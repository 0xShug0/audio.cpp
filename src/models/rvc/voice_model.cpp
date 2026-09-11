#include "engine/models/rvc/assets.h"

#include <ggml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::rvc {
namespace {

uint16_t read_u16le(const uint8_t * data) {
    return static_cast<uint16_t>(data[0]) | static_cast<uint16_t>(data[1]) << 8;
}

uint32_t read_u32le(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) | static_cast<uint32_t>(data[1]) << 8 | static_cast<uint32_t>(data[2]) << 16 |
           static_cast<uint32_t>(data[3]) << 24;
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("RVC voice checkpoint cannot be opened: " + path.string());
    }
    const std::streamsize size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error("RVC voice checkpoint has an invalid size: " + path.string());
    }
    stream.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !stream.read(reinterpret_cast<char *>(bytes.data()), size)) {
        throw std::runtime_error("RVC voice checkpoint could not be read: " + path.string());
    }
    return bytes;
}

constexpr uint32_t kEocdSignature = 0x06054b50;
constexpr uint32_t kCentralSignature = 0x02014b50;
constexpr uint32_t kLocalSignature = 0x04034b50;
constexpr uint32_t kZip64Sentinel = 0xffffffff;

struct ZipEntry {
    size_t offset = 0;
    size_t size = 0;
};

std::unordered_map<std::string, ZipEntry> read_zip_entries(const std::vector<uint8_t> & bytes) {
    if (bytes.size() < 22) {
        throw std::runtime_error("RVC voice checkpoint is too small to be a torch zip archive");
    }
    size_t eocd = bytes.size();
    const size_t scan_limit = bytes.size() >= (22 + 0xffff) ? bytes.size() - (22 + 0xffff) : 0;
    for (size_t pos = bytes.size() - 22 + 1; pos-- > scan_limit;) {
        if (read_u32le(bytes.data() + pos) == kEocdSignature) {
            eocd = pos;
            break;
        }
    }
    if (eocd == bytes.size()) {
        throw std::runtime_error("RVC voice checkpoint end-of-central-directory record not found");
    }
    const uint16_t entry_count = read_u16le(bytes.data() + eocd + 10);
    const uint32_t central_offset = read_u32le(bytes.data() + eocd + 16);
    if (central_offset == kZip64Sentinel) {
        throw std::runtime_error("RVC voice checkpoint uses zip64, which is not supported");
    }
    std::unordered_map<std::string, ZipEntry> entries;
    entries.reserve(entry_count);
    size_t cursor = central_offset;
    for (uint16_t index = 0; index < entry_count; ++index) {
        if (cursor + 46 > bytes.size() || read_u32le(bytes.data() + cursor) != kCentralSignature) {
            throw std::runtime_error("RVC voice checkpoint central directory is malformed");
        }
        const uint16_t method = read_u16le(bytes.data() + cursor + 10);
        const uint32_t compressed = read_u32le(bytes.data() + cursor + 20);
        const uint32_t uncompressed = read_u32le(bytes.data() + cursor + 24);
        const uint16_t name_len = read_u16le(bytes.data() + cursor + 28);
        const uint16_t extra_len = read_u16le(bytes.data() + cursor + 30);
        const uint16_t comment_len = read_u16le(bytes.data() + cursor + 32);
        const uint32_t local_offset = read_u32le(bytes.data() + cursor + 42);
        if (cursor + 46 + name_len > bytes.size()) {
            throw std::runtime_error("RVC voice checkpoint central directory entry is truncated");
        }
        std::string name(reinterpret_cast<const char *>(bytes.data() + cursor + 46), name_len);
        if (local_offset == kZip64Sentinel || compressed != uncompressed || method != 0) {
            throw std::runtime_error("RVC voice checkpoint entry is compressed or zip64: " + name);
        }
        if (local_offset + 30 > bytes.size() || read_u32le(bytes.data() + local_offset) != kLocalSignature) {
            throw std::runtime_error("RVC voice checkpoint local header is malformed: " + name);
        }
        const uint16_t local_name_len = read_u16le(bytes.data() + local_offset + 26);
        const uint16_t local_extra_len = read_u16le(bytes.data() + local_offset + 28);
        const size_t data_offset = local_offset + 30 + local_name_len + local_extra_len;
        if (data_offset + uncompressed > bytes.size()) {
            throw std::runtime_error("RVC voice checkpoint entry data is out of bounds: " + name);
        }
        entries.emplace(std::move(name), ZipEntry{data_offset, uncompressed});
        cursor += 46 + name_len + extra_len + comment_len;
    }
    return entries;
}

enum class GlobalKind { None, OrderedDict, RebuildTensor, StorageF32, StorageF16, StorageBF16 };

struct PickleValue {
    enum class Kind { None, Int, Bool, Str, Tuple, List, Global, Storage, Tensor, Dict } kind = Kind::None;
    int64_t integer = 0;
    bool boolean = false;
    std::string text;
    std::vector<PickleValue> items;
    GlobalKind global = GlobalKind::None;
    std::vector<int64_t> shape;
    int64_t storage_offset = 0;
};

GlobalKind classify_global(const std::string & module, const std::string & name) {
    if (module == "collections" && name == "OrderedDict") {
        return GlobalKind::OrderedDict;
    }
    if (module == "torch._utils" && name == "_rebuild_tensor_v2") {
        return GlobalKind::RebuildTensor;
    }
    if (module == "torch" && name == "FloatStorage") {
        return GlobalKind::StorageF32;
    }
    if (module == "torch" && name == "HalfStorage") {
        return GlobalKind::StorageF16;
    }
    if (module == "torch" && name == "BFloat16Storage") {
        return GlobalKind::StorageBF16;
    }
    throw std::runtime_error("RVC voice checkpoint references an unsupported pickle global: " + module + "." + name);
}

class PickleReader {
public:
    PickleReader(const uint8_t * data, size_t size) : data_(data), size_(size) {}

    PickleValue parse() {
        while (cursor_ < size_) {
            const uint8_t op = data_[cursor_++];
            switch (op) {
            case 0x80:
                expect(1);
                cursor_ += 1;
                break;
            case 0x95:
                expect(8);
                cursor_ += 8;
                break;
            case '(':
                marks_.push_back(stack_.size());
                break;
            case '0':
                pop();
                break;
            case 'N':
                push(PickleValue{});
                break;
            case 0x88:
                push(bool_value(true));
                break;
            case 0x89:
                push(bool_value(false));
                break;
            case 'K':
                push(int_value(read_byte()));
                break;
            case 'M':
                push(int_value(read_u16()));
                break;
            case 'J':
                push(int_value(read_i32()));
                break;
            case 0x8a:
                push(int_value(read_long1()));
                break;
            case 'X':
                push(str_value(read_bytes(read_u32())));
                break;
            case 0x8c:
                push(str_value(read_bytes(read_byte())));
                break;
            case 'c':
                op_global();
                break;
            case 0x93:
                op_stack_global();
                break;
            case ')':
                push(tuple_value({}));
                break;
            case ']':
                push(list_value({}));
                break;
            case '}':
                push(dict_value());
                break;
            case 0x85:
                op_tuple(1);
                break;
            case 0x86:
                op_tuple(2);
                break;
            case 0x87:
                op_tuple(3);
                break;
            case 't':
                op_tuple_mark();
                break;
            case 'q':
                op_put(read_byte());
                break;
            case 'r':
                op_put(read_u32());
                break;
            case 'h':
                op_get(read_byte());
                break;
            case 'j':
                op_get(read_u32());
                break;
            case 'Q':
                op_persid();
                break;
            case 'R':
                op_reduce();
                break;
            case 'a':
                op_append();
                break;
            case 'e':
                op_appends();
                break;
            case 's':
                op_setitem();
                break;
            case 'u':
                op_setitems();
                break;
            case 'b':
                pop();
                break;
            case '.':
                return finish();
            default:
                throw std::runtime_error("RVC voice checkpoint pickle uses an unsupported opcode: " +
                                         std::to_string(op));
            }
        }
        throw std::runtime_error("RVC voice checkpoint pickle ended without a STOP opcode");
    }

private:
    static PickleValue int_value(int64_t v) {
        PickleValue value;
        value.kind = PickleValue::Kind::Int;
        value.integer = v;
        return value;
    }
    static PickleValue bool_value(bool v) {
        PickleValue value;
        value.kind = PickleValue::Kind::Bool;
        value.boolean = v;
        return value;
    }
    static PickleValue str_value(std::string v) {
        PickleValue value;
        value.kind = PickleValue::Kind::Str;
        value.text = std::move(v);
        return value;
    }
    static PickleValue tuple_value(std::vector<PickleValue> v) {
        PickleValue value;
        value.kind = PickleValue::Kind::Tuple;
        value.items = std::move(v);
        return value;
    }
    static PickleValue list_value(std::vector<PickleValue> v) {
        PickleValue value;
        value.kind = PickleValue::Kind::List;
        value.items = std::move(v);
        return value;
    }
    static PickleValue dict_value() {
        PickleValue value;
        value.kind = PickleValue::Kind::Dict;
        return value;
    }
    void expect(size_t count) const {
        if (cursor_ + count > size_) {
            throw std::runtime_error("RVC voice checkpoint pickle is truncated");
        }
    }
    uint8_t read_byte() {
        expect(1);
        return data_[cursor_++];
    }
    uint16_t read_u16() {
        expect(2);
        const uint16_t v = read_u16le(data_ + cursor_);
        cursor_ += 2;
        return v;
    }
    uint32_t read_u32() {
        expect(4);
        const uint32_t v = read_u32le(data_ + cursor_);
        cursor_ += 4;
        return v;
    }
    int32_t read_i32() {
        return static_cast<int32_t>(read_u32());
    }
    int64_t read_long1() {
        const uint8_t length = read_byte();
        expect(length);
        int64_t value = 0;
        for (uint8_t i = 0; i < length; ++i) {
            value |= static_cast<int64_t>(data_[cursor_ + i]) << (8 * i);
        }
        if (length > 0 && length < 8 && (data_[cursor_ + length - 1] & 0x80) != 0) {
            value |= -(static_cast<int64_t>(1) << (8 * length));
        }
        cursor_ += length;
        return value;
    }
    std::string read_bytes(size_t length) {
        expect(length);
        std::string out(reinterpret_cast<const char *>(data_ + cursor_), length);
        cursor_ += length;
        return out;
    }
    std::string read_line() {
        std::string out;
        while (cursor_ < size_ && data_[cursor_] != '\n') {
            out.push_back(static_cast<char>(data_[cursor_++]));
        }
        expect(1);
        ++cursor_;
        return out;
    }
    void push(PickleValue value) {
        stack_.push_back(std::move(value));
    }
    PickleValue pop() {
        if (stack_.empty()) {
            throw std::runtime_error("RVC voice checkpoint pickle stack underflow");
        }
        PickleValue value = std::move(stack_.back());
        stack_.pop_back();
        return value;
    }
    size_t pop_mark() {
        if (marks_.empty()) {
            throw std::runtime_error("RVC voice checkpoint pickle mark underflow");
        }
        const size_t mark = marks_.back();
        marks_.pop_back();
        return mark;
    }
    void op_global() {
        const std::string module = read_line();
        const std::string name = read_line();
        PickleValue value;
        value.kind = PickleValue::Kind::Global;
        value.global = classify_global(module, name);
        push(std::move(value));
    }
    void op_stack_global() {
        const PickleValue name = pop();
        const PickleValue module = pop();
        if (name.kind != PickleValue::Kind::Str || module.kind != PickleValue::Kind::Str) {
            throw std::runtime_error("RVC voice checkpoint STACK_GLOBAL expected strings");
        }
        PickleValue value;
        value.kind = PickleValue::Kind::Global;
        value.global = classify_global(module.text, name.text);
        push(std::move(value));
    }
    void op_tuple(size_t count) {
        std::vector<PickleValue> items(count);
        for (size_t i = 0; i < count; ++i) {
            items[count - 1 - i] = pop();
        }
        push(tuple_value(std::move(items)));
    }
    void op_tuple_mark() {
        const size_t mark = pop_mark();
        std::vector<PickleValue> items(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        stack_.erase(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        push(tuple_value(std::move(items)));
    }
    void op_put(size_t id) {
        if (stack_.empty()) {
            throw std::runtime_error("RVC voice checkpoint pickle memo put on an empty stack");
        }
        if (memo_.size() <= id) {
            memo_.resize(id + 1);
        }
        memo_[id] = stack_.back();
    }
    void op_get(size_t id) {
        if (id >= memo_.size()) {
            throw std::runtime_error("RVC voice checkpoint pickle memo get references an unknown id");
        }
        push(memo_[id]);
    }
    void op_persid() {
        const PickleValue spec = pop();
        if (spec.kind != PickleValue::Kind::Tuple || spec.items.size() < 5 ||
            spec.items[1].kind != PickleValue::Kind::Global || spec.items[2].kind != PickleValue::Kind::Str ||
            spec.items[4].kind != PickleValue::Kind::Int) {
            throw std::runtime_error("RVC voice checkpoint has an unexpected storage descriptor");
        }
        PickleValue storage;
        storage.kind = PickleValue::Kind::Storage;
        storage.global = spec.items[1].global;
        storage.text = spec.items[2].text;
        push(std::move(storage));
    }
    static bool contiguous_strides(const std::vector<int64_t> & shape, const PickleValue & strides) {
        if (strides.kind != PickleValue::Kind::Tuple || strides.items.size() != shape.size()) {
            return false;
        }
        int64_t expected = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            if (strides.items[i].kind != PickleValue::Kind::Int || strides.items[i].integer != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }
    void op_reduce() {
        const PickleValue args = pop();
        const PickleValue func = pop();
        if (func.kind != PickleValue::Kind::Global) {
            throw std::runtime_error("RVC voice checkpoint pickle reduce on a non-global callable");
        }
        if (func.global == GlobalKind::OrderedDict) {
            push(dict_value());
            return;
        }
        if (func.global != GlobalKind::RebuildTensor) {
            throw std::runtime_error("RVC voice checkpoint pickle reduce on an unsupported callable");
        }
        if (args.kind != PickleValue::Kind::Tuple || args.items.size() < 4 ||
            args.items[0].kind != PickleValue::Kind::Storage || args.items[1].kind != PickleValue::Kind::Int ||
            args.items[2].kind != PickleValue::Kind::Tuple) {
            throw std::runtime_error("RVC voice checkpoint has an unexpected tensor descriptor");
        }
        PickleValue tensor;
        tensor.kind = PickleValue::Kind::Tensor;
        tensor.global = args.items[0].global;
        tensor.text = args.items[0].text;
        tensor.storage_offset = args.items[1].integer;
        if (tensor.storage_offset < 0) {
            throw std::runtime_error("RVC voice checkpoint tensor has a negative storage offset");
        }
        for (const auto & dim : args.items[2].items) {
            if (dim.kind != PickleValue::Kind::Int) {
                throw std::runtime_error("RVC voice checkpoint tensor shape is not integral");
            }
            tensor.shape.push_back(dim.integer);
        }
        if (!contiguous_strides(tensor.shape, args.items[3])) {
            throw std::runtime_error("RVC voice checkpoint contains a non-contiguous tensor");
        }
        push(std::move(tensor));
    }
    void op_append() {
        PickleValue value = pop();
        if (stack_.empty() || stack_.back().kind != PickleValue::Kind::List) {
            throw std::runtime_error("RVC voice checkpoint append without a list");
        }
        stack_.back().items.push_back(std::move(value));
    }
    void op_appends() {
        const size_t mark = pop_mark();
        std::vector<PickleValue> items(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        stack_.erase(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        if (stack_.empty() || stack_.back().kind != PickleValue::Kind::List) {
            throw std::runtime_error("RVC voice checkpoint appends without a list");
        }
        for (auto & item : items) {
            stack_.back().items.push_back(std::move(item));
        }
    }
    void op_setitem() {
        PickleValue value = pop();
        PickleValue key = pop();
        if (stack_.empty() || stack_.back().kind != PickleValue::Kind::Dict) {
            throw std::runtime_error("RVC voice checkpoint setitem without a dict");
        }
        stack_.back().items.push_back(std::move(key));
        stack_.back().items.push_back(std::move(value));
    }
    void op_setitems() {
        const size_t mark = pop_mark();
        std::vector<PickleValue> items(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        stack_.erase(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        if (stack_.empty() || stack_.back().kind != PickleValue::Kind::Dict) {
            throw std::runtime_error("RVC voice checkpoint setitems without a dict");
        }
        for (auto & item : items) {
            stack_.back().items.push_back(std::move(item));
        }
    }
    PickleValue finish() {
        PickleValue result = pop();
        if (result.kind != PickleValue::Kind::Dict || !stack_.empty()) {
            throw std::runtime_error("RVC voice checkpoint pickle did not produce a single dict");
        }
        return result;
    }

    const uint8_t * data_;
    size_t size_;
    size_t cursor_ = 0;
    std::vector<PickleValue> stack_;
    std::vector<PickleValue> memo_;
    std::vector<size_t> marks_;
};

struct TensorRecord {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

const char * storage_dtype_name(GlobalKind kind) {
    switch (kind) {
    case GlobalKind::StorageF32:
        return "F32";
    case GlobalKind::StorageF16:
        return "F16";
    case GlobalKind::StorageBF16:
        return "BF16";
    default:
        throw std::runtime_error("RVC voice checkpoint tensor has an unsupported storage dtype");
    }
}

size_t dtype_byte_size(const std::string & dtype) {
    return dtype == "F32" ? 4 : 2;
}

int64_t shape_elements(const std::vector<int64_t> & shape) {
    int64_t total = 1;
    for (const int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("RVC voice checkpoint tensor has a negative dimension");
        }
        total *= dim;
    }
    return total;
}

std::vector<float> decode_f32(const std::string & dtype, const uint8_t * data, size_t elements) {
    std::vector<float> values(elements);
    if (dtype == "F32") {
        std::memcpy(values.data(), data, elements * sizeof(float));
    } else if (dtype == "F16") {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), values.data(),
                              static_cast<int64_t>(elements));
    } else {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(data), values.data(),
                              static_cast<int64_t>(elements));
    }
    return values;
}

const PickleValue * dict_find(const PickleValue & dict, std::string_view key) {
    if (dict.kind != PickleValue::Kind::Dict) {
        throw std::runtime_error("RVC voice checkpoint metadata is not a dict");
    }
    for (size_t i = 0; i + 1 < dict.items.size(); i += 2) {
        if (dict.items[i].kind == PickleValue::Kind::Str && dict.items[i].text == key) {
            return &dict.items[i + 1];
        }
    }
    return nullptr;
}

PickleValue * dict_find(PickleValue & dict, std::string_view key) {
    return const_cast<PickleValue *>(dict_find(static_cast<const PickleValue &>(dict), key));
}

std::string require_string_metadata(const PickleValue & dict, std::string_view key) {
    const auto * value = dict_find(dict, key);
    if (value == nullptr || value->kind != PickleValue::Kind::Str) {
        throw std::runtime_error("RVC voice checkpoint missing string metadata: " + std::string(key));
    }
    return value->text;
}

int require_int_metadata(const PickleValue & dict, std::string_view key) {
    const auto * value = dict_find(dict, key);
    if (value == nullptr || value->kind != PickleValue::Kind::Int) {
        throw std::runtime_error("RVC voice checkpoint missing integer metadata: " + std::string(key));
    }
    return static_cast<int>(value->integer);
}

int sample_rate_from_checkpoint(const PickleValue & dict) {
    if (const auto * config = dict_find(dict, "config");
        config != nullptr && config->kind == PickleValue::Kind::List && !config->items.empty()) {
        const auto & last = config->items.back();
        if (last.kind == PickleValue::Kind::Int && last.integer > 0) {
            return static_cast<int>(last.integer);
        }
    }
    const auto sr = require_string_metadata(dict, "sr");
    if (sr == "32k") {
        return 32000;
    }
    if (sr == "40k") {
        return 40000;
    }
    if (sr == "48k") {
        return 48000;
    }
    throw std::runtime_error("RVC voice checkpoint has unsupported sample-rate metadata: " + sr);
}

class RvcTorchCheckpointTensorSource final : public engine::assets::TensorSource {
public:
    explicit RvcTorchCheckpointTensorSource(std::filesystem::path path)
        : path_(std::move(path)), bytes_(read_file_bytes(path_)) {
        const auto entries = read_zip_entries(bytes_);
        std::string prefix;
        const ZipEntry * pickle = nullptr;
        for (const auto & [name, entry] : entries) {
            if (name == "data.pkl" || (name.size() >= 9 && name.compare(name.size() - 9, 9, "/data.pkl") == 0)) {
                prefix = name.substr(0, name.size() - std::string("data.pkl").size());
                pickle = &entry;
                break;
            }
        }
        if (pickle == nullptr) {
            throw std::runtime_error("RVC voice checkpoint does not contain data.pkl: " + path_.string());
        }
        PickleReader reader(bytes_.data() + pickle->offset, pickle->size);
        auto root = reader.parse();
        if (const auto * version = dict_find(root, "version");
            version != nullptr && version->kind == PickleValue::Kind::Str) {
            version_ = version->text;
        }
        has_f0_ = require_int_metadata(root, "f0") != 0;
        sample_rate_ = sample_rate_from_checkpoint(root);
        PickleValue * weights = dict_find(root, "weight");
        if (weights == nullptr || weights->kind != PickleValue::Kind::Dict) {
            throw std::runtime_error("RVC voice checkpoint missing weight dict: " + path_.string());
        }
        records_.reserve(weights->items.size() / 2);
        std::string storage_name = prefix + "data/";
        const size_t storage_name_prefix_size = storage_name.size();
        for (size_t i = 0; i + 1 < weights->items.size(); i += 2) {
            PickleValue & key = weights->items[i];
            PickleValue & value = weights->items[i + 1];
            if (key.kind != PickleValue::Kind::Str || value.kind != PickleValue::Kind::Tensor) {
                throw std::runtime_error("RVC voice checkpoint weight dict contains a non-tensor entry");
            }
            storage_name.resize(storage_name_prefix_size);
            storage_name += value.text;
            const auto storage = entries.find(storage_name);
            if (storage == entries.end()) {
                throw std::runtime_error("RVC voice checkpoint is missing storage for tensor: " + key.text);
            }
            TensorRecord record;
            record.dtype = storage_dtype_name(value.global);
            record.shape = std::move(value.shape);
            const size_t element_bytes = dtype_byte_size(record.dtype);
            record.byte_size = static_cast<size_t>(shape_elements(record.shape)) * element_bytes;
            const size_t storage_offset = static_cast<size_t>(value.storage_offset) * element_bytes;
            if (storage_offset + record.byte_size > storage->second.size) {
                throw std::runtime_error("RVC voice checkpoint tensor is larger than its storage: " + key.text);
            }
            record.byte_offset = storage->second.offset + storage_offset;
            records_.emplace(std::move(key.text), std::move(record));
        }
        if (version_.empty()) {
            const auto phone = records_.find("enc_p.emb_phone.weight");
            if (phone == records_.end() || phone->second.shape.size() != 2) {
                throw std::runtime_error("RVC voice checkpoint cannot infer version from enc_p.emb_phone.weight");
            }
            if (phone->second.shape[1] == 256) {
                version_ = "v1";
            } else if (phone->second.shape[1] == 768) {
                version_ = "v2";
            } else {
                throw std::runtime_error("RVC voice checkpoint has unsupported phone embedding width: " +
                                         path_.string());
            }
        }
    }

    const std::filesystem::path & source_path() const noexcept override {
        return path_;
    }
    bool has_tensor(std::string_view name) const noexcept override {
        return records_.find(std::string(name)) != records_.end();
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        const auto & record = require_record(name);
        return engine::assets::TensorMetadata{std::string(name), record.dtype, record.shape};
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        std::vector<engine::assets::TensorMetadata> out;
        out.reserve(records_.size());
        for (const auto & [name, record] : records_) {
            out.push_back({name, record.dtype, record.shape});
        }
        return out;
    }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & record = require_record(name);
        engine::assets::RawTensorData tensor;
        tensor.metadata = engine::assets::TensorMetadata{std::string(name), record.dtype, record.shape};
        tensor.bytes.resize(record.byte_size);
        std::memcpy(tensor.bytes.data(), bytes_.data() + record.byte_offset, record.byte_size);
        return tensor;
    }
    std::vector<float> require_f32(std::string_view name,
                                   const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto & record = require_record(name);
        if (expected_shape.has_value() && *expected_shape != record.shape) {
            throw std::runtime_error("RVC voice checkpoint tensor shape mismatch for " + std::string(name));
        }
        return decode_f32(record.dtype, bytes_.data() + record.byte_offset,
                          static_cast<size_t>(shape_elements(record.shape)));
    }
    std::optional<std::vector<float>>
    optional_f32(std::string_view name, const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        throw std::runtime_error("RVC voice checkpoint does not expose i64 scalars: " + std::string(name));
    }
    const std::string & version() const noexcept {
        return version_;
    }
    int sample_rate() const noexcept {
        return sample_rate_;
    }
    bool has_f0() const noexcept {
        return has_f0_;
    }

private:
    const TensorRecord & require_record(std::string_view name) const {
        const auto it = records_.find(std::string(name));
        if (it == records_.end()) {
            throw std::runtime_error("missing RVC voice tensor: " + std::string(name));
        }
        return it->second;
    }

    std::filesystem::path path_;
    std::vector<uint8_t> bytes_;
    std::unordered_map<std::string, TensorRecord> records_;
    std::string version_;
    int sample_rate_ = 0;
    bool has_f0_ = true;
};

std::shared_ptr<const engine::assets::TensorSource>
open_rvc_voice_checkpoint(const std::filesystem::path & checkpoint_path, std::string & version, int & sample_rate,
                          bool & has_f0) {
    const auto extension = checkpoint_path.extension().string();
    if (extension == ".safetensors" || extension == ".gguf") {
        auto source = engine::assets::open_tensor_source(checkpoint_path);
        has_f0 = source->has_tensor("enc_p.emb_pitch.weight");
        const auto phone = source->require_metadata("enc_p.emb_phone.weight");
        if (phone.shape.size() != 2) {
            throw std::runtime_error("RVC voice checkpoint has invalid enc_p.emb_phone.weight shape: " +
                                     checkpoint_path.string());
        }
        if (phone.shape[1] == 256) {
            version = "v1";
        } else if (phone.shape[1] == 768) {
            version = "v2";
        } else {
            throw std::runtime_error("RVC voice checkpoint has unsupported phone embedding width: " +
                                     checkpoint_path.string());
        }
        return source;
    }
    if (extension == ".pth" || extension == ".pt") {
        auto source = std::make_shared<RvcTorchCheckpointTensorSource>(checkpoint_path);
        version = source->version();
        sample_rate = source->sample_rate();
        has_f0 = source->has_f0();
        return source;
    }
    throw std::runtime_error("unsupported RVC voice checkpoint format: " + checkpoint_path.string());
}

}  // namespace

RvcSynthesizerLayout infer_rvc_synthesizer_layout(
    const engine::assets::TensorSource & source,
    int sample_rate,
    const std::string & source_label) {
    RvcSynthesizerLayout layout;
    if (sample_rate == 32000) {
        layout.upsample_rates = {10, 8, 2, 2};
    } else if (sample_rate == 40000) {
        layout.upsample_rates = {10, 10, 2, 2};
    } else if (sample_rate == 48000) {
        layout.upsample_rates = {12, 10, 2, 2};
    } else {
        throw std::runtime_error("RVC voice checkpoint has unsupported synthesizer sample rate: " + source_label);
    }
    layout.hop_samples = 1;
    for (const auto rate : layout.upsample_rates) {
        layout.hop_samples *= rate;
    }
    if (layout.hop_samples != sample_rate / 100 || sample_rate % 100 != 0) {
        throw std::runtime_error("RVC voice checkpoint sample rate does not match a 10 ms synthesizer hop: " +
                                 source_label);
    }
    constexpr int64_t channels[4] = {256, 128, 64, 32};
    constexpr int64_t in_channels[4] = {512, 256, 128, 64};
    for (int64_t up = 0; up < 4; ++up) {
        auto name = "dec.ups." + std::to_string(up) + ".weight";
        if (!source.has_tensor(name)) {
            name = "dec.ups." + std::to_string(up) + ".weight_v";
        }
        const auto meta = source.require_metadata(name);
        if (meta.shape.size() != 3 || meta.shape[0] != in_channels[up] || meta.shape[1] != channels[up] ||
            meta.shape[2] <= 0) {
            throw std::runtime_error("RVC voice checkpoint has unsupported decoder upsample tensor shape for " +
                                     name + ": " + source_label);
        }
        layout.upsample_kernel_sizes[static_cast<size_t>(up)] = meta.shape[2];
        const auto padding = layout.upsample_kernel_sizes[static_cast<size_t>(up)] -
            layout.upsample_rates[static_cast<size_t>(up)];
        if (padding < 0 || padding % 2 != 0) {
            throw std::runtime_error("RVC voice checkpoint has unsupported decoder upsample padding for " +
                                     name + ": " + source_label);
        }
    }
    return layout;
}

RvcVoiceModel load_rvc_voice_model(const std::filesystem::path & checkpoint_path) {
    RvcVoiceModel voice;
    voice.checkpoint_path = checkpoint_path;
    voice.id = "user:" + std::filesystem::absolute(checkpoint_path).lexically_normal().string();
    voice.checkpoint = open_rvc_voice_checkpoint(checkpoint_path, voice.version, voice.sample_rate, voice.has_f0);
    if (voice.version != "v1" && voice.version != "v2") {
        throw std::runtime_error("RVC voice checkpoint has unsupported version: " + voice.version);
    }
    if (voice.sample_rate <= 0) {
        throw std::runtime_error("RVC voice checkpoint has invalid sample rate: " + checkpoint_path.string());
    }
    voice.synthesizer_layout = infer_rvc_synthesizer_layout(
        *voice.checkpoint,
        voice.sample_rate,
        checkpoint_path.string());
    voice.has_f0 = voice.checkpoint->has_tensor("enc_p.emb_pitch.weight");
    const auto speaker = voice.checkpoint->require_metadata("emb_g.weight");
    if (speaker.shape.size() != 2 || speaker.shape[0] <= 0) {
        throw std::runtime_error("RVC emb_g.weight shape is invalid: " + checkpoint_path.string());
    }
    voice.speaker_count = static_cast<int>(speaker.shape[0]);
    return voice;
}

}  // namespace engine::models::rvc
