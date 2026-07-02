#include "engine/framework/assets/torch_bin.h"

#include <ggml.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::assets {
namespace {

// ---- Byte helpers -------------------------------------------------------------------

uint16_t read_u16le(const uint8_t * data) {
    return static_cast<uint16_t>(data[0]) | static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
}

uint32_t read_u32le(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

std::vector<uint8_t> read_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("torch .bin cannot be opened: " + path.string());
    }
    const std::streamsize size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error("torch .bin has an invalid size: " + path.string());
    }
    stream.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !stream.read(reinterpret_cast<char *>(bytes.data()), size)) {
        throw std::runtime_error("torch .bin could not be read: " + path.string());
    }
    return bytes;
}

// ---- Minimal STORED-only ZIP reader -------------------------------------------------

constexpr uint32_t kEocdSignature = 0x06054b50;
constexpr uint32_t kCentralSignature = 0x02014b50;
constexpr uint32_t kLocalSignature = 0x04034b50;
constexpr uint32_t kZip64Sentinel = 0xffffffff;

struct ZipEntry {
    size_t offset = 0;  // byte offset of the entry payload inside the archive
    size_t size = 0;    // payload size in bytes
};

std::map<std::string, ZipEntry> read_zip_entries(const std::vector<uint8_t> & bytes) {
    if (bytes.size() < 22) {
        throw std::runtime_error("torch .bin is too small to be a zip archive");
    }
    // Scan backwards for the end-of-central-directory record (no archive comment in torch files).
    size_t eocd = bytes.size();
    const size_t scan_limit = bytes.size() >= (22 + 0xffff) ? bytes.size() - (22 + 0xffff) : 0;
    for (size_t pos = bytes.size() - 22 + 1; pos-- > scan_limit;) {
        if (read_u32le(bytes.data() + pos) == kEocdSignature) {
            eocd = pos;
            break;
        }
    }
    if (eocd == bytes.size()) {
        throw std::runtime_error("torch .bin end-of-central-directory record not found");
    }
    const uint16_t entry_count = read_u16le(bytes.data() + eocd + 10);
    const uint32_t central_offset = read_u32le(bytes.data() + eocd + 16);
    if (central_offset == kZip64Sentinel) {
        throw std::runtime_error("torch .bin uses zip64, which is not supported");
    }

    std::map<std::string, ZipEntry> entries;
    size_t cursor = central_offset;
    for (uint16_t index = 0; index < entry_count; ++index) {
        if (cursor + 46 > bytes.size() || read_u32le(bytes.data() + cursor) != kCentralSignature) {
            throw std::runtime_error("torch .bin central directory is malformed");
        }
        const uint16_t method = read_u16le(bytes.data() + cursor + 10);
        const uint32_t compressed = read_u32le(bytes.data() + cursor + 20);
        const uint32_t uncompressed = read_u32le(bytes.data() + cursor + 24);
        const uint16_t name_len = read_u16le(bytes.data() + cursor + 28);
        const uint16_t extra_len = read_u16le(bytes.data() + cursor + 30);
        const uint16_t comment_len = read_u16le(bytes.data() + cursor + 32);
        const uint32_t local_offset = read_u32le(bytes.data() + cursor + 42);
        if (cursor + 46 + name_len > bytes.size()) {
            throw std::runtime_error("torch .bin central directory entry is truncated");
        }
        std::string name(reinterpret_cast<const char *>(bytes.data() + cursor + 46), name_len);

        if (local_offset == kZip64Sentinel || compressed != uncompressed) {
            throw std::runtime_error("torch .bin entry is compressed or zip64, which is not supported: " + name);
        }
        if (method != 0) {
            throw std::runtime_error("torch .bin entry is not stored uncompressed: " + name);
        }
        if (local_offset + 30 > bytes.size() || read_u32le(bytes.data() + local_offset) != kLocalSignature) {
            throw std::runtime_error("torch .bin local header is malformed: " + name);
        }
        const uint16_t local_name_len = read_u16le(bytes.data() + local_offset + 26);
        const uint16_t local_extra_len = read_u16le(bytes.data() + local_offset + 28);
        const size_t data_offset = local_offset + 30 + local_name_len + local_extra_len;
        if (data_offset + uncompressed > bytes.size()) {
            throw std::runtime_error("torch .bin entry data is out of bounds: " + name);
        }
        entries.emplace(std::move(name), ZipEntry{data_offset, uncompressed});
        cursor += 46 + name_len + extra_len + comment_len;
    }
    return entries;
}

// ---- Restricted pickle interpreter --------------------------------------------------

enum class GlobalKind { None, OrderedDict, RebuildTensor, StorageF32, StorageF16, StorageBF16 };

struct PickleValue {
    enum class Kind { None, Int, Bool, Str, Tuple, Global, Storage, Tensor, Dict } kind = Kind::None;
    int64_t integer = 0;
    bool boolean = false;
    std::string text;                 // Str value, or a Storage/Tensor storage key
    std::vector<PickleValue> items;   // Tuple items, or a Dict as flat [k0, v0, k1, v1, ...]
    GlobalKind global = GlobalKind::None;  // Global callable, or a Storage/Tensor dtype
    std::vector<int64_t> shape;       // Tensor shape
    int64_t numel = 0;                // Storage/Tensor element count
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
    throw std::runtime_error("torch .bin pickle references an unsupported global: " + module + "." + name);
}

class PickleReader {
public:
    explicit PickleReader(const uint8_t * data, size_t size) : data_(data), size_(size) {}

    // Returns the state dict as a flat [key, value, key, value, ...] tuple of the final OrderedDict.
    std::vector<PickleValue> parse() {
        while (cursor_ < size_) {
            const uint8_t op = data_[cursor_++];
            switch (op) {
                case 0x80: expect(1); cursor_ += 1; break;                       // PROTO
                case 0x95: expect(8); cursor_ += 8; break;                       // FRAME
                case '(': marks_.push_back(stack_.size()); break;                // MARK
                case '0': pop(); break;                                          // POP
                case 'N': push(none_value()); break;                            // NONE
                case 0x88: push(bool_value(true)); break;                       // NEWTRUE
                case 0x89: push(bool_value(false)); break;                      // NEWFALSE
                case 'K': push(int_value(read_byte())); break;                  // BININT1
                case 'M': push(int_value(read_u16())); break;                   // BININT2
                case 'J': push(int_value(read_i32())); break;                   // BININT
                case 0x8a: push(int_value(read_long1())); break;                // LONG1
                case 'X': push(str_value(read_bytes(read_u32()))); break;       // BINUNICODE
                case 0x8c: push(str_value(read_bytes(read_byte()))); break;     // SHORT_BINUNICODE
                case 'c': op_global(); break;                                   // GLOBAL
                case 0x93: op_stack_global(); break;                            // STACK_GLOBAL
                case ')': push(tuple_value({})); break;                         // EMPTY_TUPLE
                case '}': push(dict_value()); break;                            // EMPTY_DICT
                case 0x85: op_tuple(1); break;                                  // TUPLE1
                case 0x86: op_tuple(2); break;                                  // TUPLE2
                case 0x87: op_tuple(3); break;                                  // TUPLE3
                case 't': op_tuple_mark(); break;                               // TUPLE
                case 'q': op_put(read_byte()); break;                           // BINPUT
                case 'r': op_put(read_u32()); break;                            // LONG_BINPUT
                case 'h': op_get(read_byte()); break;                           // BINGET
                case 'j': op_get(read_u32()); break;                            // LONG_BINGET
                case 'Q': op_persid(); break;                                   // BINPERSID
                case 'R': op_reduce(); break;                                   // REDUCE
                case 's': op_setitem(); break;                                  // SETITEM
                case 'u': op_setitems(); break;                                 // SETITEMS
                case 'b': pop(); break;                                          // BUILD (drop __setstate__ payload, e.g. _metadata)
                case '.': return finish();                                       // STOP
                default:
                    throw std::runtime_error(
                        "torch .bin pickle uses an unsupported opcode: " + std::to_string(op));
            }
        }
        throw std::runtime_error("torch .bin pickle ended without a STOP opcode");
    }

private:
    static PickleValue none_value() { return PickleValue{}; }
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
    static PickleValue dict_value() {
        PickleValue value;
        value.kind = PickleValue::Kind::Dict;
        return value;
    }

    void expect(size_t count) const {
        if (cursor_ + count > size_) {
            throw std::runtime_error("torch .bin pickle is truncated");
        }
    }
    uint8_t read_byte() { expect(1); return data_[cursor_++]; }
    uint16_t read_u16() { expect(2); const uint16_t v = read_u16le(data_ + cursor_); cursor_ += 2; return v; }
    uint32_t read_u32() { expect(4); const uint32_t v = read_u32le(data_ + cursor_); cursor_ += 4; return v; }
    int32_t read_i32() { return static_cast<int32_t>(read_u32()); }
    int64_t read_long1() {
        const uint8_t length = read_byte();
        expect(length);
        int64_t value = 0;
        for (uint8_t i = 0; i < length; ++i) {
            value |= static_cast<int64_t>(data_[cursor_ + i]) << (8 * i);
        }
        if (length > 0 && length < 8 && (data_[cursor_ + length - 1] & 0x80) != 0) {
            value |= -(static_cast<int64_t>(1) << (8 * length));  // sign extend
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
        expect(1);  // consume the '\n'
        ++cursor_;
        return out;
    }

    void push(PickleValue value) { stack_.push_back(std::move(value)); }
    PickleValue pop() {
        if (stack_.empty()) {
            throw std::runtime_error("torch .bin pickle stack underflow");
        }
        PickleValue value = std::move(stack_.back());
        stack_.pop_back();
        return value;
    }
    size_t pop_mark() {
        if (marks_.empty()) {
            throw std::runtime_error("torch .bin pickle mark underflow");
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
            throw std::runtime_error("torch .bin pickle put on an empty stack");
        }
        if (memo_.size() <= id) {
            memo_.resize(id + 1);
        }
        memo_[id] = stack_.back();
    }
    void op_get(size_t id) {
        if (id >= memo_.size()) {
            throw std::runtime_error("torch .bin pickle get references an unknown memo id");
        }
        push(memo_[id]);
    }
    void op_persid() {
        const PickleValue spec = pop();  // ('storage', StorageType, key, device, numel)
        if (spec.kind != PickleValue::Kind::Tuple || spec.items.size() < 5 ||
            spec.items[1].kind != PickleValue::Kind::Global ||
            spec.items[2].kind != PickleValue::Kind::Str ||
            spec.items[4].kind != PickleValue::Kind::Int) {
            throw std::runtime_error("torch .bin pickle has an unexpected storage descriptor");
        }
        PickleValue storage;
        storage.kind = PickleValue::Kind::Storage;
        storage.global = spec.items[1].global;
        storage.text = spec.items[2].text;
        storage.numel = spec.items[4].integer;
        push(std::move(storage));
    }
    void op_reduce() {
        const PickleValue args = pop();
        const PickleValue func = pop();
        if (func.kind != PickleValue::Kind::Global) {
            throw std::runtime_error("torch .bin pickle reduce on a non-global callable");
        }
        if (func.global == GlobalKind::OrderedDict) {
            push(dict_value());
            return;
        }
        if (func.global == GlobalKind::RebuildTensor) {
            if (args.kind != PickleValue::Kind::Tuple || args.items.size() < 3 ||
                args.items[0].kind != PickleValue::Kind::Storage ||
                args.items[2].kind != PickleValue::Kind::Tuple) {
                throw std::runtime_error("torch .bin pickle has an unexpected tensor descriptor");
            }
            PickleValue tensor;
            tensor.kind = PickleValue::Kind::Tensor;
            tensor.global = args.items[0].global;
            tensor.text = args.items[0].text;
            tensor.numel = args.items[0].numel;
            for (const auto & dim : args.items[2].items) {
                if (dim.kind != PickleValue::Kind::Int) {
                    throw std::runtime_error("torch .bin pickle tensor shape is not integral");
                }
                tensor.shape.push_back(dim.integer);
            }
            push(std::move(tensor));
            return;
        }
        throw std::runtime_error("torch .bin pickle reduce on an unsupported callable");
    }
    void op_setitem() {
        PickleValue value = pop();
        PickleValue key = pop();
        if (stack_.empty() || stack_.back().kind != PickleValue::Kind::Dict) {
            throw std::runtime_error("torch .bin pickle setitem without a dict");
        }
        stack_.back().items.push_back(std::move(key));
        stack_.back().items.push_back(std::move(value));
    }
    void op_setitems() {
        const size_t mark = pop_mark();
        std::vector<PickleValue> items(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        stack_.erase(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        if (stack_.empty() || stack_.back().kind != PickleValue::Kind::Dict) {
            throw std::runtime_error("torch .bin pickle setitems without a dict");
        }
        for (auto & item : items) {
            stack_.back().items.push_back(std::move(item));
        }
    }
    std::vector<PickleValue> finish() {
        const PickleValue result = pop();
        if (result.kind != PickleValue::Kind::Dict) {
            throw std::runtime_error("torch .bin pickle did not produce a state dict");
        }
        return result.items;
    }

    const uint8_t * data_;
    size_t size_;
    size_t cursor_ = 0;
    std::vector<PickleValue> stack_;
    std::vector<PickleValue> memo_;
    std::vector<size_t> marks_;
};

// ---- Tensor source ------------------------------------------------------------------

struct TensorRecord {
    std::string dtype;  // "F32", "F16", or "BF16"
    std::vector<int64_t> shape;
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

const char * storage_dtype_name(GlobalKind kind) {
    switch (kind) {
        case GlobalKind::StorageF32: return "F32";
        case GlobalKind::StorageF16: return "F16";
        case GlobalKind::StorageBF16: return "BF16";
        default:
            throw std::runtime_error("torch .bin tensor has an unsupported storage dtype");
    }
}

size_t dtype_byte_size(const std::string & dtype) {
    if (dtype == "F32") {
        return 4;
    }
    return 2;  // F16 / BF16
}

std::vector<float> decode_f32(const std::string & dtype, const uint8_t * data, size_t elements) {
    std::vector<float> values(elements);
    if (dtype == "F32") {
        std::memcpy(values.data(), data, elements * sizeof(float));
    } else if (dtype == "F16") {
        ggml_fp16_to_fp32_row(
            reinterpret_cast<const ggml_fp16_t *>(data), values.data(), static_cast<int64_t>(elements));
    } else {
        ggml_bf16_to_fp32_row(
            reinterpret_cast<const ggml_bf16_t *>(data), values.data(), static_cast<int64_t>(elements));
    }
    return values;
}

int64_t shape_elements(const std::vector<int64_t> & shape) {
    int64_t total = 1;
    for (const int64_t dim : shape) {
        total *= dim;
    }
    return total;
}

class TorchBinTensorSource final : public TensorSource {
public:
    explicit TorchBinTensorSource(std::filesystem::path path)
        : path_(std::move(path)), bytes_(read_file(path_)) {
        const auto entries = read_zip_entries(bytes_);
        std::string prefix;
        const ZipEntry * pickle = nullptr;
        for (const auto & [name, entry] : entries) {
            if (name == "data.pkl" || (name.size() >= 9 && name.compare(name.size() - 9, 9, "/data.pkl") == 0)) {
                prefix = name.substr(0, name.size() - std::string("data.pkl").size());  // keeps trailing '/'
                pickle = &entry;
                break;
            }
        }
        if (pickle == nullptr) {
            throw std::runtime_error("torch .bin does not contain a data.pkl: " + path_.string());
        }
        PickleReader reader(bytes_.data() + pickle->offset, pickle->size);
        const std::vector<PickleValue> entries_flat = reader.parse();
        for (size_t i = 0; i + 1 < entries_flat.size(); i += 2) {
            const PickleValue & key = entries_flat[i];
            const PickleValue & value = entries_flat[i + 1];
            if (key.kind != PickleValue::Kind::Str || value.kind != PickleValue::Kind::Tensor) {
                throw std::runtime_error("torch .bin state dict has a non-tensor entry: " + key.text);
            }
            const auto storage = entries.find(prefix + "data/" + value.text);
            if (storage == entries.end()) {
                throw std::runtime_error("torch .bin is missing storage for tensor: " + key.text);
            }
            TensorRecord record;
            record.dtype = storage_dtype_name(value.global);
            record.shape = value.shape;
            const size_t element_bytes = dtype_byte_size(record.dtype);
            record.byte_size = static_cast<size_t>(shape_elements(value.shape)) * element_bytes;
            record.byte_offset = storage->second.offset;
            if (record.byte_size > storage->second.size) {
                throw std::runtime_error("torch .bin tensor is larger than its storage: " + key.text);
            }
            records_.emplace(key.text, std::move(record));
        }
    }

    const std::filesystem::path & source_path() const noexcept override {
        return path_;
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return records_.find(std::string(name)) != records_.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const auto & record = require_record(name);
        return TensorMetadata{std::string(name), record.dtype, record.shape};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(records_.size());
        for (const auto & [name, record] : records_) {
            out.push_back({name, record.dtype, record.shape});
        }
        return out;
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & record = require_record(name);
        RawTensorData tensor;
        tensor.metadata = TensorMetadata{std::string(name), record.dtype, record.shape};
        tensor.bytes.resize(record.byte_size);
        std::memcpy(tensor.bytes.data(), bytes_.data() + record.byte_offset, record.byte_size);
        return tensor;
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto & record = require_record(name);
        if (expected_shape.has_value() && *expected_shape != record.shape) {
            throw std::runtime_error("torch .bin tensor shape mismatch for " + std::string(name));
        }
        return decode_f32(
            record.dtype,
            bytes_.data() + record.byte_offset,
            static_cast<size_t>(shape_elements(record.shape)));
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        throw std::runtime_error("torch .bin does not support i64 scalars: " + std::string(name));
    }

private:
    const TensorRecord & require_record(std::string_view name) const {
        const auto it = records_.find(std::string(name));
        if (it == records_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        return it->second;
    }

    std::filesystem::path path_;
    std::vector<uint8_t> bytes_;
    std::map<std::string, TensorRecord> records_;
};

}  // namespace

std::shared_ptr<const TensorSource> open_torch_bin_tensor_source(const std::filesystem::path & path) {
    return std::make_shared<TorchBinTensorSource>(path);
}

}  // namespace engine::assets
