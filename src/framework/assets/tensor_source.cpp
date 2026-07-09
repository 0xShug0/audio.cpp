#include "engine/framework/assets/tensor_source.h"

#include "engine/framework/io/binary.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <functional>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace engine::assets {
namespace {

constexpr int64_t kParallelF32ConvertElements = 1ll << 20;
constexpr int64_t kF32ConvertChunkElements = 1ll << 16;

core::TensorShape shape_from_dims(const std::vector<int64_t> & dims) {
    if (dims.empty() || dims.size() > core::kMaxTensorRank) {
        throw std::runtime_error("tensor rank must be between 1 and 4");
    }
    switch (dims.size()) {
        case 1:
            return core::TensorShape::from_dims({dims[0]});
        case 2:
            return core::TensorShape::from_dims({dims[0], dims[1]});
        case 3:
            return core::TensorShape::from_dims({dims[0], dims[1], dims[2]});
        case 4:
            return core::TensorShape::from_dims({dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::runtime_error("unsupported tensor rank");
    }
}

void validate_expected_shape(
    std::string_view name,
    const std::vector<int64_t> & actual_shape,
    const std::optional<std::vector<int64_t>> & expected_shape) {
    if (expected_shape.has_value() && actual_shape != *expected_shape) {
        throw std::runtime_error("tensor shape mismatch for " + std::string(name));
    }
}

std::string lower_ascii(std::string_view value) {
    std::string out(value);
    for (char & ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool raw_dtype_matches_ggml_type(std::string_view dtype, ggml_type type) {
    const std::string normalized = lower_ascii(dtype);
    return (normalized == "f32" && type == GGML_TYPE_F32) ||
           (normalized == "f16" && type == GGML_TYPE_F16) ||
           (normalized == "bf16" && type == GGML_TYPE_BF16) ||
           (normalized == "q4_0" && type == GGML_TYPE_Q4_0) ||
           (normalized == "q4_1" && type == GGML_TYPE_Q4_1) ||
           (normalized == "q5_0" && type == GGML_TYPE_Q5_0) ||
           (normalized == "q5_1" && type == GGML_TYPE_Q5_1) ||
           (normalized == "q4_k" && type == GGML_TYPE_Q4_K) ||
           (normalized == "q5_k" && type == GGML_TYPE_Q5_K) ||
           (normalized == "q6_k" && type == GGML_TYPE_Q6_K) ||
           (normalized == "q8_0" && type == GGML_TYPE_Q8_0);
}

void validate_raw_tensor_byte_size(std::string_view name, const core::TensorShape & shape, ggml_type type, size_t bytes) {
    const size_t expected = static_cast<size_t>(shape.prefix_elements()) *
                            ggml_row_size(type, shape.last_dim());
    if (bytes != expected) {
        throw std::runtime_error("tensor byte size mismatch for " + std::string(name));
    }
}

std::vector<std::byte> f32_bytes(const std::vector<float> & values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> f16_bytes(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> f16_values(values.size());
    ggml_fp32_to_fp16_row(values.data(), f16_values.data(), static_cast<int64_t>(values.size()));
    std::vector<std::byte> bytes(values.size() * sizeof(ggml_fp16_t));
    std::memcpy(bytes.data(), f16_values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> bf16_bytes(const std::vector<float> & values) {
    std::vector<ggml_bf16_t> bf16_values(values.size());
    ggml_fp32_to_bf16_row(values.data(), bf16_values.data(), static_cast<int64_t>(values.size()));
    std::vector<std::byte> bytes(values.size() * sizeof(ggml_bf16_t));
    std::memcpy(bytes.data(), bf16_values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> quantize_f32_rows(
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type) {
    if (!ggml_is_quantized(type)) {
        throw std::runtime_error("tensor quantization target is not a quantized ggml type");
    }
    if (ggml_quantize_requires_imatrix(type)) {
        throw std::runtime_error("tensor quantization target requires an importance matrix: " + std::string(name));
    }
    if (shape.rank < 2) {
        throw std::runtime_error("quantized tensor must have rank >= 2: " + std::string(name));
    }
    const int64_t elements_per_row = shape.last_dim();
    if (elements_per_row % ggml_blck_size(type) != 0) {
        throw std::runtime_error("quantized tensor row size is not divisible by block size: " + std::string(name));
    }
    const int64_t rows = shape.prefix_elements();
    if (rows <= 0 || elements_per_row <= 0 ||
        static_cast<size_t>(rows * elements_per_row) != values.size()) {
        throw std::runtime_error("quantized tensor shape does not match F32 value count: " + std::string(name));
    }
    std::vector<std::byte> bytes(static_cast<size_t>(rows) * ggml_row_size(type, elements_per_row));
    const size_t written = ggml_quantize_chunk(
        type,
        values.data(),
        bytes.data(),
        0,
        rows,
        elements_per_row,
        nullptr);
    if (written != bytes.size()) {
        throw std::runtime_error("quantized tensor byte size mismatch: " + std::string(name));
    }
    return bytes;
}

void set_tensor_bytes(ggml_tensor * tensor, const void * data, size_t bytes, std::string_view name) {
    if (tensor == nullptr) {
        throw std::runtime_error("cannot upload to a null backend tensor: " + std::string(name));
    }
    if (bytes != ggml_nbytes(tensor)) {
        throw std::runtime_error("backend tensor byte size mismatch for " + std::string(name));
    }
    ggml_backend_tensor_set(tensor, data, 0, bytes);
}

std::vector<float> decode_tensor_data_f32(std::string_view name, const TensorData & tensor) {
    if (tensor.type == GGML_TYPE_F32) {
        if (tensor.bytes.size() != static_cast<size_t>(tensor.shape.num_elements()) * sizeof(float)) {
            throw std::runtime_error("invalid F32 tensor byte size: " + std::string(name));
        }
        std::vector<float> values(static_cast<size_t>(tensor.shape.num_elements()));
        std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
        return values;
    }
    if (tensor.type == GGML_TYPE_F16) {
        if (tensor.bytes.size() != static_cast<size_t>(tensor.shape.num_elements()) * sizeof(ggml_fp16_t)) {
            throw std::runtime_error("invalid F16 tensor byte size: " + std::string(name));
        }
        std::vector<float> values(static_cast<size_t>(tensor.shape.num_elements()));
        ggml_fp16_to_fp32_row(
            reinterpret_cast<const ggml_fp16_t *>(tensor.bytes.data()),
            values.data(),
            tensor.shape.num_elements());
        return values;
    }
    if (tensor.type == GGML_TYPE_BF16) {
        if (tensor.bytes.size() != static_cast<size_t>(tensor.shape.num_elements()) * sizeof(ggml_bf16_t)) {
            throw std::runtime_error("invalid BF16 tensor byte size: " + std::string(name));
        }
        std::vector<float> values(static_cast<size_t>(tensor.shape.num_elements()));
        ggml_bf16_to_fp32_row(
            reinterpret_cast<const ggml_bf16_t *>(tensor.bytes.data()),
            values.data(),
            tensor.shape.num_elements());
        return values;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor.type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error("tensor type is not readable as F32 data: " + std::string(name));
    }
    if (tensor.shape.rank < 2) {
        throw std::runtime_error("quantized tensor must have rank >= 2: " + std::string(name));
    }
    const int64_t cols = tensor.shape.last_dim();
    const int64_t rows = tensor.shape.prefix_elements();
    const size_t row_bytes = ggml_row_size(tensor.type, cols);
    if (tensor.bytes.size() != static_cast<size_t>(rows) * row_bytes) {
        throw std::runtime_error("quantized tensor byte size mismatch: " + std::string(name));
    }
    std::vector<float> values(static_cast<size_t>(rows * cols));
    const std::byte * src = tensor.bytes.data();
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(
            src + static_cast<std::ptrdiff_t>(row * static_cast<int64_t>(row_bytes)),
            values.data() + static_cast<std::ptrdiff_t>(row * cols),
            cols);
    }
    return values;
}

void set_backend_tensor_from_f32(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type) {
    if (type == GGML_TYPE_F32) {
        set_tensor_bytes(tensor, values.data(), values.size() * sizeof(float), name);
        return;
    }
    if (type == GGML_TYPE_F16) {
        const auto bytes = f16_bytes(values);
        set_tensor_bytes(tensor, bytes.data(), bytes.size(), name);
        return;
    }
    if (type == GGML_TYPE_BF16) {
        const auto bytes = bf16_bytes(values);
        set_tensor_bytes(tensor, bytes.data(), bytes.size(), name);
        return;
    }
    const auto bytes = quantize_f32_rows(name, values, shape, type);
    set_tensor_bytes(tensor, bytes.data(), bytes.size(), name);
}

}  // namespace

void set_backend_tensor_from_f32_parallel(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type) {
    if (static_cast<int64_t>(values.size()) < kParallelF32ConvertElements ||
        (type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)) {
        set_backend_tensor_from_f32(tensor, name, values, shape, type);
        return;
    }

    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(values.size());
        const int64_t count = static_cast<int64_t>(values.size());
        const int64_t chunks = (count + kF32ConvertChunkElements - 1) / kF32ConvertChunkElements;
        #pragma omp parallel for schedule(static)
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            const int64_t offset = chunk * kF32ConvertChunkElements;
            const int64_t length = std::min(kF32ConvertChunkElements, count - offset);
            ggml_fp32_to_fp16_row(values.data() + offset, converted.data() + offset, length);
        }
        set_tensor_bytes(tensor, converted.data(), converted.size() * sizeof(ggml_fp16_t), name);
        return;
    }

    std::vector<ggml_bf16_t> converted(values.size());
    const int64_t count = static_cast<int64_t>(values.size());
    const int64_t chunks = (count + kF32ConvertChunkElements - 1) / kF32ConvertChunkElements;
    #pragma omp parallel for schedule(static)
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int64_t offset = chunk * kF32ConvertChunkElements;
        const int64_t length = std::min(kF32ConvertChunkElements, count - offset);
        ggml_fp32_to_bf16_row(values.data() + offset, converted.data() + offset, length);
    }
    set_tensor_bytes(tensor, converted.data(), converted.size() * sizeof(ggml_bf16_t), name);
}

namespace {

std::vector<std::byte> encode_f32_tensor_data(
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return f32_bytes(values);
    }
    if (type == GGML_TYPE_F16) {
        return f16_bytes(values);
    }
    if (type == GGML_TYPE_BF16) {
        return bf16_bytes(values);
    }
    return quantize_f32_rows(name, values, shape, type);
}

class SafeTensorSource final : public TensorSource {
public:
    explicit SafeTensorSource(std::filesystem::path path)
        : index_(engine::io::load_safetensors_index(path)),
          bytes_(engine::io::read_binary_blob(path)) {}

    const std::filesystem::path & source_path() const noexcept override {
        return index_.source_path;
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return index_.tensors.find(std::string(name)) != index_.tensors.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        return TensorMetadata{info->name, info->dtype, info->shape};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(index_.tensors.size());
        for (const auto & [name, info] : index_.tensors) {
            out.push_back({name, info.dtype, info.shape});
        }
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override {
        bytes_ = engine::io::BinaryBlob();
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        const auto [data, byte_size] = require_data_range(*info);
        RawTensorData tensor;
        tensor.metadata = TensorMetadata{info->name, info->dtype, info->shape};
        tensor.bytes.resize(byte_size);
        std::memcpy(tensor.bytes.data(), data, byte_size);
        discard_data_range(*info);
        return tensor;
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        validate_expected_shape(name, info->shape, expected_shape);
        const auto shape = shape_from_dims(expected_shape);
        const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
        const auto [data, byte_size] = require_data_range(*info);
        if (raw_dtype_matches_ggml_type(info->dtype, type)) {
            validate_raw_tensor_byte_size(name, shape, type, byte_size);
            set_tensor_bytes(tensor, data, byte_size, name);
            discard_data_range(*info);
            return;
        }
        const ggml_type raw_type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(info->dtype));
        std::vector<std::byte> raw_bytes(byte_size);
        std::memcpy(raw_bytes.data(), data, byte_size);
        const auto values = decode_tensor_data_f32(name, TensorData{shape, raw_type, std::move(raw_bytes)});
        set_backend_tensor_from_f32(tensor, name, values, shape, type);
        discard_data_range(*info);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        set_backend_tensor(tensor, name, TensorStorageType::F32, expected_shape);
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto tensor = require_tensor_data(name);
        validate_expected_shape(name, tensor.metadata.shape, expected_shape);
        const ggml_type type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(tensor.metadata.dtype));
        return decode_tensor_data_f32(name, TensorData{shape_from_dims(tensor.metadata.shape), type, tensor.bytes});
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
        const auto * info = find_info(name);
        if (info == nullptr) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        if (info->dtype != "I64" || info->data_end - info->data_begin != sizeof(int64_t)) {
            throw std::runtime_error("tensor is not an I64 scalar: " + std::string(name));
        }
        const auto [data, byte_size] = require_data_range(*info);
        (void) byte_size;
        int64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

private:
    const engine::io::SafeTensorInfo * find_info(std::string_view name) const noexcept {
        const auto it = index_.tensors.find(std::string(name));
        if (it == index_.tensors.end()) {
            return nullptr;
        }
        return &it->second;
    }

    std::pair<const std::byte *, size_t> require_data_range(const engine::io::SafeTensorInfo & info) const {
        if (bytes_.empty()) {
            bytes_ = engine::io::read_binary_blob(index_.source_path);
        }
        const size_t data_offset = index_.header_bytes + info.data_begin;
        const size_t byte_size = info.data_end - info.data_begin;
        if (data_offset + byte_size > bytes_.size()) {
            throw std::runtime_error("tensor data range is out of bounds: " + info.name);
        }
        return {bytes_.data() + static_cast<std::ptrdiff_t>(data_offset), byte_size};
    }

    void discard_data_range(const engine::io::SafeTensorInfo & info) const noexcept {
        bytes_.discard_range(index_.header_bytes + info.data_begin, info.data_end - info.data_begin);
    }

    engine::io::SafeTensorIndex index_;
    mutable engine::io::BinaryBlob bytes_;
};

std::unordered_map<std::string, std::string> parse_indexed_tensor_weight_map(
    const std::filesystem::path & index_path) {
    const auto root = engine::io::json::parse_file(index_path);
    const auto & object = root.require("weight_map").as_object();
    std::unordered_map<std::string, std::string> weight_map;
    weight_map.reserve(object.size());
    for (const auto & [name, value] : object) {
        weight_map.emplace(name, value.as_string());
    }
    if (weight_map.empty()) {
        throw std::runtime_error("indexed tensor source has an empty weight_map: " + index_path.string());
    }
    return weight_map;
}

std::vector<std::filesystem::path> indexed_tensor_source_shard_paths_from_weight_map(
    const std::filesystem::path & model_root,
    const std::unordered_map<std::string, std::string> & weight_map) {
    std::set<std::string> shard_names;
    for (const auto & [_, file_name] : weight_map) {
        shard_names.insert(file_name);
    }
    std::vector<std::filesystem::path> paths;
    paths.reserve(shard_names.size());
    for (const auto & name : shard_names) {
        const auto path = model_root / name;
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("missing indexed tensor shard: " + path.string());
        }
        paths.push_back(std::filesystem::weakly_canonical(path));
    }
    return paths;
}

class IndexedTensorSource final : public TensorSource {
public:
    IndexedTensorSource(
        std::filesystem::path index_path,
        std::unordered_map<std::string, std::string> weight_map,
        std::unordered_map<std::string, std::shared_ptr<const TensorSource>> shard_sources)
        : index_path_(std::move(index_path)),
          weight_map_(std::move(weight_map)),
          shard_sources_(std::move(shard_sources)) {}

    const std::filesystem::path & source_path() const noexcept override {
        return index_path_;
    }

    bool has_tensor(std::string_view name) const noexcept override {
        const auto route = weight_map_.find(std::string(name));
        if (route == weight_map_.end()) {
            return false;
        }
        const auto source = shard_sources_.find(route->second);
        return source != shard_sources_.end() && source->second->has_tensor(name);
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        return source_for(name)->require_metadata(name);
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(weight_map_.size());
        for (const auto & [name, _] : weight_map_) {
            out.push_back(require_metadata(name));
        }
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override {
        for (const auto & [_, source] : shard_sources_) {
            source->release_storage();
        }
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        return source_for(name)->require_tensor_data(name);
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return source_for(name)->require_f32(name, expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        source_for(name)->set_backend_tensor(tensor, name, storage_type, expected_shape);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        source_for(name)->set_backend_f32_tensor(tensor, name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return source_for(name)->require_i64_scalar(name);
    }

private:
    std::shared_ptr<const TensorSource> source_for(std::string_view name) const {
        const auto route = weight_map_.find(std::string(name));
        if (route == weight_map_.end()) {
            throw std::runtime_error("missing indexed tensor route: " + std::string(name));
        }
        const auto source = shard_sources_.find(route->second);
        if (source == shard_sources_.end()) {
            throw std::runtime_error("missing indexed tensor shard source: " + route->second);
        }
        return source->second;
    }

    std::filesystem::path index_path_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::unordered_map<std::string, std::shared_ptr<const TensorSource>> shard_sources_;
};

}  // namespace

TensorDataF32 TensorSource::require_f32_tensor(std::string_view name) const {
    const auto metadata = require_metadata(name);
    return TensorDataF32{
        shape_from_dims(metadata.shape),
        require_f32(name, metadata.shape),
    };
}

TensorDataF32 TensorSource::require_f32_tensor(
    std::string_view name,
    std::initializer_list<int64_t> expected_shape) const {
    const auto values = require_f32(name, expected_shape);
    return TensorDataF32{
        shape_from_dims(std::vector<int64_t>(expected_shape)),
        std::move(values),
    };
}

TensorData TensorSource::require_tensor(
    std::string_view name,
    TensorStorageType storage_type,
    std::initializer_list<int64_t> expected_shape) const {
    return require_tensor_as_shape(name, storage_type, expected_shape, expected_shape);
}

TensorData TensorSource::require_tensor(
    std::string_view name,
    TensorStorageType storage_type,
    const std::vector<int64_t> & expected_shape) const {
    const core::TensorShape shape = shape_from_dims(expected_shape);
    const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
    const auto raw = require_tensor_data(name);
    validate_expected_shape(name, raw.metadata.shape, expected_shape);
    if (raw_dtype_matches_ggml_type(raw.metadata.dtype, type)) {
        validate_raw_tensor_byte_size(name, shape, type, raw.bytes.size());
        return TensorData{shape, type, raw.bytes};
    }
    const ggml_type raw_type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(raw.metadata.dtype));
    const auto values = decode_tensor_data_f32(name, TensorData{shape, raw_type, raw.bytes});
    return TensorData{shape, type, encode_f32_tensor_data(name, values, shape, type)};
}

TensorData TensorSource::require_tensor_as_shape(
    std::string_view name,
    TensorStorageType storage_type,
    std::initializer_list<int64_t> expected_source_shape,
    std::initializer_list<int64_t> tensor_shape) const {
    const std::vector<int64_t> expected(expected_source_shape);
    const core::TensorShape shape = shape_from_dims(std::vector<int64_t>(tensor_shape));
    if (shape.num_elements() != std::accumulate(expected.begin(), expected.end(), int64_t{1}, std::multiplies<int64_t>())) {
        throw std::runtime_error("tensor source shape element count mismatch for " + std::string(name));
    }
    const core::TensorShape source_shape = shape_from_dims(expected);
    const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
    const auto raw = require_tensor_data(name);
    validate_expected_shape(name, raw.metadata.shape, expected);
    if (raw.metadata.shape == std::vector<int64_t>(tensor_shape) &&
        raw_dtype_matches_ggml_type(raw.metadata.dtype, type)) {
        validate_raw_tensor_byte_size(name, shape, type, raw.bytes.size());
        return TensorData{shape, type, raw.bytes};
    }
    const ggml_type raw_type = ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(raw.metadata.dtype));
    const auto values = decode_tensor_data_f32(name, TensorData{source_shape, raw_type, raw.bytes});
    return TensorData{shape, type, encode_f32_tensor_data(name, values, shape, type)};
}

void TensorSource::set_backend_tensor(
    ggml_tensor * tensor,
    std::string_view name,
    TensorStorageType storage_type,
    const std::vector<int64_t> & expected_shape) const {
    TensorData data;
    switch (expected_shape.size()) {
        case 1:
            data = require_tensor(name, storage_type, {expected_shape[0]});
            break;
        case 2:
            data = require_tensor(name, storage_type, {expected_shape[0], expected_shape[1]});
            break;
        case 3:
            data = require_tensor(name, storage_type, {expected_shape[0], expected_shape[1], expected_shape[2]});
            break;
        case 4:
            data = require_tensor(
                name,
                storage_type,
                {expected_shape[0], expected_shape[1], expected_shape[2], expected_shape[3]});
            break;
        default:
            throw std::runtime_error("tensor rank must be between 1 and 4");
    }
    set_tensor_bytes(tensor, data.bytes.data(), data.bytes.size(), name);
}

void TensorSource::set_backend_f32_tensor(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<int64_t> & expected_shape) const {
    const auto values = require_f32(name, std::optional<std::vector<int64_t>>(expected_shape));
    set_tensor_bytes(tensor, values.data(), values.size() * sizeof(float), name);
}

std::optional<TensorDataF32> TensorSource::optional_f32_tensor(std::string_view name) const {
    if (!has_tensor(name)) {
        return std::nullopt;
    }
    return require_f32_tensor(name);
}

std::optional<TensorDataF32> TensorSource::optional_f32_tensor(
    std::string_view name,
    std::initializer_list<int64_t> expected_shape) const {
    if (!has_tensor(name)) {
        return std::nullopt;
    }
    return require_f32_tensor(name, expected_shape);
}

std::optional<RawTensorData> TensorSource::optional_tensor_data(std::string_view name) const {
    if (!has_tensor(name)) {
        return std::nullopt;
    }
    return require_tensor_data(name);
}

std::optional<std::string> TensorSource::find_tensor_name(
    std::initializer_list<std::string_view> candidates) const {
    for (const auto candidate : candidates) {
        if (has_tensor(candidate)) {
            return std::string(candidate);
        }
    }
    return std::nullopt;
}

std::string TensorSource::require_tensor_name(
    std::initializer_list<std::string_view> candidates) const {
    const auto match = find_tensor_name(candidates);
    if (match.has_value()) {
        return *match;
    }
    throw std::runtime_error("none of the candidate tensor names were found");
}

TensorStorageType parse_tensor_storage_type(std::string_view value) {
    const std::string normalized = lower_ascii(value);
    if (normalized == "native" || normalized == "source" || normalized == "auto") {
        return TensorStorageType::Native;
    }
    if (normalized == "f32" || normalized == "float32") {
        return TensorStorageType::F32;
    }
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") {
        return TensorStorageType::F16;
    }
    if (normalized == "bf16" || normalized == "bfloat16") {
        return TensorStorageType::BF16;
    }
    if (normalized == "q4_0") {
        return TensorStorageType::Q4_0;
    }
    if (normalized == "q4_1") {
        return TensorStorageType::Q4_1;
    }
    if (normalized == "q5_0") {
        return TensorStorageType::Q5_0;
    }
    if (normalized == "q5_1") {
        return TensorStorageType::Q5_1;
    }
    if (normalized == "q4_k") {
        return TensorStorageType::Q4_K;
    }
    if (normalized == "q5_k") {
        return TensorStorageType::Q5_K;
    }
    if (normalized == "q6_k") {
        return TensorStorageType::Q6_K;
    }
    if (normalized == "q8_0") {
        return TensorStorageType::Q8_0;
    }
    throw std::runtime_error("unsupported tensor storage type: " + std::string(value));
}

ggml_type ggml_type_for_tensor_storage(TensorStorageType storage_type) {
    switch (storage_type) {
        case TensorStorageType::Native:
            throw std::runtime_error("native tensor storage type must be resolved before creating a ggml tensor");
        case TensorStorageType::F32:
            return GGML_TYPE_F32;
        case TensorStorageType::F16:
            return GGML_TYPE_F16;
        case TensorStorageType::BF16:
            return GGML_TYPE_BF16;
        case TensorStorageType::Q4_0:
            return GGML_TYPE_Q4_0;
        case TensorStorageType::Q4_1:
            return GGML_TYPE_Q4_1;
        case TensorStorageType::Q5_0:
            return GGML_TYPE_Q5_0;
        case TensorStorageType::Q5_1:
            return GGML_TYPE_Q5_1;
        case TensorStorageType::Q4_K:
            return GGML_TYPE_Q4_K;
        case TensorStorageType::Q5_K:
            return GGML_TYPE_Q5_K;
        case TensorStorageType::Q6_K:
            return GGML_TYPE_Q6_K;
        case TensorStorageType::Q8_0:
            return GGML_TYPE_Q8_0;
    }
    throw std::runtime_error("unknown tensor storage type");
}

TensorStorageType tensor_storage_type_for_dtype(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    if (normalized == "f32" || normalized == "float32") {
        return TensorStorageType::F32;
    }
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") {
        return TensorStorageType::F16;
    }
    if (normalized == "bf16" || normalized == "bfloat16") {
        return TensorStorageType::BF16;
    }
    if (normalized == "q4_0") {
        return TensorStorageType::Q4_0;
    }
    if (normalized == "q4_1") {
        return TensorStorageType::Q4_1;
    }
    if (normalized == "q5_0") {
        return TensorStorageType::Q5_0;
    }
    if (normalized == "q5_1") {
        return TensorStorageType::Q5_1;
    }
    if (normalized == "q4_k") {
        return TensorStorageType::Q4_K;
    }
    if (normalized == "q5_k") {
        return TensorStorageType::Q5_K;
    }
    if (normalized == "q6_k") {
        return TensorStorageType::Q6_K;
    }
    if (normalized == "q8_0") {
        return TensorStorageType::Q8_0;
    }
    throw std::runtime_error("unsupported native tensor dtype: " + std::string(dtype));
}

TensorStorageType resolve_tensor_storage_type(
    const TensorSource & source,
    std::string_view name,
    TensorStorageType requested_type) {
    if (requested_type != TensorStorageType::Native) {
        return requested_type;
    }
    return tensor_storage_type_for_dtype(source.require_metadata(name).dtype);
}

std::vector<float> tensor_data_to_f32(std::string_view name, const TensorData & tensor) {
    return decode_tensor_data_f32(name, tensor);
}

std::shared_ptr<const TensorSource> open_tensor_source(const std::filesystem::path & path) {
    if (path.extension() == ".safetensors") {
        return std::make_shared<SafeTensorSource>(path);
    }
    throw std::runtime_error("unsupported tensor source format: " + path.string());
}

std::vector<std::filesystem::path> indexed_tensor_source_shard_paths(
    const std::filesystem::path & index_path,
    const std::filesystem::path & model_root) {
    return indexed_tensor_source_shard_paths_from_weight_map(
        model_root,
        parse_indexed_tensor_weight_map(index_path));
}

std::shared_ptr<const TensorSource> open_indexed_tensor_source(
    const std::filesystem::path & index_path,
    const std::filesystem::path & model_root) {
    const auto weight_map = parse_indexed_tensor_weight_map(index_path);
    std::unordered_map<std::string, std::shared_ptr<const TensorSource>> shard_sources;
    for (const auto & path : indexed_tensor_source_shard_paths_from_weight_map(model_root, weight_map)) {
        shard_sources.emplace(path.filename().string(), open_tensor_source(path));
    }
    return std::make_shared<IndexedTensorSource>(
        index_path,
        weight_map,
        std::move(shard_sources));
}

}  // namespace engine::assets
