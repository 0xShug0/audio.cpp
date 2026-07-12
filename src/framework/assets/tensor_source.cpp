#include "engine/framework/assets/tensor_source.h"

#include "engine/framework/io/binary.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
           (normalized == "q2_k" && type == GGML_TYPE_Q2_K) ||
           (normalized == "q3_k" && type == GGML_TYPE_Q3_K) ||
           (normalized == "q4_k" && type == GGML_TYPE_Q4_K) ||
           (normalized == "q5_k" && type == GGML_TYPE_Q5_K) ||
           (normalized == "q6_k" && type == GGML_TYPE_Q6_K) ||
           (normalized == "q8_0" && type == GGML_TYPE_Q8_0);
}

ggml_type ggml_type_for_raw_dtype(std::string_view dtype) {
    const std::string normalized = lower_ascii(dtype);
    if (normalized == "f32" || normalized == "float32") return GGML_TYPE_F32;
    if (normalized == "f16" || normalized == "float16" || normalized == "fp16") return GGML_TYPE_F16;
    if (normalized == "bf16" || normalized == "bfloat16") return GGML_TYPE_BF16;
    if (normalized == "i8" || normalized == "int8") return GGML_TYPE_I8;
    if (normalized == "i16" || normalized == "int16") return GGML_TYPE_I16;
    if (normalized == "i32" || normalized == "int32") return GGML_TYPE_I32;
    if (normalized == "i64" || normalized == "int64") return GGML_TYPE_I64;
    if (normalized == "q4_0") return GGML_TYPE_Q4_0;
    if (normalized == "q4_1") return GGML_TYPE_Q4_1;
    if (normalized == "q5_0") return GGML_TYPE_Q5_0;
    if (normalized == "q5_1") return GGML_TYPE_Q5_1;
    if (normalized == "q8_0") return GGML_TYPE_Q8_0;
    if (normalized == "q2_k") return GGML_TYPE_Q2_K;
    if (normalized == "q3_k") return GGML_TYPE_Q3_K;
    if (normalized == "q4_k") return GGML_TYPE_Q4_K;
    if (normalized == "q5_k") return GGML_TYPE_Q5_K;
    if (normalized == "q6_k") return GGML_TYPE_Q6_K;
    throw std::runtime_error("unsupported tensor dtype for GGUF: " + std::string(dtype));
}

std::string dtype_for_ggml_type(ggml_type type) {
    const char * name = ggml_type_name(type);
    if (name == nullptr) {
        throw std::runtime_error("GGUF tensor has an unknown ggml type");
    }
    return lower_ascii(name);
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

struct GgufTensorInfo {
    std::string logical_name;
    std::string physical_name;
    std::string dtype;
    std::vector<int64_t> shape;
    ggml_type type = GGML_TYPE_F32;
    size_t data_offset = 0;
    size_t byte_size = 0;
};

class GgufTensorSource final : public TensorSource {
public:
    explicit GgufTensorSource(std::filesystem::path path)
        : source_path_(std::filesystem::weakly_canonical(path)) {
        ggml_context * tensor_context = nullptr;
        gguf_context * gguf = gguf_init_from_file(
            source_path_.string().c_str(),
            gguf_init_params{true, &tensor_context});
        if (gguf == nullptr || tensor_context == nullptr) {
            if (gguf != nullptr) gguf_free(gguf);
            if (tensor_context != nullptr) ggml_free(tensor_context);
            throw std::runtime_error("failed to read GGUF tensor source: " + source_path_.string());
        }
        try {
            data_begin_ = gguf_get_data_offset(gguf);
            const int64_t logical_names_key = gguf_find_key(gguf, "audiocpp.tensor_names");
            if (logical_names_key >= 0 &&
                (gguf_get_kv_type(gguf, logical_names_key) != GGUF_TYPE_ARRAY ||
                 gguf_get_arr_type(gguf, logical_names_key) != GGUF_TYPE_STRING ||
                 gguf_get_arr_n(gguf, logical_names_key) != static_cast<size_t>(gguf_get_n_tensors(gguf)))) {
                throw std::runtime_error("GGUF audiocpp.tensor_names metadata is invalid");
            }
            const int64_t count = gguf_get_n_tensors(gguf);
            const int64_t ranks_key = gguf_find_key(gguf, "audiocpp.tensor_ranks");
            const int64_t shapes_key = gguf_find_key(gguf, "audiocpp.tensor_shapes");
            const bool has_exact_shapes = ranks_key >= 0 && shapes_key >= 0;
            if ((ranks_key >= 0) != (shapes_key >= 0) ||
                (has_exact_shapes &&
                 (gguf_get_kv_type(gguf, ranks_key) != GGUF_TYPE_ARRAY ||
                  gguf_get_arr_type(gguf, ranks_key) != GGUF_TYPE_INT32 ||
                  gguf_get_arr_n(gguf, ranks_key) != static_cast<size_t>(count) ||
                  gguf_get_kv_type(gguf, shapes_key) != GGUF_TYPE_ARRAY ||
                  gguf_get_arr_type(gguf, shapes_key) != GGUF_TYPE_INT64))) {
                throw std::runtime_error("GGUF audiocpp exact tensor shape metadata is invalid");
            }
            size_t shape_cursor = 0;
            infos_.reserve(static_cast<size_t>(count));
            for (int64_t i = 0; i < count; ++i) {
                GgufTensorInfo info;
                info.physical_name = gguf_get_tensor_name(gguf, i);
                info.logical_name = logical_names_key >= 0
                    ? gguf_get_arr_str(gguf, logical_names_key, static_cast<size_t>(i))
                    : info.physical_name;
                info.type = gguf_get_tensor_type(gguf, i);
                info.dtype = dtype_for_ggml_type(info.type);
                info.data_offset = gguf_get_tensor_offset(gguf, i);
                info.byte_size = gguf_get_tensor_size(gguf, i);
                const ggml_tensor * tensor = ggml_get_tensor(tensor_context, info.physical_name.c_str());
                if (tensor == nullptr) {
                    throw std::runtime_error("GGUF tensor metadata is missing: " + info.physical_name);
                }
                if (has_exact_shapes) {
                    const auto * ranks = static_cast<const int32_t *>(gguf_get_arr_data(gguf, ranks_key));
                    const auto * shapes = static_cast<const int64_t *>(gguf_get_arr_data(gguf, shapes_key));
                    const int32_t tensor_rank = ranks[i];
                    if (tensor_rank < 1 || tensor_rank > static_cast<int32_t>(core::kMaxTensorRank) ||
                        shape_cursor + static_cast<size_t>(tensor_rank) > gguf_get_arr_n(gguf, shapes_key)) {
                        throw std::runtime_error("GGUF contains invalid exact tensor dimensions");
                    }
                    info.shape.assign(shapes + shape_cursor, shapes + shape_cursor + tensor_rank);
                    shape_cursor += static_cast<size_t>(tensor_rank);
                } else {
                    const int dimensions = ggml_n_dims(tensor);
                    info.shape.reserve(static_cast<size_t>(dimensions));
                    for (int dim = dimensions - 1; dim >= 0; --dim) {
                        info.shape.push_back(tensor->ne[dim]);
                    }
                }
                if (!info_by_name_.emplace(info.logical_name, infos_.size()).second) {
                    throw std::runtime_error("GGUF contains duplicate logical tensor name: " + info.logical_name);
                }
                infos_.push_back(std::move(info));
            }
            if (has_exact_shapes && shape_cursor != gguf_get_arr_n(gguf, shapes_key)) {
                throw std::runtime_error("GGUF exact tensor shape metadata has trailing dimensions");
            }
        } catch (...) {
            gguf_free(gguf);
            ggml_free(tensor_context);
            throw;
        }
        gguf_free(gguf);
        ggml_free(tensor_context);
        bytes_ = engine::io::read_binary_blob(source_path_);
    }

    const std::filesystem::path & source_path() const noexcept override { return source_path_; }

    bool has_tensor(std::string_view name) const noexcept override {
        return info_by_name_.find(std::string(name)) != info_by_name_.end();
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const auto & info = require_info(name);
        return {info.logical_name, info.dtype, info.shape};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(infos_.size());
        for (const auto & info : infos_) out.push_back({info.logical_name, info.dtype, info.shape});
        std::sort(out.begin(), out.end(), [](const TensorMetadata & lhs, const TensorMetadata & rhs) {
            return lhs.name < rhs.name;
        });
        return out;
    }

    void release_storage() const override { bytes_ = engine::io::BinaryBlob(); }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & info = require_info(name);
        const auto [data, byte_size] = require_data_range(info);
        RawTensorData out;
        out.metadata = {info.logical_name, info.dtype, info.shape};
        out.bytes.resize(byte_size);
        std::memcpy(out.bytes.data(), data, byte_size);
        bytes_.discard_range(data_begin_ + info.data_offset, byte_size);
        return out;
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const auto & info = require_info(name);
        validate_expected_shape(name, info.shape, expected_shape);
        const auto shape = shape_from_dims(expected_shape);
        const ggml_type type = ggml_type_for_tensor_storage(resolve_tensor_storage_type(*this, name, storage_type));
        const auto [data, byte_size] = require_data_range(info);
        if (info.type == type) {
            validate_raw_tensor_byte_size(name, shape, type, byte_size);
            set_tensor_bytes(tensor, data, byte_size, name);
            bytes_.discard_range(data_begin_ + info.data_offset, byte_size);
            return;
        }
        std::vector<std::byte> raw_bytes(byte_size);
        std::memcpy(raw_bytes.data(), data, byte_size);
        const auto values = decode_tensor_data_f32(name, TensorData{shape, info.type, std::move(raw_bytes)});
        set_backend_tensor_from_f32(tensor, name, values, shape, type);
        bytes_.discard_range(data_begin_ + info.data_offset, byte_size);
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
        return decode_tensor_data_f32(
            name,
            TensorData{shape_from_dims(tensor.metadata.shape), require_info(name).type, tensor.bytes});
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        const auto & info = require_info(name);
        if (info.type != GGML_TYPE_I64 || info.byte_size != sizeof(int64_t)) {
            throw std::runtime_error("tensor is not an I64 scalar: " + std::string(name));
        }
        const auto [data, byte_size] = require_data_range(info);
        (void) byte_size;
        int64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

private:
    const GgufTensorInfo & require_info(std::string_view name) const {
        const auto it = info_by_name_.find(std::string(name));
        if (it == info_by_name_.end()) throw std::runtime_error("missing tensor: " + std::string(name));
        return infos_[it->second];
    }

    std::pair<const std::byte *, size_t> require_data_range(const GgufTensorInfo & info) const {
        if (bytes_.empty()) bytes_ = engine::io::read_binary_blob(source_path_);
        const size_t offset = data_begin_ + info.data_offset;
        if (offset > bytes_.size() || info.byte_size > bytes_.size() - offset) {
            throw std::runtime_error("GGUF tensor data range is out of bounds: " + info.logical_name);
        }
        return {bytes_.data() + static_cast<std::ptrdiff_t>(offset), info.byte_size};
    }

    std::filesystem::path source_path_;
    size_t data_begin_ = 0;
    std::vector<GgufTensorInfo> infos_;
    std::unordered_map<std::string, size_t> info_by_name_;
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
    if (normalized == "q2_k") {
        return TensorStorageType::Q2_K;
    }
    if (normalized == "q3_k") {
        return TensorStorageType::Q3_K;
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
        case TensorStorageType::Q2_K:
            return GGML_TYPE_Q2_K;
        case TensorStorageType::Q3_K:
            return GGML_TYPE_Q3_K;
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
    if (normalized == "q2_k") {
        return TensorStorageType::Q2_K;
    }
    if (normalized == "q3_k") {
        return TensorStorageType::Q3_K;
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
    const std::string file_name = lower_ascii(path.filename().string());
    static constexpr std::string_view kIndexSuffix = ".safetensors.index.json";
    if (file_name.size() >= kIndexSuffix.size() &&
        file_name.compare(file_name.size() - kIndexSuffix.size(), kIndexSuffix.size(), kIndexSuffix) == 0) {
        return open_indexed_tensor_source(path, path.parent_path());
    }
    const std::string extension = lower_ascii(path.extension().string());
    if (extension == ".safetensors") {
        return std::make_shared<SafeTensorSource>(path);
    }
    if (extension == ".gguf") {
        return std::make_shared<GgufTensorSource>(path);
    }
    throw std::runtime_error("unsupported tensor source format: " + path.string());
}

namespace {

std::vector<std::pair<std::string, std::string>> read_gguf_embedded_sidecars(
    const std::filesystem::path & path) {
    ggml_context * tensor_context = nullptr;
    gguf_context * gguf = gguf_init_from_file(
        path.string().c_str(), gguf_init_params{true, &tensor_context});
    if (gguf == nullptr) {
        if (tensor_context != nullptr) ggml_free(tensor_context);
        throw std::runtime_error("failed to read GGUF metadata: " + path.string());
    }
    std::vector<std::pair<std::string, std::string>> result;
    try {
        const int64_t names_key = gguf_find_key(gguf, "audiocpp.embedded_files.names");
        const int64_t contents_key = gguf_find_key(gguf, "audiocpp.embedded_files.contents");
        const int64_t offsets_key = gguf_find_key(gguf, "audiocpp.embedded_files.offsets");
        const int64_t data_key = gguf_find_key(gguf, "audiocpp.embedded_files.data");
        const bool blob_layout = offsets_key >= 0 || data_key >= 0;
        if (names_key < 0 && contents_key < 0 && !blob_layout) {
            gguf_free(gguf);
            if (tensor_context != nullptr) ggml_free(tensor_context);
            return result;
        }
        if (blob_layout) {
            if (names_key < 0 || offsets_key < 0 || data_key < 0 ||
                gguf_get_kv_type(gguf, names_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, names_key) != GGUF_TYPE_STRING ||
                gguf_get_kv_type(gguf, offsets_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, offsets_key) != GGUF_TYPE_UINT64 ||
                gguf_get_arr_n(gguf, offsets_key) != gguf_get_arr_n(gguf, names_key) + 1 ||
                gguf_get_kv_type(gguf, data_key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, data_key) != GGUF_TYPE_UINT8) {
                throw std::runtime_error("GGUF embedded binary sidecar metadata is invalid");
            }
        } else if (names_key < 0 || contents_key < 0 ||
            gguf_get_kv_type(gguf, names_key) != GGUF_TYPE_ARRAY ||
            gguf_get_kv_type(gguf, contents_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(gguf, names_key) != GGUF_TYPE_STRING ||
            gguf_get_arr_type(gguf, contents_key) != GGUF_TYPE_STRING ||
            gguf_get_arr_n(gguf, names_key) != gguf_get_arr_n(gguf, contents_key)) {
            throw std::runtime_error("GGUF embedded sidecar metadata is invalid");
        }
        const size_t count = gguf_get_arr_n(gguf, names_key);
        const auto * offsets = blob_layout
            ? static_cast<const uint64_t *>(gguf_get_arr_data(gguf, offsets_key))
            : nullptr;
        const auto * data = blob_layout
            ? static_cast<const char *>(gguf_get_arr_data(gguf, data_key))
            : nullptr;
        const size_t data_size = blob_layout ? gguf_get_arr_n(gguf, data_key) : 0;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            std::string name = gguf_get_arr_str(gguf, names_key, i);
            const std::filesystem::path relative(name);
            const auto normalized = relative.lexically_normal();
            if (name.empty() || relative.is_absolute() || normalized.empty() ||
                *normalized.begin() == "..") {
                throw std::runtime_error("GGUF contains an unsafe embedded sidecar name: " + name);
            }
            if (blob_layout) {
                if (offsets[i] > offsets[i + 1] || offsets[i + 1] > data_size) {
                    throw std::runtime_error("GGUF embedded sidecar byte range is invalid");
                }
                std::string content;
                const size_t length = static_cast<size_t>(offsets[i + 1] - offsets[i]);
                if (length > 0) content.assign(data + offsets[i], length);
                result.emplace_back(normalized.generic_string(), std::move(content));
            } else {
                result.emplace_back(normalized.generic_string(), gguf_get_arr_str(gguf, contents_key, i));
            }
        }
    } catch (...) {
        gguf_free(gguf);
        if (tensor_context != nullptr) ggml_free(tensor_context);
        throw;
    }
    gguf_free(gguf);
    if (tensor_context != nullptr) ggml_free(tensor_context);
    return result;
}

}  // namespace

bool gguf_has_embedded_sidecars(const std::filesystem::path & path) {
    return !read_gguf_embedded_sidecars(path).empty();
}

std::filesystem::path materialize_gguf_sidecars(const std::filesystem::path & path) {
    const auto canonical = std::filesystem::weakly_canonical(path);
    const auto sidecars = read_gguf_embedded_sidecars(canonical);
    if (sidecars.empty()) {
        throw std::runtime_error("GGUF does not contain embedded model sidecars: " + canonical.string());
    }
    const auto fingerprint_source = canonical.string() + ":" +
        std::to_string(std::filesystem::file_size(canonical)) + ":" +
        std::to_string(std::filesystem::last_write_time(canonical).time_since_epoch().count());
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::hash<std::string>{}(fingerprint_source);
    const auto root = std::filesystem::temp_directory_path() / "audiocpp-gguf" / fingerprint.str();
    std::filesystem::create_directories(root);
    for (const auto & [name, content] : sidecars) {
        const auto output_path = root / name;
        if (engine::io::is_existing_file(output_path) &&
            engine::io::read_text_file(output_path) == content) {
            continue;
        }
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("failed to materialize GGUF sidecar: " + output_path.string());
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!output) throw std::runtime_error("failed to write GGUF sidecar: " + output_path.string());
    }
    return root;
}

PreparedModelDirectory prepare_model_directory(
    const std::filesystem::path & model_path,
    const std::filesystem::path & gguf_relative_path) {
    std::filesystem::path gguf_path;
    if (engine::io::is_existing_directory(model_path)) {
        gguf_path = model_path / gguf_relative_path;
    } else if (engine::io::is_existing_file(model_path)) {
        gguf_path = model_path;
    } else {
        throw std::runtime_error("model path does not exist: " + model_path.string());
    }
    if (engine::io::is_existing_file(gguf_path) &&
        lower_ascii(gguf_path.extension().string()) == ".gguf" &&
        gguf_has_embedded_sidecars(gguf_path)) {
        return {
            materialize_gguf_sidecars(gguf_path),
            std::filesystem::weakly_canonical(gguf_path),
        };
    }
    const auto root = engine::io::is_existing_directory(model_path)
        ? model_path
        : model_path.parent_path();
    return {std::filesystem::weakly_canonical(root), std::nullopt};
}

void convert_tensor_source_to_gguf(
    const std::filesystem::path & input_path,
    const std::filesystem::path & output_path,
    TensorStorageType weight_type,
    bool overwrite,
    bool embed_sidecars) {
    if (weight_type == TensorStorageType::Native) {
        throw std::runtime_error("GGUF conversion requires an explicit output weight type");
    }
    if (engine::io::is_existing_file(output_path) && !overwrite) {
        throw std::runtime_error("GGUF output already exists: " + output_path.string());
    }
    auto source = open_tensor_source(input_path);
    const auto metadata = source->tensors();
    if (metadata.empty()) {
        throw std::runtime_error("cannot convert an empty tensor source to GGUF");
    }

    const ggml_type requested_type = ggml_type_for_tensor_storage(weight_type);
    if (ggml_is_quantized(requested_type) && ggml_quantize_requires_imatrix(requested_type)) {
        throw std::runtime_error("requested GGUF quantization requires an importance matrix");
    }

    const size_t context_bytes = std::max<size_t>(
        1024 * 1024,
        metadata.size() * (ggml_tensor_overhead() + 256));
    ggml_context * tensor_context = ggml_init(ggml_init_params{context_bytes, nullptr, true});
    if (tensor_context == nullptr) {
        throw std::runtime_error("failed to allocate GGUF tensor metadata context");
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(tensor_context);
        throw std::runtime_error("failed to allocate GGUF context");
    }

    struct OutputTensor {
        TensorMetadata metadata;
        std::string physical_name;
        ggml_type type = GGML_TYPE_F32;
        TensorStorageType storage_type = TensorStorageType::Native;
    };
    std::vector<OutputTensor> outputs;
    outputs.reserve(metadata.size());
    std::vector<std::string> logical_names;
    logical_names.reserve(metadata.size());
    std::unordered_set<std::string> physical_names;

    try {
        gguf_set_val_str(gguf, "general.architecture", "audiocpp");
        gguf_set_val_str(gguf, "general.name", input_path.stem().string().c_str());
        gguf_set_val_str(gguf, "audiocpp.tensor_name_format", "native");
        gguf_set_val_str(gguf, "audiocpp.source_format", lower_ascii(input_path.extension().string()).c_str());
        gguf_set_val_str(gguf, "audiocpp.weight_type", lower_ascii(ggml_type_name(requested_type)).c_str());

        std::vector<std::string> embedded_names;
        std::vector<const char *> embedded_name_ptrs;
        std::vector<uint64_t> embedded_offsets{0};
        std::vector<uint8_t> embedded_data;
        if (embed_sidecars) {
            constexpr uintmax_t kMaxEmbeddedSidecarBytes = 64u * 1024u * 1024u;
            std::vector<std::filesystem::path> sidecar_paths;
            std::error_code walk_error;
            for (std::filesystem::recursive_directory_iterator it(input_path.parent_path(), walk_error), end;
                 !walk_error && it != end; it.increment(walk_error)) {
                if (!it->is_regular_file()) continue;
                const auto & path = it->path();
                const std::string extension = lower_ascii(path.extension().string());
                if (extension == ".safetensors" || extension == ".gguf" || extension == ".bin" ||
                    extension == ".pt" || extension == ".pth" || path == output_path ||
                    it->file_size() > kMaxEmbeddedSidecarBytes) {
                    continue;
                }
                sidecar_paths.push_back(path);
            }
            if (walk_error) {
                throw std::runtime_error("failed to enumerate model sidecars: " + walk_error.message());
            }
            std::sort(sidecar_paths.begin(), sidecar_paths.end());
            for (const auto & path : sidecar_paths) {
                const auto relative = std::filesystem::relative(path, input_path.parent_path());
                const std::string content = engine::io::read_text_file(path);
                embedded_names.push_back(relative.generic_string());
                embedded_data.insert(embedded_data.end(), content.begin(), content.end());
                embedded_offsets.push_back(static_cast<uint64_t>(embedded_data.size()));
            }
            embedded_name_ptrs.reserve(embedded_names.size());
            for (const auto & value : embedded_names) embedded_name_ptrs.push_back(value.c_str());
            if (!embedded_names.empty()) {
                gguf_set_arr_str(gguf, "audiocpp.embedded_files.names", embedded_name_ptrs.data(), embedded_name_ptrs.size());
                gguf_set_arr_data(gguf, "audiocpp.embedded_files.offsets", GGUF_TYPE_UINT64,
                    embedded_offsets.data(), embedded_offsets.size());
                gguf_set_arr_data(gguf, "audiocpp.embedded_files.data", GGUF_TYPE_UINT8,
                    embedded_data.data(), embedded_data.size());
            }
        }

        for (size_t i = 0; i < metadata.size(); ++i) {
            const auto & item = metadata[i];
            if (item.shape.empty() || item.shape.size() > core::kMaxTensorRank) {
                throw std::runtime_error("GGUF supports tensor ranks from 1 to 4: " + item.name);
            }
            const ggml_type source_type = ggml_type_for_raw_dtype(item.dtype);
            const bool source_is_float = source_type == GGML_TYPE_F32 || source_type == GGML_TYPE_F16 ||
                source_type == GGML_TYPE_BF16 || ggml_is_quantized(source_type);
            const std::string normalized_name = lower_ascii(item.name);
            const bool name_is_weight = item.name.size() >= 7 &&
                item.name.compare(item.name.size() - 7, 7, ".weight") == 0;
            const bool is_lookup_table = normalized_name.find("embed") != std::string::npos ||
                normalized_name.find("codebook") != std::string::npos;
            const bool can_quantize = source_is_float && name_is_weight && item.shape.size() == 2 &&
                !is_lookup_table && item.shape.back() % ggml_blck_size(requested_type) == 0;
            const bool use_requested = !ggml_is_quantized(requested_type)
                ? source_is_float
                : can_quantize;
            const bool use_f16_lookup = ggml_is_quantized(requested_type) && source_is_float && is_lookup_table;
            const ggml_type output_type = use_requested
                ? requested_type
                : (use_f16_lookup ? GGML_TYPE_F16 : source_type);
            const TensorStorageType output_storage = use_requested
                ? weight_type
                : (use_f16_lookup ? TensorStorageType::F16 : TensorStorageType::Native);

            std::string physical_name = item.name;
            if (physical_name.size() >= GGML_MAX_NAME || !physical_names.insert(physical_name).second) {
                physical_name = "_audiocpp." + std::to_string(i);
                if (!physical_names.insert(physical_name).second) {
                    throw std::runtime_error("failed to create a unique GGUF tensor alias");
                }
            }
            core::TensorShape shape = shape_from_dims(item.shape);
            const auto dims = core::to_ggml_dims(shape);
            ggml_tensor * tensor = ggml_new_tensor(
                tensor_context,
                output_type,
                static_cast<int>(shape.rank),
                dims.data());
            if (tensor == nullptr) {
                throw std::runtime_error("failed to create GGUF tensor metadata: " + item.name);
            }
            ggml_set_name(tensor, physical_name.c_str());
            gguf_add_tensor(gguf, tensor);
            outputs.push_back({item, std::move(physical_name), output_type, output_storage});
            logical_names.push_back(item.name);
        }

        std::vector<const char *> logical_name_ptrs;
        logical_name_ptrs.reserve(logical_names.size());
        for (const auto & name : logical_names) logical_name_ptrs.push_back(name.c_str());
        gguf_set_arr_str(
            gguf,
            "audiocpp.tensor_names",
            logical_name_ptrs.data(),
            logical_name_ptrs.size());
        std::vector<int32_t> exact_ranks;
        std::vector<int64_t> exact_shapes;
        exact_ranks.reserve(metadata.size());
        for (const auto & item : metadata) {
            exact_ranks.push_back(static_cast<int32_t>(item.shape.size()));
            exact_shapes.insert(exact_shapes.end(), item.shape.begin(), item.shape.end());
        }
        gguf_set_arr_data(gguf, "audiocpp.tensor_ranks", GGUF_TYPE_INT32, exact_ranks.data(), exact_ranks.size());
        gguf_set_arr_data(gguf, "audiocpp.tensor_shapes", GGUF_TYPE_INT64, exact_shapes.data(), exact_shapes.size());

        const auto parent = output_path.parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        auto temporary_path = output_path;
        temporary_path += ".tmp";
        std::filesystem::remove(temporary_path);
        if (!gguf_write_to_file(gguf, temporary_path.string().c_str(), true)) {
            throw std::runtime_error("failed to write GGUF metadata: " + temporary_path.string());
        }

        try {
            std::ofstream output(temporary_path, std::ios::binary | std::ios::app);
            if (!output) throw std::runtime_error("failed to append GGUF tensor data");
            size_t cursor = 0;
            std::vector<char> padding;
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto & item = outputs[i];
                const size_t offset = gguf_get_tensor_offset(gguf, static_cast<int64_t>(i));
                if (offset < cursor) throw std::runtime_error("GGUF tensor offsets are not monotonic");
                padding.assign(offset - cursor, '\0');
                if (!padding.empty()) output.write(padding.data(), static_cast<std::streamsize>(padding.size()));

                RawTensorData raw;
                if (item.storage_type == TensorStorageType::Native) {
                    raw = source->require_tensor_data(item.metadata.name);
                } else {
                    const auto tensor = source->require_tensor(
                        item.metadata.name,
                        item.storage_type,
                        item.metadata.shape);
                    raw.metadata = item.metadata;
                    raw.bytes = tensor.bytes;
                }
                const size_t expected_size = gguf_get_tensor_size(gguf, static_cast<int64_t>(i));
                if (raw.bytes.size() != expected_size) {
                    throw std::runtime_error("GGUF converted tensor byte size mismatch: " + item.metadata.name);
                }
                output.write(
                    reinterpret_cast<const char *>(raw.bytes.data()),
                    static_cast<std::streamsize>(raw.bytes.size()));
                if (!output) throw std::runtime_error("failed to write GGUF tensor: " + item.metadata.name);
                cursor = offset + raw.bytes.size();
            }
            output.close();
            if (!output) throw std::runtime_error("failed to finalize GGUF tensor data");
            if (engine::io::is_existing_file(output_path)) std::filesystem::remove(output_path);
            std::filesystem::rename(temporary_path, output_path);
        } catch (...) {
            std::filesystem::remove(temporary_path);
            throw;
        }
    } catch (...) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw;
    }
    gguf_free(gguf);
    ggml_free(tensor_context);
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
