#include "engine/models/voxcpm2/gguf_metadata.h"

#include "engine/framework/assets/tensor_source.h"

#include <gguf.h>

#include <stdexcept>

namespace engine::models::voxcpm2 {

GgufMetadataReader::GgufMetadataReader(const engine::assets::TensorSource & source) {
    // Metadata-only open: no_alloc=true with no ggml context parses the GGUF
    // header and KV section without ever touching tensor data.
    gguf_ = gguf_init_from_file(
        source.source_path().string().c_str(),
        gguf_init_params{true, nullptr});
}

GgufMetadataReader::~GgufMetadataReader() {
    if (gguf_ != nullptr) {
        gguf_free(gguf_);
    }
}

std::optional<std::string> GgufMetadataReader::optional_string(std::string_view key) const {
    if (gguf_ == nullptr) {
        return std::nullopt;
    }
    const int64_t idx = gguf_find_key(gguf_, std::string(key).c_str());
    if (idx < 0 || gguf_get_kv_type(gguf_, idx) != GGUF_TYPE_STRING) {
        return std::nullopt;
    }
    const char * data = gguf_get_val_str(gguf_, idx);
    if (data == nullptr) {
        return std::nullopt;
    }
    return std::string(data);
}

std::optional<uint32_t> GgufMetadataReader::optional_u32(std::string_view key) const {
    if (gguf_ == nullptr) {
        return std::nullopt;
    }
    // No KV type check: VoxCPM stores boolean flags (use_mup, no_rope,
    // mean_mode) as scalar values that gguf_get_val_u32 reads regardless of
    // their declared scalar type.
    const int64_t idx = gguf_find_key(gguf_, std::string(key).c_str());
    if (idx < 0) {
        return std::nullopt;
    }
    return gguf_get_val_u32(gguf_, idx);
}

std::optional<float> GgufMetadataReader::optional_f32(std::string_view key) const {
    if (gguf_ == nullptr) {
        return std::nullopt;
    }
    const int64_t idx = gguf_find_key(gguf_, std::string(key).c_str());
    if (idx < 0 || gguf_get_kv_type(gguf_, idx) == GGUF_TYPE_ARRAY) {
        return std::nullopt;
    }
    return gguf_get_val_f32(gguf_, idx);
}

std::optional<std::vector<std::string>> GgufMetadataReader::optional_string_array(std::string_view key) const {
    if (gguf_ == nullptr) {
        return std::nullopt;
    }
    const int64_t idx = gguf_find_key(gguf_, std::string(key).c_str());
    if (idx < 0 || gguf_get_arr_type(gguf_, idx) != GGUF_TYPE_STRING) {
        return std::nullopt;
    }
    const size_t n = gguf_get_arr_n(gguf_, idx);
    std::vector<std::string> values;
    values.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const char * v = gguf_get_arr_str(gguf_, idx, i);
        values.emplace_back(v != nullptr ? v : "");
    }
    return values;
}

std::optional<std::vector<int32_t>> GgufMetadataReader::optional_i32_array(std::string_view key) const {
    if (gguf_ == nullptr) {
        return std::nullopt;
    }
    const int64_t idx = gguf_find_key(gguf_, std::string(key).c_str());
    if (idx < 0) {
        return std::nullopt;
    }
    const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(gguf_, idx));
    const size_t n = gguf_get_arr_n(gguf_, idx);
    if (data == nullptr) {
        return n == 0 ? std::optional<std::vector<int32_t>>(std::vector<int32_t>{}) : std::nullopt;
    }
    return std::vector<int32_t>(data, data + n);
}

std::optional<std::vector<float>> GgufMetadataReader::optional_f32_array(std::string_view key) const {
    if (gguf_ == nullptr) {
        return std::nullopt;
    }
    const int64_t idx = gguf_find_key(gguf_, std::string(key).c_str());
    if (idx < 0 || gguf_get_arr_type(gguf_, idx) != GGUF_TYPE_FLOAT32) {
        return std::nullopt;
    }
    const auto * data = static_cast<const float *>(gguf_get_arr_data(gguf_, idx));
    const size_t n = gguf_get_arr_n(gguf_, idx);
    if (data == nullptr) {
        return n == 0 ? std::optional<std::vector<float>>(std::vector<float>{}) : std::nullopt;
    }
    return std::vector<float>(data, data + n);
}

std::string GgufMetadataReader::require_string(std::string_view key) const {
    auto opt = optional_string(key);
    if (opt) {
        return *opt;
    }
    throw std::runtime_error("GGUF metadata key not found: " + std::string(key));
}

uint32_t GgufMetadataReader::require_u32(std::string_view key) const {
    auto opt = optional_u32(key);
    if (opt) {
        return *opt;
    }
    throw std::runtime_error("GGUF metadata key not found: " + std::string(key));
}

std::vector<std::string> GgufMetadataReader::require_string_array(std::string_view key) const {
    auto opt = optional_string_array(key);
    if (opt) {
        return *opt;
    }
    throw std::runtime_error("GGUF metadata key not found: " + std::string(key));
}

std::vector<int32_t> GgufMetadataReader::require_i32_array(std::string_view key) const {
    auto opt = optional_i32_array(key);
    if (opt) {
        return *opt;
    }
    throw std::runtime_error("GGUF metadata key not found: " + std::string(key));
}

}  // namespace engine::models::voxcpm2