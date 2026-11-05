#include "engine/framework/assets/gguf_metadata.h"

#include <gguf.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::assets {

GgufMetadata::GgufMetadata(const std::filesystem::path & path)
    : path_(path), context_(gguf_init_from_file(path.string().c_str(), {true, nullptr})) {
    if (!context_) {
        throw std::runtime_error("cannot read GGUF metadata: " + path.string());
    }
}

void GgufMetadata::fail(const char * key, const char * message) const {
    throw std::runtime_error("GGUF field '" + std::string(key) + "' " + message + ": " + path_.string());
}

bool GgufMetadata::has(const char * key) const {
    return gguf_find_key(context_.get(), key) >= 0;
}

int64_t GgufMetadata::tensor_count() const {
    return gguf_get_n_tensors(context_.get());
}

int64_t GgufMetadata::require_key(const char * key) const {
    const auto id = gguf_find_key(context_.get(), key);
    if (id < 0) { fail(key, "is missing"); }
    return id;
}

std::string GgufMetadata::require_string(const char * key) const {
    const auto id = require_key(key);
    if (gguf_get_kv_type(context_.get(), id) != GGUF_TYPE_STRING) { fail(key, "is not a string"); }
    return gguf_get_val_str(context_.get(), id);
}

std::optional<std::string> GgufMetadata::find_string(const char * key) const {
    return has(key) ? std::optional<std::string>(require_string(key)) : std::nullopt;
}

bool GgufMetadata::require_bool(const char * key) const {
    const auto id = require_key(key);
    if (gguf_get_kv_type(context_.get(), id) != GGUF_TYPE_BOOL) { fail(key, "is not a boolean"); }
    return gguf_get_val_bool(context_.get(), id);
}

std::optional<bool> GgufMetadata::find_bool(const char * key) const {
    return has(key) ? std::optional<bool>(require_bool(key)) : std::nullopt;
}

int64_t GgufMetadata::require_int(const char * key) const {
    const auto id = require_key(key);
    switch (gguf_get_kv_type(context_.get(), id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(context_.get(), id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(context_.get(), id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(context_.get(), id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(context_.get(), id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(context_.get(), id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(context_.get(), id);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(context_.get(), id);
        case GGUF_TYPE_UINT64: {
            const auto value = gguf_get_val_u64(context_.get(), id);
            if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { fail(key, "exceeds int64 range"); }
            return static_cast<int64_t>(value);
        }
        default: fail(key, "is not an integer");
    }
}

int64_t GgufMetadata::require_int(const char * key, int64_t min, int64_t max) const {
    const auto value = require_int(key);
    if (value < min || value > max) { fail(key, "is outside the requested range"); }
    return value;
}

float GgufMetadata::require_f32(const char * key) const {
    const auto id = require_key(key);
    if (gguf_get_kv_type(context_.get(), id) != GGUF_TYPE_FLOAT32) { fail(key, "is not float32"); }
    const auto value = gguf_get_val_f32(context_.get(), id);
    if (!std::isfinite(value)) { fail(key, "is not finite"); }
    return value;
}

double GgufMetadata::require_number(const char * key) const {
    const auto id = require_key(key);
    double value;
    switch (gguf_get_kv_type(context_.get(), id)) {
        case GGUF_TYPE_FLOAT32: value = gguf_get_val_f32(context_.get(), id); break;
        case GGUF_TYPE_FLOAT64: value = gguf_get_val_f64(context_.get(), id); break;
        case GGUF_TYPE_UINT64: value = static_cast<double>(gguf_get_val_u64(context_.get(), id)); break;
        default: value = static_cast<double>(require_int(key)); break;
    }
    if (!std::isfinite(value)) { fail(key, "is not finite"); }
    return value;
}

std::optional<double> GgufMetadata::find_number(const char * key) const {
    return has(key) ? std::optional<double>(require_number(key)) : std::nullopt;
}

std::vector<int64_t> GgufMetadata::require_int_array(const char * key, bool allow_bool) const {
    const auto id = require_key(key);
    if (gguf_get_kv_type(context_.get(), id) != GGUF_TYPE_ARRAY) { fail(key, "is not an array"); }
    const auto type = gguf_get_arr_type(context_.get(), id);
    if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_UINT32 && !(allow_bool && type == GGUF_TYPE_BOOL)) {
        fail(key, "is not an int32/uint32 array or an explicitly allowed boolean array");
    }
    const auto * data = gguf_get_arr_data(context_.get(), id);
    std::vector<int64_t> values;
    values.reserve(gguf_get_arr_n(context_.get(), id));
    for (size_t i = 0; i < gguf_get_arr_n(context_.get(), id); ++i) {
        if (type == GGUF_TYPE_INT32) { values.push_back(static_cast<const int32_t *>(data)[i]); }
        else if (type == GGUF_TYPE_UINT32) { values.push_back(static_cast<const uint32_t *>(data)[i]); }
        else { values.push_back(static_cast<const bool *>(data)[i]); }
    }
    return values;
}

std::vector<std::string> GgufMetadata::require_string_array(const char * key) const {
    const auto id = require_key(key);
    if (gguf_get_kv_type(context_.get(), id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(context_.get(), id) != GGUF_TYPE_STRING) {
        fail(key, "is not a string array");
    }
    std::vector<std::string> values;
    values.reserve(gguf_get_arr_n(context_.get(), id));
    for (size_t i = 0; i < gguf_get_arr_n(context_.get(), id); ++i) {
        values.emplace_back(gguf_get_arr_str(context_.get(), id, i));
    }
    return values;
}

}  // namespace engine::assets
