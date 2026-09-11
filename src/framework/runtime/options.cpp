#include "engine/framework/runtime/options.h"

#include "engine/framework/assets/tensor_source.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace engine::runtime {

namespace {

int parse_int_value(const std::string & value, std::string_view key) {
    size_t parsed = 0;
    const long long out = std::stoll(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string(key) + " must be an integer");
    }
    if (out < std::numeric_limits<int>::min() || out > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(key) + " is outside int range");
    }
    return static_cast<int>(out);
}

float parse_float_value(const std::string & value, std::string_view key) {
    size_t parsed = 0;
    const float out = std::stof(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string(key) + " must be a float");
    }
    return out;
}

uint32_t parse_u32_value(const std::string & value, std::string_view key) {
    if (!value.empty() && value.front() == '-') {
        throw std::runtime_error(std::string(key) + " must be an unsigned integer");
    }
    size_t parsed = 0;
    const unsigned long out = std::stoul(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string(key) + " must be an unsigned integer");
    }
    if (out > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(key) + " is outside uint32 range");
    }
    return static_cast<uint32_t>(out);
}

uint64_t parse_u64_value(const std::string & value, std::string_view key) {
    if (!value.empty() && value.front() == '-') {
        throw std::runtime_error(std::string(key) + " must be an unsigned integer");
    }
    size_t parsed = 0;
    const unsigned long long out = std::stoull(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string(key) + " must be an unsigned integer");
    }
    if (out > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(std::string(key) + " is outside uint64 range");
    }
    return static_cast<uint64_t>(out);
}

int64_t parse_i64_value(const std::string & value, std::string_view key) {
    size_t parsed = 0;
    const int64_t out = std::stoll(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string(key) + " must be an integer");
    }
    return out;
}

size_t checked_mib_to_bytes(int64_t mb, std::string_view key) {
    constexpr int64_t kMiB = 1024 * 1024;
    if (mb > static_cast<int64_t>(std::numeric_limits<size_t>::max() / kMiB)) {
        throw std::runtime_error(std::string(key) + " is too large");
    }
    return static_cast<size_t>(mb) * static_cast<size_t>(kMiB);
}

std::string tensor_storage_type_name(assets::TensorStorageType storage_type) {
    switch (storage_type) {
    case assets::TensorStorageType::Native:
        return "native";
    case assets::TensorStorageType::F32:
        return "f32";
    case assets::TensorStorageType::F16:
        return "f16";
    case assets::TensorStorageType::BF16:
        return "bf16";
    case assets::TensorStorageType::I8:
        return "i8";
    case assets::TensorStorageType::Q4_0:
        return "q4_0";
    case assets::TensorStorageType::Q4_1:
        return "q4_1";
    case assets::TensorStorageType::Q5_0:
        return "q5_0";
    case assets::TensorStorageType::Q5_1:
        return "q5_1";
    case assets::TensorStorageType::Q2_K:
        return "q2_k";
    case assets::TensorStorageType::Q3_K:
        return "q3_k";
    case assets::TensorStorageType::Q4_K:
        return "q4_k";
    case assets::TensorStorageType::Q5_K:
        return "q5_k";
    case assets::TensorStorageType::Q6_K:
        return "q6_k";
    case assets::TensorStorageType::Q8_0:
        return "q8_0";
    }
    throw std::runtime_error("unknown tensor storage type");
}

std::string join_tensor_storage_types(std::initializer_list<assets::TensorStorageType> values) {
    std::string out;
    size_t index = 0;
    for (const auto value : values) {
        if (index != 0) {
            out += index + 1 == values.size() ? ", and " : ", ";
        }
        out += tensor_storage_type_name(value);
        ++index;
    }
    return out;
}

bool contains_tensor_storage_type(
    std::initializer_list<assets::TensorStorageType> values,
    assets::TensorStorageType needle) {
    for (const auto value : values) {
        if (value == needle) {
            return true;
        }
    }
    return false;
}

}  // namespace

std::optional<OptionMatch> find_option_match(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    std::optional<OptionMatch> match;
    for (const std::string_view key : keys) {
        const auto it = options.find(std::string(key));
        if (it != options.end() && !it->second.empty()) {
            if (match.has_value()) {
                if (match->value != it->second) {
                    throw std::runtime_error(
                        "conflicting option values for " + match->key + " and " + std::string(key));
                }
                continue;
            }
            match = OptionMatch{std::string(key), it->second};
        }
    }
    return match;
}

std::optional<std::string> find_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    const auto match = find_option_match(options, keys);
    return match.has_value() ? std::optional<std::string>(match->value) : std::nullopt;
}

bool parse_bool_option(const std::string & value, std::string_view key) {
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error(std::string(key) + " must be true/false, yes/no, on/off, or 1/0");
}

std::optional<int> parse_int_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        return parse_int_value(match->value, match->key);
    }
    return std::nullopt;
}

std::optional<float> parse_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        return parse_float_value(match->value, match->key);
    }
    return std::nullopt;
}

std::optional<float> parse_finite_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        const float value = parse_float_value(match->value, match->key);
        if (!std::isfinite(value)) {
            throw std::runtime_error(match->key + " must be a finite float");
        }
        return value;
    }
    return std::nullopt;
}

std::optional<float> parse_positive_finite_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        const float value = parse_float_value(match->value, match->key);
        if (!std::isfinite(value)) {
            throw std::runtime_error(match->key + " must be a finite float");
        }
        if (value <= 0.0F) {
            throw std::runtime_error(match->key + " must be positive");
        }
        return value;
    }
    return std::nullopt;
}

std::optional<uint32_t> parse_u32_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        return parse_u32_value(match->value, match->key);
    }
    return std::nullopt;
}

std::optional<uint64_t> parse_u64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        return parse_u64_value(match->value, match->key);
    }
    return std::nullopt;
}

std::optional<int64_t> parse_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto match = find_option_match(options, keys)) {
        return parse_i64_value(match->value, match->key);
    }
    return std::nullopt;
}

int64_t parse_positive_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    int64_t fallback) {
    if (const auto match = find_option_match(options, keys)) {
        const int64_t value = parse_i64_value(match->value, match->key);
        if (value <= 0) {
            throw std::runtime_error(match->key + " must be positive");
        }
        return value;
    }
    return fallback;
}

size_t parse_size_mb_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    size_t fallback_bytes) {
    if (const auto match = find_option_match(options, keys)) {
        const int64_t mb = parse_i64_value(match->value, match->key);
        if (mb <= 0) {
            throw std::runtime_error(match->key + " must be positive");
        }
        return checked_mib_to_bytes(mb, match->key);
    }
    return fallback_bytes;
}

assets::TensorStorageType parse_tensor_storage_option(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    assets::TensorStorageType fallback,
    std::initializer_list<assets::TensorStorageType> allowed) {
    if (allowed.size() == 0) {
        throw std::runtime_error("tensor storage option allowlist must not be empty");
    }
    const auto match = find_option_match(options, {key});
    if (!match.has_value()) {
        return fallback;
    }
    const auto storage_type = assets::parse_tensor_storage_type(match->value);
    if (contains_tensor_storage_type(allowed, storage_type)) {
        return storage_type;
    }
    throw std::runtime_error(match->key + " supports only " + join_tensor_storage_types(allowed));
}

assets::TensorStorageType parse_tensor_storage_option(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    std::string_view fallback_key,
    assets::TensorStorageType fallback,
    std::initializer_list<assets::TensorStorageType> allowed) {
    if (find_option_match(options, {key}).has_value()) {
        return parse_tensor_storage_option(options, key, fallback, allowed);
    }
    return parse_tensor_storage_option(options, fallback_key, fallback, allowed);
}

uint64_t random_u64_seed() {
    std::random_device random_device;
    const uint64_t hi = static_cast<uint64_t>(random_device()) << 32U;
    const uint64_t lo = static_cast<uint64_t>(random_device());
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return hi ^ lo ^ static_cast<uint64_t>(now);
}

uint32_t random_u32_seed() {
    return static_cast<uint32_t>(random_u64_seed());
}

}  // namespace engine::runtime
