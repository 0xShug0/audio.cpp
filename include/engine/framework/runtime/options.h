#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::runtime {

struct OptionMatch {
    std::string key;
    std::string value;
};

std::optional<OptionMatch> find_option_match(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<std::string> find_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

bool parse_bool_option(const std::string & value, std::string_view key);

std::optional<int> parse_int_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<float> parse_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<float> parse_finite_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<float> parse_positive_finite_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<uint32_t> parse_u32_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<uint64_t> parse_u64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<int64_t> parse_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

int64_t parse_positive_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    int64_t fallback);

size_t parse_size_mb_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    size_t fallback_bytes);

assets::TensorStorageType parse_tensor_storage_option(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    assets::TensorStorageType fallback,
    std::initializer_list<assets::TensorStorageType> allowed);

assets::TensorStorageType parse_tensor_storage_option(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    std::string_view fallback_key,
    assets::TensorStorageType fallback,
    std::initializer_list<assets::TensorStorageType> allowed);

uint32_t random_u32_seed();
uint64_t random_u64_seed();

}  // namespace engine::runtime
