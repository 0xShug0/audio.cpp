#pragma once

#include "sha256.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace engine::models::fish_drama3 {

template <typename T>
std::string_view cache_bytes(const T * data, size_t count) {
    return {reinterpret_cast<const char *>(data), count * sizeof(T)};
}

// Length delimit each field (including binary inputs) before hashing the key.
inline void cache_key_field(Sha256 & hash, std::string_view field) {
    const auto size = static_cast<uint64_t>(field.size());
    hash.update(cache_bytes(&size, 1));
    hash.update(field);
}

std::string drama3_cache_build_id();
void cache_file_identity(Sha256 & hash, const std::filesystem::path & path);

}  // namespace engine::models::fish_drama3
