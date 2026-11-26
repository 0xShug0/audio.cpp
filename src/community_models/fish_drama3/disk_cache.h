#pragma once

#include "sha256.h"

#include <filesystem>
#include <initializer_list>
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

struct CacheReadBuffer {
    void * data;
    size_t size;
};

std::string drama3_cache_build_id();
void cache_file_identity(Sha256 & hash, const std::filesystem::path & path);
// Null data may be used in an empty section. A miss leaves destination buffers
// unspecified; callers only apply them after successful checksum verification.
size_t cache_payload_size(const std::filesystem::path & path);
bool read_cache_file(const std::filesystem::path & path, std::initializer_list<CacheReadBuffer> data);
void write_cache_file(const std::filesystem::path & path, std::initializer_list<std::string_view> data, size_t keep);

}  // namespace engine::models::fish_drama3
