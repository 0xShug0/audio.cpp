#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace engine::io {

struct DiskCacheConfig {
    // Dedicated cache directory. All instances sharing it must use the same limits.
    std::filesystem::path directory;
    size_t max_entries = 128;
    uint64_t max_bytes = 1024ULL * 1024 * 1024;
};

// Local filesystem byte cache with validated reads and atomic replacement.
// Callers serialize payloads and include their compatibility identity in the key.
// A missing key returns nullopt; corrupt entries and I/O failures throw. Callers
// may explicitly warn and recompute for an optional cache; errors are never hidden.
// Operations are serialized across threads/processes using the directory lock.
// File contents are synced before atomic replacement. The parent directory is
// not synced, so the latest publication may be lost after power failure.
// Checksums detect accidental corruption, not malicious modification. Use a
// trusted directory; this is not a security boundary or a shared network cache.
class DiskCache {
public:
    explicit DiskCache(DiskCacheConfig config);

    std::optional<std::vector<std::byte>> get(std::string_view key) const;
    void put(std::string_view key, const std::vector<std::byte> & payload) const;

private:
    DiskCacheConfig config_;
};

}  // namespace engine::io
