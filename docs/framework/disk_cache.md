# Disk cache

`engine::io::DiskCache` stores byte payloads in a local directory. It provides
validated reads, atomic replacement and eviction by entry count and total size.

```cpp
#include "engine/framework/io/disk_cache.h"

engine::io::DiskCacheConfig config;
config.directory = cache_directory;
config.max_entries = 4;
config.max_bytes = 1024ULL * 1024 * 1024;
engine::io::DiskCache cache(config);

const std::string key = "features/v1/input-123";
cache.put(key, payload);              // payload: std::vector<std::byte>
if (auto cached = cache.get(key)) {
    // Use *cached, a std::vector<std::byte>.
}
```

## Keys and payloads

Callers serialize their data and construct keys that identify compatible results.
Include the payload schema version, input identity and settings that affect the
result. Model-derived data may also require weight, backend and precision identities.
Use a dedicated directory for each cache, with the same limits in every instance
that accesses it.

Keys contain 1–4096 bytes. Empty payloads are valid. Each entry stores its full key,
format version, payload length and checksum. Reads validate the complete entry
before returning its payload.

## Results and errors

- `get(key)` returns an owned byte vector on a hit, or `std::nullopt` on a miss.
- `put(key, payload)` creates or replaces an entry.
- Invalid configuration, invalid keys and entries exceeding the byte limit throw
  `std::invalid_argument`.
- Corruption and filesystem failures throw exceptions. Callers decide whether to
  propagate the error or report it and recompute an optional cached result.

A filename-hash collision is treated as a miss when the stored full key differs.
No data from the other key is returned.

## Writes and concurrency

A directory lock serializes operations across threads and processes. Writes go
through a temporary file. After its stream closes successfully, file contents
are synced with `fsync` on POSIX or `FlushFileBuffers` on Windows before atomic
replacement. Write or sync failures preserve the previous entry and propagate
to the caller.
After an interrupted write, the next operation removes the abandoned temporary
file while holding the lock. Other temporary files are untouched.

Keep the persistent lock file in place while the cache is in use. Lock acquisition
waits for the current operation; reads and writes hold the lock for their full duration.

## Capacity and eviction

Successful writes evict least-recently-used entries until both configured limits
are met. Reads refresh access order using filesystem modification time. The byte
limit includes entry headers, keys and payloads. Replacement temporarily requires
space for one additional entry.

If eviction fails after replacement, `put` throws even though the new entry may
already exist. The next successful write enforces the limits again.

## Limits

Use trusted local directories. The FNV-1a checksum detects accidental corruption;
it does not authenticate data. The parent directory is not synced after replacement,
so the latest publication may be lost after power failure. Payloads are read into
host-memory vectors.
