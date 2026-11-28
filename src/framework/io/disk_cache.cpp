// flock is a Darwin extension hidden by strict POSIX feature flags. Enable it
// only in this implementation, before system headers; never change build flags.
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "engine/framework/io/disk_cache.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace engine::io {
namespace {
constexpr std::array<char, 8> magic{'A', 'C', 'D', 'C', '0', '0', '0', '1'};
constexpr uint64_t header_bytes = 32;
constexpr size_t max_key_bytes = 4096;
constexpr const char * temporary_name = ".audiocpp-disk-cache.tmp";

// FNV-1a is an accidental-corruption checksum. Store and compare the complete key
// as well: a filename hash collision must never return another key's payload.
uint64_t checksum(uint64_t state, const void * data, size_t size) {
    const auto * bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) state = (state ^ bytes[i]) * 1099511628211ULL;
    return state;
}
constexpr uint64_t checksum_seed = 14695981039346656037ULL;

std::filesystem::path entry_path(const std::filesystem::path & directory, std::string_view key) {
    const auto hash = checksum(checksum_seed, key.data(), key.size());
    std::string name(16, '0');
    for (unsigned i = 0; i < 16; ++i) name[15 - i] = "0123456789abcdef"[(hash >> (4 * i)) & 15];
    return directory / (name + ".cache");
}

void write_number(std::ostream & output, uint64_t value) {
    std::array<char, 8> bytes{};
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 255);
    output.write(bytes.data(), bytes.size());
}

uint64_t read_number(std::istream & input) {
    std::array<unsigned char, 8> bytes{};
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= uint64_t(bytes[i]) << (8 * i);
    return value;
}

class DirectoryLock {
public:
    explicit DirectoryLock(const std::filesystem::path & directory) {
        std::filesystem::create_directories(directory);
        const auto path = directory / ".audiocpp-disk-cache.lock";
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) fail("open disk cache lock");
        OVERLAPPED overlapped{};
        if (!LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &overlapped)) {
            const auto error = GetLastError();
            CloseHandle(handle_);
            throw std::system_error(static_cast<int>(error), std::system_category(), "lock disk cache");
        }
#else
        fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open disk cache lock");
        int status;
        do { status = flock(fd_, LOCK_EX); } while (status < 0 && errno == EINTR);
        if (status < 0) {
            const int error = errno;
            close(fd_);
            throw std::system_error(error, std::generic_category(), "lock disk cache");
        }
#endif
    }
    ~DirectoryLock() {
#ifdef _WIN32
        CloseHandle(handle_);
#else
        close(fd_);
#endif
    }
    DirectoryLock(const DirectoryLock &) = delete;
    DirectoryLock & operator=(const DirectoryLock &) = delete;
private:
#ifdef _WIN32
    [[noreturn]] static void fail(const char * message) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), message);
    }
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

void validate_key(std::string_view key) {
    if (key.empty() || key.size() > max_key_bytes) throw std::invalid_argument("disk cache key must contain 1..4096 bytes");
}

void sync_file(const std::filesystem::path & path) {
    // The checked stream close flushed userspace buffers. Reopen under the
    // directory lock to sync the file before publishing it (C++17 streams have
    // no portable native-handle accessor).
#ifdef _WIN32
    const auto handle = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "open disk cache file for sync");
    const bool synced = FlushFileBuffers(handle) != 0;
    const auto error = synced ? ERROR_SUCCESS : GetLastError();
    const bool closed = CloseHandle(handle) != 0;
    if (!synced) throw std::system_error(static_cast<int>(error), std::system_category(), "sync disk cache file");
    if (!closed) throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "close disk cache sync handle");
#else
    const int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open disk cache file for sync");
    int status;
    do { status = fsync(fd); } while (status < 0 && errno == EINTR);
    const int error = status < 0 ? errno : 0;
    const int closed = close(fd);
    if (error) throw std::system_error(error, std::generic_category(), "sync disk cache file");
    if (closed < 0) throw std::system_error(errno, std::generic_category(), "close disk cache sync handle");
#endif
}

void prune(const DiskCacheConfig & config, const std::filesystem::path & written) {
    struct Entry { std::filesystem::path path; std::filesystem::file_time_type time; uint64_t bytes; };
    std::vector<Entry> entries;
    uint64_t bytes = 0;
    for (const auto & item : std::filesystem::directory_iterator(config.directory)) {
        const auto name = item.path().filename().string();
        if (name.size() != 22 || name.substr(16) != ".cache" ||
            name.substr(0, 16).find_first_not_of("0123456789abcdef") != std::string::npos) continue;
        if (!std::filesystem::is_regular_file(item.symlink_status()))
            throw std::runtime_error("disk cache entry is not a regular file: " + item.path().string());
        const auto size = item.file_size();
        if (size > (std::numeric_limits<uint64_t>::max)() - bytes)
            throw std::runtime_error("disk cache size overflow");
        bytes += size;
        entries.push_back({item.path(), item.last_write_time(), size});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry & a, const Entry & b) {
        return a.time == b.time ? a.path < b.path : a.time < b.time;
    });
    size_t count = entries.size();
    for (const auto & entry : entries) {
        if (count <= config.max_entries && bytes <= config.max_bytes) break;
        if (entry.path == written) continue;
        std::filesystem::remove(entry.path);
        bytes -= entry.bytes;
        --count;
    }
}
}  // namespace

DiskCache::DiskCache(DiskCacheConfig config) : config_(std::move(config)) {
    if (config_.directory.empty() || config_.max_entries == 0 || config_.max_bytes <= header_bytes ||
        config_.max_bytes > static_cast<uint64_t>((std::numeric_limits<std::streamsize>::max)()) ||
        config_.max_bytes > (std::numeric_limits<size_t>::max)())
        throw std::invalid_argument("invalid disk cache directory or limits");
}

std::optional<std::vector<std::byte>> DiskCache::get(std::string_view key) const {
    validate_key(key);
    DirectoryLock lock(config_.directory);
    // The lock covers all reads/writes. With it held, this can only be an
    // abandoned write; never sweep arbitrary temporary files by age or PID.
    std::filesystem::remove(config_.directory / temporary_name);
    const auto path = entry_path(config_.directory, key);
    const auto status = std::filesystem::symlink_status(path);
    if (!std::filesystem::exists(status)) return std::nullopt;
    if (!std::filesystem::is_regular_file(status))
        throw std::runtime_error("disk cache entry is not a regular file: " + path.string());
    const auto size = std::filesystem::file_size(path);
    if (size < header_bytes || size > config_.max_bytes)
        throw std::runtime_error("invalid disk cache entry size: " + path.string());
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read disk cache entry: " + path.string());
    std::array<char, 8> found_magic{};
    input.read(found_magic.data(), found_magic.size());
    const auto key_size = read_number(input);
    const auto payload_size = read_number(input);
    const auto expected_checksum = read_number(input);
    if (!input || found_magic != magic || key_size == 0 || key_size > max_key_bytes || key_size > size - header_bytes ||
        payload_size != size - header_bytes - key_size)
        throw std::runtime_error("invalid disk cache header: " + path.string());
    std::string stored_key(static_cast<size_t>(key_size), '\0');
    input.read(stored_key.data(), static_cast<std::streamsize>(stored_key.size()));
    std::vector<std::byte> payload(static_cast<size_t>(payload_size));
    if (!payload.empty()) input.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
    const auto actual_checksum = checksum(checksum(checksum_seed, stored_key.data(), stored_key.size()), payload.data(), payload.size());
    if (!input || actual_checksum != expected_checksum)
        throw std::runtime_error("disk cache checksum mismatch: " + path.string());
    if (stored_key != key) return std::nullopt;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now());
    return payload;
}

void DiskCache::put(std::string_view key, const std::vector<std::byte> & payload) const {
    validate_key(key);
    if (key.size() > config_.max_bytes - header_bytes ||
        payload.size() > config_.max_bytes - header_bytes - key.size())
        throw std::invalid_argument("disk cache entry exceeds byte limit");
    DirectoryLock lock(config_.directory);
    const auto temporary = config_.directory / temporary_name;
    std::filesystem::remove(temporary);
    const auto path = entry_path(config_.directory, key);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write disk cache temporary file: " + temporary.string());
        output.write(magic.data(), magic.size());
        write_number(output, key.size());
        write_number(output, payload.size());
        write_number(output, checksum(checksum(checksum_seed, key.data(), key.size()), payload.data(), payload.size()));
        output.write(key.data(), static_cast<std::streamsize>(key.size()));
        if (!payload.empty()) output.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
        output.close();
        if (!output) throw std::runtime_error("cannot finish disk cache temporary file: " + temporary.string());
        sync_file(temporary);
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "replace disk cache entry");
#else
        std::filesystem::rename(temporary, path);
#endif
        prune(config_, path);
    } catch (...) {
        // Preserve the actual error. A cleanup failure leaves a temporary file
        // that the next lock holder will reclaim, rather than hiding the error.
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw;
    }
}

}  // namespace engine::io
