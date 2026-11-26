#include "disk_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <fstream>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace engine::models::fish_drama3 {
namespace {
constexpr std::string_view kMagic = "DRAMA3-2";
constexpr size_t kDigestBytes = 64;

std::filesystem::path build_module_path() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&build_module_path), &module)) return {};
    std::wstring path(32768, L'\0');
    const auto size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (size == 0 || size >= path.size()) return {};
    path.resize(size);
    return path;
#else
    Dl_info module{};
    if (!dladdr(reinterpret_cast<const void *>(&build_module_path), &module) || !module.dli_fname) return {};
    return module.dli_fname;
#endif
}

bool replace_cache_file(const std::filesystem::path & source, const std::filesystem::path & target) {
#ifdef _WIN32
    return MoveFileExW(source.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code ec;
    std::filesystem::rename(source, target, ec);
    return !ec;
#endif
}

void prune(const std::filesystem::path & path, size_t keep) {
    using Entry = std::pair<std::filesystem::file_time_type, std::filesystem::path>;
    std::vector<Entry> entries;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(path.parent_path())) {
        if (entry.path().extension() != path.extension()) continue;
        const auto name = entry.path().stem().u8string();
        if (name.size() != 64 ||
            name.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
        if (!std::filesystem::is_regular_file(entry.symlink_status(ec))) continue;
        const auto time = entry.last_write_time(ec);
        if (!ec) entries.emplace_back(time, entry.path());
    }
    std::sort(entries.rbegin(), entries.rend());
    for (size_t i = keep; i < entries.size(); ++i) std::filesystem::remove(entries[i].second, ec);
}
}  // namespace

std::string drama3_cache_build_id() {
    // Hash the linked module, including dirty and incremental builds. Installed
    // byte-identical copies can share entries even at different locations.
    static const auto id = []() -> std::string {
        std::ifstream input(build_module_path(), std::ios::binary);
        if (!input) return {};
        Sha256 hash;
        std::vector<char> block(256 * 1024);
        while (input.read(block.data(), block.size()) || input.gcount() > 0) {
            hash.update({block.data(), static_cast<size_t>(input.gcount())});
        }
        return input.eof() && !input.bad() ? hash.hex() : std::string{};
    }();
    return id;
}

void cache_file_identity(Sha256 & hash, const std::filesystem::path & path) {
    const auto resolved = std::filesystem::canonical(path);
    cache_key_field(hash, resolved.generic_u8string());
    cache_key_field(hash, std::to_string(std::filesystem::file_size(resolved)));
    cache_key_field(hash, std::to_string(std::filesystem::last_write_time(resolved).time_since_epoch().count()));
    // Include the file identity and change time to detect replacement and edits
    // that retain the modification time without rereading the checkpoint.
#ifdef _WIN32
    const auto file = CreateFileW(resolved.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::filesystem::filesystem_error(
        "open cache source", resolved, std::error_code(GetLastError(), std::system_category()));
    BY_HANDLE_FILE_INFORMATION info{};
    FILE_BASIC_INFO times{};
    const bool ok = GetFileInformationByHandle(file, &info) &&
        GetFileInformationByHandleEx(file, FileBasicInfo, &times, sizeof(times));
    const auto error = ok ? ERROR_SUCCESS : GetLastError();
    CloseHandle(file);
    if (!ok) throw std::filesystem::filesystem_error(
        "stat cache source", resolved, std::error_code(error, std::system_category()));
    cache_key_field(hash, std::to_string(info.dwVolumeSerialNumber));
    cache_key_field(hash, std::to_string(info.nFileIndexHigh));
    cache_key_field(hash, std::to_string(info.nFileIndexLow));
    cache_key_field(hash, std::to_string(times.ChangeTime.QuadPart));
#else
    struct stat info{};
    if (stat(resolved.c_str(), &info) != 0) throw std::filesystem::filesystem_error(
        "stat cache source", resolved, std::error_code(errno, std::generic_category()));
    cache_key_field(hash, std::to_string(info.st_dev));
    cache_key_field(hash, std::to_string(info.st_ino));
#if defined(__APPLE__)
    const auto changed = info.st_ctimespec;
#else
    const auto changed = info.st_ctim;
#endif
    cache_key_field(hash, std::to_string(changed.tv_sec));
    cache_key_field(hash, std::to_string(changed.tv_nsec));
#endif
}

size_t cache_payload_size(const std::filesystem::path & path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path, ec))) return 0;
    const auto size = std::filesystem::file_size(path, ec);
    constexpr size_t overhead = kMagic.size() + kDigestBytes;
    return !ec && size > overhead && size <= SIZE_MAX ? static_cast<size_t>(size) - overhead : 0;
}

bool read_cache_file(const std::filesystem::path & path, std::initializer_list<CacheReadBuffer> data) {
    size_t payload = 0;
    for (const auto & section : data) payload += section.size;
    if (payload == 0 || cache_payload_size(path) != payload) return false;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(payload + kMagic.size() + kDigestBytes)) return false;
    input.seekg(0);
    std::string magic(kMagic.size(), '\0');
    input.read(magic.data(), magic.size());
    if (magic != kMagic) return false;
    Sha256 digest;
    cache_key_field(digest, path.stem().string());
    for (const auto & section : data) {
        auto * bytes = static_cast<char *>(section.data);
        if (section.size > 0 && !input.read(bytes, section.size)) return false;
        cache_key_field(digest, {bytes, section.size});
    }
    std::string checksum(kDigestBytes, '\0');
    if (!input.read(checksum.data(), checksum.size()) || digest.hex() != checksum) return false;
    input.close();
    std::error_code ec;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
    return true;
}

void write_cache_file(const std::filesystem::path & path, std::initializer_list<std::string_view> data, size_t keep) {
    static std::atomic<uint64_t> sequence{0};
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = getpid();
#endif
    auto temp = path;
    temp += "." + std::to_string(pid) + "." + std::to_string(nonce) + "." + std::to_string(sequence++) + ".tmp";
    try {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(temp, std::ios::binary | std::ios::trunc);
        output.write(kMagic.data(), kMagic.size());
        Sha256 digest;
        cache_key_field(digest, path.stem().string());
        for (const auto section : data) {
            if (!section.empty()) output.write(section.data(), section.size());
            cache_key_field(digest, section);
        }
        const auto checksum = digest.hex();
        output.write(checksum.data(), checksum.size());
        output.close();
        if (output && replace_cache_file(temp, path)) prune(path, keep);
    } catch (const std::filesystem::filesystem_error &) {
        // An unavailable optional cache must not stop synthesis.
    }
    std::error_code ec;
    std::filesystem::remove(temp, ec);
}

}  // namespace engine::models::fish_drama3
