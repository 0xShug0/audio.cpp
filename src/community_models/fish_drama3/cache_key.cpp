// Runtime targets define _POSIX_C_SOURCE for libyaml. On macOS that hides
// Dl_info and stat::st_ctimespec unless Darwin extensions stay enabled.
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include "cache_key.h"

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
#endif

namespace engine::models::fish_drama3 {
namespace {
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
    // libc++ stores file times as __int128, which std::to_string does not accept.
    cache_key_field(hash, std::to_string(static_cast<long long>(
        std::filesystem::last_write_time(resolved).time_since_epoch().count())));
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

}  // namespace engine::models::fish_drama3
