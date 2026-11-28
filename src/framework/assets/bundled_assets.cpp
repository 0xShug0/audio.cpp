// The build defines _POSIX_C_SOURCE (for libyaml), which on Apple platforms hides dladdr()
// and Dl_info unless _DARWIN_C_SOURCE is defined, as ggml does. It must come before any
// system header.
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include "engine/framework/assets/bundled_assets.h"

#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace engine::assets {
namespace {

constexpr int kParentLevels = 3;

// Its address identifies the module this file is linked into.
const char kModuleAnchor = 0;

bool exists_quietly(const std::filesystem::path & path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

// Directory of the module that contains audio.cpp: the C API library, or the executable
// when audio.cpp is linked statically. Empty when it cannot be determined.
std::filesystem::path module_directory() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&kModuleAnchor), &module)) {
        return {};
    }
    std::vector<wchar_t> buffer(MAX_PATH);
    while (true) {
        const DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return {};
        }
        if (length < buffer.size()) {
            return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    Dl_info info{};
    if (dladdr(&kModuleAnchor, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
        return {};
    }
    std::error_code ec;
    std::filesystem::path module = std::filesystem::weakly_canonical(info.dli_fname, ec);
    if (ec || !std::filesystem::is_regular_file(module, ec)) {
#if defined(__linux__)
        // For the main executable, glibc reports argv[0], which need not be a path.
        module = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (ec) {
            return {};
        }
#else
        return {};
#endif
    }
    return module.parent_path();
#endif
}

}  // namespace

std::filesystem::path resolve_bundled_asset(const std::filesystem::path & path) {
    if (path.empty() || path.has_root_path() || exists_quietly(path)) {
        return path;
    }
    static const std::filesystem::path module_dir = module_directory();
    std::filesystem::path root = module_dir;
    for (int level = 0; level <= kParentLevels && !root.empty(); ++level) {
        const std::filesystem::path candidate = root / path;
        if (exists_quietly(candidate)) {
            return candidate;
        }
        const std::filesystem::path parent = root.parent_path();
        if (parent == root) {
            break;
        }
        root = parent;
    }
    return path;
}

}  // namespace engine::assets
