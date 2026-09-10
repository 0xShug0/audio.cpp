#include "engine/framework/audio/espeak_data.h"
#include "gguf.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace engine::audio {
namespace {
namespace fs = std::filesystem;
using Context = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Files = std::map<std::string, std::string>;
constexpr size_t limit = 64 * 1024 * 1024;
constexpr auto names_key = "audiocpp.embedded_files.names";
constexpr auto offsets_key = "audiocpp.embedded_files.offsets";
constexpr auto data_key = "audiocpp.embedded_files.data";

std::string read(const fs::path & path) {
    const auto size = fs::file_size(path);
    if (size > limit) throw std::runtime_error("eSpeak data file exceeds 64 MiB limit");
    std::ifstream in(path, std::ios::binary);
    std::string bytes(static_cast<size_t>(size), '\0');
    if (!in || !in.read(bytes.data(), static_cast<std::streamsize>(size)))
        throw std::runtime_error("Cannot read eSpeak data: " + path.string());
    return bytes;
}
void validate_name(const std::string & name) {
    if (name.empty() || name.size() > 240 || name.front() == '/' || name.back() == '/')
        throw std::runtime_error("Invalid eSpeak resource path");
    // Portable allowlist excludes drive letters, alternate streams, backslashes,
    // controls and Windows trailing-dot/space aliases. Upstream uses these ASCII names.
    for (unsigned char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '/' || c == '_' || c == '-' || c == '.' || c == '!' || c == ' '))
            throw std::runtime_error("Unsafe eSpeak resource path: " + name);
    for (const auto & part : fs::path(name)) {
        const auto s = part.string();
        auto stem = s.substr(0, s.find('.'));
        for (auto & c : stem) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        if (s.empty() || s == "." || s == ".." || s.back() == '.' || s.back() == ' ' ||
            stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" ||
            (stem.size() == 4 && (stem.substr(0, 3) == "com" || stem.substr(0, 3) == "lpt") &&
             stem[3] >= '0' && stem[3] <= '9'))
            throw std::runtime_error("Unsafe eSpeak resource component: " + name);
    }
    if (name.find("//") != std::string::npos) throw std::runtime_error("Invalid eSpeak resource path");
}
std::string folded(std::string name) {
    for (auto & c : name) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return name;
}
void required(const Files & files) {
    for (const auto * name : {"phontab", "phondata", "phonindex", "intonations", "en_dict"})
        if (!files.count(name) || files.at(name).empty())
            throw std::runtime_error(std::string("eSpeak data package missing ") + name);
}
// A cache name, not a cryptographic authenticity claim: all extracted bytes are
// compared to the package before reuse, including after a fingerprint collision.
std::string fingerprint(const Files & files) {
    uint64_t hash = 14695981039346656037ULL;
    for (const auto & file : files) {
        for (const auto & value : {file.first, file.second}) {
            for (unsigned char c : value) { hash ^= c; hash *= 1099511628211ULL; }
            hash ^= 255; hash *= 1099511628211ULL;
        }
    }
    std::ostringstream out; out << std::hex << hash; return out.str();
}
bool clean_path(const fs::path & path) {
    fs::path current;
    for (const auto & part : path) {
        current /= part;
        if (fs::is_symlink(fs::symlink_status(current))) return false;
    }
    return true;
}
bool valid_cache(const fs::path & root, const Files & files) {
    try {
        if (!clean_path(root) || !fs::is_directory(root)) return false;
        for (const auto & file : files) {
            const auto path = root / "espeak-ng-data" / file.first;
            if (!clean_path(path) || !fs::is_regular_file(path) || read(path) != file.second) return false;
        }
        size_t count = 0;
        for (const auto & entry : fs::recursive_directory_iterator(root)) {
            if (entry.is_symlink()) return false;
            if (entry.is_regular_file()) ++count;
        }
        return count == files.size();
    } catch (...) { return false; }
}
fs::path cache_base() {
    // User/system cache locations may contain legitimate directory aliases
    // (for example /var -> /private/var on macOS). Resolve that anchor before
    // appending our own cache directories, whose symlink checks remain strict.
    const auto anchored = [](const fs::path & path) {
        return fs::weakly_canonical(fs::absolute(path)) / "audio.cpp" / "espeak-data";
    };
#ifdef _WIN32
    const char * base = std::getenv("LOCALAPPDATA");
    if (!base || !*base) throw std::runtime_error("LOCALAPPDATA is required for the eSpeak cache");
    return anchored(base);
#else
    const char * base = std::getenv("XDG_CACHE_HOME");
    if (base && *base && fs::path(base).is_absolute()) return anchored(base);
    base = std::getenv("HOME");
    if (!base || !*base) throw std::runtime_error("HOME is required for the eSpeak cache");
    return anchored(fs::path(base) / ".cache");
#endif
}
}

void pack_espeak_data(const fs::path & directory, const fs::path & output) {
    Files files;
    std::set<std::string> unique_names;
    size_t total = 0;
    for (const auto & entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_symlink()) throw std::runtime_error("Symlinks are not permitted in eSpeak data");
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().lexically_relative(directory).generic_string();
        validate_name(name);
        if (!unique_names.insert(folded(name)).second) throw std::runtime_error("Case-colliding eSpeak paths");
        auto bytes = read(entry.path());
        total += bytes.size();
        if (total > limit || files.size() >= 4096) throw std::runtime_error("eSpeak data package too large");
        files.emplace(name, std::move(bytes));
    }
    required(files);
    Context ctx(gguf_init_empty(), gguf_free);
    gguf_set_val_str(ctx.get(), "general.architecture", "espeak-ng-data");
    gguf_set_val_u32(ctx.get(), "espeak.data.format_version", 1);
    gguf_set_val_str(ctx.get(), "espeak.data.engine_version", "1.52.0");
    std::vector<const char *> names;
    std::vector<uint64_t> offsets{0};
    std::vector<uint8_t> bytes;
    for (const auto & file : files) {
        names.push_back(file.first.c_str());
        bytes.insert(bytes.end(), file.second.begin(), file.second.end());
        offsets.push_back(bytes.size());
    }
    gguf_set_arr_str(ctx.get(), names_key, names.data(), names.size());
    gguf_set_arr_data(ctx.get(), offsets_key, GGUF_TYPE_UINT64, offsets.data(), offsets.size());
    gguf_set_arr_data(ctx.get(), data_key, GGUF_TYPE_UINT8, bytes.data(), bytes.size());
    std::vector<uint8_t> metadata(gguf_get_meta_size(ctx.get()));
    if (metadata.size() > limit) throw std::runtime_error("eSpeak GGUF exceeds 64 MiB limit");
    gguf_get_meta_data(ctx.get(), metadata.data());
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(metadata.data()), metadata.size());
    out.close();
    if (!out) throw std::runtime_error("Cannot write eSpeak data GGUF");
}

fs::path materialize_espeak_data(const fs::path & package) {
    if (fs::file_size(package) > limit) throw std::runtime_error("eSpeak GGUF exceeds 64 MiB limit");
#ifdef _WIN32
    std::unique_ptr<FILE, decltype(&fclose)> input(_wfopen(package.c_str(), L"rb"), fclose);
#else
    std::unique_ptr<FILE, decltype(&fclose)> input(fopen(package.c_str(), "rb"), fclose);
#endif
    if (!input) throw std::runtime_error("Cannot open eSpeak data package");
    Context ctx(gguf_init_from_file_ptr(input.get(), {true, nullptr}), gguf_free);
    if (!ctx || gguf_get_n_tensors(ctx.get()) != 0) throw std::runtime_error("Invalid eSpeak data GGUF");
    const auto key = [&](const char * name, gguf_type type) {
        const auto id = gguf_find_key(ctx.get(), name);
        if (id < 0 || gguf_get_kv_type(ctx.get(), id) != type) throw std::runtime_error("Invalid eSpeak data metadata");
        return id;
    };
    if (std::string(gguf_get_val_str(ctx.get(), key("general.architecture", GGUF_TYPE_STRING))) != "espeak-ng-data" ||
        gguf_get_val_u32(ctx.get(), key("espeak.data.format_version", GGUF_TYPE_UINT32)) != 1 ||
        std::string(gguf_get_val_str(ctx.get(), key("espeak.data.engine_version", GGUF_TYPE_STRING))) != "1.52.0")
        throw std::runtime_error("Unsupported eSpeak data package version");
    const auto names = key(names_key, GGUF_TYPE_ARRAY), offsets = key(offsets_key, GGUF_TYPE_ARRAY), data = key(data_key, GGUF_TYPE_ARRAY);
    const auto count = gguf_get_arr_n(ctx.get(), names), size = gguf_get_arr_n(ctx.get(), data);
    if (count == 0 || count > 4096 || size > limit || gguf_get_arr_type(ctx.get(), names) != GGUF_TYPE_STRING ||
        gguf_get_arr_type(ctx.get(), offsets) != GGUF_TYPE_UINT64 || gguf_get_arr_n(ctx.get(), offsets) != count + 1 ||
        gguf_get_arr_type(ctx.get(), data) != GGUF_TYPE_UINT8) throw std::runtime_error("Invalid eSpeak file table");
    const auto * positions = static_cast<const uint64_t *>(gguf_get_arr_data(ctx.get(), offsets));
    const auto * bytes = static_cast<const char *>(gguf_get_arr_data(ctx.get(), data));
    if (positions[0] != 0 || positions[count] != size) throw std::runtime_error("Invalid eSpeak data offsets");
    Files files;
    std::set<std::string> unique_names;
    for (size_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_arr_str(ctx.get(), names, i);
        validate_name(name);
        if (!unique_names.insert(folded(name)).second) throw std::runtime_error("Case-colliding eSpeak paths");
        if (positions[i] > positions[i + 1] || positions[i + 1] > size) throw std::runtime_error("Invalid eSpeak file range");
        if (!files.emplace(name, std::string(bytes + positions[i], positions[i + 1] - positions[i])).second)
            throw std::runtime_error("Duplicate eSpeak file name");
    }
    required(files);
    const auto base = cache_base();
    if (!clean_path(base)) throw std::runtime_error("Symlink in eSpeak cache path");
    fs::create_directories(base);
    // Publish whole directories; another process never observes partial files.
    // Damaged entries are not modified while another engine may be using them.
    const auto hash = fingerprint(files);
    for (int revision = 0; revision < 32; ++revision) {
        const auto root = base / (hash + "-" + std::to_string(revision));
        if (valid_cache(root, files)) return root / "espeak-ng-data";
        if (fs::exists(fs::symlink_status(root))) continue;
        std::random_device random;
        const auto stage = base / (hash + ".tmp-" + std::to_string(random()) + "-" + std::to_string(random()));
        if (!fs::create_directory(stage)) continue;
        try {
            for (const auto & file : files) {
                const auto path = stage / "espeak-ng-data" / file.first;
                fs::create_directories(path.parent_path());
                std::ofstream out(path, std::ios::binary);
                out.write(file.second.data(), file.second.size()); out.close();
                if (!out) throw std::runtime_error("Cannot extract eSpeak data");
            }
            std::error_code error;
            fs::rename(stage, root, error);
            if (!error) return root / "espeak-ng-data";
            fs::remove_all(stage);
            if (valid_cache(root, files)) return root / "espeak-ng-data";
        } catch (...) { fs::remove_all(stage); throw; }
    }
    throw std::runtime_error("Cannot publish eSpeak cache entry");
}
}
