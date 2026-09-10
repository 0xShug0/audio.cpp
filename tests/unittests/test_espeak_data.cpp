#include "engine/framework/audio/espeak_data.h"
#include "test_assert.h"
#include "gguf.h"
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <random>

namespace fs = std::filesystem;
static void write(const fs::path & path, const std::string & text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary); out << text;
    if (!out) throw std::runtime_error("fixture write failed");
}
int main() try {
    using engine::audio::pack_espeak_data;
    using engine::audio::materialize_espeak_data;
    using engine::test::require;
    const auto root = fs::temp_directory_path() / ("espeak-package-test-" + std::to_string(std::random_device{}()));
    fs::create_directories(root);
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{root};
#ifdef _WIN32
    _putenv_s("LOCALAPPDATA", root.string().c_str());
#else
    // Exercise a system-style cache alias, as used by macOS /var and /tmp.
    const auto actual_cache = root / "actual-cache", cache_alias = root / "cache-alias";
    fs::create_directory(actual_cache);
    fs::create_directory_symlink(actual_cache, cache_alias);
    setenv("XDG_CACHE_HOME", cache_alias.string().c_str(), 1);
#endif
    const auto source = root / "source", package = root / "data.bin";
    for (const auto * name : {"phontab", "phondata", "phonindex", "intonations", "en_dict", "lang/en", "voices/!v/Mr serious"})
        write(source / name, std::string(name) + std::string("\0binary", 7));
    pack_espeak_data(source, package);
    std::vector<std::future<fs::path>> jobs;
    for (int i = 0; i < 8; ++i)
        jobs.push_back(std::async(std::launch::async, [&] { return materialize_espeak_data(package); }));
    const auto cached = jobs.front().get();
    for (size_t i = 1; i < jobs.size(); ++i) require(jobs[i].get() == cached, "concurrent cache identity");
    const auto timestamp = fs::last_write_time(cached / "phontab");
    require(materialize_espeak_data(package) == cached, "reuse cache");
    const auto legacy = root / "data.gguf";
    fs::copy_file(package, legacy);
    require(materialize_espeak_data(legacy) == cached, "legacy extension shares cache");
#ifndef _WIN32
    require(cached.parent_path().parent_path().parent_path() == fs::canonical(actual_cache) / "audio.cpp",
            "cache anchor resolves directory aliases");
    const auto redirected_cache = root / "redirected-cache";
    fs::create_directory(redirected_cache);
    fs::create_directory_symlink(actual_cache / "audio.cpp", redirected_cache / "audio.cpp");
    setenv("XDG_CACHE_HOME", redirected_cache.string().c_str(), 1);
    bool cache_link_rejected = false;
    try { materialize_espeak_data(package); } catch (const std::exception &) { cache_link_rejected = true; }
    require(cache_link_rejected, "reject symlink inside cache anchor");
    setenv("XDG_CACHE_HOME", cache_alias.string().c_str(), 1);
#endif
    require(timestamp == fs::last_write_time(cached / "phontab"), "cache must not be rewritten");
    for (const auto & entry : fs::recursive_directory_iterator(source)) {
        if (!entry.is_regular_file()) continue;
        const auto target = cached / entry.path().lexically_relative(source);
        std::ifstream a(entry.path(), std::ios::binary), b(target, std::ios::binary);
        require(std::string(std::istreambuf_iterator<char>(a), {}) == std::string(std::istreambuf_iterator<char>(b), {}), "binary round trip");
    }
    write(cached / "phontab", "damaged");
    const auto repaired = materialize_espeak_data(package);
    require(repaired != cached && fs::file_size(repaired / "phontab") == fs::file_size(source / "phontab"), "repair damaged cache");
    write(source / "en_dict", "updated dictionary");
    pack_espeak_data(source, package);
    require(materialize_espeak_data(package) != repaired, "content change invalidates cache");
    auto * ctx = gguf_init_from_file(package.string().c_str(), {true, nullptr});
    require(ctx != nullptr, "open fixture GGUF");
    const char * unsafe[] = {"../escaped"};
    gguf_set_arr_str(ctx, "audiocpp.embedded_files.names", unsafe, 1);
    uint64_t offsets[] = {0, 1}; uint8_t data[] = {1};
    gguf_set_arr_data(ctx, "audiocpp.embedded_files.offsets", GGUF_TYPE_UINT64, offsets, 2);
    gguf_set_arr_data(ctx, "audiocpp.embedded_files.data", GGUF_TYPE_UINT8, data, 1);
    require(gguf_write_to_file(ctx, package.string().c_str(), true), "write hostile fixture");
    gguf_free(ctx);
    bool rejected = false;
    try { materialize_espeak_data(package); } catch (const std::exception &) { rejected = true; }
    require(rejected, "reject path traversal");
    require(!fs::exists(root / "escaped"), "no escaped file");
    std::cout << "eSpeak data-package roundtrip, concurrency, reuse, repair, update and traversal tests passed\n";
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
