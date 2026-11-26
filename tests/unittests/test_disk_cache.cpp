#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "engine/framework/io/disk_cache.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/debug/trace.h"
#include "test_assert.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <locale>
#include <random>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using engine::io::DiskCache;
using engine::test::require;
using Bytes = std::vector<std::byte>;

namespace {
struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() /
        ("audiocpp-disk-cache-test-" + std::to_string(std::random_device{}()));
    TemporaryDirectory() { require(fs::create_directory(path), "create test directory"); }
    ~TemporaryDirectory() { std::error_code error; fs::remove_all(path, error); }
};

struct GroupedNumbers : std::numpunct<char> {
    char do_thousands_sep() const override { return ','; }
    std::string do_grouping() const override { return "\3"; }
};

template <typename F> void must_throw(F && call, const char * label) {
    bool threw = false;
    try { call(); } catch (const std::exception &) { threw = true; }
    require(threw, label);
}

fs::path only_entry(const fs::path & directory) {
    fs::path found;
    for (const auto & item : fs::directory_iterator(directory)) {
        if (item.path().extension() == ".cache") {
            require(found.empty(), "expected one cache entry");
            found = item.path();
        }
    }
    require(!found.empty(), "missing cache entry");
    return found;
}

void write_repeated(const fs::path & directory, unsigned value) {
    DiskCache cache({directory, 4, 1024 * 1024});
    for (unsigned i = 0; i < 12; ++i)
        cache.put("shared", Bytes(128 * 1024, std::byte(value + i)));
}
}  // namespace

int main(int argc, char ** argv) try {
    const auto started = std::chrono::steady_clock::now();
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--log") engine::debug::configure_logging({true, std::nullopt});
    }
    // Child processes use the same public API; no model or bespoke probe needed.
    if (argc == 5 && std::string(argv[1]) == "--writer") {
        write_repeated(argv[2], static_cast<unsigned>(std::stoul(argv[3])));
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        return 0;
    }
    TemporaryDirectory temporary;
    if (argc == 3 && std::string(argv[1]) == "--sync-failure") {
        // Run under strace with the second fsync injected as EIO. Keep fault
        // injection outside production code and verify the public API outcome.
        DiskCache faulted({temporary.path / "sync-failure", 2, 1024});
        const Bytes original{std::byte{1}};
        const Bytes replacement{std::byte{2}};
        faulted.put("key", original);
        bool reported = false;
        try { faulted.put("key", replacement); }
        catch (const std::system_error & error) { reported = error.code().value() == EIO; }
        require(reported, "expected injected fsync EIO");
        require(!fs::exists(temporary.path / "sync-failure/.audiocpp-disk-cache.tmp"), "failed sync removes temporary file");
        require(faulted.get("key") == original, "failed sync preserves published value");
        faulted.put("key", replacement);
        require(faulted.get("key") == replacement, "retry after failed sync");
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        std::cout << "disk_cache_test sync-failure passed\n";
        return 0;
    }
    DiskCache cache({temporary.path / "basic", 2, 1024});
    require(!cache.get("missing"), "missing key");
    const Bytes bytes{std::byte{0}, std::byte{255}, std::byte{17}};
    cache.put("schema1/model/backend/input", bytes);
    require(cache.get("schema1/model/backend/input") == bytes, "binary round trip");
    DiskCache reopened({temporary.path / "basic", 2, 1024});
    require(reopened.get("schema1/model/backend/input") == bytes, "reopen hit");
    const auto old_locale = std::locale();
    std::locale::global(std::locale(old_locale, new GroupedNumbers));
    const auto locale_hit = cache.get("schema1/model/backend/input");
    std::locale::global(old_locale);
    require(locale_hit == bytes, "cache names are locale independent");
    // Independent format fixture: FNV-1a("a") = af63dc4c8601ec8c, empty payload.
    // Verify both decoding and encoding against fixed bytes, not each other.
    const auto format_directory = temporary.path / "format";
    fs::create_directory(format_directory);
    const auto format_path = format_directory / "af63dc4c8601ec8c.cache";
    const unsigned char expected[] = {
        'A', 'C', 'D', 'C', '0', '0', '0', '1',
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0x8c, 0xec, 0x01, 0x86, 0x4c, 0xdc, 0x63, 0xaf, 'a'
    };
    {
        std::ofstream file(format_path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(expected), sizeof(expected));
    }
    DiskCache format({format_directory, 2, 1024});
    require(format.get("a") == Bytes{}, "decode independent format fixture");
    format.put("a", {});
    const auto encoded = engine::io::read_binary_file(format_path);
    require(encoded.size() == sizeof(expected), "encoded format size");
    for (size_t i = 0; i < encoded.size(); ++i)
        require(encoded[i] == std::byte{expected[i]}, "encode independent format fixture");
    cache.put("schema1/model/backend/input", {});
    require(cache.get("schema1/model/backend/input") == Bytes{}, "empty value is not a miss");
    cache.put("schema1/model/backend/input", bytes);
    must_throw([&] { cache.put("large", Bytes(1024)); }, "reject over-budget entry");
    require(cache.get("schema1/model/backend/input") == bytes, "failed write preserves old entry");
    must_throw([&] { cache.get(""); }, "reject empty key");
    must_throw([&] { DiskCache invalid({temporary.path, 0, 1024}); }, "reject zero capacity");

    const auto basic = temporary.path / "basic";
    const auto entry = only_entry(basic);
    const auto last_byte = static_cast<size_t>(fs::file_size(entry) - 1);
    for (const auto offset : {size_t{0}, size_t{8}, size_t{16}, size_t{24}, size_t{32}, last_byte}) {
        cache.put("schema1/model/backend/input", bytes);
        std::fstream file(entry, std::ios::binary | std::ios::in | std::ios::out);
        file.seekg(static_cast<std::streamoff>(offset));
        char value = 0;
        file.get(value);
        file.seekp(static_cast<std::streamoff>(offset));
        file.put(static_cast<char>(static_cast<unsigned char>(value) ^ 255));
        file.close();
        must_throw([&] { cache.get("schema1/model/backend/input"); }, "reject corrupt magic/length/checksum/key/payload");
    }
    cache.put("schema1/model/backend/input", bytes);
    fs::resize_file(entry, fs::file_size(entry) - 1);
    must_throw([&] { cache.get("schema1/model/backend/input"); }, "reject truncated payload");
    cache.put("schema1/model/backend/input", bytes);
    // A different complete key under an otherwise valid filename is a miss.
    DiskCache other({temporary.path / "other", 2, 1024});
    other.put("another-key", bytes);
    fs::copy_file(only_entry(temporary.path / "other"), entry, fs::copy_options::overwrite_existing);
    require(!cache.get("schema1/model/backend/input"), "never return another key's payload");

    // LRU and byte bounds are distinct paths. Touching a hit protects it.
    DiskCache lru({temporary.path / "lru", 2, 1024});
    lru.put("a", bytes);
    const auto a_path = only_entry(temporary.path / "lru");
    lru.put("b", bytes);
    fs::last_write_time(a_path, fs::file_time_type::clock::now() - std::chrono::hours(1));
    require(lru.get("b") == bytes, "touch recent entry");
    lru.put("c", bytes);
    require(!lru.get("a") && lru.get("b") == bytes && lru.get("c") == bytes, "evict oldest entry");
    DiskCache bounded({temporary.path / "bounded", 10, 100});
    bounded.put("a", Bytes(30));
    bounded.put("b", Bytes(30));
    require(!bounded.get("a") && bounded.get("b") == Bytes(30), "evict by total file bytes");

    // Recovery touches only the cache's own interrupted-write file.
    std::ofstream(basic / ".audiocpp-disk-cache.tmp") << "interrupted write";
    std::ofstream(basic / "unrelated.tmp") << "not owned by cache";
    require(!cache.get("absent"), "recovery miss");
    require(!fs::exists(basic / ".audiocpp-disk-cache.tmp") && fs::exists(basic / "unrelated.tmp"),
            "recover orphan without sweeping unrelated files");

    const auto bad_directory = temporary.path / "not-a-directory";
    std::ofstream(bad_directory) << "file";
    DiskCache bad({bad_directory, 2, 1024});
    must_throw([&] { bad.put("key", bytes); }, "I/O failure must not be silent");
    // Force a temporary-file cleanup error, then prove subsequent recovery.
    fs::create_directory(basic / ".audiocpp-disk-cache.tmp");
    std::ofstream(basic / ".audiocpp-disk-cache.tmp/blocked") << "file";
    must_throw([&] { cache.put("key", bytes); }, "temporary-path failure reported");
    fs::remove_all(basic / ".audiocpp-disk-cache.tmp");
    cache.put("key", bytes);
    require(cache.get("key") == bytes, "retry after I/O failure");

#ifndef _WIN32
    if (geteuid() != 0) {
        const auto read_only = temporary.path / "read-only";
        DiskCache denied({read_only, 2, 1024});
        denied.put("key", bytes);
        fs::permissions(read_only, fs::perms::owner_read | fs::perms::owner_exec);
        bool reported = false;
        try { denied.put("key", Bytes(20)); } catch (const std::exception &) { reported = true; }
        fs::permissions(read_only, fs::perms::owner_all);
        require(reported && denied.get("key") == bytes, "write permission error preserves old entry");
    }
#endif

    const auto concurrent = temporary.path / "concurrent";
    DiskCache shared({concurrent, 4, 1024 * 1024});
    shared.put("shared", Bytes(128 * 1024, std::byte{1}));
    auto writer = std::async(std::launch::async, [&] { write_repeated(concurrent, 2); });
    for (unsigned i = 0; i < 12; ++i) {
        const auto value = shared.get("shared");
        require(value && value->size() == 128 * 1024, "concurrent read size");
        for (auto item : *value) require(item == value->front(), "no partial concurrent payload");
    }
    writer.get();
    const auto executable = fs::absolute(argv[0]).string();
    auto child = [&](unsigned value) {
        const auto parameter = std::to_string(value);
#ifdef _WIN32
        const auto directory = concurrent.string();
        const char * arguments[] = {executable.c_str(), "--writer", directory.c_str(), parameter.c_str(), "--log", nullptr};
        return static_cast<int>(_spawnv(_P_WAIT, executable.c_str(), arguments));
#else
        const auto command = "\"" + executable + "\" --writer \"" + concurrent.string() + "\" " + parameter + " --log";
        return std::system(command.c_str());
#endif
    };
    auto first = std::async(std::launch::async, [&] { return child(40); });
    auto second = std::async(std::launch::async, [&] { return child(80); });
    require(first.get() == 0 && second.get() == 0, "independent process writers");
    const auto final = shared.get("shared");
    require(final && (final->front() == std::byte{51} || final->front() == std::byte{91}), "complete final process value");
    for (auto item : *final) require(item == final->front(), "complete process payload");
    require(!fs::exists(concurrent / ".audiocpp-disk-cache.tmp"), "writers leave no temporary file");

#ifndef _WIN32
    // Hold the actual directory lock with a partial temporary file. Readers
    // must wait, and a killed writer must release the lock for orphan recovery.
    const auto interrupted = temporary.path / "interrupted";
    DiskCache recovering({interrupted, 2, 1024});
    recovering.put("key", bytes);
    const auto lock_path = (interrupted / ".audiocpp-disk-cache.lock").string();
    const auto partial_path = (interrupted / ".audiocpp-disk-cache.tmp").string();
    int ready[2];
    require(pipe(ready) == 0, "create child readiness pipe");
    const auto pid = fork();
    require(pid >= 0, "fork interrupted writer");
    if (pid == 0) {
        close(ready[0]);
        const int fd = open(lock_path.c_str(), O_RDWR);
        if (fd < 0 || flock(fd, LOCK_EX) != 0) _exit(2);
        const int partial = open(partial_path.c_str(), O_CREAT | O_WRONLY, 0600);
        if (partial < 0 || write(partial, "partial", 7) != 7) _exit(3);
        close(partial);
        if (write(ready[1], "1", 1) != 1) _exit(4);
        close(ready[1]);
        for (;;) pause();
    }
    close(ready[1]);
    char acknowledged = 0;
    const auto received = read(ready[0], &acknowledged, 1);
    close(ready[0]);
    if (received != 1) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        require(false, "interrupted writer acquired lock");
    }
    auto reader = std::async(std::launch::async, [&] { return recovering.get("key"); });
    const bool waited = reader.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
    const bool preserved_active = fs::exists(partial_path);
    const int killed = kill(pid, SIGKILL);
    int status = 0;
    const auto reaped = waitpid(pid, &status, 0);
    const auto recovered = reader.get();
    require(waited && preserved_active, "never delete an active writer's temporary file");
    require(killed == 0 && reaped == pid && WIFSIGNALED(status), "writer killed and reaped");
    require(recovered == bytes && !fs::exists(partial_path), "recover after killed writer without losing old value");
#endif
    engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count());
    std::cout << "disk_cache_test passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "disk_cache_test: " << error.what() << '\n';
    return 1;
}
