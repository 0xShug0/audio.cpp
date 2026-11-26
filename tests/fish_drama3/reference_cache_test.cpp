#include "../../src/community_models/fish_drama3/reference_cache.cpp"
#include "../../src/community_models/fish_drama3/prefill_cache.h"

#include "engine/framework/runtime/errors.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#else
#include <unistd.h>
#endif

namespace {
namespace d3 = engine::models::fish_drama3;
namespace fs = std::filesystem;
using engine::runtime::AudioBuffer;
using Outcome = d3::Drama3ReferenceCache::Outcome;

void expect(bool condition, const std::string & what) {
    if (!condition) throw std::runtime_error("FAILED: " + what);
}

AudioBuffer voice(float seed, size_t count = 4096, int rate = 44100, int channels = 1) {
    AudioBuffer audio;
    audio.sample_rate = rate;
    audio.channels = channels;
    audio.samples.resize(count);
    for (size_t i = 0; i < count; ++i) audio.samples[i] = seed + static_cast<float>(i % 97) * 0.001F;
    return audio;
}

struct FakeEncoder {
    int calls = 0;
    engine::codecs::FishDacCodes operator()(const AudioBuffer & audio) {
        ++calls;
        engine::codecs::FishDacCodes codes;
        codes.codebooks = 1;
        codes.frames = 2;
        codes.codes = {static_cast<int32_t>(audio.samples.front() * 1000.0F), static_cast<int32_t>(audio.samples.size())};
        return codes;
    }
};

d3::Drama3ReferenceCache::Encoder bind(FakeEncoder & encoder) {
    return [&encoder](const AudioBuffer & audio) { return encoder(audio); };
}

int alternate(size_t slots) {
    d3::Drama3ReferenceCache cache(slots);
    FakeEncoder encoder;
    const auto a = voice(1.0F);
    const auto b = voice(2.0F);
    for (int i = 0; i < 3; ++i) {
        const auto codes_a = cache.lookup_or_encode(a, bind(encoder));
        expect(codes_a.codes.front() == 1000, "voice A returns its own codes");
        const auto codes_b = cache.lookup_or_encode(b, bind(encoder));
        expect(codes_b.codes.front() == 2000, "voice B returns its own codes");
    }
    return encoder.calls;
}

void test_alternating() {
    expect(alternate(2) == 2, "slots 2: two voices alternating encode twice in 6 lookups");
    expect(alternate(1) == 6, "slots 1: alternating encodes every time");
    expect(alternate(0) == 6, "slots 0: caching disabled");
}

void test_lru_eviction() {
    d3::Drama3ReferenceCache cache(2);
    FakeEncoder encoder;
    const auto a = voice(1.0F), b = voice(2.0F), c = voice(3.0F);
    Outcome outcome{};
    cache.lookup_or_encode(a, bind(encoder));
    cache.lookup_or_encode(b, bind(encoder));
    cache.lookup_or_encode(c, bind(encoder));  // evicts A, the least recently used
    expect(encoder.calls == 3, "three distinct voices encode three times");
    cache.lookup_or_encode(b, bind(encoder), &outcome);
    expect(outcome == Outcome::Hit && encoder.calls == 3, "B survives the eviction");
    cache.lookup_or_encode(a, bind(encoder), &outcome);
    expect(outcome == Outcome::Miss && encoder.calls == 4, "A was evicted");
    cache.lookup_or_encode(b, bind(encoder), &outcome);
    expect(outcome == Outcome::Hit, "B was used more recently than C, so C was evicted for A");
    cache.lookup_or_encode(c, bind(encoder), &outcome);
    expect(outcome == Outcome::Miss && encoder.calls == 5, "C was evicted");
}

void test_exact_match() {
    d3::Drama3ReferenceCache cache(4);
    FakeEncoder encoder;
    Outcome outcome{};
    const auto a = voice(1.0F);
    cache.lookup_or_encode(a, bind(encoder), &outcome);
    expect(outcome == Outcome::Miss, "first lookup misses");

    auto copy = voice(1.0F);  // same samples in a new buffer
    cache.lookup_or_encode(copy, bind(encoder), &outcome);
    expect(outcome == Outcome::Hit && encoder.calls == 1, "same samples in a new buffer hit");

    auto changed = a;
    changed.samples[2048] += 1e-6F;
    cache.lookup_or_encode(changed, bind(encoder), &outcome);
    expect(outcome == Outcome::Miss && encoder.calls == 2, "one changed sample misses");

    auto rate = a;
    rate.sample_rate = 48000;
    cache.lookup_or_encode(rate, bind(encoder), &outcome);
    expect(outcome == Outcome::Miss && encoder.calls == 3, "different sample rate misses");

    auto stereo = a;
    stereo.channels = 2;
    cache.lookup_or_encode(stereo, bind(encoder), &outcome);
    expect(outcome == Outcome::Miss && encoder.calls == 4, "different channel count misses");
}

void test_long_clip() {
    d3::Drama3ReferenceCache cache(2);
    FakeEncoder encoder;
    Outcome outcome{};
    const auto a = voice(1.0F), b = voice(2.0F);
    const auto long_clip = voice(5.0F, d3::Drama3ReferenceCache::max_cached_samples + 1);
    cache.lookup_or_encode(a, bind(encoder));
    cache.lookup_or_encode(b, bind(encoder));
    cache.lookup_or_encode(long_clip, bind(encoder), &outcome);
    expect(outcome == Outcome::Skip && encoder.calls == 3, "a clip over the limit is encoded and skipped");
    cache.lookup_or_encode(long_clip, bind(encoder), &outcome);
    expect(outcome == Outcome::Skip && encoder.calls == 4, "a clip over the limit is never stored");
    cache.lookup_or_encode(a, bind(encoder), &outcome);
    expect(outcome == Outcome::Hit, "voice A survives a long clip");
    cache.lookup_or_encode(b, bind(encoder), &outcome);
    expect(outcome == Outcome::Hit && encoder.calls == 4, "voice B survives a long clip");
    expect(cache.size() == 2, "the cache holds two entries");
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto base = std::filesystem::temp_directory_path();
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path = base / ("audiocpp-drama3-cache-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("Unable to create a temporary test directory");
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

class WarningLog {
public:
    WarningLog() {
        engine::debug::set_log_callback([](engine::debug::LogLevel level, const char * text, void * data) {
            if (level == engine::debug::LogLevel::Warning) static_cast<WarningLog *>(data)->messages.emplace_back(text);
        }, this);
    }
    ~WarningLog() { engine::debug::set_log_callback(nullptr, nullptr); }

    std::vector<std::string> messages;
};

void test_disk_cache() {
    TemporaryDirectory directory;
    WarningLog warnings;
    const auto audio = voice(1.0F);
    const engine::codecs::FishDacCodes expected{std::vector<int32_t>(20, 42), 10, 2};
    int calls = 0;
    const auto encode = [&](const AudioBuffer &) {
        ++calls;
        return expected;
    };
    const auto cache_for = [&](std::string identity, size_t slots = 1) {
        d3::Drama3ReferenceCache cache(slots);
        cache.set_disk_cache(directory.path, std::move(identity), 128);
        return cache;
    };
    auto original = cache_for("encoder-a");
    Outcome outcome{};
    expect(original.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::Miss && calls == 1, "cold cache encodes");

    const auto entries = directory.path / "fish_drama3-reference";
    fs::path path;
    for (const auto & entry : fs::directory_iterator(entries)) {
        if (entry.path().extension() == ".cache") path = entry.path();
    }
    expect(!path.empty(), "reference cache uses its dedicated subdirectory");
    auto warm = cache_for("encoder-a");
    expect(warm.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::DiskHit && calls == 1, "fresh cache reads the same codes from disk");
    const auto corrupt = [&] {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(-1, std::ios::end);
        file.put('x');
        expect(static_cast<bool>(file), "corrupt a payload byte without changing the file size");
    };
    corrupt();
    warm.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::Hit && calls == 1, "disk hit was promoted to the in-memory LRU");

    auto repaired = cache_for("encoder-a", 0);
    expect(repaired.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::Miss && calls == 2, "corrupt payload is recomputed");
    expect(warnings.messages.size() == 1 && warnings.messages.front().find("checksum mismatch") != std::string::npos,
        "corruption warns with its cause");
    corrupt();
    repaired.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::Miss && calls == 3 && warnings.messages.size() == 1,
        "repeated read failures in one session still recompute and warn only once");
    auto reread = cache_for("encoder-a");
    reread.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::DiskHit && calls == 3, "valid content replaces a corrupt file");

    engine::io::DiskCache disk({entries, 64, 1024 * 1024});
    const auto key = d3::reference_key("encoder-a", audio);
    auto payload = disk.get(key).value();
    payload.resize(payload.size() - 10 * sizeof(int32_t));
    disk.put(key, payload);
    expect(repaired.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::Miss && calls == 4 && warnings.messages.size() == 1,
        "a checksummed payload with the wrong size is a miss");
    auto resized = cache_for("encoder-a");
    resized.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::DiskHit && calls == 4, "wrong-sized payload is replaced");

    auto changed = cache_for("encoder-b");
    changed.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::Miss && calls == 5, "new encoder identity misses");
}

void test_unwritable_cache() {
#ifndef _WIN32
    if (geteuid() == 0) return;
#endif
    TemporaryDirectory directory;
    WarningLog warnings;
    const auto entries = directory.path / "fish_drama3-reference";
#ifdef _WIN32
    // Deny FILE_ADD_FILE on this test-only directory, including for its owner.
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    expect(ConvertStringSecurityDescriptorToSecurityDescriptorW(
        L"D:P(D;;0x2;;;WD)(A;;FA;;;OW)", SDDL_REVISION_1, &descriptor, nullptr) != 0,
        "build a read-only test directory ACL");
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE};
    const bool created = CreateDirectoryW(entries.c_str(), &attributes) != 0;
    LocalFree(descriptor);
    expect(created, "create the directory with writes denied");
#else
    fs::create_directory(entries);
    fs::permissions(entries, fs::perms::owner_read | fs::perms::owner_exec);
#endif
    d3::Drama3ReferenceCache cache(0);
    cache.set_disk_cache(directory.path, "encoder", 128);
    const engine::codecs::FishDacCodes expected{std::vector<int32_t>(20, 42), 10, 2};
    Outcome outcome{};
    const auto encode = [&](const AudioBuffer &) { return expected; };
    const bool encoded = cache.lookup_or_encode(voice(1.0F), encode, &outcome).codes == expected.codes &&
        outcome == Outcome::Miss;
    const bool repeated = cache.lookup_or_encode(voice(2.0F), encode).codes == expected.codes;
#ifndef _WIN32
    fs::permissions(entries, fs::perms::owner_all);
#endif
    expect(encoded && repeated, "unwritable cache still returns the freshly encoded conditioning");
    expect(warnings.messages.size() == 2 &&
        warnings.messages[0].find("Cache read failed") != std::string::npos &&
        warnings.messages[1].find("Cache write failed") != std::string::npos,
        "unwritable cache warns once per operation");
    expect(fs::is_empty(entries), "permission failure publishes no files");
}

void test_prefill_payload() {
    TemporaryDirectory directory;
    d3::Sha256 hash;
    hash.update("a");
    hash.update("bc");
    expect(hash.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 known answer");
    const std::vector<std::byte> backbone(5, std::byte{1}), tower(3, std::byte{2});
    const std::vector<float> hidden{0.25F, -0.5F};
    const auto payload = d3::pack_prefill(backbone, tower, hidden);
    const auto legacy_prefill = directory.path / (std::string(64, 'a') + ".prefill");
    const auto legacy_vq = directory.path / (std::string(64, 'b') + ".vq");
    std::ofstream(legacy_prefill).put('p');
    std::ofstream(legacy_vq).put('v');
    engine::io::DiskCache cache({directory.path, 1, 1024});
    cache.put("fish_drama3/prefill/v1/old", payload);
    cache.put("fish_drama3/prefill/v1/new", payload);
    expect(std::ifstream(legacy_prefill).get() == 'p' && std::ifstream(legacy_vq).get() == 'v',
        "eviction leaves legacy files intact");
    const auto stored = cache.get("fish_drama3/prefill/v1/new").value();
    const auto state = d3::unpack_prefill(stored, backbone.size(), tower.size(), hidden.size());
    expect(state && state->backbone == backbone && state->tower == tower && state->hidden == hidden,
        "round trip all three prefill sections");
    expect(!d3::unpack_prefill(stored, backbone.size(), tower.size(), hidden.size() - 1) &&
           !d3::unpack_prefill(stored, backbone.size(), tower.size(), hidden.size() + 1),
        "reject checksummed snapshots with a different payload size");
}

void expect_rejected(const std::string & value) {
    try {
        (void)d3::parse_reference_cache_slots({{"fish_drama3.reference_cache_slots", value}});
    } catch (const engine::runtime::InvalidRequestError & error) {
        expect(std::string(error.what()).find("reference_cache_slots") != std::string::npos,
               "the error for '" + value + "' names the option");
        return;
    }
    throw std::runtime_error("FAILED: reference_cache_slots '" + value + "' was accepted");
}

void test_option() {
    expect(d3::parse_reference_cache_slots({}) == 8, "default is 8 slots");
    expect(d3::parse_reference_cache_slots({{"fish_drama3.reference_cache_slots", "0"}}) == 0, "0 is accepted");
    expect(d3::parse_reference_cache_slots({{"fish_drama3.reference_cache_slots", "64"}}) == 64, "64 is accepted");
    expect_rejected("-1");
    expect_rejected("65");
    expect_rejected("abc");
}
}  // namespace

int main() {
    try {
        test_alternating();
        test_lru_eviction();
        test_exact_match();
        test_long_clip();
        test_disk_cache();
        test_unwritable_cache();
        test_prefill_payload();
        test_option();
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    std::cout << "fish_drama3 reference cache: all checks passed\n";
    return 0;
}
