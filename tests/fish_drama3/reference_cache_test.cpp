#include "../../src/community_models/fish_drama3/reference_cache.cpp"
#include "../../src/community_models/fish_drama3/disk_cache.cpp"

#include "engine/framework/runtime/errors.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace d3 = engine::models::fish_drama3;
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

void test_disk_cache() {
    TemporaryDirectory directory;
    const auto audio = voice(1.0F);
    const engine::codecs::FishDacCodes expected{std::vector<int32_t>(20, 42), 10, 2};
    int calls = 0;
    const auto encode = [&](const AudioBuffer &) {
        ++calls;
        return expected;
    };
    const auto cache_for = [&](std::string identity) {
        d3::Drama3ReferenceCache cache(1);
        cache.set_disk_cache(directory.path, std::move(identity), 128);
        return cache;
    };
    auto original = cache_for("encoder-a");
    Outcome outcome{};
    expect(original.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::Miss && calls == 1, "cold cache encodes");

    const auto path = std::filesystem::directory_iterator(directory.path)->path();
    auto warm = cache_for("encoder-a");
    expect(warm.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::DiskHit && calls == 1, "fresh cache reads the same codes from disk");
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(12);
        file.put('x');
        expect(static_cast<bool>(file), "corrupt a payload byte without changing the file size");
    }
    warm.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::Hit && calls == 1, "disk hit was promoted to the in-memory LRU");

    auto repaired = cache_for("encoder-a");
    expect(repaired.lookup_or_encode(audio, encode, &outcome).codes == expected.codes &&
        outcome == Outcome::Miss && calls == 2, "corrupt payload is recomputed");
    auto reread = cache_for("encoder-a");
    reread.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::DiskHit && calls == 2, "valid content replaces a corrupt file");
    auto changed = cache_for("encoder-b");
    changed.lookup_or_encode(audio, encode, &outcome);
    expect(outcome == Outcome::Miss && calls == 3, "new encoder identity misses");
}

void test_cache_file() {
    TemporaryDirectory directory;
    d3::Sha256 hash;
    hash.update("a");
    hash.update("bc");
    expect(hash.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 known answer");
    const auto first = directory.path / (std::string(64, 'a') + ".prefill");
    const auto second = directory.path / (std::string(64, 'b') + ".prefill");
    const auto unrelated = directory.path / "notes.prefill";
    std::ofstream(unrelated).put('x');
    d3::write_cache_file(first, {"state", "row"}, 1);
    std::array<char, 5> state{};
    std::array<char, 3> row{};
    expect(d3::read_cache_file(first, {{state.data(), state.size()}, {row.data(), row.size()}}) &&
        std::string_view(state.data(), state.size()) == "state" &&
        std::string_view(row.data(), row.size()) == "row", "read two validated snapshot sections");
    expect(!d3::read_cache_file(first, {{state.data(), state.size()}}), "reject a different snapshot size");
    std::filesystem::copy_file(first, second);
    expect(!d3::read_cache_file(second, {{state.data(), state.size()}, {row.data(), row.size()}}),
        "payload identifies its cache key");
    std::filesystem::last_write_time(first, std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    d3::write_cache_file(second, {"state", "row"}, 1);
    expect(!std::filesystem::exists(first) && std::filesystem::exists(second) && std::filesystem::exists(unrelated),
        "prune the oldest cache entry and keep unrelated files");
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
        test_cache_file();
        test_option();
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    std::cout << "fish_drama3 reference cache: all checks passed\n";
    return 0;
}
