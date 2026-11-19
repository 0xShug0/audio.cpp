#include "../../../src/community_models/fish_drama3/reference_cache.cpp"

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
        test_option();
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    std::cout << "fish_drama3 reference cache: all checks passed\n";
    return 0;
}
