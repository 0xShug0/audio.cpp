#pragma once

#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

namespace engine::models::fish_drama3 {

// Encoded reference voices, least recently used first out, keyed by the exact waveform
// (sample rate, channel count and every sample byte). Not thread-safe: Drama3Generator calls it
// only from encode_reference, which runs inside a synthesis request under the session lock.
class Drama3ReferenceCache {
public:
    enum class Outcome { Hit, Miss, Skip };
    using Encoder = std::function<codecs::FishDacCodes(const runtime::AudioBuffer &)>;

    // Clips longer than 30 s of 44.1 kHz mono (interleaved samples) are encoded but never stored.
    static constexpr std::size_t max_cached_samples = 44100 * 30;
    static constexpr std::size_t max_slots = 64;
    static constexpr std::size_t default_slots = 8;

    explicit Drama3ReferenceCache(std::size_t slots);

    // Returns the stored codes on a hit; otherwise calls encode and stores the result.
    codecs::FishDacCodes lookup_or_encode(const runtime::AudioBuffer & audio, const Encoder & encode,
                                          Outcome * outcome = nullptr);

    std::size_t size() const noexcept { return slots_.size(); }
    std::size_t capacity() const noexcept { return slots_.capacity(); }

private:
    struct SameWaveform {
        bool operator()(const runtime::AudioBuffer & a, const runtime::AudioBuffer & b) const;
    };
    runtime::CacheSlots<runtime::AudioBuffer, codecs::FishDacCodes, SameWaveform> slots_;
};

// Reads fish_drama3.reference_cache_slots: an integer from 0 (caching off) to 64, default 8.
// Throws InvalidRequestError naming the option for any other value.
std::size_t parse_reference_cache_slots(const std::unordered_map<std::string, std::string> & options);

}  // namespace engine::models::fish_drama3
