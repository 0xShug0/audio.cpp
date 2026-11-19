#include "engine/community_models/fish_drama3/reference_cache.h"

#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"

#include <cstring>

namespace engine::models::fish_drama3 {

bool Drama3ReferenceCache::SameWaveform::operator()(const runtime::AudioBuffer & a,
                                                    const runtime::AudioBuffer & b) const {
    return a.sample_rate == b.sample_rate && a.channels == b.channels && a.samples.size() == b.samples.size() &&
           (a.samples.empty() ||
            std::memcmp(a.samples.data(), b.samples.data(), a.samples.size() * sizeof(float)) == 0);
}

Drama3ReferenceCache::Drama3ReferenceCache(std::size_t slots)
    : slots_(slots) {}

codecs::FishDacCodes Drama3ReferenceCache::lookup_or_encode(const runtime::AudioBuffer & audio,
                                                             const Encoder & encode, Outcome * outcome) {
    const bool storable = slots_.capacity() > 0 && audio.samples.size() <= max_cached_samples;
    if (storable) {
        if (const auto * codes = slots_.find(audio)) {
            if (outcome) *outcome = Outcome::Hit;
            return *codes;
        }
    }
    auto codes = encode(audio);
    if (storable) slots_.put(audio, codes);
    if (outcome) *outcome = audio.samples.size() > max_cached_samples ? Outcome::Skip : Outcome::Miss;
    return codes;
}

std::size_t parse_reference_cache_slots(const std::unordered_map<std::string, std::string> & options) {
    constexpr const char * key = "fish_drama3.reference_cache_slots";
    const auto value = runtime::parse_int_option(options, {key});
    if (!value) return Drama3ReferenceCache::default_slots;
    if (*value < 0 || static_cast<std::size_t>(*value) > Drama3ReferenceCache::max_slots) {
        throw runtime::InvalidRequestError(std::string(key) + " must be between 0 and 64");
    }
    return static_cast<std::size_t>(*value);
}

}  // namespace engine::models::fish_drama3
