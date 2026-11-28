#include "engine/community_models/fish_drama3/reference_cache.h"
#include "cache_key.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>

namespace engine::models::fish_drama3 {
namespace {

constexpr size_t kCodebooks = 10;
constexpr int64_t kMaxDiskFrames = 4096;
constexpr size_t kDiskReferenceEntries = 64;
constexpr uint64_t kDiskReferenceBytes = 1024 * 1024;

std::string reference_key(std::string_view identity, const runtime::AudioBuffer & audio) {
    Sha256 key;
    cache_key_field(key, identity);
    cache_key_field(key, std::to_string(audio.sample_rate));
    cache_key_field(key, std::to_string(audio.channels));
    cache_key_field(key, cache_bytes(audio.samples.data(), audio.samples.size()));
    return "fish_drama3/reference/v1/" + key.hex();
}

codecs::FishDacCodes decode_reference_payload(const std::vector<std::byte> & payload, int64_t codebook_size) {
    // The native frame count precedes the codebook-major int32 grid.
    uint64_t frames = 0;
    if (payload.size() < sizeof(frames)) return {};
    std::memcpy(&frames, payload.data(), sizeof(frames));
    if (frames == 0 || frames > kMaxDiskFrames ||
        payload.size() != sizeof(frames) + frames * kCodebooks * sizeof(int32_t)) return {};
    std::vector<int32_t> codes(static_cast<size_t>(frames) * kCodebooks);
    std::memcpy(codes.data(), payload.data() + sizeof(frames), codes.size() * sizeof(int32_t));
    if (std::any_of(codes.begin(), codes.end(), [codebook_size](int32_t code) {
            return code < 0 || code >= codebook_size;
        })) return {};
    return {std::move(codes), kCodebooks, static_cast<int64_t>(frames)};
}

}  // namespace

bool Drama3ReferenceCache::SameWaveform::operator()(const runtime::AudioBuffer & a,
                                                    const runtime::AudioBuffer & b) const {
    return a.sample_rate == b.sample_rate && a.channels == b.channels && a.samples.size() == b.samples.size() &&
           (a.samples.empty() ||
            std::memcmp(a.samples.data(), b.samples.data(), a.samples.size() * sizeof(float)) == 0);
}

Drama3ReferenceCache::Drama3ReferenceCache(std::size_t slots)
    : slots_(slots) {}

void Drama3ReferenceCache::set_disk_cache(std::filesystem::path directory, std::string identity, int64_t codebook_size) {
    if (directory.empty()) {
        disk_cache_.reset();
    } else {
        disk_cache_.emplace(io::DiskCacheConfig{directory / "fish_drama3-reference",
            kDiskReferenceEntries, kDiskReferenceBytes});
    }
    disk_identity_ = std::move(identity);
    codebook_size_ = codebook_size;
    disk_read_warned_ = false;
    disk_write_warned_ = false;
}

codecs::FishDacCodes Drama3ReferenceCache::lookup_or_encode(const runtime::AudioBuffer & audio,
                                                             const Encoder & encode, Outcome * outcome) {
    const bool storable = slots_.capacity() > 0 && audio.samples.size() <= max_cached_samples;
    if (storable) {
        if (const auto * codes = slots_.find(audio)) {
            if (outcome) *outcome = Outcome::Hit;
            return *codes;
        }
    }
    if (disk_cache_ && !audio.samples.empty() && audio.samples.size() <= max_cached_samples) {
        return lookup_or_encode_disk(audio, encode, outcome, storable);
    }
    auto codes = encode(audio);
    if (storable) slots_.put(audio, codes);
    if (outcome) *outcome = audio.samples.size() > max_cached_samples ? Outcome::Skip : Outcome::Miss;
    return codes;
}

codecs::FishDacCodes Drama3ReferenceCache::lookup_or_encode_disk(const runtime::AudioBuffer & audio,
    const Encoder & encode, Outcome * outcome, bool storable) {
    const auto started = std::chrono::steady_clock::now();
    const auto key = reference_key(disk_identity_, audio);
    auto disk_started = std::chrono::steady_clock::now();
    codecs::FishDacCodes codes;
    try {
        if (auto payload = disk_cache_->get(key)) codes = decode_reference_payload(*payload, codebook_size_);
    } catch (const std::exception & error) {
        if (!std::exchange(disk_read_warned_, true))
            debug::log_message(debug::LogLevel::Warning, "fish_drama3.reference_cache",
                "Cache read failed, encoding the reference: " + std::string(error.what()));
    }
    debug::timing_log_scalar("drama3.reference_disk_read_ms", debug::elapsed_ms(disk_started));
    const bool hit = codes.frames > 0;
    if (!hit) codes = encode(audio);
    if (storable) slots_.put(audio, codes);
    if (!hit && codes.codebooks == kCodebooks && codes.frames > 0 && codes.frames <= kMaxDiskFrames &&
        codes.codes.size() == kCodebooks * static_cast<size_t>(codes.frames)) {
        disk_started = std::chrono::steady_clock::now();
        try {
            const auto frames = static_cast<uint64_t>(codes.frames);
            const auto bytes = codes.codes.size() * sizeof(int32_t);
            std::vector<std::byte> payload(sizeof(frames) + bytes);
            std::memcpy(payload.data(), &frames, sizeof(frames));
            std::memcpy(payload.data() + sizeof(frames), codes.codes.data(), bytes);
            disk_cache_->put(key, payload);
        } catch (const std::exception & error) {
            if (!std::exchange(disk_write_warned_, true))
                debug::log_message(debug::LogLevel::Warning, "fish_drama3.reference_cache",
                    "Cache write failed, using the encoded reference: " + std::string(error.what()));
        }
        debug::timing_log_scalar("drama3.reference_disk_write_ms", debug::elapsed_ms(disk_started));
    }
    if (outcome) *outcome = hit ? Outcome::DiskHit : Outcome::Miss;
    debug::timing_log_scalar("drama3.conditioning_codec_ms", debug::elapsed_ms(started));
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
