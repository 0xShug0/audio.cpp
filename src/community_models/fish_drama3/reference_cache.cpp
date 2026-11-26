#include "engine/community_models/fish_drama3/reference_cache.h"
#include "disk_cache.h"

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
constexpr size_t kMaxDiskCodeCount = 4096 * kCodebooks;
constexpr size_t kDiskReferenceEntries = 64;

std::filesystem::path reference_path(const std::filesystem::path & directory, std::string_view identity,
                                     const runtime::AudioBuffer & audio) {
    Sha256 key;
    cache_key_field(key, identity);
    cache_key_field(key, std::to_string(audio.sample_rate));
    cache_key_field(key, std::to_string(audio.channels));
    cache_key_field(key, cache_bytes(audio.samples.data(), audio.samples.size()));
    return directory / (key.hex() + ".vq");
}

codecs::FishDacCodes read_reference_codes(const std::filesystem::path & path, int64_t codebook_size) {
    const auto bytes = cache_payload_size(path);
    if (bytes == 0 || bytes > kMaxDiskCodeCount * sizeof(int32_t) ||
        bytes % (kCodebooks * sizeof(int32_t)) != 0) return {};
    std::vector<int32_t> codes(bytes / sizeof(int32_t));
    if (!read_cache_file(path, {{codes.data(), bytes}}) ||
        std::any_of(codes.begin(), codes.end(), [codebook_size](int32_t code) {
            return code < 0 || code >= codebook_size;
        })) return {};
    const auto frames = static_cast<int64_t>(codes.size() / kCodebooks);
    return {std::move(codes), kCodebooks, frames};
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
    disk_directory_ = std::move(directory);
    disk_identity_ = std::move(identity);
    codebook_size_ = codebook_size;
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
    if (!disk_directory_.empty() && !audio.samples.empty() && audio.samples.size() <= max_cached_samples) {
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
    const auto disk_path = reference_path(disk_directory_, disk_identity_, audio);
    auto codes = read_reference_codes(disk_path, codebook_size_);
    const bool hit = codes.frames > 0;
    if (!hit) codes = encode(audio);
    if (storable) slots_.put(audio, codes);
    if (!hit && codes.codebooks == kCodebooks && codes.frames > 0 && codes.codes.size() <= kMaxDiskCodeCount &&
        codes.codes.size() == kCodebooks * static_cast<size_t>(codes.frames)) {
        write_cache_file(disk_path, {cache_bytes(codes.codes.data(), codes.codes.size())}, kDiskReferenceEntries);
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
