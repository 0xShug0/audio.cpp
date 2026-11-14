#include "engine/community_models/zonos2/prompt.h"

#include "engine/framework/runtime/errors.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace engine::models::zonos2 {
namespace {

constexpr int kBos = 2;
constexpr int kEos = 3;
constexpr int kByteOffset = 192;  // 192 legacy symbol ids precede the 256 byte ids
constexpr int kSilenceRows = 17;

// Silence tokens for 0.2 s at 44.1 kHz, 17 frames x 9 codebooks (ZONOS2 prompt.py _SILENCE_TOKENS_0_2S).
constexpr int kSilence[kSilenceRows][9] = {
    {568, 778, 338, 524, 967, 360, 728, 550,  90},
    {568, 778,  10, 674, 364, 981, 741, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 804,  10, 674, 364, 981, 568, 378, 731},
    {568, 778, 721, 842, 264, 974, 989, 507, 308},
};

constexpr std::array<const char *, 6> kQualityFeatures = {
    "lufs", "estimated_snr", "max_pause", "estimated_bandlimit_hz", "leading_silence_s", "trailing_silence_s"};

int quality_total(const Zonos2Hparams & hp) {
    int total = 0;
    for (const int count : hp.quality_bucket_counts) total += count;
    return total;
}

// First conditioning id: [speaking rate][quality per feature][speaker background][accurate] end at text_vocab.
int conditioning_base(const Zonos2Hparams & hp) {
    return hp.text_vocab - hp.speaking_rate_buckets - quality_total(hp) - hp.speaker_bg_buckets - hp.accurate_buckets;
}

std::string trim(const std::string & s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

}  // namespace

Zonos2Prompt build_zonos2_prompt(const Zonos2Hparams & hp, const std::string & utf8, const Zonos2PromptOptions & options) {
    const int ncb = hp.n_codebooks;
    const size_t width = static_cast<size_t>(ncb) + 1;
    if (options.speaking_rate_bucket < -1 || options.speaking_rate_bucket >= hp.speaking_rate_buckets) {
        throw runtime::InvalidRequestError("zonos2 speaking_rate_bucket must be 0.." +
                                           std::to_string(hp.speaking_rate_buckets - 1));
    }
    std::vector<int> quality = options.quality_buckets;
    if (quality.empty()) {
        quality.assign(hp.quality_bucket_counts.size(), -1);
        quality[static_cast<size_t>(hp.default_quality_feature)] = hp.default_quality_bucket;
    }
    if (quality.size() != hp.quality_bucket_counts.size()) {
        throw runtime::InvalidRequestError("zonos2 quality_buckets: expected " +
                                           std::to_string(hp.quality_bucket_counts.size()) + " features");
    }
    for (size_t f = 0; f < quality.size(); ++f) {
        if (quality[f] < -1 || quality[f] >= hp.quality_bucket_counts[f]) {
            throw runtime::InvalidRequestError(std::string("zonos2 quality_buckets: ") + kQualityFeatures[f] +
                                               " bucket must be 0.." + std::to_string(hp.quality_bucket_counts[f] - 1));
        }
    }

    const int base = conditioning_base(hp);
    const int quality_base = base + hp.speaking_rate_buckets;
    const int background_base = quality_base + quality_total(hp);
    Zonos2Prompt out;
    auto pad_row = [&](int text) {
        out.ids.insert(out.ids.end(), static_cast<size_t>(ncb), hp.audio_pad_id);
        out.ids.push_back(text);
    };
    if (options.speaker) {
        out.speaker_row = 0;
        pad_row(hp.text_vocab);
        if (hp.speaker_bg_buckets > 0) {
            pad_row(background_base + (options.clean_speaker_background ? 0 : 1));
            if (hp.accurate_buckets > 0 && options.accurate_mode) pad_row(background_base + hp.speaker_bg_buckets);
        }
    }
    if (options.speaking_rate_bucket >= 0) pad_row(base + options.speaking_rate_bucket);
    int prefix = 0;
    for (size_t f = 0; f < quality.size(); ++f) {
        if (quality[f] >= 0) pad_row(quality_base + prefix + quality[f]);
        prefix += hp.quality_bucket_counts[f];
    }
    pad_row(kBos);
    for (const unsigned char byte : utf8) pad_row(kByteOffset + static_cast<int>(byte));
    pad_row(kEos);
    for (int t = 0; t < kSilenceRows; ++t) {
        for (int c = 0; c < ncb; ++c) out.ids.push_back(t >= c ? kSilence[t - c][c] : hp.audio_pad_id);
        out.ids.push_back(hp.text_vocab);
    }
    out.n_rows = static_cast<int>(out.ids.size() / width);
    return out;
}

std::vector<int> parse_zonos2_quality_buckets(const Zonos2Hparams & hp, const std::string & spec) {
    const size_t features = hp.quality_bucket_counts.size();
    std::vector<int> out(features, -1);
    if (trim(spec) == "none") return out;
    size_t pos = 0;
    while (true) {
        const size_t comma = spec.find(',', pos);
        const std::string entry = trim(spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
        const size_t colon = entry.find(':');
        if (colon == std::string::npos) {
            throw runtime::InvalidRequestError("zonos2 quality_buckets: expected feature:bucket, got '" + entry + "'");
        }
        const std::string name = trim(entry.substr(0, colon));
        const std::string value = trim(entry.substr(colon + 1));
        size_t feature = features;
        for (size_t f = 0; f < features && f < kQualityFeatures.size(); ++f) {
            if (name == kQualityFeatures[f]) feature = f;
        }
        if (feature == features) {
            throw runtime::InvalidRequestError("zonos2 quality_buckets: unknown feature '" + name +
                                               "' (lufs, estimated_snr, max_pause, estimated_bandlimit_hz, "
                                               "leading_silence_s, trailing_silence_s)");
        }
        if (out[feature] >= 0) {
            throw runtime::InvalidRequestError("zonos2 quality_buckets: " + name + " is given twice");
        }
        const int count = hp.quality_bucket_counts[feature];
        const bool digits = !value.empty() && value.size() <= 3 && value.find_first_not_of("0123456789") == std::string::npos;
        const int bucket = digits ? std::stoi(value) : -1;
        if (bucket < 0 || bucket >= count) {
            throw runtime::InvalidRequestError("zonos2 quality_buckets: " + name + " bucket must be 0.." +
                                               std::to_string(count - 1) + ", got '" + value + "'");
        }
        out[feature] = bucket;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

}  // namespace engine::models::zonos2
