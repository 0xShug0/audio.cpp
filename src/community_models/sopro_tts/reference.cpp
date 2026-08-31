#include "engine/community_models/sopro_tts/reference.h"

#include "engine/community_models/sopro_tts/semantic_encoder.h"
#include "engine/community_models/sopro_tts/speaker_encoder.h"
#include "engine/community_models/sopro_tts/vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {
namespace audio_ops {
namespace {

constexpr float kRefGainLimitDb = 30.0F;
constexpr float kRefPeakCeiling = 0.95F;
constexpr float kPauseMinSeconds = 0.10F;
constexpr float kPauseKeepSeconds = 0.15F;
constexpr float kCropForwardSeconds = 5.0F;
constexpr float kCropBackwardSeconds = 5.0F;
constexpr float kMinKeepFraction = 0.75F;
constexpr float kRoomToneSeconds = 0.25F;
constexpr float kFadeSeconds = 0.02F;
constexpr float kOnsetThresholdDb = -45.0F;
constexpr float kOnsetOverFloorDb = 15.0F;
constexpr int64_t kOnsetWindowFrames = 6;
constexpr int64_t kOnsetMinFrames = 5;
constexpr float kMinActiveSeconds = 0.4F;

// torch.unfold + RMS: floor((n - win) / hop) + 1 frames, floored at 1e-6.
std::vector<float> frame_rms(
    const std::vector<float> & wav, int64_t window, int64_t hop) {
    const auto total = static_cast<int64_t>(wav.size());
    std::vector<float> out;
    if (window <= 0 || hop <= 0 || total < window) {
        return out;
    }
    const int64_t frames = (total - window) / hop + 1;
    out.resize(static_cast<size_t>(frames), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * values = wav.data() + static_cast<size_t>(frame * hop);
        double sum = 0.0;
        for (int64_t i = 0; i < window; ++i) {
            sum += static_cast<double>(values[i]) * static_cast<double>(values[i]);
        }
        out[static_cast<size_t>(frame)] = std::max(
            static_cast<float>(std::sqrt(sum / static_cast<double>(window))), 1.0e-6F);
    }
    return out;
}

// torch.quantile's default "linear" interpolation.
float quantile(std::vector<float> values, float q) {
    if (values.empty()) {
        return 0.0F;
    }
    std::sort(values.begin(), values.end());
    const double position = static_cast<double>(q) * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(position));
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return static_cast<float>(values[lower] * (1.0 - weight) + values[upper] * weight);
}

// torch.median returns the lower of the two middle elements for even counts.
float lower_median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    const size_t index = (values.size() - 1) / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<ptrdiff_t>(index), values.end());
    return values[index];
}

struct PauseRun {
    int64_t begin = 0;
    int64_t end = 0;
};

std::vector<PauseRun> pause_runs(const std::vector<char> & quiet, int64_t min_run) {
    std::vector<PauseRun> runs;
    const auto count = static_cast<int64_t>(quiet.size());
    int64_t index = 0;
    while (index < count) {
        if (quiet[static_cast<size_t>(index)] == 0) {
            ++index;
            continue;
        }
        int64_t end = index;
        while (end < count && quiet[static_cast<size_t>(end)] != 0) {
            ++end;
        }
        if (end - index >= min_run) {
            runs.push_back({index, end});
        }
        index = end;
    }
    return runs;
}

float onset_threshold(const std::vector<float> & rms) {
    float threshold = std::pow(10.0F, kOnsetThresholdDb / 20.0F);
    if (rms.size() >= 30) {
        threshold = std::max(
            threshold, quantile(rms, 0.1F) * std::pow(10.0F, kOnsetOverFloorDb / 20.0F));
    }
    return threshold;
}

// x[:(n // win) * win].view(-1, win) RMS, i.e. non-overlapping windows.
std::vector<float> block_rms(const std::vector<float> & wav, int64_t window) {
    std::vector<float> out;
    if (window <= 0) {
        return out;
    }
    const int64_t frames = static_cast<int64_t>(wav.size()) / window;
    out.resize(static_cast<size_t>(frames), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * values = wav.data() + static_cast<size_t>(frame * window);
        double sum = 0.0;
        for (int64_t i = 0; i < window; ++i) {
            sum += static_cast<double>(values[i]) * static_cast<double>(values[i]);
        }
        out[static_cast<size_t>(frame)] = static_cast<float>(std::sqrt(sum / static_cast<double>(window)));
    }
    return out;
}

// First index where `min_hits` of the next `window` frames are above threshold.
std::optional<int64_t> first_sustained_hit(
    const std::vector<float> & rms, float threshold, int64_t window, int64_t min_hits) {
    const auto count = static_cast<int64_t>(rms.size());
    if (count < window) {
        return std::nullopt;
    }
    for (int64_t start = 0; start + window <= count; ++start) {
        int64_t hits = 0;
        for (int64_t i = 0; i < window; ++i) {
            if (rms[static_cast<size_t>(start + i)] > threshold) {
                ++hits;
            }
        }
        if (hits >= min_hits) {
            return start;
        }
    }
    return std::nullopt;
}

std::vector<float> finish_with_room_tone(
    const std::vector<float> & wav, int sample_rate, float floor_level, std::mt19937_64 & rng) {
    const auto fade = static_cast<int64_t>(kFadeSeconds * static_cast<float>(sample_rate));
    std::vector<float> out(wav);
    if (fade > 0 && static_cast<int64_t>(out.size()) >= fade) {
        const auto offset = static_cast<int64_t>(out.size()) - fade;
        for (int64_t i = 0; i < fade; ++i) {
            const float ramp = fade == 1 ? 1.0F
                                         : 1.0F - static_cast<float>(i) / static_cast<float>(fade - 1);
            out[static_cast<size_t>(offset + i)] *= ramp;
        }
    }
    const auto tone_samples = static_cast<int64_t>(kRoomToneSeconds * static_cast<float>(sample_rate));
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (int64_t i = 0; i < tone_samples; ++i) {
        out.push_back(normal(rng) * floor_level);
    }
    return out;
}

}  // namespace

std::vector<float> crop_on_pause(
    const std::vector<float> & wav, float target_seconds, int sample_rate, std::mt19937_64 & rng) {
    const auto rate = static_cast<float>(sample_rate);
    const auto window = static_cast<int64_t>(rate * 0.025F);
    const auto hop = static_cast<int64_t>(rate * 0.010F);
    const auto total = static_cast<int64_t>(wav.size());
    if (window <= 0 || hop <= 0 || total < 4 * window) {
        return wav;
    }
    const auto rms = frame_rms(wav, window, hop);
    if (rms.empty()) {
        return wav;
    }
    const float floor_level = quantile(rms, 0.1F);
    std::vector<char> quiet(rms.size(), 0);
    for (size_t i = 0; i < rms.size(); ++i) {
        quiet[i] = rms[i] < floor_level * 4.0F ? 1 : 0;
    }
    const auto runs = pause_runs(
        quiet, std::max<int64_t>(1, std::lround(kPauseMinSeconds / 0.010F)));
    const auto keep = static_cast<int64_t>(std::lround(kPauseKeepSeconds / 0.010F));
    const auto target = static_cast<int64_t>(std::lround(target_seconds * rate));

    const auto cut_at = [&](const PauseRun & run) {
        const int64_t end = (run.begin + std::min(run.end - run.begin, keep)) * hop + window;
        return std::vector<float>(
            wav.begin(), wav.begin() + static_cast<ptrdiff_t>(std::min(end, total)));
    };

    if (total <= target) {
        if (!runs.empty() && runs.back().end * hop + window >= total - hop) {
            return wav;
        }
        const auto boundary = static_cast<int64_t>(static_cast<float>(total) * kMinKeepFraction);
        for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
            if (it->begin * hop >= boundary) {
                return cut_at(*it);
            }
        }
        return finish_with_room_tone(wav, sample_rate, floor_level, rng);
    }
    const auto forward_limit = target + static_cast<int64_t>(kCropForwardSeconds * rate);
    for (const auto & run : runs) {
        if (run.begin * hop >= target && run.begin * hop <= forward_limit) {
            return cut_at(run);
        }
    }
    const auto backward_limit = target - static_cast<int64_t>(kCropBackwardSeconds * rate);
    for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
        if (it->begin * hop < target && it->end * hop >= backward_limit) {
            return cut_at(*it);
        }
    }
    std::vector<float> truncated(
        wav.begin(), wav.begin() + static_cast<ptrdiff_t>(std::min(target, total)));
    return finish_with_room_tone(truncated, sample_rate, floor_level, rng);
}

SpeechLevel speech_level_db(const std::vector<float> & wav, int sample_rate) {
    const auto rate = static_cast<float>(sample_rate);
    const auto window = static_cast<int64_t>(rate * 0.025F);
    const auto hop = static_cast<int64_t>(rate * 0.010F);
    SpeechLevel out;
    if (window <= 0 || static_cast<int64_t>(wav.size()) < window) {
        double sum = 0.0;
        for (const float value : wav) {
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const double rms = wav.empty() ? 0.0 : std::sqrt(sum / static_cast<double>(wav.size()));
        out.level_db = 20.0F * std::log10(std::max(static_cast<float>(rms), 1.0e-6F));
        return out;
    }
    const auto rms = frame_rms(wav, window, hop);
    const float threshold = quantile(rms, 0.2F) * 1.5F;
    std::vector<float> active;
    active.reserve(rms.size());
    for (const float value : rms) {
        if (value > threshold) {
            active.push_back(value);
        }
    }
    if (active.empty()) {
        active = rms;
    }
    out.level_db = 20.0F * std::log10(lower_median(active));
    out.active_seconds = static_cast<float>(active.size()) * static_cast<float>(hop) / rate;
    return out;
}

NormalizedReference normalize_reference(const std::vector<float> & wav, int sample_rate) {
    const auto level = speech_level_db(wav, sample_rate);
    // Boost only: a reference that is already hotter than the prompt level is
    // passed through untouched instead of being pulled down.
    float gain_db = std::min(std::max(kPromptLevelDb - level.level_db, 0.0F), kRefGainLimitDb);
    float peak = 0.0F;
    for (const float value : wav) {
        peak = std::max(peak, std::fabs(value));
    }
    if (peak > 0.0F) {
        // Never let the boost clip: cap it at the headroom left below 0.95.
        gain_db = std::min(gain_db, std::max(0.0F, 20.0F * std::log10(kRefPeakCeiling / peak)));
    }
    const float gain = std::pow(10.0F, gain_db / 20.0F);
    NormalizedReference out;
    out.wav = wav;
    for (auto & value : out.wav) {
        value *= gain;
    }
    out.level_db = level.level_db + gain_db;
    return out;
}

float output_gain(float prompt_level_db) {
    return std::pow(10.0F, (kOutputLevelDb - prompt_level_db) / 20.0F);
}

float match_gain(
    const std::vector<float> & wav, int sample_rate, float target_db, float prompt_level_db) {
    const auto level = speech_level_db(wav, sample_rate);
    if (level.active_seconds < kMinActiveSeconds) {
        return output_gain(prompt_level_db);
    }
    return std::pow(10.0F, (target_db - level.level_db) / 20.0F);
}

void soft_limit(std::vector<float> & wav, float knee) {
    const float span = 1.0F - knee;
    if (span <= 0.0F) {
        return;
    }
    for (auto & value : wav) {
        const float magnitude = std::fabs(value);
        if (magnitude > knee) {
            const float limited = knee + span * std::tanh((magnitude - knee) / span);
            value = value < 0.0F ? -limited : limited;
        }
    }
}

std::optional<int64_t> speech_onset(const std::vector<float> & wav, int sample_rate) {
    const auto window = static_cast<int64_t>(static_cast<float>(sample_rate) * 0.010F);
    if (window <= 0 || static_cast<int64_t>(wav.size()) < window * kOnsetWindowFrames) {
        return std::nullopt;
    }
    const auto rms = block_rms(wav, window);
    const auto hit = first_sustained_hit(
        rms, onset_threshold(rms), kOnsetWindowFrames, kOnsetMinFrames);
    if (!hit.has_value()) {
        return std::nullopt;
    }
    return *hit * window;
}

std::vector<float> trim_lead(
    const std::vector<float> & wav, int sample_rate, float lead, float skip) {
    const auto onset = speech_onset(wav, sample_rate);
    if (!onset.has_value()) {
        return wav;
    }
    const auto rate = static_cast<float>(sample_rate);
    int64_t cut = std::max(*onset - static_cast<int64_t>(lead * rate),
                           static_cast<int64_t>(skip * rate));
    cut = std::min(cut, std::max<int64_t>(0, *onset - static_cast<int64_t>(0.02F * rate)));
    cut = std::min(cut, static_cast<int64_t>(wav.size()));
    return std::vector<float>(wav.begin() + static_cast<ptrdiff_t>(cut), wav.end());
}

std::vector<float> trim_trail(const std::vector<float> & wav, int sample_rate, float trail) {
    const auto window = static_cast<int64_t>(static_cast<float>(sample_rate) * 0.010F);
    if (window <= 0 || static_cast<int64_t>(wav.size()) < window) {
        return wav;
    }
    const auto rms = block_rms(wav, window);
    const float threshold = onset_threshold(rms);
    int64_t last = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(rms.size()); ++i) {
        if (rms[static_cast<size_t>(i)] > threshold) {
            last = i;
        }
    }
    if (last < 0) {
        return wav;
    }
    const int64_t end = std::min<int64_t>(
        static_cast<int64_t>(wav.size()),
        (last + 1) * window + static_cast<int64_t>(trail * static_cast<float>(sample_rate)));
    return std::vector<float>(wav.begin(), wav.begin() + static_cast<ptrdiff_t>(end));
}

void fade_edges(
    std::vector<float> & wav, int sample_rate, bool fade_in, bool fade_out, float fade_seconds) {
    const auto fade = static_cast<int64_t>(fade_seconds * static_cast<float>(sample_rate));
    if (fade <= 1 || static_cast<int64_t>(wav.size()) <= 2 * fade) {
        return;
    }
    for (int64_t i = 0; i < fade; ++i) {
        const float ramp = static_cast<float>(i) / static_cast<float>(fade - 1);
        if (fade_in) {
            wav[static_cast<size_t>(i)] *= ramp;
        }
        if (fade_out) {
            wav[wav.size() - static_cast<size_t>(fade) + static_cast<size_t>(i)] *= 1.0F - ramp;
        }
    }
}

std::vector<float> join_segments(std::vector<std::vector<float>> parts, int sample_rate) {
    std::vector<float> out;
    const auto count = static_cast<int64_t>(parts.size());
    for (int64_t i = 0; i < count; ++i) {
        fade_edges(parts[static_cast<size_t>(i)], sample_rate, i > 0, i + 1 < count);
        out.insert(
            out.end(),
            parts[static_cast<size_t>(i)].begin(),
            parts[static_cast<size_t>(i)].end());
    }
    return out;
}

}  // namespace audio_ops

SoproReferenceBuilder::SoproReferenceBuilder(
    const SoproTTSAssets & assets,
    const SoproSpeakerEncoderRuntime & speaker_encoder,
    const SoproSemanticEncoderRuntime & semantic_encoder,
    const SoproVocoderRuntime & vocoder)
    : config_(assets.config),
      speaker_encoder_(speaker_encoder),
      semantic_encoder_(semantic_encoder),
      vocoder_(vocoder),
      mel_mean_(assets.config.model.acoustic_mel_mean),
      mel_std_(assets.config.model.acoustic_mel_std) {
    const auto & source = *assets.model_weights;
    const int64_t in_dim = config_.model.cond_in_dim;
    const int64_t hidden = config_.model.cond_hidden_dim;
    cond_proj_w0 = source.require_f32("cond_proj.0.weight", {hidden, in_dim});
    cond_proj_b0 = source.require_f32("cond_proj.0.bias", {hidden});
    cond_proj_w3 = source.require_f32("cond_proj.3.weight", {hidden, hidden});
    cond_proj_b3 = source.require_f32("cond_proj.3.bias", {hidden});
}

SoproReference SoproReferenceBuilder::build(
    const std::vector<float> & audio24,
    float ref_seconds,
    std::mt19937_64 & rng) const {
    if (audio24.empty()) {
        throw std::runtime_error("Sopro requires non-empty reference audio");
    }
    const auto sample_rate = static_cast<int>(config_.sample_rate);
    auto cropped = audio_ops::crop_on_pause(audio24, ref_seconds, sample_rate, rng);
    auto normalized = audio_ops::normalize_reference(cropped, sample_rate);
    const float reference_level_db = normalized.level_db;
    auto wav = std::move(normalized.wav);

    const auto speaker_rate = static_cast<int>(config_.speaker_encoder.sample_rate);
    const auto wav16 = engine::audio::resample_mono_torchaudio_sinc_hann(
        wav, sample_rate, speaker_rate);
    const auto embeddings = speaker_encoder_.encode(wav16);

    const int64_t in_dim = config_.model.cond_in_dim;
    const int64_t hidden = config_.model.cond_hidden_dim;
    std::vector<float> conditioning;
    conditioning.reserve(static_cast<size_t>(in_dim));
    conditioning.insert(conditioning.end(), embeddings.id_emb.begin(), embeddings.id_emb.end());
    conditioning.insert(conditioning.end(), embeddings.style_emb.begin(), embeddings.style_emb.end());
    conditioning.insert(conditioning.end(), embeddings.style_ctrl.begin(), embeddings.style_ctrl.end());
    if (static_cast<int64_t>(conditioning.size()) != in_dim) {
        throw std::runtime_error(
            "Sopro speaker embeddings do not add up to model.cond_in_dim; check config.json");
    }
    std::vector<float> projected(static_cast<size_t>(hidden), 0.0F);
    for (int64_t o = 0; o < hidden; ++o) {
        const float * row = cond_proj_w0.data() + static_cast<size_t>(o * in_dim);
        double sum = cond_proj_b0[static_cast<size_t>(o)];
        for (int64_t i = 0; i < in_dim; ++i) {
            sum += static_cast<double>(row[i]) * static_cast<double>(conditioning[static_cast<size_t>(i)]);
        }
        const auto value = static_cast<float>(sum);
        projected[static_cast<size_t>(o)] = value / (1.0F + std::exp(-value));  // SiLU
    }
    SoproReference out;
    out.cond_vec.assign(static_cast<size_t>(hidden), 0.0F);
    for (int64_t o = 0; o < hidden; ++o) {
        const float * row = cond_proj_w3.data() + static_cast<size_t>(o * hidden);
        double sum = cond_proj_b3[static_cast<size_t>(o)];
        for (int64_t i = 0; i < hidden; ++i) {
            sum += static_cast<double>(row[i]) * static_cast<double>(projected[static_cast<size_t>(i)]);
        }
        out.cond_vec[static_cast<size_t>(o)] = static_cast<float>(sum);
    }

    out.semantic_tokens = semantic_encoder_.encode(wav);

    auto mel = vocoder_.log_mel(wav);
    const int64_t n_mels = vocoder_.n_mels();
    if (n_mels <= 0 || mel.size() % static_cast<size_t>(n_mels) != 0) {
        throw std::runtime_error("Sopro reference mel has an unexpected shape");
    }
    out.mel_frames = static_cast<int64_t>(mel.size()) / n_mels;
    for (int64_t c = 0; c < n_mels; ++c) {
        const float mean = mel_mean_[static_cast<size_t>(c)];
        const float scale = mel_std_[static_cast<size_t>(c)];
        float * row = mel.data() + static_cast<size_t>(c * out.mel_frames);
        for (int64_t t = 0; t < out.mel_frames; ++t) {
            row[t] = (row[t] - mean) / scale;
        }
    }
    out.mel = std::move(mel);
    out.level_db = reference_level_db;
    return out;
}

}  // namespace engine::community_models::sopro_tts
