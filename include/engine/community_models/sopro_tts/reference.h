#pragma once

#include "engine/community_models/sopro_tts/assets.h"

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace engine::community_models::sopro_tts {

class SoproSpeakerEncoderRuntime;
class SoproSemanticEncoderRuntime;
class SoproVocosRuntime;

// Ports of sopro/audio.py. All of them operate on mono float waveforms.
namespace audio_ops {

constexpr float kPromptLevelDb = -19.8F;
constexpr float kOutputLevelDb = -23.0F;
constexpr float kLimiterKnee = 0.9F;
constexpr float kLeadInSeconds = 0.08F;
constexpr float kSegmentLeadSeconds = 0.30F;
constexpr float kSegmentSkipSeconds = 0.10F;
constexpr float kTrailSeconds = 0.30F;
constexpr float kJoinFadeSeconds = 0.01F;
constexpr float kFinalFadeSeconds = 0.08F;
constexpr float kGateHoldSeconds = 2.0F;

struct SpeechLevel {
    float level_db = 0.0F;
    float active_seconds = 0.0F;
};

// normalize_reference returns the boosted waveform together with the speech
// level it ends up at, so the output gain can be derived from the reference
// the model actually heard.
struct NormalizedReference {
    std::vector<float> wav;
    float level_db = kPromptLevelDb;
};

std::vector<float> crop_on_pause(
    const std::vector<float> & wav, float target_seconds, int sample_rate, std::mt19937_64 & rng);
SpeechLevel speech_level_db(const std::vector<float> & wav, int sample_rate);
// Boost-only and peak-guarded: a reference already at or above the prompt level
// is left alone, and the boost never pushes the peak past 0.95.
NormalizedReference normalize_reference(const std::vector<float> & wav, int sample_rate);
// The output gain both synthesis modes apply: maps the prompt's speech level
// to kOutputLevelDb.
float output_gain(float prompt_level_db = kPromptLevelDb);
void soft_limit(std::vector<float> & wav, float knee = kLimiterKnee);
std::optional<int64_t> speech_onset(const std::vector<float> & wav, int sample_rate);
// First sample of `min_frames` consecutive 10 ms frames over the noise floor.
std::optional<int64_t> energy_onset(
    const std::vector<float> & wav, int sample_rate, float over_floor_db = 10.0F, int64_t min_frames = 3);
// Share of the 10 ms frames in [start, end) that are over the noise floor.
float energy_fraction(
    const std::vector<float> & wav, int sample_rate, int64_t start, int64_t end, float over_floor_db = 10.0F);
// Where the lead-in is cut for speech starting at `onset`: `lead` seconds before
// it, never before `skip` seconds, and at least 20 ms before the onset.
int64_t lead_cut(int64_t onset, int sample_rate, float lead, float skip);
std::vector<float> trim_lead(
    const std::vector<float> & wav, int sample_rate,
    float lead = kLeadInSeconds, float skip = 0.0F);
std::vector<float> trim_trail(
    const std::vector<float> & wav, int sample_rate, float trail = kTrailSeconds);
void fade_edges(
    std::vector<float> & wav, int sample_rate, bool fade_in, bool fade_out,
    float fade_seconds = kJoinFadeSeconds);
std::vector<float> join_segments(std::vector<std::vector<float>> parts, int sample_rate);

}  // namespace audio_ops

// The per-voice state the semantic LM and the acoustic head both condition on.
struct SoproReference {
    std::vector<float> cond_vec;           // [cond_hidden_dim]
    std::vector<int32_t> semantic_tokens;  // one id per 1024 reference samples
    std::vector<float> mel;                // [n_mels, mel_frames], normalised
    int64_t mel_frames = 0;
    // Speech level of the normalised reference, i.e. what the output gain is
    // derived from (Reference.level_db).
    float level_db = audio_ops::kPromptLevelDb;
};

// SoproTTS.prepare_reference: crop on a pause, level-normalise, then run the
// speaker encoder, the semantic encoder and the analysis mel in one pass.
class SoproReferenceBuilder final {
public:
    SoproReferenceBuilder(
        const SoproTTSAssets & assets,
        const SoproSpeakerEncoderRuntime & speaker_encoder,
        const SoproSemanticEncoderRuntime & semantic_encoder,
        const SoproVocosRuntime & vocoder);

    // audio24: mono 24 kHz reference waveform.
    SoproReference build(
        const std::vector<float> & audio24, float ref_seconds, std::mt19937_64 & rng) const;

private:
    const SoproTTSConfig & config_;
    const SoproSpeakerEncoderRuntime & speaker_encoder_;
    const SoproSemanticEncoderRuntime & semantic_encoder_;
    const SoproVocosRuntime & vocoder_;
    // SoproModel.cond_proj = Sequential(Linear, SiLU, Identity, Linear).
    std::vector<float> cond_proj_w0;
    std::vector<float> cond_proj_b0;
    std::vector<float> cond_proj_w3;
    std::vector<float> cond_proj_b3;
    std::vector<float> mel_mean_;
    std::vector<float> mel_std_;
};

}  // namespace engine::community_models::sopro_tts
