#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::ace_step {

constexpr int kAceStepAudioSampleRate = 48000;
constexpr int kAceStepAudioChannels = 2;

enum class AceStepTaskType {
    TextToMusic,
    Cover,
    CoverNoFsq,
    Repaint,
    Extract,
    Lego,
    Complete,
};

struct AceStepGenerationOptions {
    float duration_seconds = -1.0F;
    bool thinking = true;
    bool use_cot_metas = true;
    bool use_cot_caption = true;
    bool use_cot_language = true;
    int64_t num_inference_steps = 8;
    float guidance_scale = 1.0F;
    bool use_adg = false;
    float cfg_interval_start = 0.0F;
    float cfg_interval_end = 1.0F;
    float lm_temperature = 0.85F;
    float lm_cfg_scale = 2.0F;
    int64_t lm_top_k = 0;
    float lm_top_p = 0.9F;
    float lm_repetition_penalty = 1.0F;
    uint32_t seed = 1234;
    float shift = 1.0F;
    std::string infer_method = "ode";
    std::vector<float> timesteps;
    float audio_cover_strength = 1.0F;
    float cover_noise_strength = 0.0F;
    std::optional<uint32_t> retake_seed = std::nullopt;
    float retake_variance = 0.0F;
    std::string sampler_mode = "euler";
    float velocity_norm_threshold = 0.0F;
    float velocity_ema_factor = 0.0F;
    bool dcw_enabled = false;
    std::string dcw_mode = "double";
    float dcw_scaler = 0.05F;
    float dcw_high_scaler = 0.02F;
    std::string dcw_wavelet = "haar";
    int64_t repaint_crossfade_frames = 10;
    float repaint_injection_ratio = 0.5F;
    std::string noise_file;
    bool flow_edit_morph = false;
};

struct AceStepReferenceCondition {
    std::optional<runtime::AudioBuffer> audio = std::nullopt;
    std::optional<runtime::Transcript> lyrics = std::nullopt;
};

struct AceStepRequest {
    AceStepTaskType task = AceStepTaskType::TextToMusic;
    std::string task_type = "text2music";
    std::string instruction;
    std::string prompt;
    std::string lyrics;
    std::string audio_codes_text;
    std::vector<int32_t> audio_code_ids;
    std::string vocal_language = "en";
    std::string track_name;
    std::vector<std::string> complete_track_classes;
    std::optional<int64_t> bpm = std::nullopt;
    std::optional<std::string> keyscale = std::nullopt;
    std::optional<std::string> timesignature = std::nullopt;
    std::string negative_prompt = "NO USER INPUT";
    std::string chunk_mask_mode;
    std::optional<float> repainting_start_seconds = std::nullopt;
    std::optional<float> repainting_end_seconds = std::nullopt;
    std::optional<runtime::AudioBuffer> source_audio = std::nullopt;
    std::optional<AceStepReferenceCondition> reference = std::nullopt;
    AceStepGenerationOptions generation;
};

struct AceStepMetadata {
    std::optional<int64_t> bpm = std::nullopt;
    std::optional<int64_t> duration = std::nullopt;
    std::optional<std::string> keyscale = std::nullopt;
    std::optional<std::string> timesignature = std::nullopt;
    std::optional<std::string> language = std::nullopt;
    std::optional<std::string> genres = std::nullopt;
};

struct AceStepPlan {
    std::string caption;
    std::string cot_caption;
    AceStepMetadata metadata;
    std::string audio_codes_text;
    std::vector<int32_t> audio_code_ids;
    int64_t frames_5hz = 0;
};

struct AceStepTokenizedText {
    std::string text;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
};

struct AceStepTextConditioning {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct AceStepEncoderConditioning {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct AceStepLatents {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

struct AceStepPreDitInputs {
    AceStepPlan plan;
    AceStepTokenizedText text_prompt;
    AceStepTokenizedText non_cover_text_prompt;
    AceStepTokenizedText lyrics_prompt;
    AceStepTextConditioning text_hidden_states;
    AceStepTextConditioning non_cover_text_hidden_states;
    AceStepTextConditioning lyric_token_embeddings;
    AceStepLatents lm_hints_25hz;
    AceStepLatents target_latents;
    AceStepLatents src_latents;
    AceStepLatents chunk_mask;
    std::vector<int32_t> latent_attention_mask;
    std::vector<int32_t> repaint_mask;
    bool is_cover = false;
    AceStepEncoderConditioning encoder_hidden_states;
    AceStepEncoderConditioning encoder_hidden_states_non_cover;
    AceStepLatents context_latents;
    AceStepLatents context_latents_non_cover;
    std::optional<runtime::AudioBuffer> repaint_splice_audio = std::nullopt;
    float repaint_splice_start_seconds = 0.0F;
    float repaint_splice_end_seconds = 0.0F;
};

struct AceStepDiffusionConditioning {
    AceStepRequest request;
    AceStepPreDitInputs pre_dit;
};

struct AceStepResult {
    runtime::AudioBuffer audio;
    std::optional<AceStepPlan> plan = std::nullopt;
};

}  // namespace engine::models::ace_step
