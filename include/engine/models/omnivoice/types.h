#pragma once

#include "engine/framework/text/chunking.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::omnivoice {

enum class OmniVoicePromptMode {
    AutoVoice,
    VoiceClone,
    VoiceDesign,
};

struct OmniVoiceGenerationOptions {
    std::optional<uint32_t> seed = std::nullopt;
    int64_t num_inference_steps = 32;
    float guidance_scale = 2.0F;
    float speed = 1.0F;
    std::optional<float> duration_seconds = std::nullopt;
    float t_shift = 0.1F;
    bool denoise = true;
    bool postprocess_output = true;
    float layer_penalty_factor = 5.0F;
    float position_temperature = 5.0F;
    float class_temperature = 0.0F;
    bool preprocess_prompt = true;
    // Longest stretch of reference speech the model is conditioned on. Past
    // roughly 15 s cloning quality collapses and generation slows sharply, so
    // the reference is cut back to the last pause inside this window. Zero
    // disables the limit.
    float reference_max_seconds = 15.0F;
    // Exact digital silence written onto both ends of a clamped reference.
    int64_t reference_pad_ms = 150;
    float audio_chunk_duration_seconds = 15.0F;
    float audio_chunk_threshold_seconds = 30.0F;
    std::optional<int64_t> text_chunk_size = std::nullopt;
    engine::text::TextChunkMode text_chunk_mode = engine::text::TextChunkMode::TagAware;
};

struct OmniVoiceAudioTokens {
    std::vector<int32_t> token_ids;
    int64_t frames = 0;
    int64_t codebooks = 0;
    float reference_rms = 0.0F;
    // Fraction of the caller's reference audio these tokens actually cover, 1.0
    // when nothing was dropped. The length clamp cuts the audio but cannot cut
    // the caller's transcript, and estimate_target_tokens derives the output
    // duration from the ref_text-to-ref_frames ratio -- so a clamped reference
    // paired with a full transcript collapses the implied speaking rate and
    // truncates the output. prompt_builder trims the transcript by this
    // fraction to keep the pair consistent.
    double retained_fraction = 1.0;
};

struct OmniVoiceRequest {
    std::string text;
    std::string language;
    std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
    std::optional<OmniVoiceAudioTokens> reference_audio_tokens = std::nullopt;
    std::string reference_text;
    std::string instruct;
    float reference_rms = 0.0F;
    OmniVoiceGenerationOptions generation;
};

struct OmniVoicePrompt {
    OmniVoicePromptMode mode = OmniVoicePromptMode::AutoVoice;
    std::vector<int32_t> style_token_ids;
    std::vector<int32_t> text_token_ids;
    std::string text;
    std::string language;
    std::string instruct;
    std::string reference_text;
    std::optional<OmniVoiceAudioTokens> reference_audio_tokens = std::nullopt;
    int64_t target_audio_tokens = 0;
    float reference_rms = 0.0F;
};

struct OmniVoiceGeneratedAudioTokens {
    std::vector<int32_t> token_ids;
    int64_t frames = 0;
    int64_t codebooks = 0;
    bool graph_rebuilt = false;
    int64_t graph_total_token_capacity = 0;
    int64_t graph_target_frame_capacity = 0;
    double forward_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    double scoring_ms = 0.0;
    double update_ms = 0.0;
    int64_t decode_steps = 0;
};

struct OmniVoiceResult {
    runtime::AudioBuffer audio;
};

}  // namespace engine::models::omnivoice
