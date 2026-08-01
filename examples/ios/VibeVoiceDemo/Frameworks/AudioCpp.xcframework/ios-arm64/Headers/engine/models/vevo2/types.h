#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vevo2 {

enum class Vevo2InferencePath {
    TextProsodyToTargetVoice,
    SourceAudioToTargetVoice,
};

enum class Vevo2RouteKind {
    ZeroShotTts,
    TextToSinging,
    Svs,
    StylePreservedVoiceConversion,
    StylePreservedSingingConversion,
    StyleConvertedVoiceConversion,
    StyleConvertedSingingConversion,
    Editing,
    SingingStyleConversion,
    MelodyControl,
};

struct Vevo2GenerationOptions {
    bool use_prosody_code = true;
    bool predict_target_prosody = false;
    int top_k = 25;
    float top_p = 0.8F;
    float temperature = 1.0F;
    float repetition_penalty = 1.1F;
    int max_new_tokens = 500;
    uint32_t seed = 1234;
    bool use_pitch_shift = false;
    int source_shift_steps = 0;
    int prosody_shift_steps = 0;
    int style_shift_steps = 0;
    std::optional<float> target_duration_seconds = std::nullopt;
    std::optional<float> reference_duration_seconds = std::nullopt;
    int num_inference_steps = 32;
    std::string fm_noise_file;
};

struct Vevo2ReferenceInputs {
    std::string target_text;
    std::string style_ref_text;
    std::optional<runtime::AudioBuffer> source_audio = std::nullopt;
    std::optional<runtime::AudioBuffer> prosody_audio = std::nullopt;
    std::optional<runtime::AudioBuffer> style_ref_audio = std::nullopt;
    runtime::AudioBuffer timbre_ref_audio;
};

struct Vevo2Request {
    Vevo2InferencePath path = Vevo2InferencePath::TextProsodyToTargetVoice;
    Vevo2RouteKind route = Vevo2RouteKind::ZeroShotTts;
    Vevo2ReferenceInputs refs;
    Vevo2GenerationOptions generation;
};

struct Vevo2PromptParts {
    std::string text_prompt;
    std::string prosody_prompt;
    std::string content_style_prompt;
    std::string full_prompt;
};

struct Vevo2TokenizedPrompt {
    std::string text;
    std::vector<int32_t> input_ids;
};

struct Vevo2TokenSequence {
    std::vector<int32_t> ids;
};

struct Vevo2MelSequence {
    int64_t frames = 0;
    int64_t mel_bins = 0;
    std::vector<float> values;
};

}  // namespace engine::models::vevo2
