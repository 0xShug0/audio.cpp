#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vietneu_tts {

enum class VietneuTTSVariant {
    Base,
};

struct VietneuTTSGenerationOptions {
    int64_t max_new_tokens = 2048;
    bool do_sample = true;
    bool subtalker_do_sample = true;
    float temperature = 0.9F;
    int top_k = 50;
    float top_p = 1.0F;
    float repetition_penalty = 1.05F;
    float subtalker_temperature = 0.9F;
    int subtalker_top_k = 50;
    float subtalker_top_p = 1.0F;
    uint32_t seed = 1234;
};

enum class Qwen3VoiceCloneMode {
    Icl,
    SpeakerEmbeddingOnly,
};

struct Qwen3VoiceCloneInput {
    runtime::AudioBuffer reference_audio;
    std::string reference_text;
    Qwen3VoiceCloneMode mode = Qwen3VoiceCloneMode::Icl;
    std::optional<std::vector<float>> speaker_embedding = std::nullopt;
};

struct VietneuTTSRequest {
    std::string text;
    std::string language = "Auto";
    std::optional<Qwen3VoiceCloneInput> voice_clone = std::nullopt;
    VietneuTTSGenerationOptions generation;
};

struct VietneuTTSResult {
    runtime::AudioBuffer audio;
    std::vector<int32_t> codec_codes;
};

struct Qwen3SpeechCodes {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t code_groups = 0;
};

struct VietneuSpeakerEmbedding {
    std::vector<float> values;
    int64_t dims = 0;
};

}  // namespace engine::models::vietneu_tts
