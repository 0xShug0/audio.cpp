#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vieneu_v3_turbo {

enum class VieNeuTTSVariant {
    Base,
};

// Defaults follow `Vieneu.infer()` in the Python package (v3 Turbo): one sampler
// setting drives all 16 codebooks of the acoustic decoder, and the repetition
// penalty only looks at the last `repetition_window` frames of each codebook.
struct VieNeuTTSGenerationOptions {
    int64_t max_new_tokens = 300;
    bool do_sample = true;
    bool subtalker_do_sample = true;
    float temperature = 0.8F;
    int top_k = 25;
    float top_p = 0.95F;
    float repetition_penalty = 1.2F;
    int64_t repetition_window = 64;
    // Cap max_new_tokens by the phoneme count of the chunk (Python `frame_cap`).
    bool frame_cap = true;
    /// Re-generations allowed for a chunk the babble guard flags; 0 disables it.
    /// Needs sampling — a greedy retry returns the same frames.
    int64_t babble_retries = 2;
    // Acoustic-decoder overrides; when left negative they follow the main values.
    float subtalker_temperature = -1.0F;
    int subtalker_top_k = -1;
    float subtalker_top_p = -1.0F;
    uint32_t seed = 1234;
};

enum class Qwen3VoiceCloneMode {
    Icl,
    SpeakerEmbeddingOnly,
};

struct Qwen3SpeechCodes {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t code_groups = 0;
};

struct Qwen3VoiceCloneInput {
    runtime::AudioBuffer reference_audio;
    std::string reference_text;
    Qwen3VoiceCloneMode mode = Qwen3VoiceCloneMode::Icl;
    std::optional<std::vector<float>> speaker_embedding = std::nullopt;
    // Pre-encoded reference codes (frames x code_groups, row-major). When present they
    // replace the codec encoder pass, which lets packaged voices (codes + speaker
    // embedding) run without any reference audio.
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
};

struct VieNeuTTSRequest {
    std::string text;
    std::string language = "Auto";
    std::optional<Qwen3VoiceCloneInput> voice_clone = std::nullopt;
    VieNeuTTSGenerationOptions generation;
};

struct VieNeuTTSResult {
    runtime::AudioBuffer audio;
    std::vector<int32_t> codec_codes;
};

struct VieNeuSpeakerEmbedding {
    std::vector<float> values;
    int64_t dims = 0;
};

}  // namespace engine::models::vieneu_v3_turbo
