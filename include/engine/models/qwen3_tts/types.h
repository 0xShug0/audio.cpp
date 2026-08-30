#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::qwen3_tts {

enum class Qwen3TTSVariant {
    Base,
    VoiceDesign,
    CustomVoice,
};

enum class Qwen3TTSPerfMode {
    Standard,
    FlashAttention,
};

// The speech tokenizer emits one codec frame per 1920 samples at 24 kHz
// (src/models/qwen3_tts/tokenizer_speech_encoder.cpp:45-46), so the talker
// produces 12.5 frames of audio per second.
inline constexpr double kQwen3TTSCodecFrameRateHz = 24000.0 / 1920.0;
// Codepoints of prose a speaker gets through in one second. 13 is a brisk
// English delivery (~155 wpm); languages that pack more speech into fewer
// characters are covered by the safety margin below.
inline constexpr double kQwen3TTSCodepointsPerSecond = 13.0;
// Headroom left for punctuation, prompt tokens, and slower deliveries. The
// chunk is sized so a typical chunk finishes well before the token cap rather
// than exactly at it.
inline constexpr double kQwen3TTSChunkSafetyMargin = 0.75;
inline constexpr int64_t kQwen3TTSMinTextChunkSize = 64;

// Largest text chunk the talker can be expected to finish within
// `max_new_tokens` codec frames.
//
// F6.2: the previous fixed default of 8192 codepoints was roughly five times
// what a 2048-frame cap can synthesise (2048 frames is 163.8 s of audio, about
// a fifth of what 8192 characters of prose needs), so long-form input was
// truncated mid-sentence with no diagnostic.
constexpr int64_t qwen3_tts_default_text_chunk_size(int64_t max_new_tokens) noexcept {
    if (max_new_tokens <= 0) {
        return kQwen3TTSMinTextChunkSize;
    }
    const double seconds = static_cast<double>(max_new_tokens) / kQwen3TTSCodecFrameRateHz;
    const double codepoints = seconds * kQwen3TTSCodepointsPerSecond * kQwen3TTSChunkSafetyMargin;
    const auto budget = static_cast<int64_t>(codepoints);
    return budget < kQwen3TTSMinTextChunkSize ? kQwen3TTSMinTextChunkSize : budget;
}

// Codec frames needed to speak `codepoints` characters at the same rate. The
// inverse of the function above, so the two can be checked against each other.
constexpr int64_t qwen3_tts_frames_for_codepoints(int64_t codepoints) noexcept {
    if (codepoints <= 0) {
        return 0;
    }
    const double seconds = static_cast<double>(codepoints) / kQwen3TTSCodepointsPerSecond;
    return static_cast<int64_t>(seconds * kQwen3TTSCodecFrameRateHz) + 1;
}

struct Qwen3TTSGenerationOptions {
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
};

struct Qwen3VoiceDesignInput {
    std::string instruct;
};

struct Qwen3CustomVoiceInput {
    std::string speaker;
    std::string instruct;
};

struct Qwen3TTSRequest {
    std::string text;
    std::string language = "Auto";
    std::optional<Qwen3VoiceCloneInput> voice_clone = std::nullopt;
    std::optional<Qwen3VoiceDesignInput> voice_design = std::nullopt;
    std::optional<Qwen3CustomVoiceInput> custom_voice = std::nullopt;
    Qwen3TTSGenerationOptions generation;
};

struct Qwen3TTSResult {
    runtime::AudioBuffer audio;
    std::vector<int32_t> codec_codes;
};

struct Qwen3SpeechCodes {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t code_groups = 0;
};

struct Qwen3SpeakerEmbedding {
    std::vector<float> values;
    int64_t dims = 0;
};

}  // namespace engine::models::qwen3_tts
