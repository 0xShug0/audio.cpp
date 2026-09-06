#pragma once

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/breeze_tts/assets.h"
#include "engine/models/breeze_tts/speech_decoder.h"
#include "engine/models/breeze_tts/text_encoder.h"
#include "engine/models/breeze_tts/tokenizer_text.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::breeze_tts {

struct BreezeGenerationRequest {
    std::string text;
    std::string instruction;
    std::string reference_text;
    std::optional<engine::runtime::AudioBuffer> reference_audio;
    std::optional<BreezeSpeechCodes> reference_codes;
    float guidance_scale = 1.0F;
    float temperature = 0.9F;
    float depth_temperature = 0.9F;
    int64_t top_k = 50;
    float top_p = 1.0F;
    int64_t max_tokens = 1500;
    uint64_t seed = 0;
};

struct BreezeStreamStep {
    std::vector<int32_t> new_codes;
    int64_t new_frames = 0;
    bool done = false;
};

class BreezeGeneratorRuntime {
public:
    BreezeGeneratorRuntime(
        std::shared_ptr<const BreezeTTSAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        engine::core::AttentionPreference attention_preference = engine::core::AttentionPreference::Auto);
    ~BreezeGeneratorRuntime();

    engine::runtime::AudioBuffer generate(const BreezeGenerationRequest & request);
    BreezeSpeechCodes encode_reference(const engine::runtime::AudioBuffer & audio) const;

    // Resumable sub-chunk streaming. begin_stream runs prompts + prefills;
    // step_stream runs up to max_new_frames AR frames; end_stream releases
    // graphs. Only one stream may be active at a time. decode_codes runs the
    // codec over a code prefix without releasing graphs (caller brackets with
    // begin/end_stream).
    void begin_stream(const BreezeGenerationRequest & request);
    BreezeStreamStep step_stream(size_t max_new_frames);
    void end_stream();
    engine::runtime::AudioBuffer decode_codes(const std::vector<int32_t> & codes, int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::breeze_tts
