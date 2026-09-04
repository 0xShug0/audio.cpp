#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/chatterbox/s3gen_inference.h"
#include "engine/community_models/chatterbox_turbo/assets.h"
#include "engine/community_models/chatterbox_turbo/s3gen_turbo.h"
#include "engine/community_models/chatterbox_turbo/t3_turbo_component.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::chatterbox_turbo {

struct ChatterboxTurboGenerateConfig {
    float temperature = 0.8f;
    float top_p = 0.95f;
    int64_t top_k = 1000;
    float repetition_penalty = 1.2f;
    int64_t max_new_tokens = 1000;
    uint32_t seed = 0;
    // exaggeration/cfg_weight/min_p from the base Chatterbox request surface are accepted but
    // ignored for Turbo (matches upstream tts_turbo.py's logger.warning(...ignored...)); no
    // fields here since this MVP path only drives the built-in voice (see ChatterboxTurboAssets).
};

// Orchestrates the full Turbo TTS pipeline (T3 GPT2 backbone -> S3Gen meanflow decoder -> HiFT
// vocoder) using only the built-in default voice baked into the T3 GGUF's `conds.*` tensors.
class ChatterboxTurboTtsComponent {
public:
    ChatterboxTurboTtsComponent(
        std::shared_ptr<const ChatterboxTurboAssets> assets,
        const engine::core::ExecutionContext & execution_context);

    engine::models::chatterbox::S3GenInferenceOutputs generate(
        const std::string & text,
        const ChatterboxTurboGenerateConfig & config) const;

private:
    std::shared_ptr<const ChatterboxTurboAssets> assets_;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    std::unique_ptr<T3TurboInferenceComponent> t3_;
    std::shared_ptr<ChatterboxTurboS3Gen> s3gen_;
    std::vector<float> builtin_speaker_embedding_;
    std::vector<int32_t> builtin_cond_prompt_speech_tokens_;
    engine::models::chatterbox::EmbedReferenceOutputs builtin_ref_dict_;
};

}  // namespace engine::community_models::chatterbox_turbo
