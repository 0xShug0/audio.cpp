#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/firered_audio/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::firered_audio {

struct FireRedAudioGenerationRequest {
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> audio_no_latent_mask;
    std::optional<engine::runtime::AudioBuffer> prompt_audio;
    bool prompt_audio_is_assistant = false;
    std::string generated_text_language = "zh";
    uint32_t seed = 1234;
    int64_t num_inference_steps = 10;
    float guidance_scale = 2.0F;
    int64_t max_new_audio_steps = 750;
    int64_t min_new_audio_steps = 6;
    int64_t max_new_text_tokens = 512;
};

struct FireRedAudioGenerationResult {
    engine::runtime::AudioBuffer audio;
    std::string generated_text;
};

class FireRedAudioGenerationRuntime {
public:
    FireRedAudioGenerationRuntime(
        std::shared_ptr<const FireRedAudioAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        size_t reference_cache_slots,
        bool mem_saver);
    ~FireRedAudioGenerationRuntime();

    FireRedAudioGenerationRuntime(const FireRedAudioGenerationRuntime &) = delete;
    FireRedAudioGenerationRuntime & operator=(const FireRedAudioGenerationRuntime &) = delete;

    FireRedAudioGenerationResult generate(const FireRedAudioGenerationRequest & request);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::firered_audio
