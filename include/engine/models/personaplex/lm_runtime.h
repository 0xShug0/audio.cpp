#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/models/personaplex/assets.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::personaplex {

constexpr size_t kPersonaPlexDelayedStreamCount = 17;

struct PersonaPlexLMWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::QwenCausalDecoderWeights main;
    core::TensorValue text_embedding;
    std::vector<core::TensorValue> audio_embeddings;
};

modules::QwenCausalDecoderConfig personaplex_lm_decoder_config(
    const PersonaPlexConfig & config,
    core::BackendType backend_type);

std::shared_ptr<const PersonaPlexLMWeights> load_personaplex_lm_weights(
    const PersonaPlexAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type);

struct PersonaPlexMainStepOutput {
    std::vector<float> hidden;
    std::vector<float> text_logits;
};

struct PersonaPlexDelayedStep {
    std::array<int32_t, kPersonaPlexDelayedStreamCount> tokens{};
    std::array<int32_t, kPersonaPlexDelayedStreamCount> target{};
    std::array<uint8_t, kPersonaPlexDelayedStreamCount> provided{};
    int64_t model_input_position = 0;
    int64_t target_position = 0;
};

class PersonaPlexMainStepGraph {
public:
    PersonaPlexMainStepGraph(
        std::shared_ptr<const PersonaPlexLMWeights> weights,
        PersonaPlexConfig config,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes);
    ~PersonaPlexMainStepGraph();

    PersonaPlexMainStepOutput run_embedding_step(const std::vector<float> & embedding);
    PersonaPlexMainStepOutput run_token_step(const std::array<int32_t, kPersonaPlexDelayedStreamCount> & tokens);
    void reset();
    int64_t valid_steps() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::personaplex
