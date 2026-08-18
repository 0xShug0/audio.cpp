#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/models/personaplex/assets.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <memory>
#include <vector>

namespace engine::models::personaplex {

constexpr size_t kPersonaPlexDepformerAudioStreams = 16;

struct PersonaPlexDepformerLayerWeights {
    modules::NormWeights norm1;
    modules::AttentionWeights attention;
    modules::NormWeights norm2;
    std::vector<modules::LinearWeights> gate_up;
    std::vector<modules::LinearWeights> down;
};

struct PersonaPlexDepformerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<modules::LinearWeights> input_from_lm;
    core::TensorValue text_embedding;
    std::vector<core::TensorValue> audio_embeddings;
    std::vector<PersonaPlexDepformerLayerWeights> layers;
    std::vector<modules::LinearWeights> heads;
};

std::shared_ptr<const PersonaPlexDepformerWeights> load_personaplex_depformer_weights(
    const PersonaPlexAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type);

struct PersonaPlexDepformerSamplingOptions {
    bool do_sample = true;
    float temperature = 0.8F;
    int64_t top_k = 250;
};

struct PersonaPlexDepformerOutput {
    std::array<int32_t, kPersonaPlexDepformerAudioStreams> sampled_audio_tokens{};
};

class PersonaPlexDepformerRuntime {
public:
    PersonaPlexDepformerRuntime(
        std::shared_ptr<const PersonaPlexDepformerWeights> weights,
        PersonaPlexConfig config,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes);
    ~PersonaPlexDepformerRuntime();

    PersonaPlexDepformerOutput run(
        int32_t text_token,
        const std::vector<float> & transformer_hidden,
        const std::array<int32_t, kPersonaPlexDepformerAudioStreams> & audio_target,
        const std::array<uint8_t, kPersonaPlexDepformerAudioStreams> & audio_provided,
        const PersonaPlexDepformerSamplingOptions & sampling,
        engine::sampling::HfSamplerScratch & scratch,
        std::mt19937 & fallback_rng,
        const engine::sampling::HfTorchSamplingState * torch_state);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::personaplex
