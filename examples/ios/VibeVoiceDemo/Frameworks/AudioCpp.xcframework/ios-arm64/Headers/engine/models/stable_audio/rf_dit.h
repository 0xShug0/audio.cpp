#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/conditioner_runtime.h"
#include "engine/models/stable_audio/sampler.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::stable_audio {

struct StableAudioRfAttentionWeights {
    modules::LinearWeights to_qkv;
    modules::LinearWeights to_q;
    modules::LinearWeights to_kv;
    modules::LinearWeights to_out;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    bool differential = false;
};

struct StableAudioRfFeedForwardWeights {
    modules::LinearWeights in_proj;
    modules::LinearWeights out_proj;
};

struct StableAudioRfLayerWeights {
    core::TensorValue pre_norm_gamma;
    core::TensorValue cross_attend_norm_gamma;
    core::TensorValue ff_norm_gamma;
    StableAudioRfAttentionWeights self_attn;
    StableAudioRfAttentionWeights cross_attn;
    StableAudioRfFeedForwardWeights ff;
    modules::LinearWeights local_embed_in;
    modules::LinearWeights local_embed_out;
    core::TensorValue scale_shift_gate;
};

struct StableAudioRfDitWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights to_cond_embed_0;
    modules::LinearWeights to_cond_embed_2;
    modules::LinearWeights to_global_embed_0;
    modules::LinearWeights to_global_embed_2;
    modules::LinearWeights timestep_embed_0;
    modules::LinearWeights timestep_embed_2;
    modules::LinearWeights preprocess_conv;
    modules::LinearWeights postprocess_conv;
    modules::LinearWeights project_in;
    modules::LinearWeights project_out;
    core::TensorValue memory_tokens;
    modules::LinearWeights global_cond_embedder_0;
    modules::LinearWeights global_cond_embedder_2;
    std::vector<StableAudioRfLayerWeights> layers;
};

StableAudioRfDitWeights load_stable_audio_rf_dit_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class StableAudioRfDitRuntime {
public:
    StableAudioRfDitRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~StableAudioRfDitRuntime();

    StableAudioRfDitRuntime(const StableAudioRfDitRuntime &) = delete;
    StableAudioRfDitRuntime & operator=(const StableAudioRfDitRuntime &) = delete;

    void prepare(const StableAudioSamplingState & sampling, float cfg_scale) const;
    std::vector<float> sample(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
        std::string_view sampler_type,
        float cfg_scale,
        float apg_scale) const;
    void release_runtime_graphs() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const StableAudioAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<StableAudioRfDitWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::stable_audio
