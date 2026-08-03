#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/stable_audio/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::stable_audio {

struct StableAudioSameNormWeights {
    core::TensorValue alpha;
    core::TensorValue beta;
    core::TensorValue gamma;
};

struct StableAudioSameAttentionWeights {
    modules::LinearWeights to_qkv;
    modules::LinearWeights to_out;
    StableAudioSameNormWeights q_norm;
    StableAudioSameNormWeights k_norm;
};

struct StableAudioSameFeedForwardWeights {
    modules::LinearWeights in_proj;
    modules::LinearWeights out_proj;
};

struct StableAudioSameTransformerWeights {
    StableAudioSameNormWeights pre_norm;
    StableAudioSameNormWeights ff_norm;
    StableAudioSameAttentionWeights self_attn;
    StableAudioSameFeedForwardWeights ff;
};

struct StableAudioSameResamplingBlockWeights {
    modules::Conv1dWeights mapping;
    core::TensorValue new_tokens;
    std::vector<StableAudioSameTransformerWeights> transformers;
};

struct StableAudioSameWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue bottleneck_bias;
    core::TensorValue bottleneck_scaling_factor;
    core::TensorValue bottleneck_running_std;
    modules::LinearWeights decoder_in;
    StableAudioSameResamplingBlockWeights decoder_block;
    StableAudioSameResamplingBlockWeights encoder_block;
    modules::LinearWeights encoder_out;
};

StableAudioSameWeights load_stable_audio_same_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class StableAudioSameRuntime {
public:
    StableAudioSameRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~StableAudioSameRuntime();

    StableAudioSameRuntime(const StableAudioSameRuntime &) = delete;
    StableAudioSameRuntime & operator=(const StableAudioSameRuntime &) = delete;

    void prepare_decode(int64_t batch, int64_t latent_tokens, bool chunked) const;
    std::vector<runtime::AudioBuffer> decode(
        const std::vector<float> & latents,
        int64_t batch,
        int64_t latent_tokens,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        bool chunked) const;

    std::vector<float> encode(
        const runtime::AudioBuffer & audio,
        int64_t audio_sample_size,
        uint64_t seed,
        uint64_t & rng_offset_blocks) const;
    void release_runtime_graphs() const;

private:
    class Graph;
    class EncoderGraph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const StableAudioAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    engine::sampling::TorchCudaSamplingPolicy rng_policy_;
    mutable std::unique_ptr<StableAudioSameWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
    mutable std::unique_ptr<EncoderGraph> encoder_graph_;
};

}  // namespace engine::models::stable_audio
