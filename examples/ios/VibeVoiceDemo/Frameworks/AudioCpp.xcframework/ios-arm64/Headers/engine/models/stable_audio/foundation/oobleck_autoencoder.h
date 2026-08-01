#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
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

namespace engine::models::stable_audio::foundation {

struct OobleckSnakeWeights {
    core::TensorValue alpha;
    core::TensorValue beta;
};

struct OobleckResidualUnitWeights {
    OobleckSnakeWeights snake_0;
    modules::Conv1dWeights conv_1;
    OobleckSnakeWeights snake_2;
    modules::Conv1dWeights conv_3;
};

struct OobleckDecoderBlockWeights {
    OobleckSnakeWeights snake_0;
    modules::ConvTranspose1dWeights upsample;
    OobleckResidualUnitWeights residual_2;
    OobleckResidualUnitWeights residual_3;
    OobleckResidualUnitWeights residual_4;
};

struct OobleckEncoderBlockWeights {
    OobleckResidualUnitWeights residual_0;
    OobleckResidualUnitWeights residual_1;
    OobleckResidualUnitWeights residual_2;
    OobleckSnakeWeights snake_3;
    modules::Conv1dWeights downsample;
};

struct OobleckAutoencoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights encoder_in_conv;
    std::vector<OobleckEncoderBlockWeights> encoder_blocks;
    OobleckSnakeWeights encoder_final_snake;
    modules::Conv1dWeights encoder_out_conv;
    modules::Conv1dWeights in_conv;
    std::vector<OobleckDecoderBlockWeights> blocks;
    OobleckSnakeWeights final_snake;
    modules::Conv1dWeights out_conv;
};

OobleckAutoencoderWeights load_oobleck_autoencoder_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class OobleckAutoencoderRuntime {
public:
    OobleckAutoencoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~OobleckAutoencoderRuntime();

    OobleckAutoencoderRuntime(const OobleckAutoencoderRuntime &) = delete;
    OobleckAutoencoderRuntime & operator=(const OobleckAutoencoderRuntime &) = delete;

    void prepare_decode(int64_t batch, int64_t latent_tokens) const;
    std::vector<float> encode(
        const runtime::AudioBuffer & audio,
        int64_t target_frames,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & rng_policy) const;
    std::vector<runtime::AudioBuffer> decode(
        const std::vector<float> & latents,
        int64_t batch,
        int64_t latent_tokens) const;

private:
    class EncodeGraph;
    class DecodeGraph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const StableAudioAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<OobleckAutoencoderWeights> weights_;
    mutable std::unique_ptr<EncodeGraph> encode_graph_;
    mutable std::unique_ptr<DecodeGraph> decode_graph_;
};

}  // namespace engine::models::stable_audio::foundation
