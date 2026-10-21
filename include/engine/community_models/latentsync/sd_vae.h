#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncSdVaeConfig {
    std::array<int64_t, 4> channels{128, 256, 512, 512};
    int64_t latent_channels = 4;
    int64_t layers_per_block = 2;
    int64_t norm_groups = 32;
    bool cuda_cudnn = false;
    bool cuda_f16_activations = false;
};

struct SdVaeResnetWeights {
    modules::NormWeights norm1, norm2;
    modules::Conv2dWeights conv1, conv2;
    std::optional<modules::Conv2dWeights> shortcut;
};

struct SdVaeStageWeights {
    std::vector<SdVaeResnetWeights> resnets;
    std::optional<modules::Conv2dWeights> resample;
};

struct SdVaeTowerWeights {
    modules::Conv2dWeights input, output, quant;
    modules::NormWeights output_norm, attention_norm;
    struct AttentionWeights {
        modules::LinearWeights query, key, value, out;
    } attention;
    std::array<SdVaeResnetWeights, 2> mid_resnets;
    std::vector<SdVaeStageWeights> stages;
};

SdVaeTowerWeights load_sd_vae_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const LatentSyncSdVaeConfig & config,
    bool encoder, assets::TensorStorageType type);

class LatentSyncSdVae {
public:
    explicit LatentSyncSdVae(LatentSyncSdVaeConfig config);
    core::TensorValue build_encoder(core::ModuleBuildContext & ctx,
        const core::TensorValue & image, const SdVaeTowerWeights & weights) const;
    core::TensorValue build_decoder(core::ModuleBuildContext & ctx,
        const core::TensorValue & latent, const SdVaeTowerWeights & weights) const;

private:
    core::TensorValue build_resnet(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const SdVaeResnetWeights & weights) const;
    core::TensorValue build_mid(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const SdVaeTowerWeights & weights) const;
    LatentSyncSdVaeConfig config_;
};

class LatentSyncSdVaeRuntime {
public:
    LatentSyncSdVaeRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
        assets::TensorStorageType weight_type, int64_t resolution = 512,
        bool cuda_cudnn = false, bool cuda_f16_activations = false,
        bool share_graph_workspace = false);
    ~LatentSyncSdVaeRuntime();
    std::vector<float> encode_moments(const std::vector<float> & image);
    std::vector<float> encode(const std::vector<float> & image, const std::vector<float> & posterior_noise);
    std::vector<float> decode(const std::vector<float> & scaled_latent);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
