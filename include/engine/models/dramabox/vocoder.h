#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/models/dramabox/assets.h"
#include "engine/models/dramabox/audio_vae.h"

#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::dramabox {

struct DramaBoxVocoderConv1dWeights {
    modules::Conv1dWeights conv;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;
    bool use_bias = true;
};

struct DramaBoxVocoderConvTranspose1dWeights {
    modules::ConvTranspose1dWeights conv;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t padding = 0;
    bool use_bias = true;
};

struct DramaBoxVocoderActivationWeights {
    core::TensorValue alpha;
    core::TensorValue inv_beta;
    core::TensorValue up_filter;
    core::TensorValue down_filter;
};

struct DramaBoxVocoderResBlockWeights {
    std::vector<DramaBoxVocoderConv1dWeights> convs1;
    std::vector<DramaBoxVocoderConv1dWeights> convs2;
    std::vector<DramaBoxVocoderActivationWeights> activations;
};

struct DramaBoxBigVganWeights {
    DramaBoxVocoderConv1dWeights conv_pre;
    std::vector<DramaBoxVocoderConvTranspose1dWeights> ups;
    std::vector<DramaBoxVocoderResBlockWeights> resblocks;
    DramaBoxVocoderActivationWeights activation_post;
    DramaBoxVocoderConv1dWeights conv_post;
    int64_t sample_rate = 0;
    int64_t num_mels = 0;
    int64_t initial_channel = 0;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
};

struct DramaBoxVocoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    DramaBoxBigVganWeights vocoder;
    DramaBoxBigVganWeights bwe;
    core::TensorValue mel_basis;
    core::TensorValue stft_forward_basis;
    core::TensorValue resampler_filter;
};

struct DramaBoxVocoderOutput {
    int64_t channels = 0;
    int64_t samples = 0;
    int64_t sample_rate = 0;
    std::vector<float> waveform;
};

DramaBoxVocoderWeights load_dramabox_vocoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class DramaBoxVocoderRuntime {
public:
    DramaBoxVocoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~DramaBoxVocoderRuntime();

    DramaBoxVocoderRuntime(const DramaBoxVocoderRuntime &) = delete;
    DramaBoxVocoderRuntime & operator=(const DramaBoxVocoderRuntime &) = delete;

    void prepare(int64_t mel_frames) const;
    DramaBoxVocoderOutput synthesize(const DramaBoxDecodedMel & mel) const;

private:
    class VocoderGraph;
    class BweGraph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const DramaBoxAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<DramaBoxVocoderWeights> weights_;
    mutable std::unique_ptr<VocoderGraph> vocoder_graph_;
    mutable std::unique_ptr<BweGraph> bwe_graph_;
};

}  // namespace engine::models::dramabox
