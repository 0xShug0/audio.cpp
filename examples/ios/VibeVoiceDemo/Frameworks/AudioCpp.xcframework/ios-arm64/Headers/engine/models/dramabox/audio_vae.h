#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/runtime/model.h"
#include "engine/models/dramabox/assets.h"

#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::dramabox {

struct DramaBoxVaeResnetBlockWeights {
    modules::PixelNormCausalConv2dResBlockConfig config;
    modules::PixelNormCausalConv2dResBlockWeights block;
};

struct DramaBoxVaeUpStageWeights {
    std::vector<DramaBoxVaeResnetBlockWeights> blocks;
    std::optional<modules::Conv2dWeights> upsample;
    int64_t channels = 0;
};

struct DramaBoxAudioVaeDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<float> latent_mean;
    std::vector<float> latent_std;
    modules::Conv2dWeights conv_in;
    DramaBoxVaeResnetBlockWeights mid_block_1;
    DramaBoxVaeResnetBlockWeights mid_block_2;
    std::vector<DramaBoxVaeUpStageWeights> up;
    modules::Conv2dWeights conv_out;
};

struct DramaBoxVaeDownStageWeights {
    std::vector<DramaBoxVaeResnetBlockWeights> blocks;
    std::optional<modules::Conv2dWeights> downsample;
    int64_t channels = 0;
};

struct DramaBoxAudioVaeEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<float> latent_mean;
    std::vector<float> latent_std;
    modules::Conv2dWeights conv_in;
    std::vector<DramaBoxVaeDownStageWeights> down;
    DramaBoxVaeResnetBlockWeights mid_block_1;
    DramaBoxVaeResnetBlockWeights mid_block_2;
    modules::Conv2dWeights conv_out;
};

struct DramaBoxDecodedMel {
    int64_t batch = 0;
    int64_t channels = 0;
    int64_t frames = 0;
    int64_t mel_bins = 0;
    std::vector<float> values;
    const ggml_tensor * device_values = nullptr;
};

struct DramaBoxEncodedReferenceLatents {
    int64_t tokens = 0;
    std::vector<float> values;
};

DramaBoxAudioVaeDecoderWeights load_dramabox_audio_vae_decoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

DramaBoxAudioVaeEncoderWeights load_dramabox_audio_vae_encoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

std::vector<float> reference_log_mel(
    const runtime::AudioBuffer & audio_buffer,
    const DramaBoxConfig & config,
    float ref_duration,
    int threads,
    int64_t & frames_out);

class DramaBoxAudioVaeDecoderRuntime {
public:
    DramaBoxAudioVaeDecoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~DramaBoxAudioVaeDecoderRuntime();

    DramaBoxAudioVaeDecoderRuntime(const DramaBoxAudioVaeDecoderRuntime &) = delete;
    DramaBoxAudioVaeDecoderRuntime & operator=(const DramaBoxAudioVaeDecoderRuntime &) = delete;

    void prepare(int64_t batch, int64_t latent_frames) const;
    DramaBoxDecodedMel decode(const std::vector<float> & patch_latents, int64_t batch, int64_t latent_frames) const;
    DramaBoxDecodedMel decode_to_device(const std::vector<float> & patch_latents, int64_t batch, int64_t latent_frames) const;
    void release_runtime_state() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const DramaBoxAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<DramaBoxAudioVaeDecoderWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
};

class DramaBoxAudioVaeEncoderRuntime {
public:
    DramaBoxAudioVaeEncoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        assets::TensorStorageType weight_storage_type);
    ~DramaBoxAudioVaeEncoderRuntime();

    DramaBoxAudioVaeEncoderRuntime(const DramaBoxAudioVaeEncoderRuntime &) = delete;
    DramaBoxAudioVaeEncoderRuntime & operator=(const DramaBoxAudioVaeEncoderRuntime &) = delete;

    void prepare(int64_t batch, int64_t mel_frames) const;
    DramaBoxEncodedReferenceLatents encode(const std::vector<float> & mel, int64_t batch, int64_t mel_frames) const;
    void release_runtime_state() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const DramaBoxAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    mutable std::unique_ptr<DramaBoxAudioVaeEncoderWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::dramabox
