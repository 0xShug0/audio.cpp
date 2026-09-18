#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/liveavatar/assets.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::liveavatar {

class LiveAvatarPipelineState;

struct LiveAvatarGenerateRequest {
    std::string prompt;
    std::string negative_prompt;
    std::string reference_image_path;
    engine::runtime::AudioBuffer audio;
    int64_t height = 704;
    int64_t width = 1024;
    int64_t frames_per_clip = 48;
    int64_t steps = 4;
    int64_t max_clips = 1;
    uint64_t seed = 420;
    float guidance_scale = 0.0F;
    float shift = 3.0F;
    bool blockwise_generation = false;
    bool fused_cfg = true;
    bool sage_attention = true;
    bool memory_saver = true;
    bool denoiser_layerwise = false;
    int64_t denoiser_layerwise_batch = 16;
    int64_t vae_encoder_chunk_size = 16;
    int64_t vae_decoder_tile_size = 0;
    int64_t target_cache_blocks = 0;
    bool vae_cache_f16 = true;
};

struct LiveAvatarVideoResult {
    int64_t width = 0;
    int64_t height = 0;
    int64_t frames = 0;
    int64_t fps = 16;
    std::vector<std::byte> rgb24;
};

using LiveAvatarVideoChunkCallback = std::function<void(LiveAvatarVideoResult)>;

class LiveAvatarPipelineRuntime final {
public:
    LiveAvatarPipelineRuntime(
        std::shared_ptr<const LiveAvatarAssets> assets,
        engine::core::ExecutionContext & execution,
        bool denoiser_weight_streaming = false);
    ~LiveAvatarPipelineRuntime();

    LiveAvatarVideoResult generate(
        const LiveAvatarGenerateRequest & request,
        const LiveAvatarVideoChunkCallback & chunk_callback = {});

private:
    std::shared_ptr<const LiveAvatarAssets> assets_;
    engine::core::ExecutionContext & execution_;
    std::unique_ptr<LiveAvatarPipelineState> impl_;
    bool denoiser_weight_streaming_ = false;
};

struct LiveAvatarVAEProbeRequest {
    std::vector<float> input;
    int64_t channels = 3;
    int64_t frames = 1;
    int64_t height = 0;
    int64_t width = 0;
    bool scale_encoded_latents = false;
};

struct LiveAvatarVAEProbeResult {
    std::vector<float> output;
    engine::core::TensorShape output_shape;
};

LiveAvatarVAEProbeResult run_liveavatar_vae_encode_probe(
    const LiveAvatarAssets & assets,
    engine::core::ExecutionContext & execution,
    const LiveAvatarVAEProbeRequest & request);

}  // namespace engine::community_models::liveavatar
