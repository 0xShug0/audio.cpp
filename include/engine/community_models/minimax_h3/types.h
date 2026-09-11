#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::minimax_h3 {

enum class MiniMaxH3DitAccelerationMode {
    None,
    FirstBlockCache,
    Spectrum,
};

enum class MiniMaxH3SamplerMode {
    Euler,
    ResMultistep,
    Dpmpp2m,
    UniPC,
};

struct MiniMaxH3GenerateRequest {
    std::string prompt;
    std::string negative_prompt = " ";
    int64_t steps = 0;
    uint32_t seed = 42;
    int64_t height = 0;
    int64_t width = 0;
    int64_t num_frames = 0;
    float guidance_scale = 1.0F;
    MiniMaxH3SamplerMode sampler = MiniMaxH3SamplerMode::Euler;
    float flow_shift = 12.0F;
    float audio_flow_shift = 3.0F;
    bool return_video = false;
    bool text_layerwise = false;
    bool dit_layerwise = false;
    int64_t text_layerwise_batch = 1;
    int64_t dit_layerwise_batch = 1;
    int64_t dit_mlp_chunk_tokens = 0;
    MiniMaxH3DitAccelerationMode dit_acceleration = MiniMaxH3DitAccelerationMode::None;
    float first_block_cache_threshold = 0.10F;
    float first_block_cache_start_percent = 0.10F;
    float first_block_cache_end_percent = 0.95F;
    float first_block_cache_start_sigma = 0.95F;
    float first_block_cache_end_sigma = 0.10F;
    bool first_block_cache_sigma_window = false;
    int64_t first_block_cache_max_consecutive = 2;
    int64_t spectrum_warmup_steps = 1;
    float spectrum_initial_window = 2.0F;
    float spectrum_flex_window = 0.75F;
    int64_t spectrum_degree = 1;
    int64_t spectrum_history_size = 8;
    float spectrum_ridge_lambda = 0.1F;
};

struct MiniMaxH3VideoFrames {
    int width = 0;
    int height = 0;
    int frames = 0;
    int fps = 24;
    std::vector<std::byte> rgb24;
};

struct MiniMaxH3GenerateResult {
    int sample_rate = 32000;
    int channels = 1;
    std::vector<float> samples;
    std::optional<MiniMaxH3VideoFrames> video;
};

}  // namespace engine::models::minimax_h3
