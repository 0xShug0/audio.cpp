#include "engine/community_models/minimax_h3/generation_plan.h"

#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::minimax_h3 {

namespace core = engine::core;

constexpr int64_t kHeightDivision = 32;
constexpr int64_t kWidthDivision = 32;
constexpr int64_t kTimeDivision = 17;
constexpr int64_t kTimeRemainder = 5;

std::vector<float> h3_sigmas(int64_t steps, float shift) {
    std::vector<float> sigmas(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        const float base = 1.0F - static_cast<float>(i) / static_cast<float>(steps);
        sigmas[static_cast<size_t>(i)] = shift * base / (1.0F + (shift - 1.0F) * base);
    }
    return sigmas;
}

std::vector<float> build_step_timestep_features(
    const H3Config & cfg,
    const std::vector<float> & audio_sigmas,
    const std::vector<float> & video_sigmas) {
    std::vector<float> out(static_cast<size_t>(cfg.denoise_steps * 2 * cfg.timestep_input_dim));
    const int64_t half = cfg.timestep_input_dim / 2;
    for (int64_t step = 0; step < cfg.denoise_steps; ++step) {
        const float audio_timestep = 1.0F - audio_sigmas[static_cast<size_t>(step)];
        const float video_timestep = 1.0F - video_sigmas[static_cast<size_t>(step)];
        const size_t offset = static_cast<size_t>(step * 2 * cfg.timestep_input_dim);
        for (int64_t i = 0; i < half; ++i) {
            const float freq = std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
            const float video_arg = video_timestep * freq;
            const float audio_arg = audio_timestep * freq;
            out[offset + static_cast<size_t>(i)] = std::cos(video_arg);
            out[offset + static_cast<size_t>(half + i)] = std::sin(video_arg);
            out[offset + static_cast<size_t>(cfg.timestep_input_dim + i)] = std::cos(audio_arg);
            out[offset + static_cast<size_t>(cfg.timestep_input_dim + half + i)] = std::sin(audio_arg);
        }
    }
    return out;
}

std::vector<float> build_step_timestep_values(
    const H3Config & cfg,
    const std::vector<float> & audio_sigmas,
    const std::vector<float> & video_sigmas) {
    std::vector<float> out(static_cast<size_t>(cfg.denoise_steps * 2));
    for (int64_t step = 0; step < cfg.denoise_steps; ++step) {
        out[static_cast<size_t>(step * 2)] = 1.0F - video_sigmas[static_cast<size_t>(step)];
        out[static_cast<size_t>(step * 2 + 1)] = 1.0F - audio_sigmas[static_cast<size_t>(step)];
    }
    return out;
}

std::vector<float> build_step_adaln_curve_inputs(
    const H3Config & cfg,
    const std::vector<float> & timestep_values,
    const std::vector<float> & table) {
    if (cfg.adaln_curve_grid < 2 || cfg.time_embed_dim <= 0) {
        throw std::runtime_error("MiniMax-H3 AdaLN curve config is invalid");
    }
    const size_t expected = static_cast<size_t>(cfg.adaln_curve_grid * cfg.time_embed_dim);
    if (table.size() != expected) {
        throw std::runtime_error("MiniMax-H3 AdaLN curve table size mismatch");
    }
    std::vector<float> out(static_cast<size_t>(cfg.denoise_steps * 2 * cfg.time_embed_dim));
    const float last = static_cast<float>(cfg.adaln_curve_grid - 1);
    for (int64_t row = 0; row < cfg.denoise_steps * 2; ++row) {
        const float clamped = std::clamp(timestep_values[static_cast<size_t>(row)], 0.0F, 1.0F);
        const float pos = clamped * last;
        const int64_t lower = std::min<int64_t>(static_cast<int64_t>(std::floor(pos)), cfg.adaln_curve_grid - 2);
        const float alpha = pos - static_cast<float>(lower);
        const size_t dst = static_cast<size_t>(row * cfg.time_embed_dim);
        const size_t lo = static_cast<size_t>(lower * cfg.time_embed_dim);
        const size_t hi = static_cast<size_t>((lower + 1) * cfg.time_embed_dim);
        for (int64_t dim = 0; dim < cfg.time_embed_dim; ++dim) {
            const float a = table[lo + static_cast<size_t>(dim)];
            const float b = table[hi + static_cast<size_t>(dim)];
            out[dst + static_cast<size_t>(dim)] = a + alpha * (b - a);
        }
    }
    return out;
}


InitialLatents generate_initial_latents(
    const H3Config & cfg,
    const core::ExecutionContext & execution,
    uint32_t seed) {
    const size_t video_tensor_count = static_cast<size_t>(cfg.video_latents_dim * cfg.video_latent_t * cfg.video_latent_h * cfg.video_latent_w);
    const size_t audio_tensor_count = static_cast<size_t>(cfg.audio_channels * cfg.audio_latents_dim * cfg.audio_steps);
    auto pack_video = [&cfg](const std::vector<float> & tensor) {
        std::vector<float> rows(static_cast<size_t>(cfg.video_patches * cfg.video_latents_dim * 4));
        const int64_t ph = cfg.video_latent_h / 2;
        const int64_t pw = cfg.video_latent_w / 2;
        for (int64_t t = 0; t < cfg.video_latent_t; ++t) {
            for (int64_t hp = 0; hp < ph; ++hp) {
                for (int64_t wp = 0; wp < pw; ++wp) {
                    const int64_t row = (t * ph + hp) * pw + wp;
                    for (int64_t c = 0; c < cfg.video_latents_dim; ++c) {
                        for (int64_t ih = 0; ih < 2; ++ih) {
                            for (int64_t iw = 0; iw < 2; ++iw) {
                                const int64_t dst_col = (c * 2 + ih) * 2 + iw;
                                const int64_t src = ((c * cfg.video_latent_t + t) * cfg.video_latent_h + hp * 2 + ih) * cfg.video_latent_w + wp * 2 + iw;
                                rows[static_cast<size_t>(row * cfg.video_latents_dim * 4 + dst_col)] = tensor[static_cast<size_t>(src)];
                            }
                        }
                    }
                }
            }
        }
        return rows;
    };
    auto pack_audio = [&cfg](const std::vector<float> & tensor) {
        std::vector<float> rows(static_cast<size_t>(cfg.audio_channels * cfg.audio_steps * cfg.audio_latents_dim));
        for (int64_t channel = 0; channel < cfg.audio_channels; ++channel) {
            for (int64_t t = 0; t < cfg.audio_steps; ++t) {
                const int64_t row = channel * cfg.audio_steps + t;
                for (int64_t d = 0; d < cfg.audio_latents_dim; ++d) {
                    const int64_t src = (channel * cfg.audio_latents_dim + d) * cfg.audio_steps + t;
                    rows[static_cast<size_t>(row * cfg.audio_latents_dim + d)] = tensor[static_cast<size_t>(src)];
                }
            }
        }
        return rows;
    };
    if (execution.backend_type() == core::BackendType::Cuda) {
        const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
            execution.backend_type(),
            execution.config().device,
            "minimax_h3.rng",
            "MiniMax-H3");
        uint64_t offset_blocks = 0;
        auto video = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
            video_tensor_count,
            seed,
            offset_blocks,
            rng_policy,
            engine::sampling::TorchRandnPrecision::BFloat16);
        offset_blocks = 0;
        auto audio = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
            audio_tensor_count,
            seed,
            offset_blocks,
            rng_policy,
            engine::sampling::TorchRandnPrecision::BFloat16);
        return {pack_video(video), pack_audio(audio)};
    }
    auto video = engine::sampling::generate_torch_cuda_randn(video_tensor_count, seed);
    auto audio = engine::sampling::generate_torch_cuda_randn(
        audio_tensor_count,
        seed,
        engine::sampling::TorchRandnPrecision::BFloat16,
        0);
    return {pack_video(video), pack_audio(audio)};
}

H3Config resolve_generation_config(const H3Config & base, const MiniMaxH3GenerateRequest & request) {
    H3Config out = base;
    out.denoise_steps = request.steps > 0 ? request.steps : base.denoise_steps;
    out.flow_shift = request.flow_shift;
    out.audio_flow_shift = request.audio_flow_shift;
    const int64_t requested_height = request.height > 0 ? request.height : base.height;
    const int64_t requested_width = request.width > 0 ? request.width : base.width;
    out.height = ((requested_height + kHeightDivision - 1) / kHeightDivision) * kHeightDivision;
    out.width = ((requested_width + kWidthDivision - 1) / kWidthDivision) * kWidthDivision;
    out.num_frames = request.num_frames > 0 ? request.num_frames : base.num_frames;
    if (out.num_frames % kTimeDivision != kTimeRemainder) {
        out.num_frames =
            ((out.num_frames - kTimeRemainder + kTimeDivision - 1) / kTimeDivision) * kTimeDivision + kTimeRemainder;
    }
    out.video_latent_t = ((out.num_frames - 5) / 17) * 5 + 2;
    out.video_latent_h = out.height / 16;
    out.video_latent_w = out.width / 16;
    out.audio_steps = static_cast<int64_t>(std::llround(static_cast<double>(out.num_frames) / 24.0 * 40.0));
    out.video_patches = out.video_latent_t * (out.video_latent_h / 2) * (out.video_latent_w / 2);
    return out;
}

}  // namespace engine::models::minimax_h3
