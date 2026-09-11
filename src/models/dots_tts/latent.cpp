#include "engine/models/dots_tts/latent.h"

#include "engine/framework/sampling/torch_random.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts {

DotsLatentCodec::DotsLatentCodec(std::shared_ptr<const DotsAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr || assets_->latent_stats == nullptr) {
        throw std::runtime_error("DotTTS latent codec requires latent stats");
    }
    mean_ = assets_->latent_stats->require_f32("mean", {assets_->config.latent_dim});
    const auto var = assets_->latent_stats->require_f32("var", {assets_->config.latent_dim});
    std_.resize(var.size());
    for (size_t i = 0; i < var.size(); ++i) {
        if (var[i] <= 0.0F) {
            throw std::runtime_error("DotTTS latent variance must be positive");
        }
        std_[i] = std::sqrt(var[i]);
    }
}

DotsLatentMatrix DotsLatentCodec::sample_from_encoder_latents(
    const std::vector<float> & latents,
    int64_t channels,
    int64_t frames,
    uint64_t seed,
    uint64_t offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & sampling_policy) const {
    const int64_t dims = assets_->config.latent_dim;
    if (channels != dims * 2 || frames <= 0 || static_cast<int64_t>(latents.size()) != channels * frames) {
        throw std::runtime_error("DotTTS encoder latent shape mismatch");
    }
    auto noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(dims * frames),
        seed,
        offset_blocks,
        sampling_policy,
        engine::sampling::TorchRandnPrecision::Float32
    );
    DotsLatentMatrix out;
    out.frames = frames;
    out.dims = dims;
    out.values.resize(static_cast<size_t>(frames * dims));
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t dim = 0; dim < dims; ++dim) {
            const float mean = latents[static_cast<size_t>(dim * frames + frame)];
            const float log_std = latents[static_cast<size_t>((dim + dims) * frames + frame)];
            out.values[static_cast<size_t>(frame * dims + dim)] =
                mean + noise[static_cast<size_t>(dim * frames + frame)] * std::exp(log_std);
        }
    }
    return out;
}

DotsLatentMatrix DotsLatentCodec::normalize(const DotsLatentMatrix & input) const {
    if (input.dims != assets_->config.latent_dim || static_cast<int64_t>(input.values.size()) != input.frames * input.dims) {
        throw std::runtime_error("DotTTS latent normalization shape mismatch");
    }
    DotsLatentMatrix out = input;
    for (int64_t frame = 0; frame < input.frames; ++frame) {
        for (int64_t dim = 0; dim < input.dims; ++dim) {
            float & value = out.values[static_cast<size_t>(frame * input.dims + dim)];
            value = (value - mean_[static_cast<size_t>(dim)]) / std_[static_cast<size_t>(dim)];
        }
    }
    return out;
}

DotsLatentMatrix DotsLatentCodec::denormalize(const DotsLatentMatrix & input) const {
    if (input.dims != assets_->config.latent_dim || static_cast<int64_t>(input.values.size()) != input.frames * input.dims) {
        throw std::runtime_error("DotTTS latent denormalization shape mismatch");
    }
    DotsLatentMatrix out = input;
    for (int64_t frame = 0; frame < input.frames; ++frame) {
        for (int64_t dim = 0; dim < input.dims; ++dim) {
            float & value = out.values[static_cast<size_t>(frame * input.dims + dim)];
            value = value * std_[static_cast<size_t>(dim)] + mean_[static_cast<size_t>(dim)];
        }
    }
    return out;
}

}  // namespace engine::models::dots_tts
