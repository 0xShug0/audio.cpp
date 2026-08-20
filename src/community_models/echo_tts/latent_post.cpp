#include "engine/community_models/echo_tts/latent_post.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::echo_tts {

std::vector<float> pca_project(
    const EchoPcaState & pca,
    const EchoTtsConfig & config,
    const std::vector<float> & z_q,
    int64_t frames) {
    const int64_t features = config.ae_latent_dim;
    const int64_t components = config.latent_size;
    if (static_cast<int64_t>(z_q.size()) != frames * features) {
        throw std::runtime_error("Echo-TTS PCA projection received a mis-shaped z_q buffer");
    }
    if (static_cast<int64_t>(pca.components.size()) != components * features ||
        static_cast<int64_t>(pca.mean.size()) != features) {
        throw std::runtime_error("Echo-TTS PCA state has unexpected dimensions");
    }

    std::vector<float> out(static_cast<size_t>(frames * components), 0.0F);
    for (int64_t f = 0; f < frames; ++f) {
        const float * row = z_q.data() + f * features;
        float * dst = out.data() + f * components;
        for (int64_t c = 0; c < components; ++c) {
            const float * basis = pca.components.data() + c * features;
            double acc = 0.0;
            for (int64_t k = 0; k < features; ++k) {
                acc += static_cast<double>(row[k] - pca.mean[static_cast<size_t>(k)]) *
                       static_cast<double>(basis[k]);
            }
            dst[c] = static_cast<float>(acc) * pca.latent_scale;
        }
    }
    return out;
}

std::vector<float> pca_unproject(
    const EchoPcaState & pca,
    const EchoTtsConfig & config,
    const std::vector<float> & latents,
    int64_t frames) {
    const int64_t features = config.ae_latent_dim;
    const int64_t components = config.latent_size;
    if (static_cast<int64_t>(latents.size()) != frames * components) {
        throw std::runtime_error("Echo-TTS PCA inverse received a mis-shaped latent buffer");
    }
    if (static_cast<int64_t>(pca.components.size()) != components * features ||
        static_cast<int64_t>(pca.mean.size()) != features) {
        throw std::runtime_error("Echo-TTS PCA state has unexpected dimensions");
    }
    if (pca.latent_scale == 0.0F) {
        throw std::runtime_error("Echo-TTS PCA latent_scale must be non-zero");
    }

    std::vector<float> out(static_cast<size_t>(frames * features), 0.0F);
    const float inv_scale = 1.0F / pca.latent_scale;
    for (int64_t f = 0; f < frames; ++f) {
        const float * row = latents.data() + f * components;
        float * dst = out.data() + f * features;
        for (int64_t k = 0; k < features; ++k) {
            dst[k] = pca.mean[static_cast<size_t>(k)];
        }
        for (int64_t c = 0; c < components; ++c) {
            const float coeff = row[c] * inv_scale;
            if (coeff == 0.0F) {
                continue;
            }
            const float * basis = pca.components.data() + c * features;
            for (int64_t k = 0; k < features; ++k) {
                dst[k] += coeff * basis[k];
            }
        }
    }
    return out;
}

int64_t find_flattening_point(
    const std::vector<float> & latents,
    int64_t frames,
    int64_t latent_size,
    int64_t window_size,
    float std_threshold,
    float target_value) {
    if (frames <= 0 || latent_size <= 0 || window_size <= 0) {
        return frames;
    }
    if (static_cast<int64_t>(latents.size()) != frames * latent_size) {
        throw std::runtime_error("Echo-TTS flattening search received a mis-shaped latent buffer");
    }

    // Upstream pads the sequence with `window_size` zero frames before scanning,
    // so a generation that runs to the end of the window still terminates.
    const int64_t padded_frames = frames + window_size;
    const int64_t count = window_size * latent_size;
    if (count < 2) {
        return frames;
    }

    auto value_at = [&](int64_t frame, int64_t channel) -> double {
        if (frame >= frames) {
            return 0.0;
        }
        return static_cast<double>(latents[static_cast<size_t>(frame * latent_size + channel)]);
    };

    for (int64_t start = 0; start < padded_frames - window_size; ++start) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t f = start; f < start + window_size; ++f) {
            for (int64_t c = 0; c < latent_size; ++c) {
                const double v = value_at(f, c);
                sum += v;
                sum_sq += v * v;
            }
        }
        const double mean = sum / static_cast<double>(count);
        // torch.std defaults to the unbiased estimator (correction = 1).
        const double variance =
            (sum_sq - sum * mean) / static_cast<double>(count - 1);
        const double stddev = variance > 0.0 ? std::sqrt(variance) : 0.0;
        if (stddev < static_cast<double>(std_threshold) &&
            std::abs(mean - static_cast<double>(target_value)) < 0.1) {
            return start;
        }
    }
    return frames;
}

}  // namespace engine::models::echo_tts
