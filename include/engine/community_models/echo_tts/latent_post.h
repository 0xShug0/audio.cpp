#pragma once

#include "engine/community_models/echo_tts/config.h"

#include <cstdint>
#include <vector>

namespace engine::models::echo_tts {

// Forward PCA: Fish z_q (frames, ae_latent_dim) -> DiT latents (frames, latent_size).
// Mirrors inference.py::ae_encode, minus the transposes, which the caller owns.
std::vector<float> pca_project(
    const EchoPcaState & pca,
    const EchoTtsConfig & config,
    const std::vector<float> & z_q,
    int64_t frames);

// Inverse PCA: DiT latents (frames, latent_size) -> Fish z_q (frames, ae_latent_dim).
// Mirrors inference.py::ae_decode.
std::vector<float> pca_unproject(
    const EchoPcaState & pca,
    const EchoTtsConfig & config,
    const std::vector<float> & latents,
    int64_t frames);

// Port of inference.py::find_flattening_point. `latents` is (frames, latent_size)
// row-major. Returns the number of leading frames to keep.
//
// The generated latent tail goes flat once the model has finished speaking, and
// this heuristic is what upstream uses to find that point. It is deliberately
// bit-for-bit faithful, including the unbiased (N-1) variance, because the crop
// index directly sets the output duration.
int64_t find_flattening_point(
    const std::vector<float> & latents,
    int64_t frames,
    int64_t latent_size,
    int64_t window_size = 20,
    float std_threshold = 0.05F,
    float target_value = 0.0F);

}  // namespace engine::models::echo_tts
