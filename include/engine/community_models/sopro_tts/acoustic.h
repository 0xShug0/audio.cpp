#pragma once

#include "engine/community_models/sopro_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}
namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::community_models::sopro_tts {

struct SoproAcousticWeights;
struct SoproAcousticGraphs;

struct SoproAcousticRequest {
    // Reference tokens followed by the generated ones; the acoustic head sees
    // the whole span so the prompt mel and the new audio stay phase-coherent.
    std::vector<int32_t> semantic_tokens;
    std::vector<float> cond_vec;    // [cond_hidden_dim]
    std::vector<float> prompt_mel;  // [n_mels, prompt_frames], normalised
    int64_t prompt_frames = 0;
    int64_t total_frames = 0;
    int64_t steps = 2;
    uint64_t seed = 0;
};

// sopro/nn/acoustic.py AcousticHead.solve, offline (unchunked) path: a
// rectified-flow DiT with adaptive layer norm conditioning, solved with the
// Euler steps of a sway-sampled time grid while the prompt frames are pinned
// to the reference mel at every step.
class SoproAcousticRuntime final {
public:
    SoproAcousticRuntime(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~SoproAcousticRuntime();

    SoproAcousticRuntime(const SoproAcousticRuntime &) = delete;
    SoproAcousticRuntime & operator=(const SoproAcousticRuntime &) = delete;

    // Returns the normalised mel [n_mels, total_frames], channel-major.
    std::vector<float> solve(const SoproAcousticRequest & request) const;

private:
    const SoproModelConfig & config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const SoproAcousticWeights> weights_;
    mutable std::unique_ptr<SoproAcousticGraphs> graphs_;
};

// build_time_grid: linspace(0, 1, steps + 1) warped by the sway coefficient.
std::vector<float> build_time_grid(int64_t steps, float sway_coefficient);

// sinusoidal_time_embedding(t, dim, scale=1000).
std::vector<float> sinusoidal_time_embedding(float t, int64_t dim, float scale = 1000.0F);

}  // namespace engine::community_models::sopro_tts
