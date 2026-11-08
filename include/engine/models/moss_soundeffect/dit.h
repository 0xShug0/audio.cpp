#pragma once

#include "engine/models/moss_soundeffect/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/attention_fallback.h"

#include <memory>
#include <vector>

namespace engine::models::moss_soundeffect {

// Owned by the model session; execution must outlive this runtime.
class DitRuntime {
public:
    DitRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets,
               assets::TensorStorageType storage_type,
               core::AttentionPreference attention = core::AttentionPreference::Auto);
    ~DitRuntime();

    DitRuntime(const DitRuntime &) = delete;
    DitRuntime & operator=(const DitRuntime &) = delete;

    // Each context is the flattened [1, text_length, text_dim] text-encoder output.
    // Padding must already be zeroed as in WanPrompter, before text projection.
    void prepare_context(const std::vector<float> & positive, const std::vector<float> & negative);

    // Latents and returned velocity are flattened [1, latent_dim, frames].
    // Positive and negative predictions are independent batch-one evaluations.
    std::vector<float> predict(const std::vector<float> & latents, float timestep, bool negative = false);

    // Same B1 latent input; returns flattened [2, latent_dim, frames], positive then negative.
    std::vector<float> predict_cfg(const std::vector<float> & latents, float timestep);

    // Upload noise once; CFG/Euler and latent state remain on the backend.
    std::vector<float> sample(const std::vector<float> & noise, int steps, float guidance, float shift);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_soundeffect
