#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <unordered_map>

namespace engine::models::sam_audio {

class DiTModule {
public:
    DiTModule(const assets::TensorSource & source, core::ExecutionContext & execution,
              const std::filesystem::path & config);
    core::TensorValue build(core::ModuleBuildContext & ctx,
                            const core::TensorValue & input,
                            const core::TensorValue & memory,
                            const core::TensorValue & time_embedding,
                            const core::TensorValue & positions) const;
    core::TensorValue align_inputs(core::ModuleBuildContext & ctx,
                                  const core::TensorValue & concatenated_audio,
                                  const core::TensorValue & video,
                                  const core::TensorValue & aligned_anchor_ids) const;
    core::TensorValue project_text(core::ModuleBuildContext & ctx,
                                  const core::TensorValue & text) const;
    int64_t dim() const noexcept { return dim_; }
    int64_t frequency_dim() const noexcept { return frequency_dim_; }
    int64_t max_positions() const noexcept { return max_positions_; }

private:
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> weights_;
    core::TensorValue ones_;
    int64_t dim_, heads_, layers_, frequency_dim_, max_positions_;
    float norm_eps_, rope_theta_;
    bool qk_norm_, context_norm_;
};

struct DiTConditioning {
    int64_t frames = 0;
    int64_t tokens = 0;
    std::vector<float> audio;
    std::vector<float> text;
    std::vector<float> video;
    std::vector<int32_t> anchors;
};

class DiTRuntime {
public:
    DiTRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution,
               const std::filesystem::path & config, bool memory_bounded = false);
    ~DiTRuntime();
    std::vector<float> sample(const DiTConditioning & conditioning, const std::vector<float> & noise, int steps);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sam_audio
