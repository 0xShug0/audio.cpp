#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/dramabox/assets.h"
#include "engine/models/dramabox/prompt_connector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::dramabox {

struct DramaBoxAdaLayerNormWeights {
    modules::LinearWeights timestep_linear_1;
    modules::LinearWeights timestep_linear_2;
    modules::LinearWeights output_linear;
    int64_t output_coefficient = 0;
};

struct DramaBoxDitSelfAttentionWeights {
    modules::LinearWeights qkv_gate;
    std::optional<modules::LinearWeights> q;
    std::optional<modules::LinearWeights> k;
    std::optional<modules::LinearWeights> v;
    std::optional<modules::LinearWeights> gate;
    modules::LinearWeights out;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
};

struct DramaBoxDitCrossAttentionWeights {
    modules::LinearWeights q_gate;
    modules::LinearWeights kv;
    modules::LinearWeights out;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
};

struct DramaBoxDitBlockWeights {
    core::TensorValue audio_scale_shift_table;
    core::TensorValue audio_prompt_scale_shift_table;
    DramaBoxDitSelfAttentionWeights self_attention;
    DramaBoxDitCrossAttentionWeights cross_attention;
    modules::LinearWeights ff_in;
    modules::LinearWeights ff_out;
};

struct DramaBoxDitWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights patchify_proj;
    DramaBoxAdaLayerNormWeights adaln;
    DramaBoxAdaLayerNormWeights prompt_adaln;
    std::vector<DramaBoxDitBlockWeights> blocks;
    core::TensorValue output_scale_shift_table;
    modules::LinearWeights output_proj;
};

struct DramaBoxDitInputs {
    int64_t batch = 0;
    int64_t tokens = 0;
    bool stg_enabled = false;
    int64_t ref_tokens = 0;
    const std::vector<float> * latent = nullptr;
    const std::vector<float> * sigma_features = nullptr;
};

DramaBoxDitWeights load_dramabox_dit_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type,
    DramaBoxPerfMode perf_mode = DramaBoxPerfMode::Exact);

class DramaBoxDitRuntime {
public:
    DramaBoxDitRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        assets::TensorStorageType weight_storage_type,
        DramaBoxPerfMode perf_mode);
    ~DramaBoxDitRuntime();

    DramaBoxDitRuntime(const DramaBoxDitRuntime &) = delete;
    DramaBoxDitRuntime & operator=(const DramaBoxDitRuntime &) = delete;

    void prepare(int64_t batch, int64_t tokens, int64_t context_tokens, bool stg_enabled, int64_t ref_tokens) const;
    void prepare_static_inputs(
        int64_t batch,
        int64_t tokens,
        bool stg_enabled,
        int64_t ref_tokens,
        const DramaBoxConditioningEncoding & conditioning,
        const std::vector<float> & rope_cos,
        const std::vector<float> & rope_sin,
        const std::vector<float> & timestep_mask) const;
    std::vector<float> forward(const DramaBoxDitInputs & inputs) const;
    void release_runtime_state() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const DramaBoxAssets> assets_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    DramaBoxPerfMode perf_mode_ = DramaBoxPerfMode::Exact;
    mutable std::unique_ptr<DramaBoxDitWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
};

std::vector<float> make_dramabox_timestep_features(const std::vector<float> & timesteps);
void fill_dramabox_timestep_features(
    const std::vector<float> & timesteps,
    std::vector<float> & out);
void make_dramabox_audio_rope(
    const std::vector<float> & positions,
    int64_t batch,
    int64_t tokens,
    const DramaBoxConfig & config,
    std::vector<float> & cos,
    std::vector<float> & sin);
void make_dramabox_audio_rope_repeated(
    const std::vector<float> & positions,
    int64_t repeat_count,
    int64_t tokens,
    const DramaBoxConfig & config,
    std::vector<float> & cos,
    std::vector<float> & sin);

}  // namespace engine::models::dramabox
