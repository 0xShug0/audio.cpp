#pragma once

#include "engine/models/ming3_audio/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <array>

namespace engine::models::ming3_audio {

struct ThinkerOptions {
    bool disk_experts = false;
    bool pin_experts = true;
    int64_t expert_cache_size = 8;
    int64_t disk_expert_read_threads = 1;
};

struct KdaWeights {
    modules::LinearWeights input_proj;
    std::array<core::TensorValue, 3> conv;
    modules::LinearWeights output;
    core::TensorValue a_log;
    core::TensorValue dt_bias;
    modules::NormWeights norm;
};

struct MlaWeights {
    modules::LinearWeights query_down;
    modules::LinearWeights query_up;
    modules::NormWeights query_norm;
    modules::LinearWeights kv_down;
    modules::LinearWeights kv_up;
    modules::NormWeights kv_norm;
    modules::LinearWeights gate;
    modules::LinearWeights output;
};

struct ThinkerLayerWeights {
    modules::NormWeights input_norm;
    modules::NormWeights post_norm;
    KdaWeights kda;
    MlaWeights mla;
    modules::LinearWeights mlp_gate_up;
    modules::LinearWeights mlp_down;
    core::TensorValue router;
    core::TensorValue router_bias;
    core::TensorValue expert_gate;
    core::TensorValue expert_up;
    core::TensorValue expert_down;
    core::DiskExpertWeightStore * disk_store = nullptr;
    size_t disk_group = 0;
};

struct ThinkerWeights {
    ThinkerWeights(core::ExecutionContext & execution, const assets::TensorSource & source,
                   const ThinkerConfig & config, const ThinkerOptions & options = {});

    core::BackendWeightStore store;
    std::unique_ptr<core::DiskExpertWeightStore> disk_store;
    core::TensorValue embedding;
    core::TensorValue head;
    core::TensorValue group_identity;
    modules::NormWeights norm;
    std::vector<ThinkerLayerWeights> layers;
};

struct KdaOutput {
    core::TensorValue hidden;
    std::array<core::TensorValue, 3> conv_tails;
    core::TensorValue recurrent_state;
};

core::TensorValue build_mla(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                            const core::TensorValue & positions, const core::TensorValue & attention_mask,
                            const core::TensorValue & key_cache, const core::TensorValue & value_cache,
                            const MlaWeights & weights, const ThinkerConfig & config);

KdaOutput build_kda(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                    const std::array<core::TensorValue, 3> & tails,
                    const core::TensorValue & state, const KdaWeights & weights,
                    const ThinkerConfig & config);

core::TensorValue build_thinker_mlp(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                   const ThinkerLayerWeights & weights, const ThinkerConfig & config,
                                   const core::TensorValue & group_identity, bool routed);

class Thinker {
public:
    Thinker(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
            ThinkerConfig config, int64_t capacity, ThinkerOptions options = {});
    ~Thinker();

    void reset();
    void reserve(int64_t capacity);
    void release_workspace();
    std::vector<float> prefill(const std::vector<int32_t> & ids,
                               const std::vector<float> & audio_embeddings = {},
                               int32_t audio_token_id = -1);
    std::vector<float> step(int32_t token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ming3_audio
