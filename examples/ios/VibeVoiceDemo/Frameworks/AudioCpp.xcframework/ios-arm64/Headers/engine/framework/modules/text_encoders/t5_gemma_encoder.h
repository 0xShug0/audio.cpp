#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"

#include <vector>

namespace engine::modules {

struct T5GemmaEncoderConfig {
    int64_t hidden_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    float rope_theta = 10000.0F;
    float rms_norm_eps = 1.0e-6F;
    float attn_logit_softcap = 50.0F;
    float query_pre_attn_scalar = 64.0F;
    bool scale_embeddings = true;
};

struct T5GemmaEncoderLayerWeights {
    core::TensorValue pre_self_attn_norm;
    core::TensorValue post_self_attn_norm;
    core::TensorValue pre_ff_norm;
    core::TensorValue post_ff_norm;
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
    LinearWeights gate_proj;
    LinearWeights up_proj;
    LinearWeights down_proj;
};

struct T5GemmaEncoderWeights {
    core::TensorValue embed_tokens;
    std::vector<T5GemmaEncoderLayerWeights> layers;
    core::TensorValue norm;
};

class T5GemmaEncoderModule {
public:
    explicit T5GemmaEncoderModule(T5GemmaEncoderConfig config);

    const T5GemmaEncoderConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input_ids,
        const core::TensorValue & positions,
        const core::TensorValue & additive_attention_mask,
        const T5GemmaEncoderWeights & weights) const;

private:
    T5GemmaEncoderConfig config_;
};

}  // namespace engine::modules
