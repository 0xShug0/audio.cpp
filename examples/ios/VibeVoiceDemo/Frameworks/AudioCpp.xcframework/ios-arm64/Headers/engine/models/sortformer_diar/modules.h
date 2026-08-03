#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <vector>

namespace engine::models::sortformer_diar {

struct SortformerMaskedSelfAttentionWeights {
    modules::LinearWeights query;
    modules::LinearWeights key;
    modules::LinearWeights value;
    modules::LinearWeights output;
};

class SortformerMaskedSelfAttentionModule {
public:
    explicit SortformerMaskedSelfAttentionModule(modules::AttentionConfig config);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const SortformerMaskedSelfAttentionWeights & weights,
        const core::TensorValue & attention_mask) const;

private:
    modules::AttentionConfig config_;
};

struct SortformerTransformerBlockWeights {
    modules::NormWeights self_attention_norm;
    SortformerMaskedSelfAttentionWeights self_attention;
    modules::NormWeights feed_forward_norm;
    modules::LinearWeights feed_forward_fc1;
    modules::LinearWeights feed_forward_fc2;
};

class SortformerPostNormTransformerEncoderBlockModule {
public:
    SortformerPostNormTransformerEncoderBlockModule(
        int64_t hidden_size,
        int64_t num_heads,
        int64_t intermediate_size,
        float eps);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & attention_mask,
        const SortformerTransformerBlockWeights & weights) const;

private:
    int64_t hidden_size_ = 0;
    int64_t num_heads_ = 0;
    int64_t intermediate_size_ = 0;
    float eps_ = 1.0e-5f;
};

struct SortformerTransformerStackWeights {
    std::vector<SortformerTransformerBlockWeights> layers;
};

class SortformerPostNormTransformerStackModule {
public:
    SortformerPostNormTransformerStackModule(
        int64_t hidden_size,
        int64_t num_heads,
        int64_t intermediate_size,
        int64_t num_layers,
        float eps);

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & attention_mask,
        const SortformerTransformerStackWeights & weights) const;

private:
    int64_t hidden_size_ = 0;
    int64_t num_heads_ = 0;
    int64_t intermediate_size_ = 0;
    int64_t num_layers_ = 0;
    float eps_ = 1.0e-5f;
};

}  // namespace engine::models::sortformer_diar
