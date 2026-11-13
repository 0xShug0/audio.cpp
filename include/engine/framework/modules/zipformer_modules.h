#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <functional>
#include <vector>

namespace engine::modules {

using ZipformerConstantFactory =
    std::function<core::TensorValue(const core::TensorShape &, std::vector<float>)>;

struct ZipformerLinearWeights {
    core::TensorValue weight;
    core::TensorValue bias;
};

enum class ZipformerAttentionImpl { Default, Batched };

struct ZipformerRelativeAttentionConfig {
    int64_t num_heads = 0;
    int64_t query_head_dim = 0;
    int64_t position_dim = 48;
    int64_t position_head_dim = 4;
    ZipformerAttentionImpl impl = ZipformerAttentionImpl::Default;
};

struct ZipformerRelativeAttentionWeights {
    ZipformerLinearWeights in_proj;
    std::vector<float> linear_pos_weight;
};

struct ZipformerRelativeAttentionOutputs {
    std::vector<core::TensorValue> weights;
    core::TensorValue key_state;
    core::TensorValue batched_weights;  // [heads, query frames, source frames]
};

class ZipformerRelativeAttentionModule {
public:
    explicit ZipformerRelativeAttentionModule(ZipformerRelativeAttentionConfig config);

    // An empty cached_key selects full-sequence attention; key_state is then empty.
    // An empty padding_mask selects unmasked attention.
    ZipformerRelativeAttentionOutputs build(
        core::ModuleBuildContext & ctx,
        ZipformerConstantFactory & constant_factory,
        const core::TensorValue & input,
        const core::TensorValue & cached_key,
        const core::TensorValue & padding_mask,
        const ZipformerRelativeAttentionWeights & weights) const;

private:
    ZipformerRelativeAttentionOutputs build_default(
        core::ModuleBuildContext & ctx,
        ZipformerConstantFactory & constant_factory,
        const core::TensorValue & input,
        const core::TensorValue & cached_key,
        const core::TensorValue & padding_mask,
        const ZipformerRelativeAttentionWeights & weights) const;

    ZipformerRelativeAttentionOutputs build_batched(
        core::ModuleBuildContext & ctx,
        ZipformerConstantFactory & constant_factory,
        const core::TensorValue & input,
        const core::TensorValue & cached_key,
        const core::TensorValue & padding_mask,
        const ZipformerRelativeAttentionWeights & weights) const;

    ZipformerRelativeAttentionConfig config_;
};

struct ZipformerNonlinearAttentionWeights {
    ZipformerLinearWeights in_proj;
    ZipformerLinearWeights out_proj;
};

struct ZipformerStatefulOutputs {
    core::TensorValue output;
    core::TensorValue state;
};

class ZipformerNonlinearAttentionModule {
public:
    // An empty cached_value selects full-sequence attention without persistent state.
    ZipformerStatefulOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & attention,
        const core::TensorValue & cached_value,
        const ZipformerNonlinearAttentionWeights & weights) const;
};

struct ZipformerConvolutionConfig {
    int64_t kernel_size = 0;
};

struct ZipformerConvolutionWeights {
    ZipformerLinearWeights in_proj;
    DepthwiseConv1dWeights causal_conv;
    DepthwiseConv1dWeights chunkwise_conv;
    std::vector<float> chunk_scale_left;
    std::vector<float> chunk_scale_right;
    ZipformerLinearWeights out_proj;
};

class ZipformerConvolutionModule {
public:
    explicit ZipformerConvolutionModule(ZipformerConvolutionConfig config);

    ZipformerStatefulOutputs build(
        core::ModuleBuildContext & ctx,
        ZipformerConstantFactory & constant_factory,
        const core::TensorValue & input,
        const core::TensorValue & cached_value,
        const ZipformerConvolutionWeights & weights) const;

private:
    ZipformerConvolutionConfig config_;
};

}  // namespace engine::modules
