#pragma once

#include "engine/models/niagara_asr/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <memory>
#include <vector>

namespace engine::models::niagara_asr {

struct NiagaraSubsamplingWeights {
    engine::modules::BatchNorm2dEvalWeights bn0;
    engine::modules::BatchNorm2dEvalWeights bn1;
    engine::modules::Conv2dWeights input_projection;
    engine::modules::Conv2dWeights conv1;
    engine::modules::Conv2dWeights conv2;
    engine::modules::LinearWeights dense;
};

struct NiagaraSelfAttentionWeights {
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights query;
    engine::modules::LinearWeights key;
    engine::modules::LinearWeights value;
    engine::modules::LinearWeights global_attention;
    engine::modules::LinearWeights out;
    engine::core::TensorValue relative_position;
    engine::core::TensorValue u_bias;
    engine::core::TensorValue v_bias;
};

struct NiagaraFeedForwardWeights {
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights dense1;
    engine::modules::LinearWeights dense2;
    engine::core::TensorValue residual_scale;
};

struct NiagaraStateSpaceWeights {
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights dense1;
    engine::modules::LinearWeights dense2;
    engine::core::TensorValue conv_kernel;
};

struct NiagaraLayerWeights {
    NiagaraFeedForwardWeights ffn;
    NiagaraSelfAttentionWeights self_attention;
    NiagaraStateSpaceWeights state_space;
    NiagaraFeedForwardWeights ffn1;
    engine::modules::NormWeights out_norm;
};

struct NiagaraWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    NiagaraSubsamplingWeights subsampling;
    std::vector<NiagaraLayerWeights> layers;
    engine::modules::LinearWeights ctc;
};

std::shared_ptr<const NiagaraWeights> load_niagara_weights(
    const NiagaraAsrAssets & assets,
    engine::core::ExecutionContext & execution,
    engine::assets::TensorStorageType storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::niagara_asr
