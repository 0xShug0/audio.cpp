#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/xtts_v2/assets.h"

#include <memory>
#include <vector>

namespace engine::models::xtts_v2 {

struct XttsV2GptLayerWeights {
    modules::NormWeights attn_norm;
    modules::LinearWeights qkv;
    modules::LinearWeights attn_out;
    modules::NormWeights mlp_norm;
    modules::LinearWeights mlp_in;
    modules::LinearWeights mlp_out;
};

struct XttsV2GptWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue audio_embedding;
    core::TensorValue text_positions;
    core::TensorValue audio_positions;
    std::vector<XttsV2GptLayerWeights> layers;
    modules::NormWeights transformer_norm;
    modules::NormWeights output_norm;
    modules::LinearWeights audio_head;
};

std::shared_ptr<const XttsV2GptWeights> load_xtts_v2_gpt_weights(
    const XttsV2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type);

}  // namespace engine::models::xtts_v2
