#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/community_models/granite5asr/assets.h"
#include "engine/community_models/granite5asr/frontend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::granite5asr {

struct Granite5LayerWeights {
    modules::NormWeights ffn1_norm;
    modules::LinearWeights ffn1_fc1;
    modules::LinearWeights ffn1_fc2;

    modules::NormWeights norm_self_att;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    modules::LinearWeights o_proj;
    core::TensorValue rel_pos_emb;

    modules::NormWeights norm_conv;
    modules::LinearWeights conv_pw1;
    core::TensorValue conv_dw_weight;
    core::TensorValue conv_dw_bias;
    modules::LinearWeights conv_pw2;

    modules::NormWeights ffn2_norm;
    modules::LinearWeights ffn2_fc1;
    modules::LinearWeights ffn2_fc2;

    modules::NormWeights norm_out;

    bool is_subsample = false;
};

struct Granite5EncoderWeights {
    modules::LinearWeights input_linear;
    std::vector<Granite5LayerWeights> layers;
    modules::LinearWeights out;
    modules::LinearWeights out_mid;
};

class Granite5EncoderRuntime {
public:
    Granite5EncoderRuntime(
        std::shared_ptr<const Granite5ASRAssets> assets,
        engine::core::ExecutionContext & execution_context,
        assets::TensorStorageType storage_type,
        size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull);

    std::vector<int32_t> transcribe_features(
        const Granite5FrontendFeatures & features);

    const Granite5ASRAssets & assets() const noexcept { return *assets_; }

private:
    std::shared_ptr<const Granite5ASRAssets> assets_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    engine::core::BackendWeightStore weight_store_;
    Granite5EncoderWeights weights_;
    size_t graph_arena_bytes_;
};

}  // namespace engine::community_models::granite5asr
