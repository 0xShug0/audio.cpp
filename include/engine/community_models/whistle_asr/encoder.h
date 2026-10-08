#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/frontend.h"
#include "engine/community_models/whistle_asr/graph.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace engine::community_models::whistle_asr {

struct WhistleEncoderLayerWeights {
    WhistleMhcWeights mhc;
    modules::NormWeights pre_hada_norm_0;
    WhistleHadamardWeights hadamard_0;
    modules::NormWeights attn_norm;
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::NormWeights q_norm;
    modules::NormWeights k_norm;
    modules::LinearWeights gate_proj;
    modules::LinearWeights out_proj;
    modules::NormWeights post_attn_norm;
    float attn_gate = 0.0f;
    modules::NormWeights conv_norm;
    modules::LinearWeights pw1;
    modules::DepthwiseConv1dWeights dw;
    modules::NormWeights conv_out_norm;
    modules::LinearWeights pw2;
    modules::NormWeights pre_hada_norm;
    WhistleHadamardWeights hadamard;
};

struct WhistleCrossProjectionWeights {
    modules::LinearWeights k_proj;
    modules::NormWeights k_norm;
    modules::LinearWeights v_proj;
};

struct WhistleEncoderWeights {
    modules::Conv2dWeights stem_conv;
    modules::Conv2dWeights stem_dw_1;
    modules::Conv2dWeights stem_pw_1;
    modules::Conv2dWeights stem_dw_2;
    modules::Conv2dWeights stem_pw_2;
    modules::LinearWeights stem_out;
    std::array<WhistleEncoderLayerWeights, kWhistleLayers> layers;
    modules::NormWeights final_norm;
    core::TensorValue positional;
    std::array<WhistleCrossProjectionWeights, kWhistleLayers> cross;
    WhistleGraphConstants constants;
};

struct WhistleEncoderOutput {
    size_t frames = 0;
    std::vector<float> memory;
    std::array<std::vector<float>, kWhistleLayers> cross_k;
    std::array<std::vector<float>, kWhistleLayers> cross_v;
};

class WhistleEncoderRuntime {
public:
    WhistleEncoderRuntime(
        std::shared_ptr<const WhistleAssets> assets,
        core::ExecutionContext & execution_context);

    [[nodiscard]] WhistleEncoderOutput encode(const MelFeatures & mel);

private:
    std::shared_ptr<const WhistleAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    core::BackendWeightStore weight_store_;
    WhistleEncoderWeights weights_;
};

}  // namespace engine::community_models::whistle_asr
