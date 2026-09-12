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

struct XttsV2GptPrefillResult {
    std::vector<float> logits;
    std::vector<float> latent;
};

struct XttsV2GptGeneration {
    std::vector<int32_t> codes;
    std::vector<float> latents;  // frame-major [codes, 1024]
};

std::shared_ptr<const XttsV2GptWeights> load_xtts_v2_gpt_weights(
    const XttsV2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type);

class XttsV2GptRuntime {
public:
    XttsV2GptRuntime(
        const XttsV2Assets & assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        assets::TensorStorageType storage_type);
    ~XttsV2GptRuntime();

    XttsV2GptPrefillResult prefill(
        const std::vector<float> & conditioning_latent,
        const std::vector<int32_t> & text_tokens);
    XttsV2GptGeneration generate(
        const std::vector<float> & conditioning_latent,
        const std::vector<int32_t> & text_tokens,
        const XttsV2GenerationOptions & options);

private:
    class PrefillGraph;
    core::ExecutionContext & execution_;
    size_t graph_context_bytes_;
    std::shared_ptr<const XttsV2GptWeights> weights_;
    std::unique_ptr<PrefillGraph> prefill_;
};

}  // namespace engine::models::xtts_v2
