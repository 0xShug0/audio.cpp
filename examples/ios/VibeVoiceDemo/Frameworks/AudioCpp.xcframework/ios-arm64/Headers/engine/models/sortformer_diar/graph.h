#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/modules.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::sortformer_diar {

struct SortformerInferenceGraph {
    int64_t feature_frames = 0;
    int64_t encoder_frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_projection_graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int compute_threads = 1;

    core::TensorValue input;
    core::TensorValue mask1;
    core::TensorValue mask2;
    core::TensorValue encoder_keep_mask;
    core::TensorValue pos_emb;
    core::TensorValue transformer_mask;
    std::vector<core::TensorValue> projected_pos_emb;
    std::vector<core::TensorValue> projected_pos_emb_computed;
    core::TensorValue output_probabilities;

    ~SortformerInferenceGraph();
};

int64_t sortformer_conv_valid_length(int64_t valid, int64_t kernel, int64_t stride, int64_t padding);

void ensure_sortformer_inference_graph(
    std::unique_ptr<SortformerInferenceGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerAssets & assets,
    const SortformerDiarWeights & weights,
    size_t graph_context_bytes,
    int64_t feature_frames,
    int64_t encoder_frames);

void fill_sortformer_keep_mask(std::vector<int32_t> & mask, int64_t frames, int64_t valid_frames);

void fill_sortformer_transformer_attention_mask(
    std::vector<float> & mask,
    int64_t frames,
    int64_t valid_frames);

}  // namespace engine::models::sortformer_diar
