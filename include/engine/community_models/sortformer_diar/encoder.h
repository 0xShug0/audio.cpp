#pragma once

#include "engine/community_models/sortformer_diar/assets.h"
#include "engine/community_models/sortformer_diar/frontend.h"
#include "engine/community_models/sortformer_diar/weights.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::sortformer_diar {

struct SortformerV2InferenceGraph {
    int64_t feature_frames = 0;
    int64_t state_frames = 0;
    int64_t chunk_encoder_frames = 0;
    int64_t encoder_frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_projection_graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int compute_threads = 1;

    core::TensorValue input;
    core::TensorValue state_input;
    core::TensorValue mask1;
    core::TensorValue mask2;
    core::TensorValue encoder_keep_mask;
    core::TensorValue pos_emb;
    core::TensorValue transformer_mask;
    std::vector<core::TensorValue> projected_pos_emb;
    std::vector<core::TensorValue> projected_pos_emb_computed;
    core::TensorValue chunk_embeddings;
    core::TensorValue encoder_output;
    core::TensorValue encoder_projection;
    core::TensorValue transformer_output;
    core::TensorValue output_probabilities;

    ~SortformerV2InferenceGraph();
};

int64_t sortformer_v2_conv_valid_length(int64_t valid, int64_t kernel, int64_t stride, int64_t padding);
void fill_sortformer_v2_keep_mask(std::vector<int32_t> & mask, int64_t frames, int64_t valid_frames);
void fill_sortformer_v2_transformer_attention_mask(
    std::vector<float> & mask,
    int64_t frames,
    int64_t valid_frames);

void ensure_sortformer_v2_inference_graph(
    std::unique_ptr<SortformerV2InferenceGraph> & graph,
    const core::ExecutionContext & execution_context,
    const SortformerV2Assets & assets,
    const SortformerV2Weights & weights,
    size_t graph_context_bytes,
    int64_t feature_frames,
    int64_t encoder_frames,
    int64_t state_frames = 0);

}  // namespace engine::community_models::sortformer_diar
