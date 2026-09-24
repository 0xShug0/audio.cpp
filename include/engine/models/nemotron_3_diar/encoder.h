#pragma once

#include "engine/models/nemotron_3_diar/assets.h"
#include "engine/framework/core/execution_context.h"
#include "ggml-alloc.h"

#include <memory>
#include <vector>

namespace engine::models::nemotron_3_diar {

struct PreEncodeGraph {
    int64_t batch = 0;
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_gallocr_t allocator = nullptr;
    int threads = 1;
    core::TensorValue input;
    core::TensorValue output;
    ~PreEncodeGraph();
};

struct EncoderGraph {
    int64_t batch = 0;
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_gallocr_t allocator = nullptr;
    int threads = 1;
    core::TensorValue input;
    core::TensorValue attention_mask;
    core::TensorValue frame_mask;
    core::TensorValue rope_cos;
    core::TensorValue rope_sin;
    core::TensorValue probabilities;
    ~EncoderGraph();
};

void ensure_pre_encode_graph(
    std::unique_ptr<PreEncodeGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames);

void ensure_encoder_graph(
    std::unique_ptr<EncoderGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames,
    bool use_flash_attention);

std::vector<float> rope_table(int64_t batch, int64_t heads, int64_t frames, int64_t head_dim, float theta, bool cosine);
std::vector<float> attention_mask(const std::vector<int64_t> & lengths, int64_t frames);
std::vector<float> frame_mask(const std::vector<int64_t> & lengths, int64_t frames);

}  // namespace engine::models::nemotron_3_diar
