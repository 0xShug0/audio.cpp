#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::controlfoley {

struct ControlFoleyFlowConfig {
    bool v2 = false;
    int64_t latent_dim = 40;
    int64_t clip_dim = 1024;
    int64_t visual_dim = 768;
    int64_t sync_dim = 768;
    int64_t text_dim = 1024;
    int64_t audio_dim = 512;
    int64_t timbre_dim = 1536;
    int64_t hidden_dim = 896;
    int64_t depth = 54;
    int64_t fused_depth = 36;
    int64_t num_heads = 14;
    float mlp_ratio = 4.0F;
    int64_t latent_seq_len = 345;
    int64_t clip_seq_len = 64;
    int64_t visual_seq_len = 32;
    int64_t sync_seq_len = 192;
    int64_t text_seq_len = 77;
    int64_t audio_seq_len = 1;
    int64_t timbre_seq_len = 1;
    float layer_norm_eps = 1.0e-5F;
    float rms_norm_eps = 1.1920928955078125e-7F;
    float rope_theta = 10000.0F;
};

struct ControlFoleyFlowRuntimeOptions {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    size_t weight_context_bytes = 6144ull * 1024ull * 1024ull;
    size_t condition_graph_arena_bytes = 512ull * 1024ull * 1024ull;
    size_t flow_graph_arena_bytes = 2048ull * 1024ull * 1024ull;
    size_t graph_nodes = 1048576;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
};

struct ControlFoleyFlowConditionInput {
    int64_t batch = 0;
    std::vector<float> clip;
    std::vector<float> visual;
    std::vector<float> sync;
    std::vector<float> text;
    std::vector<float> audio;
    std::vector<float> timbre;
};

struct ControlFoleyFlowConditions {
    int64_t batch = 0;
    std::vector<float> clip;
    std::vector<float> sync;
    std::vector<float> text;
    std::vector<float> audio;
    std::vector<float> timbre;
    std::vector<float> clip_cond;
    std::vector<float> text_cond;
};

struct ControlFoleyFlowPrediction {
    int64_t batch = 0;
    int64_t latent_tokens = 0;
    int64_t latent_dim = 0;
    std::vector<float> flow;
    std::vector<float> multimodal;
    std::vector<float> hidden;
};

class ControlFoleyFlowDenoiserRuntime {
public:
    ControlFoleyFlowDenoiserRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        ControlFoleyFlowConfig config = {},
        ControlFoleyFlowRuntimeOptions options = {});
    ~ControlFoleyFlowDenoiserRuntime();

    ControlFoleyFlowDenoiserRuntime(const ControlFoleyFlowDenoiserRuntime &) = delete;
    ControlFoleyFlowDenoiserRuntime & operator=(const ControlFoleyFlowDenoiserRuntime &) = delete;

    ControlFoleyFlowConditions preprocess_conditions(const ControlFoleyFlowConditionInput & input);

    void update_config(ControlFoleyFlowConfig config);

    ControlFoleyFlowPrediction predict_flow(
        const std::vector<float> & latent,
        const std::vector<float> & timesteps,
        const ControlFoleyFlowConditions & conditions);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::controlfoley
