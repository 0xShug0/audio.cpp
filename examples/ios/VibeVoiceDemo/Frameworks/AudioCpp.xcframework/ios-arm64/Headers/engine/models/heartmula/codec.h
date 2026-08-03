#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/heartmula/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::heartmula {

class HeartCodecFlowEstimatorGraph;
class HeartCodecConditioningGraph;
class HeartCodecScalarDecoderGraph;
struct HeartMuLaGenerationOptions;

struct HeartCodecProjectLayerWeights {
    modules::Conv1dWeights ffn_1;
    modules::LinearWeights ffn_2;
};

struct HeartCodecAdaLayerNormWeights {
    modules::LinearWeights timestep_linear_1;
    modules::LinearWeights timestep_linear_2;
    modules::LinearWeights linear;
};

struct HeartCodecTransformerBlockWeights {
    modules::NormWeights attn_norm;
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights o_proj;
    modules::NormWeights mlp_norm;
    modules::LinearWeights mlp_gate;
    modules::LinearWeights mlp_up;
    modules::LinearWeights mlp_down;
    core::TensorValue scale_shift_table;
};

struct HeartCodecFlowEstimatorWeights {
    HeartCodecProjectLayerWeights proj_in;
    std::vector<HeartCodecTransformerBlockWeights> transformer_blocks;
    HeartCodecProjectLayerWeights connection_proj;
    std::vector<HeartCodecTransformerBlockWeights> transformer_blocks_2;
    core::TensorValue scale_shift_table;
    core::TensorValue scale_shift_table_2;
    HeartCodecProjectLayerWeights proj_out;
    HeartCodecAdaLayerNormWeights adaln_single;
    HeartCodecAdaLayerNormWeights adaln_single_2;
};

struct HeartCodecFlowWeights {
    modules::LinearWeights vq_project_out;
    std::vector<core::TensorValue> vq_codebooks;
    modules::LinearWeights cond_feature_emb;
    core::TensorValue zero_cond_embedding;
    assets::TensorDataF32 zero_cond_embedding_host;
    HeartCodecFlowEstimatorWeights estimator;
};

struct HeartCodecPreluWeights {
    core::TensorValue weight;
};

struct HeartCodecResidualUnitWeights {
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
    HeartCodecPreluWeights activation1;
    HeartCodecPreluWeights activation2;
};

struct HeartCodecDecoderBlockWeights {
    modules::ConvTranspose1dWeights up_conv;
    std::vector<HeartCodecResidualUnitWeights> residual_units;
};

struct HeartCodecPostProcessorWeights {
    modules::Conv1dWeights conv;
    HeartCodecPreluWeights activation;
};

struct HeartCodecScalarDecoderWeights {
    modules::Conv1dWeights input_conv;
    std::vector<HeartCodecDecoderBlockWeights> blocks;
    HeartCodecPostProcessorWeights post_processor;
    modules::Conv1dWeights output_conv;
};

struct HeartCodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    HeartCodecFlowWeights flow;
    HeartCodecScalarDecoderWeights scalar_decoder;
};

struct HeartCodecFlowEstimate {
    std::vector<float> values;
    int64_t batch_size = 0;
    int64_t frames = 0;
    int64_t channels = 0;
};

struct HeartCodecDecodedAudio {
    std::vector<float> values;
    int64_t batch_size = 0;
    int64_t channels = 0;
    int64_t samples = 0;
};

struct HeartCodecConditioning {
    std::vector<float> values;
    int64_t batch_size = 0;
    int64_t frames = 0;
    int64_t channels = 0;
};

class HeartCodecWeightsRuntime final {
public:
    HeartCodecWeightsRuntime(
        std::shared_ptr<const HeartMuLaAssets> assets,
        core::ExecutionContext & execution_context,
        size_t weight_context_bytes = 512ull * 1024ull * 1024ull,
        size_t flow_estimator_graph_arena_bytes = 2048ull * 1024ull * 1024ull,
        size_t conditioning_graph_arena_bytes = 512ull * 1024ull * 1024ull,
        size_t scalar_decoder_graph_arena_bytes = 1536ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);
    ~HeartCodecWeightsRuntime();

    HeartCodecWeightsRuntime(const HeartCodecWeightsRuntime &) = delete;
    HeartCodecWeightsRuntime & operator=(const HeartCodecWeightsRuntime &) = delete;

    const HeartMuLaAssets & assets() const noexcept;
    const HeartCodecWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    int device() const noexcept;
    int threads() const noexcept;
    size_t flow_estimator_graph_arena_bytes() const noexcept;
    size_t conditioning_graph_arena_bytes() const noexcept;
    size_t scalar_decoder_graph_arena_bytes() const noexcept;

    HeartCodecFlowEstimate estimate_flow(
        const std::vector<float> & hidden_states,
        int64_t batch_size,
        int64_t frames,
        const std::vector<float> & timesteps) const;
    HeartCodecDecodedAudio decode_scalar_latents(
        const std::vector<float> & latents,
        int64_t batch_size,
        int64_t frames) const;
    HeartCodecConditioning build_conditioning(
        const std::vector<int32_t> & codes,
        int64_t batch_size,
        int64_t code_frames) const;
    HeartCodecDecodedAudio detokenize_codes(
        const std::vector<int32_t> & codes,
        int64_t frames,
        int64_t codebooks,
        const HeartMuLaGenerationOptions & options,
        uint64_t seed,
        uint64_t randn_philox_offset = 0,
        uint64_t randn_call_offset_blocks = 1) const;
    void clear_graph_cache() const;

private:
    std::shared_ptr<const HeartMuLaAssets> assets_;
    std::shared_ptr<const HeartCodecWeights> weights_;
    mutable std::unique_ptr<HeartCodecFlowEstimatorGraph> flow_estimator_graph_;
    mutable std::unique_ptr<HeartCodecConditioningGraph> conditioning_graph_;
    mutable std::unique_ptr<HeartCodecScalarDecoderGraph> scalar_decoder_graph_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t flow_estimator_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t conditioning_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t scalar_decoder_graph_arena_bytes_ = 1536ull * 1024ull * 1024ull;
};

HeartCodecWeights load_heartcodec_weights(
    const HeartMuLaAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

}  // namespace engine::models::heartmula
