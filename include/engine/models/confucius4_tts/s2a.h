#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/audio_features.h"
#include "engine/models/confucius4_tts/style_encoder.h"
#include "engine/models/confucius4_tts/t2s.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusS2ASequence {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct ConfuciusS2AMel {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct ConfuciusS2AInputEmbeddingWeights {
    engine::core::TensorValue token_embedding;
    engine::modules::Conv1dWeights token_projection;
    engine::modules::LinearWeights encoder_projection;
    engine::core::TensorValue prompt_cond;
};

struct ConfuciusS2ALengthRegulatorWeights {
    engine::modules::LinearWeights content_projection;
    std::vector<engine::modules::Conv1dWeights> convs;
    std::vector<engine::modules::NormWeights> norms;
    engine::modules::Conv1dWeights output;
};

struct ConfuciusS2AAdaLayerNormWeights {
    engine::core::TensorValue norm_weight;
    engine::modules::LinearWeights project;
};

struct ConfuciusS2ADitLayerWeights {
    ConfuciusS2AAdaLayerNormWeights attention_norm;
    engine::modules::LinearWeights qkv;
    engine::modules::LinearWeights attention_out;
    ConfuciusS2AAdaLayerNormWeights ffn_norm;
    engine::modules::LinearWeights ffn_w1;
    engine::modules::LinearWeights ffn_w2;
    engine::modules::LinearWeights ffn_w3;
    engine::modules::LinearWeights skip_in;
};

struct ConfuciusS2AWavenetLayerWeights {
    engine::modules::Conv1dWeights in_layer;
    engine::modules::Conv1dWeights res_skip_layer;
};

struct ConfuciusS2ACfmWeights {
    engine::modules::LinearWeights input_mu_projection;
    engine::modules::LinearWeights input_projection;
    engine::modules::LinearWeights skip_linear;
    engine::core::TensorValue time_freqs;
    engine::modules::LinearWeights time_mlp0;
    engine::modules::LinearWeights time_mlp2;
    engine::core::TensorValue time2_freqs;
    engine::modules::LinearWeights time2_mlp0;
    engine::modules::LinearWeights time2_mlp2;
    std::vector<ConfuciusS2ADitLayerWeights> dit_layers;
    ConfuciusS2AAdaLayerNormWeights dit_norm;
    engine::modules::LinearWeights conv1;
    engine::modules::LinearWeights res_projection;
    engine::modules::Conv1dWeights wavenet_cond;
    std::vector<ConfuciusS2AWavenetLayerWeights> wavenet_layers;
    engine::modules::LinearWeights final_modulation;
    engine::modules::LinearWeights final_linear;
    engine::modules::Conv1dWeights conv2;
};

struct ConfuciusS2AWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    ConfuciusS2AInputEmbeddingWeights input_embedding;
    ConfuciusS2ALengthRegulatorWeights length_regulator;
    ConfuciusS2ACfmWeights cfm;
};

std::shared_ptr<const ConfuciusS2AWeights> load_confucius_s2a_weights(
    const ConfuciusAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class ConfuciusS2ARuntime {
public:
    ConfuciusS2ARuntime(
        std::shared_ptr<const ConfuciusAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~ConfuciusS2ARuntime();

    ConfuciusS2ARuntime(const ConfuciusS2ARuntime &) = delete;
    ConfuciusS2ARuntime & operator=(const ConfuciusS2ARuntime &) = delete;

    void prepare_condition(int64_t semantic_tokens, int64_t target_frames);
    void prepare_cfm(int64_t total_frames, bool use_cfg);
    ConfuciusS2AMel infer_mel(
        const ConfuciusT2SSemanticGeneration & generation,
        const ConfuciusMelOutput & reference_mel,
        const ConfuciusStyleEmbedding & style,
        int64_t target_frames,
        int64_t num_inference_steps,
        float guidance_scale,
        uint32_t seed,
        uint64_t & rng_offset_blocks);
    void release_pre_cfm_graphs();
    void release_cfm_graph();

private:
    class ConditionGraph;
    class CfmGraph;

    std::shared_ptr<const ConfuciusAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const ConfuciusS2AWeights> weights_;
    std::unique_ptr<ConditionGraph> condition_graph_;
    std::unique_ptr<CfmGraph> cfm_graph_;
};

}  // namespace engine::models::confucius4_tts
