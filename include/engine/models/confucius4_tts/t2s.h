#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/semantic_encoder.h"
#include "engine/models/confucius4_tts/types.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusT2SSemanticGeneration {
    std::vector<int32_t> semantic_codes;
    std::vector<float> latent;
    int64_t frames = 0;
    int64_t dims = 0;
    uint64_t rng_offset_blocks = 0;
};

struct ConfuciusT2SGenerationRequest {
    std::vector<int32_t> text_tokens;
    ConfuciusSemanticEmbedding semantic_condition;
    ConfuciusGenerationOptions options;
    uint64_t rng_offset_blocks = 0;
};

struct ConfuciusT2SLayerWeights {
    engine::modules::NormWeights attn_norm;
    engine::modules::LinearWeights qkv;
    engine::modules::LinearWeights attn_out;
    engine::modules::NormWeights mlp_norm;
    engine::modules::LinearWeights mlp_in;
    engine::modules::LinearWeights mlp_out;
};

struct ConfuciusT2SConvWeights {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t dilation = 1;
    engine::modules::Conv1dWeights weights;
};

struct ConfuciusT2SSERes2NetWeights {
    ConfuciusT2SConvWeights tdnn1;
    std::vector<ConfuciusT2SConvWeights> res2net;
    ConfuciusT2SConvWeights tdnn2;
    ConfuciusT2SConvWeights se_conv1;
    ConfuciusT2SConvWeights se_conv2;
};

struct ConfuciusT2SSpeakerWeights {
    ConfuciusT2SConvWeights block0;
    std::vector<ConfuciusT2SSERes2NetWeights> blocks;
    ConfuciusT2SConvWeights mfa;
    ConfuciusT2SConvWeights asp_tdnn;
    ConfuciusT2SConvWeights asp_conv;
    ConfuciusT2SConvWeights fc;
};

struct ConfuciusT2SWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue text_embedding;
    engine::modules::LinearWeights text_fc1;
    engine::modules::LinearWeights text_fc2;
    engine::core::TensorValue semantic_embedding;
    engine::core::TensorValue text_pos_embedding;
    engine::core::TensorValue semantic_pos_embedding;
    std::vector<ConfuciusT2SLayerWeights> layers;
    engine::modules::NormWeights gpt_final_norm;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights semantic_head;
    ConfuciusT2SSpeakerWeights speaker;
};

std::shared_ptr<const ConfuciusT2SWeights> load_confucius_t2s_weights(
    const ConfuciusAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class ConfuciusT2SRuntime {
public:
    ConfuciusT2SRuntime(
        std::shared_ptr<const ConfuciusAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~ConfuciusT2SRuntime();

    ConfuciusT2SRuntime(const ConfuciusT2SRuntime &) = delete;
    ConfuciusT2SRuntime & operator=(const ConfuciusT2SRuntime &) = delete;

    void prepare_generation(int64_t text_tokens, int64_t max_semantic_tokens, int64_t num_beams);
    ConfuciusT2SSemanticGeneration generate(const ConfuciusT2SGenerationRequest & request);
    void release_graphs();

private:
    class SpeakerConditionGraph;
    class PrefillGraph;
    class DecodeGraph;
    class ForwardGraph;

    std::shared_ptr<const ConfuciusAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const ConfuciusT2SWeights> weights_;
    std::unique_ptr<SpeakerConditionGraph> speaker_graph_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
    std::unique_ptr<ForwardGraph> forward_graph_;
};

}  // namespace engine::models::confucius4_tts
