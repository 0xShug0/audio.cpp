#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/audio_features.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusWav2Vec2BertAttentionWeights {
    engine::modules::LinearWeights q;
    engine::modules::LinearWeights k;
    engine::modules::LinearWeights v;
    engine::modules::LinearWeights out;
    engine::core::TensorValue distance_embedding;
};

struct ConfuciusWav2Vec2BertConvWeights {
    engine::modules::NormWeights layer_norm;
    engine::modules::Conv1dWeights pointwise_in;
    engine::modules::DepthwiseConv1dWeights depthwise;
    engine::modules::NormWeights depthwise_layer_norm;
    engine::modules::Conv1dWeights pointwise_out;
};

struct ConfuciusWav2Vec2BertLayerWeights {
    engine::modules::NormWeights ffn1_norm;
    engine::modules::LinearWeights ffn1_in;
    engine::modules::LinearWeights ffn1_out;
    engine::modules::NormWeights self_attn_norm;
    ConfuciusWav2Vec2BertAttentionWeights self_attn;
    ConfuciusWav2Vec2BertConvWeights conv;
    engine::modules::NormWeights ffn2_norm;
    engine::modules::LinearWeights ffn2_in;
    engine::modules::LinearWeights ffn2_out;
    engine::modules::NormWeights final_norm;
};

struct ConfuciusWav2Vec2BertWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::NormWeights feature_norm;
    engine::modules::LinearWeights feature_projection;
    std::vector<ConfuciusWav2Vec2BertLayerWeights> layers;
    engine::core::TensorValue semantic_mean;
    engine::core::TensorValue semantic_std;
};

struct ConfuciusSemanticEmbedding {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

std::shared_ptr<const ConfuciusWav2Vec2BertWeights> load_confucius_wav2vec2bert_weights(
    const ConfuciusAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class ConfuciusWav2Vec2BertRuntime {
public:
    ConfuciusWav2Vec2BertRuntime(
        std::shared_ptr<const ConfuciusAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~ConfuciusWav2Vec2BertRuntime();

    ConfuciusWav2Vec2BertRuntime(const ConfuciusWav2Vec2BertRuntime &) = delete;
    ConfuciusWav2Vec2BertRuntime & operator=(const ConfuciusWav2Vec2BertRuntime &) = delete;

    void prepare(int64_t frames);
    ConfuciusSemanticEmbedding encode(const ConfuciusSemanticFeatureOutput & features);
    void release_graph();

private:
    class Graph;

    std::shared_ptr<const ConfuciusAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const ConfuciusWav2Vec2BertWeights> weights_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::confucius4_tts
