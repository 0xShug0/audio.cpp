#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/index_tts2_5/assets.h"
#include "engine/models/index_tts2_5/audio_features.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25Wav2Vec2BertAttentionWeights {
    engine::modules::LinearWeights q;
    engine::modules::LinearWeights k;
    engine::modules::LinearWeights v;
    engine::modules::LinearWeights out;
    engine::core::TensorValue distance_embedding;
};

struct IndexTTS25Wav2Vec2BertConvWeights {
    engine::modules::NormWeights layer_norm;
    engine::modules::Conv1dWeights pointwise_in;
    engine::modules::DepthwiseConv1dWeights depthwise;
    engine::modules::NormWeights depthwise_layer_norm;
    engine::modules::Conv1dWeights pointwise_out;
};

struct IndexTTS25Wav2Vec2BertLayerWeights {
    engine::modules::NormWeights ffn1_norm;
    engine::modules::LinearWeights ffn1_in;
    engine::modules::LinearWeights ffn1_out;
    engine::modules::NormWeights self_attn_norm;
    IndexTTS25Wav2Vec2BertAttentionWeights self_attn;
    IndexTTS25Wav2Vec2BertConvWeights conv;
    engine::modules::NormWeights ffn2_norm;
    engine::modules::LinearWeights ffn2_in;
    engine::modules::LinearWeights ffn2_out;
    engine::modules::NormWeights final_norm;
};

struct IndexTTS25Wav2Vec2BertWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::NormWeights feature_norm;
    engine::modules::LinearWeights feature_projection;
    std::vector<IndexTTS25Wav2Vec2BertLayerWeights> layers;
    engine::core::TensorValue semantic_mean;
    engine::core::TensorValue semantic_std;
};

struct IndexTTS25SemanticEmbedding {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

std::shared_ptr<const IndexTTS25Wav2Vec2BertWeights> load_index_tts2_5_wav2vec2bert_weights(
    const IndexTTS25Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class IndexTTS25Wav2Vec2BertRuntime {
public:
    IndexTTS25Wav2Vec2BertRuntime(
        std::shared_ptr<const IndexTTS25Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~IndexTTS25Wav2Vec2BertRuntime();

    IndexTTS25Wav2Vec2BertRuntime(const IndexTTS25Wav2Vec2BertRuntime &) = delete;
    IndexTTS25Wav2Vec2BertRuntime & operator=(const IndexTTS25Wav2Vec2BertRuntime &) = delete;

    void prepare(int64_t frames);
    IndexTTS25SemanticEmbedding encode(const IndexTTS25SemanticFeatureOutput & features);
    void release_graph();

private:
    class Graph;

    std::shared_ptr<const IndexTTS25Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS25Wav2Vec2BertWeights> weights_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::index_tts2_5
