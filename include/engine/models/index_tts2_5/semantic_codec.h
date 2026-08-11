#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/index_tts2_5/assets.h"
#include "engine/models/index_tts2_5/semantic_encoder.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25VocosConvNeXtBlockWeights {
    engine::modules::DepthwiseConv1dWeights depthwise;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pointwise_in;
    engine::modules::LinearWeights pointwise_out;
    engine::core::TensorValue gamma;
};

struct IndexTTS25VocosBackboneWeights {
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<IndexTTS25VocosConvNeXtBlockWeights> blocks;
    engine::modules::NormWeights final_norm;
};

struct IndexTTS25SemanticCodecWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    IndexTTS25VocosBackboneWeights encoder_backbone;
    engine::modules::LinearWeights encoder_projection;
    engine::modules::Conv1dWeights quantizer_in;
    engine::core::TensorValue codebook;
    engine::core::TensorValue normalized_codebook;
    engine::modules::Conv1dWeights quantizer_out;
    IndexTTS25VocosBackboneWeights decoder_backbone;
    engine::modules::LinearWeights decoder_projection;
    engine::modules::Conv1dWeights up;
};

struct IndexTTS25SemanticCodecOutput {
    std::vector<int32_t> codes;
    std::vector<float> embedding_channel_first;
    int64_t frames = 0;
    int64_t dims = 0;
};

std::shared_ptr<const IndexTTS25SemanticCodecWeights> load_index_tts2_5_semantic_codec_weights(
    const IndexTTS25Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class IndexTTS25SemanticCodecRuntime {
public:
    IndexTTS25SemanticCodecRuntime(
        std::shared_ptr<const IndexTTS25Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~IndexTTS25SemanticCodecRuntime();

    IndexTTS25SemanticCodecRuntime(const IndexTTS25SemanticCodecRuntime &) = delete;
    IndexTTS25SemanticCodecRuntime & operator=(const IndexTTS25SemanticCodecRuntime &) = delete;

    void prepare_quantize(int64_t frames);
    void prepare_codes(int64_t frames);
    IndexTTS25SemanticCodecOutput quantize(const IndexTTS25SemanticEmbedding & semantic);
    IndexTTS25SemanticCodecOutput codes_to_embedding(const std::vector<int32_t> & codes, int64_t frames);
    void release_graphs();

private:
    class QuantizeGraph;
    class CodesGraph;

    std::shared_ptr<const IndexTTS25Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS25SemanticCodecWeights> weights_;
    std::unique_ptr<QuantizeGraph> quantize_graph_;
    std::unique_ptr<CodesGraph> codes_graph_;
};

}  // namespace engine::models::index_tts2_5
