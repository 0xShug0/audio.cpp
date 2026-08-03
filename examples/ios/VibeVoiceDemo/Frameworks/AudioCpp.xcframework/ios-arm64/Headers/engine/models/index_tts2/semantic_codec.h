#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/index_tts2/assets.h"
#include "engine/models/index_tts2/semantic_encoder.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2VocosConvNeXtBlockWeights {
    engine::modules::DepthwiseConv1dWeights depthwise;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pointwise_in;
    engine::modules::LinearWeights pointwise_out;
    engine::core::TensorValue gamma;
};

struct IndexTTS2VocosBackboneWeights {
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<IndexTTS2VocosConvNeXtBlockWeights> blocks;
    engine::modules::NormWeights final_norm;
};

struct IndexTTS2SemanticCodecWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    IndexTTS2VocosBackboneWeights encoder_backbone;
    engine::modules::LinearWeights encoder_projection;
    engine::modules::Conv1dWeights quantizer_in;
    engine::core::TensorValue codebook;
    engine::core::TensorValue normalized_codebook;
    engine::modules::Conv1dWeights quantizer_out;
    IndexTTS2VocosBackboneWeights decoder_backbone;
    engine::modules::LinearWeights decoder_projection;
};

struct IndexTTS2SemanticCodecOutput {
    std::vector<int32_t> codes;
    std::vector<float> embedding_channel_first;
    int64_t frames = 0;
    int64_t dims = 0;
};

std::shared_ptr<const IndexTTS2SemanticCodecWeights> load_index_tts2_semantic_codec_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class IndexTTS2SemanticCodecRuntime {
public:
    IndexTTS2SemanticCodecRuntime(
        std::shared_ptr<const IndexTTS2Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~IndexTTS2SemanticCodecRuntime();

    IndexTTS2SemanticCodecRuntime(const IndexTTS2SemanticCodecRuntime &) = delete;
    IndexTTS2SemanticCodecRuntime & operator=(const IndexTTS2SemanticCodecRuntime &) = delete;

    void prepare_quantize(int64_t frames);
    void prepare_codes(int64_t frames);
    IndexTTS2SemanticCodecOutput quantize(const IndexTTS2SemanticEmbedding & semantic);
    IndexTTS2SemanticCodecOutput codes_to_embedding(const std::vector<int32_t> & codes, int64_t frames);
    void release_graphs();

private:
    class QuantizeGraph;
    class CodesGraph;

    std::shared_ptr<const IndexTTS2Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS2SemanticCodecWeights> weights_;
    std::unique_ptr<QuantizeGraph> quantize_graph_;
    std::unique_ptr<CodesGraph> codes_graph_;
};

}  // namespace engine::models::index_tts2
