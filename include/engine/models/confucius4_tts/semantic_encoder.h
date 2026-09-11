#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/wav2vec2_bert_encoder.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/audio_features.h"

#include <memory>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusSemanticEmbedding {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

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
    engine::modules::Wav2Vec2BertEncoderComponent component_;
};

}  // namespace engine::models::confucius4_tts
