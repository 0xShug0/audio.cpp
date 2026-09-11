#include "engine/models/confucius4_tts/semantic_encoder.h"

#include <stdexcept>
#include <utility>

namespace engine::models::confucius4_tts {

ConfuciusWav2Vec2BertRuntime::ConfuciusWav2Vec2BertRuntime(
    std::shared_ptr<const ConfuciusAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    if (assets == nullptr) {
        throw std::runtime_error("Confucius4-TTS Wav2Vec2-BERT runtime requires assets");
    }
    if (graph_arena_bytes == 0) {
        throw std::runtime_error("Confucius4-TTS Wav2Vec2-BERT graph arena must be non-zero");
    }
    if (assets->semantic_encoder_weights == nullptr || assets->semantic_stats == nullptr) {
        throw std::runtime_error("Confucius4-TTS Wav2Vec2-BERT requires model and stats tensor sources");
    }
    engine::modules::Wav2Vec2BertEncoderWeightBinding binding;
    binding.matmul_storage_type = matmul_storage_type;
    binding.conv_storage_type = conv_storage_type;
    binding.stats_storage_type = matmul_storage_type;
    component_ = engine::modules::Wav2Vec2BertEncoderComponent::load_from_tensor_source(
        assets->semantic_encoder_weights,
        assets->semantic_stats,
        execution,
        graph_arena_bytes,
        weight_context_bytes,
        engine::modules::Wav2Vec2BertEncoderConfig{},
        binding);
}

ConfuciusWav2Vec2BertRuntime::~ConfuciusWav2Vec2BertRuntime() = default;

void ConfuciusWav2Vec2BertRuntime::prepare(int64_t frames) {
    component_.prepare(frames);
}

ConfuciusSemanticEmbedding ConfuciusWav2Vec2BertRuntime::encode(const ConfuciusSemanticFeatureOutput & features) {
    engine::modules::Wav2Vec2BertEncoderInput input;
    input.values = features.values;
    input.attention_mask = features.attention_mask;
    input.frames = features.frames;
    input.dims = features.dims;

    const auto output = component_.encode(input);
    return {output.values, output.frames, output.dims};
}

void ConfuciusWav2Vec2BertRuntime::release_graph() {
    component_.release_runtime_graph();
}

}  // namespace engine::models::confucius4_tts
