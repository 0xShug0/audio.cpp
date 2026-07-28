#include "engine/models/confucius4_tts/style_encoder.h"

#include <stdexcept>
#include <utility>

namespace engine::models::confucius4_tts {

ConfuciusStyleEncoder::ConfuciusStyleEncoder(
    std::shared_ptr<const ConfuciusAssets> assets,
    core::BackendConfig backend,
    engine::assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Confucius4-TTS style encoder requires assets");
    }
    engine::modules::CampplusEncoderConfig config;
    config.feat_dim = assets_->config.style_encoder.feat_dim;
    config.embedding_size = assets_->config.style_encoder.embedding_size;
    config.weight_storage_type = weight_storage_type;
    component_ = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        assets_->style_encoder_weights,
        std::move(backend),
        config);
}

ConfuciusStyleEmbedding ConfuciusStyleEncoder::embed_fbank(
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims) const {
    const auto out = component_.embed_from_features(features, frames, dims);
    return {out.embedding, out.embedding_size};
}

void ConfuciusStyleEncoder::release_runtime_graph() {
    component_.release_runtime_graph();
}

}  // namespace engine::models::confucius4_tts
