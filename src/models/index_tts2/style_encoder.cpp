#include "engine/models/index_tts2/style_encoder.h"

#include <stdexcept>
#include <utility>

namespace engine::models::index_tts2 {

IndexTTS2StyleEncoder::IndexTTS2StyleEncoder(
    std::shared_ptr<const IndexTTS2Assets> assets,
    core::BackendConfig backend,
    engine::assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 style encoder requires assets");
    }
    engine::modules::CampplusEncoderConfig config;
    config.feat_dim = assets_->config.s2mel.n_mels;
    config.embedding_size = assets_->config.s2mel.style_dim;
    config.weight_storage_type = weight_storage_type;
    component_ = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        assets_->campplus_weights,
        std::move(backend),
        config);
}

IndexTTS2StyleEmbedding IndexTTS2StyleEncoder::embed_fbank(
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims) const {
    const auto out = component_.embed_from_features(features, frames, dims);
    return {out.embedding, out.embedding_size};
}

void IndexTTS2StyleEncoder::release_graph() {
    component_.release_runtime_graph();
}

}  // namespace engine::models::index_tts2
