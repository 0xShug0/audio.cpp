#include "engine/models/index_tts2/vocoder.h"

#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

constexpr int64_t kChunkedVocoderFrames = 768;
constexpr int64_t kChunkedVocoderOverlapFrames = 32;

engine::modules::BigVganVocoderConfig parse_bigvgan_config(
    const IndexTTS2Assets & assets,
    engine::assets::TensorStorageType weight_storage_type) {
    const auto root = assets.resources.parse_json("bigvgan_config");
    engine::modules::BigVganVocoderConfig config;
    config.sampling_rate = engine::io::json::require_i64(root, "sampling_rate");
    config.num_mels = engine::io::json::require_i64(root, "num_mels");
    config.n_fft = engine::io::json::require_i64(root, "n_fft");
    config.hop_size = engine::io::json::require_i64(root, "hop_size");
    config.win_size = engine::io::json::require_i64(root, "win_size");
    config.upsample_initial_channel = engine::io::json::require_i64(root, "upsample_initial_channel");
    config.snake_logscale = engine::io::json::require_bool(root, "snake_logscale");
    config.upsample_rates = engine::io::json::require_i64_array(root, "upsample_rates");
    config.upsample_kernel_sizes = engine::io::json::require_i64_array(root, "upsample_kernel_sizes");
    config.resblock_kernel_sizes = engine::io::json::require_i64_array(root, "resblock_kernel_sizes");
    config.weight_storage_type = weight_storage_type;
    if (config.sampling_rate != assets.config.s2mel.sample_rate ||
        config.num_mels != assets.config.s2mel.n_mels ||
        config.n_fft != assets.config.s2mel.n_fft ||
        config.hop_size != assets.config.s2mel.hop_length ||
        config.win_size != assets.config.s2mel.win_length) {
        throw std::runtime_error("IndexTTS2 BigVGAN config does not match S2Mel mel config");
    }
    return config;
}

}  // namespace

IndexTTS2BigVganVocoder::IndexTTS2BigVganVocoder(
    std::shared_ptr<const IndexTTS2Assets> assets,
    core::BackendConfig backend,
    engine::assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 BigVGAN vocoder requires assets");
    }
    component_ = engine::modules::BigVganVocoderComponent::load_from_tensor_source(
        assets_->bigvgan_weights,
        std::move(backend),
        parse_bigvgan_config(*assets_, weight_storage_type));
}

IndexTTS2VocoderOutput IndexTTS2BigVganVocoder::synthesize(
    const std::vector<float> & mel,
    int64_t frames) const {
    const auto out = frames > kChunkedVocoderFrames
        ? component_.synthesize_chunked(mel, frames, kChunkedVocoderFrames, kChunkedVocoderOverlapFrames)
        : component_.synthesize(mel, frames);
    return {out.waveform, out.samples, static_cast<int>(out.sample_rate)};
}

void IndexTTS2BigVganVocoder::release_runtime_graph() {
    component_.release_runtime_graph();
}

}  // namespace engine::models::index_tts2
