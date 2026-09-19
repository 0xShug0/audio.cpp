#include "engine/community_models/granite5asr/frontend.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::granite5asr {
namespace {

audio::NemoMelFrontend make_frontend(const std::shared_ptr<const Granite5ASRAssets> & assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Granite 5 ASR frontend requires valid assets");
    }
    const auto & source = assets->config.frontend;
    audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.n_mels;
    config.stft = {source.n_fft, source.hop_length, source.win_length, true,
                   audio::STFTPadMode::Reflect, audio::STFTFamily::Default};
    config.window = audio::MelWindow::PeriodicF64;
    config.mel_bank = audio::MelBank::HTK;
    config.mel_path = audio::MelPath::DensePowerLog10;
    config.log_zero_guard = 1e-10f;
    config.max_log_drop = source.logmel_floor_db;
    config.log_scale = 0.25f;
    config.log_bias = 1.0f;
    config.add_deltas = source.deltas;
    config.stack_frames = source.stack_factor;
    config.layout = audio::MelLayout::TimeMajor;
    return audio::NemoMelFrontend(std::move(config));
}

}  // namespace

Granite5Frontend::Granite5Frontend(std::shared_ptr<const Granite5ASRAssets> assets)
    : frontend_(make_frontend(assets)) {}

Granite5FrontendFeatures Granite5Frontend::extract(const runtime::AudioBuffer & audio) const {
    auto mel = frontend_.extract_audio(audio.samples, audio.sample_rate, audio.channels, {});
    Granite5FrontendFeatures features;
    features.values = std::move(mel.values);
    features.frames = mel.frames;
    features.feature_dim = mel.feature_size;
    return features;
}

}  // namespace engine::community_models::granite5asr
