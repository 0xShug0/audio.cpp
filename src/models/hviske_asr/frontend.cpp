#include "engine/models/hviske_asr/frontend.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

using Clock = std::chrono::steady_clock;

engine::audio::NemoMelFrontend make_frontend(const HviskeASRAssets & assets) {
    const auto & source = assets.config.frontend;
    engine::audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.features;
    config.stft = {source.n_fft, source.hop_length, source.win_length,
                   true, engine::audio::STFTPadMode::Constant, engine::audio::STFTFamily::Default};
    config.preemphasis = source.preemph;
    config.dither_stddev = source.dither;
    config.dither_method = engine::audio::DitherMethod::Normal;
    config.window = engine::audio::MelWindow::SymmetricPrecise;
    config.mel_path = engine::audio::MelPath::ComplexPowerLn;
    config.log_zero_guard = source.log_zero_guard;
    config.norm = engine::audio::MelNorm::PerBinMixed;
    config.layout = engine::audio::MelLayout::FeatureMajor;
    config.frame_multiple = std::max<int64_t>(1, source.pad_to);

    if (assets.model_weights == nullptr) {
        return engine::audio::NemoMelFrontend(std::move(config));
    }
    const bool has_filterbank = assets.model_weights->has_tensor("preprocessor.featurizer.fb");
    const bool has_window = assets.model_weights->has_tensor("preprocessor.featurizer.window");
    if (has_filterbank != has_window) {
        throw std::runtime_error("Hviske checkpoint requires both window and filterbank tensors");
    }
    if (!has_filterbank) {
        return engine::audio::NemoMelFrontend(std::move(config));
    }
    config.window = engine::audio::MelWindow::FromArgument;
    config.mel_bank = engine::audio::MelBank::FromArgument;
    const int64_t freq_bins = source.n_fft / 2 + 1;
    return engine::audio::NemoMelFrontend(
        std::move(config),
        assets.model_weights->require_f32("preprocessor.featurizer.window", {source.win_length}),
        engine::audio::AudioTensor{
            assets.model_weights->require_f32("preprocessor.featurizer.fb", {1, source.features, freq_bins}),
            {source.features, freq_bins}});
}

}  // namespace

HviskeFrontend::HviskeFrontend(std::shared_ptr<const HviskeASRAssets> assets)
    : frontend_(assets ? make_frontend(*assets) : throw std::runtime_error("Hviske frontend requires assets")) {}

HviskeFrontendFeatures HviskeFrontend::extract(const engine::runtime::AudioBuffer & audio) const {
    const auto wall_start = Clock::now();
    auto mel = frontend_.extract_audio(
        audio.samples, audio.sample_rate, audio.channels,
        {true, engine::audio::ValidFrameRule::FloorHops});
    HviskeFrontendFeatures out;
    out.values = std::move(mel.values);
    out.feature_dim = mel.feature_size;
    out.frames = mel.frames;
    out.valid_frames = mel.valid_frames;
    debug::timing_log_scalar("hviske_asr.frontend_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("hviske_asr.frontend_frames", out.frames);
    debug::trace_log_scalar("hviske_asr.frontend_valid_frames", out.valid_frames);
    return out;
}

}  // namespace engine::models::hviske_asr
