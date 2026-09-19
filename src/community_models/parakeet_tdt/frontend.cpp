#include "engine/community_models/parakeet_tdt/frontend.h"

#include "engine/framework/debug/profiler.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;

engine::audio::NemoMelFrontendConfig mel_config(const ParakeetTDTAssets & assets) {
    const auto & source = assets.config.frontend;
    engine::audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.feature_size;
    config.stft = {source.n_fft, source.hop_length, source.win_length,
                   true, engine::audio::STFTPadMode::Constant, engine::audio::STFTFamily::Default};
    config.preemphasis = source.preemphasis;
    config.window = engine::audio::MelWindow::SymmetricPrecise;
    config.log_zero_guard = source.log_zero_guard;
    config.norm = engine::audio::MelNorm::PerBinF64;
    return config;
}

}  // namespace

ParakeetFrontend::ParakeetFrontend(std::shared_ptr<const ParakeetTDTAssets> assets)
    : frontend_(assets ? mel_config(*assets) : throw std::runtime_error("Parakeet TDT frontend requires assets")) {}

ParakeetFrontendFeatures ParakeetFrontend::extract(
    const engine::runtime::AudioBuffer & audio,
    bool center) const {
    const auto wall_start = Clock::now();
    auto mel = frontend_.extract_audio(
        audio.samples, audio.sample_rate, audio.channels,
        {center, center ? engine::audio::ValidFrameRule::FloorHops
                        : engine::audio::ValidFrameRule::StftFrames});
    ParakeetFrontendFeatures out;
    out.values = std::move(mel.values);
    out.frames = mel.frames;
    out.valid_frames = mel.valid_frames;
    out.feature_dim = mel.feature_size;
    debug::timing_log_scalar("parakeet.frontend_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("parakeet.frontend.frames", out.frames);
    debug::trace_log_scalar("parakeet.frontend.valid_frames", out.valid_frames);
    debug::trace_log_scalar("parakeet.frontend.center", center);
    return out;
}

}  // namespace engine::community_models::parakeet_tdt
