#include "engine/models/nemotron_asr/frontend.h"

#include "engine/framework/debug/profiler.h"

#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

engine::audio::NemoMelFrontend make_frontend(const std::shared_ptr<const NemotronASRAssets> & assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Nemotron ASR frontend requires assets");
    }
    const auto & source = assets->config.frontend;
    engine::audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.feature_size;
    config.stft = {source.n_fft, source.hop_length, source.win_length, true,
                   engine::audio::STFTPadMode::Constant, engine::audio::STFTFamily::Default};
    config.preemphasis = source.preemphasis;
    config.window = engine::audio::MelWindow::SymmetricPrecise;
    config.mel_bank = engine::audio::MelBank::Slaney;
    config.mel_path = engine::audio::MelPath::SparsePowerLn;
    config.log_zero_guard = source.log_zero_guard;
    config.layout = engine::audio::MelLayout::TimeMajor;
    return engine::audio::NemoMelFrontend(std::move(config));
}

}  // namespace

NemotronFrontend::NemotronFrontend(std::shared_ptr<const NemotronASRAssets> assets)
    : frontend_(make_frontend(assets)) {}

NemotronFrontendFeatures NemotronFrontend::extract(
    const engine::runtime::AudioBuffer & audio,
    bool center) const {
    return extract_waveform(prepare_waveform(audio), center);
}

std::vector<float> NemotronFrontend::prepare_waveform(const engine::runtime::AudioBuffer & audio) const {
    return frontend_.prepare_audio(audio.samples, audio.sample_rate, audio.channels);
}

NemotronFrontendFeatures NemotronFrontend::extract_waveform(
    const std::vector<float> & waveform,
    bool center) const {
    const auto wall_start = Clock::now();
    engine::audio::NemoMelRunConfig run;
    run.center = center;
    run.valid_frame_rule = center ? engine::audio::ValidFrameRule::FloorHops
                              : engine::audio::ValidFrameRule::FullWindows;
    auto features = frontend_.extract_mono(waveform, run);

    NemotronFrontendFeatures out;
    out.values = std::move(features.values);
    out.frames = features.frames;
    out.valid_frames = features.valid_frames;
    out.feature_dim = features.feature_size;
    debug::timing_log_scalar("nemotron_asr.frontend_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.frontend.frames", out.frames);
    debug::trace_log_scalar("nemotron_asr.frontend.valid_frames", out.valid_frames);
    debug::trace_log_scalar("nemotron_asr.frontend.center", center);
    return out;
}

}  // namespace engine::models::nemotron_asr
