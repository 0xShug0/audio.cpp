#include "engine/community_models/nemotron_transcribe/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_transcribe {
namespace {

audio::NemoMelFrontendConfig make_config(const FrontendConfig & config, const std::vector<float> & window) {
    if (config.sample_rate <= 0 || config.mel_bins <= 0 || config.n_fft <= 0 ||
        config.window_length <= 0 || config.window_length > config.n_fft ||
        config.hop_length <= 0 || !std::isfinite(config.preemphasis)) {
        throw std::runtime_error("Nemotron Transcribe frontend configuration is invalid");
    }
    audio::NemoMelFrontendConfig result;
    result.sample_rate = config.sample_rate;
    result.n_mels = config.mel_bins;
    result.stft = {config.n_fft, config.hop_length, config.window_length,
                   true, audio::STFTPadMode::Constant, audio::STFTFamily::Default};
    if (std::any_of(window.begin(), window.end(), [](float x) { return !std::isfinite(x); })) {
        throw std::runtime_error("Nemotron Transcribe trained window contains non-finite values");
    }
    result.window = window.empty() ? audio::MelWindow::SymmetricPrecise : audio::MelWindow::FromArgument;
    result.input_rate = audio::MelInputRate::RequireMatch;
    result.preemphasis = config.preemphasis;
    result.mel_path = audio::MelPath::LogMelSpectrogram;
    result.frame_multiple = 1;
    result.pad_basis = audio::PadBasis::ValidFrames;
    result.mel_bank = audio::MelBank::FromArgument;
    // Inference is deterministic, matching NVIDIA's FE and parity test. Training
    // config dither is not applied at inference.
    return result;
}

}  // namespace

Frontend::Frontend(FrontendConfig config, std::vector<float> trained_filterbank, std::vector<float> trained_window)
    : config_(config),
      frontend_(make_config(config, trained_window), trained_window, audio::AudioTensor{
          std::move(trained_filterbank), {config.mel_bins, config.n_fft / 2 + 1}}) {}

MelFeatures Frontend::extract(const runtime::AudioBuffer & input, int threads) const {
    if (input.channels != 1 || input.sample_rate != config_.sample_rate || input.samples.empty()) {
        throw std::runtime_error("Nemotron Transcribe requires nonempty 16 kHz mono audio");
    }
    if (std::any_of(input.samples.begin(), input.samples.end(), [](float x) { return !std::isfinite(x); })) {
        throw std::runtime_error("Nemotron Transcribe audio contains a non-finite sample");
    }
    auto raw = frontend_.extract_audio(input.samples, input.sample_rate, input.channels,
        {true, audio::ValidFrameRule::FloorHops}, static_cast<size_t>(std::max(1, threads)));
    MelFeatures result;
    result.frames = raw.valid_frames;
    result.bins = config_.mel_bins;
    result.duration_seconds = static_cast<double>(input.samples.size()) / input.sample_rate;
    result.values = std::move(raw.values);
    result.values.resize(static_cast<size_t>(result.frames * result.bins));
    if (result.frames < 1) throw std::runtime_error("Nemotron Transcribe audio has no valid mel frames");
    return result;
}

std::vector<float> normalize_asr_features(const MelFeatures & features) {
    if (features.frames <= 0 || features.bins <= 0 ||
        features.values.size() != static_cast<size_t>(features.frames * features.bins)) {
        throw std::runtime_error("Nemotron Transcribe mel shape is invalid");
    }
    std::vector<float> normalized(features.values.size());
    for (int f = 0; f < features.bins; ++f) {
        double sum = 0.0;
        for (int64_t t = 0; t < features.frames; ++t) sum += features.values[t * features.bins + f];
        const double mean = sum / features.frames;
        double variance = 0.0;
        for (int64_t t = 0; t < features.frames; ++t) {
            const double delta = features.values[t * features.bins + f] - mean;
            variance += delta * delta;
        }
        const float stddev = static_cast<float>(features.frames > 1
            ? std::sqrt(variance / (features.frames - 1)) : 0.0) + 1e-5f;
        for (int64_t t = 0; t < features.frames; ++t) {
            normalized[t * features.bins + f] = static_cast<float>(
                (features.values[t * features.bins + f] - mean) / stddev);
        }
    }
    return normalized;
}

bool is_near_silent(const std::vector<float> & samples) {
    double sum = 0.0;
    for (const double sample : samples) {
        if (std::abs(sample) >= 1e-3) return false;
        sum += sample * sample;
    }
    return samples.empty() || std::sqrt(sum / samples.size()) < 3e-5;
}

}  // namespace engine::models::nemotron_transcribe
