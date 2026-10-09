#include "engine/models/ced/frontend.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>

namespace engine::models::ced {

CedLogMelFrontend::CedLogMelFrontend(const CedConfig & config, const assets::TensorSource & source)
    : stft_{config.fft_size, config.hop_size, config.window_size, config.center, audio::STFTPadMode::Reflect},
      window_(source.require_f32("frontend.window", {config.window_size})) {
    audio::AudioTensor bank;
    bank.shape = {config.mel_bins, config.fft_size / 2 + 1};
    bank.values = source.require_f32("frontend.mel_filterbank", bank.shape);
    filterbank_ = audio::MelFilterbank().prepare_sparse(bank);
}

audio::AudioTensor CedLogMelFrontend::extract(const std::vector<float> & mono, size_t threads) const {
    const auto stft_started = std::chrono::steady_clock::now();
    const auto magnitude = audio::STFT().compute_magnitude(mono, window_, 1, mono.size(), stft_, threads);
    debug::timing_log_scalar("ced.frontend.stft_ms", debug::elapsed_ms(stft_started));
    const auto mel_started = std::chrono::steady_clock::now();
    const int64_t frames = magnitude.shape.at(2);
    auto mel = audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values, 1, magnitude.shape.at(1), frames, frames, filterbank_);
    for (auto & value : mel.values) value = 10.0f * std::log10(std::max(value, 1e-10f));
    const float floor = *std::max_element(mel.values.begin(), mel.values.end()) - 120.0f;
    for (auto & value : mel.values) value = std::max(value, floor);
    debug::timing_log_scalar("ced.frontend.mel_ms", debug::elapsed_ms(mel_started));
    return mel;
}

}  // namespace engine::models::ced
