#include "engine/models/gigaam_asr/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::gigaam_asr {

GigaAMFrontend::GigaAMFrontend(
    audio::STFTConfig config, std::vector<float> window, audio::AudioTensor filterbank)
    : config_(config), window_(std::move(window)), filterbank_(audio::MelFilterbank().prepare_sparse(filterbank)) {}

audio::AudioTensor GigaAMFrontend::extract(const std::vector<float> & mono_16khz, size_t threads) const {
    if (mono_16khz.size() < static_cast<size_t>(config_.n_fft)) {
        throw std::runtime_error("GigaAM audio is shorter than one analysis window");
    }
    auto spectrum = audio::STFT().compute_magnitude(
        mono_16khz, window_, 1, static_cast<int64_t>(mono_16khz.size()), config_, threads);
    auto mel = audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        spectrum.values, 1, spectrum.shape[1], spectrum.shape[2], spectrum.shape[2], filterbank_);
    for (auto & value : mel.values) {
        value = std::log(std::clamp(value, 1e-9f, 1e9f));
    }
    return mel;
}

}  // namespace engine::models::gigaam_asr
