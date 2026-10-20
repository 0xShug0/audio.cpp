#include "engine/community_models/xtts_v2/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::xtts_v2 {

MelFrontend::MelFrontend(MelPurpose purpose) : purpose_(purpose) {
    const bool conditioning = purpose == MelPurpose::Conditioning;
    const int sample_rate = conditioning ? 22050 : 16000;
    const int bands = conditioning ? 80 : 64;
    stft_ = {conditioning ? 2048 : 512, conditioning ? 256 : 160,
             conditioning ? 1024 : 400, true, audio::STFTPadMode::Reflect};
    window_.resize(stft_.win_length);
    const float pi = std::acos(-1.f);
    for (size_t i = 0; i < window_.size(); ++i) {
        const float cosine = std::cos(2 * pi * i / window_.size());
        window_[i] = conditioning ? .5f * (1 - cosine) : .54f - .46f * cosine;
    }
    const int bins = stft_.n_fft / 2 + 1;
    const float max_mel = 2595.f * std::log10(1 + 8000.f / 700);
    std::vector<float> edges(bands + 2);
    for (int i = 0; i < bands + 2; ++i) {
        edges[i] = 700.f * (std::pow(10.f, max_mel * i / (bands + 1) / 2595.f) - 1);
    }
    filterbank_.shape = {bands, bins};
    filterbank_.values.resize(bands * bins);
    for (int band = 0; band < bands; ++band) {
        const float scale = conditioning ? 2.f / (edges[band + 2] - edges[band]) : 1.f;
        for (int bin = 0; bin < bins; ++bin) {
            const float frequency = float(sample_rate) * bin / stft_.n_fft;
            const float rising = (frequency - edges[band]) / (edges[band + 1] - edges[band]);
            const float falling = (edges[band + 2] - frequency) / (edges[band + 2] - edges[band + 1]);
            filterbank_.values[band * bins + bin] = std::max(0.f, std::min(rising, falling)) * scale;
        }
    }
}

std::vector<float> MelFrontend::compute(const std::vector<float> & pcm, size_t threads) const {
    if (pcm.size() <= static_cast<size_t>(stft_.n_fft / 2)) {
        throw std::runtime_error("XTTS v2 reference is too short for reflect-padded STFT");
    }
    const auto spectrum = audio::STFT().compute_complex(pcm, window_, 1, pcm.size(), stft_, threads);
    std::vector<float> power(spectrum.values.size() / 2);
    for (size_t i = 0; i < power.size(); ++i) {
        const float real = spectrum.values[2 * i], imag = spectrum.values[2 * i + 1];
        power[i] = real * real + imag * imag;
    }
    auto mel = audio::MelFilterbank().compute_custom(power, 1, spectrum.shape[1], spectrum.shape[2], filterbank_);
    if (purpose_ == MelPurpose::Conditioning) {
        for (auto & value : mel.values) value = std::log(std::max(value, 1e-5f));
    }
    return std::move(mel.values);
}
}  // namespace engine::models::xtts_v2
