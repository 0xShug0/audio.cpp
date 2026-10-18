#include "engine/models/open_wake_word/frontend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::models::open_wake_word {

Frontend::Frontend(const FrontendWeights &weights)
    : window_(weights.window), filterbank_{weights.mel_filter, {32, 257}} {}

audio::AudioTensor Frontend::extract(const std::vector<float> &pcm, size_t threads) const {
    if (pcm.size() < 512) {
        throw std::invalid_argument("OpenWakeWord frontend needs at least 512 samples");
    }
    audio::STFTConfig config;
    config.n_fft = config.win_length = 512;
    config.hop_length = 160;
    config.center = false;
    config.pad_mode = audio::STFTPadMode::Constant;
    const auto complex = audio::STFT().compute_complex(pcm, window_, 1, pcm.size(), config, threads);
    const auto frames = complex.shape[2];
    std::vector<float> power(complex.values.size() / 2);
    for (size_t i = 0; i < power.size(); ++i) {
        const float real = complex.values[i * 2], imag = complex.values[i * 2 + 1];
        power[i] = real * real + imag * imag;
    }
    auto mel = audio::MelFilterbank().compute_custom(power, 1, 257, frames, filterbank_);
    float maximum = -std::numeric_limits<float>::infinity();
    for (auto &value : mel.values) {
        value = std::log(std::max(value, 1e-10F)) * 10.0F / 2.3025851249694824F;
        maximum = std::max(maximum, value);
    }
    audio::AudioTensor output;
    output.shape = {frames, 32};
    output.values.resize(mel.values.size());
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t m = 0; m < 32; ++m) {
            output.values[t * 32 + m] = std::max(mel.values[m * frames + t], maximum - 80.0F) / 10.0F + 2.0F;
        }
    }
    return output;
}

} // namespace engine::models::open_wake_word
