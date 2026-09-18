#include "engine/models/universr/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::universr {

UniverSRFeatures analyze_audio(const UniverSRAssets & assets,
    const std::vector<float> & interleaved, int channels, int file_sample_rate,
    int effective_sample_rate, int threads) {
    const auto started = std::chrono::steady_clock::now();
    const auto & config = assets.config;
    const auto rate = config.sr_to_lr_bins.find(effective_sample_rate / 1000);
    if (effective_sample_rate % 1000 != 0 || rate == config.sr_to_lr_bins.end()) {
        throw std::runtime_error("UniverSR effective input rate must be 8000, 12000, 16000, or 24000 Hz");
    }
    if (interleaved.empty() || channels < 1 || interleaved.size() % channels != 0 || file_sample_rate <= 0) {
        throw std::runtime_error("UniverSR requires nonempty, valid interleaved audio");
    }
    auto waveform = audio::mixdown_interleaved_to_mono_average(interleaved, channels);
    const auto resampling = audio::torchaudio_sinc_hann_float32_options();
    if (file_sample_rate == config.sample_rate) {
        const size_t original = waveform.size();
        waveform = audio::resample_mono_torchaudio_sinc_hann(waveform, config.sample_rate, effective_sample_rate, resampling);
        waveform = audio::resample_mono_torchaudio_sinc_hann(waveform, effective_sample_rate, config.sample_rate, resampling);
        waveform.resize(original);
    } else {
        waveform = audio::resample_mono_torchaudio_sinc_hann(waveform, file_sample_rate, config.sample_rate, resampling);
    }
    UniverSRFeatures result;
    result.original_samples = static_cast<int64_t>(waveform.size());
    result.low_bins = rate->second;
    waveform.resize(std::max<size_t>(32768, waveform.size()), 0.0f);
    const audio::STFTConfig stft_config{config.n_fft, config.hop_length, config.n_fft, true, audio::STFTPadMode::Reflect};
    const auto spectrum = audio::STFT().compute_complex(waveform, assets.window, 1,
        static_cast<int64_t>(waveform.size()), stft_config, static_cast<size_t>(threads));
    result.frames = spectrum.shape[2];
    const int64_t plane = result.low_bins * result.frames;
    result.low_spectrum.resize(static_cast<size_t>(2 * plane));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (int64_t i = 0; i < plane; ++i) {
        const float real = spectrum.values[static_cast<size_t>(2 * i)] + config.comp_eps;
        const float imag = spectrum.values[static_cast<size_t>(2 * i + 1)];
        const float magnitude = std::pow(std::hypot(real, imag), config.alpha) * config.beta;
        const float phase = std::atan2(imag, real);
        result.low_spectrum[static_cast<size_t>(i)] = magnitude * std::cos(phase);
        result.low_spectrum[static_cast<size_t>(plane + i)] = magnitude * std::sin(phase);
    }
    debug::timing_log_scalar("universr.frontend.ms", debug::elapsed_ms(started));
    return result;
}

std::vector<float> synthesize_audio(const UniverSRAssets & assets,
    const UniverSRFeatures & features, const std::vector<float> & generated_high_spectrum,
    int threads) {
    const auto started = std::chrono::steady_clock::now();
    const auto & config = assets.config;
    const int64_t frames = features.frames;
    const int64_t low_plane = features.low_bins * frames;
    const int64_t high_plane = config.hr_freq_bins * frames;
    if (frames < 2 || features.low_bins < config.total_freq_bins - config.hr_freq_bins ||
        features.low_bins > config.total_freq_bins || features.original_samples < 1 ||
        features.low_spectrum.size() != static_cast<size_t>(2 * low_plane) ||
        generated_high_spectrum.size() != static_cast<size_t>(2 * high_plane)) {
        throw std::runtime_error("UniverSR synthesis spectrum shape mismatch");
    }
    const int64_t plane = config.total_freq_bins * frames;
    std::vector<float> spectrum(static_cast<size_t>(2 * (config.total_freq_bins + 1) * frames), 0.0f);
    const int64_t high_start = config.total_freq_bins - config.hr_freq_bins;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (int64_t i = 0; i < plane; ++i) {
        const int64_t bin = i / frames;
        const bool low = bin < features.low_bins;
        const int64_t index = low ? i : i - high_start * frames;
        const auto & input = low ? features.low_spectrum : generated_high_spectrum;
        const int64_t channel_stride = low ? low_plane : high_plane;
        const float real = input[static_cast<size_t>(index)] / config.beta;
        const float imag = input[static_cast<size_t>(channel_stride + index)] / config.beta;
        const float magnitude = std::pow(std::hypot(real, imag), 1.0f / config.alpha);
        const float phase = std::atan2(imag, real);
        spectrum[static_cast<size_t>(2 * i)] = magnitude * std::cos(phase);
        spectrum[static_cast<size_t>(2 * i + 1)] = magnitude * std::sin(phase);
    }
    // Upstream calls istft without length, then crops back to the unpadded waveform length.
    const int64_t reconstructed_samples = (frames - 1) * config.hop_length;
    const audio::STFTConfig stft_config{config.n_fft, config.hop_length, config.n_fft, true, audio::STFTPadMode::Reflect};
    auto waveform = audio::ISTFT().compute(spectrum, assets.window, 1, config.total_freq_bins + 1,
        frames, reconstructed_samples, stft_config, static_cast<size_t>(threads)).values;
    waveform.resize(static_cast<size_t>(std::min(features.original_samples, reconstructed_samples)));
    debug::timing_log_scalar("universr.istft.ms", debug::elapsed_ms(started));
    return waveform;
}

}  // namespace engine::models::universr
