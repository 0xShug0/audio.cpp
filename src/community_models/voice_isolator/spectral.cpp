#include "engine/community_models/voice_isolator/spectral.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::voice_isolator {
namespace {

audio::STFTConfig stft_config(const Config & config, bool center, audio::STFTPadMode pad_mode) {
    return {config.n_fft, config.hop_length, config.n_fft, center, pad_mode, audio::STFTFamily::PeriodicF32};
}

}  // namespace

const std::vector<float> & analysis_window(const Config & config) {
    return audio::get_cached_stft_window(stft_config(config, true, audio::STFTPadMode::Reflect));
}

std::vector<float> clip_spectra(const Config & config, const std::vector<float> & samples, int64_t & frames) {
    const auto n = static_cast<int64_t>(samples.size());
    if (n == 0) {
        throw std::runtime_error("Voice Isolator needs at least one sample");
    }
    const auto pad = n > config.n_fft / 2 ? audio::STFTPadMode::Reflect : audio::STFTPadMode::Constant;
    auto spectra = audio::STFT().compute_complex(samples, analysis_window(config), 1, n, stft_config(config, true, pad));
    frames = spectra.shape[2];
    return std::move(spectra.values);
}

std::vector<float> clip_waveform(
    const Config & config, const std::vector<float> & spectra, int64_t frames, int64_t samples) {
    return audio::ISTFT().compute(spectra, analysis_window(config), 1, config.bins(), frames, samples,
        stft_config(config, true, audio::STFTPadMode::Reflect)).values;
}

std::vector<float> stream_spectra(
    const Config & config,
    const float * held,
    int64_t held_start,
    int64_t held_count,
    int64_t first,
    int64_t count) {
    const int64_t start = first * config.hop_length - config.n_fft / 2;
    const int64_t length = (count - 1) * config.hop_length + config.n_fft;
    std::vector<float> segment(static_cast<size_t>(length), 0.0F);
    const int64_t lo = std::max(start, held_start);
    const int64_t hi = std::min(start + length, held_start + held_count);
    if (hi > lo) {
        std::copy(held + (lo - held_start), held + (hi - held_start), segment.begin() + (lo - start));
    }
    return audio::STFT().compute_complex(segment, analysis_window(config), 1, length,
        stft_config(config, false, audio::STFTPadMode::Constant)).values;
}

void pack_bands(const BandLayout & layout, const float * spectra, int64_t frames, float * packed) {
    const int64_t inputs = layout.encoder_inputs();
    std::fill(packed, packed + layout.bands() * frames * inputs, 0.0F);
    for (int64_t band = 0; band < layout.bands(); ++band) {
        const auto [first_bin, end_bin] = layout.edges[static_cast<size_t>(band)];
        for (int64_t t = 0; t < frames; ++t) {
            float * row = packed + (band * frames + t) * inputs;
            for (int64_t bin = first_bin; bin < end_bin; ++bin) {
                const float * value = spectra + (bin * frames + t) * 2;
                row[2 * (bin - first_bin)] = value[0];
                row[2 * (bin - first_bin) + 1] = value[1];
            }
        }
    }
}

void spectral_change(
    const BandLayout & layout,
    float bound,
    const float * raw,
    int64_t raw_frames,
    int64_t raw_frame,
    const float * spectra,
    int64_t frames,
    int64_t frame,
    float * change) {
    const int64_t outputs = layout.decoder_outputs();
    for (int64_t band = 0; band < layout.bands(); ++band) {
        const auto [first_bin, end_bin] = layout.edges[static_cast<size_t>(band)];
        const float * head = raw + (band * raw_frames + raw_frame) * outputs;
        for (int64_t bin = first_bin; bin < end_bin; ++bin) {
            const float * r = head + 4 * (bin - first_bin);
            const float * s = spectra + (bin * frames + frame) * 2;
            float * out = change + (bin * frames + frame) * 2;
            const float mask_re = 2.0F * std::tanh(r[0]);
            const float mask_im = 2.0F * std::tanh(r[1]);
            const float res_re = std::tanh(r[2]);
            const float res_im = std::tanh(r[3]);
            const float norm = std::max(std::sqrt(res_re * res_re + res_im * res_im), 1.0F);
            const float scale = bound * std::sqrt(s[0] * s[0] + s[1] * s[1]) / norm;
            out[0] = s[0] * mask_re - s[1] * mask_im + res_re * scale;
            out[1] = s[0] * mask_im + s[1] * mask_re + res_im * scale;
        }
    }
}

StreamSynthesizer::StreamSynthesizer(const Config & config)
    : n_fft_(config.n_fft),
      hop_(config.hop_length),
      window_(&analysis_window(config)),
      plan_(audio::get_real_fft_plan(static_cast<size_t>(config.n_fft))) {}

StreamSynthesizer::~StreamSynthesizer() = default;

void StreamSynthesizer::add(int64_t first, int64_t count, const float * change) {
    if (count <= 0) {
        return;
    }
    frames_.resize(static_cast<size_t>(count * n_fft_));
    const auto complex_bytes = static_cast<std::ptrdiff_t>(sizeof(std::complex<float>));
    plan_->inverse(
        {static_cast<size_t>(count), static_cast<size_t>(n_fft_)},
        {complex_bytes, count * complex_bytes},
        {static_cast<std::ptrdiff_t>(n_fft_ * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))},
        1,
        reinterpret_cast<const std::complex<float> *>(change),
        frames_.data(),
        1.0F / static_cast<float>(n_fft_));
    const int64_t needed = (first + count - 1) * hop_ - base_ + n_fft_;
    if (needed > static_cast<int64_t>(sum_.size())) {
        sum_.resize(static_cast<size_t>(needed), 0.0F);
        envelope_.resize(static_cast<size_t>(needed), 0.0F);
    }
    const auto & window = *window_;
    for (int64_t k = 0; k < count; ++k) {
        const int64_t at = (first + k) * hop_ - base_;
        if (at < 0) {
            throw std::runtime_error("Voice Isolator synthesized a frame that was already released");
        }
        const float * frame = frames_.data() + k * n_fft_;
        for (int64_t n = 0; n < n_fft_; ++n) {
            sum_[static_cast<size_t>(at + n)] += frame[n] * window[static_cast<size_t>(n)];
            envelope_[static_cast<size_t>(at + n)] += window[static_cast<size_t>(n)] * window[static_cast<size_t>(n)];
        }
    }
}

void StreamSynthesizer::take(int64_t start, int64_t stop, float * out) {
    const int64_t offset = n_fft_ / 2 - base_;
    for (int64_t s = start; s < stop; ++s) {
        const auto index = static_cast<size_t>(s + offset);
        out[s - start] = index < sum_.size() ? sum_[index] / std::max(envelope_[index], 1.0e-11F) : 0.0F;
    }
    const int64_t drop = std::min<int64_t>(stop - base_, static_cast<int64_t>(sum_.size()));
    if (drop > 0) {
        sum_.erase(sum_.begin(), sum_.begin() + drop);
        envelope_.erase(envelope_.begin(), envelope_.begin() + drop);
        base_ += drop;
    }
}

void StreamSynthesizer::reset() {
    sum_.clear();
    envelope_.clear();
    base_ = 0;
}

std::vector<RenderWindow> plan_render_windows(int64_t samples, int64_t chunk, int64_t overlap) {
    if (chunk <= 0 || overlap < 0 || overlap > chunk / 2) {
        throw std::runtime_error("Voice Isolator render windows need a positive chunk and an overlap of at most half of it");
    }
    if (samples <= chunk) {
        return {{0, samples, overlap}};
    }
    std::vector<RenderWindow> windows;
    for (int64_t start = 0; start < samples - chunk; start += chunk - overlap) {
        windows.push_back({start, chunk, overlap});
    }
    windows.push_back({samples - chunk, chunk, overlap});
    for (size_t index = 0; index < windows.size(); ++index) {
        auto & window = windows[index];
        if (index > 0) {
            window.fade_in = windows[index - 1].start + chunk - overlap - window.start;
        }
        window.fade_out = overlap > 0 && window.start + chunk < samples;
    }
    return windows;
}

std::vector<float> RenderWindow::weights() const {
    // [ramp up | ramp down] over `overlap` samples each: torch.linspace(0, 1) and (1, 0).
    std::vector<float> ramps;
    if (overlap > 1) {
        ramps = audio::make_linear_fade_window(2 * overlap, overlap);
    } else if (overlap == 1) {
        ramps = {0.0F, 1.0F};
    }
    std::vector<float> w(static_cast<size_t>(length), 1.0F);
    if (fade_in >= 0) {
        std::fill(w.begin(), w.begin() + fade_in, 0.0F);
        std::copy(ramps.begin(), ramps.begin() + overlap, w.begin() + fade_in);
    }
    if (fade_out) {
        std::copy(ramps.begin() + overlap, ramps.end(), w.end() - overlap);
    }
    return w;
}

}  // namespace engine::models::voice_isolator
