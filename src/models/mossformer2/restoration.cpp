#include "engine/models/mossformer2/restoration.h"
#include "engine/models/mossformer2/routing.h"
#include "engine/models/mossformer2/runtime.h"
#include "engine/models/mossformer2/vocoder.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::mossformer2 {
namespace {

std::vector<double> zero_phase_butterworth(const std::vector<float> & input, double cutoff, bool highpass) {
    // Match scipy.signal.butter(4, cutoff / 24000) and filtfilt's odd extension.
    constexpr size_t edge = 15;
    if (input.size() <= edge || cutoff <= 0 || cutoff >= 24000)
        throw std::runtime_error("MossFormer2 SR cannot filter this input bandwidth or length");
    const double pi = std::acos(-1.0), warped = std::tan(pi * cutoff / 48000.0);
    std::array<std::complex<double>, 5> polynomial{};
    polynomial[0] = 1;
    for (int i = 0; i < 4; ++i) {
        const auto p = warped * std::polar(1.0, pi * (2 * i + 5) / 8.0);
        const auto z = (1.0 + p) / (1.0 - p);
        for (int j = i + 1; j > 0; --j) polynomial[j] -= z * polynomial[j - 1];
    }
    std::array<double, 5> a{}, b{1, 4, 6, 4, 1};
    double gain = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = polynomial[i].real();
        gain += highpass && i % 2 ? -a[i] : a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) b[i] *= gain / 16.0 * (highpass && i % 2 ? -1.0 : 1.0);
    const double dc = std::accumulate(b.begin(), b.end(), 0.0) / std::accumulate(a.begin(), a.end(), 0.0);
    std::array<double, 4> zi{};
    for (int i = 3; i >= 0; --i) zi[i] = b[i + 1] - a[i + 1] * dc + (i == 3 ? 0 : zi[i + 1]);
    std::vector<double> x(input.size() + 2 * edge);
    for (size_t i = 0; i < edge; ++i) x[i] = 2.0 * input.front() - input[edge - i];
    std::copy(input.begin(), input.end(), x.begin() + edge);
    for (size_t i = 0; i < edge; ++i) x[edge + input.size() + i] = 2.0 * input.back() - input[input.size() - 2 - i];
    for (int pass = 0; pass < 2; ++pass) {
        auto state = zi;
        for (auto & value : state) value *= x.front();
        for (auto & value : x) {
            const auto y = b[0] * value + state[0];
            for (size_t i = 0; i < state.size(); ++i)
                state[i] = (i == 3 ? 0 : state[i + 1]) + b[i + 1] * value - a[i + 1] * y;
            value = y;
        }
        std::reverse(x.begin(), x.end());
    }
    return {x.begin() + edge, x.end() - edge};
}

double estimate_bandwidth(const std::vector<float> & input, size_t threads) {
    auto padded = input;
    padded.resize((padded.size() + 127) / 128 * 128, 0.0F);
    const audio::STFTConfig config{256, 128, 256, true, audio::STFTPadMode::Constant};
    const auto spectrum = audio::STFT().compute_complex(padded, audio::get_cached_stft_window(config),
        1, padded.size(), config, threads);
    const auto frames = spectrum.shape[2];
    std::array<double, 129> energy{};
    for (size_t f = 0; f < energy.size(); ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            const size_t index = (f * frames + t) * 2;
            energy[f] += double(spectrum.values[index]) * spectrum.values[index] +
                double(spectrum.values[index + 1]) * spectrum.values[index + 1];
        }
    }
    const double total = std::accumulate(energy.begin(), energy.end(), 0.0);
    if (total == 0) return 0;
    double cumulative = 0;
    size_t bin = 0;
    for (; bin < energy.size(); ++bin) {
        cumulative += energy[bin];
        if (cumulative / total >= 0.9996) break;
    }
    return static_cast<double>(bin) * 48000.0 / 256.0;
}

std::vector<float> substitute_low_band(const std::vector<float> & input, const std::vector<float> & generated,
                                      double cutoff) {
    const auto low = zero_phase_butterworth(input, cutoff, false);
    const auto high = zero_phase_butterworth(generated, cutoff, true);
    std::vector<float> result(std::min(low.size(), high.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        const double fade = std::min(1.0, static_cast<double>(i) / 4799.0);
        result[i] = static_cast<float>((1 - fade) * input[i] + fade * (low[i] + high[i]));
    }
    return result;
}
}  // namespace

class MossFormer2RestorationRuntime::Impl {
public:
    Impl(std::shared_ptr<const MossFormer2Assets> assets, core::ExecutionContext & execution)
        : assets(std::move(assets)), execution(execution), backbone(this->assets, execution) {
        const auto & c = this->assets->config;
        if (this->assets->impl->use_vocoder) {
            vocoder = std::make_unique<MossFormer2VocoderRuntime>(this->assets, execution);
            audio::MelSpectrogramFrontendConfig config;
            config.sample_rate = c.sample_rate;
            config.n_fft = config.win_length = c.fft_length;
            config.hop_length = c.hop_length;
            config.n_mels = c.mel_bins;
            config.mel_fmin = c.mel_low;
            config.mel_fmax = c.mel_high;
            config.filterbank_projection = audio::MelFilterbankProjection::SparseF32;
            config.magnitude_epsilon = 1e-9F;
            mel = audio::get_cached_mel_spectrogram_frontend(config);
        } else {
            stft = {c.fft_length, c.hop_length, c.fft_length, false, audio::STFTPadMode::Constant};
            window.resize(c.fft_length);
            window_squared.resize(c.fft_length);
            const float step = 2.0F * std::acos(-1.0F) / (c.fft_length - 1);
            for (int i = 0; i < c.fft_length; ++i) {
                window[i] = 0.54F - 0.46F * std::cos(step * i);
                window_squared[i] = window[i] * window[i];
            }
        }
        this->assets->tensors->release_storage();
    }

    std::vector<float> enhance(const std::vector<float> & input) {
        const auto & c = assets->config;
        if (input.size() < static_cast<size_t>(c.fft_length))
            throw std::runtime_error("MossFormer2 SE input is shorter than its analysis window");
        const auto frames = 1 + (input.size() - c.fft_length + c.hop_length - 1) / c.hop_length;
        std::vector<float> framed(frames * c.fft_length), scaled(c.fft_length + (frames - 1) * c.hop_length, 0);
        for (size_t i = 0; i < input.size(); ++i) scaled[i] = input[i] * 32768.0F;
        std::normal_distribution<float> noise;
        for (size_t t = 0; t < frames; ++t)
            for (int i = 0; i < c.fft_length; ++i) framed[t * c.fft_length + i] = scaled[t * c.hop_length + i] + noise(rng);
        audio::KaldiFbankOptions options;
        options.sample_rate = c.sample_rate;
        options.num_mels = c.mel_bins;
        options.frame_length_ms = 1000.0F * c.fft_length / c.sample_rate;
        // Frames are already overlapped and independently dithered, as in Kaldi.
        options.frame_shift_ms = options.frame_length_ms;
        options.lfr_m = options.lfr_n = 1;
        options.sparse_filterbank = true;
        auto fbank = audio::extract_kaldi_fbank(framed, options);
        if (static_cast<size_t>(fbank.frames) != frames) throw std::runtime_error("MossFormer2 SE filterbank frame mismatch");
        std::vector<float> features(3 * c.mel_bins * frames);
        for (int m = 0; m < c.mel_bins; ++m)
            for (size_t t = 0; t < frames; ++t) features[m * frames + t] = fbank.values[t * c.mel_bins + m];
        for (int order = 1; order < 3; ++order) {
            for (int m = 0; m < c.mel_bins; ++m) {
                const auto base = ((order - 1) * c.mel_bins + m) * frames;
                for (size_t t = 0; t < frames; ++t) {
                    float delta = 0;
                    for (int d = 1; d <= 2; ++d) delta += d * (features[base + std::min(frames - 1, t + d)] - features[base + (t < size_t(d) ? 0 : t - d)]);
                    features[(order * c.mel_bins + m) * frames + t] = delta / 10.0F;
                }
            }
        }
        auto spectrum = audio::STFT().compute_complex(scaled, window, 1, scaled.size(), stft, execution.config().threads);
        const auto mask = backbone.process_features(features, frames);
        for (size_t i = 0; i < mask.size(); ++i) {
            spectrum.values[i * 2] *= mask[i];
            spectrum.values[i * 2 + 1] *= mask[i];
        }
        std::vector<std::complex<float>> complex(spectrum.values.size() / 2);
        for (size_t i = 0; i < complex.size(); ++i) complex[i] = {spectrum.values[i * 2], spectrum.values[i * 2 + 1]};
        std::vector<float> inverse(frames * c.fft_length);
        audio::real_fft_inverse({frames, static_cast<size_t>(c.fft_length)},
            {sizeof(std::complex<float>), static_cast<std::ptrdiff_t>(frames * sizeof(std::complex<float>))},
            {static_cast<std::ptrdiff_t>(c.fft_length * sizeof(float)), sizeof(float)}, 1, complex.data(), inverse.data(),
            1.0F / c.fft_length, execution.config().threads);
        std::vector<float> result(scaled.size(), 0), weights(scaled.size(), 0), chunk(c.fft_length);
        for (size_t t = 0; t < frames; ++t) {
            for (int i = 0; i < c.fft_length; ++i) chunk[i] = inverse[t * c.fft_length + i] / window[i];
            audio::AudioChunkSpan span;
            span.output_start_sample = t * c.hop_length;
            span.valid_samples = c.fft_length;
            audio::overlap_add_planar_chunk(result, weights, chunk, 1, result.size(), span, window_squared,
                audio::AudioChunkCounterMode::SharedAcrossLanes);
        }
        audio::normalize_overlap_added_planar(result, weights, 1, result.size(), audio::AudioChunkCounterMode::SharedAcrossLanes);
        for (auto & value : result) value /= 32768.0F;
        result.resize(input.size());
        return result;
    }

    std::vector<float> super_resolve(const std::vector<float> & input) {
        const auto features = mel->extract_mono(input, execution.config().threads);
        return vocoder->synthesize(backbone.process_features(features.values, features.frames), features.frames);
    }

    std::shared_ptr<const MossFormer2Assets> assets;
    core::ExecutionContext & execution;
    MossFormer2GatedFSMNRuntime backbone;
    std::unique_ptr<MossFormer2VocoderRuntime> vocoder;
    audio::STFTConfig stft;
    std::shared_ptr<const audio::MelSpectrogramFrontend> mel;
    std::vector<float> window, window_squared;
    std::mt19937 rng{std::random_device{}()};
};

MossFormer2RestorationRuntime::MossFormer2RestorationRuntime(std::shared_ptr<const MossFormer2Assets> assets,
    core::ExecutionContext & execution) : impl_(std::make_unique<Impl>(std::move(assets), execution)) {}
MossFormer2RestorationRuntime::~MossFormer2RestorationRuntime() = default;

std::vector<float> MossFormer2RestorationRuntime::restore(const std::vector<float> & input, int64_t window, int64_t overlap) {
    auto & p = *impl_;
    const bool se = !p.assets->impl->use_vocoder;
    const double cutoff = se ? 0 : estimate_bandwidth(input, p.execution.config().threads);
    // No missing band exists for silence or input extending to Nyquist.
    if (!se && (cutoff == 0 || cutoff >= 24000)) return input;
    const auto run = [&](const std::vector<float> & x) { return se ? p.enhance(x) : p.super_resolve(x); };
    std::vector<float> result;
    if (window == 0 || input.size() <= static_cast<size_t>(window)) {
        result = run(input);
    } else {
        const auto hop = window - overlap;
        int64_t padded = input.size();
        if (padded < window + hop) padded = window + hop;
        else if ((padded - window) % hop) padded += padded - (padded - window) / hop * hop;
        auto samples = input;
        samples.resize(padded, 0);
        result.resize(padded, 0);
        const int64_t discard = overlap / 2;
        for (int64_t offset = 0; offset + window <= padded; offset += hop) {
            const auto chunk = run({samples.begin() + offset, samples.begin() + offset + window});
            const auto first = offset == 0 ? 0 : discard;
            const bool final_window = offset + window + hop > padded;
            const auto last = std::min<int64_t>(window - (final_window ? 0 : discard), chunk.size());
            if (last <= first) throw std::runtime_error("MossFormer2 restoration chunk is too short for its overlap");
            std::copy(chunk.begin() + first, chunk.begin() + last, result.begin() + offset + first);
        }
        if (se) result.resize(input.size());
    }
    return se ? result : substitute_low_band(input, result, cutoff);
}

}  // namespace engine::models::mossformer2
