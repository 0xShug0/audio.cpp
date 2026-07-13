#include "engine/models/index_tts2/audio_features.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::index_tts2 {
namespace {

struct MelFilterbankKey {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t num_mels = 0;
    float fmin = 0.0F;
    float fmax = 0.0F;

    bool operator==(const MelFilterbankKey & other) const noexcept {
        return sample_rate == other.sample_rate && n_fft == other.n_fft && num_mels == other.num_mels &&
            fmin == other.fmin && fmax == other.fmax;
    }
};

struct MelFilterbankKeyHash {
    size_t operator()(const MelFilterbankKey & key) const noexcept {
        size_t seed = std::hash<int64_t>{}(key.sample_rate);
        seed ^= std::hash<int64_t>{}(key.n_fft) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.num_mels) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.fmin) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.fmax) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct KaldiFilterbankKey {
    int64_t sample_rate = 0;
    int64_t padded_window_size = 0;
    int64_t num_mels = 0;
    float low_freq = 0.0F;
    float high_freq = 0.0F;

    bool operator==(const KaldiFilterbankKey & other) const noexcept {
        return sample_rate == other.sample_rate &&
            padded_window_size == other.padded_window_size &&
            num_mels == other.num_mels &&
            low_freq == other.low_freq &&
            high_freq == other.high_freq;
    }
};

struct KaldiFilterbankKeyHash {
    size_t operator()(const KaldiFilterbankKey & key) const noexcept {
        size_t seed = 0;
        seed ^= std::hash<int64_t>{}(key.sample_rate) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.padded_window_size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.num_mels) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.low_freq) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.high_freq) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

std::vector<float> make_povey_window(int64_t window_size) {
    std::vector<float> window(static_cast<size_t>(window_size), 0.0F);
    constexpr float kPi = 3.14159265358979323846F;
    for (int64_t i = 0; i < window_size; ++i) {
        const float hann =
            0.5F - 0.5F * std::cos(2.0F * kPi * static_cast<float>(i) / static_cast<float>(window_size - 1));
        window[static_cast<size_t>(i)] = std::pow(hann, 0.85F);
    }
    return window;
}

std::vector<float> make_kaldi_mel_filterbank(
    int64_t sample_rate,
    int64_t n_fft,
    int64_t n_mels,
    float low_freq,
    float high_freq) {
    const int64_t num_fft_bins = n_fft / 2 + 1;
    const float nyquist = 0.5F * static_cast<float>(sample_rate);
    if (high_freq <= 0.0F) {
        high_freq += nyquist;
    }
    const float fft_bin_width = static_cast<float>(sample_rate) / static_cast<float>(n_fft);
    const float mel_low = 1127.0F * std::log(1.0F + low_freq / 700.0F);
    const float mel_high = 1127.0F * std::log(1.0F + high_freq / 700.0F);
    const float mel_delta = (mel_high - mel_low) / static_cast<float>(n_mels + 1);

    std::vector<float> filterbank(static_cast<size_t>(n_mels * num_fft_bins), 0.0F);
    for (int64_t mel_bin = 0; mel_bin < n_mels; ++mel_bin) {
        const float left_mel = mel_low + static_cast<float>(mel_bin) * mel_delta;
        const float center_mel = mel_low + static_cast<float>(mel_bin + 1) * mel_delta;
        const float right_mel = mel_low + static_cast<float>(mel_bin + 2) * mel_delta;
        for (int64_t fft_bin = 0; fft_bin < num_fft_bins; ++fft_bin) {
            const float freq = fft_bin_width * static_cast<float>(fft_bin);
            const float mel = 1127.0F * std::log(1.0F + freq / 700.0F);
            const float up_slope = (mel - left_mel) / std::max(center_mel - left_mel, 1.0e-12F);
            const float down_slope = (right_mel - mel) / std::max(right_mel - center_mel, 1.0e-12F);
            filterbank[static_cast<size_t>(mel_bin * num_fft_bins + fft_bin)] =
                std::max(0.0F, std::min(up_slope, down_slope));
        }
    }
    return filterbank;
}

const std::vector<float> & cached_mel_filterbank(const IndexTTS2S2MelConfig & config) {
    static std::mutex mutex;
    static std::unordered_map<MelFilterbankKey, std::vector<float>, MelFilterbankKeyHash> cache;
    const float fmax = config.fmax.value_or(static_cast<float>(config.sample_rate) / 2.0F);
    const MelFilterbankKey key{config.sample_rate, config.n_fft, config.n_mels, config.fmin, fmax};
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    const auto filterbank = engine::audio::MelFilterbank().build({
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.fmin,
        fmax,
        true,
    });
    return cache.emplace(
        key,
        filterbank.values).first->second;
}

const std::vector<float> & cached_povey_window(int64_t window_size) {
    static std::mutex mutex;
    static std::unordered_map<int64_t, std::vector<float>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(window_size);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(window_size, make_povey_window(window_size)).first->second;
}

const std::vector<float> & cached_kaldi_mel_filterbank(
    int64_t sample_rate,
    int64_t padded_window_size,
    int64_t num_mels,
    float low_freq,
    float high_freq) {
    static std::mutex mutex;
    static std::unordered_map<KaldiFilterbankKey, std::vector<float>, KaldiFilterbankKeyHash> cache;
    const KaldiFilterbankKey key{sample_rate, padded_window_size, num_mels, low_freq, high_freq};
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(
        key,
        make_kaldi_mel_filterbank(sample_rate, padded_window_size, num_mels, low_freq, high_freq)).first->second;
}

struct RealDftTables {
    std::vector<double> cos;
    std::vector<double> sin;
};

const RealDftTables & cached_real_dft_tables_512() {
    static const RealDftTables tables = [] {
        constexpr int64_t kFft = 512;
        constexpr int64_t kFreqBins = kFft / 2 + 1;
        constexpr double kPi = 3.14159265358979323846264338327950288;
        RealDftTables out;
        out.cos.resize(static_cast<size_t>(kFreqBins * kFft));
        out.sin.resize(static_cast<size_t>(kFreqBins * kFft));
        for (int64_t freq = 0; freq < kFreqBins; ++freq) {
            for (int64_t n = 0; n < kFft; ++n) {
                const double angle = 2.0 * kPi * static_cast<double>(freq * n) / static_cast<double>(kFft);
                out.cos[static_cast<size_t>(freq * kFft + n)] = std::cos(angle);
                out.sin[static_cast<size_t>(freq * kFft + n)] = std::sin(angle);
            }
        }
        return out;
    }();
    return tables;
}

std::vector<float> require_mono_samples(const std::vector<float> & samples, int channels) {
    if (channels <= 0) {
        throw std::runtime_error("IndexTTS2 audio channel count must be positive");
    }
    if (samples.empty()) {
        throw std::runtime_error("IndexTTS2 audio must not be empty");
    }
    if (samples.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("IndexTTS2 audio sample count must be divisible by channel count");
    }
    if (channels == 1) {
        return samples;
    }
    return engine::audio::mixdown_interleaved_to_mono_average(samples, channels);
}

std::vector<float> resample_mono(const std::vector<float> & input, int input_sample_rate, int output_sample_rate) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("IndexTTS2 resampling requires positive sample rates");
    }
    if (input_sample_rate == output_sample_rate || input.empty()) {
        return input;
    }
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float32ComputationStoredAsFloat32;
    options.accumulation = engine::audio::TorchaudioSincHannAccumulation::Float32;
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, input_sample_rate, output_sample_rate, options);
}

std::vector<float> resample_mono_librosa(const std::vector<float> & input, int input_sample_rate, int output_sample_rate) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("IndexTTS2 librosa-style resampling requires positive sample rates");
    }
    if (input_sample_rate == output_sample_rate || input.empty()) {
        return input;
    }
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::ExplicitFloat32Runtime;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.require_full_input = true;
    if (auto output = engine::audio::try_resample_mono_soxr(input, input_sample_rate, output_sample_rate, options)) {
        return *output;
    }
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, input_sample_rate, output_sample_rate);
}

}  // namespace

IndexTTS2MelOutput compute_index_tts2_mel_spectrogram(
    const std::vector<float> & waveform,
    const IndexTTS2S2MelConfig & config,
    size_t threads) {
    if (config.sample_rate <= 0 || config.n_fft <= 0 || config.win_length <= 0 ||
        config.hop_length <= 0 || config.n_mels <= 0) {
        throw std::runtime_error("IndexTTS2 mel spectrogram config is invalid");
    }
    if (waveform.empty()) {
        throw std::runtime_error("IndexTTS2 mel spectrogram requires non-empty waveform");
    }

    const int64_t pad = (config.n_fft - config.hop_length) / 2;
    const auto padded = engine::audio::reflect_pad_samples(waveform, pad, pad);
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        false,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    auto magnitude = engine::audio::STFT().compute_magnitude(
        padded,
        window,
        1,
        static_cast<int64_t>(padded.size()),
        stft_config,
        threads);
    for (float & value : magnitude.values) {
        value = std::sqrt(value * value + 1.0e-9F);
    }

    const auto & filterbank = cached_mel_filterbank(config);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    if (static_cast<int64_t>(filterbank.size()) != config.n_mels * freq_bins) {
        throw std::runtime_error("IndexTTS2 mel filterbank shape mismatch");
    }

    IndexTTS2MelOutput output;
    output.channels = config.n_mels;
    output.frames = frames;
    output.values.assign(static_cast<size_t>(config.n_mels * frames), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(config.n_mels * frames >= 4096)
#endif
    for (int64_t mel = 0; mel < config.n_mels; ++mel) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0F;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                sum += filterbank[static_cast<size_t>(mel * freq_bins + freq)] *
                    magnitude.values[static_cast<size_t>(freq * frames + frame)];
            }
            output.values[static_cast<size_t>(mel * frames + frame)] = std::log(std::max(sum, 1.0e-5F));
        }
    }
    return output;
}

IndexTTS2FbankOutput compute_index_tts2_campplus_fbank_16k(const std::vector<float> & waveform_16k) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kWindowSize = 400;
    constexpr int64_t kWindowShift = 160;
    constexpr int64_t kPaddedWindowSize = 512;
    constexpr int64_t kNumMels = 80;
    constexpr float kLowFreq = 20.0F;
    constexpr float kHighFreq = 0.0F;
    constexpr float kPreemphasis = 0.97F;
    constexpr float kEpsilon = std::numeric_limits<float>::epsilon();

    if (static_cast<int64_t>(waveform_16k.size()) < kWindowSize) {
        throw std::runtime_error("IndexTTS2 CAMPPlus fbank requires at least one 25 ms frame");
    }

    const int64_t frames = 1 + (static_cast<int64_t>(waveform_16k.size()) - kWindowSize) / kWindowShift;
    const auto & window = cached_povey_window(kWindowSize);
    const auto & mel_filterbank = cached_kaldi_mel_filterbank(
        kSampleRate,
        kPaddedWindowSize,
        kNumMels,
        kLowFreq,
        kHighFreq);

    std::vector<float> frame(static_cast<size_t>(kWindowSize), 0.0F);
    std::vector<float> stft_batch(static_cast<size_t>(frames * kPaddedWindowSize), 0.0F);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        const int64_t start = frame_index * kWindowShift;
        float mean = 0.0F;
        for (int64_t i = 0; i < kWindowSize; ++i) {
            const float sample = waveform_16k[static_cast<size_t>(start + i)];
            frame[static_cast<size_t>(i)] = sample;
            mean += sample;
        }
        mean /= static_cast<float>(kWindowSize);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            frame[static_cast<size_t>(i)] -= mean;
        }
        for (int64_t i = kWindowSize - 1; i > 0; --i) {
            frame[static_cast<size_t>(i)] -= kPreemphasis * frame[static_cast<size_t>(i - 1)];
        }
        frame[0] -= kPreemphasis * frame[0];
        for (int64_t i = 0; i < kWindowSize; ++i) {
            stft_batch[static_cast<size_t>(frame_index * kPaddedWindowSize + i)] =
                frame[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }
    }

    std::vector<float> stft_window(static_cast<size_t>(kPaddedWindowSize), 1.0F);
    const engine::audio::STFTConfig stft_config{
        kPaddedWindowSize,
        kPaddedWindowSize,
        kPaddedWindowSize,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        stft_batch,
        stft_window,
        frames,
        kPaddedWindowSize,
        stft_config);

    const int64_t freq_bins = (kPaddedWindowSize / 2) + 1;
    IndexTTS2FbankOutput output;
    output.frames = frames;
    output.dims = kNumMels;
    output.values.assign(static_cast<size_t>(frames * kNumMels), 0.0F);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
            float energy = 0.0F;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(frame_index * freq_bins + freq)];
                energy += (mag * mag) * mel_filterbank[static_cast<size_t>(mel_bin * freq_bins + freq)];
            }
            output.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)] =
                std::log(std::max(energy, kEpsilon));
        }
    }

    for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
        float mean = 0.0F;
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            mean += output.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)];
        }
        mean /= static_cast<float>(frames);
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            output.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)] -= mean;
        }
    }
    return output;
}

IndexTTS2SemanticFeatureOutput compute_index_tts2_semantic_features_16k(const std::vector<float> & waveform_16k) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kWindowSize = 400;
    constexpr int64_t kWindowShift = 160;
    constexpr int64_t kFftSize = 512;
    constexpr int64_t kFreqBins = kFftSize / 2 + 1;
    constexpr int64_t kNumMels = 80;
    constexpr float kLowFreq = 20.0F;
    constexpr float kHighFreq = 8000.0F;
    constexpr float kPreemphasis = 0.97F;
    constexpr double kInputScale = 32768.0;
    constexpr double kMelFloor = 1.192092955078125e-07;

    if (static_cast<int64_t>(waveform_16k.size()) < kWindowSize) {
        throw std::runtime_error("IndexTTS2 semantic fbank requires at least one 25 ms frame");
    }

    const int64_t frames = 1 + (static_cast<int64_t>(waveform_16k.size()) - kWindowSize) / kWindowShift;
    const auto & window = cached_povey_window(kWindowSize);
    const auto & mel_filterbank = cached_kaldi_mel_filterbank(
        kSampleRate,
        kFftSize,
        kNumMels,
        kLowFreq,
        kHighFreq);
    const auto & dft = cached_real_dft_tables_512();

    IndexTTS2FbankOutput fbank;
    fbank.frames = frames;
    fbank.dims = kNumMels;
    fbank.values.assign(static_cast<size_t>(frames * kNumMels), 0.0F);

#ifdef _OPENMP
#pragma omp parallel for if(frames >= 8)
#endif
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        const int64_t start = frame_index * kWindowShift;
        double buffer[kFftSize] = {};
        double mean = 0.0;
        for (int64_t i = 0; i < kWindowSize; ++i) {
            const double sample = static_cast<double>(waveform_16k[static_cast<size_t>(start + i)]) * kInputScale;
            buffer[i] = sample;
            mean += sample;
        }
        mean /= static_cast<double>(kWindowSize);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            buffer[i] -= mean;
        }
        for (int64_t i = kWindowSize - 1; i > 0; --i) {
            buffer[i] -= kPreemphasis * buffer[i - 1];
        }
        buffer[0] *= (1.0 - kPreemphasis);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            buffer[i] *= static_cast<double>(window[static_cast<size_t>(i)]);
        }

        double power[kFreqBins] = {};
        for (int64_t freq = 0; freq < kFreqBins; ++freq) {
            double re = 0.0;
            double im = 0.0;
            const size_t table_offset = static_cast<size_t>(freq * kFftSize);
            for (int64_t n = 0; n < kFftSize; ++n) {
                const double sample = buffer[n];
                re += sample * dft.cos[table_offset + static_cast<size_t>(n)];
                im -= sample * dft.sin[table_offset + static_cast<size_t>(n)];
            }
            power[freq] = re * re + im * im;
        }

        for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
            double energy = 0.0;
            for (int64_t freq = 0; freq < kFreqBins; ++freq) {
                energy += static_cast<double>(mel_filterbank[static_cast<size_t>(mel_bin * kFreqBins + freq)]) *
                    power[freq];
            }
            fbank.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)] =
                static_cast<float>(std::log(std::max(energy, kMelFloor)));
        }
    }

    for (int64_t mel_bin = 0; mel_bin < fbank.dims; ++mel_bin) {
        double mean = 0.0;
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            mean += static_cast<double>(fbank.values[static_cast<size_t>(frame * fbank.dims + mel_bin)]);
        }
        mean /= static_cast<double>(fbank.frames);

        double variance = 0.0;
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            const double diff =
                static_cast<double>(fbank.values[static_cast<size_t>(frame * fbank.dims + mel_bin)]) - mean;
            variance += diff * diff;
        }
        variance = fbank.frames > 1 ? variance / static_cast<double>(fbank.frames - 1) : 0.0;
        const float scale = static_cast<float>(1.0 / std::sqrt(variance + 1.0e-7));
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            float & value = fbank.values[static_cast<size_t>(frame * fbank.dims + mel_bin)];
            value = (value - static_cast<float>(mean)) * scale;
        }
    }

    const int64_t padded_frames = fbank.frames + (fbank.frames % 2);
    std::vector<float> padded(static_cast<size_t>(padded_frames * fbank.dims), 1.0F);
    std::copy(fbank.values.begin(), fbank.values.end(), padded.begin());

    IndexTTS2SemanticFeatureOutput output;
    output.frames = padded_frames / 2;
    output.dims = fbank.dims * 2;
    output.values.assign(static_cast<size_t>(output.frames * output.dims), 0.0F);
    output.attention_mask.assign(static_cast<size_t>(output.frames), 0);
    for (int64_t pair = 0; pair < output.frames; ++pair) {
        const int64_t first_frame = pair * 2;
        const int64_t second_frame = first_frame + 1;
        std::copy_n(
            padded.data() + static_cast<size_t>(first_frame * fbank.dims),
            static_cast<size_t>(fbank.dims),
            output.values.data() + static_cast<size_t>(pair * output.dims));
        std::copy_n(
            padded.data() + static_cast<size_t>(second_frame * fbank.dims),
            static_cast<size_t>(fbank.dims),
            output.values.data() + static_cast<size_t>(pair * output.dims + fbank.dims));
        output.attention_mask[static_cast<size_t>(pair)] = second_frame < fbank.frames ? 1 : 0;
    }
    return output;
}

IndexTTS2PreparedReferenceAudio prepare_index_tts2_reference_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    const IndexTTS2S2MelConfig & mel_config,
    size_t threads,
    bool speaker_load_semantic) {
    constexpr int64_t kMaxReferenceSeconds = 15;
    auto mono = require_mono_samples(samples, channels);
    const int64_t max_input_samples = static_cast<int64_t>(sample_rate) * kMaxReferenceSeconds;
    if (max_input_samples > 0 && static_cast<int64_t>(mono.size()) > max_input_samples) {
        engine::audio::truncate_samples_to_count(mono, static_cast<size_t>(max_input_samples));
    }

    IndexTTS2PreparedReferenceAudio output;
    output.waveform_22k = resample_mono_librosa(mono, sample_rate, mel_config.sample_rate);
    output.waveform_16k = speaker_load_semantic
        ? resample_mono(output.waveform_22k, mel_config.sample_rate, 16000)
        : resample_mono_librosa(mono, sample_rate, 16000);
    output.mel = compute_index_tts2_mel_spectrogram(output.waveform_22k, mel_config, threads);
    output.campplus_fbank = compute_index_tts2_campplus_fbank_16k(output.waveform_16k);
    output.semantic_features = compute_index_tts2_semantic_features_16k(output.waveform_16k);
    return output;
}

}  // namespace engine::models::index_tts2
