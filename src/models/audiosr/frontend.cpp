#include "engine/models/audiosr/frontend.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/runtime/host_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::audiosr {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::vector<float> mono_audio(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("AudioSR input audio sample rate and channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("AudioSR input audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("AudioSR input audio samples must be divisible by channel count");
    }
    if (audio.channels == 1) {
        return audio.samples;
    }
    return engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

void normalize_like_reference(std::vector<float> & samples) {
    if (samples.empty()) {
        throw std::runtime_error("AudioSR normalization received empty audio");
    }
    double mean = 0.0;
    for (float sample : samples) {
        mean += static_cast<double>(sample);
    }
    mean /= static_cast<double>(samples.size());
    float peak = 0.0F;
    for (float & sample : samples) {
        sample -= static_cast<float>(mean);
        peak = std::max(peak, std::abs(sample));
    }
    const float scale = 0.5F / (peak + 1.0e-8F);
    for (float & sample : samples) {
        sample *= scale;
    }
}

int64_t padded_samples_for_reference(int64_t samples, const AudioSRConfig & config) {
    const int64_t unit = static_cast<int64_t>(config.sample_rate) *
        static_cast<int64_t>(config.segment_seconds) /
        static_cast<int64_t>(config.segment_seconds_divisor);
    if (unit <= 0) {
        throw std::runtime_error("AudioSR segment unit is invalid");
    }
    return ((samples + unit - 1) / unit) * unit;
}

struct AudioSRFeatureTensors {
    std::vector<float> mel;
    std::vector<float> stft;
    int64_t frames = 0;
};

AudioSRFeatureTensors extract_features(
    const std::vector<float> & waveform,
    const AudioSRConfig & config,
    int64_t target_frames,
    size_t threads) {
    const int64_t reflect = (config.n_fft - config.hop_length) / 2;
    if (reflect <= 0 || static_cast<int64_t>(waveform.size()) <= reflect) {
        throw std::runtime_error("AudioSR log-mel input is too short for reference reflect padding");
    }
    std::vector<float> padded(static_cast<size_t>(static_cast<int64_t>(waveform.size()) + 2 * reflect));
    for (int64_t i = 0; i < reflect; ++i) {
        padded[static_cast<size_t>(i)] = waveform[static_cast<size_t>(reflect - i)];
    }
    std::copy(waveform.begin(), waveform.end(), padded.begin() + reflect);
    for (int64_t i = 0; i < reflect; ++i) {
        padded[static_cast<size_t>(reflect + static_cast<int64_t>(waveform.size()) + i)] =
            waveform[static_cast<size_t>(static_cast<int64_t>(waveform.size()) - 2 - i)];
    }
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const engine::audio::STFT stft;
    const auto mag = stft.compute_magnitude(
        padded,
        window,
        1,
        static_cast<int64_t>(padded.size()),
        stft_config,
        threads);
    const engine::audio::MelFilterbankConfig mel_config{
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.mel_fmin,
        config.mel_fmax,
        true,
    };
    const auto filterbank = engine::audio::MelFilterbank().build(mel_config);
    auto mel = engine::audio::MelFilterbank().compute_custom(
        mag.values,
        1,
        config.n_fft / 2 + 1,
        mag.shape.at(2),
        filterbank);
    for (float & value : mel.values) {
        value = std::log(std::max(value, 1.0e-5F));
    }

    AudioSRFeatureTensors out;
    out.frames = target_frames;
    out.mel.assign(static_cast<size_t>(target_frames * config.n_mels), 0.0F);
    out.stft.assign(static_cast<size_t>(target_frames * (config.n_fft / 2)), 0.0F);
    const int64_t source_frames = mag.shape.at(2);
    const int64_t copied_frames = std::min(target_frames, source_frames);
    for (int64_t frame = 0; frame < copied_frames; ++frame) {
        for (int64_t bin = 0; bin < config.n_mels; ++bin) {
            out.mel[static_cast<size_t>(frame * config.n_mels + bin)] =
                mel.values[static_cast<size_t>(bin * source_frames + frame)];
        }
        for (int64_t bin = 0; bin < config.n_fft / 2; ++bin) {
            out.stft[static_cast<size_t>(frame * (config.n_fft / 2) + bin)] =
                mag.values[static_cast<size_t>(bin * source_frames + frame)];
        }
    }
    return out;
}

struct CutoffInfo {
    int64_t bin = 0;
    float hz = 0.0F;
};

CutoffInfo locate_cutoff(const std::vector<float> & stft, int64_t frames, const AudioSRConfig & config) {
    const int64_t bins = config.n_fft / 2;
    if (frames <= 0 || static_cast<int64_t>(stft.size()) != frames * bins) {
        throw std::runtime_error("AudioSR cutoff detection received invalid STFT shape");
    }
    std::vector<double> energy(static_cast<size_t>(bins), 0.0);
    for (int64_t bin = 0; bin < bins; ++bin) {
        double sum = 0.0;
        for (int64_t frame = 0; frame < frames; ++frame) {
            sum += std::abs(static_cast<double>(stft[static_cast<size_t>(frame * bins + bin)]));
        }
        energy[static_cast<size_t>(bin)] = sum + (bin == 0 ? 0.0 : energy[static_cast<size_t>(bin - 1)]);
    }
    const double threshold = energy.back() * 0.985;
    int64_t cutoff_bin = 0;
    for (int64_t i = 1; i <= bins; ++i) {
        const int64_t bin = bins - i;
        if (energy[static_cast<size_t>(bin)] < threshold) {
            cutoff_bin = bin;
            break;
        }
    }
    float cutoff = static_cast<float>(cutoff_bin) / static_cast<float>(bins) *
        (0.5F * static_cast<float>(config.sample_rate));
    if (cutoff < 1000.0F) {
        cutoff_bin = bins;
        cutoff = 0.5F * static_cast<float>(config.sample_rate);
    }
    return {cutoff_bin, cutoff};
}

std::vector<double> odd_extension(const std::vector<double> & input, int64_t edge) {
    if (edge <= 0) {
        return input;
    }
    if (static_cast<int64_t>(input.size()) <= edge) {
        throw std::runtime_error("AudioSR sosfiltfilt input is shorter than padding edge");
    }
    std::vector<double> out(static_cast<size_t>(static_cast<int64_t>(input.size()) + 2 * edge));
    const double first = input.front();
    const double last = input.back();
    for (int64_t i = 0; i < edge; ++i) {
        out[static_cast<size_t>(i)] = 2.0 * first - input[static_cast<size_t>(edge - i)];
    }
    std::copy(input.begin(), input.end(), out.begin() + edge);
    const int64_t size = static_cast<int64_t>(input.size());
    for (int64_t i = 0; i < edge; ++i) {
        out[static_cast<size_t>(edge + size + i)] = 2.0 * last - input[static_cast<size_t>(size - 2 - i)];
    }
    return out;
}

std::array<double, 2> lfilter_zi_second_order(const float * sos) {
    const double b0 = sos[0];
    const double b1 = sos[1];
    const double b2 = sos[2];
    const double a0 = sos[3];
    const double a1 = sos[4] / a0;
    const double a2 = sos[5] / a0;
    const double nb0 = b0 / a0;
    const double nb1 = b1 / a0;
    const double nb2 = b2 / a0;
    const double rhs0 = nb1 - a1 * nb0;
    const double rhs1 = nb2 - a2 * nb0;
    const double z0 = (rhs0 + rhs1) / (1.0 + a1 + a2);
    const double z1 = rhs1 - a2 * z0;
    return {z0, z1};
}

std::vector<std::array<double, 2>> sosfilt_zi(const float * sos, int64_t sections) {
    std::vector<std::array<double, 2>> zi(static_cast<size_t>(sections));
    double scale = 1.0;
    for (int64_t section = 0; section < sections; ++section) {
        const float * coeff = sos + section * 6;
        const auto section_zi = lfilter_zi_second_order(coeff);
        zi[static_cast<size_t>(section)] = {scale * section_zi[0], scale * section_zi[1]};
        const double bsum = static_cast<double>(coeff[0]) + static_cast<double>(coeff[1]) + static_cast<double>(coeff[2]);
        const double asum = static_cast<double>(coeff[3]) + static_cast<double>(coeff[4]) + static_cast<double>(coeff[5]);
        scale *= bsum / asum;
    }
    return zi;
}

std::vector<double> sosfilt(const float * sos, int64_t sections, const std::vector<double> & input, double initial_value) {
    auto zi_base = sosfilt_zi(sos, sections);
    for (auto & item : zi_base) {
        item[0] *= initial_value;
        item[1] *= initial_value;
    }
    std::vector<double> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        double value = input[i];
        for (int64_t section = 0; section < sections; ++section) {
            const float * coeff = sos + section * 6;
            const double b0 = coeff[0] / coeff[3];
            const double b1 = coeff[1] / coeff[3];
            const double b2 = coeff[2] / coeff[3];
            const double a1 = coeff[4] / coeff[3];
            const double a2 = coeff[5] / coeff[3];
            auto & state = zi_base[static_cast<size_t>(section)];
            const double y = b0 * value + state[0];
            state[0] = b1 * value - a1 * y + state[1];
            state[1] = b2 * value - a2 * y;
            value = y;
        }
        out[i] = value;
    }
    return out;
}

std::vector<float> sosfiltfilt(const std::vector<float> & input, const float * sos, int64_t sections) {
    int64_t zero_b2 = 0;
    int64_t zero_a2 = 0;
    for (int64_t section = 0; section < sections; ++section) {
        const float * coeff = sos + section * 6;
        if (coeff[2] == 0.0F) {
            ++zero_b2;
        }
        if (coeff[5] == 0.0F) {
            ++zero_a2;
        }
    }
    const int64_t ntaps = 2 * sections + 1 - std::min(zero_b2, zero_a2);
    const int64_t edge = 3 * ntaps;
    std::vector<double> work(input.begin(), input.end());
    work = odd_extension(work, edge);
    work = sosfilt(sos, sections, work, work.front());
    std::reverse(work.begin(), work.end());
    work = sosfilt(sos, sections, work, work.front());
    std::reverse(work.begin(), work.end());
    std::vector<float> out(input.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(work[static_cast<size_t>(edge) + i]);
    }
    return out;
}

double bessel_i0(double x) {
    const double ax = std::abs(x);
    if (ax < 3.75) {
        const double y = (x / 3.75) * (x / 3.75);
        return 1.0 +
            y * (3.5156229 +
            y * (3.0899424 +
            y * (1.2067492 +
            y * (0.2659732 +
            y * (0.0360768 +
            y * 0.0045813)))));
    }
    const double y = 3.75 / ax;
    return (std::exp(ax) / std::sqrt(ax)) *
        (0.39894228 +
        y * (0.01328592 +
        y * (0.00225319 +
        y * (-0.00157565 +
        y * (0.00916281 +
        y * (-0.02057706 +
        y * (0.02635537 +
        y * (-0.01647633 +
        y * 0.00392377))))))));
}

std::vector<double> firwin_kaiser(int64_t taps, double cutoff, double beta) {
    if (taps <= 0 || taps % 2 == 0) {
        throw std::runtime_error("AudioSR resample_poly requires an odd positive FIR length");
    }
    const int64_t half = (taps - 1) / 2;
    const double denom = bessel_i0(beta);
    std::vector<double> h(static_cast<size_t>(taps));
    for (int64_t i = 0; i < taps; ++i) {
        const int64_t centered = i - half;
        const double sinc_arg = cutoff * static_cast<double>(centered);
        const double sinc = centered == 0
            ? 1.0
            : std::sin(kPi * sinc_arg) / (kPi * sinc_arg);
        const double ratio = static_cast<double>(centered) / static_cast<double>(half);
        const double window = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
        h[static_cast<size_t>(i)] = cutoff * sinc * window;
    }
    const double sum = std::accumulate(h.begin(), h.end(), 0.0);
    for (double & value : h) {
        value /= sum;
    }
    return h;
}

std::vector<float> resample_poly_reference(const std::vector<float> & input, int up, int down) {
    if (up <= 0 || down <= 0) {
        throw std::runtime_error("AudioSR resample_poly requires positive rates");
    }
    const int divisor = std::gcd(up, down);
    up /= divisor;
    down /= divisor;
    if (up == 1 && down == 1) {
        return input;
    }
    const int64_t n_in = static_cast<int64_t>(input.size());
    const int64_t n_out = (n_in * up + down - 1) / down;
    const int64_t max_rate = std::max(up, down);
    const int64_t half_len = 10 * max_rate;
    auto h = firwin_kaiser(2 * half_len + 1, 1.0 / static_cast<double>(max_rate), 5.0);
    for (double & value : h) {
        value *= static_cast<double>(up);
    }
    const int64_t n_pre_pad = down - (half_len % down);
    const int64_t n_pre_remove = (half_len + n_pre_pad) / down;
    std::vector<double> padded(static_cast<size_t>(n_pre_pad) + h.size(), 0.0);
    std::copy(h.begin(), h.end(), padded.begin() + n_pre_pad);
    h = std::move(padded);
    std::vector<float> out(static_cast<size_t>(n_out), 0.0F);
    for (int64_t out_index = 0; out_index < n_out; ++out_index) {
        const int64_t raw_index = out_index + n_pre_remove;
        const int64_t center = raw_index * down;
        int64_t coeff_index = center % up;
        double sum = 0.0;
        for (; coeff_index < static_cast<int64_t>(h.size()); coeff_index += up) {
            const int64_t input_index = (center - coeff_index) / up;
            if (input_index >= 0 && input_index < n_in) {
                sum += h[static_cast<size_t>(coeff_index)] * static_cast<double>(input[static_cast<size_t>(input_index)]);
            }
        }
        out[static_cast<size_t>(out_index)] = static_cast<float>(sum);
    }
    return out;
}

int filter_index_for_seed(uint32_t seed, uint64_t offset) {
    std::mt19937 rng(seed);
    for (uint64_t i = 0; i < offset; ++i) {
        (void)rng();
    }
    return static_cast<int>(rng() % 4U);
}

std::vector<float> lowpass_reference(
    const std::vector<float> & waveform,
    int64_t cutoff_bin,
    const AudioSRConfig & config,
    const std::vector<float> & sos_table,
    const std::vector<float> & sos_valid,
    uint32_t seed,
    uint64_t lowpass_rng_offset) {
    if (cutoff_bin < 0 || cutoff_bin > 1024 ||
        sos_valid.size() != 1025 ||
        sos_table.size() != static_cast<size_t>(4 * 1025 * 4 * 6)) {
        throw std::runtime_error("AudioSR lowpass filter table shape is invalid");
    }
    if (sos_valid[static_cast<size_t>(cutoff_bin)] == 0.0F) {
        return waveform;
    }
    const int filter_index = filter_index_for_seed(seed, lowpass_rng_offset);
    const size_t sos_offset = static_cast<size_t>(((filter_index * 1025 + cutoff_bin) * 4) * 6);
    const float * sos = sos_table.data() + sos_offset;
    auto filtered = sosfiltfilt(waveform, sos, 4);
    const int down_rate = static_cast<int>(std::floor(2.0F * (static_cast<float>(cutoff_bin) / 1024.0F) * (0.5F * static_cast<float>(config.sample_rate))));
    if (down_rate <= 0 || down_rate >= config.sample_rate) {
        return filtered;
    }
    filtered = resample_poly_reference(filtered, down_rate, config.sample_rate);
    filtered = resample_poly_reference(filtered, config.sample_rate, down_rate);
    filtered.resize(waveform.size(), 0.0F);
    filtered = sosfiltfilt(filtered, sos, 4);
    filtered.resize(waveform.size(), 0.0F);
    return filtered;
}

}  // namespace

AudioSRFrontend::AudioSRFrontend(const AudioSRAssets & assets)
    : config_(assets.config),
      lowpass_sos_(assets.frontend_lowpass_sos),
      lowpass_sos_valid_(assets.frontend_lowpass_sos_valid) {}

AudioSRFrontendOutput AudioSRFrontend::compute(
    const engine::runtime::AudioBuffer & audio,
    uint32_t seed,
    uint64_t lowpass_rng_offset,
    size_t threads) const {
    auto mono = mono_audio(audio);
    if (audio.sample_rate != config_.sample_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            config_.sample_rate,
            engine::audio::torchaudio_sinc_hann_float32_options());
    }
    const int64_t original_samples = static_cast<int64_t>(mono.size());
    normalize_like_reference(mono);
    const int64_t target_samples = padded_samples_for_reference(static_cast<int64_t>(mono.size()), config_);
    mono.resize(static_cast<size_t>(target_samples), 0.0F);

    AudioSRFrontendOutput out;
    out.waveform = std::move(mono);
    out.samples = target_samples;
    out.original_samples = original_samples;
    const int64_t target_frames = target_samples / config_.hop_length;
    auto full_features = extract_features(out.waveform, config_, target_frames, threads);
    out.mel = std::move(full_features.mel);
    out.mel_frames = full_features.frames;
    out.latent_time = out.mel_frames / 8;
    const auto cutoff = locate_cutoff(full_features.stft, full_features.frames, config_);
    out.lowpass_waveform = lowpass_reference(
        out.waveform,
        cutoff.bin,
        config_,
        lowpass_sos_,
        lowpass_sos_valid_,
        seed,
        lowpass_rng_offset);
    out.lowpass_mel = extract_features(out.lowpass_waveform, config_, target_frames, threads).mel;
    return out;
}

}  // namespace engine::models::audiosr
