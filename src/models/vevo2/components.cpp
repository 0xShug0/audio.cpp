#include "engine/models/vevo2/components.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::models::vevo2 {

struct Vevo2CocoConvNeXtBlockWeights {
    engine::modules::DepthwiseConv1dWeights dwconv;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pwconv1;
    engine::modules::LinearWeights pwconv2;
    engine::core::TensorValue gamma;
};

struct Vevo2CocoVocosEncoderWeights {
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<Vevo2CocoConvNeXtBlockWeights> convnext;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights output;
};

struct Vevo2CocoFvqWeights {
    engine::modules::Conv1dWeights in_project;
    engine::core::TensorValue codebook;
    engine::modules::Conv1dWeights out_project;
};

struct Vevo2CocoTokenizerWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::optional<engine::modules::LinearWeights> whisper_input;
    std::optional<engine::modules::LinearWeights> chromagram_input;
    std::vector<engine::modules::Conv1dWeights> downsample_layers;
    Vevo2CocoVocosEncoderWeights encoder;
    Vevo2CocoFvqWeights quantizer;
};

namespace {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t require_power_of_two_log2(int64_t value, const char * name) {
    if (value <= 0 || (value & (value - 1)) != 0) {
        throw std::runtime_error(std::string(name) + " must be a positive power of two");
    }
    int64_t log2 = 0;
    while (value > 1) {
        value >>= 1;
        ++log2;
    }
    return log2;
}

int64_t conv1d_output_frames(int64_t frames, int64_t kernel, int64_t stride, int64_t padding, int64_t dilation = 1) {
    return (frames + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t coco_output_frames(int64_t input_frames, int64_t downsample_rate) {
    int64_t frames = input_frames;
    const int64_t layers = require_power_of_two_log2(downsample_rate, "Vevo2 Coco downsample_rate");
    for (int64_t layer = 0; layer < layers; ++layer) {
        frames = conv1d_output_frames(frames, 3, 2, 1);
    }
    if (frames <= 0) {
        throw std::runtime_error("Vevo2 Coco tokenizer produced non-positive output frame count");
    }
    return frames;
}

double hz_to_octs(double frequency, double tuning, int64_t bins_per_octave) {
    const double a440 = 440.0 * std::pow(2.0, tuning / static_cast<double>(bins_per_octave));
    return std::log2(frequency / (a440 / 16.0));
}

double parabolic_shift(const std::vector<float> & power, int64_t freq_bins, int64_t frames, int64_t freq, int64_t frame) {
    if (freq <= 0 || freq >= freq_bins - 1) {
        return 0.0;
    }
    const auto at = [&](int64_t f) {
        return static_cast<double>(power[static_cast<size_t>(f * frames + frame)]);
    };
    const double a = at(freq + 1) + at(freq - 1) - 2.0 * at(freq);
    const double b = (at(freq + 1) - at(freq - 1)) / 2.0;
    if (std::abs(b) >= std::abs(a) || a == 0.0) {
        return 0.0;
    }
    return -b / a;
}

double gradient_along_frequency(
    const std::vector<float> & power,
    int64_t freq_bins,
    int64_t frames,
    int64_t freq,
    int64_t frame) {
    const auto at = [&](int64_t f) {
        return static_cast<double>(power[static_cast<size_t>(f * frames + frame)]);
    };
    if (freq == 0) {
        return at(1) - at(0);
    }
    if (freq == freq_bins - 1) {
        return at(freq_bins - 1) - at(freq_bins - 2);
    }
    return (at(freq + 1) - at(freq - 1)) / 2.0;
}

double estimate_chroma_tuning(
    const std::vector<float> & power,
    int64_t freq_bins,
    int64_t frames,
    int64_t sample_rate,
    int64_t n_fft,
    int64_t bins_per_octave) {
    std::vector<double> selected_pitches;
    std::vector<double> selected_magnitudes;
    selected_pitches.reserve(static_cast<size_t>(frames));
    selected_magnitudes.reserve(static_cast<size_t>(frames));

    for (int64_t frame = 0; frame < frames; ++frame) {
        double frame_max = 0.0;
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            frame_max = std::max(frame_max, static_cast<double>(power[static_cast<size_t>(freq * frames + frame)]));
        }
        const double threshold = 0.1 * frame_max;
        for (int64_t freq = 1; freq < freq_bins; ++freq) {
            const double fft_freq = static_cast<double>(freq) * static_cast<double>(sample_rate) /
                static_cast<double>(n_fft);
            if (fft_freq < 150.0 || fft_freq >= 4000.0) {
                continue;
            }
            const double value = static_cast<double>(power[static_cast<size_t>(freq * frames + frame)]);
            if (value <= threshold) {
                continue;
            }
            const double previous = static_cast<double>(power[static_cast<size_t>((freq - 1) * frames + frame)]);
            const bool is_local_max = freq == freq_bins - 1
                ? value > previous
                : (value > previous &&
                   value >= static_cast<double>(power[static_cast<size_t>((freq + 1) * frames + frame)]));
            if (!is_local_max) {
                continue;
            }
            const double shift = parabolic_shift(power, freq_bins, frames, freq, frame);
            const double dskew = 0.5 * gradient_along_frequency(power, freq_bins, frames, freq, frame) * shift;
            selected_pitches.push_back((static_cast<double>(freq) + shift) * static_cast<double>(sample_rate) /
                                       static_cast<double>(n_fft));
            selected_magnitudes.push_back(value + dskew);
        }
    }

    if (selected_pitches.empty()) {
        return 0.0;
    }
    std::vector<double> magnitudes = selected_magnitudes;
    const auto mid = magnitudes.begin() + static_cast<std::ptrdiff_t>(magnitudes.size() / 2);
    std::nth_element(magnitudes.begin(), mid, magnitudes.end());
    double median = *mid;
    if (magnitudes.size() % 2 == 0) {
        const auto prev = std::max_element(magnitudes.begin(), mid);
        median = (*prev + *mid) / 2.0;
    }

    constexpr double kResolution = 0.01;
    constexpr int64_t kBins = 100;
    std::array<int64_t, kBins> counts{};
    for (size_t i = 0; i < selected_pitches.size(); ++i) {
        if (selected_magnitudes[i] < median || selected_pitches[i] <= 0.0) {
            continue;
        }
        double residual = std::fmod(static_cast<double>(bins_per_octave) * hz_to_octs(selected_pitches[i], 0.0, bins_per_octave), 1.0);
        if (residual < 0.0) {
            residual += 1.0;
        }
        if (residual >= 0.5) {
            residual -= 1.0;
        }
        int64_t bin = static_cast<int64_t>(std::floor((residual + 0.5) / kResolution));
        bin = std::max<int64_t>(0, std::min<int64_t>(bin, kBins - 1));
        ++counts[static_cast<size_t>(bin)];
    }
    const auto best = std::max_element(counts.begin(), counts.end());
    if (*best == 0) {
        return 0.0;
    }
    return -0.5 + static_cast<double>(std::distance(counts.begin(), best)) * kResolution;
}

std::vector<float> build_chroma_filterbank(int64_t sample_rate, int64_t n_fft, int64_t n_chroma, double tuning) {
    const int64_t freq_bins = n_fft / 2 + 1;
    std::vector<double> frqbins(static_cast<size_t>(n_fft), 0.0);
    for (int64_t i = 1; i < n_fft; ++i) {
        const double frequency = static_cast<double>(sample_rate) * static_cast<double>(i) / static_cast<double>(n_fft);
        frqbins[static_cast<size_t>(i)] =
            static_cast<double>(n_chroma) * hz_to_octs(frequency, tuning, n_chroma);
    }
    frqbins[0] = frqbins[1] - 1.5 * static_cast<double>(n_chroma);

    std::vector<double> binwidth(static_cast<size_t>(n_fft), 1.0);
    for (int64_t i = 0; i < n_fft - 1; ++i) {
        binwidth[static_cast<size_t>(i)] = std::max(frqbins[static_cast<size_t>(i + 1)] - frqbins[static_cast<size_t>(i)], 1.0);
    }

    std::vector<double> full(static_cast<size_t>(n_chroma * n_fft), 0.0);
    const double half = std::round(static_cast<double>(n_chroma) / 2.0);
    for (int64_t chroma = 0; chroma < n_chroma; ++chroma) {
        for (int64_t freq = 0; freq < n_fft; ++freq) {
            double d = frqbins[static_cast<size_t>(freq)] - static_cast<double>(chroma);
            d = std::fmod(d + half + 10.0 * static_cast<double>(n_chroma), static_cast<double>(n_chroma)) - half;
            const double ratio = 2.0 * d / binwidth[static_cast<size_t>(freq)];
            full[static_cast<size_t>(chroma * n_fft + freq)] = std::exp(-0.5 * ratio * ratio);
        }
    }

    for (int64_t freq = 0; freq < n_fft; ++freq) {
        double norm = 0.0;
        for (int64_t chroma = 0; chroma < n_chroma; ++chroma) {
            const double value = full[static_cast<size_t>(chroma * n_fft + freq)];
            norm += value * value;
        }
        norm = std::sqrt(norm);
        if (norm <= std::numeric_limits<float>::min()) {
            norm = 1.0;
        }
        for (int64_t chroma = 0; chroma < n_chroma; ++chroma) {
            full[static_cast<size_t>(chroma * n_fft + freq)] /= norm;
        }
    }

    for (int64_t freq = 0; freq < n_fft; ++freq) {
        const double octave = frqbins[static_cast<size_t>(freq)] / static_cast<double>(n_chroma);
        const double octave_weight = std::exp(-0.5 * std::pow((octave - 5.0) / 2.0, 2.0));
        for (int64_t chroma = 0; chroma < n_chroma; ++chroma) {
            full[static_cast<size_t>(chroma * n_fft + freq)] *= octave_weight;
        }
    }

    const int64_t roll = -3 * (n_chroma / 12);
    std::vector<float> out(static_cast<size_t>(n_chroma * freq_bins), 0.0F);
    for (int64_t chroma = 0; chroma < n_chroma; ++chroma) {
        int64_t source_chroma = chroma - roll;
        source_chroma %= n_chroma;
        if (source_chroma < 0) {
            source_chroma += n_chroma;
        }
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            out[static_cast<size_t>(chroma * freq_bins + freq)] =
                static_cast<float>(full[static_cast<size_t>(source_chroma * n_fft + freq)]);
        }
    }
    return out;
}

std::vector<float> interpolate_chromagram(
    const std::vector<float> & chroma,
    int64_t frames,
    int64_t bins,
    int64_t target_frames) {
    if (target_frames == frames) {
        return chroma;
    }
    std::vector<float> out(static_cast<size_t>(target_frames * bins), 0.0F);
    if (target_frames <= 1 || frames <= 1) {
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            for (int64_t bin = 0; bin < bins; ++bin) {
                out[static_cast<size_t>(frame * bins + bin)] = chroma[static_cast<size_t>(bin)];
            }
        }
        return out;
    }
    const double scale = static_cast<double>(frames) / static_cast<double>(target_frames);
    for (int64_t out_frame = 0; out_frame < target_frames; ++out_frame) {
        const double in_pos = (static_cast<double>(out_frame) + 0.5) * scale - 0.5;
        const double clamped = std::max(0.0, std::min<double>(in_pos, static_cast<double>(frames - 1)));
        const int64_t left = static_cast<int64_t>(std::floor(clamped));
        const int64_t right = std::min<int64_t>(left + 1, frames - 1);
        const float frac = static_cast<float>(clamped - static_cast<double>(left));
        for (int64_t bin = 0; bin < bins; ++bin) {
            const float l = chroma[static_cast<size_t>(left * bins + bin)];
            const float r = chroma[static_cast<size_t>(right * bins + bin)];
            out[static_cast<size_t>(out_frame * bins + bin)] = l * (1.0F - frac) + r * frac;
        }
    }
    return out;
}

std::vector<float> fix_length(std::vector<float> input, size_t size) {
    if (input.size() > size) {
        input.resize(size);
    } else if (input.size() < size) {
        input.resize(size, 0.0F);
    }
    return input;
}

std::vector<float> resample_mono_linear_ratio(const std::vector<float> & input, double ratio) {
    if (ratio <= 0.0 || !std::isfinite(ratio)) {
        throw std::runtime_error("Vevo2 pitch shift requires a finite positive resample ratio");
    }
    if (input.empty()) {
        return {};
    }
    const size_t output_samples = static_cast<size_t>(std::llround(static_cast<double>(input.size()) * ratio));
    if (output_samples == 0) {
        return {};
    }
    if (ratio == 1.0) {
        return input;
    }
    std::vector<float> output(output_samples, 0.0F);
    for (size_t i = 0; i < output_samples; ++i) {
        const double src_pos = static_cast<double>(i) / ratio;
        const size_t left = static_cast<size_t>(std::floor(src_pos));
        const size_t right = std::min(left + 1, input.size() - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(left));
        output[i] = input[left] * (1.0F - frac) + input[right] * frac;
    }
    return output;
}

float wrap_phase(float phase) {
    constexpr float kTwoPi = 2.0F * 3.14159265358979323846264338327950288F;
    return phase - kTwoPi * std::round(phase / kTwoPi);
}

std::vector<float> librosa_like_time_stretch(const std::vector<float> & mono, double rate) {
    if (rate <= 0.0 || !std::isfinite(rate)) {
        throw std::runtime_error("Vevo2 pitch shift requires a finite positive stretch rate");
    }
    constexpr int64_t kNfft = 2048;
    constexpr int64_t kHop = 512;
    const engine::audio::STFTConfig config{
        kNfft,
        kHop,
        kNfft,
        true,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(config);
    const auto spectrum = engine::audio::STFT().compute_complex(
        mono,
        window,
        1,
        static_cast<int64_t>(mono.size()),
        config);
    const int64_t freq_bins = spectrum.shape[1];
    const int64_t frames = spectrum.shape[2];
    std::vector<double> time_steps;
    for (int64_t t = 0;; ++t) {
        const double step = static_cast<double>(t) * rate;
        if (step >= static_cast<double>(frames)) {
            break;
        }
        time_steps.push_back(step);
    }
    const int64_t out_frames = static_cast<int64_t>(time_steps.size());
    std::vector<float> stretched(static_cast<size_t>(freq_bins * out_frames * 2), 0.0F);
    std::vector<float> phase_acc(static_cast<size_t>(freq_bins), 0.0F);
    for (int64_t bin = 0; bin < freq_bins; ++bin) {
        const size_t base = static_cast<size_t>((bin * frames) * 2);
        phase_acc[static_cast<size_t>(bin)] = std::atan2(spectrum.values[base + 1], spectrum.values[base]);
    }

    constexpr float kTwoPi = 2.0F * 3.14159265358979323846264338327950288F;
    for (int64_t t = 0; t < out_frames; ++t) {
        const double step = time_steps[static_cast<size_t>(t)];
        const int64_t left_frame = static_cast<int64_t>(step);
        const int64_t right_frame = left_frame + 1;
        const float alpha = static_cast<float>(step - static_cast<double>(left_frame));
        for (int64_t bin = 0; bin < freq_bins; ++bin) {
            const auto read_complex = [&](int64_t frame) -> std::pair<float, float> {
                if (frame < 0 || frame >= frames) {
                    return {0.0F, 0.0F};
                }
                const size_t base = static_cast<size_t>(((bin * frames) + frame) * 2);
                return {spectrum.values[base], spectrum.values[base + 1]};
            };
            const auto left = read_complex(left_frame);
            const auto right = read_complex(right_frame);
            const float left_mag = std::hypot(left.first, left.second);
            const float right_mag = std::hypot(right.first, right.second);
            const float mag = (1.0F - alpha) * left_mag + alpha * right_mag;
            const float phase = phase_acc[static_cast<size_t>(bin)];
            const size_t out_base = static_cast<size_t>(((bin * out_frames) + t) * 2);
            stretched[out_base] = mag * std::cos(phase);
            stretched[out_base + 1] = mag * std::sin(phase);

            const float left_phase = std::atan2(left.second, left.first);
            const float right_phase = std::atan2(right.second, right.first);
            const float phi_advance = static_cast<float>(kHop) * kTwoPi * static_cast<float>(bin) /
                static_cast<float>(kNfft);
            const float dphase = wrap_phase(right_phase - left_phase - phi_advance);
            phase_acc[static_cast<size_t>(bin)] += phi_advance + dphase;
        }
    }

    const int64_t requested_samples = static_cast<int64_t>(std::llround(static_cast<double>(mono.size()) / rate));
    const int64_t min_safe_samples = std::max<int64_t>(1, (out_frames - 1) * kHop);
    const int64_t stretched_samples = std::max(requested_samples, min_safe_samples);
    return engine::audio::ISTFT().compute(
        stretched,
        window,
        1,
        freq_bins,
        out_frames,
        stretched_samples,
        config).values;
}

std::vector<float> librosa_like_pitch_shift_24k(const std::vector<float> & mono, int steps) {
    const double rate = std::pow(2.0, -static_cast<double>(steps) / 12.0);
    auto stretched = librosa_like_time_stretch(mono, rate);
    auto shifted = resample_mono_linear_ratio(stretched, rate);
    return fix_length(std::move(shifted), mono.size());
}

std::vector<float> compute_vevo2_chromagram(
    const runtime::AudioBuffer & audio,
    int64_t target_frames,
    double frame_len_ratio = 1.0,
    bool use_pitch_shifted_waveform = false,
    int pitch_shift_steps = 0) {
    constexpr int64_t kSampleRate = 24000;
    constexpr int64_t kNfft = 1920;
    constexpr int64_t kHop = 480;
    constexpr int64_t kChroma = 24;
    if (target_frames <= 0) {
        throw std::runtime_error("Vevo2 chromagram requires positive target frame count");
    }
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        24000);
    if (use_pitch_shifted_waveform) {
        mono = librosa_like_pitch_shift_24k(mono, pitch_shift_steps);
    }
    const engine::audio::STFTConfig stft_config{
        kNfft,
        kHop,
        kNfft,
        true,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        mono,
        window,
        1,
        static_cast<int64_t>(mono.size()),
        stft_config);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    std::vector<float> power(magnitude.values.size(), 0.0F);
    for (size_t i = 0; i < magnitude.values.size(); ++i) {
        power[i] = magnitude.values[i] * magnitude.values[i];
    }

    const double tuning = estimate_chroma_tuning(power, freq_bins, stft_frames, kSampleRate, kNfft, kChroma);
    const auto chroma_filter = build_chroma_filterbank(kSampleRate, kNfft, kChroma, tuning);

    std::vector<float> chroma(static_cast<size_t>(target_frames * kChroma), 0.0F);
    for (int64_t frame = 0; frame < target_frames; ++frame) {
        const int64_t source_frame = std::min(frame, stft_frames - 1);
        float frame_max = 0.0F;
        for (int64_t bin = 0; bin < kChroma; ++bin) {
            double sum = 0.0;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                sum += static_cast<double>(chroma_filter[static_cast<size_t>(bin * freq_bins + freq)]) *
                    static_cast<double>(power[static_cast<size_t>(freq * stft_frames + source_frame)]);
            }
            const float value = static_cast<float>(sum);
            chroma[static_cast<size_t>(frame * kChroma + bin)] = value;
            frame_max = std::max(frame_max, std::abs(value));
        }
        if (frame_max > std::numeric_limits<float>::min()) {
            for (int64_t bin = 0; bin < kChroma; ++bin) {
                chroma[static_cast<size_t>(frame * kChroma + bin)] /= frame_max;
            }
        }
    }

    if (frame_len_ratio != 1.0) {
        const int64_t resampled_frames = static_cast<int64_t>(static_cast<double>(target_frames) * frame_len_ratio);
        if (resampled_frames <= 0) {
            throw std::runtime_error("Vevo2 chromagram duration ratio produced no frames");
        }
        chroma = interpolate_chromagram(chroma, target_frames, kChroma, resampled_frames);
        target_frames = resampled_frames;
    }
    return chroma;
}

int64_t vevo2_frame_count_24k(const runtime::AudioBuffer & audio) {
    return static_cast<int64_t>(engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        24000).size()) / 480;
}

engine::core::TensorValue ensure_contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

engine::core::TensorValue repeat_like(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value,
    const engine::core::TensorValue & like) {
    return engine::core::wrap_tensor(ggml_repeat(ctx.ggml, value.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

engine::core::TensorValue scale_last_dim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & scale) {
    const auto view = engine::core::reshape_tensor(
        ctx,
        scale,
        engine::core::TensorShape::from_dims({1, 1, scale.shape.dims[0]}));
    const auto repeated = repeat_like(ctx, view, input);
    return engine::core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, repeated.tensor), input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue transpose_btc_to_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

engine::core::TensorValue transpose_bct_to_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

std::vector<float> effective_weight_norm_conv1d(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    const auto g = source.require_f32(prefix + ".weight_g", {out_channels, 1, 1});
    const auto v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel_size});
    std::vector<float> weight(v.size());
    const int64_t row_size = in_channels * kernel_size;
    for (int64_t out = 0; out < out_channels; ++out) {
        double sum = 0.0;
        const int64_t base = out * row_size;
        for (int64_t i = 0; i < row_size; ++i) {
            const float value = v[static_cast<size_t>(base + i)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("Vevo2 weight-norm tensor has zero norm: " + prefix);
        }
        const float scale = static_cast<float>(static_cast<double>(g[static_cast<size_t>(out)]) / norm);
        for (int64_t i = 0; i < row_size; ++i) {
            weight[static_cast<size_t>(base + i)] = v[static_cast<size_t>(base + i)] * scale;
        }
    }
    return weight;
}

std::vector<float> l2_normalized_rows(
    std::vector<float> values,
    int64_t rows,
    int64_t cols,
    const std::string & name) {
    if (rows <= 0 || cols <= 0 || static_cast<int64_t>(values.size()) != rows * cols) {
        throw std::runtime_error("Vevo2 L2 row normalization shape mismatch: " + name);
    }
    for (int64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        const int64_t base = row * cols;
        for (int64_t col = 0; col < cols; ++col) {
            const float value = values[static_cast<size_t>(base + col)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("Vevo2 L2 row normalization found zero norm: " + name);
        }
        const float inv = static_cast<float>(1.0 / norm);
        for (int64_t col = 0; col < cols; ++col) {
            values[static_cast<size_t>(base + col)] *= inv;
        }
    }
    return values;
}

std::pair<std::vector<float>, std::vector<float>> load_whisper_stats(
    const engine::assets::TensorSource & source,
    int64_t whisper_dim) {
    auto mean = source.require_f32("mean", {whisper_dim});
    auto std = source.require_f32("std", {whisper_dim});
    source.release_storage();
    return {std::move(mean), std::move(std)};
}

engine::modules::Conv1dWeights load_weight_norm_conv1d_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    bool use_bias) {
    engine::modules::Conv1dWeights weights;
    weights.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_channels, in_channels, kernel_size}),
        storage_type,
        effective_weight_norm_conv1d(source, prefix, out_channels, in_channels, kernel_size));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

Vevo2CocoVocosEncoderWeights load_coco_vocos_encoder_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const Vevo2CocoTokenizerConfig & config,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    Vevo2CocoVocosEncoderWeights weights;
    weights.embed = engine::modules::binding::conv1d_from_source(
        store,
        source,
        "encoder.0.embed",
        conv_storage_type,
        config.vocos_dim,
        config.hidden_size,
        7,
        true);
    weights.norm = engine::modules::binding::norm_from_source(store, source, "encoder.0.norm", config.vocos_dim);
    weights.convnext.reserve(static_cast<size_t>(config.vocos_num_layers));
    for (int64_t layer = 0; layer < config.vocos_num_layers; ++layer) {
        const std::string prefix = "encoder.0.convnext." + std::to_string(layer);
        Vevo2CocoConvNeXtBlockWeights block;
        block.dwconv = engine::modules::binding::depthwise_conv1d_from_source(
            store,
            source,
            prefix + ".dwconv",
            conv_storage_type,
            config.vocos_dim,
            7,
            true);
        block.norm = engine::modules::binding::norm_from_source(store, source, prefix + ".norm", config.vocos_dim);
        block.pwconv1 = engine::modules::binding::linear_from_source(
            store,
            source,
            prefix + ".pwconv1",
            matmul_storage_type,
            config.vocos_intermediate_dim,
            config.vocos_dim,
            true);
        block.pwconv2 = engine::modules::binding::linear_from_source(
            store,
            source,
            prefix + ".pwconv2",
            matmul_storage_type,
            config.vocos_dim,
            config.vocos_intermediate_dim,
            true);
        block.gamma = store.load_f32_tensor(source, prefix + ".gamma", {config.vocos_dim});
        weights.convnext.push_back(std::move(block));
    }
    weights.final_norm = engine::modules::binding::norm_from_source(store, source, "encoder.0.final_layer_norm", config.vocos_dim);
    weights.output = engine::modules::binding::linear_from_source(
        store,
        source,
        "encoder.1",
        matmul_storage_type,
        config.hidden_size,
        config.vocos_dim,
        true);
    return weights;
}

Vevo2CocoFvqWeights load_coco_fvq_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const Vevo2CocoTokenizerConfig & config,
    engine::assets::TensorStorageType conv_storage_type) {
    Vevo2CocoFvqWeights weights;
    constexpr const char * prefix = "quantizer.quantizers.0";
    weights.in_project = load_weight_norm_conv1d_weights(
        store,
        source,
        std::string(prefix) + ".in_project",
        conv_storage_type,
        config.codebook_dim,
        config.hidden_size,
        1,
        true);
    weights.codebook = store.make_f32(
        engine::core::TensorShape::from_dims({config.codebook_size, config.codebook_dim}),
        l2_normalized_rows(
            source.require_f32(std::string(prefix) + ".codebook.weight", {config.codebook_size, config.codebook_dim}),
            config.codebook_size,
            config.codebook_dim,
            std::string(prefix) + ".codebook.weight"));
    weights.out_project = load_weight_norm_conv1d_weights(
        store,
        source,
        std::string(prefix) + ".out_project",
        conv_storage_type,
        config.hidden_size,
        config.codebook_dim,
        1,
        true);
    return weights;
}

std::shared_ptr<const Vevo2CocoTokenizerWeights> load_coco_tokenizer_weights(
    const Vevo2CocoTokenizerConfig & config,
    const char * name,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    const engine::assets::TensorSource & source) {
    if (config.coco_type != "style" && config.coco_type != "content_style") {
        throw std::runtime_error("Vevo2 Coco tokenizer weights require style or content_style config");
    }
    auto weights = std::make_shared<Vevo2CocoTokenizerWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        std::string("vevo2.") + name + ".weights",
        weight_context_bytes);
    if (config.coco_type == "content_style") {
        weights->whisper_input = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            "whisper_input_layer",
            matmul_storage_type,
            config.hidden_size,
            config.whisper_dim,
            true);
    }
    weights->chromagram_input = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "chromagram_input_layer",
        matmul_storage_type,
        config.hidden_size,
        config.chromagram_dim,
        true);
    const int64_t downsample_layers = require_power_of_two_log2(config.downsample_rate, "Vevo2 Coco downsample_rate");
    weights->downsample_layers.reserve(static_cast<size_t>(downsample_layers));
    for (int64_t layer = 0; layer < downsample_layers; ++layer) {
        weights->downsample_layers.push_back(engine::modules::binding::conv1d_from_source(
            *weights->store,
            source,
            "downsample_layers." + std::to_string(layer * 2),
            conv_storage_type,
            config.hidden_size,
            config.hidden_size,
            3,
            true));
    }
    weights->encoder = load_coco_vocos_encoder_weights(
        *weights->store,
        source,
        config,
        matmul_storage_type,
        conv_storage_type);
    weights->quantizer = load_coco_fvq_weights(*weights->store, source, config, conv_storage_type);
    weights->store->upload();
    return weights;
}

engine::core::TensorValue build_coco_convnext_block(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const Vevo2CocoConvNeXtBlockWeights & weights,
    const Vevo2CocoTokenizerConfig & config) {
    auto hidden = engine::modules::DepthwiseConv1dModule({
        config.vocos_dim,
        7,
        1,
        3,
        1,
        weights.dwconv.bias.has_value(),
    }).build(ctx, input_bct, weights.dwconv);
    hidden = transpose_bct_to_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.vocos_dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = engine::modules::LinearModule({
        config.vocos_dim,
        config.vocos_intermediate_dim,
        true,
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.pwconv1);
    hidden = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = engine::modules::LinearModule({
        config.vocos_intermediate_dim,
        config.vocos_dim,
        true,
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.pwconv2);
    hidden = scale_last_dim(ctx, hidden, weights.gamma);
    hidden = transpose_btc_to_bct(ctx, hidden);
    return engine::modules::AddModule{}.build(ctx, input_bct, hidden);
}

engine::core::TensorValue build_coco_vocos_encoder(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const Vevo2CocoTokenizerWeights & weights,
    const Vevo2CocoTokenizerConfig & config) {
    auto hidden = engine::modules::Conv1dModule({
        config.hidden_size,
        config.vocos_dim,
        7,
        1,
        3,
        1,
        weights.encoder.embed.bias.has_value(),
    }).build(ctx, input_bct, weights.encoder.embed);
    hidden = transpose_bct_to_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.vocos_dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.encoder.norm);
    hidden = transpose_btc_to_bct(ctx, hidden);
    for (const auto & block : weights.encoder.convnext) {
        hidden = build_coco_convnext_block(ctx, hidden, block, config);
    }
    hidden = transpose_bct_to_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.vocos_dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.encoder.final_norm);
    return engine::modules::LinearModule({
        config.vocos_dim,
        config.hidden_size,
        true,
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.encoder.output);
}

engine::core::TensorValue build_l2_normalized_rows(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    auto contiguous = ensure_contiguous(ctx, input);
    auto squared = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous.tensor, contiguous.tensor),
        contiguous.shape,
        GGML_TYPE_F32);
    auto summed = engine::modules::ReduceSumModule({1}).build(ctx, squared);
    auto norm = engine::core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, summed.tensor, 1.0F, 1.0e-20F)),
        summed.shape,
        GGML_TYPE_F32);
    auto repeated_norm = repeat_like(ctx, norm, contiguous);
    return engine::core::wrap_tensor(
        ggml_div(ctx.ggml, contiguous.tensor, repeated_norm.tensor),
        contiguous.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue build_coco_quantizer_ids(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & encoder_output_btd,
    const Vevo2CocoTokenizerWeights & weights,
    const Vevo2CocoTokenizerConfig & config) {
    auto hidden_bdt = transpose_btc_to_bct(ctx, encoder_output_btd);
    auto projected_bdt = engine::modules::Conv1dModule({
        config.hidden_size,
        config.codebook_dim,
        1,
        1,
        0,
        1,
        weights.quantizer.in_project.bias.has_value(),
    }).build(ctx, hidden_bdt, weights.quantizer.in_project);
    auto projected_btd = transpose_bct_to_btc(ctx, projected_bdt);
    auto flat = engine::core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, projected_btd),
        engine::core::TensorShape::from_dims({projected_btd.shape.dims[1], config.codebook_dim}));
    auto normalized = build_l2_normalized_rows(ctx, flat);
    auto logits = engine::modules::LinearModule({
        config.codebook_dim,
        config.codebook_size,
        false,
        GGML_PREC_F32,
    }).build(ctx, normalized, {weights.quantizer.codebook, std::nullopt});
    auto ids = engine::core::wrap_tensor(
        ggml_argmax(ctx.ggml, ensure_contiguous(ctx, logits).tensor),
        engine::core::TensorShape::from_dims({projected_btd.shape.dims[1]}),
        GGML_TYPE_I32);
    return engine::core::reshape_tensor(
        ctx,
        ids,
        engine::core::TensorShape::from_dims({1, projected_btd.shape.dims[1]}));
}

}  // namespace

struct Vevo2CocoTokenizerGraph {
    Vevo2CocoTokenizerGraph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const Vevo2CocoTokenizerConfig & config,
        const Vevo2CocoTokenizerWeights & weights,
        int64_t feature_frames)
        : backend(backend),
          uses_whisper(config.coco_type == "content_style"),
          feature_frames(feature_frames),
          output_frames(coco_output_frames(feature_frames, config.downsample_rate)) {
        if (backend == nullptr) {
            throw std::runtime_error("Vevo2 Coco tokenizer graph backend is not initialized");
        }
        if (feature_frames <= 0) {
            throw std::runtime_error("Vevo2 Coco tokenizer graph requires positive feature frames");
        }
        if (uses_whisper && !weights.whisper_input.has_value()) {
            throw std::runtime_error("Vevo2 content-style tokenizer graph requires Whisper input weights");
        }
        if (!weights.chromagram_input.has_value()) {
            throw std::runtime_error("Vevo2 Coco tokenizer graph requires chromagram input weights");
        }

        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 Coco tokenizer graph context");
        }

        engine::core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.coco_tokenizer", backend_type};
        chromagram_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, feature_frames, config.chromagram_dim})).tensor;
        auto chroma = engine::core::wrap_tensor(
            chromagram_input,
            engine::core::TensorShape::from_dims({1, feature_frames, config.chromagram_dim}),
            GGML_TYPE_F32);
        auto hidden = engine::modules::LinearModule({
            config.chromagram_dim,
            config.hidden_size,
            true,
            GGML_PREC_F32,
        }).build(build_ctx, chroma, *weights.chromagram_input);

        if (uses_whisper) {
            whisper_input = engine::core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, feature_frames, config.whisper_dim})).tensor;
            auto whisper = engine::core::wrap_tensor(
                whisper_input,
                engine::core::TensorShape::from_dims({1, feature_frames, config.whisper_dim}),
                GGML_TYPE_F32);
            auto whisper_hidden = engine::modules::LinearModule({
                config.whisper_dim,
                config.hidden_size,
                true,
                GGML_PREC_F32,
            }).build(build_ctx, whisper, *weights.whisper_input);
            hidden = engine::modules::AddModule{}.build(build_ctx, hidden, whisper_hidden);
        }

        hidden = transpose_btc_to_bct(build_ctx, hidden);
        for (const auto & downsample : weights.downsample_layers) {
            hidden = engine::modules::Conv1dModule({
                config.hidden_size,
                config.hidden_size,
                3,
                2,
                1,
                1,
                downsample.bias.has_value(),
            }).build(build_ctx, hidden, downsample);
            hidden = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(build_ctx, hidden);
        }
        hidden = build_coco_vocos_encoder(build_ctx, hidden, weights, config);
        const auto ids = build_coco_quantizer_ids(build_ctx, hidden, weights, config);
        ids_output = ids.tensor;
        ggml_set_output(ids_output);

        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, ids_output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate Vevo2 Coco tokenizer graph");
        }
    }

    ~Vevo2CocoTokenizerGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(int64_t frames, bool whisper) const noexcept {
        return feature_frames == frames && uses_whisper == whisper;
    }

    Vevo2TokenSequence run(
        const std::vector<float> & chromagram_features,
        const std::optional<std::vector<float>> & whisper_features) {
        if (static_cast<int64_t>(chromagram_features.size()) !=
            feature_frames * chromagram_input->ne[0]) {
            throw std::runtime_error("Vevo2 Coco tokenizer chromagram feature size mismatch");
        }
        ggml_backend_tensor_set(
            chromagram_input,
            chromagram_features.data(),
            0,
            chromagram_features.size() * sizeof(float));
        if (uses_whisper) {
            if (!whisper_features.has_value() ||
                static_cast<int64_t>(whisper_features->size()) != feature_frames * whisper_input->ne[0]) {
                throw std::runtime_error("Vevo2 Coco tokenizer Whisper feature size mismatch");
            }
            ggml_backend_tensor_set(
                whisper_input,
                whisper_features->data(),
                0,
                whisper_features->size() * sizeof(float));
        }
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Vevo2 Coco tokenizer graph compute failed");
        }
        Vevo2TokenSequence out;
        out.ids.resize(static_cast<size_t>(output_frames));
        ggml_backend_tensor_get(ids_output, out.ids.data(), 0, out.ids.size() * sizeof(int32_t));
        return out;
    }

    ggml_backend_t backend = nullptr;
    bool uses_whisper = false;
    int64_t feature_frames = 0;
    int64_t output_frames = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * chromagram_input = nullptr;
    ggml_tensor * whisper_input = nullptr;
    ggml_tensor * ids_output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

Vevo2ProsodyTokenizerRuntime::Vevo2ProsodyTokenizerRuntime(
    const Vevo2Assets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_weight_storage_type,
    engine::assets::TensorStorageType conv_weight_storage_type)
    : config_(assets.config.prosody_tokenizer),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weight_source_(assets.prosody_tokenizer_weights),
      weights_(load_coco_tokenizer_weights(
          config_,
          "prosody_tokenizer",
          execution_context.backend(),
          execution_context.backend_type(),
          weight_context_bytes,
          matmul_weight_storage_type,
          conv_weight_storage_type,
          *weight_source_)),
      name_("prosody_tokenizer") {
    weight_source_->release_storage();
}

Vevo2ProsodyTokenizerRuntime::~Vevo2ProsodyTokenizerRuntime() = default;

Vevo2CocoTokenizerGraph & Vevo2ProsodyTokenizerRuntime::ensure_graph(
    int64_t feature_frames,
    bool uses_whisper) const {
    if (uses_whisper) {
        throw std::runtime_error("Vevo2 prosody tokenizer graph does not accept Whisper features");
    }
    if (graph_ == nullptr || !graph_->matches(feature_frames, false)) {
        graph_ = std::make_unique<Vevo2CocoTokenizerGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            *weights_,
            feature_frames);
    }
    return *graph_;
}

Vevo2TokenSequence Vevo2ProsodyTokenizerRuntime::encode_chromagram_features(
    const std::vector<float> & chromagram_features,
    int64_t feature_frames) const {
    if (static_cast<int64_t>(chromagram_features.size()) != feature_frames * config_.chromagram_dim) {
        throw std::runtime_error("Vevo2 prosody chromagram feature size mismatch");
    }
    return ensure_graph(feature_frames, false).run(chromagram_features, std::nullopt);
}

Vevo2TokenSequence Vevo2ProsodyTokenizerRuntime::encode(
    const runtime::AudioBuffer & prosody_audio,
    const std::optional<runtime::AudioBuffer> & style_ref_audio,
    const Vevo2GenerationOptions & generation) const {
    if (generation.predict_target_prosody) {
        throw std::runtime_error("Vevo2 predict_target_prosody is not implemented in the reference path");
    }
    const int64_t prosody_frames = vevo2_frame_count_24k(prosody_audio);
    double frame_len_ratio = 1.0;
    if (generation.target_duration_seconds.has_value()) {
        const auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            prosody_audio.samples,
            prosody_audio.sample_rate,
            prosody_audio.channels,
            24000);
        const double duration = static_cast<double>(mono.size()) / 24000.0;
        if (duration <= 0.0) {
            throw std::runtime_error("Vevo2 prosody duration ratio requires non-empty audio");
        }
        frame_len_ratio = static_cast<double>(*generation.target_duration_seconds) / duration;
    }
    auto prosody_tokens = encode_chromagram_features(
        compute_vevo2_chromagram(
            prosody_audio,
            prosody_frames,
            frame_len_ratio,
            generation.use_pitch_shift,
            generation.use_pitch_shift ? generation.prosody_shift_steps : 0),
        static_cast<int64_t>(static_cast<double>(prosody_frames) * frame_len_ratio));
    if (!style_ref_audio.has_value()) {
        return prosody_tokens;
    }

    const int64_t style_frames = vevo2_frame_count_24k(*style_ref_audio);
    auto style_tokens = encode_chromagram_features(
        compute_vevo2_chromagram(
            *style_ref_audio,
            style_frames,
            1.0,
            generation.use_pitch_shift,
            generation.use_pitch_shift ? generation.style_shift_steps : 0),
        style_frames);
    style_tokens.ids.insert(style_tokens.ids.end(), prosody_tokens.ids.begin(), prosody_tokens.ids.end());
    return style_tokens;
}

Vevo2ContentStyleTokenizerRuntime::Vevo2ContentStyleTokenizerRuntime(
    const Vevo2Assets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_weight_storage_type,
    engine::assets::TensorStorageType conv_weight_storage_type)
    : config_(assets.config.content_style_tokenizer),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weight_source_(assets.content_style_tokenizer_weights),
      weights_(load_coco_tokenizer_weights(
          config_,
          "content_style_tokenizer",
          execution_context.backend(),
          execution_context.backend_type(),
          weight_context_bytes,
          matmul_weight_storage_type,
          conv_weight_storage_type,
          *weight_source_)),
      name_("content_style_tokenizer") {
    if (config_.use_normed_whisper) {
        auto stats = load_whisper_stats(*assets.fm_whisper_stats, config_.whisper_dim);
        whisper_mean_ = std::move(stats.first);
        whisper_std_ = std::move(stats.second);
    }
    weight_source_->release_storage();
}

Vevo2ContentStyleTokenizerRuntime::~Vevo2ContentStyleTokenizerRuntime() = default;

Vevo2CocoTokenizerGraph & Vevo2ContentStyleTokenizerRuntime::ensure_graph(
    int64_t feature_frames,
    bool uses_whisper) const {
    if (!uses_whisper) {
        throw std::runtime_error("Vevo2 content-style tokenizer graph requires Whisper features");
    }
    if (graph_ == nullptr || !graph_->matches(feature_frames, true)) {
        graph_ = std::make_unique<Vevo2CocoTokenizerGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            *weights_,
            feature_frames);
    }
    return *graph_;
}

Vevo2TokenSequence Vevo2ContentStyleTokenizerRuntime::encode_feature_frames(
    const std::vector<float> & whisper_features,
    const std::vector<float> & chromagram_features,
    int64_t feature_frames) const {
    if (static_cast<int64_t>(whisper_features.size()) != feature_frames * config_.whisper_dim) {
        throw std::runtime_error("Vevo2 content-style Whisper feature size mismatch");
    }
    if (static_cast<int64_t>(chromagram_features.size()) != feature_frames * config_.chromagram_dim) {
        throw std::runtime_error("Vevo2 content-style chromagram feature size mismatch");
    }
    std::vector<float> normalized_whisper = whisper_features;
    if (config_.use_normed_whisper) {
        if (static_cast<int64_t>(whisper_mean_.size()) != config_.whisper_dim ||
            static_cast<int64_t>(whisper_std_.size()) != config_.whisper_dim) {
            throw std::runtime_error("Vevo2 content-style Whisper stats are not loaded");
        }
        for (int64_t frame = 0; frame < feature_frames; ++frame) {
            const int64_t base = frame * config_.whisper_dim;
            for (int64_t dim = 0; dim < config_.whisper_dim; ++dim) {
                const float stddev = whisper_std_[static_cast<size_t>(dim)];
                if (stddev == 0.0F) {
                    throw std::runtime_error("Vevo2 content-style Whisper std contains zero");
                }
                const size_t index = static_cast<size_t>(base + dim);
                normalized_whisper[index] =
                    (normalized_whisper[index] - whisper_mean_[static_cast<size_t>(dim)]) / stddev;
            }
        }
    }
    return ensure_graph(feature_frames, true).run(
        chromagram_features,
        std::optional<std::vector<float>>(std::move(normalized_whisper)));
}

Vevo2TokenSequence Vevo2ContentStyleTokenizerRuntime::encode_style_reference(
    const std::optional<runtime::AudioBuffer> & style_ref_audio,
    const std::optional<std::vector<float>> & whisper_features,
    int64_t feature_frames,
    const Vevo2GenerationOptions & generation) const {
    if (!style_ref_audio.has_value()) {
        return {};
    }
    if (!whisper_features.has_value()) {
        throw std::runtime_error("Vevo2 style reference tokenization requires Whisper features");
    }
    return encode_feature_frames(
        *whisper_features,
        compute_vevo2_chromagram(
            *style_ref_audio,
            feature_frames,
            1.0,
            generation.use_pitch_shift,
            generation.use_pitch_shift ? generation.style_shift_steps : 0),
        feature_frames);
}

Vevo2TokenSequence Vevo2ContentStyleTokenizerRuntime::encode_shifted_reference(
    const runtime::AudioBuffer & audio,
    const std::vector<float> & whisper_features,
    int64_t feature_frames,
    int pitch_shift_steps) const {
    return encode_feature_frames(
        whisper_features,
        compute_vevo2_chromagram(audio, feature_frames, 1.0, true, pitch_shift_steps),
        feature_frames);
}

Vevo2TokenSequence Vevo2ContentStyleTokenizerRuntime::encode_timbre_reference(
    const runtime::AudioBuffer & timbre_ref_audio,
    const std::vector<float> & whisper_features,
    int64_t feature_frames) const {
    return encode_feature_frames(
        whisper_features,
        compute_vevo2_chromagram(timbre_ref_audio, feature_frames),
        feature_frames);
}

}  // namespace engine::models::vevo2
