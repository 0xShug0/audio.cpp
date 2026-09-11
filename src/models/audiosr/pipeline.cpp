#include "engine/models/audiosr/pipeline.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/audiosr/autoencoder.h"
#include "engine/models/audiosr/ddim.h"
#include "engine/models/audiosr/frontend.h"
#include "engine/models/audiosr/hifigan.h"
#include "engine/models/audiosr/unet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::audiosr {
namespace {

constexpr int kAudioSRLongAudioSampleRate = 48000;

std::shared_ptr<const AudioSRAssets> require_assets(std::shared_ptr<const AudioSRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("AudioSR pipeline requires assets");
    }
    return assets;
}

uint32_t parse_seed(const std::unordered_map<std::string, std::string> & options) {
    const auto seed = engine::runtime::parse_i64_option(options, {"seed"}).value_or(42);
    if (seed < 0) {
        return 42;
    }
    if (seed > 2147483647) {
        throw std::runtime_error("AudioSR seed is outside int range");
    }
    return static_cast<uint32_t>(seed);
}

int64_t locate_cutoff_mel_bin(const std::vector<float> & mel_bft, int64_t frames, const AudioSRConfig & config) {
    if (frames <= 0 || static_cast<int64_t>(mel_bft.size()) != config.n_mels * frames) {
        throw std::runtime_error("AudioSR mel cutoff received invalid mel shape");
    }
    std::vector<double> energy(static_cast<size_t>(config.n_mels), 0.0);
    for (int64_t bin = 0; bin < config.n_mels; ++bin) {
        double sum = 0.0;
        for (int64_t frame = 0; frame < frames; ++frame) {
            sum += std::exp(static_cast<double>(mel_bft[static_cast<size_t>(frame * config.n_mels + bin)]));
        }
        energy[static_cast<size_t>(bin)] = sum + (bin == 0 ? 0.0 : energy[static_cast<size_t>(bin - 1)]);
    }
    const double threshold = energy.back() * 0.985;
    for (int64_t i = 1; i <= config.n_mels; ++i) {
        const int64_t bin = config.n_mels - i;
        if (energy[static_cast<size_t>(bin)] < threshold) {
            return bin;
        }
    }
    return 0;
}

std::vector<float> replace_low_mel_bins(
    std::vector<float> generated,
    const std::vector<float> & lowpass,
    int64_t frames,
    const AudioSRConfig & config) {
    if (static_cast<int64_t>(generated.size()) != config.n_mels * frames ||
        static_cast<int64_t>(lowpass.size()) != config.n_mels * frames) {
        throw std::runtime_error("AudioSR mel replacement shape mismatch");
    }
    const int64_t cutoff = locate_cutoff_mel_bin(lowpass, frames, config);
    for (int64_t bin = 0; bin < cutoff; ++bin) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            generated[static_cast<size_t>(bin * frames + frame)] =
                lowpass[static_cast<size_t>(frame * config.n_mels + bin)];
        }
    }
    return generated;
}

void normalize_generated_waveform(std::vector<float> & samples) {
    if (samples.empty()) {
        return;
    }
    float peak = 0.0F;
    for (float sample : samples) {
        peak = std::max(peak, std::abs(sample));
    }
    if (peak > 0.0F) {
        const float scale = 0.5F / peak;
        for (float & sample : samples) {
            sample *= scale;
        }
    }
    double mean = 0.0;
    for (float sample : samples) {
        mean += static_cast<double>(sample);
    }
    mean /= static_cast<double>(samples.size());
    for (float & sample : samples) {
        sample -= static_cast<float>(mean);
    }
}

int64_t locate_cutoff_stft_bin(const std::vector<float> & samples, const engine::audio::STFTConfig & stft_config) {
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto complex = engine::audio::STFT().compute_complex(
        samples,
        window,
        1,
        static_cast<int64_t>(samples.size()),
        stft_config,
        0);
    if (complex.shape.size() != 4 || complex.shape[0] != 1 || complex.shape[3] != 2) {
        throw std::runtime_error("AudioSR postprocess STFT shape mismatch");
    }
    const int64_t freq_bins = complex.shape[1];
    const int64_t frames = complex.shape[2];
    std::vector<double> energy(static_cast<size_t>(freq_bins), 0.0);
    for (int64_t f = 0; f < freq_bins; ++f) {
        double sum = 0.0;
        for (int64_t t = 0; t < frames; ++t) {
            const size_t base = static_cast<size_t>((f * frames + t) * 2);
            const double real = complex.values[base];
            const double imag = complex.values[base + 1];
            sum += std::sqrt(real * real + imag * imag);
        }
        energy[static_cast<size_t>(f)] = sum + (f == 0 ? 0.0 : energy[static_cast<size_t>(f - 1)]);
    }
    const double threshold = energy.back() * 0.985;
    for (int64_t i = 1; i <= freq_bins; ++i) {
        const int64_t bin = freq_bins - i;
        if (energy[static_cast<size_t>(bin)] < threshold) {
            return bin;
        }
    }
    return 0;
}

void postprocess_waveform_with_lowpass(std::vector<float> & generated, const std::vector<float> & lowpass) {
    if (generated.empty() || lowpass.empty()) {
        throw std::runtime_error("AudioSR postprocess requires non-empty waveforms");
    }
    const engine::audio::STFTConfig stft_config{
        2048,
        512,
        2048,
        true,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto gt = engine::audio::STFT().compute_complex(
        lowpass,
        window,
        1,
        static_cast<int64_t>(lowpass.size()),
        stft_config,
        0);
    auto out = engine::audio::STFT().compute_complex(
        generated,
        window,
        1,
        static_cast<int64_t>(generated.size()),
        stft_config,
        0);
    if (gt.shape.size() != 4 || out.shape.size() != 4 || gt.shape[0] != 1 || out.shape[0] != 1 ||
        gt.shape[1] != out.shape[1] || gt.shape[2] != out.shape[2] || gt.shape[3] != 2 || out.shape[3] != 2) {
        throw std::runtime_error("AudioSR postprocess STFT shape mismatch");
    }
    const int64_t freq_bins = out.shape[1];
    const int64_t frames = out.shape[2];
    const int64_t cutoff = locate_cutoff_stft_bin(lowpass, stft_config);
    if (cutoff < 0 || cutoff >= freq_bins) {
        throw std::runtime_error("AudioSR postprocess cutoff is outside STFT range");
    }
    double gt_energy = 0.0;
    double out_energy = 0.0;
    for (int64_t t = 0; t < frames; ++t) {
        const size_t base = static_cast<size_t>((cutoff * frames + t) * 2);
        const double gt_real = gt.values[base];
        const double gt_imag = gt.values[base + 1];
        const double out_real = out.values[base];
        const double out_imag = out.values[base + 1];
        gt_energy += std::sqrt(gt_real * gt_real + gt_imag * gt_imag);
        out_energy += std::sqrt(out_real * out_real + out_imag * out_imag);
    }
    if (!(out_energy > 0.0)) {
        throw std::runtime_error("AudioSR postprocess generated STFT cutoff energy is zero");
    }
    const float energy_ratio = static_cast<float>(std::min(1.2, std::max(0.8, gt_energy / out_energy)));
    for (int64_t f = 0; f < cutoff; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            const size_t base = static_cast<size_t>((f * frames + t) * 2);
            out.values[base] = gt.values[base] / energy_ratio;
            out.values[base + 1] = gt.values[base + 1] / energy_ratio;
        }
    }
    generated = engine::audio::ISTFT()
        .compute(
            out.values,
            window,
            1,
            freq_bins,
            frames,
            static_cast<int64_t>(generated.size()),
            stft_config,
            0)
        .values;
}

std::vector<float> audio_to_mono_48k(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("AudioSR input audio sample rate and channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("AudioSR input audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("AudioSR input audio samples must be divisible by channel count");
    }
    std::vector<float> mono;
    if (audio.channels == 1) {
        mono = audio.samples;
    } else {
        mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    }
    if (audio.sample_rate != kAudioSRLongAudioSampleRate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            kAudioSRLongAudioSampleRate,
            engine::audio::torchaudio_sinc_hann_float32_options());
    }
    return mono;
}

int64_t samples_from_seconds(float seconds, const char * name) {
    if (!std::isfinite(seconds) || seconds <= 0.0F) {
        throw std::runtime_error(std::string("AudioSR ") + name + " must be positive");
    }
    const double samples = static_cast<double>(seconds) * static_cast<double>(kAudioSRLongAudioSampleRate);
    if (samples < 1.0 || samples > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(std::string("AudioSR ") + name + " is outside supported range");
    }
    return static_cast<int64_t>(samples);
}

float hann_window_value(int64_t index, int64_t size) {
    if (size <= 1) {
        return 1.0F;
    }
    constexpr double kPi = 3.141592653589793238462643383279502884;
    return static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(index) / static_cast<double>(size - 1))));
}

float peak_abs(const std::vector<float> & samples, size_t start, size_t count) {
    float peak = 0.0F;
    for (size_t i = 0; i < count; ++i) {
        peak = std::max(peak, std::abs(samples[start + i]));
    }
    return peak + 1.0e-8F;
}

engine::runtime::AudioBuffer chunk_audio_buffer(const std::vector<float> & mono_48k, size_t start, size_t valid, size_t padded) {
    engine::runtime::AudioBuffer out;
    out.sample_rate = kAudioSRLongAudioSampleRate;
    out.channels = 1;
    out.samples.assign(padded, 0.0F);
    std::copy_n(mono_48k.begin() + static_cast<std::ptrdiff_t>(start), valid, out.samples.begin());
    return out;
}

}  // namespace

struct AudioSRPipelineRuntime::Impl {
    Impl(
        std::shared_ptr<const AudioSRAssets> assets_in,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type)
        : assets(require_assets(std::move(assets_in))),
          frontend(*assets),
          autoencoder(assets, execution, weight_type),
          unet(assets, execution, weight_type),
          sampler(
              assets->config,
              engine::sampling::resolve_torch_cuda_sampling_policy(
                  execution.backend_type(),
                  execution.config().device,
                  "audiosr.ddim.rng",
                  "AudioSR",
                  engine::sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)),
          vocoder(assets, execution, weight_type) {}

    std::shared_ptr<const AudioSRAssets> assets;
    AudioSRFrontend frontend;
    AudioSRAutoencoderRuntime autoencoder;
    AudioSRUnetRuntime unet;
    AudioSRDdimSampler sampler;
    AudioSRHiFiGanRuntime vocoder;

    engine::runtime::AudioBuffer run_one_shot(
        const engine::runtime::AudioBuffer & input,
        const AudioSROptions & options,
        uint64_t lowpass_rng_offset,
        uint64_t & torch_rng_offset_blocks) {
        const auto prepared = frontend.compute(input, options.seed, lowpass_rng_offset, 0);
        const auto cond = autoencoder.encode_condition(prepared.lowpass_mel, prepared.mel_frames, options.seed);
        autoencoder.release_encoder_graph();
        AudioSRLatent uncond = cond;
        uncond.values.assign(uncond.values.size(), assets->config.unconditional_lowpass_value);
        const auto latent = sampler.sample(
            unet,
            cond,
            uncond,
            options.num_inference_steps,
            options.guidance_scale,
            options.ddim_eta,
            options.seed,
            torch_rng_offset_blocks);
        unet.release_runtime_graphs();
        auto mel = autoencoder.decode_first_stage(latent);
        autoencoder.release_decoder_graph();
        mel = replace_low_mel_bins(std::move(mel), prepared.lowpass_mel, prepared.mel_frames, assets->config);
        auto audio = vocoder.synthesize(mel, prepared.mel_frames);
        vocoder.release_runtime_graphs();
        postprocess_waveform_with_lowpass(audio.samples, prepared.lowpass_waveform);
        normalize_generated_waveform(audio.samples);
        if (static_cast<int64_t>(audio.samples.size()) > prepared.original_samples) {
            audio.samples.resize(static_cast<size_t>(prepared.original_samples));
        }
        return audio;
    }

    engine::runtime::AudioBuffer run_one_shot(
        const engine::runtime::AudioBuffer & input,
        const AudioSROptions & options) {
        uint64_t torch_rng_offset_blocks = 0;
        return run_one_shot(input, options, 0, torch_rng_offset_blocks);
    }

    engine::runtime::AudioBuffer run_long_audio(
        const std::vector<float> & mono_48k,
        const AudioSROptions & options,
        int64_t chunk_samples,
        int64_t overlap_samples) {
        if (chunk_samples <= overlap_samples) {
            throw std::runtime_error("AudioSR audio_chunk_duration_sec must be greater than audio_chunk_overlap_sec");
        }
        const int64_t step_samples = chunk_samples - overlap_samples;
        const size_t total_samples = mono_48k.size();
        std::vector<float> final_waveform(total_samples, 0.0F);
        std::vector<float> contribution(total_samples, 0.0F);
        uint64_t torch_rng_offset_blocks = 0;
        uint64_t lowpass_rng_offset = 0;
        const int64_t fade_size = overlap_samples * 2;

        for (size_t start = 0; start < total_samples; start += static_cast<size_t>(step_samples)) {
            const size_t valid = std::min(static_cast<size_t>(chunk_samples), total_samples - start);
            const float original_peak = peak_abs(mono_48k, start, valid);
            auto chunk = chunk_audio_buffer(mono_48k, start, valid, static_cast<size_t>(chunk_samples));
            auto processed = run_one_shot(chunk, options, lowpass_rng_offset, torch_rng_offset_blocks);
            ++lowpass_rng_offset;
            if (processed.samples.size() > valid) {
                processed.samples.resize(valid);
            }
            const float processed_peak = peak_abs(processed.samples, 0, processed.samples.size());
            const float scale = original_peak / processed_peak;
            for (float & sample : processed.samples) {
                sample *= scale;
            }

            if (overlap_samples > 0 && start > 0) {
                const size_t fade_count = std::min(static_cast<size_t>(overlap_samples), processed.samples.size());
                for (size_t i = 0; i < fade_count; ++i) {
                    processed.samples[i] *= hann_window_value(static_cast<int64_t>(i), fade_size);
                }
            }
            if (overlap_samples > 0 && start + static_cast<size_t>(chunk_samples) < total_samples) {
                const size_t fade_count = std::min(static_cast<size_t>(overlap_samples), processed.samples.size());
                const size_t fade_start = processed.samples.size() - fade_count;
                for (size_t i = 0; i < fade_count; ++i) {
                    processed.samples[fade_start + i] *=
                        hann_window_value(overlap_samples + static_cast<int64_t>(i), fade_size);
                }
            }

            const bool has_left_overlap = overlap_samples > 0 && start > 0;
            const bool has_right_overlap =
                overlap_samples > 0 && start + static_cast<size_t>(chunk_samples) < total_samples;
            const size_t left_fade_count = has_left_overlap
                ? std::min(static_cast<size_t>(overlap_samples), processed.samples.size())
                : 0;
            const size_t right_fade_count = has_right_overlap
                ? std::min(static_cast<size_t>(overlap_samples), processed.samples.size())
                : 0;
            const size_t right_fade_start = processed.samples.size() - right_fade_count;
            for (size_t i = 0; i < processed.samples.size(); ++i) {
                float weight = 1.0F;
                if (i < left_fade_count) {
                    weight = hann_window_value(static_cast<int64_t>(i), fade_size);
                }
                if (i >= right_fade_start) {
                    const size_t fade_i = i - right_fade_start;
                    weight = hann_window_value(overlap_samples + static_cast<int64_t>(fade_i), fade_size);
                }
                final_waveform[start + i] += processed.samples[i];
                contribution[start + i] += weight;
            }
        }

        for (size_t i = 0; i < final_waveform.size(); ++i) {
            if (contribution[i] != 0.0F) {
                final_waveform[i] /= contribution[i];
            }
            final_waveform[i] = std::clamp(final_waveform[i], -1.0F, 1.0F);
        }
        return engine::runtime::AudioBuffer{kAudioSRLongAudioSampleRate, 1, std::move(final_waveform)};
    }
};

AudioSROptions parse_audiosr_options(const std::unordered_map<std::string, std::string> & options) {
    AudioSROptions out;
    out.num_inference_steps = static_cast<int>(
        engine::runtime::parse_i64_option(options, {"num_inference_steps"}).value_or(out.num_inference_steps));
    if (out.num_inference_steps <= 0) {
        throw std::runtime_error("AudioSR num_inference_steps must be positive");
    }
    out.guidance_scale = engine::runtime::parse_finite_float_option(
        options,
        {"guidance_scale"}).value_or(out.guidance_scale);
    if (!std::isfinite(out.guidance_scale)) {
        throw std::runtime_error("AudioSR guidance_scale must be finite");
    }
    out.ddim_eta = engine::runtime::parse_finite_float_option(
        options,
        {"ddim_eta"}).value_or(out.ddim_eta);
    if (!std::isfinite(out.ddim_eta) || out.ddim_eta < 0.0F) {
        throw std::runtime_error("AudioSR ddim_eta must be finite and non-negative");
    }
    out.audio_chunk_duration_sec = engine::runtime::parse_positive_finite_float_option(
        options,
        {"audio_chunk_duration_sec", "chunk_duration", "chunk_duration_sec"}).value_or(out.audio_chunk_duration_sec);
    out.audio_chunk_overlap_sec = engine::runtime::parse_finite_float_option(
        options,
        {"audio_chunk_overlap_sec", "overlap_duration", "overlap_duration_sec"}).value_or(out.audio_chunk_overlap_sec);
    if (!std::isfinite(out.audio_chunk_overlap_sec) || out.audio_chunk_overlap_sec < 0.0F) {
        throw std::runtime_error("AudioSR audio_chunk_overlap_sec must be finite and non-negative");
    }
    if (out.audio_chunk_duration_sec <= out.audio_chunk_overlap_sec) {
        throw std::runtime_error("AudioSR audio_chunk_duration_sec must be greater than audio_chunk_overlap_sec");
    }
    out.seed = parse_seed(options);
    return out;
}

AudioSRPipelineRuntime::AudioSRPipelineRuntime(
    std::shared_ptr<const AudioSRAssets> assets,
    engine::core::ExecutionContext & execution,
    engine::assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, weight_type)) {}

AudioSRPipelineRuntime::~AudioSRPipelineRuntime() = default;

engine::runtime::AudioBuffer AudioSRPipelineRuntime::run(
    const engine::runtime::AudioBuffer & input,
    const AudioSROptions & options) {
    const int64_t chunk_samples = samples_from_seconds(options.audio_chunk_duration_sec, "audio_chunk_duration_sec");
    const int64_t overlap_samples = options.audio_chunk_overlap_sec == 0.0F
        ? 0
        : samples_from_seconds(options.audio_chunk_overlap_sec, "audio_chunk_overlap_sec");
    if (overlap_samples >= chunk_samples) {
        throw std::runtime_error("AudioSR audio_chunk_duration_sec must be greater than audio_chunk_overlap_sec");
    }
    const auto mono_48k = audio_to_mono_48k(input);
    if (static_cast<int64_t>(mono_48k.size()) > chunk_samples) {
        return impl_->run_long_audio(mono_48k, options, chunk_samples, overlap_samples);
    }
    return impl_->run_one_shot(input, options);
}

}  // namespace engine::models::audiosr
