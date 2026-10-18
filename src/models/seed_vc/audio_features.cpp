#include "engine/models/seed_vc/audio_features.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::seed_vc {
namespace {

std::vector<float> validated_seed_vc_mono_samples(const std::vector<float> & samples, int channels) {
    if (channels <= 0) {
        throw std::runtime_error("Seed-VC audio channel count must be positive");
    }
    if (samples.empty()) {
        throw std::runtime_error("Seed-VC audio must not be empty");
    }
    if (samples.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("Seed-VC audio sample count must be divisible by channels");
    }
    if (channels == 1) {
        return samples;
    }
    return engine::audio::mixdown_interleaved_to_mono_average(samples, channels);
}

}  // namespace

namespace {

std::vector<float> seed_vc_resample_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("Seed-VC resampling requires positive sample rates");
    }
    if (input_sample_rate == output_sample_rate || input.empty()) {
        return input;
    }
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::ExplicitFloat32Runtime;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.require_full_input = true;
    options.warning_context = "Seed-VC";
    options.fallback_description = "torchaudio-compatible resampling";
    if (auto output = engine::audio::try_resample_mono_soxr(input, input_sample_rate, output_sample_rate, options)) {
        return *output;
    }
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, input_sample_rate, output_sample_rate);
}

}  // namespace

SeedVcPreparedAudio seed_vc_prepare_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    int64_t max_22k_samples) {
    auto wave_22k = seed_vc_resample_mono(validated_seed_vc_mono_samples(samples, channels), sample_rate, 22050);
    if (max_22k_samples > 0 && static_cast<int64_t>(wave_22k.size()) > max_22k_samples) {
        engine::audio::truncate_samples_to_count(wave_22k, static_cast<size_t>(max_22k_samples));
    }
    SeedVcPreparedAudio out;
    out.waveform_16k = seed_vc_resample_mono(wave_22k, 22050, 16000);
    out.waveform_22k = std::move(wave_22k);
    return out;
}

SeedVcPreparedAudioForSampleRate seed_vc_prepare_audio_for_sample_rate(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    int output_sample_rate,
    int64_t max_output_samples) {
    auto waveform = seed_vc_resample_mono(validated_seed_vc_mono_samples(samples, channels), sample_rate, output_sample_rate);
    if (max_output_samples > 0 && static_cast<int64_t>(waveform.size()) > max_output_samples) {
        engine::audio::truncate_samples_to_count(waveform, static_cast<size_t>(max_output_samples));
    }
    SeedVcPreparedAudioForSampleRate out;
    out.waveform_16k = engine::audio::resample_mono_torchaudio_sinc_hann(waveform, output_sample_rate, 16000);
    out.waveform = std::move(waveform);
    return out;
}

SeedVcMelSpectrogramOutput compute_seed_vc_mel_spectrogram(
    const std::vector<float> & waveform,
    const SeedVcMelConfig & config,
    size_t threads) {
    engine::audio::MelSpectrogramFrontendConfig mel;
    mel.sample_rate = config.sample_rate;
    mel.n_fft = config.n_fft;
    mel.hop_length = config.hop_size;
    mel.win_length = config.win_size;
    mel.n_mels = config.num_mels;
    mel.mel_fmin = config.fmin;
    mel.mel_fmax = config.fmax;
    mel.filterbank_projection = engine::audio::MelFilterbankProjection::SparseF32;
    mel.magnitude_epsilon = 1.0e-9f;
    const auto frontend = engine::audio::get_cached_mel_spectrogram_frontend(mel);
    auto features = frontend->extract_mono(waveform, threads);
    return {std::move(features.values), features.mel_bins, features.frames};
}

SeedVcCampplusFbankOutput compute_seed_vc_campplus_fbank_16k(const std::vector<float> & waveform_16k) {
    if (waveform_16k.size() < 400) return {};
    auto features = engine::audio::ReferenceAudioFrontend::campplus_fbank_16k(waveform_16k);
    return {std::move(features.values), features.frames, features.feature_dim};
}

}  // namespace engine::models::seed_vc
