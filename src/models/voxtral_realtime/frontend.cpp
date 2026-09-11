#include "engine/models/voxtral_realtime/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::voxtral_realtime {
namespace {

using Clock = std::chrono::steady_clock;

int64_t raw_audio_length_per_token(const VoxtralRealtimeConfig & config) {
    return config.audio_length_per_tok * config.frontend.hop_length;
}

std::vector<float> padded_streaming_audio(const VoxtralRealtimeConfig & config, std::vector<float> audio) {
    const int64_t unit = raw_audio_length_per_token(config);
    const int64_t left_pad = 32 * unit;
    const int64_t right_base = (unit - (static_cast<int64_t>(audio.size()) % unit)) % unit;
    const int64_t right_extra = (config.default_num_delay_tokens + 1 + 10) * unit;
    std::vector<float> padded(static_cast<size_t>(left_pad + static_cast<int64_t>(audio.size()) + right_base + right_extra), 0.0F);
    std::copy(audio.begin(), audio.end(), padded.begin() + left_pad);
    return padded;
}

std::vector<float> padded_first_stream_audio(const VoxtralRealtimeConfig & config, std::vector<float> audio) {
    const int64_t unit = raw_audio_length_per_token(config);
    const int64_t left_pad = 32 * unit;
    std::vector<float> padded(static_cast<size_t>(left_pad + static_cast<int64_t>(audio.size())), 0.0F);
    std::copy(audio.begin(), audio.end(), padded.begin() + left_pad);
    return padded;
}

engine::audio::STFTConfig make_stft_config(const VoxtralRealtimeFrontendConfig & config, bool center) {
    engine::audio::STFTConfig stft_config;
    stft_config.n_fft = config.n_fft;
    stft_config.hop_length = config.hop_length;
    stft_config.win_length = config.win_length;
    stft_config.center = center;
    stft_config.pad_mode = engine::audio::STFTPadMode::Reflect;
    stft_config.family = engine::audio::STFTFamily::Kokoro;
    return stft_config;
}

void cache_last_magnitude_frame(
    const engine::audio::AudioTensor & magnitude,
    VoxtralRealtimeFrontendStreamState & state) {
    const int64_t freq_bins = magnitude.shape.at(1);
    const int64_t stft_frames = magnitude.shape.at(2);
    if (freq_bins <= 0 || stft_frames <= 0) {
        throw std::runtime_error("VoxTral streaming frontend cannot cache empty STFT output");
    }
    state.cached_magnitude_frame.resize(static_cast<size_t>(freq_bins));
    for (int64_t freq = 0; freq < freq_bins; ++freq) {
        state.cached_magnitude_frame[static_cast<size_t>(freq)] =
            magnitude.values[static_cast<size_t>(freq * stft_frames + (stft_frames - 1))];
    }
    state.cached_freq_bins = freq_bins;
    state.cached_frame_ready = true;
}

VoxtralRealtimeFeatures features_from_magnitude(
    const VoxtralRealtimeFrontendConfig & config,
    const engine::audio::AudioTensor & magnitude,
    const engine::audio::SparseMelFilterbank & filterbank,
    int64_t frames,
    const Clock::time_point & total_start,
    bool log_timing) {
    const int64_t freq_bins = magnitude.shape.at(1);
    const int64_t stft_frames = magnitude.shape.at(2);
    if (frames <= 0 || frames > stft_frames) {
        throw std::runtime_error("VoxTral frontend has invalid STFT frame count");
    }
    const auto mel_start = Clock::now();
    auto mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        freq_bins,
        stft_frames,
        frames,
        filterbank);
    if (log_timing) {
        engine::debug::timing_log_scalar(
            "voxtral_realtime.frontend.mel_ms",
            engine::debug::elapsed_ms(mel_start));
    }
    const auto log_start = Clock::now();
    const float floor_value = config.global_log_mel_max - 8.0F;
    for (float & value : mel.values) {
        value = (std::max(std::log10(std::max(value, 1.0e-10F)), floor_value) + 4.0F) / 4.0F;
    }
    if (log_timing) {
        engine::debug::timing_log_scalar(
            "voxtral_realtime.frontend.log_scale_ms",
            engine::debug::elapsed_ms(log_start));
    }
    VoxtralRealtimeFeatures out;
    out.values = std::move(mel.values);
    out.mel_bins = config.feature_size;
    out.frames = frames;
    if (log_timing) {
        engine::debug::timing_log_scalar("voxtral_realtime.frontend.frames", out.frames);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.frontend.feature_extract_ms",
            engine::debug::elapsed_ms(total_start));
    }
    return out;
}

engine::audio::AudioTensor magnitude_with_cached_first_frame(
    const engine::audio::AudioTensor & tail_magnitude,
    const VoxtralRealtimeFrontendStreamState & state) {
    const int64_t freq_bins = tail_magnitude.shape.at(1);
    const int64_t tail_stft_frames = tail_magnitude.shape.at(2);
    const int64_t tail_output_frames = tail_stft_frames - 1;
    if (!state.cached_frame_ready || state.cached_freq_bins != freq_bins ||
        static_cast<int64_t>(state.cached_magnitude_frame.size()) != freq_bins) {
        throw std::runtime_error("VoxTral streaming frontend STFT cache shape mismatch");
    }
    if (tail_output_frames <= 0) {
        throw std::runtime_error("VoxTral streaming frontend tail STFT frame count mismatch");
    }
    engine::audio::AudioTensor magnitude;
    magnitude.shape = {1, freq_bins, tail_output_frames + 1};
    magnitude.values.assign(static_cast<size_t>(freq_bins * (tail_output_frames + 1)), 0.0F);
    for (int64_t freq = 0; freq < freq_bins; ++freq) {
        const size_t out_base = static_cast<size_t>(freq * (tail_output_frames + 1));
        magnitude.values[out_base] = state.cached_magnitude_frame[static_cast<size_t>(freq)];
        for (int64_t frame = 0; frame < tail_output_frames; ++frame) {
            magnitude.values[out_base + static_cast<size_t>(frame + 1)] =
                tail_magnitude.values[static_cast<size_t>(freq * tail_stft_frames + frame)];
        }
    }
    return magnitude;
}

VoxtralRealtimeFeatures extract_features_from_mono(
    const VoxtralRealtimeFrontendConfig & config,
    const std::vector<float> & mono,
    bool center,
    const engine::audio::SparseMelFilterbank & filterbank,
    VoxtralRealtimeFrontendStreamState * stream_state,
    bool reuse_cached_prefix,
    bool log_timing) {
    const auto total_start = Clock::now();
    if (mono.empty()) {
        throw std::runtime_error("VoxTral frontend requires non-empty audio");
    }
    const auto stft_config = make_stft_config(config, center);
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    if (reuse_cached_prefix && stream_state != nullptr && stream_state->cached_frame_ready) {
        if (center || static_cast<int64_t>(mono.size()) <= config.hop_length) {
            throw std::runtime_error("VoxTral streaming frontend cached STFT path received invalid chunk");
        }
        const std::vector<float> tail(
            mono.begin() + static_cast<std::ptrdiff_t>(config.hop_length),
            mono.end());
        const auto stft_start = Clock::now();
        auto tail_magnitude = engine::audio::STFT().compute_magnitude(
            tail,
            window,
            1,
            static_cast<int64_t>(tail.size()),
            stft_config,
            1);
        if (log_timing) {
            engine::debug::timing_log_scalar(
                "voxtral_realtime.frontend.stft_ms",
                engine::debug::elapsed_ms(stft_start));
        }
        auto magnitude = magnitude_with_cached_first_frame(tail_magnitude, *stream_state);
        cache_last_magnitude_frame(tail_magnitude, *stream_state);
        return features_from_magnitude(config, magnitude, filterbank, magnitude.shape.at(2), total_start, log_timing);
    }

    const auto stft_start = Clock::now();
    const size_t stft_threads = (stream_state != nullptr && !center) ? 1 : 0;
    auto magnitude = engine::audio::STFT().compute_magnitude(
        mono,
        window,
        1,
        static_cast<int64_t>(mono.size()),
        stft_config,
        stft_threads);
    if (log_timing) {
        engine::debug::timing_log_scalar(
            "voxtral_realtime.frontend.stft_ms",
            engine::debug::elapsed_ms(stft_start));
    }
    const int64_t stft_frames = magnitude.shape.at(2);
    const int64_t frames = stft_frames - 1;
    if (frames <= 0) {
        throw std::runtime_error("VoxTral frontend requires at least two STFT frames");
    }
    if (stream_state != nullptr && !center) {
        cache_last_magnitude_frame(magnitude, *stream_state);
    }
    return features_from_magnitude(config, magnitude, filterbank, frames, total_start, log_timing);
}

}  // namespace

VoxtralRealtimeFrontend::VoxtralRealtimeFrontend(std::shared_ptr<const VoxtralRealtimeAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VoxTral frontend requires assets");
    }
    const auto & config = assets_->config.frontend;
    mel_filterbank_ = engine::audio::MelFilterbank().build_sparse(
        engine::audio::MelFilterbankConfig{
            config.sample_rate,
            config.n_fft,
            config.feature_size,
            0.0F,
            static_cast<float>(config.sample_rate) / 2.0F,
            true});
}

VoxtralRealtimeFeatures VoxtralRealtimeFrontend::extract(const runtime::AudioBuffer & audio, bool first_chunk) const {
    const auto total_start = Clock::now();
    const auto & config = assets_->config.frontend;
    const auto resample_start = Clock::now();
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("VoxTral audio input requires positive sample_rate and channels");
    }
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(config.sample_rate));
    engine::debug::timing_log_scalar(
        "voxtral_realtime.frontend.resample_ms",
        engine::debug::elapsed_ms(resample_start));
    const auto pad_start = Clock::now();
    mono = padded_streaming_audio(assets_->config, std::move(mono));
    engine::debug::timing_log_scalar(
        "voxtral_realtime.frontend.pad_ms",
        engine::debug::elapsed_ms(pad_start));
    auto out = extract_features_from_mono(config, mono, first_chunk, mel_filterbank_, nullptr, false, true);
    engine::debug::timing_log_scalar("voxtral_realtime.frontend.total_ms", engine::debug::elapsed_ms(total_start));
    return out;
}

VoxtralRealtimeFeatures VoxtralRealtimeFrontend::extract_stream_chunk(
    const runtime::AudioBuffer & audio,
    bool first_chunk,
    VoxtralRealtimeFrontendStreamState & state,
    int64_t steady_tokens) const {
    const auto & config = assets_->config.frontend;
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("VoxTral audio input requires positive sample_rate and channels");
    }
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(config.sample_rate));
    if (first_chunk) {
        if (first_stream_chunk_samples() <= 0) {
            throw std::runtime_error("VoxTral streaming frontend requires positive first chunk samples");
        }
        mono.resize(static_cast<size_t>(first_stream_chunk_samples()), 0.0F);
        mono = padded_first_stream_audio(assets_->config, std::move(mono));
    } else {
        if (steady_stream_chunk_samples(steady_tokens) <= 0) {
            throw std::runtime_error("VoxTral streaming frontend requires positive steady chunk samples");
        }
        mono.resize(static_cast<size_t>(steady_stream_chunk_samples(steady_tokens)), 0.0F);
    }
    const bool reuse_cached_prefix = !first_chunk && state.cached_frame_ready;
    return extract_features_from_mono(config, mono, first_chunk, mel_filterbank_, &state, reuse_cached_prefix, false);
}

int64_t VoxtralRealtimeFrontend::first_stream_chunk_samples() const {
    const auto & config = assets_->config;
    return ((config.default_num_delay_tokens + 1) * config.audio_length_per_tok - 1) *
        config.frontend.hop_length + config.frontend.win_length / 2;
}

int64_t VoxtralRealtimeFrontend::steady_stream_chunk_samples(int64_t steady_tokens) const {
    const auto & config = assets_->config;
    if (steady_tokens <= 0) {
        throw std::runtime_error("VoxTral streaming frontend requires a positive steady token count");
    }
    return steady_tokens * config.audio_length_per_tok * config.frontend.hop_length + config.frontend.win_length;
}

int64_t VoxtralRealtimeFrontend::first_stream_chunk_advance_samples() const {
    const auto & config = assets_->config;
    const int64_t frames = (config.default_num_delay_tokens + 1) * config.audio_length_per_tok;
    return frames * config.frontend.hop_length - config.frontend.n_fft / 2;
}

int64_t VoxtralRealtimeFrontend::steady_stream_chunk_advance_samples(int64_t steady_tokens) const {
    const auto & config = assets_->config;
    if (steady_tokens <= 0) {
        throw std::runtime_error("VoxTral streaming frontend requires a positive steady token count");
    }
    return steady_tokens * config.audio_length_per_tok * config.frontend.hop_length;
}

}  // namespace engine::models::voxtral_realtime
