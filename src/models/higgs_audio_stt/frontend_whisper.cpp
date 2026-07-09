#include "engine/models/higgs_audio_stt/frontend_whisper.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {
namespace {

constexpr float kLogFloor = 1.0e-10F;
using Clock = std::chrono::steady_clock;

void validate_audio_input(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Higgs Audio STT audio sample_rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("Higgs Audio STT audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Higgs Audio STT audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Higgs Audio STT interleaved audio size is not divisible by channel count");
    }
}

std::vector<float> mono_resampled(const runtime::AudioBuffer & audio, int sample_rate) {
    validate_audio_input(audio);
    return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        sample_rate);
}

int64_t ceil_to_multiple(int64_t value, int64_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

struct ChunkFeatures {
    std::vector<float> values;
    int64_t frames = 0;
};

ChunkFeatures compute_chunk_features(
    const std::vector<float> & samples,
    const engine::audio::SparseMelFilterbank & filterbank,
    const HiggsAudioSTTFrontendConfig & config) {
    std::vector<float> chunk = samples;
    if (static_cast<int64_t>(chunk.size()) < config.n_fft) {
        chunk.resize(static_cast<size_t>(config.n_fft), 0.0F);
    }
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    auto magnitude = engine::audio::STFT().compute_magnitude(
        chunk,
        window,
        1,
        static_cast<int64_t>(chunk.size()),
        stft_config);
    if (magnitude.shape.size() != 3) {
        throw std::runtime_error("Higgs Audio STT STFT returned unexpected rank");
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (stft_frames <= 1) {
        throw std::runtime_error("Higgs Audio STT STFT produced too few frames");
    }
    ChunkFeatures out;
    out.frames = stft_frames - 1;
    auto mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        freq_bins,
        stft_frames,
        out.frames,
        filterbank);
    if (mel.shape.size() != 3 || mel.shape[1] != config.feature_size || mel.shape[2] != out.frames) {
        throw std::runtime_error("Higgs Audio STT mel frontend returned unexpected shape");
    }
    float max_log = -INFINITY;
    for (float & value : mel.values) {
        value = std::log10(std::max(value, kLogFloor));
        max_log = std::max(max_log, value);
    }
    const float floor = max_log - 8.0F;
    for (float & value : mel.values) {
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    }
    out.values = std::move(mel.values);
    return out;
}

}  // namespace

HiggsAudioSTTWhisperFrontend::HiggsAudioSTTWhisperFrontend(std::shared_ptr<const HiggsAudioSTTAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs Audio STT Whisper frontend requires assets");
    }
    const auto & config = assets_->config.frontend;
    filterbank_ = engine::audio::MelFilterbank().build_sparse(
        engine::audio::MelFilterbankConfig{
            config.sample_rate,
            config.n_fft,
            config.feature_size,
            0.0F,
            static_cast<float>(config.sample_rate) / 2.0F,
            true,
        });
}

HiggsAudioSTTAudioFeatures HiggsAudioSTTWhisperFrontend::extract(const runtime::AudioBuffer & audio) const {
    const auto wall_start = Clock::now();
    const auto & config = assets_->config.frontend;
    if (config.sample_rate <= 0 || config.feature_size <= 0 || config.hop_length <= 0 || config.n_fft <= 0) {
        throw std::runtime_error("Higgs Audio STT Whisper frontend config is invalid");
    }
    if (!(config.chunk_size_seconds > 0.0)) {
        throw std::runtime_error("Higgs Audio STT chunk_size_seconds must be positive");
    }
    const auto resample_start = Clock::now();
    const auto samples = mono_resampled(audio, config.sample_rate);
    const auto resample_end = Clock::now();
    const int64_t chunk_samples = static_cast<int64_t>(std::llround(config.chunk_size_seconds * config.sample_rate));
    if (chunk_samples <= 0) {
        throw std::runtime_error("Higgs Audio STT chunk_size_seconds produced an empty chunk");
    }
    std::vector<ChunkFeatures> chunks;
    for (int64_t offset = 0; offset < static_cast<int64_t>(samples.size()); offset += chunk_samples) {
        const int64_t length = std::min<int64_t>(chunk_samples, static_cast<int64_t>(samples.size()) - offset);
        std::vector<float> chunk(
            samples.begin() + static_cast<std::ptrdiff_t>(offset),
            samples.begin() + static_cast<std::ptrdiff_t>(offset + length));
        chunks.push_back(compute_chunk_features(chunk, filterbank_, config));
    }
    if (chunks.empty()) {
        throw std::runtime_error("Higgs Audio STT produced no audio chunks");
    }

    int64_t max_frames = 0;
    for (const auto & chunk : chunks) {
        max_frames = std::max(max_frames, chunk.frames);
    }
    max_frames = ceil_to_multiple(max_frames, 16);

    HiggsAudioSTTAudioFeatures out;
    out.mel_bins = config.feature_size;
    out.chunks = static_cast<int64_t>(chunks.size());
    out.frames = max_frames;
    out.values.assign(static_cast<size_t>(out.chunks * out.mel_bins * out.frames), 0.0F);
    out.attention_mask.assign(static_cast<size_t>(out.chunks * out.frames), 0);
    out.valid_frames.reserve(chunks.size());
    out.encoder_tokens_per_chunk.reserve(chunks.size());
    const int64_t projector_stride = assets_->config.projector_temporal_downsample;
    for (int64_t b = 0; b < out.chunks; ++b) {
        const auto & chunk = chunks[static_cast<size_t>(b)];
        out.valid_frames.push_back(chunk.frames);
        const int64_t audio_tokens = higgs_audio_stt_audio_encoder_token_count(chunk.frames, projector_stride);
        out.encoder_tokens_per_chunk.push_back(audio_tokens);
        out.encoder_tokens += audio_tokens;
        for (int64_t mel = 0; mel < out.mel_bins; ++mel) {
            const size_t src = static_cast<size_t>(mel * chunk.frames);
            const size_t dst = static_cast<size_t>((b * out.mel_bins + mel) * out.frames);
            std::copy_n(chunk.values.begin() + static_cast<std::ptrdiff_t>(src), static_cast<size_t>(chunk.frames), out.values.begin() + static_cast<std::ptrdiff_t>(dst));
        }
        std::fill_n(
            out.attention_mask.begin() + static_cast<std::ptrdiff_t>(b * out.frames),
            static_cast<size_t>(chunk.frames),
            1);
    }

    debug::timing_log_scalar("higgs_audio_stt.frontend.resample_ms", engine::debug::elapsed_ms(resample_start, resample_end));
    debug::timing_log_scalar("higgs_audio_stt.frontend.total_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("higgs_audio_stt.frontend.chunks", out.chunks);
    debug::trace_log_scalar("higgs_audio_stt.frontend.frames", out.frames);
    debug::trace_log_scalar("higgs_audio_stt.frontend.audio_tokens", out.encoder_tokens);
    return out;
}

}  // namespace engine::models::higgs_audio_stt
