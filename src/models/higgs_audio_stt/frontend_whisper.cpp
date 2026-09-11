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
    const engine::audio::WhisperLogMelExtractor & extractor,
    const HiggsAudioSTTFrontendConfig & config) {
    std::vector<float> chunk = samples;
    if (static_cast<int64_t>(chunk.size()) < config.n_fft) {
        chunk.resize(static_cast<size_t>(config.n_fft), 0.0F);
    }
    auto features = extractor.compute(chunk);
    return {std::move(features.values), features.frames};
}

engine::audio::WhisperLogMelConfig make_extractor_config(const std::shared_ptr<const HiggsAudioSTTAssets> & assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs Audio STT Whisper frontend requires assets");
    }
    const auto & config = assets->config.frontend;
    return {
        config.sample_rate,
        config.n_fft,
        config.hop_length,
        config.feature_size,
        engine::audio::STFTFamily::Default,
    };
}

}  // namespace

HiggsAudioSTTWhisperFrontend::HiggsAudioSTTWhisperFrontend(std::shared_ptr<const HiggsAudioSTTAssets> assets)
    : assets_(std::move(assets)),
      extractor_(make_extractor_config(assets_)) {}

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
        chunks.push_back(compute_chunk_features(chunk, extractor_, config));
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
