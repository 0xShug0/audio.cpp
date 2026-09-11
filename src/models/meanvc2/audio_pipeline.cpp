#include "engine/models/meanvc2/audio_pipeline.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace engine::models::meanvc2 {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kFbankDims = 80;
constexpr int64_t kInputChunkSamples = 2560;
constexpr int64_t kSamplesCacheLen = 720;
constexpr int64_t kFbankCacheFrames = 3;
constexpr int64_t kBnWindow = 19;
constexpr int64_t kBnStride = 16;
constexpr int64_t kRequiredCacheSize = 8;
constexpr int64_t kAsrOffsetInit = 8;
constexpr int64_t kAsrOffsetReset = 4000;
constexpr int64_t kAsrOffsetStep = 4;
constexpr int64_t kBnDim = 256;
constexpr int64_t kVcConditionFrames = 16;

std::vector<float> mono_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("MeanVC2 audio input must contain non-empty audio with valid format");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MeanVC2 audio input sample count is not divisible by channel count");
    }
    return audio.channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

engine::audio::KaldiFbankOptions meanvc2_fbank_options() {
    engine::audio::KaldiFbankOptions options;
    options.sample_rate = kSampleRate;
    options.num_mels = kFbankDims;
    options.frame_length_ms = 25.0F;
    options.frame_shift_ms = 10.0F;
    options.lfr_m = 1;
    options.lfr_n = 1;
    options.window_type = engine::audio::KaldiFbankWindowType::Povey;
    options.preemphasis = 0.97F;
    options.low_frequency = 20.0F;
    options.high_frequency = 0.0F;
    options.remove_dc_offset = true;
    options.upscale_samples = true;
    options.apply_cmvn = false;
    return options;
}

}  // namespace

MeanVC2StreamingFrontend::MeanVC2StreamingFrontend() {
    reset();
}

void MeanVC2StreamingFrontend::reset() {
    samples_cache_.assign(static_cast<size_t>(kSamplesCacheLen), -0.5F);
    fbank_cache_.clear();
    has_fbank_cache_ = false;
    asr_offset_ = kAsrOffsetInit;
}

std::vector<MeanVC2FbankWindow> MeanVC2StreamingFrontend::encode_chunk(
    const float * samples,
    int64_t sample_count) {
    if (samples == nullptr || sample_count <= 0) {
        return {};
    }
    if (sample_count > kInputChunkSamples) {
        throw std::runtime_error("MeanVC2 frontend chunk exceeds 160 ms");
    }

    std::vector<float> padded;
    padded.reserve(samples_cache_.size() + static_cast<size_t>(sample_count));
    padded.insert(padded.end(), samples_cache_.begin(), samples_cache_.end());
    padded.insert(padded.end(), samples, samples + sample_count);
    const size_t cache_start = padded.size() > static_cast<size_t>(kSamplesCacheLen)
        ? padded.size() - static_cast<size_t>(kSamplesCacheLen)
        : 0;
    samples_cache_.assign(padded.begin() + static_cast<std::ptrdiff_t>(cache_start), padded.end());

    auto fbank = engine::audio::extract_kaldi_fbank(padded, meanvc2_fbank_options());
    if (fbank.frames <= 0) {
        return {};
    }
    if (fbank.feature_dim != kFbankDims) {
        throw std::runtime_error("MeanVC2 frontend fbank shape mismatch");
    }
    if (has_fbank_cache_) {
        std::vector<float> merged;
        merged.reserve(fbank_cache_.size() + fbank.values.size());
        merged.insert(merged.end(), fbank_cache_.begin(), fbank_cache_.end());
        merged.insert(merged.end(), fbank.values.begin(), fbank.values.end());
        fbank.frames += static_cast<int>(fbank_cache_.size() / static_cast<size_t>(kFbankDims));
        fbank.values = std::move(merged);
    }

    const int64_t cache_frames = std::min<int64_t>(kFbankCacheFrames, fbank.frames);
    fbank_cache_.assign(
        fbank.values.end() - static_cast<std::ptrdiff_t>(cache_frames * kFbankDims),
        fbank.values.end());
    has_fbank_cache_ = true;

    if (asr_offset_ >= kAsrOffsetReset) {
        asr_offset_ = std::max<int64_t>(kRequiredCacheSize, asr_offset_ - kAsrOffsetReset);
    }

    std::vector<MeanVC2FbankWindow> windows;
    for (int64_t start = 0; start + kBnWindow <= fbank.frames; start += kBnStride) {
        MeanVC2FbankWindow window;
        window.offset = asr_offset_;
        window.values.assign(
            fbank.values.begin() + static_cast<std::ptrdiff_t>(start * kFbankDims),
            fbank.values.begin() + static_cast<std::ptrdiff_t>((start + kBnWindow) * kFbankDims));
        windows.push_back(std::move(window));
        asr_offset_ += kAsrOffsetStep;
    }
    return windows;
}

void MeanVC2BnStreamAdapter::reset() {
    last_frame_.clear();
}

std::vector<float> MeanVC2BnStreamAdapter::append_encoded_frames(const std::vector<float> & encoded_frames) {
    if (encoded_frames.empty()) {
        return {};
    }
    if (encoded_frames.size() % static_cast<size_t>(kBnDim) != 0) {
        throw std::runtime_error("MeanVC2 ASR encoded frame shape mismatch");
    }

    std::vector<float> frames;
    frames.reserve(last_frame_.size() + encoded_frames.size());
    frames.insert(frames.end(), last_frame_.begin(), last_frame_.end());
    frames.insert(frames.end(), encoded_frames.begin(), encoded_frames.end());

    const int64_t input_frames = static_cast<int64_t>(frames.size() / static_cast<size_t>(kBnDim));
    last_frame_.assign(
        frames.end() - static_cast<std::ptrdiff_t>(kBnDim),
        frames.end());
    if (input_frames < 2) {
        return {};
    }

    std::vector<float> out(static_cast<size_t>(kVcConditionFrames * kBnDim), 0.0F);
    const float input_scale = static_cast<float>(input_frames - 1) / static_cast<float>(kVcConditionFrames);
    for (int64_t out_frame = 0; out_frame < kVcConditionFrames; ++out_frame) {
        const float input_pos = static_cast<float>(out_frame + 1) * input_scale;
        const int64_t left = std::min<int64_t>(static_cast<int64_t>(input_pos), input_frames - 1);
        const int64_t right = std::min<int64_t>(left + 1, input_frames - 1);
        const float alpha = input_pos - static_cast<float>(left);
        for (int64_t dim = 0; dim < kBnDim; ++dim) {
            const float a = frames[static_cast<size_t>(left * kBnDim + dim)];
            const float b = frames[static_cast<size_t>(right * kBnDim + dim)];
            out[static_cast<size_t>(out_frame * kBnDim + dim)] = a + (b - a) * alpha;
        }
    }
    return out;
}

MeanVC2PreparedAudio prepare_meanvc2_audio_16k(const runtime::AudioBuffer & audio) {
    MeanVC2PreparedAudio prepared;
    prepared.mono_16k = mono_audio(audio);
    if (audio.sample_rate != kSampleRate) {
        prepared.mono_16k = engine::audio::resample_mono_torchaudio_sinc_hann(
            prepared.mono_16k,
            audio.sample_rate,
            kSampleRate);
    }
    return prepared;
}

MeanVC2PreparedAudio prepare_meanvc2_source_audio_16k(const runtime::AudioBuffer & audio) {
    MeanVC2PreparedAudio prepared;
    prepared.mono_16k = mono_audio(audio);
    if (audio.sample_rate != kSampleRate) {
        engine::audio::SoxrResampleOptions options;
        options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
        options.warning_context = "MeanVC2 source";
        options.fallback_description = "torchaudio sinc-Hann resampling";
        if (auto resampled = engine::audio::try_resample_mono_soxr(
                prepared.mono_16k,
                audio.sample_rate,
                kSampleRate,
                options)) {
            prepared.mono_16k = std::move(*resampled);
        } else {
            prepared.mono_16k = engine::audio::resample_mono_torchaudio_sinc_hann(
                prepared.mono_16k,
                audio.sample_rate,
                kSampleRate);
        }
    }
    return prepared;
}

}  // namespace engine::models::meanvc2
