#include "engine/models/vibevoice_asr/speech_encoder.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace {

constexpr int64_t kStreamingSegmentSeconds = 60;

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("VibeVoice-ASR speech encoder requires non-empty audio");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("VibeVoice-ASR speech encoder audio shape mismatch");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

void append_latents(VibeVoiceTokenizerLatents & dst, const VibeVoiceTokenizerLatents & src) {
    if (src.frames <= 0 || src.dim <= 0) {
        throw std::runtime_error("VibeVoice-ASR streaming tokenizer returned empty latents");
    }
    if (static_cast<int64_t>(src.values.size()) != src.frames * src.dim) {
        throw std::runtime_error("VibeVoice-ASR streaming tokenizer latent payload mismatch");
    }
    if (dst.frames == 0) {
        dst.dim = src.dim;
    } else if (dst.dim != src.dim) {
        throw std::runtime_error("VibeVoice-ASR streaming tokenizer latent dimension mismatch");
    }
    dst.frames += src.frames;
    dst.values.insert(dst.values.end(), src.values.begin(), src.values.end());
}

}  // namespace

VibeVoiceASRSpeechEncoder::VibeVoiceASRSpeechEncoder(
    std::shared_ptr<const VibeVoiceAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t tokenizer_weight_context_bytes,
    size_t connector_weight_context_bytes,
    assets::TensorStorageType tokenizer_weight_storage_type,
    assets::TensorStorageType connector_weight_storage_type)
    : assets_(std::move(assets)),
      sampling_policy_(
          backend_type == core::BackendType::Cuda
              ? std::optional<sampling::TorchCudaSamplingPolicy>(
                    sampling::resolve_torch_cuda_sampling_policy(
                        backend_type,
                        device,
                        "vibevoice_asr",
                        "VibeVoice-ASR",
                        sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda))
              : std::nullopt),
      tokenizer_(
          assets_,
          backend_type,
          device,
          threads,
          tokenizer_weight_context_bytes,
          128ull * 1024ull * 1024ull,
          tokenizer_weight_storage_type),
      connector_(
          assets_,
          backend_type,
          device,
          threads,
          connector_weight_context_bytes,
          64ull * 1024ull * 1024ull,
          connector_weight_storage_type) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeVoice-ASR speech encoder requires assets");
    }
}

VibeVoiceASRSpeechFeatures VibeVoiceASRSpeechEncoder::encode(
    const runtime::AudioBuffer & audio,
    uint64_t seed) const {
    const int64_t total_frames = audio_frame_count(audio);
    const int64_t segment_samples = static_cast<int64_t>(audio.sample_rate) * kStreamingSegmentSeconds;
    VibeVoiceTokenizerLatents acoustic_mean;
    VibeVoiceTokenizerLatents semantic_tokens;
    if (total_frames > segment_samples) {
        VibeVoiceTokenizerStreamingState acoustic_state;
        VibeVoiceTokenizerStreamingState semantic_state;
        for (int64_t start = 0; start < total_frames; start += segment_samples) {
            runtime::TimeSpan span;
            span.start_sample = start;
            span.end_sample = std::min(total_frames, start + segment_samples);
            const auto chunk = audio::slice_audio_buffer(audio, span);
            const bool is_final_chunk = span.end_sample == total_frames;
            append_latents(acoustic_mean, tokenizer_.encode_acoustic_streaming(chunk, acoustic_state, is_final_chunk));
            append_latents(semantic_tokens, tokenizer_.encode_semantic_streaming(chunk, semantic_state, is_final_chunk));
        }
    } else {
        acoustic_mean = tokenizer_.encode_acoustic(audio);
        semantic_tokens = tokenizer_.encode_semantic(audio);
    }
    if (acoustic_mean.frames != semantic_tokens.frames) {
        const int64_t frames = std::min(acoustic_mean.frames, semantic_tokens.frames);
        acoustic_mean.frames = frames;
        semantic_tokens.frames = frames;
        acoustic_mean.values.resize(static_cast<size_t>(frames * acoustic_mean.dim));
        semantic_tokens.values.resize(static_cast<size_t>(frames * semantic_tokens.dim));
    }
    const auto sampled = sample_vibevoice_acoustic_latents_gaussian(
        {acoustic_mean},
        assets_->config.acoustic_tokenizer.fix_std,
        seed,
        0,
        sampling::TorchRandnPrecision::BFloat16,
        sampling_policy_ ? &*sampling_policy_ : nullptr);
    const auto acoustic = connector_.project_acoustic(
        sampled.latents.front().values,
        sampled.latents.front().frames,
        sampled.latents.front().dim);
    const auto semantic = connector_.project_semantic(
        semantic_tokens.values,
        semantic_tokens.frames,
        semantic_tokens.dim);
    if (acoustic.frames != semantic.frames || acoustic.hidden_size != semantic.hidden_size) {
        throw std::runtime_error("VibeVoice-ASR acoustic and semantic connector shapes mismatch");
    }
    VibeVoiceASRSpeechFeatures out;
    out.frames = acoustic.frames;
    out.hidden_size = acoustic.hidden_size;
    out.next_rng_index = sampled.next_rng_index;
    out.values = acoustic.values;
    for (size_t i = 0; i < out.values.size(); ++i) {
        out.values[i] += semantic.values[i];
    }
    return out;
}

}  // namespace engine::models::vibevoice_asr
