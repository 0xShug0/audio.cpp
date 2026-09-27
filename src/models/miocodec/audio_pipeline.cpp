#include "engine/models/miocodec/audio_pipeline.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::models::miocodec {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMioCodecSslSampleRate = 16000;
constexpr int kMioCodecSslHopSize = 320;
constexpr int64_t kMioCodecHiddenSize = 768;
constexpr size_t kMaxCachedGlobalReferences = 4;

int64_t minimum_wavlm_input_length(int64_t desired_output_length) {
    static constexpr int kernels[] = {10, 3, 3, 3, 3, 2, 2};
    static constexpr int strides[] = {5, 2, 2, 2, 2, 2, 2};
    int64_t length = desired_output_length;
    for (int i = 6; i >= 0; --i) {
        length = (length - 1) * strides[i] + kernels[i];
    }
    return length;
}

int64_t waveform_padding_samples(int64_t audio_samples, int sample_rate) {
    if (audio_samples <= 0) {
        throw std::runtime_error("MioCodec padding calculation requires positive audio length");
    }
    const double samples_after_resampling =
        static_cast<double>(audio_samples) / static_cast<double>(sample_rate) * static_cast<double>(kMioCodecSslSampleRate);
    const auto expected_ssl_frames =
        static_cast<int64_t>(std::ceil(samples_after_resampling / static_cast<double>(kMioCodecSslHopSize)));
    const auto required_ssl_samples = minimum_wavlm_input_length(expected_ssl_frames);
    const double required_samples =
        static_cast<double>(required_ssl_samples) / static_cast<double>(kMioCodecSslSampleRate) * static_cast<double>(sample_rate);
    return std::max<int64_t>(0, static_cast<int64_t>(std::ceil((required_samples - static_cast<double>(audio_samples)) / 2.0)));
}

std::vector<float> zero_pad_sides(const std::vector<float> & samples, int64_t pad) {
    if (pad <= 0) {
        return samples;
    }
    std::vector<float> out(static_cast<size_t>(static_cast<int64_t>(samples.size()) + 2 * pad), 0.0F);
    std::copy(samples.begin(), samples.end(), out.begin() + static_cast<std::ptrdiff_t>(pad));
    return out;
}

const std::vector<float> & require_layer(
    const engine::modules::WavlmEncoderLayerOutput & layers,
    int64_t layer_index) {
    const auto it = std::find(layers.layer_indices.begin(), layers.layer_indices.end(), layer_index);
    if (it == layers.layer_indices.end()) {
        throw std::runtime_error("MioCodec WavLM output is missing requested layer");
    }
    return layers.hidden_states[static_cast<size_t>(std::distance(layers.layer_indices.begin(), it))];
}

std::vector<float> average_layers(
    const engine::modules::WavlmEncoderLayerOutput & layers,
    const std::vector<int> & layer_indices) {
    if (layer_indices.empty()) {
        throw std::runtime_error("MioCodec SSL layer list must not be empty");
    }
    const size_t count = static_cast<size_t>(layers.tokens * layers.hidden_size);
    if (layer_indices.size() == 1) {
        const auto & values = require_layer(layers, layer_indices.front());
        if (values.size() != count) {
            throw std::runtime_error("MioCodec WavLM layer output size mismatch");
        }
        return values;
    }
    if (layer_indices.size() == 2) {
        const auto & first = require_layer(layers, layer_indices.front());
        const auto & second = require_layer(layers, layer_indices.back());
        if (first.size() != count || second.size() != count) {
            throw std::runtime_error("MioCodec WavLM layer output size mismatch");
        }
        std::vector<float> out(count);
        for (size_t i = 0; i < count; ++i) {
            out[i] = (first[i] + second[i]) * 0.5F;
        }
        return out;
    }
    std::vector<float> out(count, 0.0F);
    for (const int layer : layer_indices) {
        const auto & values = require_layer(layers, layer);
        if (values.size() != count) {
            throw std::runtime_error("MioCodec WavLM layer output size mismatch");
        }
        for (size_t i = 0; i < count; ++i) {
            out[i] += values[i];
        }
    }
    const float scale = 1.0F / static_cast<float>(layer_indices.size());
    for (float & value : out) {
        value *= scale;
    }
    return out;
}

void normalize_ssl_features_in_place(std::vector<float> & features, int64_t frames, int64_t dim) {
    if (frames <= 1 || dim != kMioCodecHiddenSize ||
        static_cast<int64_t>(features.size()) != frames * dim) {
        throw std::runtime_error("MioCodec SSL normalization requires a valid [frames, 768] tensor with at least two frames");
    }
    std::array<double, kMioCodecHiddenSize> mean{};
    for (int64_t t = 0; t < frames; ++t) {
        const auto * row = features.data() + static_cast<size_t>(t * dim);
        for (int64_t c = 0; c < dim; ++c) {
            mean[static_cast<size_t>(c)] += static_cast<double>(row[c]);
        }
    }
    const double inv_frames = 1.0 / static_cast<double>(frames);
    for (double & value : mean) {
        value *= inv_frames;
    }
    std::array<double, kMioCodecHiddenSize> variance{};
    for (int64_t t = 0; t < frames; ++t) {
        const auto * row = features.data() + static_cast<size_t>(t * dim);
        for (int64_t c = 0; c < dim; ++c) {
            const double centered = static_cast<double>(row[c]) - mean[static_cast<size_t>(c)];
            variance[static_cast<size_t>(c)] += centered * centered;
        }
    }
    const double inv_variance_count = 1.0 / static_cast<double>(frames - 1);
    std::array<double, kMioCodecHiddenSize> inv_stddev{};
    for (int64_t c = 0; c < dim; ++c) {
        inv_stddev[static_cast<size_t>(c)] =
            1.0 / (std::sqrt(variance[static_cast<size_t>(c)] * inv_variance_count) + 1.0e-8);
    }
    for (int64_t t = 0; t < frames; ++t) {
        auto * row = features.data() + static_cast<size_t>(t * dim);
        for (int64_t c = 0; c < dim; ++c) {
            row[c] = static_cast<float>(
                (static_cast<double>(row[c]) - mean[static_cast<size_t>(c)]) *
                inv_stddev[static_cast<size_t>(c)]);
        }
    }
}

engine::modules::WavlmEncoderLayerOutput encode_ssl_layers(
    const engine::modules::WavlmEncoderComponent & wavlm,
    const std::vector<float> & mono_44k,
    const MioCodecAssets & assets,
    const std::vector<int> & layer_indices,
    const std::string & timing_prefix) {
    auto timing_start = Clock::now();
    const int64_t padding = waveform_padding_samples(static_cast<int64_t>(mono_44k.size()), assets.config.sample_rate);
    const auto padded = zero_pad_sides(mono_44k, padding);
    engine::debug::timing_log_scalar(timing_prefix + ".pad_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    const auto wavlm_input = engine::audio::resample_mono_torchaudio_sinc_hann(
        padded,
        assets.config.sample_rate,
        kMioCodecSslSampleRate);
    engine::debug::timing_log_scalar(timing_prefix + ".resample_ms", engine::debug::elapsed_ms(timing_start));
    std::vector<int64_t> output_layers;
    output_layers.reserve(layer_indices.size());
    for (const int layer : layer_indices) {
        output_layers.push_back(layer);
    }
    timing_start = Clock::now();
    const auto layers = wavlm.encode_layers(
        wavlm_input,
        1,
        static_cast<int64_t>(wavlm_input.size()),
        output_layers);
    engine::debug::timing_log_scalar(timing_prefix + ".wavlm_ms", engine::debug::elapsed_ms(timing_start));
    if (layers.batch != 1 || layers.tokens <= 0 || layers.hidden_size != kMioCodecHiddenSize) {
        throw std::runtime_error("MioCodec WavLM output shape mismatch");
    }
    return layers;
}

}  // namespace

std::vector<float> prepare_miocodec_mono_audio(
    const runtime::AudioBuffer & audio,
    int target_sample_rate) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("MioCodec audio input must contain non-empty audio with a valid sample rate and channel count");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MioCodec audio input sample count is not divisible by channel count");
    }
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != target_sample_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            target_sample_rate);
    }
    float max_abs = 0.0F;
    for (const float sample : mono) {
        max_abs = std::max(max_abs, std::abs(sample));
    }
    const float scale = 1.0F / (max_abs + 1.0e-8F);
    for (float & sample : mono) {
        sample *= scale;
    }
    return mono;
}

MioCodecSslFeatureExtractor::MioCodecSslFeatureExtractor(
    std::shared_ptr<const MioCodecAssets> assets,
    engine::modules::WavlmEncoderComponent wavlm,
    std::vector<int> layer_indices,
    bool normalize_features,
    std::string timing_prefix)
    : assets_(std::move(assets)),
      wavlm_(std::move(wavlm)),
      layer_indices_(std::move(layer_indices)),
      normalize_features_(normalize_features),
      timing_prefix_(std::move(timing_prefix)) {}

MioCodecSslFeatures MioCodecSslFeatureExtractor::extract(const std::vector<float> & mono_44k) const {
    const auto total_start = Clock::now();
    const auto layers = encode_ssl_layers(wavlm_, mono_44k, *assets_, layer_indices_, timing_prefix_);
    auto timing_start = Clock::now();
    MioCodecSslFeatures out;
    out.frames = layers.tokens;
    out.values = average_layers(layers, layer_indices_);
    if (normalize_features_) {
        normalize_ssl_features_in_place(out.values, out.frames, layers.hidden_size);
    }
    engine::debug::timing_log_scalar(timing_prefix_ + ".postprocess_ms", engine::debug::elapsed_ms(timing_start));
    engine::debug::timing_log_scalar(timing_prefix_ + ".total_ms", engine::debug::elapsed_ms(total_start));
    return out;
}

const engine::modules::WavlmEncoderComponent & MioCodecSslFeatureExtractor::wavlm() const noexcept {
    return wavlm_;
}

MioCodecGlobalReferenceEncoder::MioCodecGlobalReferenceEncoder(
    MioCodecSslFeatureExtractor ssl_extractor,
    MioCodecGlobalEncoderRuntime & global_encoder)
    : ssl_extractor_(std::move(ssl_extractor)),
      global_encoder_(&global_encoder) {}

MioCodecGlobalEmbedding MioCodecGlobalReferenceEncoder::embedding_for_reference(
    const std::vector<float> & reference_audio) const {
    const auto total_start = Clock::now();
    for (auto it = cached_references_.begin(); it != cached_references_.end(); ++it) {
        if (it->audio == reference_audio) {
            auto cached = std::move(*it);
            cached_references_.erase(it);
            auto embedding = cached.embedding;
            cached_references_.insert(cached_references_.begin(), std::move(cached));
            engine::debug::timing_log_scalar("miocodec.global_reference.cache_hit", true);
            engine::debug::timing_log_scalar("miocodec.global_reference.ssl_ms", 0.0);
            engine::debug::timing_log_scalar("miocodec.global_reference.encoder_ms", 0.0);
            engine::debug::timing_log_scalar("miocodec.global_reference.total_ms", engine::debug::elapsed_ms(total_start));
            return embedding;
        }
    }
    engine::debug::timing_log_scalar("miocodec.global_reference.cache_hit", false);
    auto timing_start = Clock::now();
    const auto reference_ssl = ssl_extractor_.extract(reference_audio);
    engine::debug::timing_log_scalar("miocodec.global_reference.ssl_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    auto embedding = global_encoder_->encode(reference_ssl.values, reference_ssl.frames);
    engine::debug::timing_log_scalar("miocodec.global_reference.encoder_ms", engine::debug::elapsed_ms(timing_start));
    cached_references_.insert(
        cached_references_.begin(),
        CachedReference{reference_audio, embedding});
    if (cached_references_.size() > kMaxCachedGlobalReferences) {
        cached_references_.pop_back();
    }
    engine::debug::timing_log_scalar("miocodec.global_reference.total_ms", engine::debug::elapsed_ms(total_start));
    return embedding;
}

MioCodecWaveformReconstructor::MioCodecWaveformReconstructor(
    std::shared_ptr<const MioCodecAssets> assets,
    core::ExecutionContext & execution_context)
    : assets_(std::move(assets)),
      execution_context_(&execution_context) {}

std::vector<float> MioCodecWaveformReconstructor::reconstruct(
    const MioCodecWaveHead & head,
    const std::vector<float> & window) {
    const auto threads = static_cast<size_t>(std::max(1, execution_context_->config().threads));
    if (execution_context_->backend_type() != engine::core::BackendType::Cuda) {
        if (host_istft_ == nullptr || host_istft_frames_ != head.frames) {
            host_istft_ = std::make_unique<engine::audio::HostLogMagnitudePhaseISTFT>(
                engine::audio::HostLogMagnitudePhaseISTFTConfig{
                    head.frames,
                    assets_->config.n_fft,
                    assets_->config.hop_length,
                    assets_->config.n_fft + 2,
                    threads,
                });
            host_istft_frames_ = head.frames;
            engine::debug::timing_log_scalar("miocodec.istft.workspace_rebuilt", true);
        } else {
            engine::debug::timing_log_scalar("miocodec.istft.workspace_rebuilt", false);
        }
        const auto result = host_istft_->compute(head.values, window);
        engine::debug::timing_log_scalar("miocodec.istft.spectrum_ms", result.timing.spectrum_ms);
        engine::debug::timing_log_scalar("miocodec.istft.framed_clear_ms", result.timing.framed_clear_ms);
        engine::debug::timing_log_scalar("miocodec.istft.fft_inverse_ms", result.timing.fft_inverse_ms);
        engine::debug::timing_log_scalar("miocodec.istft.fold_clear_ms", result.timing.fold_clear_ms);
        engine::debug::timing_log_scalar("miocodec.istft.overlap_add_ms", result.timing.overlap_add_ms);
        engine::debug::timing_log_scalar("miocodec.istft.normalize_ms", result.timing.normalize_ms);
        engine::debug::timing_log_scalar("miocodec.istft.total_ms", result.timing.total_ms);
        return result.audio;
    }
    if (cuda_istft_ == nullptr || cuda_istft_frames_ != head.frames) {
        const auto runtime_build_start = Clock::now();
        cuda_istft_ = std::make_unique<engine::audio::CudaLogMagnitudePhaseISTFT>(
            engine::audio::CudaLogMagnitudePhaseISTFTConfig{
                head.frames,
                assets_->config.n_fft,
                assets_->config.hop_length,
                assets_->config.n_fft + 2,
                execution_context_->config().device,
                true,
            });
        cuda_istft_frames_ = head.frames;
        engine::debug::timing_log_scalar("miocodec.istft.runtime_rebuilt", true);
        engine::debug::timing_log_scalar(
            "miocodec.istft.runtime_build_ms",
            engine::debug::elapsed_ms(runtime_build_start));
    } else {
        engine::debug::timing_log_scalar("miocodec.istft.runtime_rebuilt", false);
        engine::debug::timing_log_scalar("miocodec.istft.runtime_build_ms", 0.0);
    }
    const auto result = cuda_istft_->compute(head.values, window);
    engine::debug::timing_log_scalar("miocodec.istft.input_upload_ms", result.timing.input_upload_ms);
    engine::debug::timing_log_scalar("miocodec.istft.spectrum_kernel_ms", result.timing.spectrum_kernel_ms);
    engine::debug::timing_log_scalar("miocodec.istft.fft_inverse_ms", result.timing.fft_inverse_ms);
    engine::debug::timing_log_scalar("miocodec.istft.overlap_add_ms", result.timing.overlap_add_ms);
    engine::debug::timing_log_scalar("miocodec.istft.normalize_ms", result.timing.normalize_ms);
    engine::debug::timing_log_scalar("miocodec.istft.audio_read_ms", result.timing.audio_read_ms);
    engine::debug::timing_log_scalar("miocodec.istft.total_ms", result.timing.total_ms);
    return result.audio;
}

}  // namespace engine::models::miocodec
