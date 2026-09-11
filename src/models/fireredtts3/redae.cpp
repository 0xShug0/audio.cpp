#include "engine/models/fireredtts3/redae.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/codecs/redae_codec_runtime.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::fireredtts3 {
namespace {

namespace codecs = engine::codecs;

std::vector<float> mono_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("FireRedTTS3 prompt audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("FireRedTTS3 prompt audio channel count must be positive");
    }
    if (audio.channels == 1) {
        return audio.samples;
    }
    return audio::mixdown_interleaved_to_mono_average(
        audio.samples,
        audio.channels,
        audio::MonoMixAccumulation::Float32);
}

std::vector<float> pad_left_to_multiple(std::vector<float> audio, int64_t multiple) {
    if (multiple <= 0) {
        throw std::runtime_error("FireRedTTS3 audio pad multiple must be positive");
    }
    const int64_t samples = static_cast<int64_t>(audio.size());
    const int64_t target = ((samples + multiple - 1) / multiple) * multiple;
    const int64_t pad = target - samples;
    if (pad <= 0) {
        return audio;
    }
    std::vector<float> out(static_cast<size_t>(target), 0.0F);
    std::copy(audio.begin(), audio.end(), out.begin() + static_cast<std::ptrdiff_t>(pad));
    return out;
}

codecs::RedAeCodecConfig codec_config_from(const FireRedTTS3RedAeConfig & c) {
    codecs::RedAeCodecConfig out;
    out.sample_rate = c.sample_rate;
    out.audio_patch_size = c.audio_patch_size;
    out.bottleneck_dim = c.bottleneck_dim;
    out.enc_hidden_size = c.enc_hidden_size;
    out.enc_intermediate_size = c.enc_intermediate_size;
    out.enc_layers = c.enc_layers;
    out.enc_heads = c.enc_heads;
    out.enc_kv_heads = c.enc_kv_heads;
    out.enc_head_dim = c.enc_head_dim;
    out.enc_sliding_window = c.enc_sliding_window;
    out.enc_extra_downsample_rate = c.enc_extra_downsample_rate;
    out.enc_downsample_layers = c.enc_downsample_layers;
    out.dec_hidden_size = c.dec_hidden_size;
    out.dec_intermediate_size = c.dec_intermediate_size;
    out.dec_layers = c.dec_layers;
    out.dec_heads = c.dec_heads;
    out.dec_kv_heads = c.dec_kv_heads;
    out.dec_head_dim = c.dec_head_dim;
    out.dec_sliding_window = c.dec_sliding_window;
    return out;
}

codecs::RedAeCodecSources codec_sources_from(const std::shared_ptr<const FireRedTTS3Assets> & assets) {
    if (assets == nullptr || assets->redae_weights == nullptr) {
        throw std::runtime_error("FireRedTTS3 RedAE runtime requires RedAE tensor source");
    }
    codecs::RedAeCodecSources out;
    out.encoder = assets->redae_weights;
    out.decoder = assets->redae_weights;
    return out;
}

codecs::RedAeCodecRuntimeOptions codec_options_from(
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    codecs::RedAeCodecRuntimeOptions out;
    out.graph_arena_bytes = graph_arena_bytes;
    out.weight_context_bytes = weight_context_bytes;
    out.weight_storage_type = storage_type;
    return out;
}

}  // namespace

std::vector<float> prepare_firered_prompt_audio_24k(
    const engine::runtime::AudioBuffer & audio,
    const FireRedTTS3RedAeConfig & config,
    int64_t patch_size) {
    auto mono = mono_audio(audio);
    if (audio.sample_rate != config.sample_rate) {
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, static_cast<int>(config.sample_rate));
    }
    return pad_left_to_multiple(mono, config.audio_patch_size * config.enc_extra_downsample_rate * patch_size);
}

class FireRedRedAeRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          runtime_(
              codec_sources_from(assets_),
              execution,
              codec_config_from(assets_->redae),
              codecs::RedAeCodecWeightBinding{
                  "fireredtts3.redae",
                  "fireredtts3.redae.weights"},
              codec_options_from(graph_arena_bytes, weight_context_bytes, storage_type)) {}

    std::vector<float> encode(const std::vector<float> & audio_24k) {
        return runtime_.encode(audio_24k);
    }

    runtime::AudioBuffer decode(const std::vector<float> & latents) {
        return runtime_.decode(latents);
    }

    void release_graphs() {
        runtime_.release_runtime_graphs();
    }

private:
    std::shared_ptr<const FireRedTTS3Assets> assets_;
    codecs::RedAeCodecRuntime runtime_;
};

FireRedRedAeRuntime::FireRedRedAeRuntime(
    std::shared_ptr<const FireRedTTS3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

FireRedRedAeRuntime::~FireRedRedAeRuntime() = default;

std::vector<float> FireRedRedAeRuntime::encode(const std::vector<float> & audio_24k) {
    return impl_->encode(audio_24k);
}

engine::runtime::AudioBuffer FireRedRedAeRuntime::decode(const std::vector<float> & latents) {
    return impl_->decode(latents);
}

void FireRedRedAeRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::fireredtts3
