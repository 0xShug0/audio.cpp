#include "engine/models/firered_audio/redae.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/codecs/redae_codec_runtime.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::firered_audio {
namespace {

namespace codecs = engine::codecs;

std::vector<float> mono_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("FireRedAudio prompt audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("FireRedAudio prompt audio channel count must be positive");
    }
    if (audio.channels == 1) {
        return audio.samples;
    }
    return audio::mixdown_interleaved_to_mono_average(
        audio.samples,
        audio.channels,
        audio::MonoMixAccumulation::Float32);
}

std::vector<float> pad_right_to_multiple(std::vector<float> audio, int64_t multiple) {
    if (multiple <= 0) {
        throw std::runtime_error("FireRedAudio audio pad multiple must be positive");
    }
    const int64_t samples = static_cast<int64_t>(audio.size());
    const int64_t target = ((samples + multiple - 1) / multiple) * multiple;
    if (target <= samples) {
        return audio;
    }
    audio.resize(static_cast<size_t>(target), 0.0F);
    return audio;
}

codecs::RedAeCodecConfig codec_config_from(const FireRedAudioRedAeConfig & c) {
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

codecs::RedAeCodecSources codec_sources_from(const std::shared_ptr<const FireRedAudioAssets> & assets) {
    if (assets == nullptr || assets->model_weights == nullptr || assets->redae_decoder_weights == nullptr) {
        throw std::runtime_error("FireRedAudio RedAE runtime requires encoder and decoder tensor sources");
    }
    codecs::RedAeCodecSources out;
    out.encoder = assets->model_weights;
    out.decoder = assets->redae_decoder_weights;
    return out;
}

codecs::RedAeCodecWeightBinding codec_binding() {
    codecs::RedAeCodecWeightBinding out;
    out.trace_prefix = "firered_audio.redae";
    out.weight_store_name = "firered_audio.redae.weights";
    out.enc_in0 = "red_vae.in_proj.0";
    out.enc_in1 = "red_vae.in_proj.1";
    out.encoder_qwen = "red_vae.qwen3";
    out.downsample_cls = "red_vae.downsample.cls_tok";
    out.downsample_qwen = "red_vae.downsample.qwen3";
    out.enc_out = "red_vae.out_proj";
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
    const FireRedAudioRedAeConfig & config,
    int64_t patch_size) {
    auto mono = mono_audio(audio);
    if (audio.sample_rate != config.sample_rate) {
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, static_cast<int>(config.sample_rate));
    }
    return pad_right_to_multiple(mono, config.audio_patch_size * config.enc_extra_downsample_rate * patch_size);
}

class FireRedAudioRedAeRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedAudioAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          runtime_(
              codec_sources_from(assets_),
              execution,
              codec_config_from(assets_->redae),
              codec_binding(),
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
    std::shared_ptr<const FireRedAudioAssets> assets_;
    codecs::RedAeCodecRuntime runtime_;
};

FireRedAudioRedAeRuntime::FireRedAudioRedAeRuntime(
    std::shared_ptr<const FireRedAudioAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

FireRedAudioRedAeRuntime::~FireRedAudioRedAeRuntime() = default;

std::vector<float> FireRedAudioRedAeRuntime::encode(const std::vector<float> & audio_24k) {
    return impl_->encode(audio_24k);
}

engine::runtime::AudioBuffer FireRedAudioRedAeRuntime::decode(const std::vector<float> & latents) {
    return impl_->decode(latents);
}

void FireRedAudioRedAeRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::firered_audio
