#include "engine/models/chatterbox_turbo/s3gen_turbo.h"

#include "components/dot_prefix_source.h"
#include "components/s3gen_name_bridge.h"

#include <stdexcept>

namespace engine::models::chatterbox_turbo {

namespace {

engine::modules::HiftVocoderConfig make_turbo_hift_config(engine::assets::TensorStorageType weight_storage_type) {
    // Matches the S3Gen GGUF's `chatterbox.s3gen.*` metadata and base Chatterbox's own HiFT
    // config (src/models/chatterbox/hift_vocoder_impl.cpp) — the vocoder architecture is
    // unchanged by the meanflow distillation, only the tensor names differ (see
    // s3gen_name_bridge.h). The turbo GGUF folds torch weight-norm parametrization into a plain
    // "weight" tensor at conversion time (no "parametrizations.weight.original0/1" keys), so
    // this uses the Canonical (plain) layout rather than TorchParametrizedWeightNorm.
    engine::modules::HiftVocoderConfig config;
    config.in_channels = 80;
    config.base_channels = 512;
    config.nb_harmonics = 8;
    config.sampling_rate = 24000;
    config.nsf_alpha = 0.1F;
    config.nsf_sigma = 0.003F;
    config.nsf_voiced_threshold = 10.0F;
    config.upsample_rates = {8, 5, 3};
    config.upsample_kernel_sizes = {16, 11, 7};
    config.istft_n_fft = 16;
    config.istft_hop = 4;
    config.resblock_kernel_sizes = {3, 7, 11};
    config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.source_resblock_kernel_sizes = {7, 7, 11};
    config.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.lrelu_slope = 0.1F;
    config.audio_limit = 0.99F;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
    config.weight_storage_type = weight_storage_type;
    // The "v." vocoder namespace is applied inside translate_vocoder_name (see
    // s3gen_name_bridge.cpp) rather than here, since that translation runs on top of the raw
    // dot-prefixed source and needs to see the plain (unprefixed) name HiftVocoderComponent asks
    // for in order to match its own prefix rules.
    config.tensor_prefix = "";
    config.weight_layout = engine::modules::HiftVocoderWeightLayout::Canonical;
    return config;
}

}  // namespace

std::filesystem::path derive_turbo_s3gen_gguf_path(const std::filesystem::path & t3_gguf_path) {
    const std::string filename = t3_gguf_path.filename().string();
    const std::string marker = "-t3-";
    const auto pos = filename.find(marker);
    if (pos == std::string::npos) {
        throw std::runtime_error(
            "Chatterbox Turbo T3 GGUF filename does not match the expected '...-t3-<quant>.gguf' naming: " + filename);
    }
    const std::string s3gen_filename = filename.substr(0, pos) + "-s3gen-" + filename.substr(pos + marker.size());
    return t3_gguf_path.parent_path() / s3gen_filename;
}

std::shared_ptr<ChatterboxTurboS3Gen> ChatterboxTurboS3Gen::load(
    const std::filesystem::path & s3gen_gguf_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto raw_source = make_dot_prefixed_tensor_source(engine::assets::open_tensor_source(s3gen_gguf_path), "s3");
    auto flow_source = make_turbo_s3gen_flow_bridge(raw_source);
    auto vocoder_source = make_turbo_vocoder_bridge(raw_source);

    auto out = std::make_shared<ChatterboxTurboS3Gen>();
    out->execution_context_ = &execution_context;
    out->encoder_weights_ =
        engine::models::chatterbox::load_s3_flow_encoder_weights(*flow_source, execution_context, weight_storage_type);
    out->decoder_weights_ =
        engine::models::chatterbox::load_s3_flow_decoder_weights(*flow_source, execution_context, weight_storage_type);
    if (!engine::models::chatterbox::s3_flow_decoder_is_meanflow(*out->decoder_weights_)) {
        throw std::runtime_error(
            "Chatterbox Turbo S3Gen GGUF is missing the meanflow time_embed_mixer tensor (flow.decoder.estimator.time_embed_mixer)");
    }
    out->vocoder_ = std::make_shared<engine::modules::HiftVocoderComponent>(
        engine::modules::HiftVocoderComponent::load_from_tensor_source(
            vocoder_source, execution_context.config(), make_turbo_hift_config(weight_storage_type)));
    return out;
}

engine::models::chatterbox::S3GenInferenceOutputs ChatterboxTurboS3Gen::synthesize(
    const engine::models::chatterbox::EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    uint64_t flow_seed,
    uint64_t vocoder_seed) const {
    // n_cfm_timesteps=2 matches tts_turbo.py's ChatterboxTurboTTS.generate default; cfg_rate and
    // cosine_schedule are unused on the meanflow path (see s3gen_inference.cpp's
    // decoder_weights.meanflow branch).
    const auto mel = engine::models::chatterbox::compute_s3_token2mel_inference(
        cache_,
        *encoder_weights_,
        *decoder_weights_,
        ref_dict,
        speech_tokens,
        static_cast<int64_t>(speech_tokens.size()),
        /*num_steps=*/2,
        /*cfg_rate=*/0.0f,
        /*cosine_schedule=*/false,
        /*full_noise=*/{},
        flow_seed,
        execution_context_->config(),
        /*timing=*/nullptr);

    const auto voc = vocoder_->synthesize(mel.mel, mel.frames, vocoder_seed);

    engine::models::chatterbox::S3GenInferenceOutputs outputs;
    outputs.waveform = voc.waveform;
    outputs.samples = voc.samples;
    outputs.mel = mel.mel;
    outputs.mel_channels = mel.channels;
    outputs.mel_frames = mel.frames;
    return outputs;
}

}  // namespace engine::models::chatterbox_turbo
