#include "engine/models/confucius4_tts/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/yaml.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::confucius4_tts {
namespace {

namespace json = engine::io::json;
namespace yaml = engine::io::yaml;

ConfuciusT2SConfig parse_t2s_config(const yaml::FlattenedDocument & document) {
    ConfuciusT2SConfig config;
    config.num_layers = yaml::require_i64(document, "t2s_model.num_layers");
    config.model_dim = yaml::require_i64(document, "t2s_model.model_dim");
    config.num_heads = yaml::require_i64(document, "t2s_model.num_heads");
    config.max_text_seq_lens = yaml::require_i64(document, "t2s_model.max_text_seq_lens");
    config.max_semantic_seq_lens = yaml::require_i64(document, "t2s_model.max_semantic_seq_lens");
    config.vocab_size = yaml::require_i64(document, "t2s_model.vocab_size");
    config.semantic_vocab_size = yaml::require_i64(document, "t2s_model.semantic_vocab_size");
    config.text_embedding_dim = yaml::require_i64(document, "t2s_model.text_embedding_dim");
    config.speaker_embedding_dim = yaml::require_i64(document, "t2s_model.speaker_embedding_dim");
    config.start_semantic_token = yaml::require_i64(document, "t2s_model.start_semantic_token");
    config.stop_semantic_token = yaml::require_i64(document, "t2s_model.stop_semantic_token");
    return config;
}

ConfuciusS2AConfig parse_s2a_config(const yaml::FlattenedDocument & document) {
    ConfuciusS2AConfig config;
    config.input_size = yaml::require_i64(document, "s2a_model.input_size");
    config.output_size = yaml::require_i64(document, "s2a_model.output_size");
    config.spk_embed_dim = yaml::require_i64(document, "s2a_model.spk_embed_dim");
    config.semantic_embed_dim = yaml::require_i64(document, "s2a_model.semantic_embed_dim");
    config.lm_latent_dim = yaml::require_i64(document, "s2a_model.lm_latent_dim");
    config.estimator_mlp_ratio = yaml::require_float(document, "s2a_model.estimator_mlp_ratio");
    return config;
}

ConfuciusAudioConfig parse_audio_config(const yaml::FlattenedDocument & document) {
    ConfuciusAudioConfig config;
    config.sample_rate = yaml::require_int(document, "audio.target_sample_rate");
    config.prompt_sample_rate = yaml::require_int(document, "audio.prompt_sample_rate");
    config.n_fft = yaml::require_i64(document, "audio.n_fft");
    config.hop_length = yaml::require_i64(document, "audio.hop_length");
    config.win_length = yaml::require_i64(document, "audio.win_length");
    config.n_mels = yaml::require_i64(document, "audio.n_mels");
    config.fmin = yaml::optional_f32(document, "audio.fmin", config.fmin);
    config.fmax = yaml::optional_nullable_f32(document, "audio.fmax");
    return config;
}

ConfuciusStyleEncoderConfig parse_style_config(const yaml::FlattenedDocument & document) {
    ConfuciusStyleEncoderConfig config;
    config.feat_dim = yaml::require_i64(document, "paths.style_encoder.init_args.feat_dim");
    config.embedding_size = yaml::require_i64(document, "paths.style_encoder.init_args.embedding_size");
    return config;
}

ConfuciusPreprocessorConfig parse_preprocessor_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("w2v_preprocessor_config");
    ConfuciusPreprocessorConfig config;
    config.feature_extractor_type = json::require_string(root, "feature_extractor_type");
    config.feature_size = json::require_i64(root, "feature_size");
    config.num_mel_bins = json::require_i64(root, "num_mel_bins");
    config.padding_side = json::require_string(root, "padding_side");
    config.padding_value = json::optional_f32(root, "padding_value", config.padding_value);
    config.processor_class = json::require_string(root, "processor_class");
    config.return_attention_mask = json::optional_bool(root, "return_attention_mask", config.return_attention_mask);
    config.sampling_rate = static_cast<int>(json::require_i64(root, "sampling_rate"));
    config.stride = json::require_i64(root, "stride");
    return config;
}

ConfuciusBigVganConfig parse_vocoder_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("bigvgan_config");
    ConfuciusBigVganConfig config;
    config.sample_rate = static_cast<int>(json::require_i64(root, "sampling_rate"));
    config.n_fft = json::require_i64(root, "n_fft");
    config.hop_size = json::require_i64(root, "hop_size");
    config.win_size = json::require_i64(root, "win_size");
    config.num_mels = json::require_i64(root, "num_mels");
    config.upsample_initial_channel = json::require_i64(root, "upsample_initial_channel");
    return config;
}

ConfuciusConfig parse_config(const assets::ResourceBundle & resources) {
    const auto document = resources.parse_flattened_yaml("config");
    ConfuciusConfig config;
    config.t2s = parse_t2s_config(document);
    config.s2a = parse_s2a_config(document);
    config.audio = parse_audio_config(document);
    config.style_encoder = parse_style_config(document);
    config.preprocessor = parse_preprocessor_config(resources);
    config.vocoder = parse_vocoder_config(resources);
    return config;
}

void validate_config(const ConfuciusConfig & config) {
    engine::io::require_positive(config.t2s.num_layers, "t2s num_layers");
    engine::io::require_positive(config.t2s.model_dim, "t2s model_dim");
    engine::io::require_positive(config.t2s.num_heads, "t2s num_heads");
    engine::io::require_divisible(config.t2s.model_dim, config.t2s.num_heads, "t2s model_dim / num_heads");
    engine::io::require_positive(config.t2s.vocab_size, "t2s vocab_size");
    engine::io::require_positive(config.t2s.semantic_vocab_size, "t2s semantic_vocab_size");
    engine::io::require_positive(config.s2a.input_size, "s2a input_size");
    engine::io::require_positive(config.s2a.output_size, "s2a output_size");
    engine::io::require_positive(config.s2a.spk_embed_dim, "s2a spk_embed_dim");
    engine::io::require_positive(config.audio.sample_rate, "audio target_sample_rate");
    engine::io::require_positive(config.audio.prompt_sample_rate, "audio prompt_sample_rate");
    engine::io::require_positive(config.audio.n_mels, "audio n_mels");
    if (config.style_encoder.feat_dim != config.audio.n_mels) {
        throw std::runtime_error("Confucius4-TTS style encoder feat_dim must match audio n_mels");
    }
    if (config.s2a.output_size != config.audio.n_mels) {
        throw std::runtime_error("Confucius4-TTS S2A output_size must match audio n_mels");
    }
    if (config.s2a.spk_embed_dim != config.style_encoder.embedding_size) {
        throw std::runtime_error("Confucius4-TTS S2A speaker dimension must match style encoder embedding size");
    }
    if (config.preprocessor.sampling_rate != config.audio.prompt_sample_rate) {
        throw std::runtime_error("Confucius4-TTS W2V preprocessor sampling_rate must match prompt_sample_rate");
    }
    if (config.preprocessor.num_mel_bins != config.style_encoder.feat_dim) {
        throw std::runtime_error("Confucius4-TTS W2V preprocessor num_mel_bins must match style feat_dim");
    }
    if (config.vocoder.sample_rate != config.audio.sample_rate ||
        config.vocoder.n_fft != config.audio.n_fft ||
        config.vocoder.hop_size != config.audio.hop_length ||
        config.vocoder.win_size != config.audio.win_length ||
        config.vocoder.num_mels != config.audio.n_mels) {
        throw std::runtime_error("Confucius4-TTS BigVGAN config does not match audio config");
    }
}

void validate_t2s_weights(const ConfuciusConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "text_projector.embed.weight", {config.t2s.vocab_size, config.t2s.text_embedding_dim});
    assets::require_tensor_shape(source, "text_projector.text_projection_fc2.weight", {config.t2s.model_dim, config.t2s.text_embedding_dim});
    assets::require_tensor_shape(source, "semantic_embedding.weight", {config.t2s.semantic_vocab_size, config.t2s.model_dim});
    assets::require_tensor_shape(source, "semantic_position_embedding.embedding.weight", {config.t2s.max_semantic_seq_lens, config.t2s.model_dim});
    assets::require_tensor_shape(source, "transformer.h.0.attn.c_attn.weight", {config.t2s.model_dim, config.t2s.model_dim * 3});
    assets::require_tensor_shape(source, "transformer.h.0.attn.c_proj.weight", {config.t2s.model_dim, config.t2s.model_dim});
    assets::require_tensor_shape(source, "transformer.h.0.mlp.c_fc.weight", {config.t2s.model_dim, config.t2s.model_dim * 4});
    assets::require_tensor_shape(source, "speaker_encoder.blocks.0.conv.weight", {512, config.t2s.speaker_embedding_dim, 5});
    assets::require_tensor_shape(source, "final_norm.weight", {config.t2s.model_dim});
    assets::require_tensor_shape(source, "semantic_head.weight", {config.t2s.semantic_vocab_size, config.t2s.model_dim});
}

void validate_s2a_weights(const ConfuciusConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "input_embedding.embedding.weight", {8192, 8});
    assets::require_tensor_shape(source, "input_embedding.out_project.weight", {config.s2a.semantic_embed_dim, 8, 1});
    assets::require_tensor_shape(source, "encoder_proj.weight", {config.s2a.input_size * 2, config.s2a.lm_latent_dim + config.s2a.semantic_embed_dim});
    assets::require_tensor_shape(source, "length_regulator.model.0.weight", {config.s2a.input_size, config.s2a.input_size, 3});
    assets::require_tensor_shape(
        source,
        "decoder.estimator.input_embed.proj.weight",
        {config.s2a.input_size, config.s2a.output_size * 2 + config.s2a.input_size + config.s2a.spk_embed_dim});
    assets::require_tensor_shape(source, "decoder.estimator.final_layer.adaLN_modulation.1.weight", {config.s2a.input_size * 2, config.s2a.input_size});
    assets::require_tensor_shape(source, "decoder.estimator.conv2.weight", {config.s2a.output_size, config.s2a.input_size, 1});
    assets::require_tensor_shape(source, "prompt_cond", {1, 1, config.s2a.input_size});
}

void validate_semantic_encoder_weights(const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "feature_projection.projection.weight", {1024, 160});
    assets::require_tensor_shape(source, "encoder.layers.0.self_attn.linear_k.weight", {1024, 1024});
    assets::require_tensor_shape(source, "encoder.layers.0.conv_module.depthwise_conv.weight", {1024, 1, 31});
}

void validate_semantic_encoder_shaw_weights(const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "encoder.layers.0.self_attn.k_proj.weight", {1024, 1024});
    assets::require_tensor_shape(source, "encoder.layers.0.conv.depthwise_conv.weight", {1024, 1, 31});
    assets::require_tensor_shape(source, "final_proj.weight", {768, 1024});
}

void validate_style_encoder_weights(const ConfuciusConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "speaker_encoder.head.conv1.weight", {32, 1, 3, 3});
    assets::require_tensor_shape(source, "speaker_encoder.xvector.transit3.linear.weight", {512, 1024, 1});
    assets::require_tensor_shape(source, "speaker_encoder.xvector.transit3.nonlinear.batchnorm.weight", {1024});
    assets::require_tensor_shape(source, "speaker_encoder.head.bn1.weight", {32});
    if (config.style_encoder.embedding_size != 192) {
        throw std::runtime_error("Confucius4-TTS CAMPPlus style embedding size must be 192");
    }
}

void validate_vocoder_weights(const ConfuciusConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "conv_pre.weight_v", {config.vocoder.upsample_initial_channel, config.audio.n_mels, 7});
    assets::require_tensor_shape(source, "ups.0.0.weight_v", {config.vocoder.upsample_initial_channel, 768, 8});
    assets::require_tensor_shape(source, "ups.5.0.weight_v", {48, 24, 4});
    assets::require_tensor_shape(source, "conv_post.weight_v", {1, 24, 7});
}

void validate_semantic_stats(const ConfuciusConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "mean", {config.s2a.semantic_embed_dim});
    assets::require_tensor_shape(source, "var", {config.s2a.semantic_embed_dim});
}

void validate_weight_anchors(const ConfuciusAssets & assets) {
    validate_t2s_weights(assets.config, *assets.t2s_weights);
    validate_s2a_weights(assets.config, *assets.s2a_weights);
    validate_semantic_encoder_weights(*assets.semantic_encoder_weights);
    validate_semantic_encoder_shaw_weights(*assets.semantic_encoder_shaw_weights);
    validate_semantic_stats(assets.config, *assets.semantic_stats);
    validate_style_encoder_weights(assets.config, *assets.style_encoder_weights);
    validate_vocoder_weights(assets.config, *assets.vocoder_weights);
}

}  // namespace

std::shared_ptr<const ConfuciusAssets> load_confucius_assets(const std::filesystem::path & model_path) {
    ConfuciusAssets assets;
    assets.resources = model_spec::load_resource_bundle(
        model_path,
        model_spec::default_spec_path("confucius4_tts"));
    assets.config = parse_config(assets.resources);
    validate_config(assets.config);
    assets.t2s_weights = assets.resources.open_tensor_source("t2s");
    assets.s2a_weights = assets.resources.open_tensor_source("s2a");
    assets.semantic_encoder_weights = assets.resources.open_tensor_source("semantic_encoder");
    assets.semantic_encoder_shaw_weights = assets.resources.open_tensor_source("semantic_encoder_shaw");
    assets.semantic_stats = assets.resources.open_tensor_source("semantic_stats");
    assets.style_encoder_weights = assets.resources.open_tensor_source("style_encoder");
    assets.vocoder_weights = assets.resources.open_tensor_source("vocoder");
    validate_weight_anchors(assets);
    return std::make_shared<ConfuciusAssets>(std::move(assets));
}

}  // namespace engine::models::confucius4_tts
