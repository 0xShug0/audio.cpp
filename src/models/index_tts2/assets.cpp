#include "engine/models/index_tts2/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/yaml.h"

#include <stdexcept>

namespace engine::models::index_tts2 {
namespace {

namespace json = engine::io::json;
namespace yaml = engine::io::yaml;

IndexTTS2Config parse_config(const assets::ResourceBundle & resources) {
    const auto document = resources.parse_flattened_yaml("config");
    IndexTTS2Config config;
    config.version = yaml::optional_string(document, "version", config.version);
    config.dataset_sample_rate = static_cast<int>(yaml::require_i64(document, "dataset.sample_rate"));
    config.dataset_squeeze = yaml::optional_bool(document, "dataset.squeeze", config.dataset_squeeze);
    config.dataset_mel_sample_rate = static_cast<int>(yaml::require_i64(document, "dataset.mel.sample_rate"));
    config.dataset_mel_n_fft = yaml::require_i64(document, "dataset.mel.n_fft");
    config.dataset_mel_hop_length = yaml::require_i64(document, "dataset.mel.hop_length");
    config.dataset_mel_win_length = yaml::require_i64(document, "dataset.mel.win_length");
    config.dataset_mel_n_mels = yaml::require_i64(document, "dataset.mel.n_mels");
    config.dataset_mel_fmin = yaml::optional_f32(document, "dataset.mel.mel_fmin", config.dataset_mel_fmin);
    config.dataset_mel_normalize = yaml::optional_bool(document, "dataset.mel.normalize", config.dataset_mel_normalize);

    config.gpt.model_dim = yaml::require_i64(document, "gpt.model_dim");
    config.gpt.max_mel_tokens = yaml::require_i64(document, "gpt.max_mel_tokens");
    config.gpt.max_text_tokens = yaml::require_i64(document, "gpt.max_text_tokens");
    config.gpt.heads = yaml::require_i64(document, "gpt.heads");
    config.gpt.use_mel_codes_as_input = yaml::optional_bool(document, "gpt.use_mel_codes_as_input", config.gpt.use_mel_codes_as_input);
    config.gpt.mel_length_compression = yaml::require_i64(document, "gpt.mel_length_compression");
    config.gpt.layers = yaml::require_i64(document, "gpt.layers");
    config.gpt.number_text_tokens = yaml::require_i64(document, "gpt.number_text_tokens");
    config.gpt.number_mel_codes = yaml::require_i64(document, "gpt.number_mel_codes");
    config.gpt.start_mel_token = yaml::require_i64(document, "gpt.start_mel_token");
    config.gpt.stop_mel_token = yaml::require_i64(document, "gpt.stop_mel_token");
    config.gpt.start_text_token = yaml::require_i64(document, "gpt.start_text_token");
    config.gpt.stop_text_token = yaml::require_i64(document, "gpt.stop_text_token");
    config.gpt.train_solo_embeddings = yaml::optional_bool(document, "gpt.train_solo_embeddings", config.gpt.train_solo_embeddings);
    config.gpt.condition_type = yaml::require_string(document, "gpt.condition_type");
    config.gpt.condition_output_size = yaml::require_i64(document, "gpt.condition_module.output_size");
    config.gpt.condition_linear_units = yaml::require_i64(document, "gpt.condition_module.linear_units");
    config.gpt.condition_attention_heads = yaml::require_i64(document, "gpt.condition_module.attention_heads");
    config.gpt.condition_num_blocks = yaml::require_i64(document, "gpt.condition_module.num_blocks");
    config.gpt.condition_input_layer = yaml::require_string(document, "gpt.condition_module.input_layer");
    config.gpt.condition_perceiver_mult = yaml::require_i64(document, "gpt.condition_module.perceiver_mult");
    config.gpt.emo_condition_output_size = yaml::require_i64(document, "gpt.emo_condition_module.output_size");
    config.gpt.emo_condition_linear_units = yaml::require_i64(document, "gpt.emo_condition_module.linear_units");
    config.gpt.emo_condition_attention_heads = yaml::require_i64(document, "gpt.emo_condition_module.attention_heads");
    config.gpt.emo_condition_num_blocks = yaml::require_i64(document, "gpt.emo_condition_module.num_blocks");
    config.gpt.emo_condition_input_layer = yaml::require_string(document, "gpt.emo_condition_module.input_layer");
    config.gpt.emo_condition_perceiver_mult = yaml::require_i64(document, "gpt.emo_condition_module.perceiver_mult");

    config.semantic_codec.codebook_size = yaml::require_i64(document, "semantic_codec.codebook_size");
    config.semantic_codec.hidden_size = yaml::require_i64(document, "semantic_codec.hidden_size");
    config.semantic_codec.codebook_dim = yaml::require_i64(document, "semantic_codec.codebook_dim");
    config.semantic_codec.vocos_dim = yaml::require_i64(document, "semantic_codec.vocos_dim");
    config.semantic_codec.vocos_intermediate_dim = yaml::require_i64(document, "semantic_codec.vocos_intermediate_dim");
    config.semantic_codec.vocos_num_layers = yaml::require_i64(document, "semantic_codec.vocos_num_layers");

    config.s2mel.sample_rate = static_cast<int>(yaml::require_i64(document, "s2mel.preprocess_params.sr"));
    config.s2mel.n_fft = yaml::require_i64(document, "s2mel.preprocess_params.spect_params.n_fft");
    config.s2mel.win_length = yaml::require_i64(document, "s2mel.preprocess_params.spect_params.win_length");
    config.s2mel.hop_length = yaml::require_i64(document, "s2mel.preprocess_params.spect_params.hop_length");
    config.s2mel.n_mels = yaml::require_i64(document, "s2mel.preprocess_params.spect_params.n_mels");
    config.s2mel.fmin = yaml::optional_f32(document, "s2mel.preprocess_params.spect_params.fmin", config.s2mel.fmin);
    config.s2mel.fmax = yaml::optional_nullable_f32(document, "s2mel.preprocess_params.spect_params.fmax");
    config.s2mel.dit_type = yaml::require_string(document, "s2mel.dit_type");
    config.s2mel.reg_loss_type = yaml::require_string(document, "s2mel.reg_loss_type");
    config.s2mel.style_dim = yaml::require_i64(document, "s2mel.style_encoder.dim");
    config.s2mel.length_regulator_channels = yaml::require_i64(document, "s2mel.length_regulator.channels");
    config.s2mel.length_regulator_is_discrete = yaml::optional_bool(document, "s2mel.length_regulator.is_discrete", config.s2mel.length_regulator_is_discrete);
    config.s2mel.length_regulator_in_channels = yaml::require_i64(document, "s2mel.length_regulator.in_channels");
    config.s2mel.length_regulator_content_codebook_size = yaml::require_i64(document, "s2mel.length_regulator.content_codebook_size");
    config.s2mel.length_regulator_sampling_ratios = yaml::require_list_i64(document, "s2mel.length_regulator.sampling_ratios");
    config.s2mel.length_regulator_vector_quantize = yaml::optional_bool(document, "s2mel.length_regulator.vector_quantize", config.s2mel.length_regulator_vector_quantize);
    config.s2mel.length_regulator_n_codebooks = yaml::require_i64(document, "s2mel.length_regulator.n_codebooks");
    config.s2mel.length_regulator_quantizer_dropout = yaml::optional_f32(document, "s2mel.length_regulator.quantizer_dropout", config.s2mel.length_regulator_quantizer_dropout);
    config.s2mel.length_regulator_f0_condition = yaml::optional_bool(document, "s2mel.length_regulator.f0_condition", config.s2mel.length_regulator_f0_condition);
    config.s2mel.length_regulator_n_f0_bins = yaml::require_i64(document, "s2mel.length_regulator.n_f0_bins");
    config.s2mel.dit_hidden_dim = yaml::require_i64(document, "s2mel.DiT.hidden_dim");
    config.s2mel.dit_num_heads = yaml::require_i64(document, "s2mel.DiT.num_heads");
    config.s2mel.dit_depth = yaml::require_i64(document, "s2mel.DiT.depth");
    config.s2mel.dit_class_dropout_prob = yaml::optional_f32(document, "s2mel.DiT.class_dropout_prob", config.s2mel.dit_class_dropout_prob);
    config.s2mel.dit_block_size = yaml::require_i64(document, "s2mel.DiT.block_size");
    config.s2mel.dit_in_channels = yaml::require_i64(document, "s2mel.DiT.in_channels");
    config.s2mel.dit_style_condition = yaml::optional_bool(document, "s2mel.DiT.style_condition", config.s2mel.dit_style_condition);
    config.s2mel.dit_final_layer_type = yaml::require_string(document, "s2mel.DiT.final_layer_type");
    config.s2mel.dit_target = yaml::require_string(document, "s2mel.DiT.target");
    config.s2mel.dit_content_dim = yaml::require_i64(document, "s2mel.DiT.content_dim");
    config.s2mel.dit_content_codebook_size = yaml::require_i64(document, "s2mel.DiT.content_codebook_size");
    config.s2mel.dit_content_type = yaml::require_string(document, "s2mel.DiT.content_type");
    config.s2mel.dit_f0_condition = yaml::optional_bool(document, "s2mel.DiT.f0_condition", config.s2mel.dit_f0_condition);
    config.s2mel.dit_n_f0_bins = yaml::require_i64(document, "s2mel.DiT.n_f0_bins");
    config.s2mel.dit_content_codebooks = yaml::require_i64(document, "s2mel.DiT.content_codebooks");
    config.s2mel.dit_is_causal = yaml::optional_bool(document, "s2mel.DiT.is_causal", config.s2mel.dit_is_causal);
    config.s2mel.dit_long_skip_connection = yaml::optional_bool(document, "s2mel.DiT.long_skip_connection", config.s2mel.dit_long_skip_connection);
    config.s2mel.dit_zero_prompt_speech_token = yaml::optional_bool(document, "s2mel.DiT.zero_prompt_speech_token", config.s2mel.dit_zero_prompt_speech_token);
    config.s2mel.dit_time_as_token = yaml::optional_bool(document, "s2mel.DiT.time_as_token", config.s2mel.dit_time_as_token);
    config.s2mel.dit_style_as_token = yaml::optional_bool(document, "s2mel.DiT.style_as_token", config.s2mel.dit_style_as_token);
    config.s2mel.dit_uvit_skip_connection = yaml::optional_bool(document, "s2mel.DiT.uvit_skip_connection", config.s2mel.dit_uvit_skip_connection);
    config.s2mel.dit_add_resblock_in_transformer = yaml::optional_bool(document, "s2mel.DiT.add_resblock_in_transformer", config.s2mel.dit_add_resblock_in_transformer);
    config.s2mel.wavenet_hidden_dim = yaml::require_i64(document, "s2mel.wavenet.hidden_dim");
    config.s2mel.wavenet_num_layers = yaml::require_i64(document, "s2mel.wavenet.num_layers");
    config.s2mel.wavenet_kernel_size = yaml::require_i64(document, "s2mel.wavenet.kernel_size");
    config.s2mel.wavenet_dilation_rate = yaml::require_i64(document, "s2mel.wavenet.dilation_rate");
    config.s2mel.wavenet_dropout = yaml::optional_f32(document, "s2mel.wavenet.p_dropout", config.s2mel.wavenet_dropout);
    config.s2mel.wavenet_style_condition = yaml::optional_bool(document, "s2mel.wavenet.style_condition", config.s2mel.wavenet_style_condition);

    config.emo_num = yaml::require_list_i64(document, "emo_num");
    return config;
}

void validate_qwen_emotion_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("qwen_emotion_config");
    if (json::optional_string(root, "model_type", "") != "qwen3") {
        throw std::runtime_error("IndexTTS2 Qwen emotion model must have model_type=qwen3");
    }
    if (json::optional_i64(root, "hidden_size", 0) != 1024 ||
        json::optional_i64(root, "num_hidden_layers", 0) != 28 ||
        json::optional_i64(root, "num_attention_heads", 0) != 16) {
        throw std::runtime_error("IndexTTS2 Qwen emotion model config does not match expected 0.6B architecture");
    }
}

void validate_config(const IndexTTS2Config & config, const assets::ResourceBundle & resources) {
    engine::io::require_positive(config.dataset_sample_rate, "dataset.sample_rate");
    engine::io::require_positive(config.dataset_mel_sample_rate, "dataset.mel.sample_rate");
    engine::io::require_positive(config.dataset_mel_n_fft, "dataset.mel.n_fft");
    engine::io::require_positive(config.gpt.model_dim, "gpt.model_dim");
    engine::io::require_positive(config.gpt.layers, "gpt.layers");
    engine::io::require_divisible(config.gpt.model_dim, config.gpt.heads, "gpt.model_dim / gpt.heads");
    engine::io::require_positive(config.semantic_codec.codebook_size, "semantic_codec.codebook_size");
    engine::io::require_positive(config.s2mel.sample_rate, "s2mel.sample_rate");
    engine::io::require_positive(config.s2mel.n_mels, "s2mel.n_mels");
    engine::io::require_positive(config.s2mel.dit_hidden_dim, "s2mel.DiT.hidden_dim");
    engine::io::require_divisible(config.s2mel.dit_hidden_dim, config.s2mel.dit_num_heads, "s2mel.DiT.hidden_dim / num_heads");
    engine::io::require_nonnegative(config.s2mel.length_regulator_quantizer_dropout, "length_regulator.quantizer_dropout");
    engine::io::require_positive(config.s2mel.wavenet_dropout + 1.0F, "wavenet.p_dropout");
    if (config.emo_num.empty()) {
        throw std::runtime_error("IndexTTS2 config emo_num must not be empty");
    }
    validate_qwen_emotion_config(resources);
}

void validate_gpt_weights(const IndexTTS2Config & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "text_embedding.weight", {config.gpt.number_text_tokens + 1, config.gpt.model_dim});
    assets::require_tensor_shape(source, "mel_embedding.weight", {config.gpt.number_mel_codes, config.gpt.model_dim});
    assets::require_tensor_shape(source, "gpt.h.0.attn.c_attn.weight", {config.gpt.model_dim, config.gpt.model_dim * 3});
    assets::require_tensor_shape(source, "gpt.h.0.attn.c_proj.weight", {config.gpt.model_dim, config.gpt.model_dim});
    assets::require_tensor_shape(source, "gpt.h.0.mlp.c_fc.weight", {config.gpt.model_dim, config.gpt.model_dim * 4});
    assets::require_tensor_shape(source, "gpt.h.0.mlp.c_proj.weight", {config.gpt.model_dim * 4, config.gpt.model_dim});
    assets::require_tensor_shape(source, "conditioning_encoder.after_norm.weight", {config.gpt.condition_output_size});
    assets::require_tensor_shape(source, "emo_conditioning_encoder.after_norm.weight", {config.gpt.emo_condition_output_size});
}

void validate_s2mel_weights(const IndexTTS2Config & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "gpt_layer.0.weight", {256, config.gpt.model_dim});
    assets::require_tensor_shape(source, "gpt_layer.2.weight", {config.s2mel.length_regulator_in_channels, 128});
    assets::require_tensor_shape(source, "length_regulator.model.0.weight", {config.s2mel.length_regulator_channels, config.s2mel.length_regulator_channels, 3});
    assets::require_tensor_shape(source, "cfm.estimator.x_embedder.weight_v", {config.s2mel.dit_hidden_dim, config.s2mel.dit_in_channels});
    assets::require_tensor_shape(source, "cfm.estimator.transformer.layers.0.attention.wqkv.weight", {config.s2mel.dit_hidden_dim * 3, config.s2mel.dit_hidden_dim});
    assets::require_tensor_shape(source, "cfm.estimator.final_layer.adaLN_modulation.1.weight", {config.s2mel.dit_hidden_dim * 2, config.s2mel.dit_hidden_dim});
}

void validate_matrix_weights(
    const IndexTTS2Config & config,
    const assets::TensorSource & speaker_matrix,
    const assets::TensorSource & emotion_matrix) {
    int64_t total = 0;
    for (const int64_t count : config.emo_num) {
        engine::io::require_positive(count, "emo_num item");
        total += count;
    }
    assets::require_tensor_shape(speaker_matrix, "tensor", {total, config.s2mel.style_dim});
    assets::require_tensor_shape(emotion_matrix, "tensor", {total, config.gpt.model_dim});
}

void validate_w2v_stats(const IndexTTS2Config & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "mean", {config.semantic_codec.hidden_size});
    assets::require_tensor_shape(source, "var", {config.semantic_codec.hidden_size});
}

void validate_w2v_weights(const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "feature_projection.projection.weight", {1024, 160});
    assets::require_tensor_shape(source, "encoder.layers.0.self_attn.linear_k.weight", {1024, 1024});
    assets::require_tensor_shape(source, "encoder.layers.0.conv_module.depthwise_conv.weight", {1024, 1, 31});
}

void validate_semantic_codec_weights(const IndexTTS2Config & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "quantizer.quantizers.0.codebook.weight", {config.semantic_codec.codebook_size, config.semantic_codec.codebook_dim});
    assets::require_tensor_shape(source, "encoder.1.weight", {config.semantic_codec.hidden_size, config.semantic_codec.vocos_dim});
    assets::require_tensor_shape(source, "decoder.1.weight", {config.semantic_codec.hidden_size, config.semantic_codec.vocos_dim});
}

void validate_qwen_weights(const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "model.embed_tokens.weight", {151936, 1024});
    assets::require_tensor_shape(source, "model.layers.0.self_attn.q_proj.weight", {2048, 1024});
    assets::require_tensor_shape(source, "model.layers.0.self_attn.k_proj.weight", {1024, 1024});
    assets::require_tensor_shape(source, "model.layers.0.mlp.gate_proj.weight", {3072, 1024});
    assets::require_tensor_shape(source, "model.norm.weight", {1024});
}

void validate_weight_anchors(const IndexTTS2Assets & assets) {
    validate_gpt_weights(assets.config, *assets.gpt_weights);
    validate_s2mel_weights(assets.config, *assets.s2mel_weights);
    validate_matrix_weights(assets.config, *assets.speaker_matrix, *assets.emotion_matrix);
    validate_w2v_stats(assets.config, *assets.wav2vec2bert_stats);
    validate_w2v_weights(*assets.wav2vec2bert_weights);
    validate_semantic_codec_weights(assets.config, *assets.semantic_codec_weights);
    validate_qwen_weights(*assets.qwen_emotion_weights);
}

}  // namespace

std::shared_ptr<const IndexTTS2Assets> load_index_tts2_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<IndexTTS2Assets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("index_tts2"));
    assets->config = parse_config(assets->resources);
    validate_config(assets->config, assets->resources);

    assets->gpt_weights = assets->resources.open_tensor_source("gpt");
    assets->s2mel_weights = assets->resources.open_tensor_source("s2mel");
    assets->speaker_matrix = assets->resources.open_tensor_source("speaker_matrix");
    assets->emotion_matrix = assets->resources.open_tensor_source("emotion_matrix");
    assets->wav2vec2bert_stats = assets->resources.open_tensor_source("wav2vec2bert_stats");
    assets->wav2vec2bert_weights = assets->resources.open_tensor_source("wav2vec2bert");
    assets->semantic_codec_weights = assets->resources.open_tensor_source("semantic_codec");
    assets->campplus_weights = assets->resources.open_tensor_source("campplus");
    assets->bigvgan_weights = assets->resources.open_tensor_source("bigvgan");
    assets->qwen_emotion_weights = assets->resources.open_tensor_source("qwen_emotion");

    validate_weight_anchors(*assets);
    return assets;
}

}  // namespace engine::models::index_tts2
