#include "engine/models/index_tts2/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/io/yaml.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::models::index_tts2 {
namespace {

namespace yaml = engine::io::yaml;

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("IndexTTS2 model path does not exist: " + model_path.string());
}

std::filesystem::path require_file(const std::filesystem::path & path, const char * label) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("missing IndexTTS2 ") + label + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path require_dir(const std::filesystem::path & path, const char * label) {
    if (!engine::io::is_existing_directory(path)) {
        throw std::runtime_error(std::string("missing IndexTTS2 ") + label + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::string require_string(const yaml::FlattenedDocument & document, const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end()) {
        throw std::runtime_error("IndexTTS2 config missing key: " + key);
    }
    return it->second;
}

std::string optional_string(
    const yaml::FlattenedDocument & document,
    const std::string & key,
    const std::string & fallback) {
    const auto it = document.scalars.find(key);
    return it == document.scalars.end() ? fallback : it->second;
}

int64_t require_i64(const yaml::FlattenedDocument & document, const std::string & key) {
    return static_cast<int64_t>(yaml::require_int(document, key));
}

float optional_f32(const yaml::FlattenedDocument & document, const std::string & key, float fallback) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end()) {
        return fallback;
    }
    size_t parsed = 0;
    const float value = std::stof(it->second, &parsed);
    if (parsed != it->second.size() || !std::isfinite(value)) {
        throw std::runtime_error("IndexTTS2 config key must be a finite float: " + key);
    }
    return value;
}

bool optional_bool(const yaml::FlattenedDocument & document, const std::string & key, bool fallback) {
    const auto it = document.scalars.find(key);
    return it == document.scalars.end() ? fallback : yaml::parse_bool_scalar(it->second, key);
}

std::optional<float> optional_nullable_fmax(const yaml::FlattenedDocument & document, const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end() || it->second == "None" || it->second == "null") {
        return std::nullopt;
    }
    size_t parsed = 0;
    const float value = std::stof(it->second, &parsed);
    if (parsed != it->second.size() || !std::isfinite(value)) {
        throw std::runtime_error("IndexTTS2 config fmax must be None or a finite float");
    }
    return value;
}

std::vector<int64_t> require_i64_list(const yaml::FlattenedDocument & document, const std::string & key) {
    const auto it = document.lists.find(key);
    if (it == document.lists.end()) {
        throw std::runtime_error("IndexTTS2 config missing list key: " + key);
    }
    std::vector<int64_t> out;
    out.reserve(it->second.size());
    for (const auto & item : it->second) {
        size_t parsed = 0;
        const long value = std::stol(item, &parsed);
        if (parsed != item.size()) {
            throw std::runtime_error("IndexTTS2 config list key contains a non-integer value: " + key);
        }
        out.push_back(static_cast<int64_t>(value));
    }
    return out;
}

void require_positive(int64_t value, const char * label) {
    if (value <= 0) {
        throw std::runtime_error(std::string("IndexTTS2 config contains non-positive ") + label);
    }
}

void require_positive_float(float value, const char * label) {
    if (!(value > 0.0F) || !std::isfinite(value)) {
        throw std::runtime_error(std::string("IndexTTS2 config contains invalid ") + label);
    }
}

void require_nonnegative_float(float value, const char * label) {
    if (value < 0.0F || !std::isfinite(value)) {
        throw std::runtime_error(std::string("IndexTTS2 config contains invalid ") + label);
    }
}

void require_divisible(int64_t value, int64_t divisor, const char * label) {
    if (divisor <= 0 || value % divisor != 0) {
        throw std::runtime_error(std::string("IndexTTS2 config invalid divisibility for ") + label);
    }
}

IndexTTS2Config parse_config(const std::filesystem::path & path) {
    const auto document = yaml::parse_flattened_document(engine::io::read_text_file(path));
    IndexTTS2Config config;
    config.version = optional_string(document, "version", config.version);
    config.bpe_model = require_string(document, "dataset.bpe_model");
    config.dataset_sample_rate = static_cast<int>(require_i64(document, "dataset.sample_rate"));
    config.dataset_squeeze = optional_bool(document, "dataset.squeeze", config.dataset_squeeze);
    config.dataset_mel_sample_rate = static_cast<int>(require_i64(document, "dataset.mel.sample_rate"));
    config.dataset_mel_n_fft = require_i64(document, "dataset.mel.n_fft");
    config.dataset_mel_hop_length = require_i64(document, "dataset.mel.hop_length");
    config.dataset_mel_win_length = require_i64(document, "dataset.mel.win_length");
    config.dataset_mel_n_mels = require_i64(document, "dataset.mel.n_mels");
    config.dataset_mel_fmin = optional_f32(document, "dataset.mel.mel_fmin", config.dataset_mel_fmin);
    config.dataset_mel_normalize = optional_bool(document, "dataset.mel.normalize", config.dataset_mel_normalize);

    config.gpt.model_dim = require_i64(document, "gpt.model_dim");
    config.gpt.max_mel_tokens = require_i64(document, "gpt.max_mel_tokens");
    config.gpt.max_text_tokens = require_i64(document, "gpt.max_text_tokens");
    config.gpt.heads = require_i64(document, "gpt.heads");
    config.gpt.use_mel_codes_as_input = optional_bool(document, "gpt.use_mel_codes_as_input", config.gpt.use_mel_codes_as_input);
    config.gpt.mel_length_compression = require_i64(document, "gpt.mel_length_compression");
    config.gpt.layers = require_i64(document, "gpt.layers");
    config.gpt.number_text_tokens = require_i64(document, "gpt.number_text_tokens");
    config.gpt.number_mel_codes = require_i64(document, "gpt.number_mel_codes");
    config.gpt.start_mel_token = require_i64(document, "gpt.start_mel_token");
    config.gpt.stop_mel_token = require_i64(document, "gpt.stop_mel_token");
    config.gpt.start_text_token = require_i64(document, "gpt.start_text_token");
    config.gpt.stop_text_token = require_i64(document, "gpt.stop_text_token");
    config.gpt.train_solo_embeddings = optional_bool(document, "gpt.train_solo_embeddings", config.gpt.train_solo_embeddings);
    config.gpt.condition_type = require_string(document, "gpt.condition_type");
    config.gpt.condition_output_size = require_i64(document, "gpt.condition_module.output_size");
    config.gpt.condition_linear_units = require_i64(document, "gpt.condition_module.linear_units");
    config.gpt.condition_attention_heads = require_i64(document, "gpt.condition_module.attention_heads");
    config.gpt.condition_num_blocks = require_i64(document, "gpt.condition_module.num_blocks");
    config.gpt.condition_input_layer = require_string(document, "gpt.condition_module.input_layer");
    config.gpt.condition_perceiver_mult = require_i64(document, "gpt.condition_module.perceiver_mult");
    config.gpt.emo_condition_output_size = require_i64(document, "gpt.emo_condition_module.output_size");
    config.gpt.emo_condition_linear_units = require_i64(document, "gpt.emo_condition_module.linear_units");
    config.gpt.emo_condition_attention_heads = require_i64(document, "gpt.emo_condition_module.attention_heads");
    config.gpt.emo_condition_num_blocks = require_i64(document, "gpt.emo_condition_module.num_blocks");
    config.gpt.emo_condition_input_layer = require_string(document, "gpt.emo_condition_module.input_layer");
    config.gpt.emo_condition_perceiver_mult = require_i64(document, "gpt.emo_condition_module.perceiver_mult");

    config.semantic_codec.codebook_size = require_i64(document, "semantic_codec.codebook_size");
    config.semantic_codec.hidden_size = require_i64(document, "semantic_codec.hidden_size");
    config.semantic_codec.codebook_dim = require_i64(document, "semantic_codec.codebook_dim");
    config.semantic_codec.vocos_dim = require_i64(document, "semantic_codec.vocos_dim");
    config.semantic_codec.vocos_intermediate_dim = require_i64(document, "semantic_codec.vocos_intermediate_dim");
    config.semantic_codec.vocos_num_layers = require_i64(document, "semantic_codec.vocos_num_layers");

    config.s2mel.sample_rate = static_cast<int>(require_i64(document, "s2mel.preprocess_params.sr"));
    config.s2mel.n_fft = require_i64(document, "s2mel.preprocess_params.spect_params.n_fft");
    config.s2mel.win_length = require_i64(document, "s2mel.preprocess_params.spect_params.win_length");
    config.s2mel.hop_length = require_i64(document, "s2mel.preprocess_params.spect_params.hop_length");
    config.s2mel.n_mels = require_i64(document, "s2mel.preprocess_params.spect_params.n_mels");
    config.s2mel.fmin = optional_f32(document, "s2mel.preprocess_params.spect_params.fmin", config.s2mel.fmin);
    config.s2mel.fmax = optional_nullable_fmax(document, "s2mel.preprocess_params.spect_params.fmax");
    config.s2mel.dit_type = require_string(document, "s2mel.dit_type");
    config.s2mel.reg_loss_type = require_string(document, "s2mel.reg_loss_type");
    config.s2mel.style_dim = require_i64(document, "s2mel.style_encoder.dim");
    config.s2mel.length_regulator_channels = require_i64(document, "s2mel.length_regulator.channels");
    config.s2mel.length_regulator_is_discrete = optional_bool(document, "s2mel.length_regulator.is_discrete", config.s2mel.length_regulator_is_discrete);
    config.s2mel.length_regulator_in_channels = require_i64(document, "s2mel.length_regulator.in_channels");
    config.s2mel.length_regulator_content_codebook_size = require_i64(document, "s2mel.length_regulator.content_codebook_size");
    config.s2mel.length_regulator_sampling_ratios = require_i64_list(document, "s2mel.length_regulator.sampling_ratios");
    config.s2mel.length_regulator_vector_quantize = optional_bool(document, "s2mel.length_regulator.vector_quantize", config.s2mel.length_regulator_vector_quantize);
    config.s2mel.length_regulator_n_codebooks = require_i64(document, "s2mel.length_regulator.n_codebooks");
    config.s2mel.length_regulator_quantizer_dropout = optional_f32(document, "s2mel.length_regulator.quantizer_dropout", config.s2mel.length_regulator_quantizer_dropout);
    config.s2mel.length_regulator_f0_condition = optional_bool(document, "s2mel.length_regulator.f0_condition", config.s2mel.length_regulator_f0_condition);
    config.s2mel.length_regulator_n_f0_bins = require_i64(document, "s2mel.length_regulator.n_f0_bins");
    config.s2mel.dit_hidden_dim = require_i64(document, "s2mel.DiT.hidden_dim");
    config.s2mel.dit_num_heads = require_i64(document, "s2mel.DiT.num_heads");
    config.s2mel.dit_depth = require_i64(document, "s2mel.DiT.depth");
    config.s2mel.dit_class_dropout_prob = optional_f32(document, "s2mel.DiT.class_dropout_prob", config.s2mel.dit_class_dropout_prob);
    config.s2mel.dit_block_size = require_i64(document, "s2mel.DiT.block_size");
    config.s2mel.dit_in_channels = require_i64(document, "s2mel.DiT.in_channels");
    config.s2mel.dit_style_condition = optional_bool(document, "s2mel.DiT.style_condition", config.s2mel.dit_style_condition);
    config.s2mel.dit_final_layer_type = require_string(document, "s2mel.DiT.final_layer_type");
    config.s2mel.dit_target = require_string(document, "s2mel.DiT.target");
    config.s2mel.dit_content_dim = require_i64(document, "s2mel.DiT.content_dim");
    config.s2mel.dit_content_codebook_size = require_i64(document, "s2mel.DiT.content_codebook_size");
    config.s2mel.dit_content_type = require_string(document, "s2mel.DiT.content_type");
    config.s2mel.dit_f0_condition = optional_bool(document, "s2mel.DiT.f0_condition", config.s2mel.dit_f0_condition);
    config.s2mel.dit_n_f0_bins = require_i64(document, "s2mel.DiT.n_f0_bins");
    config.s2mel.dit_content_codebooks = require_i64(document, "s2mel.DiT.content_codebooks");
    config.s2mel.dit_is_causal = optional_bool(document, "s2mel.DiT.is_causal", config.s2mel.dit_is_causal);
    config.s2mel.dit_long_skip_connection = optional_bool(document, "s2mel.DiT.long_skip_connection", config.s2mel.dit_long_skip_connection);
    config.s2mel.dit_zero_prompt_speech_token = optional_bool(document, "s2mel.DiT.zero_prompt_speech_token", config.s2mel.dit_zero_prompt_speech_token);
    config.s2mel.dit_time_as_token = optional_bool(document, "s2mel.DiT.time_as_token", config.s2mel.dit_time_as_token);
    config.s2mel.dit_style_as_token = optional_bool(document, "s2mel.DiT.style_as_token", config.s2mel.dit_style_as_token);
    config.s2mel.dit_uvit_skip_connection = optional_bool(document, "s2mel.DiT.uvit_skip_connection", config.s2mel.dit_uvit_skip_connection);
    config.s2mel.dit_add_resblock_in_transformer = optional_bool(document, "s2mel.DiT.add_resblock_in_transformer", config.s2mel.dit_add_resblock_in_transformer);
    config.s2mel.wavenet_hidden_dim = require_i64(document, "s2mel.wavenet.hidden_dim");
    config.s2mel.wavenet_num_layers = require_i64(document, "s2mel.wavenet.num_layers");
    config.s2mel.wavenet_kernel_size = require_i64(document, "s2mel.wavenet.kernel_size");
    config.s2mel.wavenet_dilation_rate = require_i64(document, "s2mel.wavenet.dilation_rate");
    config.s2mel.wavenet_dropout = optional_f32(document, "s2mel.wavenet.p_dropout", config.s2mel.wavenet_dropout);
    config.s2mel.wavenet_style_condition = optional_bool(document, "s2mel.wavenet.style_condition", config.s2mel.wavenet_style_condition);

    config.gpt_checkpoint = require_string(document, "gpt_checkpoint");
    config.w2v_stat = require_string(document, "w2v_stat");
    config.s2mel_checkpoint = require_string(document, "s2mel_checkpoint");
    config.emo_matrix = require_string(document, "emo_matrix");
    config.spk_matrix = require_string(document, "spk_matrix");
    config.emo_num = require_i64_list(document, "emo_num");
    config.qwen_emo_path = require_string(document, "qwen_emo_path");
    config.vocoder_type = require_string(document, "vocoder.type");
    config.vocoder_name = require_string(document, "vocoder.name");

    require_positive(config.dataset_sample_rate, "dataset.sample_rate");
    require_positive(config.dataset_mel_sample_rate, "dataset.mel.sample_rate");
    require_positive(config.dataset_mel_n_fft, "dataset.mel.n_fft");
    require_positive(config.gpt.model_dim, "gpt.model_dim");
    require_positive(config.gpt.layers, "gpt.layers");
    require_divisible(config.gpt.model_dim, config.gpt.heads, "gpt.model_dim / gpt.heads");
    require_positive(config.semantic_codec.codebook_size, "semantic_codec.codebook_size");
    require_positive(config.s2mel.sample_rate, "s2mel.sample_rate");
    require_positive(config.s2mel.n_mels, "s2mel.n_mels");
    require_positive(config.s2mel.dit_hidden_dim, "s2mel.DiT.hidden_dim");
    require_divisible(config.s2mel.dit_hidden_dim, config.s2mel.dit_num_heads, "s2mel.DiT.hidden_dim / num_heads");
    require_nonnegative_float(config.s2mel.length_regulator_quantizer_dropout, "length_regulator.quantizer_dropout");
    require_positive_float(config.s2mel.wavenet_dropout + 1.0F, "wavenet.p_dropout");
    if (config.emo_num.empty()) {
        throw std::runtime_error("IndexTTS2 config emo_num must not be empty");
    }
    return config;
}

std::filesystem::path safetensor_peer(
    const std::filesystem::path & root,
    const std::string & configured_path,
    const char * label) {
    std::filesystem::path relative(configured_path);
    relative.replace_extension(".safetensors");
    return require_file(root / relative, label);
}

void require_tensor_shape(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape != expected_shape) {
        throw std::runtime_error("IndexTTS2 tensor shape mismatch for " + name);
    }
}

void validate_gpt_weights(const IndexTTS2Config & config, const assets::TensorSource & source) {
    require_tensor_shape(source, "text_embedding.weight", {config.gpt.number_text_tokens + 1, config.gpt.model_dim});
    require_tensor_shape(source, "mel_embedding.weight", {config.gpt.number_mel_codes, config.gpt.model_dim});
    require_tensor_shape(source, "gpt.h.0.attn.c_attn.weight", {config.gpt.model_dim, config.gpt.model_dim * 3});
    require_tensor_shape(source, "gpt.h.0.attn.c_proj.weight", {config.gpt.model_dim, config.gpt.model_dim});
    require_tensor_shape(source, "gpt.h.0.mlp.c_fc.weight", {config.gpt.model_dim, config.gpt.model_dim * 4});
    require_tensor_shape(source, "gpt.h.0.mlp.c_proj.weight", {config.gpt.model_dim * 4, config.gpt.model_dim});
    require_tensor_shape(source, "conditioning_encoder.after_norm.weight", {config.gpt.condition_output_size});
    require_tensor_shape(source, "emo_conditioning_encoder.after_norm.weight", {config.gpt.emo_condition_output_size});
}

void validate_s2mel_weights(const IndexTTS2Config & config, const assets::TensorSource & source) {
    require_tensor_shape(source, "gpt_layer.0.weight", {256, config.gpt.model_dim});
    require_tensor_shape(source, "gpt_layer.2.weight", {config.s2mel.length_regulator_in_channels, 128});
    require_tensor_shape(source, "length_regulator.model.0.weight", {config.s2mel.length_regulator_channels, config.s2mel.length_regulator_channels, 3});
    require_tensor_shape(source, "cfm.estimator.x_embedder.weight_v", {config.s2mel.dit_hidden_dim, config.s2mel.dit_in_channels});
    require_tensor_shape(source, "cfm.estimator.transformer.layers.0.attention.wqkv.weight", {config.s2mel.dit_hidden_dim * 3, config.s2mel.dit_hidden_dim});
    require_tensor_shape(source, "cfm.estimator.final_layer.adaLN_modulation.1.weight", {config.s2mel.dit_hidden_dim * 2, config.s2mel.dit_hidden_dim});
}

void validate_matrix_weights(
    const IndexTTS2Config & config,
    const assets::TensorSource & speaker_matrix,
    const assets::TensorSource & emotion_matrix) {
    int64_t total = 0;
    for (const int64_t count : config.emo_num) {
        require_positive(count, "emo_num item");
        total += count;
    }
    require_tensor_shape(speaker_matrix, "tensor", {total, config.s2mel.style_dim});
    require_tensor_shape(emotion_matrix, "tensor", {total, config.gpt.model_dim});
}

void validate_w2v_stats(const IndexTTS2Config & config, const assets::TensorSource & source) {
    require_tensor_shape(source, "mean", {config.semantic_codec.hidden_size});
    require_tensor_shape(source, "var", {config.semantic_codec.hidden_size});
}

void validate_w2v_weights(const assets::TensorSource & source) {
    require_tensor_shape(source, "feature_projection.projection.weight", {1024, 160});
    require_tensor_shape(source, "encoder.layers.0.self_attn.linear_k.weight", {1024, 1024});
    require_tensor_shape(source, "encoder.layers.0.conv_module.depthwise_conv.weight", {1024, 1, 31});
}

void validate_semantic_codec_weights(const IndexTTS2Config & config, const assets::TensorSource & source) {
    require_tensor_shape(source, "quantizer.quantizers.0.codebook.weight", {config.semantic_codec.codebook_size, config.semantic_codec.codebook_dim});
    require_tensor_shape(source, "encoder.1.weight", {config.semantic_codec.hidden_size, config.semantic_codec.vocos_dim});
    require_tensor_shape(source, "decoder.1.weight", {config.semantic_codec.hidden_size, config.semantic_codec.vocos_dim});
}

void validate_qwen_config(const std::filesystem::path & config_path) {
    const auto root = engine::io::json::parse(engine::io::read_text_file(config_path));
    if (engine::io::json::optional_string(root, "model_type", "") != "qwen3") {
        throw std::runtime_error("IndexTTS2 Qwen emotion model must have model_type=qwen3");
    }
    if (engine::io::json::optional_i64(root, "hidden_size", 0) != 1024 ||
        engine::io::json::optional_i64(root, "num_hidden_layers", 0) != 28 ||
        engine::io::json::optional_i64(root, "num_attention_heads", 0) != 16) {
        throw std::runtime_error("IndexTTS2 Qwen emotion model config does not match expected 0.6B architecture");
    }
}

void validate_qwen_weights(const assets::TensorSource & source) {
    require_tensor_shape(source, "model.embed_tokens.weight", {151936, 1024});
    require_tensor_shape(source, "model.layers.0.self_attn.q_proj.weight", {2048, 1024});
    require_tensor_shape(source, "model.layers.0.self_attn.k_proj.weight", {1024, 1024});
    require_tensor_shape(source, "model.layers.0.mlp.gate_proj.weight", {3072, 1024});
    require_tensor_shape(source, "model.norm.weight", {1024});
}

}  // namespace

IndexTTS2AssetPaths resolve_index_tts2_assets(const std::filesystem::path & model_path) {
    IndexTTS2AssetPaths paths;
    paths.model_root = resolve_model_root(model_path);
    paths.config_yaml_path = require_file(paths.model_root / "config.yaml", "config.yaml");
    const auto config = parse_config(paths.config_yaml_path);

    paths.bpe_model_path = require_file(paths.model_root / config.bpe_model, "BPE model");
    paths.gpt_weights_path = safetensor_peer(paths.model_root, config.gpt_checkpoint, "GPT safetensors");
    paths.s2mel_weights_path = safetensor_peer(paths.model_root, config.s2mel_checkpoint, "S2Mel safetensors");
    paths.speaker_matrix_path = safetensor_peer(paths.model_root, config.spk_matrix, "speaker matrix safetensors");
    paths.emotion_matrix_path = safetensor_peer(paths.model_root, config.emo_matrix, "emotion matrix safetensors");
    paths.wav2vec2bert_stats_path = safetensor_peer(paths.model_root, config.w2v_stat, "Wav2Vec2BERT stats safetensors");

    paths.wav2vec2bert_root = require_dir(paths.model_root / "w2v-bert-2.0", "Wav2Vec2BERT directory");
    paths.wav2vec2bert_config_path = require_file(paths.wav2vec2bert_root / "config.json", "Wav2Vec2BERT config");
    paths.wav2vec2bert_preprocessor_config_path = require_file(paths.wav2vec2bert_root / "preprocessor_config.json", "Wav2Vec2BERT preprocessor config");
    paths.wav2vec2bert_weights_path = require_file(paths.wav2vec2bert_root / "model.safetensors", "Wav2Vec2BERT safetensors");
    paths.semantic_codec_weights_path = require_file(paths.model_root / "semantic_codec_model.safetensors", "semantic codec safetensors");
    paths.campplus_weights_path = require_file(paths.model_root / "campplus.safetensors", "CAMPPlus safetensors");
    paths.bigvgan_root = require_dir(paths.model_root / "bigvgan", "BigVGAN directory");
    paths.bigvgan_config_path = require_file(paths.bigvgan_root / "config.json", "BigVGAN config");
    paths.bigvgan_weights_path = require_file(paths.bigvgan_root / "model.safetensors", "BigVGAN safetensors");

    paths.qwen_emotion_root = require_dir(paths.model_root / config.qwen_emo_path, "Qwen emotion directory");
    paths.qwen_emotion_config_path = require_file(paths.qwen_emotion_root / "config.json", "Qwen emotion config");
    paths.qwen_emotion_generation_config_path = require_file(paths.qwen_emotion_root / "generation_config.json", "Qwen emotion generation config");
    paths.qwen_emotion_tokenizer_json_path = require_file(paths.qwen_emotion_root / "tokenizer.json", "Qwen emotion tokenizer");
    paths.qwen_emotion_tokenizer_config_path = require_file(paths.qwen_emotion_root / "tokenizer_config.json", "Qwen emotion tokenizer config");
    paths.qwen_emotion_vocab_path = require_file(paths.qwen_emotion_root / "vocab.json", "Qwen emotion vocab");
    paths.qwen_emotion_merges_path = require_file(paths.qwen_emotion_root / "merges.txt", "Qwen emotion merges");
    paths.qwen_emotion_weights_path = require_file(paths.qwen_emotion_root / "model.safetensors", "Qwen emotion safetensors");
    return paths;
}

std::shared_ptr<const IndexTTS2Assets> load_index_tts2_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<IndexTTS2Assets>();
    assets->paths = resolve_index_tts2_assets(model_path);
    assets->config = parse_config(assets->paths.config_yaml_path);

    validate_qwen_config(assets->paths.qwen_emotion_config_path);
    assets->gpt_weights = assets::open_tensor_source(assets->paths.gpt_weights_path);
    assets->s2mel_weights = assets::open_tensor_source(assets->paths.s2mel_weights_path);
    assets->speaker_matrix = assets::open_tensor_source(assets->paths.speaker_matrix_path);
    assets->emotion_matrix = assets::open_tensor_source(assets->paths.emotion_matrix_path);
    assets->wav2vec2bert_stats = assets::open_tensor_source(assets->paths.wav2vec2bert_stats_path);
    assets->wav2vec2bert_weights = assets::open_tensor_source(assets->paths.wav2vec2bert_weights_path);
    assets->semantic_codec_weights = assets::open_tensor_source(assets->paths.semantic_codec_weights_path);
    assets->campplus_weights = assets::open_tensor_source(assets->paths.campplus_weights_path);
    assets->bigvgan_weights = assets::open_tensor_source(assets->paths.bigvgan_weights_path);
    assets->qwen_emotion_weights = assets::open_tensor_source(assets->paths.qwen_emotion_weights_path);

    validate_gpt_weights(assets->config, *assets->gpt_weights);
    validate_s2mel_weights(assets->config, *assets->s2mel_weights);
    validate_matrix_weights(assets->config, *assets->speaker_matrix, *assets->emotion_matrix);
    validate_w2v_stats(assets->config, *assets->wav2vec2bert_stats);
    validate_w2v_weights(*assets->wav2vec2bert_weights);
    validate_semantic_codec_weights(assets->config, *assets->semantic_codec_weights);
    validate_qwen_weights(*assets->qwen_emotion_weights);
    return assets;
}

}  // namespace engine::models::index_tts2
