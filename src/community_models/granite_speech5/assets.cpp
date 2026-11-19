#include "engine/community_models/granite_speech5/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::community_models::granite_speech5 {
namespace json = engine::io::json;
namespace {

void require_shape(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape != expected) {
        std::string message = "granite_speech5 tensor shape mismatch for " + name + ": expected [";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i != 0) {
                message += "x";
            }
            message += std::to_string(expected[i]);
        }
        message += "], got [";
        for (size_t i = 0; i < metadata.shape.size(); ++i) {
            if (i != 0) {
                message += "x";
            }
            message += std::to_string(metadata.shape[i]);
        }
        message += "] in " + source.source_path().string();
        throw std::runtime_error(message);
    }
}

void require_vector(const assets::TensorSource & source, const std::string & name, int64_t size) {
    require_shape(source, name, {size});
}

void require_present(const assets::TensorSource & source, const std::string & name) {
    if (!source.has_tensor(name)) {
        throw std::runtime_error("granite_speech5 missing required tensor: " + name);
    }
}

void check_encoder_layer(const assets::TensorSource & source, int layer) {
    const std::string prefix = "encoder.layers." + std::to_string(layer) + ".";
    require_vector(source, prefix + "attn.pre_norm.weight", 1024);
    require_vector(source, prefix + "attn.pre_norm.bias", 1024);
    require_shape(source, prefix + "attn.rel_pos_emb.weight", {1025, 128});
    require_shape(source, prefix + "attn.to_kv.weight", {2048, 1024});
    require_vector(source, prefix + "attn.to_out.bias", 1024);
    require_shape(source, prefix + "attn.to_out.weight", {1024, 1024});
    require_shape(source, prefix + "attn.to_q.weight", {1024, 1024});
    require_vector(source, prefix + "conv.batch_norm.bias", 2048);
    require_vector(source, prefix + "conv.batch_norm.running_mean", 2048);
    require_vector(source, prefix + "conv.batch_norm.running_var", 2048);
    require_vector(source, prefix + "conv.batch_norm.weight", 2048);
    require_present(source, prefix + "conv.batch_norm.num_batches_tracked");
    require_shape(source, prefix + "conv.depth_conv.conv.weight", {2048, 1, 15});
    require_vector(source, prefix + "conv.down_conv.bias", 1024);
    require_shape(source, prefix + "conv.down_conv.weight", {1024, 2048, 1});
    require_vector(source, prefix + "conv.norm.bias", 1024);
    require_vector(source, prefix + "conv.norm.weight", 1024);
    require_vector(source, prefix + "conv.up_conv.bias", 4096);
    require_shape(source, prefix + "conv.up_conv.weight", {4096, 1024, 1});
    for (const char * block : {"ff1", "ff2"}) {
        const std::string block_prefix = prefix + block + ".";
        require_vector(source, block_prefix + "down_proj.bias", 1024);
        require_shape(source, block_prefix + "down_proj.weight", {1024, 4096});
        require_vector(source, block_prefix + "pre_norm.bias", 1024);
        require_vector(source, block_prefix + "pre_norm.weight", 1024);
        require_vector(source, block_prefix + "up_proj.bias", 4096);
        require_shape(source, block_prefix + "up_proj.weight", {4096, 1024});
    }
    require_vector(source, prefix + "post_norm.bias", 1024);
    require_vector(source, prefix + "post_norm.weight", 1024);
}

void check_qformer_layer(const assets::TensorSource & source, int layer) {
    const std::string prefix = "projector.qformer.encoder.layer." + std::to_string(layer) + ".";
    for (const char * block : {"attention", "crossattention"}) {
        const std::string block_prefix = prefix + block + ".attention.";
        for (const char * proj : {"query", "key", "value"}) {
            require_vector(source, block_prefix + proj + ".bias", 1024);
            require_shape(source, block_prefix + proj + ".weight", {1024, 1024});
        }
        const std::string out_prefix = prefix + block + ".output.";
        require_vector(source, out_prefix + "LayerNorm.bias", 1024);
        require_vector(source, out_prefix + "LayerNorm.weight", 1024);
        require_vector(source, out_prefix + "dense.bias", 1024);
        require_shape(source, out_prefix + "dense.weight", {1024, 1024});
    }
    require_vector(source, prefix + "intermediate_query.dense.bias", 4096);
    require_shape(source, prefix + "intermediate_query.dense.weight", {4096, 1024});
    require_vector(source, prefix + "output_query.LayerNorm.bias", 1024);
    require_vector(source, prefix + "output_query.LayerNorm.weight", 1024);
    require_vector(source, prefix + "output_query.dense.bias", 1024);
    require_shape(source, prefix + "output_query.dense.weight", {1024, 4096});
}

void check_text_layer(const assets::TensorSource & source, int layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
    require_vector(source, prefix + "input_layernorm.weight", 2048);
    require_vector(source, prefix + "post_attention_layernorm.weight", 2048);
    require_shape(source, prefix + "mlp.down_proj.weight", {2048, 4096});
    require_shape(source, prefix + "mlp.gate_proj.weight", {4096, 2048});
    require_shape(source, prefix + "mlp.up_proj.weight", {4096, 2048});
    require_shape(source, prefix + "self_attn.k_proj.weight", {512, 2048});
    require_shape(source, prefix + "self_attn.o_proj.weight", {2048, 2048});
    require_shape(source, prefix + "self_attn.q_proj.weight", {2048, 2048});
    require_shape(source, prefix + "self_attn.v_proj.weight", {512, 2048});
}

GraniteSpeech5EncoderConfig parse_encoder_config(const json::Value & value) {
    GraniteSpeech5EncoderConfig config;
    config.input_dim = json::require_i64(value, "input_dim");
    config.hidden_dim = json::require_i64(value, "hidden_dim");
    config.num_layers = json::require_i64(value, "num_layers");
    config.num_heads = json::require_i64(value, "num_heads");
    config.dim_head = json::require_i64(value, "dim_head");
    config.context_size = json::require_i64(value, "context_size");
    config.output_dim = json::require_i64(value, "output_dim");
    config.conv_expansion_factor = json::optional_i64(value, "conv_expansion_factor", 2);
    config.conv_kernel_size = json::optional_i64(value, "conv_kernel_size", 15);
    config.max_pos_emb = json::optional_i64(value, "max_pos_emb", 512);
    config.feedforward_mult = json::optional_i64(value, "feedforward_mult", 4);
    const auto model_type = json::optional_string(value, "model_type", "granite_speech_encoder");
    if (model_type != "granite_speech_encoder") {
        throw std::runtime_error(
            "granite_speech5 requires encoder model_type 'granite_speech_encoder', got: " + model_type);
    }
    return config;
}

GraniteSpeech5ProjectorConfig parse_projector_config(
    const json::Value & value, const json::Value & root) {
    GraniteSpeech5ProjectorConfig config;
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.intermediate_size = json::optional_i64(value, "intermediate_size", 4096);
    config.window_size = json::optional_i64(root, "window_size", 15);
    config.downsample_rate = json::optional_i64(root, "downsample_rate", 5);
    const auto model_type = json::optional_string(value, "model_type", "blip_2_qformer");
    if (model_type != "blip_2_qformer") {
        throw std::runtime_error(
            "granite_speech5 requires projector model_type 'blip_2_qformer', got: " + model_type);
    }
    return config;
}

GraniteSpeech5TextConfig parse_text_config(const json::Value & value) {
    GraniteSpeech5TextConfig config;
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.bos_token_id = json::require_i64(value, "bos_token_id");
    config.eos_token_id = json::require_i64(value, "eos_token_id");
    config.pad_token_id = json::require_i64(value, "pad_token_id");
    config.audio_token_id = json::optional_i64(value, "audio_token_id", config.audio_token_id);
    config.embedding_multiplier = json::optional_i64(value, "embedding_multiplier", 12);
    config.attention_multiplier = json::optional_f32(value, "attention_multiplier", 0.0078125F);
    config.logits_scaling = json::optional_i64(value, "logits_scaling", 8);
    config.residual_multiplier = json::optional_f32(value, "residual_multiplier", 0.22F);
    config.rope_theta = json::optional_i64(value, "rope_theta", 10000);
    config.max_position_embeddings = json::optional_i64(value, "max_position_embeddings", 16384);
    if (const auto * rope = value.find("rope_scaling"); rope != nullptr && rope->is_object()) {
        config.rope_factor = json::optional_f32(*rope, "factor", config.rope_factor);
        config.rope_original_max_position_embeddings = json::optional_i64(
            *rope, "original_max_position_embeddings", config.rope_original_max_position_embeddings);
        config.rope_type = json::optional_string(*rope, "rope_type", config.rope_type);
    }
    const auto model_type = json::optional_string(value, "model_type", "granite");
    if (model_type != "granite") {
        throw std::runtime_error(
            "granite_speech5 requires text model_type 'granite', got: " + model_type);
    }
    return config;
}

GraniteSpeech5FrontendConfig parse_frontend_config(const assets::ResourceBundle & resources) {
    GraniteSpeech5FrontendConfig config;
    if (!resources.has_file("preprocessor_config")) {
        return config;
    }
    const auto root = resources.parse_json("preprocessor_config");
    const auto * mel = root.find("melspec_kwargs");
    const auto & frontend = (mel != nullptr && mel->is_object()) ? *mel : root;
    config.sample_rate = json::optional_i64(frontend, "sampling_rate", config.sample_rate);
    config.n_fft = json::optional_i64(frontend, "n_fft", config.n_fft);
    config.win_length = json::optional_i64(frontend, "win_length", config.win_length);
    config.hop_length = json::optional_i64(frontend, "hop_length", config.hop_length);
    config.n_mels = json::optional_i64(frontend, "n_mels", config.n_mels);
    return config;
}

GraniteSpeech5Config parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");

    GraniteSpeech5Config config;
    config.model_type = json::require_string(root, "model_type");
    if (config.model_type != kGraniteSpeech5ModelType) {
        throw std::runtime_error(
            "granite_speech5 requires model_type 'granite_speech', got: " + config.model_type);
    }
    config.has_lora_adapter = json::optional_bool(root, "has_lora_adapter", false);
    if (config.has_lora_adapter) {
        throw std::runtime_error("granite_speech5 snapshot carries an external LoRA adapter");
    }

    config.encoder = parse_encoder_config(root.require("encoder_config"));
    config.projector = parse_projector_config(root.require("projector_config"), root);
    config.text = parse_text_config(root.require("text_config"));
    if (const auto * audio_token = root.find("audio_token_index");
        audio_token != nullptr && audio_token->is_number()) {
        config.text.audio_token_id = audio_token->as_i64();
    }
    config.frontend = parse_frontend_config(resources);
    return config;
}

void validate_config(const GraniteSpeech5Config & config) {
    const auto & encoder = config.encoder;
    if (encoder.input_dim != 160 || encoder.hidden_dim != 1024 || encoder.num_layers != 16 ||
        encoder.num_heads != 8 || encoder.dim_head != 128 || encoder.output_dim != 348) {
        throw std::runtime_error("granite_speech5 encoder dimensions do not match the pinned snapshot");
    }
    if (encoder.context_size <= 0 || encoder.max_pos_emb <= 0 ||
        encoder.conv_expansion_factor <= 0 || encoder.conv_kernel_size <= 0 ||
        encoder.feedforward_mult <= 0) {
        throw std::runtime_error("granite_speech5 invalid encoder bounds");
    }
    const auto & projector = config.projector;
    if (projector.hidden_size != 1024 || projector.num_hidden_layers != 2 ||
        projector.num_attention_heads != 16 || projector.intermediate_size != 4096) {
        throw std::runtime_error("granite_speech5 projector dimensions do not match the pinned snapshot");
    }
    if (projector.window_size != 15 || projector.downsample_rate != 5 ||
        projector.hidden_size % projector.num_attention_heads != 0) {
        throw std::runtime_error("granite_speech5 invalid projector window bounds");
    }
    const auto & text = config.text;
    if (text.num_hidden_layers != 40 || text.hidden_size != 2048 ||
        text.num_attention_heads != 16 || text.num_key_value_heads != 4 ||
        text.intermediate_size != 4096 || text.vocab_size != 100353) {
        throw std::runtime_error("granite_speech5 text dimensions do not match the pinned snapshot");
    }
    if (text.hidden_size % text.num_attention_heads != 0) {
        throw std::runtime_error("granite_speech5 invalid text attention layout");
    }
    for (int64_t id : {text.bos_token_id, text.eos_token_id, text.pad_token_id, text.audio_token_id}) {
        if (id < 0 || id >= text.vocab_size) {
            throw std::runtime_error("granite_speech5 vocabulary ID out of range");
        }
    }
    if (text.max_position_embeddings <= 0 || text.rope_theta <= 0 || text.rope_factor <= 0.0F ||
        text.rope_original_max_position_embeddings <= 0) {
        throw std::runtime_error("granite_speech5 invalid context/window bounds");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.n_fft != 512 ||
        config.frontend.win_length != 400 || config.frontend.hop_length != 160 ||
        config.frontend.n_mels != 80) {
        throw std::runtime_error("granite_speech5 frontend constants do not match the pinned snapshot");
    }
}

assets::ResourceBundle build_resource_bundle(const std::filesystem::path & model_path) {
    if (!std::filesystem::is_directory(model_path)) {
        throw std::runtime_error(
            "granite_speech5 expects a native checkpoint directory, got: " + model_path.string());
    }
    // Manual bundle for native checkpoint directories only (Task D path,
    // unchanged): the logical tensor entry stays `weights` (native root
    // `model:.` here, `model:model.safetensors` file), the same name the
    // spec-backed GGUF `$gguf` source binds for packages.
    assets::ResourceBundle resources(model_path);
    resources.add_model_files({
        {"config", "config.json", true},
        {"preprocessor_config", "preprocessor_config.json", true},
        {"processor_config", "processor_config.json", true},
        {"tokenizer_json", "tokenizer.json", true},
        {"tokenizer_config", "tokenizer_config.json", true},
        {"generation_config", "generation_config.json", true},
        {"special_tokens_map", "special_tokens_map.json", true},
        {"chat_template", "chat_template.jinja", true},
    });
    resources.add_tensor_source("weights", model_path / "model.safetensors");
    return resources;
}

bool is_gguf_file_path(const std::filesystem::path & path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".gguf";
}

// A GGUF package is a standalone .gguf file, or a directory that holds the
// single package GGUF instead of a native checkpoint. Native checkpoint
// directories (config.json triple, model.safetensors) never take this path,
// even if a stray GGUF sits beside them.
bool is_gguf_package_path(const std::filesystem::path & model_path) {
    if (is_gguf_file_path(model_path)) {
        return true;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(model_path, ec)) {
        return false;
    }
    if (granite_speech5_can_load(model_path)) {
        return false;
    }
    // A foreign native checkpoint directory (its own weights on disk) is not
    // an installed package, even with a stray GGUF beside it.
    if (granite_speech5_directory_has_native_weights(model_path)) {
        return false;
    }
    return engine::assets::find_directory_gguf(model_path).has_value();
}

assets::ResourceBundle load_granite_speech5_resources(const std::filesystem::path & model_path) {
    if (is_gguf_package_path(model_path)) {
        // Spec-backed bundle: validates the embedded model spec family and
        // materializes the sidecars; tensor entry keeps the logical `weights`
        // name so the inventory check below is unchanged.
        return engine::model_spec::load_resource_bundle_for_family(model_path, kGraniteSpeech5Family);
    }
    return build_resource_bundle(model_path);
}

}  // namespace

bool granite_speech5_directory_has_native_weights(const std::filesystem::path & dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return false;
    }
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            return false;
        }
        if (!entry.is_regular_file(ec) || ec) {
            continue;
        }
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        if (extension == ".safetensors" || extension == ".bin") {
            return true;
        }
    }
    return false;
}

bool granite_speech5_can_load(const std::filesystem::path & model_path) {
    try {
        if (!std::filesystem::is_directory(model_path)) {
            return false;
        }
        const auto config_path = model_path / "config.json";
        std::error_code ec;
        if (!std::filesystem::is_regular_file(config_path, ec)) {
            return false;
        }
        const auto root = json::parse_file(config_path);
        if (json::optional_string(root, "model_type", "") != kGraniteSpeech5ModelType) {
            return false;
        }
        if (json::optional_bool(root, "has_lora_adapter", false)) {
            return false;
        }
        const auto * encoder = root.find("encoder_config");
        const auto * projector = root.find("projector_config");
        const auto * text = root.find("text_config");
        if (encoder == nullptr || !encoder->is_object() || projector == nullptr ||
            !projector->is_object() || text == nullptr || !text->is_object()) {
            return false;
        }
        // Conditional generation triple: TurboCTC snapshots (granite5asr /
        // granite_speech5_ctc, including ones stamped `granite_speech5`) carry
        // no Q-Former projector or Granite text decoder.
        return json::optional_string(*encoder, "model_type", "") == "granite_speech_encoder" &&
            json::optional_string(*projector, "model_type", "") == "blip_2_qformer" &&
            json::optional_string(*text, "model_type", "") == "granite";
    } catch (...) {
        return false;
    }
}

void check_granite_speech5_encoder_weight_shapes(const assets::TensorSource & source) {
    require_shape(source, "encoder.input_linear.weight", {1024, 160});
    require_vector(source, "encoder.input_linear.bias", 1024);
    require_shape(source, "encoder.out.weight", {348, 1024});
    require_vector(source, "encoder.out.bias", 348);
    require_shape(source, "encoder.out_mid.weight", {1024, 348});
    require_vector(source, "encoder.out_mid.bias", 1024);
    // Validate the first block before cloning the checks across all 16.
    check_encoder_layer(source, 0);
    for (int layer = 1; layer < 16; ++layer) {
        check_encoder_layer(source, layer);
    }
    if (source.has_tensor("encoder.layers.16.attn.to_q.weight")) {
        throw std::runtime_error("granite_speech5 encoder has more than 16 layers");
    }
}

void check_granite_speech5_projector_weight_shapes(const assets::TensorSource & source) {
    require_shape(source, "projector.query", {1, 3, 1024});
    require_vector(source, "projector.qformer.layernorm.weight", 1024);
    require_vector(source, "projector.qformer.layernorm.bias", 1024);
    require_shape(source, "projector.linear.weight", {2048, 1024});
    require_vector(source, "projector.linear.bias", 2048);
    check_qformer_layer(source, 0);
    check_qformer_layer(source, 1);
    if (source.has_tensor("projector.qformer.encoder.layer.2.attention.attention.query.weight")) {
        throw std::runtime_error("granite_speech5 projector has more than 2 Q-Former layers");
    }
}

void check_granite_speech5_tensor_inventory(const assets::TensorSource & source) {
    check_granite_speech5_encoder_weight_shapes(source);
    check_granite_speech5_projector_weight_shapes(source);

    require_shape(source, "language_model.model.embed_tokens.weight", {100353, 2048});
    require_shape(source, "language_model.lm_head.weight", {100353, 2048});
    require_vector(source, "language_model.model.norm.weight", 2048);
    for (int layer = 0; layer < 40; ++layer) {
        check_text_layer(source, layer);
    }

    // Exact layer counts: no encoder layer 16, Q-Former layer 2, or text
    // layer 40 may exist.
    if (source.has_tensor("encoder.layers.16.attn.to_q.weight") ||
        source.has_tensor("projector.qformer.encoder.layer.2.attention.attention.query.weight") ||
        source.has_tensor("language_model.model.layers.40.input_layernorm.weight")) {
        throw std::runtime_error("granite_speech5 tensor inventory found layers beyond 40/16/2");
    }
}

std::shared_ptr<const GraniteSpeech5Assets> load_granite_speech5_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<GraniteSpeech5Assets>();
    assets->resources = load_granite_speech5_resources(model_path);
    assets->config = parse_config(assets->resources);
    validate_config(assets->config);

    // Open the tensor source and check ALL required bindings before any large
    // weight upload. The source stays reopenable in the bundle; B2 releases
    // host storage only after its graph uploads finish.
    assets->source = assets->resources.open_tensor_source("weights");
    check_granite_speech5_tensor_inventory(*assets->source);
    return assets;
}

}  // namespace engine::community_models::granite_speech5
