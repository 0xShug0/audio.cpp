#include "engine/community_models/confucius4_r2t2/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::community_models::confucius4_r2t2 {
namespace json = engine::io::json;
namespace {

R2T2ASRAudioEncoderConfig parse_audio_encoder_config(const json::Value & value) {
    R2T2ASRAudioEncoderConfig config;
    config.num_mel_bins = json::require_i64(value, "num_mel_bins");
    config.encoder_layers = json::require_i64(value, "encoder_layers");
    config.encoder_attention_heads = json::require_i64(value, "encoder_attention_heads");
    config.encoder_ffn_dim = json::require_i64(value, "encoder_ffn_dim");
    config.d_model = json::require_i64(value, "d_model");
    // The Transformers checkpoint calls a small RoPE setting
    // max_position_embeddings. The audio tower still uses the same 1500-frame
    // sinusoidal table as the original Qwen checkpoint.
    config.max_source_positions = json::optional_i64(value, "max_source_positions", 1500);
    config.n_window = json::require_i64(value, "n_window");
    config.n_window_infer = json::require_i64(value, "n_window_infer");
    config.conv_chunksize = json::require_i64(value, "conv_chunksize");
    config.downsample_hidden_size = json::require_i64(value, "downsample_hidden_size");
    config.output_dim = json::require_i64(value, "output_dim");
    config.activation_function = json::require_string(value, "activation_function");
    if (config.activation_function != "gelu") {
        throw std::runtime_error("R2T2 ASR currently supports gelu audio activation");
    }
    return config;
}

R2T2ASRTextDecoderConfig parse_text_decoder_config(
    const json::Value & thinker_config,
    const json::Value & text_config) {
    R2T2ASRTextDecoderConfig config;
    config.vocab_size = json::require_i64(text_config, "vocab_size");
    config.output_size = json::optional_i64(thinker_config, "classify_num", config.vocab_size);
    config.hidden_size = json::require_i64(text_config, "hidden_size");
    config.intermediate_size = json::require_i64(text_config, "intermediate_size");
    config.num_hidden_layers = json::require_i64(text_config, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(text_config, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(text_config, "num_key_value_heads");
    config.head_dim = json::optional_i64(text_config, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(text_config, "max_position_embeddings");
    config.audio_token_id = json::require_i64(thinker_config, "audio_token_id");
    config.audio_start_token_id = json::optional_i64(thinker_config, "audio_start_token_id", 0);
    config.audio_end_token_id = json::optional_i64(thinker_config, "audio_end_token_id", 0);
    config.pad_token_id = json::optional_i64(thinker_config, "pad_token_id", config.pad_token_id);
    config.pad_token_id = json::optional_i64(text_config, "pad_token_id", config.pad_token_id);
    config.rms_norm_eps = json::optional_f32(text_config, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(text_config, "rope_theta", config.rope_theta);
    config.attention_bias = json::optional_bool(text_config, "attention_bias", config.attention_bias);
    const auto * rope_parameters = text_config.find("rope_parameters");
    if (rope_parameters != nullptr && rope_parameters->is_object()) {
        config.rope_theta = json::optional_f32(*rope_parameters, "rope_theta", config.rope_theta);
    }
    const auto * rope_scaling = text_config.find("rope_scaling");
    if (rope_scaling != nullptr && rope_scaling->is_object()) {
        config.mrope_section = json::optional_i64_array_or_scalar(*rope_scaling, "mrope_section", config.mrope_section);
    }
    return config;
}

int64_t require_added_token_id(const assets::ResourceBundle & resources, std::string_view content) {
    const auto tokenizer = resources.parse_json("tokenizer_json");
    for (const auto & item : tokenizer.require("added_tokens").as_array()) {
        const auto * token_content = item.find("content");
        const auto * token_id = item.find("id");
        if (token_content != nullptr && token_content->is_string() &&
            token_id != nullptr && token_id->is_number() && token_content->as_string() == content) {
            return token_id->as_i64();
        }
    }
    throw std::runtime_error("R2T2 ASR tokenizer.json is missing token: " + std::string(content));
}

void add_supported_languages(R2T2ASRConfig & config, const json::Value & root) {
    static constexpr const char * kLanguages[] = {
        "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
        "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
        "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
        "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
    };
    config.supported_languages = {"Auto"};
    const auto * languages = root.find("support_languages");
    if (languages != nullptr && languages->is_array()) {
        for (const auto & language : languages->as_array()) {
            config.supported_languages.push_back(language.as_string());
        }
        return;
    }
    for (const char * language : kLanguages) {
        config.supported_languages.emplace_back(language);
    }
}

R2T2ASRConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto * legacy_thinker = root.find("thinker_config");
    const bool hf_layout = legacy_thinker == nullptr;
    const auto & thinker_config = hf_layout ? root : *legacy_thinker;
    const auto & audio_config = thinker_config.require("audio_config");
    const auto & text_config = thinker_config.require("text_config");

    R2T2ASRConfig config;
    config.hf_transformers_layout = hf_layout;
    config.model_type = json::require_string(root, "model_type");
    config.thinker_model_type = json::optional_string(thinker_config, "model_type", config.model_type);
    config.model_size = json::optional_string(root, "model_size", hf_layout ? "R2T2-1.7B-hf" : config.model_type);
    config.classify_num = json::optional_i64(thinker_config, "classify_num", 0);
    config.timestamp_token_id = json::optional_i64(root, "timestamp_token_id", 0);
    config.tie_word_embeddings = json::optional_bool(
        root,
        "tie_word_embeddings",
        json::optional_bool(text_config, "tie_word_embeddings", false));
    config.audio_encoder = parse_audio_encoder_config(audio_config);
    config.text_decoder = parse_text_decoder_config(thinker_config, text_config);
    if (hf_layout) {
        config.text_decoder.audio_start_token_id = require_added_token_id(resources, "<|audio_start|>");
        config.text_decoder.audio_end_token_id = require_added_token_id(resources, "<|audio_end|>");
    }

    const auto generation = resources.parse_json("generation_config");
    config.max_new_tokens = json::optional_i64(generation, "max_new_tokens", config.max_new_tokens);
    config.text_decoder.pad_token_id = json::optional_i64(generation, "pad_token_id", config.text_decoder.pad_token_id);
    config.text_decoder.eos_token_ids = json::require_i64_array_or_scalar(generation, "eos_token_id");

    const auto processor = resources.parse_json(resources.has_file("processor_config") ? "processor_config" : "preprocessor_config");
    const auto * feature_extractor = processor.find("feature_extractor");
    const auto & frontend = feature_extractor != nullptr && feature_extractor->is_object() ? *feature_extractor : processor;
    config.frontend.sample_rate = static_cast<int>(json::optional_i64(frontend, "sampling_rate", config.frontend.sample_rate));
    config.frontend.feature_size = json::require_i64(frontend, "feature_size");
    config.frontend.hop_length = json::require_i64(frontend, "hop_length");
    config.frontend.n_fft = json::require_i64(frontend, "n_fft");
    config.timestamp_segment_time_ms = json::optional_i64(processor, "timestamp_segment_time", 0);
    if (config.timestamp_segment_time_ms == 0) {
        config.timestamp_segment_time_ms = json::optional_i64(root, "timestamp_segment_time", 0);
    }
    config.sample_rate = config.frontend.sample_rate;
    if (config.frontend.feature_size != config.audio_encoder.num_mel_bins) {
        throw std::runtime_error("R2T2 ASR frontend feature size does not match audio encoder config");
    }

    add_supported_languages(config, root);
    return config;
}

assets::ResourceBundle make_resource_bundle(
    const std::filesystem::path & model_path,
    std::string_view package_family) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, package_family);
    if (!resources.has_file("preprocessor_config") && !resources.has_file("processor_config")) {
        throw std::runtime_error("R2T2 ASR requires preprocessor_config.json or processor_config.json");
    }
    const bool has_legacy_tokenizer = resources.has_file("vocab") && resources.has_file("merges");
    if (!has_legacy_tokenizer && !resources.has_file("tokenizer_json")) {
        throw std::runtime_error("R2T2 ASR requires vocab.json plus merges.txt, or tokenizer.json");
    }
    return resources;
}

/// Locates the checkpoint without going through the resource bundle's
/// canonicalized tensor path. Canonicalizing dereferences symlinked weights
/// (Hugging Face cache snapshots, `hf download --local-dir` layouts), and the
/// blob target has no file extension, which the tensor source opener uses to
/// pick a format. Keeping this resolution inside the family avoids changing
/// shared framework behavior for every other model.
///
/// A GGUF checkpoint wins when both formats sit in the same directory, matching
/// the convention documented for the Qwen family ("loaders prefer model.gguf
/// when both formats are present").
std::shared_ptr<const assets::TensorSource> open_model_weights(
    const assets::ResourceBundle & resources) {
    const auto & root = resources.model_root();
    if (!root.empty() && engine::io::is_existing_directory(root)) {
        std::filesystem::path safetensors;
        std::filesystem::path gguf;
        std::error_code ec;
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) {
                continue;
            }
            std::string extension = it->path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (extension == ".gguf" && gguf.empty()) {
                gguf = it->path();
            } else if (extension == ".safetensors" && safetensors.empty()) {
                safetensors = it->path();
            }
        }
        if (!gguf.empty()) {
            return assets::open_tensor_source(gguf);
        }
        if (!safetensors.empty()) {
            // Sharded safetensors ship an index beside the shards and are loaded
            // through the package path instead (the index maps tensor names).
            if (safetensors.extension() != ".safetensors" ||
                !engine::io::is_existing_file(safetensors.parent_path() / "model.safetensors.index.json")) {
                return assets::open_tensor_source(safetensors);
            }
        }
    }
    // Sharded safetensors and other layouts stay on the package path.
    return resources.open_tensor_source("weights");
}

/// R2T2 runs f32/f16/bf16/q8_0 weights. ggml's 4-bit and k-quant kernels are
/// not validated for this graph — a Q4_K checkpoint loads and then decodes to
/// an empty transcript — so reject lower precisions at load time with the fix
/// in the message instead of failing silently at inference time.
void validate_checkpoint_weight_types(const assets::TensorSource & source) {
    static constexpr std::array<std::string_view, 4> kSupported = {"f32", "f16", "bf16", "q8_0"};
    for (const auto & meta : source.tensors()) {
        if (meta.name.size() < 7 || meta.name.compare(meta.name.size() - 7, 7, ".weight") != 0) {
            continue;
        }
        std::string dtype = meta.dtype;
        std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (std::find(kSupported.begin(), kSupported.end(), dtype) != kSupported.end()) {
            continue;
        }
        throw std::runtime_error(
            "R2T2 ASR supports f32/f16/bf16/q8_0 weights, but tensor '" + meta.name + "' is " + meta.dtype +
            ". Reconvert the checkpoint with --type q8_0 (or f16).");
    }
}

std::shared_ptr<const R2T2ASRAssets> make_assets(
    assets::ResourceBundle resources) {
    if (!resources.has_file("preprocessor_config") &&
        !resources.has_file("processor_config")) {
        throw std::runtime_error(
            "R2T2 ASR requires preprocessor_config.json or processor_config.json");
    }
    const bool has_legacy_tokenizer =
        resources.has_file("vocab") && resources.has_file("merges");
    if (!has_legacy_tokenizer && !resources.has_file("tokenizer_json")) {
        throw std::runtime_error(
            "R2T2 ASR requires vocab.json plus merges.txt, or tokenizer.json");
    }
    R2T2ASRAssets assets;
    assets.resources = std::move(resources);
    assets.config = parse_config(assets.resources);
    assets.model_weights = open_model_weights(assets.resources);
    validate_checkpoint_weight_types(*assets.model_weights);
    return std::make_shared<R2T2ASRAssets>(std::move(assets));
}

}  // namespace

std::shared_ptr<const R2T2ASRAssets> load_confucius4_r2t2_assets(const std::filesystem::path & model_path) {
    return load_confucius4_r2t2_assets(model_path, "confucius4_r2t2");
}

std::shared_ptr<const R2T2ASRAssets> load_confucius4_r2t2_assets(
    const std::filesystem::path & model_path,
    std::string_view package_family) {
    return make_assets(make_resource_bundle(model_path, package_family));
}

std::shared_ptr<const R2T2ASRAssets> load_confucius4_r2t2_assets(
    assets::ResourceBundle resources) {
    return make_assets(std::move(resources));
}

}  // namespace engine::community_models::confucius4_r2t2
