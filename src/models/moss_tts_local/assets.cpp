#include "engine/models/moss_tts_local/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_tts_local {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("MOSS-TTS-Local model path does not exist: " + model_path.string());
}

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"config", "config.json", true},
        {"tokenizer_json", "tokenizer.json", false},
        {"tokenizer_config", "tokenizer_config.json", false},
        {"tokenizer_vocab", "vocab.json", false},
        {"tokenizer_merges", "merges.txt", false},
    });
    return resources;
}

void require_positive(int64_t value, const char * label) {
    if (value <= 0) {
        throw std::runtime_error(std::string("MOSS-TTS-Local config has non-positive ") + label);
    }
}

MossBackboneConfig parse_backbone_config(const engine::io::json::Value & value) {
    MossBackboneConfig config;
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    require_positive(config.num_attention_heads, "backbone num_attention_heads");
    config.head_dim =
        json::optional_i64(value, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    require_positive(config.hidden_size, "backbone hidden_size");
    require_positive(config.intermediate_size, "backbone intermediate_size");
    require_positive(config.num_hidden_layers, "backbone num_hidden_layers");
    require_positive(config.num_key_value_heads, "backbone num_key_value_heads");
    require_positive(config.head_dim, "backbone head_dim");
    require_positive(config.vocab_size, "backbone vocab_size");
    return config;
}

MossLocalTransformerConfig parse_local_config(const engine::io::json::Value & value) {
    MossLocalTransformerConfig config;
    config.hidden_size = json::require_i64(value, "n_embd");
    config.intermediate_size = json::optional_i64(value, "n_inner", config.hidden_size * 4);
    config.num_layers = json::require_i64(value, "n_layer");
    config.num_heads = json::require_i64(value, "n_head");
    config.max_positions = json::optional_i64(value, "n_positions", config.max_positions);
    config.layer_norm_eps = json::optional_f32(value, "layer_norm_epsilon", config.layer_norm_eps);
    config.rope_base = json::optional_f32(value, "rope_base", config.rope_base);
    require_positive(config.hidden_size, "local n_embd");
    require_positive(config.num_layers, "local n_layer");
    require_positive(config.num_heads, "local n_head");
    return config;
}

std::filesystem::path huggingface_hub_root() {
    if (const char * hf_home = std::getenv("HF_HOME")) {
        return std::filesystem::path(hf_home) / "hub";
    }
    const char * home = std::getenv("USERPROFILE");
    if (home == nullptr) {
        home = std::getenv("HOME");
    }
    if (home == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local cannot locate the Hugging Face cache: no HF_HOME/USERPROFILE/HOME");
    }
    return std::filesystem::path(home) / ".cache" / "huggingface" / "hub";
}

std::string repo_id_to_cache_folder(const std::string & repo_id) {
    // "Org/Name" -> "models--Org--Name" (Hugging Face hub cache naming).
    std::string folder = "models--";
    for (const char ch : repo_id) {
        if (ch == '/') {
            folder += "--";
        } else {
            folder += ch;
        }
    }
    return folder;
}

}  // namespace

MossTTSLocalAssetPaths resolve_moss_tts_local_assets(const std::filesystem::path & model_path) {
    const auto root = resolve_model_root(model_path);
    MossTTSLocalAssetPaths paths;
    paths.model_root = root;
    paths.config_path = root / "config.json";
    paths.model_weights_path = root / "model.safetensors";
    const auto add_optional_file = [&](std::optional<std::filesystem::path> & slot, const char * name) {
        const auto path = root / name;
        if (engine::io::is_existing_file(path)) {
            slot = path;
        }
    };
    add_optional_file(paths.tokenizer_json_path, "tokenizer.json");
    add_optional_file(paths.tokenizer_config_path, "tokenizer_config.json");
    add_optional_file(paths.tokenizer_vocab_path, "vocab.json");
    add_optional_file(paths.tokenizer_merges_path, "merges.txt");
    const auto audio_tokenizer_root = root / "audio_tokenizer";
    if (engine::io::is_existing_directory(audio_tokenizer_root)) {
        paths.audio_tokenizer_root = audio_tokenizer_root;
    }
    return paths;
}

MossTTSLocalConfig load_moss_tts_local_config(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    const auto root = resources.parse_json("config");
    const auto model_type = json::optional_string(root, "model_type", "");
    if (model_type != "moss_tts_local") {
        throw std::runtime_error(
            "MOSS-TTS-Local config model_type mismatch: expected moss_tts_local, got " + model_type);
    }
    MossTTSLocalConfig config;
    config.backbone = parse_backbone_config(root.require("qwen3_config"));
    config.local = parse_local_config(root.require("gpt2_config"));
    config.num_codebooks = json::require_i64(root, "n_vq");
    config.audio_vocab_size = json::optional_i64(root, "audio_vocab_size", 0);
    config.audio_codebook_sizes = json::optional_i64_array(root, "audio_codebook_sizes");
    config.audio_pad_token_id = json::optional_i64(root, "audio_pad_token_id", 0);
    config.pad_token_id = json::optional_i64(root, "pad_token_id", 0);
    config.im_start_token_id = json::optional_i64(root, "im_start_token_id", 0);
    config.im_end_token_id = json::optional_i64(root, "im_end_token_id", 0);
    config.audio_start_token_id = json::optional_i64(root, "audio_start_token_id", 0);
    config.audio_end_token_id = json::optional_i64(root, "audio_end_token_id", 0);
    config.audio_user_slot_token_id = json::optional_i64(root, "audio_user_slot_token_id", 0);
    config.audio_assistant_slot_token_id = json::optional_i64(root, "audio_assistant_slot_token_id", 0);
    config.sampling_rate = json::optional_i64(root, "sampling_rate", 0);
    config.local_text_head_mode = json::optional_string(root, "local_text_head_mode", "");
    config.audio_tokenizer_name_or_path = json::optional_string(root, "audio_tokenizer_name_or_path", "");
    require_positive(config.num_codebooks, "n_vq");
    if (!config.audio_codebook_sizes.empty() &&
        static_cast<int64_t>(config.audio_codebook_sizes.size()) != config.num_codebooks) {
        throw std::runtime_error("MOSS-TTS-Local audio_codebook_sizes length does not match n_vq");
    }
    return config;
}

std::shared_ptr<const MossTTSLocalAssets> load_moss_tts_local_assets(const std::filesystem::path & model_path) {
    MossTTSLocalAssets assets;
    assets.paths = resolve_moss_tts_local_assets(model_path);
    assets.config = load_moss_tts_local_config(model_path);
    if (!engine::io::is_existing_file(assets.paths.model_weights_path)) {
        throw std::runtime_error(
            "MOSS-TTS-Local weights not found: " + assets.paths.model_weights_path.string());
    }
    assets.model_weights = assets::open_tensor_source(assets.paths.model_weights_path);
    return std::make_shared<MossTTSLocalAssets>(std::move(assets));
}

std::filesystem::path resolve_moss_codec_dir(const MossTTSLocalAssets & assets) {
    if (assets.paths.audio_tokenizer_root.has_value() &&
        engine::io::is_existing_file(*assets.paths.audio_tokenizer_root / "config.json")) {
        return *assets.paths.audio_tokenizer_root;
    }
    const std::string & repo_id = assets.config.audio_tokenizer_name_or_path;
    if (repo_id.empty()) {
        throw std::runtime_error(
            "MOSS-TTS-Local audio tokenizer not found: no local audio_tokenizer/ and no repo id in config");
    }
    const auto snapshots = huggingface_hub_root() / repo_id_to_cache_folder(repo_id) / "snapshots";
    if (!engine::io::is_existing_directory(snapshots)) {
        throw std::runtime_error(
            "MOSS-TTS-Local codec not found in Hugging Face cache: " + snapshots.string() +
            " (download " + repo_id + " or place it under the model's audio_tokenizer/ directory)");
    }
    for (const auto & entry : std::filesystem::directory_iterator(snapshots)) {
        if (entry.is_directory() && engine::io::is_existing_file(entry.path() / "config.json")) {
            return entry.path();
        }
    }
    throw std::runtime_error("MOSS-TTS-Local codec snapshot with config.json not found under " + snapshots.string());
}

}  // namespace engine::models::moss_tts_local
