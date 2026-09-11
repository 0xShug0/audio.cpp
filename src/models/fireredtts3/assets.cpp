#include "engine/models/fireredtts3/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::fireredtts3 {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "fireredtts3";

class DotPrefixedTensorSource final : public engine::assets::TensorSource {
public:
    DotPrefixedTensorSource(std::shared_ptr<const engine::assets::TensorSource> source, std::string prefix)
        : source_(std::move(source)),
          prefix_(std::move(prefix)),
          marker_(prefix_ + ".") {
        if (source_ == nullptr || prefix_.empty()) {
            throw std::runtime_error("FireRedTTS3 tensor source prefix requires source and prefix");
        }
    }

    const std::filesystem::path & source_path() const noexcept override {
        return source_->source_path();
    }

    bool has_tensor(std::string_view name) const noexcept override {
        const auto stripped = strip(name);
        return stripped.empty() ? false : source_->has_tensor(stripped);
    }

    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        auto metadata = source_->require_metadata(require_stripped(name));
        metadata.name = std::string(name);
        return metadata;
    }

    std::vector<engine::assets::TensorMetadata> tensors() const override {
        auto out = source_->tensors();
        for (auto & tensor : out) {
            tensor.name = marker_ + tensor.name;
        }
        return out;
    }

    void release_storage() const override {
        source_->release_storage();
    }

    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        auto data = source_->require_tensor_data(require_stripped(name));
        data.metadata.name = std::string(name);
        return data;
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return source_->require_f32(require_stripped(name), expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto stripped = strip(name);
        if (stripped.empty() || !source_->has_tensor(stripped)) {
            return std::nullopt;
        }
        return source_->require_f32(stripped, expected_shape);
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        engine::assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        source_->set_backend_tensor(tensor, require_stripped(name), storage_type, expected_shape);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        source_->set_backend_f32_tensor(tensor, require_stripped(name), expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return source_->require_i64_scalar(require_stripped(name));
    }

private:
    std::string strip(std::string_view name) const {
        if (name.size() <= marker_.size() || name.substr(0, marker_.size()) != marker_) {
            return {};
        }
        return std::string(name.substr(marker_.size()));
    }

    std::string require_stripped(std::string_view name) const {
        auto stripped = strip(name);
        if (stripped.empty()) {
            throw std::runtime_error("FireRedTTS3 tensor name is outside prefix: " + std::string(name));
        }
        return stripped;
    }

    std::shared_ptr<const engine::assets::TensorSource> source_;
    std::string prefix_;
    std::string marker_;
};

FireRedTTS3BaseConfig parse_core_config(const engine::assets::ResourceBundle & resources, const std::string & key) {
    const auto root = resources.parse_json(key);
    FireRedTTS3BaseConfig config;
    config.redae_dim = json::optional_i64(root, "redae_dim", config.redae_dim);
    config.history_patches = json::optional_i64(root, "num_history_patches", config.history_patches);
    config.speaker_dim = json::optional_i64(root, "spk_in_dim", config.speaker_dim);
    config.patch_size = json::optional_i64(root, "patch_size", config.patch_size);
    config.patch_hidden_size = json::optional_i64(root, "patch_encoder_hidden_size", config.patch_hidden_size);
    config.patch_layers = json::optional_i64(root, "patch_encoder_depth", config.patch_layers);
    config.patch_heads = json::optional_i64(root, "patch_encoder_num_heads", config.patch_heads);
    const int64_t patch_mlp_ratio = json::optional_i64(root, "patch_encoder_mlp_ratio", 4);
    config.patch_intermediate_size = config.patch_hidden_size * patch_mlp_ratio;
    config.dit_hidden_size = json::optional_i64(root, "dit_hidden_size", config.dit_hidden_size);
    config.dit_layers = json::optional_i64(root, "dit_depth", config.dit_layers);
    config.dit_heads = json::optional_i64(root, "dit_num_heads", config.dit_heads);
    const int64_t dit_mlp_ratio = json::optional_i64(root, "dit_mlp_ratio", 3);
    config.dit_intermediate_size = config.dit_hidden_size * dit_mlp_ratio;
    engine::io::require_positive(config.patch_size, "FireRedTTS3 patch size");
    engine::io::require_positive(config.redae_dim, "FireRedTTS3 RedAE dimension");
    engine::io::require_positive(config.dit_hidden_size, "FireRedTTS3 DiT hidden size");
    return config;
}

FireRedTTS3RedAeConfig parse_redae_config(const engine::assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("redae_config");
    FireRedTTS3RedAeConfig config;
    config.sample_rate = json::optional_i64(root, "audio_sample_rate", config.sample_rate);
    config.audio_patch_size = json::optional_i64(root, "audio_patch_size", config.audio_patch_size);
    config.bottleneck_dim = json::optional_i64(root, "bottleneck_dim", config.bottleneck_dim);
    config.enc_hidden_size = json::optional_i64(root, "enc_hidden_size", config.enc_hidden_size);
    config.enc_intermediate_size = json::optional_i64(root, "enc_intermediate_size", config.enc_intermediate_size);
    config.enc_layers = json::optional_i64(root, "enc_num_hidden_layers", config.enc_layers);
    config.enc_heads = json::optional_i64(root, "enc_num_attention_heads", config.enc_heads);
    config.enc_kv_heads = json::optional_i64(root, "enc_num_key_value_heads", config.enc_kv_heads);
    const bool enc_use_sliding_window = json::optional_bool(root, "enc_use_sliding_window", false);
    const int64_t enc_max_window_layers = json::optional_i64(root, "enc_max_window_layers", config.enc_layers);
    if (enc_use_sliding_window) {
        if (enc_max_window_layers != 0) {
            throw std::runtime_error("FireRedTTS3 RedAE encoder supports only all-layer sliding attention");
        }
        config.enc_sliding_window = json::optional_i64(root, "enc_sliding_window", 0);
    }
    config.enc_extra_downsample_rate = json::optional_i64(root, "enc_extra_downsample_rate", config.enc_extra_downsample_rate);
    config.enc_downsample_layers = json::optional_i64(root, "enc_downsample_num_hidden_layers", config.enc_downsample_layers);
    config.dec_hidden_size = json::optional_i64(root, "dec_hidden_size", config.dec_hidden_size);
    config.dec_intermediate_size = json::optional_i64(root, "dec_intermediate_size", config.dec_intermediate_size);
    config.dec_layers = json::optional_i64(root, "dec_num_hidden_layers", config.dec_layers);
    config.dec_heads = json::optional_i64(root, "dec_num_attention_heads", config.dec_heads);
    config.dec_kv_heads = json::optional_i64(root, "dec_num_key_value_heads", config.dec_kv_heads);
    const bool dec_use_sliding_window = json::optional_bool(root, "dec_use_sliding_window", false);
    const int64_t dec_max_window_layers = json::optional_i64(root, "dec_max_window_layers", config.dec_layers);
    if (dec_use_sliding_window) {
        if (dec_max_window_layers != 0) {
            throw std::runtime_error("FireRedTTS3 RedAE decoder supports only all-layer sliding attention");
        }
        config.dec_sliding_window = json::optional_i64(root, "dec_sliding_window", 0);
    }
    config.enc_head_dim = config.enc_hidden_size / config.enc_heads;
    config.dec_head_dim = config.dec_hidden_size / config.dec_heads;
    engine::io::require_positive(config.sample_rate, "FireRedTTS3 RedAE sample rate");
    engine::io::require_positive(config.audio_patch_size, "FireRedTTS3 RedAE audio patch size");
    engine::io::require_positive(config.bottleneck_dim, "FireRedTTS3 RedAE bottleneck dimension");
    if (enc_use_sliding_window) {
        engine::io::require_positive(config.enc_sliding_window, "FireRedTTS3 RedAE encoder sliding window");
    }
    if (dec_use_sliding_window) {
        engine::io::require_positive(config.dec_sliding_window, "FireRedTTS3 RedAE decoder sliding window");
    }
    return config;
}

void validate_base_tensor_shapes(const engine::assets::TensorSource & source, FireRedTTS3BaseConfig & config) {
    const auto embedding = source.require_metadata("backbone_llm.embed_tokens.weight");
    if (embedding.shape.size() != 2) {
        throw std::runtime_error("FireRedTTS3 Base token embedding must be rank 2");
    }
    config.vocab_size = embedding.shape.at(0);
    config.hidden_size = embedding.shape.at(1);
    engine::assets::require_tensor_shape(source, "spk_proj_llm.weight", {config.hidden_size, config.speaker_dim});
    engine::assets::require_tensor_shape(source, "spk_proj_dit.weight", {config.speaker_dim, config.speaker_dim});
    engine::assets::require_tensor_shape(source, "patch_encoder.in_proj.weight", {config.patch_hidden_size, config.redae_dim});
    engine::assets::require_tensor_shape(source, "patch_encoder.out_proj.linear.weight", {config.hidden_size, config.patch_hidden_size});
    engine::assets::require_tensor_shape(source, "dit.in_proj.weight", {config.dit_hidden_size, config.redae_dim + config.speaker_dim + config.dit_hidden_size});
    engine::assets::require_tensor_shape(source, "dit_head.weight", {config.dit_hidden_size, config.hidden_size});
    engine::assets::require_tensor_shape(source, "stop_head.weight", {1, config.hidden_size});
}

void validate_instruct_tensor_shapes(const engine::assets::TensorSource & source, FireRedTTS3BaseConfig & config) {
    config.speaker_dim = 0;
    const auto embedding = source.require_metadata("backbone_llm.model.embed_tokens.weight");
    if (embedding.shape.size() != 2) {
        throw std::runtime_error("FireRedTTS3 Instruct token embedding must be rank 2");
    }
    config.vocab_size = embedding.shape.at(0);
    config.hidden_size = embedding.shape.at(1);
    engine::assets::require_tensor_shape(source, "patch_encoder.in_proj.weight", {config.patch_hidden_size, config.redae_dim});
    engine::assets::require_tensor_shape(source, "patch_encoder.out_proj.linear.weight", {config.hidden_size, config.patch_hidden_size});
    engine::assets::require_tensor_shape(source, "dit.in_proj.weight", {config.dit_hidden_size, config.redae_dim + config.dit_hidden_size});
    engine::assets::require_tensor_shape(source, "dit_head.weight", {config.dit_hidden_size, config.hidden_size});
    engine::assets::require_tensor_shape(source, "stop_head.weight", {1, config.hidden_size});
}

void validate_redae_qwen_shapes(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t heads,
    int64_t kv_heads,
    int64_t & head_dim) {
    const auto q = source.require_metadata(prefix + ".layers.0.self_attn.q_proj.weight");
    if (q.shape.size() != 2 || q.shape.at(1) != hidden || q.shape.at(0) % heads != 0) {
        throw std::runtime_error("FireRedTTS3 RedAE Q projection shape mismatch for " + prefix);
    }
    head_dim = q.shape.at(0) / heads;
    engine::assets::require_tensor_shape(source, prefix + ".layers.0.self_attn.k_proj.weight", {kv_heads * head_dim, hidden});
    engine::assets::require_tensor_shape(source, prefix + ".layers.0.self_attn.v_proj.weight", {kv_heads * head_dim, hidden});
    engine::assets::require_tensor_shape(source, prefix + ".layers.0.self_attn.o_proj.weight", {hidden, heads * head_dim});
}

void validate_redae_tensor_shapes(const engine::assets::TensorSource & source, FireRedTTS3RedAeConfig & config) {
    engine::assets::require_tensor_shape(source, "encoder.in_proj.0.weight", {config.enc_hidden_size, config.audio_patch_size});
    engine::assets::require_tensor_shape(source, "encoder.in_proj.1.weight", {config.enc_hidden_size, config.enc_hidden_size});
    engine::assets::require_tensor_shape(source, "encoder.out_proj.weight", {config.bottleneck_dim, config.enc_hidden_size});
    engine::assets::require_tensor_shape(source, "decoder.in_proj.weight", {config.enc_extra_downsample_rate * config.dec_hidden_size, config.bottleneck_dim});
    engine::assets::require_tensor_shape(source, "decoder.istft_head.out.weight", {config.audio_patch_size * 4 + 2, config.dec_hidden_size});
    engine::assets::require_tensor_shape(source, "decoder.istft_head.istft.window", {config.audio_patch_size * 4});
    validate_redae_qwen_shapes(
        source, "encoder.qwen3", config.enc_hidden_size, config.enc_heads, config.enc_kv_heads, config.enc_head_dim);
    validate_redae_qwen_shapes(
        source, "encoder.downsample.qwen3", config.enc_hidden_size, config.enc_heads, config.enc_kv_heads, config.enc_head_dim);
    validate_redae_qwen_shapes(
        source, "decoder.qwen3", config.dec_hidden_size, config.dec_heads, config.dec_kv_heads, config.dec_head_dim);
}

std::filesystem::path find_gguf_path(const engine::assets::ResourceBundle & resources) {
    for (const auto & file : resources.files()) {
        if (file.path.extension() == ".gguf") {
            return file.path;
        }
    }
    return {};
}

std::shared_ptr<const engine::assets::TensorSource> optional_tensor_source(
    const engine::assets::ResourceBundle & resources,
    std::string_view id) {
    if (!resources.has_file(id)) {
        return nullptr;
    }
    try {
        return resources.open_tensor_source(id);
    } catch (const std::exception & error) {
        const std::string message = error.what();
        if (message.find("packed GGUF namespace does not exist") != std::string::npos) {
            return nullptr;
        }
        throw;
    }
}

}  // namespace

std::shared_ptr<const FireRedTTS3Assets> load_fireredtts3_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<FireRedTTS3Assets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    assets->model_root = assets->resources.model_root();
    assets->gguf_path = find_gguf_path(assets->resources);
    assets->redae = parse_redae_config(assets->resources);
    assets->redae_weights = assets->resources.open_tensor_source("redae_weights");
    assets->base_weights = optional_tensor_source(assets->resources, "base_weights");
    assets->instruct_weights = optional_tensor_source(assets->resources, "instruct_weights");
    const bool has_base =
        assets->resources.has_file("base_config") &&
        assets->base_weights != nullptr &&
        assets->base_weights->has_tensor("backbone_llm.embed_tokens.weight");
    const bool has_instruct =
        assets->resources.has_file("instruct_config") &&
        assets->instruct_weights != nullptr &&
        assets->instruct_weights->has_tensor("backbone_llm.model.embed_tokens.weight");
    if (has_base == has_instruct) {
        throw std::runtime_error("FireRedTTS3 package must contain exactly one Base or Instruct model variant");
    }
    if (has_base) {
        assets->variant = FireRedTTS3Variant::Base;
        assets->base = parse_core_config(assets->resources, "base_config");
        assets->campplus_weights = std::make_shared<DotPrefixedTensorSource>(
            assets->resources.open_tensor_source("campplus_weights"),
            "campplus");
        validate_base_tensor_shapes(*assets->base_weights, assets->base);
    } else {
        assets->variant = FireRedTTS3Variant::Instruct;
        assets->base = parse_core_config(assets->resources, "instruct_config");
        validate_instruct_tensor_shapes(*assets->instruct_weights, assets->base);
    }
    validate_redae_tensor_shapes(*assets->redae_weights, assets->redae);
    return assets;
}

}  // namespace engine::models::fireredtts3
