#include "engine/community_models/voxcpm1/assets.h"
#include "engine/community_models/voxcpm1/tokenizer_gguf.h"
#include "engine/community_models/voxcpm1/config_gguf.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

namespace engine::community_models::voxcpm1 {
namespace json = engine::io::json;
namespace {

VoxCPM1RopeScalingConfig parse_rope_scaling(const json::Value & value) {
    VoxCPM1RopeScalingConfig config;
    config.type = json::optional_string(value, "type", "");
    config.long_factor = json::optional_f32_array(value, "long_factor");
    config.short_factor = json::optional_f32_array(value, "short_factor");
    config.original_max_position_embeddings =
        json::optional_i64(value, "original_max_position_embeddings", 0);
    return config;
}

VoxCPM1MiniCPMConfig parse_lm_config(const json::Value & value) {
    VoxCPM1MiniCPMConfig config;
    config.bos_token_id = json::optional_i64(value, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(value, "eos_token_id", config.eos_token_id);
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.max_position_embeddings = json::require_i64(value, "max_position_embeddings");
    config.num_attention_heads = json::require_i64(value, "num_attention_heads");
    config.num_hidden_layers = json::require_i64(value, "num_hidden_layers");
    config.num_key_value_heads = json::require_i64(value, "num_key_value_heads");
    config.kv_channels = json::optional_i64(value, "kv_channels", config.hidden_size / config.num_attention_heads);
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.scale_emb = json::optional_i64(value, "scale_emb", config.scale_emb);
    config.dim_model_base = json::optional_i64(value, "dim_model_base", config.dim_model_base);
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(value, "rope_theta", config.rope_theta);
    config.scale_depth = json::optional_f32(value, "scale_depth", config.scale_depth);
    config.use_mup = json::optional_bool(value, "use_mup", config.use_mup);
    if (const auto * rope_scaling = value.find("rope_scaling"); rope_scaling != nullptr) {
        config.rope_scaling = parse_rope_scaling(*rope_scaling);
    }
    engine::io::require_positive(config.hidden_size, "lm hidden_size");
    engine::io::require_positive(config.intermediate_size, "lm intermediate_size");
    engine::io::require_positive(config.max_position_embeddings, "lm max_position_embeddings");
    engine::io::require_positive(config.num_attention_heads, "lm num_attention_heads");
    engine::io::require_positive(config.num_hidden_layers, "lm num_hidden_layers");
    engine::io::require_positive(config.num_key_value_heads, "lm num_key_value_heads");
    engine::io::require_positive(config.kv_channels, "lm kv_channels");
    engine::io::require_positive(config.vocab_size, "lm vocab_size");
    engine::io::require_divisible(config.hidden_size, config.num_attention_heads, "lm hidden_size / num_attention_heads");
    engine::io::require_divisible(config.num_attention_heads, config.num_key_value_heads, "lm attention heads");
    if (!config.rope_scaling.type.empty()) {
        if (config.rope_scaling.type != "longrope") {
            throw std::runtime_error("VoxCPM1 currently expects longrope rope_scaling");
        }
        const int64_t expected = config.hidden_size / config.num_attention_heads / 2;
        if (static_cast<int64_t>(config.rope_scaling.long_factor.size()) != expected ||
            static_cast<int64_t>(config.rope_scaling.short_factor.size()) != expected) {
            throw std::runtime_error("VoxCPM1 rope_scaling factor length does not match head_dim / 2");
        }
    }
    return config;
}

VoxCPM1LocalTransformerConfig parse_local_transformer_config(
    const json::Value & value,
    const char * label) {
    VoxCPM1LocalTransformerConfig config;
    config.hidden_dim = json::require_i64(value, "hidden_dim");
    config.ffn_dim = json::require_i64(value, "ffn_dim");
    config.num_heads = json::require_i64(value, "num_heads");
    config.num_layers = json::require_i64(value, "num_layers");
    config.kv_channels = json::optional_i64(value, "kv_channels", config.hidden_dim / config.num_heads);
    engine::io::require_positive(config.hidden_dim, label);
    engine::io::require_positive(config.ffn_dim, label);
    engine::io::require_positive(config.num_heads, label);
    engine::io::require_positive(config.num_layers, label);
    engine::io::require_positive(config.kv_channels, label);
    engine::io::require_divisible(config.hidden_dim, config.num_heads, label);
    return config;
}

VoxCPM1DiTConfig parse_dit_config(const json::Value & value) {
    const auto base = parse_local_transformer_config(value, "dit transformer");
    VoxCPM1DiTConfig config;
    config.hidden_dim = base.hidden_dim;
    config.ffn_dim = base.ffn_dim;
    config.num_heads = base.num_heads;
    config.num_layers = base.num_layers;
    config.kv_channels = base.kv_channels;
    config.mean_mode = json::optional_bool(value, "dit_mean_mode", json::optional_bool(value, "mean_mode", false));
    const auto & cfm = value.require("cfm_config");
    config.cfm.sigma_min = json::optional_f32(cfm, "sigma_min", config.cfm.sigma_min);
    config.cfm.solver = json::optional_string(cfm, "solver", config.cfm.solver);
    config.cfm.t_scheduler = json::optional_string(cfm, "t_scheduler", config.cfm.t_scheduler);
    config.cfm.inference_cfg_rate = json::optional_f32(cfm, "inference_cfg_rate", config.cfm.inference_cfg_rate);
    if (config.cfm.solver != "euler") {
        throw std::runtime_error("VoxCPM1 CFM currently expects euler solver");
    }
    if (config.cfm.t_scheduler != "log-norm") {
        throw std::runtime_error("VoxCPM1 CFM currently expects log-norm scheduler");
    }
    return config;
}

VoxCPM1AudioVAEConfig parse_audio_vae_config(const json::Value & value) {
    VoxCPM1AudioVAEConfig config;
    config.encoder_dim = json::require_i64(value, "encoder_dim");
    config.encoder_rates = json::require_i64_array(value, "encoder_rates");
    config.latent_dim = json::require_i64(value, "latent_dim");
    config.decoder_dim = json::require_i64(value, "decoder_dim");
    config.decoder_rates = json::require_i64_array(value, "decoder_rates");
    config.sample_rate_bin_boundaries = json::optional_i64_array(value, "sr_bin_boundaries");
    config.sample_rate = static_cast<int>(json::require_i64(value, "sample_rate"));
    config.output_sample_rate = static_cast<int>(json::require_i64(value, "out_sample_rate"));
    engine::io::require_positive(config.encoder_dim, "AudioVAE encoder_dim");
    engine::io::require_positive(config.latent_dim, "AudioVAE latent_dim");
    engine::io::require_positive(config.decoder_dim, "AudioVAE decoder_dim");
    engine::io::require_positive(config.sample_rate, "AudioVAE sample_rate");
    engine::io::require_positive(config.output_sample_rate, "AudioVAE out_sample_rate");
    if (config.encoder_rates.empty() || config.decoder_rates.empty()) {
        throw std::runtime_error("VoxCPM1 AudioVAE rates must be non-empty");
    }
    for (const auto rate : config.encoder_rates) {
        engine::io::require_positive(rate, "AudioVAE encoder rate");
    }
    for (const auto rate : config.decoder_rates) {
        engine::io::require_positive(rate, "AudioVAE decoder rate");
    }
    return config;
}

VoxCPM1Config parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    VoxCPM1Config config;
    config.architecture = json::require_string(root, "architecture");
    if (config.architecture != "voxcpm") {
        throw std::runtime_error("VoxCPM1 config architecture mismatch: " + config.architecture);
    }
    config.lm = parse_lm_config(root.require("lm_config"));
    config.patch_size = json::optional_i64(root, "patch_size", config.patch_size);
    config.feat_dim = json::optional_i64(root, "feat_dim", config.feat_dim);
    config.residual_lm_num_layers =
        json::optional_i64(root, "residual_lm_num_layers", config.residual_lm_num_layers);
    config.residual_lm_no_rope = json::optional_bool(root, "residual_lm_no_rope", config.residual_lm_no_rope);
    config.scalar_quantization_latent_dim =
        json::optional_i64(root, "scalar_quantization_latent_dim", config.scalar_quantization_latent_dim);
    config.scalar_quantization_scale =
        json::optional_i64(root, "scalar_quantization_scale", config.scalar_quantization_scale);
    config.encoder = parse_local_transformer_config(root.require("encoder_config"), "local encoder transformer");
    config.dit = parse_dit_config(root.require("dit_config"));
    config.audio_vae = parse_audio_vae_config(root.require("audio_vae_config"));
    config.max_length = json::optional_i64(root, "max_length", config.max_length);
    config.device = json::optional_string(root, "device", config.device);
    config.dtype = json::optional_string(root, "dtype", config.dtype);
    engine::io::require_positive(config.patch_size, "patch_size");
    engine::io::require_positive(config.feat_dim, "feat_dim");
    engine::io::require_positive(config.residual_lm_num_layers, "residual_lm_num_layers");
    engine::io::require_positive(config.scalar_quantization_latent_dim, "scalar_quantization_latent_dim");
    engine::io::require_positive(config.scalar_quantization_scale, "scalar_quantization_scale");
    engine::io::require_positive(config.max_length, "max_length");
    if (config.feat_dim != config.audio_vae.latent_dim) {
        throw std::runtime_error("VoxCPM1 feat_dim must match AudioVAE latent_dim");
    }
    if (config.residual_lm_num_layers > config.lm.num_hidden_layers) {
        throw std::runtime_error("VoxCPM1 residual_lm_num_layers exceeds lm num_hidden_layers");
    }
    return config;
}

namespace assets = engine::assets;

namespace {
core::TensorShape make_tensor_shape(const std::vector<int64_t> & dims) {
    if (dims.empty() || dims.size() > core::kMaxTensorRank) {
        throw std::runtime_error("tensor rank must be between 1 and 4");
    }
    switch (dims.size()) {
        case 1:
            return core::TensorShape::from_dims({dims[0]});
        case 2:
            return core::TensorShape::from_dims({dims[0], dims[1]});
        case 3:
            return core::TensorShape::from_dims({dims[0], dims[1], dims[2]});
        case 4:
            return core::TensorShape::from_dims({dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::runtime_error("unsupported tensor rank");
    }
}
}  // namespace

class VoxCPM1TensorSource final : public assets::TensorSource {
public:
    VoxCPM1TensorSource(
        std::shared_ptr<const assets::TensorSource> source,
        const VoxCPM1Config & config)
        : source_(std::move(source)), config_(config) {
        build_synthesized();
    }

    const std::filesystem::path & source_path() const noexcept override {
        return source_->source_path();
    }

    bool has_tensor(std::string_view name) const noexcept override {
        if (synthesized_.find(std::string(name)) != synthesized_.end()) {
            return true;
        }
        return source_->has_tensor(name);
    }

    assets::TensorMetadata require_metadata(std::string_view name) const override {
        const auto it = synthesized_.find(std::string(name));
        if (it != synthesized_.end()) {
            return it->second;
        }
        return source_->require_metadata(name);
    }

    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> out;
        out.reserve(synthesized_.size());
        for (const auto & [name, meta] : synthesized_) {
            out.push_back(meta);
        }
        auto src = source_->tensors();
        out.insert(out.end(), src.begin(), src.end());
        std::sort(out.begin(), out.end(),
            [](const assets::TensorMetadata & lhs, const assets::TensorMetadata & rhs) {
                return lhs.name < rhs.name;
            });
        return out;
    }

    void release_storage() const override { source_->release_storage(); }

    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto it = synthesized_.find(std::string(name));
        if (it != synthesized_.end()) {
            return generate_synthesized_tensor(name);
        }
        return source_->require_tensor_data(name);
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto it = synthesized_.find(std::string(name));
        if (it != synthesized_.end()) {
            return generate_synthesized_f32(name);
        }
        if (expected_shape.has_value()) {
            const auto meta = source_->require_metadata(name);
            const int64_t expected_elems = element_count("expected", *expected_shape);
            const int64_t actual_elems = element_count(name, meta.shape);
            if (expected_elems == actual_elems && meta.shape != *expected_shape) {
                // GGUF stores this tensor in a different rank (e.g. 1D) but
                // with the same element count. Return raw data so the caller
                // can reshape it into the expected logical shape.
                return source_->require_f32(name, std::nullopt);
            }
        }
        return source_->require_f32(name, expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const auto it = synthesized_.find(std::string(name));
        if (it != synthesized_.end()) {
            const auto values = generate_synthesized_f32(name);
            engine::assets::set_backend_tensor_from_f32_parallel(tensor, name, values,
                make_tensor_shape(expected_shape),
                engine::assets::ggml_type_for_tensor_storage(storage_type));
            return;
        }
        {
            const auto meta = source_->require_metadata(name);
            const int64_t expected_elems = element_count(name, expected_shape);
            const int64_t actual_elems = element_count(name, meta.shape);
            if (expected_elems == actual_elems && meta.shape != expected_shape) {
                const auto values = source_->require_f32(name, std::nullopt);
                const ggml_type type = engine::assets::ggml_type_for_tensor_storage(
                    engine::assets::resolve_tensor_storage_type(*this, name, storage_type));
                engine::assets::set_backend_tensor_from_f32_parallel(tensor, name, values,
                    make_tensor_shape(expected_shape), type);
                return;
            }
        }
        source_->set_backend_tensor(tensor, name, storage_type, expected_shape);
    }

    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        set_backend_tensor(tensor, name, assets::TensorStorageType::F32, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return source_->require_i64_scalar(name);
    }

private:
    static int64_t element_count(std::string_view name, const std::vector<int64_t> & shape) {
        int64_t count = 1;
        for (const int64_t dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("tensor shape contains a non-positive dimension: " + std::string(name));
            }
            if (count > std::numeric_limits<int64_t>::max() / dim) {
                throw std::runtime_error("tensor element count overflow: " + std::string(name));
            }
            count *= dim;
        }
        return count;
    }

    void build_synthesized() {
        // VoxCPM1 GGUFs do not contain SR-conditioning tensors; the shared
        // decoder loader still requires scale_embed / bias_embed per block.
        // Register identity-initialized placeholders under the V1 names.
        const auto & vae = config_.audio_vae;
        const size_t num_blocks = vae.decoder_rates.size();
        for (size_t i = 0; i < num_blocks; ++i) {
            const int64_t input_channels =
                vae.decoder_dim / (int64_t{1} << static_cast<int>(i));
            const std::string prefix =
                "audio_vae.decoder.sr_cond_model." + std::to_string(i + 2) + ".";
            synthesized_[prefix + "scale_embed.weight"] =
                assets::TensorMetadata{prefix + "scale_embed.weight", "F32", {1, input_channels}};
            synthesized_[prefix + "bias_embed.weight"] =
                assets::TensorMetadata{prefix + "bias_embed.weight", "F32", {1, input_channels}};
        }
    }

    std::vector<float> generate_synthesized_f32(std::string_view name) const {
        const auto it = synthesized_.find(std::string(name));
        if (it == synthesized_.end()) {
            throw std::runtime_error("no synthesized tensor: " + std::string(name));
        }
        const auto & metadata = it->second;
        const int64_t num_elements = element_count(name, metadata.shape);
        constexpr std::string_view kScaleEmbed = "scale_embed.weight";
        if (name.size() >= kScaleEmbed.size() &&
            name.substr(name.size() - kScaleEmbed.size()) == kScaleEmbed) {
            return std::vector<float>(static_cast<size_t>(num_elements), 1.0F);
        }
        return std::vector<float>(static_cast<size_t>(num_elements), 0.0F);
    }

    assets::RawTensorData generate_synthesized_tensor(std::string_view name) const {
        const auto it = synthesized_.find(std::string(name));
        if (it == synthesized_.end()) {
            throw std::runtime_error("no synthesized tensor: " + std::string(name));
        }
        const auto & metadata = it->second;
        const int64_t num_elements = element_count(name, metadata.shape);
        std::vector<std::byte> bytes(static_cast<size_t>(num_elements) * sizeof(float));
        const auto values = generate_synthesized_f32(name);
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return {metadata, std::move(bytes)};
    }

    std::shared_ptr<const assets::TensorSource> source_;
    VoxCPM1Config config_;
    std::unordered_map<std::string, assets::TensorMetadata> synthesized_;
};

void require_vae_weight_v_shape(const assets::TensorSource & source,
                                std::string_view name,
                                const std::vector<int64_t> & expected_shape,
                                bool relaxed_rank) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape == expected_shape) {
        return;
    }
    if (!relaxed_rank) {
        throw std::runtime_error("tensor shape mismatch for " + std::string(name));
    }
    int64_t expected_elems = 1;
    for (const int64_t dim : expected_shape) {
        expected_elems *= dim;
    }
    int64_t actual_elems = 1;
    for (const int64_t dim : metadata.shape) {
        actual_elems *= dim;
    }
    if (actual_elems != expected_elems) {
        throw std::runtime_error("tensor element count mismatch for " + std::string(name));
    }
}

void validate_weight_anchors(const VoxCPM1Assets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.model_weights;
    assets::require_tensor_shape(weights, "token_embd.weight", {config.lm.vocab_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "output_norm.weight", {config.lm.hidden_size});
    assets::require_tensor_shape(weights, "blk.0.attn_q.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "blk.0.attn_k.weight",
        {config.lm.num_key_value_heads * config.lm.kv_channels, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "blk.0.ffn_gate.weight", {config.lm.intermediate_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "residual_lm.output_norm.weight", {config.lm.hidden_size});
    require_vae_weight_v_shape(weights, "locenc.special_token", {1, 1, 1, config.encoder.hidden_dim}, true);
    assets::require_tensor_shape(weights, "locenc.in_proj.weight", {config.encoder.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "locenc.output_norm.weight", {config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "locdit.in_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "locdit.cond_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "locdit.out_proj.weight", {config.feat_dim, config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "locdit.output_norm.weight", {config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "fsq.in_proj.weight", {config.scalar_quantization_latent_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "fsq.out_proj.weight", {config.lm.hidden_size, config.scalar_quantization_latent_dim});
    assets::require_tensor_shape(weights, "proj.enc_to_lm.weight", {config.lm.hidden_size, config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "proj.lm_to_dit.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "proj.res_to_dit.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "stop.stop_proj.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "stop.stop_head.weight", {2, config.lm.hidden_size});

    const auto & vae = *assets.audiovae_weights;
    int64_t encoder_in_channels = config.audio_vae.encoder_dim;
    for (size_t i = 0; i < config.audio_vae.encoder_rates.size(); ++i) {
        encoder_in_channels *= 2;
    }
    require_vae_weight_v_shape(vae, "audio_vae.encoder.fc_mu.weight", {config.audio_vae.latent_dim, encoder_in_channels, 3}, true);
    assets::require_tensor_shape(vae, "audio_vae.encoder.fc_mu.bias", {config.audio_vae.latent_dim});
    require_vae_weight_v_shape(vae, "audio_vae.decoder.model.0.weight", {config.audio_vae.latent_dim, 1, 7}, true);
    require_vae_weight_v_shape(vae, "audio_vae.decoder.model.1.weight", {config.audio_vae.decoder_dim, config.audio_vae.latent_dim, 1}, true);
}

}

std::shared_ptr<const VoxCPM1Assets> load_voxcpm1_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<VoxCPM1Assets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("voxcpm1"));

    {
        auto raw_model_weights = out->resources.open_tensor_source("weights");

        bool has_tokenizer = VoxCPM1GgufTokenizer::has_tokenizer_metadata(*raw_model_weights);
        bool has_config = has_voxcpm1_config_metadata(*raw_model_weights);

        if (has_tokenizer && has_config) {
            out->config = load_voxcpm1_config_from_gguf(*raw_model_weights);
            out->config.v1 = true;
            out->gguf_tokenizer = std::make_shared<VoxCPM1GgufTokenizer>(raw_model_weights);
        } else {
            out->config = parse_config(out->resources);
            out->config.v1 = true;
        }
    }

    auto raw_model_weights = out->resources.open_tensor_source("weights");
    auto raw_audiovae_weights = out->resources.open_tensor_source("audiovae_weights");
    out->model_weights = std::make_shared<VoxCPM1TensorSource>(raw_model_weights, out->config);
    out->audiovae_weights = std::make_shared<VoxCPM1TensorSource>(raw_audiovae_weights, out->config);
    validate_weight_anchors(*out);
    return out;
}

}  // namespace engine::community_models::voxcpm1
