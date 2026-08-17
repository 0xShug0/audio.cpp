#include "engine/models/voxcpm2/assets.h"

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

namespace engine::models::voxcpm2 {
namespace json = engine::io::json;
namespace {

VoxCPM2RopeScalingConfig parse_rope_scaling(const json::Value & value) {
    VoxCPM2RopeScalingConfig config;
    config.type = json::optional_string(value, "type", "");
    config.long_factor = json::optional_f32_array(value, "long_factor");
    config.short_factor = json::optional_f32_array(value, "short_factor");
    config.original_max_position_embeddings =
        json::optional_i64(value, "original_max_position_embeddings", 0);
    return config;
}

VoxCPM2MiniCPMConfig parse_lm_config(const json::Value & value) {
    VoxCPM2MiniCPMConfig config;
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
            throw std::runtime_error("VoxCPM2 currently expects longrope rope_scaling");
        }
        const int64_t expected = config.hidden_size / config.num_attention_heads / 2;
        if (static_cast<int64_t>(config.rope_scaling.long_factor.size()) != expected ||
            static_cast<int64_t>(config.rope_scaling.short_factor.size()) != expected) {
            throw std::runtime_error("VoxCPM2 rope_scaling factor length does not match head_dim / 2");
        }
    }
    return config;
}

VoxCPM2LocalTransformerConfig parse_local_transformer_config(
    const json::Value & value,
    const char * label) {
    VoxCPM2LocalTransformerConfig config;
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

VoxCPM2DiTConfig parse_dit_config(const json::Value & value) {
    const auto base = parse_local_transformer_config(value, "dit transformer");
    VoxCPM2DiTConfig config;
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
        throw std::runtime_error("VoxCPM2 CFM currently expects euler solver");
    }
    if (config.cfm.t_scheduler != "log-norm") {
        throw std::runtime_error("VoxCPM2 CFM currently expects log-norm scheduler");
    }
    return config;
}

VoxCPM2AudioVAEConfig parse_audio_vae_config(const json::Value & value) {
    VoxCPM2AudioVAEConfig config;
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
        throw std::runtime_error("VoxCPM2 AudioVAE rates must be non-empty");
    }
    for (const auto rate : config.encoder_rates) {
        engine::io::require_positive(rate, "AudioVAE encoder rate");
    }
    for (const auto rate : config.decoder_rates) {
        engine::io::require_positive(rate, "AudioVAE decoder rate");
    }
    return config;
}

VoxCPM2Config parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    VoxCPM2Config config;
    config.architecture = json::require_string(root, "architecture");
    if (config.architecture != "voxcpm2" && config.architecture != "voxcpm") {
        throw std::runtime_error("VoxCPM config architecture mismatch: " + config.architecture);
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
        throw std::runtime_error("VoxCPM2 feat_dim must match AudioVAE latent_dim");
    }
    if (config.residual_lm_num_layers > config.lm.num_hidden_layers) {
        throw std::runtime_error("VoxCPM2 residual_lm_num_layers exceeds lm num_hidden_layers");
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

class TransformingTensorSource final : public assets::TensorSource {
public:
    TransformingTensorSource(
        std::shared_ptr<const assets::TensorSource> source,
        const VoxCPM2Config & config,
        bool is_v1)
        : source_(std::move(source)), config_(config), is_v1_(is_v1) {
        build_routes();
    }

    const std::filesystem::path & source_path() const noexcept override {
        return source_->source_path();
    }

    bool has_tensor(std::string_view name) const noexcept override {
        const std::string key{std::string(name)};
        if (routes_.find(key) != routes_.end() ||
            synthesized_tensors_.find(key) != synthesized_tensors_.end()) {
            return true;
        }
        if (is_v1_) {
            // v1 GGUF stores folded AudioVAE conv weights; the loader asks for
            // decomposed weight_v/weight_g names which we synthesize from the
            // folded tensors on demand.
            const auto base = folded_base_name(key);
            if (!base.empty() && folded_convs_.find(base) != folded_convs_.end()) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool is_synthesized(std::string_view name) const noexcept override {
        const std::string key{std::string(name)};
        return synthesized_tensors_.find(key) != synthesized_tensors_.end();
    }

    assets::TensorMetadata require_metadata(std::string_view name) const override {
        const auto it = synthesized_tensors_.find(std::string(name));
        if (it != synthesized_tensors_.end()) {
            return it->second;
        }
        const std::string key{std::string(name)};
        if (is_v1_) {
            const auto base = folded_base_name(key);
            if (!base.empty()) {
                const auto folded_it = folded_convs_.find(base);
                if (folded_it != folded_convs_.end()) {
                    auto metadata = source_->require_metadata(folded_it->second);
                    metadata.name = key;
                    if (has_suffix(key, ".weight_g") && !metadata.shape.empty()) {
                        metadata.shape = {metadata.shape.front(), 1, 1};
                    }
                    return metadata;
                }
            }
        }
        const auto route_it = routes_.find(key);
        if (route_it == routes_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        auto metadata = source_->require_metadata(route_it->second);
        metadata.name = key;
        // Apply shape transformations if needed
        if (reshape_map_.find(key) != reshape_map_.end()) {
            metadata.shape = reshape_map_.at(key);
        }
        return metadata;
    }

    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> out;
        out.reserve(routes_.size() + synthesized_tensors_.size());
        for (const auto & [name, route] : routes_) {
            out.push_back(require_metadata(name));
        }
        for (const auto & [name, metadata] : synthesized_tensors_) {
            out.push_back(metadata);
        }
        std::sort(out.begin(), out.end(),
            [](const assets::TensorMetadata & lhs, const assets::TensorMetadata & rhs) {
                return lhs.name < rhs.name;
            });
        return out;
    }

    void release_storage() const override { source_->release_storage(); }

    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto it = synthesized_tensors_.find(std::string(name));
        if (it != synthesized_tensors_.end()) {
            return generate_synthesized_tensor(name);
        }
        const std::string key{std::string(name)};
        if (is_v1_) {
            const auto base = folded_base_name(key);
            if (!base.empty() && folded_convs_.find(base) != folded_convs_.end()) {
                auto data = source_->require_tensor_data(folded_convs_.at(base));
                data.metadata.name = key;
                return data;
            }
        }
        const auto route_it = routes_.find(key);
        if (route_it == routes_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        auto data = source_->require_tensor_data(route_it->second);
        data.metadata.name = key;
        // Apply transformations
        if (reshape_map_.find(key) != reshape_map_.end()) {
            const auto & target_shape = reshape_map_.at(key);
            if (data.metadata.shape != target_shape) {
                // Reshape the data
                data = reshape_tensor_data(data, target_shape);
            }
        }
        return data;
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto it = synthesized_tensors_.find(std::string(name));
        if (it != synthesized_tensors_.end()) {
            return generate_synthesized_f32(name);
        }
        const std::string key{std::string(name)};
        if (is_v1_) {
            const auto base = folded_base_name(key);
            if (!base.empty()) {
                const auto folded_it = folded_convs_.find(base);
                if (folded_it != folded_convs_.end()) {
                    const auto folded = source_->require_f32(folded_it->second, std::nullopt);
                    if (has_suffix(key, ".weight_g")) {
                        return folded_weight_g(folded, folded_it->second, expected_shape);
                    }
                    return folded;
                }
            }
        }
        const auto route_it = routes_.find(key);
        if (route_it == routes_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        if (is_v1_ && expected_shape.has_value()) {
            const auto meta = source_->require_metadata(route_it->second);
            const int64_t expected_elems = checked_element_count("expected", *expected_shape);
            const int64_t actual_elems = checked_element_count(route_it->second, meta.shape);
            if (expected_elems == actual_elems && meta.shape != *expected_shape) {
                return source_->require_f32(route_it->second, std::nullopt);
            }
        }
        // Check if we need to reshape
        if (reshape_map_.find(key) != reshape_map_.end()) {
            const auto & target_shape = reshape_map_.at(key);
            if (expected_shape.has_value() && *expected_shape != target_shape) {
                // We'll fetch with target shape and then it will be validated
            }
            return source_->require_f32(route_it->second, target_shape);
        }
        return source_->require_f32(route_it->second, expected_shape);
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
        const auto it = synthesized_tensors_.find(std::string(name));
        if (it != synthesized_tensors_.end()) {
            const auto values = generate_synthesized_f32(name);
            engine::assets::set_backend_tensor_from_f32_parallel(tensor, name, values,
                make_tensor_shape(expected_shape),
                engine::assets::ggml_type_for_tensor_storage(storage_type));
            return;
        }
        const auto route_it = routes_.find(std::string(name));
        if (route_it == routes_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        // Check for weight norm decomposition (weight_v + weight_g)
        const std::string logical_name = std::string(name);
        if (weight_norm_map_.find(logical_name) != weight_norm_map_.end()) {
            const auto & wn = weight_norm_map_.at(logical_name);
            const auto weight_v = source_->require_f32(wn.weight_v_name, wn.weight_v_shape);
            const auto weight_g = source_->require_f32(wn.weight_g_name, wn.weight_g_shape);
            const auto folded = fold_weight_norm(weight_v, weight_g, wn.out_channels, wn.in_channels, wn.kernel_size);
            const auto shape = make_tensor_shape(expected_shape);
            const ggml_type type = engine::assets::ggml_type_for_tensor_storage(
                engine::assets::resolve_tensor_storage_type(*this, name, storage_type));
            engine::assets::set_backend_tensor_from_f32_parallel(tensor, name, folded, shape, type);
            return;
        }
        // Check for reshape
        if (reshape_map_.find(logical_name) != reshape_map_.end()) {
            const auto & target_shape = reshape_map_.at(logical_name);
            const auto values = source_->require_f32(route_it->second, target_shape);
            const auto shape = make_tensor_shape(expected_shape);
            const ggml_type type = engine::assets::ggml_type_for_tensor_storage(
                engine::assets::resolve_tensor_storage_type(*this, name, storage_type));
            engine::assets::set_backend_tensor_from_f32_parallel(tensor, name, values, shape, type);
            return;
        }
        // Special handling for V1 embedding weight: token_embd.weight is transposed in GGUF
        // V1 GGUF stores [hidden_size, vocab_size] but we need [vocab_size, hidden_size]
        if (is_v1_ && logical_name == "base_lm.embed_tokens.weight") {
            const auto source_values = source_->require_f32(route_it->second, std::nullopt);
            const auto source_meta = source_->require_metadata(route_it->second);
            if (source_meta.shape.size() == 2) {
                const int64_t src_rows = source_meta.shape[0];
                const int64_t src_cols = source_meta.shape[1];
                const int64_t dst_rows = expected_shape.size() > 0 ? expected_shape[0] : src_cols;
                const int64_t dst_cols = expected_shape.size() > 1 ? expected_shape[1] : src_rows;
                if (src_rows == dst_cols && src_cols == dst_rows) {
                    // Transpose the weight matrix
                    std::vector<float> transposed(static_cast<size_t>(dst_rows * dst_cols));
                    for (int64_t i = 0; i < src_rows; ++i) {
                        for (int64_t j = 0; j < src_cols; ++j) {
                            transposed[static_cast<size_t>(j * dst_rows + i)] = source_values[static_cast<size_t>(i * src_cols + j)];
                        }
                    }
                    const auto shape = make_tensor_shape(expected_shape);
                    const ggml_type type = engine::assets::ggml_type_for_tensor_storage(
                        engine::assets::resolve_tensor_storage_type(*this, name, storage_type));
                    engine::assets::set_backend_tensor_from_f32_parallel(tensor, name, transposed, shape, type);
                    return;
                }
            }
        }
        source_->set_backend_tensor(tensor, route_it->second, storage_type, expected_shape);
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
    struct WeightNormInfo {
        std::string weight_v_name;
        std::string weight_g_name;
        std::vector<int64_t> weight_v_shape;
        std::vector<int64_t> weight_g_shape;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel_size = 0;
    };

    void build_routes() {
        // V1 -> V2 tensor name mapping
        std::unordered_map<std::string, std::string> rename_map = {
            // LM embeddings
            {"token_embd.weight", "base_lm.embed_tokens.weight"},
            // LM blocks
            {"blk.", "base_lm.layers."},
            {"attn_q.weight", "self_attn.q_proj.weight"},
            {"attn_k.weight", "self_attn.k_proj.weight"},
            {"attn_v.weight", "self_attn.v_proj.weight"},
            {"attn_norm.weight", "input_layernorm.weight"},
            {"attn_output.weight", "self_attn.o_proj.weight"},
            {"ffn_norm.weight", "post_attention_layernorm.weight"},
            {"ffn_gate.weight", "mlp.gate_proj.weight"},
            {"ffn_up.weight", "mlp.up_proj.weight"},
            {"ffn_down.weight", "mlp.down_proj.weight"},
            // Output norm
            {"output_norm.weight", "base_lm.norm.weight"},
            // Residual LM
            {"residual_lm.blk.", "residual_lm.layers."},
            {"residual_lm.output_norm.weight", "residual_lm.norm.weight"},
            // Local encoder (feat_encoder)
            {"locenc.in_proj.weight", "feat_encoder.in_proj.weight"},
            {"locenc.in_proj.bias", "feat_encoder.in_proj.bias"},
            {"locenc.special_token", "feat_encoder.special_token"},
            {"locenc.blk.", "feat_encoder.encoder.layers."},
            {"locenc.output_norm.weight", "feat_encoder.encoder.norm.weight"},
            // Local DiT (feat_decoder)
            {"locdit.in_proj.weight", "feat_decoder.estimator.in_proj.weight"},
            {"locdit.in_proj.bias", "feat_decoder.estimator.in_proj.bias"},
            {"locdit.cond_proj.weight", "feat_decoder.estimator.cond_proj.weight"},
            {"locdit.cond_proj.bias", "feat_decoder.estimator.cond_proj.bias"},
            {"locdit.out_proj.weight", "feat_decoder.estimator.out_proj.weight"},
            {"locdit.out_proj.bias", "feat_decoder.estimator.out_proj.bias"},
            {"locdit.time_mlp.linear_1.weight", "feat_decoder.estimator.time_mlp.linear_1.weight"},
            {"locdit.time_mlp.linear_1.bias", "feat_decoder.estimator.time_mlp.linear_1.bias"},
            {"locdit.time_mlp.linear_2.weight", "feat_decoder.estimator.time_mlp.linear_2.weight"},
            {"locdit.time_mlp.linear_2.bias", "feat_decoder.estimator.time_mlp.linear_2.bias"},
            {"locdit.delta_time_mlp.linear_1.weight", "feat_decoder.estimator.delta_time_mlp.linear_1.weight"},
            {"locdit.delta_time_mlp.linear_1.bias", "feat_decoder.estimator.delta_time_mlp.linear_1.bias"},
            {"locdit.delta_time_mlp.linear_2.weight", "feat_decoder.estimator.delta_time_mlp.linear_2.weight"},
            {"locdit.delta_time_mlp.linear_2.bias", "feat_decoder.estimator.delta_time_mlp.linear_2.bias"},
            {"locdit.output_norm.weight", "feat_decoder.estimator.decoder.norm.weight"},
            {"locdit.blk.", "feat_decoder.estimator.decoder.layers."},
            // Projections
            {"proj.enc_to_lm.weight", "enc_to_lm_proj.weight"},
            {"proj.enc_to_lm.bias", "enc_to_lm_proj.bias"},
            {"proj.lm_to_dit.weight", "lm_to_dit_proj.weight"},
            {"proj.lm_to_dit.bias", "lm_to_dit_proj.bias"},
            {"proj.res_to_dit.weight", "res_to_dit_proj.weight"},
            {"proj.res_to_dit.bias", "res_to_dit_proj.bias"},
            // V1→V2 mapping for fusion_concat_proj (critical for V1 models with fusion)
            {"proj.fusion_concat.weight", "fusion_concat_proj.weight"},
            {"proj.fusion_concat.bias", "fusion_concat_proj.bias"},
            {"fusion_concat_proj.weight", "fusion_concat_proj.weight"},
            {"stop.stop_proj.weight", "stop_proj.weight"},
            {"stop.stop_proj.bias", "stop_proj.bias"},
            {"stop.stop_head.weight", "stop_head.weight"},
            // FSQ
            {"fsq.in_proj.weight", "fsq_layer.in_proj.weight"},
            {"fsq.in_proj.bias", "fsq_layer.in_proj.bias"},
            {"fsq.out_proj.weight", "fsq_layer.out_proj.weight"},
            {"fsq.out_proj.bias", "fsq_layer.out_proj.bias"},
            // Audio VAE (prefixed with audio_vae.)
            {"audio_vae.encoder.block.", "encoder.block."},
            {"audio_vae.encoder.fc_mu", "encoder.fc_mu"},
            {"audio_vae.decoder.model.", "decoder.model."},
            {"audio_vae.decoder.sr_cond_model.", "decoder.sr_cond_model."},
        };

        // Build routes by scanning source tensors
        for (const auto & tensor : source_->tensors()) {
            std::string v1_name = tensor.name;
            std::string v2_name = v1_name;

            // Apply prefix replacements
            for (const auto & [from, to] : rename_map) {
                if (v2_name.rfind(from, 0) == 0) {
                    v2_name = to + v2_name.substr(from.size());
                    break;
                }
            }

            // Handle blk.N.* -> layers.N.* (base LM, residual LM, locenc, locdit)
            constexpr std::string_view kBlk = "blk.";
            const size_t blk_pos = v1_name.find(kBlk);
            if (blk_pos != std::string::npos) {
                const size_t layer_start = blk_pos + kBlk.size();
                const size_t dot = v1_name.find('.', layer_start);
                if (dot != std::string::npos) {
                    const std::string layer_idx = v1_name.substr(layer_start, dot - layer_start);
                    const std::string rest = v1_name.substr(dot + 1);
                    if (v1_name.rfind("residual_lm.", 0) == 0) {
                        v2_name = "residual_lm.layers." + layer_idx + "." + rest;
                    } else if (v1_name.rfind("locenc.", 0) == 0) {
                        v2_name = "feat_encoder.encoder.layers." + layer_idx + "." + rest;
                    } else if (v1_name.rfind("locdit.", 0) == 0) {
                        v2_name = "feat_decoder.estimator.decoder.layers." + layer_idx + "." + rest;
                    } else {
                        v2_name = "base_lm.layers." + layer_idx + "." + rest;
                    }
                    // Further sub-replacements
                    for (const auto & [from, to] : rename_map) {
                        size_t pos = v2_name.find(from);
                        if (pos != std::string::npos) {
                            v2_name.replace(pos, from.size(), to);
                        }
                    }
                }
            }

            routes_[v2_name] = v1_name;
        }

        // Reshape map
        reshape_map_ = {
            // feat_quant: {N, F} -> {N, F, 1}
            // merge: {N, D} -> {N, D, 1}
            // downsample/upsample: {out, in} -> {out, in, k, k} (k=3 for 3x3)
        };

        // Folded AudioVAE conv weights: v1 GGUF stores weight-norm weights
        // already folded into a single `.weight` tensor, while the v2 loader
        // requests decomposed `.weight_v`/`.weight_g` names. Register every
        // audio_vae conv weight so those logical names resolve to the folded
        // data (weight_v) and its per-channel row norms (weight_g), which makes
        // the loader's fold_weight_norm an exact identity.
        if (is_v1_) {
            std::vector<std::pair<std::string, std::string>> folded;
            for (const auto & [logical, source] : routes_) {
                if (source.rfind("audio_vae.", 0) == 0 && has_suffix(logical, ".weight")) {
                    folded.emplace_back(
                        logical.substr(0, logical.size() - 7), source);
                }
            }
            for (const auto & [base, source] : folded) {
                folded_convs_[base] = source;
            }
        }

        // Synthesized tensors for V1
        const int64_t encoder_hidden = config_.encoder.hidden_dim;
        const int64_t feat_dim = config_.feat_dim;
        const int64_t lm_hidden = config_.lm.hidden_size;

        // feat_encoder.scale_embed (identity buckets)
        synthesized_tensors_["feat_encoder.scale_embed.weight"] =
            assets::TensorMetadata{"feat_encoder.scale_embed.weight", "F32", {32, encoder_hidden}};
        synthesized_tensors_["feat_encoder.bias_embed.weight"] =
            assets::TensorMetadata{"feat_encoder.bias_embed.weight", "F32", {32, encoder_hidden}};

        // feat_encoder.fc_logvar (zeros)
        synthesized_tensors_["feat_encoder.fc_logvar.weight"] =
            assets::TensorMetadata{"feat_encoder.fc_logvar.weight", "F32", {feat_dim, encoder_hidden}};

        // feat_encoder.diag (identity)
        synthesized_tensors_["feat_encoder.diag"] =
            assets::TensorMetadata{"feat_encoder.diag", "F32", {feat_dim}};

        // feat_encoder.special_token (from token_embd)
        synthesized_tensors_["feat_encoder.special_token"] =
            assets::TensorMetadata{"feat_encoder.special_token", "F32", {1, 1, 1, encoder_hidden}};

        // token_embd.extra_bias (from logit_scale or zeros)
        synthesized_tensors_["token_embd.extra_bias"] =
            assets::TensorMetadata{"token_embd.extra_bias", "F32", {lm_hidden}};

        // feat_encoder.merge (zeros)
        synthesized_tensors_["feat_encoder.merge.weight"] =
            assets::TensorMetadata{"feat_encoder.merge.weight", "F32", {encoder_hidden, feat_dim}};

        // Identity SR-condition embeddings for V1 decoder blocks. VoxCPM1
        // GGUFs contain no sr_cond_model tensors (no SR conditioning), but the
        // shared decoder loader requires scale_embed/bias_embed.
        {
            const auto & vae = config_.audio_vae;
            const size_t num_blocks = vae.decoder_rates.size();
            for (size_t i = 0; i < num_blocks; ++i) {
                const int64_t input_channels =
                    vae.decoder_dim / (int64_t{1} << static_cast<int>(i));
                const std::string prefix =
                    "decoder.sr_cond_model." + std::to_string(i + 2) + ".";
                synthesized_tensors_[prefix + "scale_embed.weight"] =
                    assets::TensorMetadata{prefix + "scale_embed.weight", "F32", {1, input_channels}};
                synthesized_tensors_[prefix + "bias_embed.weight"] =
                    assets::TensorMetadata{prefix + "bias_embed.weight", "F32", {1, input_channels}};
            }
        }

        // Missing projection weights for V1 (not in VoxCPM1 GGUF)
        synthesized_tensors_["fusion_concat_proj.weight"] =
            assets::TensorMetadata{"fusion_concat_proj.weight", "F32", {lm_hidden, lm_hidden * 2}};
        synthesized_tensors_["fusion_concat_proj.bias"] =
            assets::TensorMetadata{"fusion_concat_proj.bias", "F32", {lm_hidden}};
    }

    std::vector<float> fold_weight_norm(
        const std::vector<float> & weight_v,
        const std::vector<float> & weight_g,
        int64_t out_channels, int64_t in_channels, int64_t kernel_size) const {
        if (static_cast<int64_t>(weight_v.size()) != out_channels * in_channels * kernel_size ||
            static_cast<int64_t>(weight_g.size()) != out_channels) {
            throw std::runtime_error("VoxCPM1 weight-norm shape mismatch");
        }
        std::vector<float> out(weight_v.size(), 0.0F);
        for (int64_t d0 = 0; d0 < out_channels; ++d0) {
            const size_t base = static_cast<size_t>(d0 * in_channels * kernel_size);
            double norm_sq = 0.0;
            for (int64_t i = 0; i < in_channels * kernel_size; ++i) {
                const double value = weight_v[base + static_cast<size_t>(i)];
                norm_sq += value * value;
            }
            const float scale = weight_g[static_cast<size_t>(d0)] /
                                static_cast<float>(std::sqrt(norm_sq + 1e-8));
            for (int64_t i = 0; i < in_channels * kernel_size; ++i) {
                out[base + static_cast<size_t>(i)] = weight_v[base + static_cast<size_t>(i)] * scale;
            }
        }
        return out;
    }

    assets::RawTensorData reshape_tensor_data(const assets::RawTensorData & data,
                                               const std::vector<int64_t> & target_shape) const {
        // For now, just return the data as-is (validation happens elsewhere)
        // The actual reshape happens in require_f32
        return data;
    }

    assets::RawTensorData generate_synthesized_tensor(std::string_view name) const {
        const auto it = synthesized_tensors_.find(std::string(name));
        if (it == synthesized_tensors_.end()) {
            throw std::runtime_error("no synthesized tensor: " + std::string(name));
        }
        const auto & metadata = it->second;
        const int64_t num_elements = std::accumulate(metadata.shape.begin(), metadata.shape.end(), 1, std::multiplies<int64_t>());
        std::vector<std::byte> bytes(num_elements * sizeof(float));
        std::memset(bytes.data(), 0, bytes.size());
        return {metadata, std::move(bytes)};
    }

    std::vector<float> generate_synthesized_f32(std::string_view name) const {
        const auto it = synthesized_tensors_.find(std::string(name));
        if (it == synthesized_tensors_.end()) {
            throw std::runtime_error("no synthesized tensor: " + std::string(name));
        }
        const auto & metadata = it->second;
        const int64_t num_elements = std::accumulate(metadata.shape.begin(), metadata.shape.end(), 1, std::multiplies<int64_t>());
        if (name == "feat_encoder.diag") {
            std::vector<float> out(num_elements, 1.0F);
            return out;
        }
        if (std::string_view prefix = "decoder.sr_cond_model.";
            name.rfind(prefix, 0) == 0 && has_suffix(name, ".scale_embed.weight")) {
            return std::vector<float>(num_elements, 1.0F);
        }
        if (name == "feat_encoder.scale_embed.weight" || name == "feat_encoder.bias_embed.weight") {
            // Identity-like initialization
            std::vector<float> out(num_elements, 0.0F);
            // Fill with small values
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = 0.01F;
            }
            return out;
        }
        if (name == "fusion_concat_proj.weight") {
            // Xavier/Glorot initialization for fusion_concat_proj weight
            // shape is [lm_hidden, lm_hidden * 2]
            std::vector<float> out(num_elements);
            const float scale = std::sqrt(2.0f / (config_.lm.hidden_size + config_.lm.hidden_size * 2));
            for (size_t i = 0; i < out.size(); ++i) {
                // Simple uniform distribution in [-scale, scale]
                out[i] = (static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
            }
            return out;
        }
        return std::vector<float>(num_elements, 0.0F);
    }

    static bool has_suffix(std::string_view value, std::string_view suffix) {
        return value.size() >= suffix.size() &&
               value.substr(value.size() - suffix.size()) == suffix;
    }

    std::string folded_base_name(const std::string & key) const {
        constexpr std::string_view kWeightV = ".weight_v";
        constexpr std::string_view kWeightG = ".weight_g";
        if (has_suffix(key, kWeightV)) {
            return key.substr(0, key.size() - kWeightV.size());
        }
        if (has_suffix(key, kWeightG)) {
            return key.substr(0, key.size() - kWeightG.size());
        }
        return "";
    }

    std::vector<float> folded_weight_g(
        const std::vector<float> & folded,
        const std::string & folded_source_name,
        const std::optional<std::vector<int64_t>> & expected_shape) const {
        const auto meta = source_->require_metadata(folded_source_name);
        const int64_t groups = expected_shape.has_value() && !expected_shape->empty()
            ? expected_shape->front()
            : (meta.shape.empty() ? 0 : meta.shape.front());
        const int64_t rows = checked_element_count(folded_source_name, meta.shape);
        if (groups <= 0 || rows == 0 || rows % groups != 0) {
            throw std::runtime_error("folded weight_g shape mismatch: " + folded_source_name);
        }
        const int64_t inner = rows / groups;
        std::vector<float> out(static_cast<size_t>(groups), 0.0F);
        for (int64_t g = 0; g < groups; ++g) {
            double norm_sq = 0.0;
            for (int64_t i = 0; i < inner; ++i) {
                const float value = folded[static_cast<size_t>(g * inner + i)];
                norm_sq += static_cast<double>(value) * static_cast<double>(value);
            }
            out[static_cast<size_t>(g)] = static_cast<float>(std::sqrt(norm_sq));
        }
        return out;
    }

    static int64_t checked_element_count(std::string_view name, const std::vector<int64_t> & shape) {
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

    std::shared_ptr<const assets::TensorSource> source_;
    VoxCPM2Config config_;
    bool is_v1_;
    std::unordered_map<std::string, std::string> routes_;
    std::unordered_map<std::string, std::vector<int64_t>> reshape_map_;
    std::unordered_map<std::string, WeightNormInfo> weight_norm_map_;
    std::unordered_map<std::string, assets::TensorMetadata> synthesized_tensors_;
    std::unordered_map<std::string, std::string> folded_convs_;
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

void validate_weight_anchors(const VoxCPM2Assets & assets) {
    const auto & config = assets.config;
    const auto & weights = *assets.model_weights;
    assets::require_tensor_shape(weights, "base_lm.embed_tokens.weight", {config.lm.vocab_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.norm.weight", {config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.layers.0.self_attn.q_proj.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.layers.0.self_attn.k_proj.weight",
        {config.lm.num_key_value_heads * config.lm.kv_channels, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "base_lm.layers.0.mlp.gate_proj.weight", {config.lm.intermediate_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "residual_lm.norm.weight", {config.lm.hidden_size});
    assets::require_tensor_shape(weights, "feat_encoder.special_token", {1, 1, 1, config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "feat_encoder.in_proj.weight", {config.encoder.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "feat_encoder.encoder.norm.weight", {config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.in_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.cond_proj.weight", {config.dit.hidden_dim, config.feat_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.out_proj.weight", {config.feat_dim, config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "feat_decoder.estimator.decoder.norm.weight", {config.dit.hidden_dim});
    assets::require_tensor_shape(weights, "fsq_layer.in_proj.weight", {config.scalar_quantization_latent_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "fsq_layer.out_proj.weight", {config.lm.hidden_size, config.scalar_quantization_latent_dim});
    assets::require_tensor_shape(weights, "enc_to_lm_proj.weight", {config.lm.hidden_size, config.encoder.hidden_dim});
    assets::require_tensor_shape(weights, "lm_to_dit_proj.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "res_to_dit_proj.weight", {config.dit.hidden_dim, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "fusion_concat_proj.weight", {config.lm.hidden_size, config.lm.hidden_size * 2});
    assets::require_tensor_shape(weights, "stop_proj.weight", {config.lm.hidden_size, config.lm.hidden_size});
    assets::require_tensor_shape(weights, "stop_head.weight", {2, config.lm.hidden_size});

    const auto & vae = *assets.audiovae_weights;
    int64_t encoder_in_channels = config.audio_vae.encoder_dim;
    for (size_t i = 0; i < config.audio_vae.encoder_rates.size(); ++i) {
        encoder_in_channels *= 2;
    }
    require_vae_weight_v_shape(vae, "encoder.fc_mu.weight_v", {config.audio_vae.latent_dim, encoder_in_channels, 3}, config.v1);
    assets::require_tensor_shape(vae, "encoder.fc_mu.bias", {config.audio_vae.latent_dim});
    require_vae_weight_v_shape(vae, "decoder.model.0.weight_v", {config.audio_vae.latent_dim, 1, 7}, config.v1);
    require_vae_weight_v_shape(vae, "decoder.model.1.weight_v", {config.audio_vae.decoder_dim, config.audio_vae.latent_dim, 1}, config.v1);
}

}

std::shared_ptr<const VoxCPM2Assets> load_voxcpm2_assets(const std::filesystem::path & model_path, bool is_v1) {
    auto out = std::make_shared<VoxCPM2Assets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path(is_v1 ? "voxcpm1" : "voxcpm2"));
    out->config = parse_config(out->resources);
    out->config.v1 = is_v1;
    auto raw_model_weights = out->resources.open_tensor_source("weights");
    auto raw_audiovae_weights = out->resources.open_tensor_source("audiovae_weights");
    if (is_v1) {
        out->model_weights = std::make_shared<TransformingTensorSource>(raw_model_weights, out->config, true);
        out->audiovae_weights = std::make_shared<TransformingTensorSource>(raw_audiovae_weights, out->config, true);
    } else {
        out->model_weights = raw_model_weights;
        out->audiovae_weights = raw_audiovae_weights;
    }
    validate_weight_anchors(*out);
    return out;
}

}  // namespace engine::models::voxcpm2
