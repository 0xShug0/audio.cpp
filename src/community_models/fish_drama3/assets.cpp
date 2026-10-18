#include "engine/community_models/fish_drama3/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/model_spec/package.h"

#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_drama3 {
namespace json = engine::io::json;
namespace {

// Drama 3 packs Q, K, V into consecutive rows. Present the split names expected
// by FishDacCodecRuntime without changing that shared codec or its decode graph.
class Drama3CodecSource final : public assets::TensorSource {
public:
    explicit Drama3CodecSource(std::shared_ptr<const assets::TensorSource> source) : source_(std::move(source)) {
        if (!source_) throw std::invalid_argument("Drama 3 codec source is null");
        const std::string suffix = ".attention.wqkv.weight";
        const char * projections[] = {"q_proj", "k_proj", "v_proj"};
        for (const auto & tensor : source_->tensors()) {
            if (tensor.name.size() < suffix.size() ||
                tensor.name.compare(tensor.name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
            const auto prefix = tensor.name.substr(0, tensor.name.size() - suffix.size()) + ".attention.";
            int split_count = 0;
            for (const auto * projection : projections) split_count += source_->has_tensor(prefix + projection + ".weight");
            if (split_count == 3) continue;
            if (split_count != 0) throw std::runtime_error("Drama 3 codec has incomplete split QKV weights: " + prefix);
            if (tensor.shape.size() != 2 || tensor.shape[1] <= 0 || tensor.shape[0] % 3 != 0 ||
                tensor.shape[0] / 3 != tensor.shape[1]) {
                throw std::runtime_error("Drama 3 codec packed QKV shape is invalid: " + tensor.name);
            }
            for (size_t part = 0; part < 3; ++part) {
                auto metadata = tensor;
                metadata.name = prefix + projections[part] + ".weight";
                metadata.shape[0] /= 3;
                slices_.emplace(metadata.name, Slice{tensor.name, part, metadata});
            }
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return source_->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override {
        return slices_.find(name) != slices_.end() || source_->has_tensor(name);
    }
    assets::TensorMetadata require_metadata(std::string_view name) const override {
        const auto found = slices_.find(name);
        return found == slices_.end() ? source_->require_metadata(name) : found->second.metadata;
    }
    std::vector<assets::TensorMetadata> tensors() const override {
        auto result = source_->tensors();
        for (const auto & slice : slices_) result.push_back(slice.second.metadata);
        return result;
    }
    void release_storage() const override { source_->release_storage(); }
    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto found = slices_.find(name);
        if (found == slices_.end()) return source_->require_tensor_data(name);
        const auto & slice = found->second;
        const auto packed = source_->require_tensor_data(slice.packed_name);
        const auto columns = slice.metadata.shape[1];
        const auto type = assets::ggml_type_for_tensor_dtype(slice.metadata.dtype);
        if (columns % ggml_blck_size(type) != 0) throw std::runtime_error("Drama 3 codec QKV row is not block aligned");
        const auto row_bytes = ggml_row_size(type, columns);
        const auto rows = static_cast<size_t>(slice.metadata.shape[0]);
        if (rows > std::numeric_limits<size_t>::max() / row_bytes / 3 || packed.bytes.size() != rows * row_bytes * 3) {
            throw std::runtime_error("Drama 3 codec packed QKV byte count is invalid: " + slice.packed_name);
        }
        const auto count = rows * row_bytes;
        const auto begin = packed.bytes.begin() + slice.part * count;
        return {slice.metadata, {begin, begin + count}};
    }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const override {
        const auto found = slices_.find(name);
        if (found == slices_.end()) return source_->require_f32(name, expected_shape);
        const auto & metadata = found->second.metadata;
        if (expected_shape && *expected_shape != metadata.shape) throw std::runtime_error("Drama 3 codec QKV shape mismatch");
        const auto data = require_tensor(name, assets::TensorStorageType::Native, metadata.shape);
        return assets::tensor_data_to_f32(name, data);
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override { return source_->require_i64_scalar(name); }

private:
    struct Slice {
        std::string packed_name;
        size_t part;
        assets::TensorMetadata metadata;
    };
    std::shared_ptr<const assets::TensorSource> source_;
    std::map<std::string, Slice, std::less<>> slices_;
};

}  // namespace

std::shared_ptr<const assets::TensorSource> make_drama3_codec_source(std::shared_ptr<const assets::TensorSource> source) {
    return std::make_shared<Drama3CodecSource>(std::move(source));
}

std::shared_ptr<const Drama3Assets> load_drama3_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<Drama3Assets>();
    out->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "fish_drama3");
    const auto root = out->resources.parse_json("config");
    if (json::require_string(root, "model_type") != "fish_omni" ||
        json::require_string(root, "speech_template_version") != "reference-bracket-v1") {
        throw std::runtime_error("Fish Drama 3 requires a fish_omni reference-bracket-v1 checkpoint");
    }
    const auto & text = root.require("text_config");
    auto & b = out->backbone;
    b.vocab_size = json::require_i64(text, "vocab_size");
    b.hidden_size = json::require_i64(text, "hidden_size");
    b.intermediate_size = json::require_i64(text, "moe_intermediate_size");
    if (json::require_i64(text, "shared_expert_intermediate_size") != b.intermediate_size) {
        throw std::runtime_error("Drama 3 shared and routed expert widths must match");
    }
    b.experts = json::require_i64(text, "num_experts");
    b.experts_per_token = json::require_i64(text, "num_experts_per_tok");
    b.layers = json::require_i64(text, "num_hidden_layers");
    b.heads = json::require_i64(text, "num_attention_heads");
    b.kv_heads = json::require_i64(text, "num_key_value_heads");
    b.head_dim = json::require_i64(text, "head_dim");
    b.linear_conv_kernel_dim = json::require_i64(text, "linear_conv_kernel_dim");
    b.linear_key_head_dim = json::require_i64(text, "linear_key_head_dim");
    b.linear_value_head_dim = json::require_i64(text, "linear_value_head_dim");
    b.linear_num_key_heads = json::require_i64(text, "linear_num_key_heads");
    b.linear_num_value_heads = json::require_i64(text, "linear_num_value_heads");
    b.rms_norm_eps = json::require_f32(text, "rms_norm_eps");
    b.rope_theta = json::require_f32(text.require("rope_parameters"), "rope_theta");
    b.partial_rotary_factor = json::require_f32(text.require("rope_parameters"), "partial_rotary_factor");
    b.tie_word_embeddings = json::optional_bool(text, "tie_word_embeddings", false);
    const auto & layer_types = text.require("layer_types").as_array();
    if (layer_types.size() != static_cast<size_t>(b.layers)) throw std::runtime_error("Drama 3 layer type count mismatch");
    for (int64_t layer = 0; layer < b.layers; ++layer) {
        const auto expected = (layer + 1) % 4 == 0 ? "full_attention" : "linear_attention";
        if (layer_types[layer].as_string() != expected) throw std::runtime_error("unsupported Drama 3 hybrid layer layout");
    }
    if (b.experts <= 0 || b.experts_per_token <= 0 || b.experts_per_token > b.experts ||
        !json::optional_bool(text, "norm_topk_prob", true) || !json::require_bool(text, "layernorm_zero_centered_gamma")) {
        throw std::runtime_error("unsupported Drama 3 expert or normalization configuration");
    }
    const auto & audio = root.require("audio_decoder_config");
    auto & t = out->tower;
    t.hidden_size = json::require_i64(audio, "dim");
    t.intermediate_size = json::require_i64(audio, "intermediate_size");
    t.heads = json::require_i64(audio, "n_head");
    t.kv_heads = json::require_i64(audio, "n_local_heads");
    t.head_dim = json::require_i64(audio, "head_dim");
    t.codebook_size = json::require_i64(audio, "vocab_size");
    t.shared_layers = json::require_i64(audio, "shared_enrichment_layers");
    t.residual_layers = json::require_i64(audio, "num_codebooks") - 1;
    t.backbone_codebooks = json::require_i64(audio, "vq_backbone_codebooks");
    t.norm_eps = json::require_f32(audio, "norm_eps");
    t.rope_theta = json::require_f32(audio, "rope_base");
    t.rotary_fraction = json::require_f32(audio, "rotary_percent");
    t.semantic_begin = json::require_i32(root, "semantic_start_token_id");
    out->eos = json::require_i32(root, "eos_token_id");
    if (t.hidden_size != b.hidden_size || t.residual_layers != 9 ||
        json::require_i64(root, "semantic_end_token_id") + 1 - t.semantic_begin != t.codebook_size ||
        !json::require_bool(audio, "rotary_interleaved") ||
        !json::require_bool(audio, "layernorm_zero_centered_gamma") ||
        !json::require_bool(audio, "attention_output_gate") ||
        json::require_string(audio, "decoder_type") != "codebook_mtp_shared") {
        throw std::runtime_error("unsupported Drama 3 tower or semantic range");
    }
    b.lm_head_token_ids.push_back(out->eos);
    for (int32_t token = 0; token < t.codebook_size; ++token)
        b.lm_head_token_ids.push_back(t.semantic_begin + token);
    out->model_weights = out->resources.open_tensor_source("model_weights");
    out->codec_weights = make_drama3_codec_source(out->resources.open_tensor_source("codec_weights"));
    (void) out->model_weights->require_metadata("model.language_model.layers.0.mlp.experts.gate_proj.weight");
    (void) out->model_weights->require_metadata("audio_decoder_mtp_shared.layers.0.eh_proj.weight");
    const auto & residual_codebook = out->codec_weights->require_metadata("quantizer.quantizer.quantizers.0.codebook.weight");
    out->codec.residual_codebook_size = residual_codebook.shape[0];
    return out;
}

}  // namespace engine::models::fish_drama3
