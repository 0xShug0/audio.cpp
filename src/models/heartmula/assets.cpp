#include "engine/models/heartmula/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::heartmula {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("HeartMuLa model path does not exist: " + model_path.string());
}

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"tokenizer_json", "tokenizer.json", true},
        {"generation_config", "gen_config.json", true},
        {"mula_config", "HeartMuLa-oss-3B/config.json", true},
        {"mula_index", "HeartMuLa-oss-3B/model.safetensors.index.json", true},
        {"codec_config", "HeartCodec-oss/config.json", true},
        {"codec_index", "HeartCodec-oss/model.safetensors.index.json", true},
    });
    return resources;
}

void require_string_value(const std::string & actual, const char * expected, const char * label) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string("HeartMuLa config ") + label + " mismatch: expected " + expected + ", got " + actual);
    }
}

void require_equal(int64_t actual, int64_t expected, const char * label) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string("HeartMuLa config ") + label + " mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

void require_non_empty(const std::vector<int64_t> & values, const char * label) {
    if (values.empty()) {
        throw std::runtime_error(std::string("HeartMuLa config contains empty ") + label);
    }
}

int64_t pow2_i64(size_t exponent) {
    int64_t value = 1;
    for (size_t i = 0; i < exponent; ++i) {
        value *= 2;
    }
    return value;
}

int64_t llama_mlp_hidden_dim(int64_t dim) {
    const int64_t multiple_of = 256;
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    return multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
}

HeartMuLaTransformerConfig transformer_flavor_config(const std::string & flavor) {
    HeartMuLaTransformerConfig config;
    config.flavor = flavor;
    if (flavor == "llama-3B") {
        config.num_layers = 28;
        config.num_heads = 24;
        config.num_kv_heads = 8;
        config.embed_dim = 3072;
        config.max_seq_len = 8192;
        config.intermediate_dim = 8192;
    } else if (flavor == "llama-300M") {
        config.num_layers = 3;
        config.num_heads = 8;
        config.num_kv_heads = 4;
        config.embed_dim = 3072;
        config.max_seq_len = 2048;
        config.intermediate_dim = 8192;
    } else {
        throw std::runtime_error("HeartMuLa unsupported transformer flavor: " + flavor);
    }
    engine::io::require_divisible(config.embed_dim, config.num_heads, (flavor + " hidden/head").c_str());
    engine::io::require_divisible(config.num_heads, config.num_kv_heads, (flavor + " grouped query heads").c_str());
    config.head_dim = config.embed_dim / config.num_heads;
    return config;
}

HeartMuLaConfig parse_mula_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("mula_config");
    HeartMuLaConfig config;
    config.model_type = json::optional_string(root, "model_type", "");
    require_string_value(config.model_type, "heartmula", "model_type");
    config.torch_dtype = json::optional_string(root, "torch_dtype", config.torch_dtype);
    require_string_value(config.torch_dtype, "float32", "torch_dtype");
    config.backbone_flavor = json::require_string(root, "backbone_flavor");
    config.decoder_flavor = json::require_string(root, "decoder_flavor");
    config.text_vocab_size = json::require_i64(root, "text_vocab_size");
    config.audio_vocab_size = json::require_i64(root, "audio_vocab_size");
    config.audio_num_codebooks = json::require_i64(root, "audio_num_codebooks");
    config.muq_dim = json::require_i64(root, "muq_dim");
    config.backbone = transformer_flavor_config(config.backbone_flavor);
    config.decoder = transformer_flavor_config(config.decoder_flavor);

    engine::io::require_positive(config.text_vocab_size, "text_vocab_size");
    engine::io::require_positive(config.audio_vocab_size, "audio_vocab_size");
    engine::io::require_positive(config.audio_num_codebooks, "audio_num_codebooks");
    engine::io::require_positive(config.muq_dim, "muq_dim");
    if (config.audio_num_codebooks < 2) {
        throw std::runtime_error("HeartMuLa audio_num_codebooks must be at least 2");
    }
    return config;
}

HeartMuLaGenerationConfig parse_generation_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("generation_config");
    HeartMuLaGenerationConfig config;
    config.text_bos_id = json::require_i64(root, "text_bos_id");
    config.text_eos_id = json::require_i64(root, "text_eos_id");
    config.audio_eos_id = json::require_i64(root, "audio_eos_id");
    config.empty_id = json::require_i64(root, "empty_id");
    engine::io::require_positive(config.text_bos_id, "text_bos_id");
    engine::io::require_positive(config.text_eos_id, "text_eos_id");
    engine::io::require_positive(config.audio_eos_id, "audio_eos_id");
    return config;
}

HeartCodecConfig parse_codec_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("codec_config");
    HeartCodecConfig config;
    config.model_type = json::optional_string(root, "model_type", "");
    require_string_value(config.model_type, "heartcodec", "codec model_type");
    config.torch_dtype = json::optional_string(root, "torch_dtype", config.torch_dtype);
    require_string_value(config.torch_dtype, "float32", "codec torch_dtype");
    config.dim = json::require_i64(root, "dim");
    config.codebook_size = json::require_i64(root, "codebook_size");
    config.decay = json::optional_f32(root, "decay", config.decay);
    config.commitment_weight = json::optional_f32(root, "commitment_weight", config.commitment_weight);
    config.threshold_ema_dead_code = json::require_i64(root, "threshold_ema_dead_code");
    config.use_cosine_sim = json::optional_bool(root, "use_cosine_sim", config.use_cosine_sim);
    config.codebook_dim = json::require_i64(root, "codebook_dim");
    config.num_quantizers = json::require_i64(root, "num_quantizers");
    config.attention_head_dim = json::require_i64(root, "attention_head_dim");
    config.in_channels = json::require_i64(root, "in_channels");
    config.norm_type = json::require_string(root, "norm_type");
    config.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.num_layers = json::require_i64(root, "num_layers");
    config.num_layers_2 = json::require_i64(root, "num_layers_2");
    config.out_channels = json::require_i64(root, "out_channels");
    config.num_bands = json::require_i64(root, "num_bands");
    config.sample_rate = static_cast<int>(json::require_i64(root, "sample_rate"));
    config.causal = json::optional_bool(root, "causal", config.causal);
    config.num_samples = json::require_i64(root, "num_samples");
    config.downsample_factors = json::require_i64_array(root, "downsample_factors");
    config.downsample_kernel_sizes = json::require_i64_array(root, "downsample_kernel_sizes");
    config.upsample_factors = json::require_i64_array(root, "upsample_factors");
    config.upsample_kernel_sizes = json::require_i64_array(root, "upsample_kernel_sizes");
    config.latent_hidden_dim = json::require_i64(root, "latent_hidden_dim");
    config.default_kernel_size = json::require_i64(root, "default_kernel_size");
    config.delay_kernel_size = json::require_i64(root, "delay_kernel_size");
    config.init_channel = json::require_i64(root, "init_channel");
    config.res_kernel_size = json::require_i64(root, "res_kernel_size");

    engine::io::require_positive(config.dim, "codec dim");
    engine::io::require_positive(config.codebook_size, "codec codebook_size");
    engine::io::require_positive(config.decay, "codec decay");
    engine::io::require_positive(config.commitment_weight, "codec commitment_weight");
    engine::io::require_positive(config.threshold_ema_dead_code, "codec threshold_ema_dead_code");
    engine::io::require_positive(config.codebook_dim, "codec codebook_dim");
    engine::io::require_positive(config.num_quantizers, "codec num_quantizers");
    engine::io::require_positive(config.attention_head_dim, "codec attention_head_dim");
    engine::io::require_positive(config.in_channels, "codec in_channels");
    engine::io::require_positive(config.num_attention_heads, "codec num_attention_heads");
    engine::io::require_positive(config.num_layers, "codec num_layers");
    engine::io::require_positive(config.num_layers_2, "codec num_layers_2");
    engine::io::require_positive(config.out_channels, "codec out_channels");
    engine::io::require_positive(config.num_bands, "codec num_bands");
    engine::io::require_positive(config.sample_rate, "codec sample_rate");
    engine::io::require_positive(config.num_samples, "codec num_samples");
    engine::io::require_positive(config.latent_hidden_dim, "codec latent_hidden_dim");
    engine::io::require_positive(config.default_kernel_size, "codec default_kernel_size");
    engine::io::require_positive(config.delay_kernel_size, "codec delay_kernel_size");
    engine::io::require_positive(config.init_channel, "codec init_channel");
    engine::io::require_positive(config.res_kernel_size, "codec res_kernel_size");
    require_non_empty(config.downsample_factors, "downsample_factors");
    require_non_empty(config.downsample_kernel_sizes, "downsample_kernel_sizes");
    require_non_empty(config.upsample_factors, "upsample_factors");
    require_non_empty(config.upsample_kernel_sizes, "upsample_kernel_sizes");
    engine::io::require_all_positive(config.downsample_factors, "downsample_factors");
    engine::io::require_all_positive(config.downsample_kernel_sizes, "downsample_kernel_sizes");
    engine::io::require_all_positive(config.upsample_factors, "upsample_factors");
    engine::io::require_all_positive(config.upsample_kernel_sizes, "upsample_kernel_sizes");
    require_equal(
        static_cast<int64_t>(config.downsample_factors.size()),
        static_cast<int64_t>(config.downsample_kernel_sizes.size()),
        "downsample factor/kernel count");
    require_equal(
        static_cast<int64_t>(config.upsample_factors.size()),
        static_cast<int64_t>(config.upsample_kernel_sizes.size()),
        "upsample factor/kernel count");
    require_string_value(config.norm_type, "ada_norm_single", "codec norm_type");
    if (!config.causal) {
        throw std::runtime_error("HeartCodec config must be causal");
    }
    return config;
}

void fill_paths(
    HeartMuLaAssetPaths & paths,
    const assets::ResourceBundle & resources) {
    paths.model_root = resources.model_root();
    paths.tokenizer_json_path = resources.require_file("tokenizer_json");
    paths.generation_config_path = resources.require_file("generation_config");
    paths.mula_config_path = resources.require_file("mula_config");
    paths.mula_index_path = resources.require_file("mula_index");
    paths.codec_config_path = resources.require_file("codec_config");
    paths.codec_index_path = resources.require_file("codec_index");
    paths.mula_shard_paths = engine::assets::indexed_tensor_source_shard_paths(
        paths.mula_index_path,
        paths.mula_index_path.parent_path());
    paths.codec_shard_paths = engine::assets::indexed_tensor_source_shard_paths(
        paths.codec_index_path,
        paths.codec_index_path.parent_path());
}

void validate_mula_weight_anchors(const HeartMuLaAssets & assets) {
    const auto & config = assets.mula_config;
    const auto & weights = *assets.mula_weights;
    const auto & backbone = config.backbone;
    const auto & decoder = config.decoder;
    assets::require_tensor_shape(weights, "text_embeddings.weight", {config.text_vocab_size, backbone.embed_dim});
    assets::require_tensor_shape(
        weights,
        "audio_embeddings.weight",
        {config.audio_vocab_size * config.audio_num_codebooks, backbone.embed_dim});
    assets::require_tensor_shape(weights, "unconditional_text_embedding.weight", {1, backbone.embed_dim});
    assets::require_tensor_shape(weights, "projection.weight", {decoder.embed_dim, backbone.embed_dim});
    assets::require_tensor_shape(weights, "codebook0_head.weight", {config.audio_vocab_size, backbone.embed_dim});
    assets::require_tensor_shape(
        weights,
        "audio_head",
        {config.audio_num_codebooks - 1, decoder.embed_dim, config.audio_vocab_size});
    assets::require_tensor_shape(weights, "muq_linear.weight", {backbone.embed_dim, config.muq_dim});
    assets::require_tensor_shape(weights, "muq_linear.bias", {backbone.embed_dim});
    assets::require_tensor_shape(weights, "backbone.norm.scale", {backbone.embed_dim});
    assets::require_tensor_shape(weights, "decoder.norm.scale", {decoder.embed_dim});
    assets::require_tensor_shape(weights, "backbone.layers.0.attn.q_proj.weight", {backbone.embed_dim, backbone.embed_dim});
    assets::require_tensor_shape(
        weights,
        "backbone.layers.0.attn.k_proj.weight",
        {backbone.num_kv_heads * backbone.head_dim, backbone.embed_dim});
    assets::require_tensor_shape(
        weights,
        "backbone.layers.0.attn.v_proj.weight",
        {backbone.num_kv_heads * backbone.head_dim, backbone.embed_dim});
    assets::require_tensor_shape(
        weights,
        "backbone.layers.0.attn.output_proj.weight",
        {backbone.embed_dim, backbone.embed_dim});
    assets::require_tensor_shape(weights, "backbone.layers.0.mlp.w1.weight", {backbone.intermediate_dim, backbone.embed_dim});
    assets::require_tensor_shape(weights, "backbone.layers.0.mlp.w2.weight", {backbone.embed_dim, backbone.intermediate_dim});
    assets::require_tensor_shape(weights, "backbone.layers.0.mlp.w3.weight", {backbone.intermediate_dim, backbone.embed_dim});
    assets::require_tensor_shape(weights, "backbone.layers.0.sa_norm.scale", {backbone.embed_dim});
    assets::require_tensor_shape(weights, "backbone.layers.0.mlp_norm.scale", {backbone.embed_dim});
    assets::require_tensor_shape(weights, "decoder.layers.0.attn.q_proj.weight", {decoder.embed_dim, decoder.embed_dim});
    assets::require_tensor_shape(
        weights,
        "decoder.layers.0.attn.k_proj.weight",
        {decoder.num_kv_heads * decoder.head_dim, decoder.embed_dim});
    assets::require_tensor_shape(
        weights,
        "decoder.layers.0.attn.v_proj.weight",
        {decoder.num_kv_heads * decoder.head_dim, decoder.embed_dim});
    assets::require_tensor_shape(
        weights,
        "decoder.layers.0.attn.output_proj.weight",
        {decoder.embed_dim, decoder.embed_dim});
    assets::require_tensor_shape(weights, "decoder.layers.0.mlp.w1.weight", {decoder.intermediate_dim, decoder.embed_dim});
    assets::require_tensor_shape(weights, "decoder.layers.0.mlp.w2.weight", {decoder.embed_dim, decoder.intermediate_dim});
    assets::require_tensor_shape(weights, "decoder.layers.0.mlp.w3.weight", {decoder.intermediate_dim, decoder.embed_dim});
    assets::require_tensor_shape(weights, "decoder.layers.0.sa_norm.scale", {decoder.embed_dim});
    assets::require_tensor_shape(weights, "decoder.layers.0.mlp_norm.scale", {decoder.embed_dim});
}

void validate_codec_weight_anchors(const HeartMuLaAssets & assets) {
    const auto & config = assets.codec_config;
    const auto & weights = *assets.codec_weights;
    const auto estimator_dim = config.num_attention_heads * config.attention_head_dim;
    const auto estimator_mlp_dim = llama_mlp_hidden_dim(estimator_dim);
    const auto decoder_in_channels = config.init_channel * pow2_i64(config.upsample_factors.size());
    const auto first_decoder_out_channels = decoder_in_channels / 2;
    assets::require_tensor_shape(weights, "flow_matching.cond_feature_emb.weight", {config.dim, config.dim});
    assets::require_tensor_shape(weights, "flow_matching.cond_feature_emb.bias", {config.dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.vq_embed.layers.0._codebook.embed",
        {1, config.codebook_size, config.codebook_dim});
    assets::require_tensor_shape(weights, "flow_matching.zero_cond_embedding1", {config.dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.proj_in.ffn_1.weight",
        {estimator_dim, config.in_channels, 3});
    assets::require_tensor_shape(weights, "flow_matching.estimator.proj_in.ffn_1.bias", {estimator_dim});
    assets::require_tensor_shape(weights, "flow_matching.estimator.proj_in.ffn_2.weight", {estimator_dim, estimator_dim});
    assets::require_tensor_shape(weights, "flow_matching.estimator.scale_shift_table", {2, estimator_dim});
    assets::require_tensor_shape(weights, "flow_matching.estimator.scale_shift_table_2", {2, estimator_dim * 2});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.attn.q_proj.weight",
        {estimator_dim, estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.attn.k_proj.weight",
        {estimator_dim, estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.attn.v_proj.weight",
        {estimator_dim, estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.attn.o_proj.weight",
        {estimator_dim, estimator_dim});
    assets::require_tensor_shape(weights, "flow_matching.estimator.transformer_blocks.0.attn_norm.weight", {estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.mlp.gate.weight",
        {estimator_mlp_dim, estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.mlp.up.weight",
        {estimator_mlp_dim, estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.mlp.down.weight",
        {estimator_dim, estimator_mlp_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.transformer_blocks.0.scale_shift_table",
        {6, estimator_dim});
    assets::require_tensor_shape(
        weights,
        "flow_matching.estimator.proj_out.ffn_2.weight",
        {config.out_channels, config.out_channels});
    assets::require_tensor_shape(
        weights,
        "scalar_model.encoder.0.parametrizations.weight.original1",
        {config.init_channel, config.num_bands, config.default_kernel_size});
    assets::require_tensor_shape(
        weights,
        "scalar_model.encoder.1.conv.weight",
        {config.init_channel, config.init_channel, config.default_kernel_size});
    assets::require_tensor_shape(
        weights,
        "scalar_model.decoder.0.parametrizations.weight.original1",
        {decoder_in_channels, config.latent_hidden_dim, config.delay_kernel_size});
    assets::require_tensor_shape(
        weights,
        "scalar_model.decoder.1.up_conv.layer.parametrizations.weight.original1",
        {decoder_in_channels, first_decoder_out_channels, config.upsample_kernel_sizes.front()});
}

}  // namespace

HeartMuLaAssetPaths resolve_heartmula_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    HeartMuLaAssetPaths paths;
    fill_paths(paths, resources);
    return paths;
}

std::shared_ptr<const HeartMuLaAssets> load_heartmula_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    HeartMuLaAssets assets;
    fill_paths(assets.paths, resources);
    assets.mula_config = parse_mula_config(resources);
    assets.generation_config = parse_generation_config(resources);
    assets.codec_config = parse_codec_config(resources);
    assets.mula_weights = engine::assets::open_indexed_tensor_source(
        assets.paths.mula_index_path,
        assets.paths.mula_index_path.parent_path());
    assets.codec_weights = engine::assets::open_indexed_tensor_source(
        assets.paths.codec_index_path,
        assets.paths.codec_index_path.parent_path());
    validate_mula_weight_anchors(assets);
    validate_codec_weight_anchors(assets);
    return std::make_shared<HeartMuLaAssets>(std::move(assets));
}

}  // namespace engine::models::heartmula
