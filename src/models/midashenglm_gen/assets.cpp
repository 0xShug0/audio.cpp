#include "engine/models/midashenglm_gen/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>

namespace engine::models::midashenglm_gen {
namespace {

namespace json = engine::io::json;

MiDashengLmGenConfig parse_config(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("MiDashengLM-Gen config.json not found: " + path.string());
    }
    const auto root = json::parse_file(path);
    MiDashengLmGenConfig config;
    config.hidden_size = json::optional_i64(root, "llm_emb_dim", config.hidden_size);
    config.audio_embedding_size = json::optional_i64(root, "audio_emb_size", config.audio_embedding_size);
    config.target_embedding_size = json::optional_i64(root, "target_emb_dim", config.target_embedding_size);
    config.sequence_length = json::optional_i64(root, "seq_len", config.sequence_length);
    config.flow_hidden_size = json::optional_i64(root, "flow_dit_width", config.flow_hidden_size);
    config.flow_depth = json::optional_i64(root, "flow_dit_depth", config.flow_depth);
    config.flow_heads = json::optional_i64(root, "flow_dit_heads", config.flow_heads);
    const float flow_mlp_ratio = json::optional_f32(root, "flow_dit_mlp_ratio", 4.0F);
    config.flow_intermediate_size =
        static_cast<int64_t>(static_cast<float>(config.flow_hidden_size) * flow_mlp_ratio);
    config.patch_size = 5;
    config.block_size = 640 * config.patch_size;
    config.sample_rate = json::optional_i64(root, "sample_rate", config.sample_rate);
    engine::io::require_positive(config.hidden_size, "MiDashengLM-Gen hidden size");
    engine::io::require_positive(config.audio_embedding_size, "MiDashengLM-Gen audio embedding size");
    engine::io::require_positive(config.target_embedding_size, "MiDashengLM-Gen target embedding size");
    engine::io::require_positive(config.sequence_length, "MiDashengLM-Gen sequence length");
    engine::io::require_positive(config.flow_hidden_size, "MiDashengLM-Gen flow hidden size");
    engine::io::require_positive(config.flow_depth, "MiDashengLM-Gen flow depth");
    engine::io::require_positive(config.flow_heads, "MiDashengLM-Gen flow head count");
    engine::io::require_positive(config.flow_intermediate_size, "MiDashengLM-Gen flow intermediate size");
    engine::io::require_positive(config.sample_rate, "MiDashengLM-Gen sample rate");
    return config;
}

void validate_tensor_shapes(const engine::assets::TensorSource & source, MiDashengLmGenConfig & config) {
    const auto embedding = source.require_metadata("model.mdl_model.decoder.model.embed_tokens.weight");
    if (embedding.shape.size() != 2) {
        throw std::runtime_error("MiDashengLM-Gen token embedding must be rank 2");
    }
    config.vocab_size = embedding.shape.at(0);
    config.hidden_size = embedding.shape.at(1);

    engine::assets::require_tensor_shape(
        source,
        "model.mdl_model.audio_projector.net.0.weight",
        {config.hidden_size, config.audio_embedding_size * config.patch_size});
    engine::assets::require_tensor_shape(source, "model.mdl_model.audio_projector.net.0.bias", {config.hidden_size});
    engine::assets::require_tensor_shape(
        source,
        "model.mdl_model.audio_projector.net.2.weight",
        {config.hidden_size, config.hidden_size});
    engine::assets::require_tensor_shape(source, "model.mdl_model.audio_projector.net.2.bias", {config.hidden_size});
    engine::assets::require_tensor_shape(
        source,
        "model.stop_head.weight",
        {2, config.hidden_size});
    engine::assets::require_tensor_shape(source, "model.stop_head.bias", {2});
    engine::assets::require_tensor_shape(
        source,
        "model.flowloss.cfm.model.x_embedder.weight",
        {config.flow_hidden_size, config.target_embedding_size});
    engine::assets::require_tensor_shape(
        source,
        "model.flowloss.cfm.model.c_embedder.cond_embedder.weight",
        {config.flow_hidden_size, config.hidden_size});
    engine::assets::require_tensor_shape(
        source,
        "model.flowloss.cfm.model.final_layer.linear.weight",
        {config.target_embedding_size, config.flow_hidden_size});

    engine::assets::require_tensor_shape(
        source,
        "model.audio_upsampler.weight",
        {config.audio_embedding_size, config.audio_embedding_size, 2});
    engine::assets::require_tensor_shape(source, "model.audio_upsampler.bias", {config.audio_embedding_size});
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.backbone.embed.weight",
        {config.audio_embedding_size, config.audio_embedding_size, 7});
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.backbone.embed.bias",
        {config.audio_embedding_size});
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.backbone.norm.weight",
        {config.audio_embedding_size});
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.backbone.norm.bias",
        {config.audio_embedding_size});
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        const auto prefix = "model.audio_decoder.backbone.convnext." + std::to_string(layer);
        engine::assets::require_tensor_shape(source, prefix + ".dwconv.weight", {config.audio_embedding_size, 1, 7});
        engine::assets::require_tensor_shape(source, prefix + ".dwconv.bias", {config.audio_embedding_size});
        engine::assets::require_tensor_shape(source, prefix + ".norm.weight", {config.audio_embedding_size});
        engine::assets::require_tensor_shape(source, prefix + ".norm.bias", {config.audio_embedding_size});
        engine::assets::require_tensor_shape(
            source,
            prefix + ".pwconv1.weight",
            {config.decoder_intermediate_size, config.audio_embedding_size});
        engine::assets::require_tensor_shape(source, prefix + ".pwconv1.bias", {config.decoder_intermediate_size});
        engine::assets::require_tensor_shape(
            source,
            prefix + ".pwconv2.weight",
            {config.audio_embedding_size, config.decoder_intermediate_size});
        engine::assets::require_tensor_shape(source, prefix + ".pwconv2.bias", {config.audio_embedding_size});
        engine::assets::require_tensor_shape(source, prefix + ".gamma", {config.audio_embedding_size});
    }
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.backbone.final_layer_norm.weight",
        {config.audio_embedding_size});
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.backbone.final_layer_norm.bias",
        {config.audio_embedding_size});
    engine::assets::require_tensor_shape(
        source,
        "model.audio_decoder.head.out.weight",
        {config.istft_n_fft + 2, config.audio_embedding_size});
    engine::assets::require_tensor_shape(source, "model.audio_decoder.head.out.bias", {config.istft_n_fft + 2});
    engine::assets::require_tensor_shape(source, "model.audio_decoder.head.istft.window", {config.istft_n_fft});
}

}  // namespace

std::shared_ptr<const MiDashengLmGenAssets> load_midashenglm_gen_assets(
    const std::filesystem::path & model_path) {
    const auto root = std::filesystem::weakly_canonical(model_path);
    auto assets = std::make_shared<MiDashengLmGenAssets>();
    assets->model_root = root;
    assets->config = parse_config(root / "config.json");
    assets->weights = engine::assets::open_indexed_tensor_source(root / "model.safetensors.index.json", root);
    validate_tensor_shapes(*assets->weights, assets->config);
    return assets;
}

}  // namespace engine::models::midashenglm_gen
