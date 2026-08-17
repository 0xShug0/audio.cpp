#include "engine/models/ace_step/dit_weights_runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "helper_utils.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::ace_step {
namespace {

AceStepConditionEncoderLayerWeights load_condition_encoder_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const AceStepDiffusionConfig & config) {
    const int64_t dim = ace_step_diffusion_attention_head_dim(config, "ACE-Step DiT");
    AceStepConditionEncoderLayerWeights layer;
    layer.input_norm = store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
    layer.post_norm = store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
    layer.q_norm = store.load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
    layer.k_norm = store.load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
    layer.q_proj = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.num_attention_heads * dim, config.hidden_size});
    layer.k_proj = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    layer.v_proj = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    layer.o_proj = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.num_attention_heads * dim});
    layer.gate_proj = store.load_tensor(
        source,
        prefix + ".mlp.gate_proj.weight",
        storage_type,
        {config.intermediate_size, config.hidden_size});
    layer.up_proj = store.load_tensor(
        source,
        prefix + ".mlp.up_proj.weight",
        storage_type,
        {config.intermediate_size, config.hidden_size});
    layer.down_proj = store.load_tensor(
        source,
        prefix + ".mlp.down_proj.weight",
        storage_type,
        {config.hidden_size, config.intermediate_size});
    return layer;
}

std::shared_ptr<const AceStepConditionEncoderWeights> load_condition_encoder_weights(
    const std::shared_ptr<core::BackendWeightStore> & store,
    const AceStepAssets & assets,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.encoder;
    const auto & source = *assets.dit_weights;
    auto weights = std::make_shared<AceStepConditionEncoderWeights>();
    weights->store = store;

    weights->text_projector = store->load_tensor(
        source,
        "encoder.text_projector.weight",
        storage_type,
        {config.hidden_size, config.text_hidden_dim});

    weights->lyric_embed_weight = store->load_tensor(
        source,
        "encoder.lyric_encoder.embed_tokens.weight",
        storage_type,
        {config.hidden_size, config.text_hidden_dim});
    weights->lyric_embed_bias = store->load_tensor(
        source,
        "encoder.lyric_encoder.embed_tokens.bias",
        assets::TensorStorageType::F32,
        {config.hidden_size});
    weights->lyric_layers.reserve(static_cast<size_t>(config.num_lyric_encoder_hidden_layers));
    for (int64_t i = 0; i < config.num_lyric_encoder_hidden_layers; ++i) {
        weights->lyric_layers.push_back(load_condition_encoder_layer_weights(
            *store,
            source,
            "encoder.lyric_encoder.layers." + std::to_string(i),
            storage_type,
            config));
    }
    weights->lyric_norm = store->load_f32_tensor(source, "encoder.lyric_encoder.norm.weight", {config.hidden_size});

    weights->timbre_embed_weight = store->load_tensor(
        source,
        "encoder.timbre_encoder.embed_tokens.weight",
        storage_type,
        {config.hidden_size, config.timbre_hidden_dim});
    weights->timbre_embed_bias = store->load_tensor(
        source,
        "encoder.timbre_encoder.embed_tokens.bias",
        assets::TensorStorageType::F32,
        {config.hidden_size});
    weights->timbre_layers.reserve(static_cast<size_t>(config.num_timbre_encoder_hidden_layers));
    for (int64_t i = 0; i < config.num_timbre_encoder_hidden_layers; ++i) {
        weights->timbre_layers.push_back(load_condition_encoder_layer_weights(
            *store,
            source,
            "encoder.timbre_encoder.layers." + std::to_string(i),
            storage_type,
            config));
    }
    weights->timbre_norm = store->load_f32_tensor(source, "encoder.timbre_encoder.norm.weight", {config.hidden_size});
    // Every ACE-Step package carries this parameter; only the XL class prepends
    // it to the reference frames, so loading it anywhere else would pin a tensor
    // the graph never reads.
    if (config.timbre_special_token) {
        weights->timbre_special_token_host = source.require_f32(
            "encoder.timbre_encoder.special_token",
            {1, 1, config.hidden_size});
    }
    return weights;
}

std::shared_ptr<const AceStepDetokenizerWeights> load_detokenizer_weights(
    const std::shared_ptr<core::BackendWeightStore> & store,
    const AceStepAssets & assets,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.encoder;
    const auto & source = *assets.dit_weights;
    const int64_t dim = config.head_dim;

    auto weights = std::make_shared<AceStepDetokenizerWeights>();
    weights->store = store;
    weights->embed_tokens = {
        store->load_tensor(source, "detokenizer.embed_tokens.weight", storage_type, {config.hidden_size, config.hidden_size}),
        store->load_tensor(source, "detokenizer.embed_tokens.bias", assets::TensorStorageType::F32, {config.hidden_size}),
    };
    weights->special_tokens_host = source.require_f32(
        "detokenizer.special_tokens",
        {1, config.pool_window_size, config.hidden_size});

    weights->layers.layers.reserve(static_cast<size_t>(config.num_attention_pooler_hidden_layers));
    for (int64_t i = 0; i < config.num_attention_pooler_hidden_layers; ++i) {
        const std::string prefix = "detokenizer.layers." + std::to_string(i);
        modules::QwenDecoderLayerWeights layer;
        layer.input_norm.weight = store->load_f32_tensor(
            source, prefix + ".input_layernorm.weight", {config.hidden_size});
        layer.post_norm.weight = store->load_f32_tensor(
            source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        layer.q_norm.weight = store->load_f32_tensor(
            source, prefix + ".self_attn.q_norm.weight", {dim});
        layer.k_norm.weight = store->load_f32_tensor(
            source, prefix + ".self_attn.k_norm.weight", {dim});
        layer.self_attention.q_weight = store->load_tensor(
            source, prefix + ".self_attn.q_proj.weight", storage_type, {config.num_attention_heads * dim, config.hidden_size});
        layer.self_attention.k_weight = store->load_tensor(
            source, prefix + ".self_attn.k_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        layer.self_attention.v_weight = store->load_tensor(
            source, prefix + ".self_attn.v_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        layer.self_attention.out_weight = store->load_tensor(
            source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.num_attention_heads * dim});
        layer.mlp.gate_proj = {
            store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}),
            std::nullopt,
        };
        layer.mlp.up_proj = {
            store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}),
            std::nullopt,
        };
        layer.mlp.down_proj = {
            store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size}),
            std::nullopt,
        };
        weights->layers.layers.push_back(std::move(layer));
    }

    weights->norm = store->load_f32_tensor(source, "detokenizer.norm.weight", {config.hidden_size});
    weights->proj_out = {
        store->load_tensor(source, "detokenizer.proj_out.weight", storage_type, {config.latent_channels, config.hidden_size}),
        store->load_tensor(source, "detokenizer.proj_out.bias", assets::TensorStorageType::F32, {config.latent_channels}),
    };
    weights->quantizer_project_out_weight = source.require_f32(
        "tokenizer.quantizer.project_out.weight",
        {config.hidden_size, config.fsq_dim});
    weights->quantizer_project_out_bias = source.require_f32(
        "tokenizer.quantizer.project_out.bias",
        {config.hidden_size});
    return weights;
}

AceStepTimeEmbeddingWeights load_time_embedding_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size) {
    AceStepTimeEmbeddingWeights weights;
    weights.fc1 = {
        store.load_tensor(source, prefix + ".linear_1.weight", storage_type, {hidden_size, 256}),
        store.load_tensor(source, prefix + ".linear_1.bias", assets::TensorStorageType::F32, {hidden_size}),
    };
    weights.fc2 = {
        store.load_tensor(source, prefix + ".linear_2.weight", storage_type, {hidden_size, hidden_size}),
        store.load_tensor(source, prefix + ".linear_2.bias", assets::TensorStorageType::F32, {hidden_size}),
    };
    weights.time_proj = {
        store.load_tensor(source, prefix + ".time_proj.weight", storage_type, {hidden_size * 6, hidden_size}),
        store.load_tensor(source, prefix + ".time_proj.bias", assets::TensorStorageType::F32, {hidden_size * 6}),
    };
    return weights;
}

AceStepDiTAttentionWeights load_dit_attention_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const AceStepDiffusionConfig & config) {
    const int64_t dim = ace_step_diffusion_attention_head_dim(config, "ACE-Step DiT");
    return {
        store.load_tensor(source, prefix + ".q_proj.weight", storage_type, {config.num_attention_heads * dim, config.hidden_size}),
        store.load_tensor(source, prefix + ".k_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size}),
        store.load_tensor(source, prefix + ".v_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size}),
        store.load_tensor(source, prefix + ".o_proj.weight", storage_type, {config.hidden_size, config.num_attention_heads * dim}),
        store.load_f32_tensor(source, prefix + ".q_norm.weight", {dim}),
        store.load_f32_tensor(source, prefix + ".k_norm.weight", {dim}),
    };
}

std::shared_ptr<const AceStepDiffusionWeights> load_diffusion_weights(
    const std::shared_ptr<core::BackendWeightStore> & store,
    const AceStepAssets & assets,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.diffusion;
    // The two places the DiT meets the condition encoder, and the only shapes in
    // this function that are not square in the DiT's own width.
    const int64_t encoder_hidden_size = assets.config.encoder.hidden_size;
    const auto & source = *assets.dit_weights;
    auto weights = std::make_shared<AceStepDiffusionWeights>();
    weights->store = store;
    weights->one = store->make_f32(core::TensorShape::from_dims({1}), std::vector<float>{1.0F});
    weights->proj_in = {
        store->load_tensor(source, "decoder.proj_in.1.weight", storage_type, {config.hidden_size, config.in_channels, config.patch_size}),
        store->load_tensor(source, "decoder.proj_in.1.bias", assets::TensorStorageType::F32, {config.hidden_size}),
    };
    weights->time_embed = load_time_embedding_weights(*store, source, "decoder.time_embed", storage_type, config.hidden_size);
    weights->time_embed_r = load_time_embedding_weights(*store, source, "decoder.time_embed_r", storage_type, config.hidden_size);
    weights->condition_embedder = {
        store->load_tensor(source, "decoder.condition_embedder.weight", storage_type, {config.hidden_size, encoder_hidden_size}),
        store->load_tensor(source, "decoder.condition_embedder.bias", assets::TensorStorageType::F32, {config.hidden_size}),
    };
    if (!config.is_turbo) {
        // Stands in for the encoder output under classifier-free guidance, so it
        // is sized in the encoder's width and the condition embedder projects it
        // like any other conditioning.
        weights->null_condition_emb_host = source.require_f32("null_condition_emb", {1, 1, encoder_hidden_size});
    }
    weights->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "decoder.layers." + std::to_string(i);
        AceStepDiTLayerWeights layer;
        layer.self_attn_norm = store->load_f32_tensor(source, prefix + ".self_attn_norm.weight", {config.hidden_size});
        layer.self_attn = load_dit_attention_weights(*store, source, prefix + ".self_attn", storage_type, config);
        layer.cross_attn_norm = store->load_f32_tensor(source, prefix + ".cross_attn_norm.weight", {config.hidden_size});
        layer.cross_attn = load_dit_attention_weights(*store, source, prefix + ".cross_attn", storage_type, config);
        layer.mlp_norm = store->load_f32_tensor(source, prefix + ".mlp_norm.weight", {config.hidden_size});
        layer.mlp_gate = {
            store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}),
            std::nullopt,
        };
        layer.mlp_up = {
            store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}),
            std::nullopt,
        };
        layer.mlp_down = {
            store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size}),
            std::nullopt,
        };
        layer.scale_shift_table = store->load_tensor(
            source,
            prefix + ".scale_shift_table",
            storage_type,
            {1, 6, config.hidden_size});
        weights->layers.push_back(std::move(layer));
    }
    weights->norm_out = store->load_f32_tensor(source, "decoder.norm_out.weight", {config.hidden_size});
    weights->proj_out = {
        store->load_tensor(source, "decoder.proj_out.1.weight", storage_type, {config.hidden_size, config.latent_channels, config.patch_size}),
        store->load_tensor(source, "decoder.proj_out.1.bias", assets::TensorStorageType::F32, {config.latent_channels}),
    };
    weights->final_scale_shift_table = store->load_tensor(
        source,
        "decoder.scale_shift_table",
        storage_type,
        {1, 2, config.hidden_size});
    return weights;
}

}  // namespace

AceStepDitWeightsRuntime::AceStepDitWeightsRuntime(
    std::shared_ptr<const AceStepAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t weight_context_bytes)
    : assets_(std::move(assets)) {
    if (!assets_) {
        throw std::runtime_error("ACE-Step DiT weights runtime requires assets");
    }
    if (execution.backend() == nullptr) {
        throw std::runtime_error("ACE-Step DiT weights runtime backend is not initialized");
    }
    store_ = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "ace_step.dit.weights",
        weight_context_bytes);
    detokenizer_weights_ = load_detokenizer_weights(store_, *assets_, storage_type);
    condition_encoder_weights_ = load_condition_encoder_weights(store_, *assets_, storage_type);
    diffusion_weights_ = load_diffusion_weights(store_, *assets_, storage_type);
    store_->upload();
    assets_->dit_weights->release_storage();
}

const std::shared_ptr<const AceStepAssets> & AceStepDitWeightsRuntime::assets() const noexcept {
    return assets_;
}

std::shared_ptr<const AceStepDetokenizerWeights> AceStepDitWeightsRuntime::detokenizer_weights() const noexcept {
    return detokenizer_weights_;
}

std::shared_ptr<const AceStepConditionEncoderWeights> AceStepDitWeightsRuntime::condition_encoder_weights() const noexcept {
    return condition_encoder_weights_;
}

std::shared_ptr<const AceStepDiffusionWeights> AceStepDitWeightsRuntime::diffusion_weights() const noexcept {
    return diffusion_weights_;
}

}  // namespace engine::models::ace_step
