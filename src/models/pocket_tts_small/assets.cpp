#include "engine/models/pocket_tts_small/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/io/yaml.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>

namespace engine::models::pocket_tts_small {

namespace {

modules::AttentionWeights make_packed_attention_weights(
    core::ModuleBuildContext & ctx,
    const PackedAttentionWeights & weights,
    int64_t hidden_size,
    int64_t attention_size = 0,
    int64_t kv_size = 0) {
    if (attention_size == 0) attention_size = hidden_size;
    if (kv_size == 0) kv_size = hidden_size;
    if (weights.in_proj_weight.shape.rank != 2
        || weights.in_proj_weight.shape.dims[0] != attention_size + 2 * kv_size
        || weights.in_proj_weight.shape.dims[1] != hidden_size) {
        throw std::runtime_error("Packed attention weight must have shape [attention_size + 2 * kv_size, hidden_size]");
    }
    const size_t row_stride = weights.in_proj_weight.tensor->nb[1];
    auto q = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, weights.in_proj_weight.tensor, hidden_size, attention_size, row_stride, 0),
        core::TensorShape::from_dims({attention_size, hidden_size}), weights.in_proj_weight.type);
    auto k = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, weights.in_proj_weight.tensor, hidden_size, kv_size, row_stride,
            static_cast<size_t>(attention_size) * row_stride),
        core::TensorShape::from_dims({kv_size, hidden_size}), weights.in_proj_weight.type);
    auto v = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, weights.in_proj_weight.tensor, hidden_size, kv_size, row_stride,
            static_cast<size_t>(attention_size + kv_size) * row_stride),
        core::TensorShape::from_dims({kv_size, hidden_size}), weights.in_proj_weight.type);
    modules::AttentionWeights attention = {};
    attention.q_weight = q;
    attention.k_weight = k;
    attention.v_weight = v;
    attention.out_weight = weights.out_proj_weight;
    return attention;
}

struct Descriptor {
    float default_temperature = 0.7F;
    bool pad_with_spaces_for_short_inputs = false;
    bool remove_semicolons = false;
    bool insert_bos_before_voice = false;
    std::optional<int> model_recommended_frames_after_eos;
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    int flow_layers = 6;
    // Residual blocks of the flow net (upstream `flow_lm.flow.depth`). Distinct from the
    // transformer depth: the *_24l models have 24 transformer layers but a 6-block flow net.
    int flow_depth = 6;
    int flow_dim = 1024;
    int flow_heads = 16;
    int flow_hidden_size = 512;
    int flow_intermediate_size = 4096;
    int latent_dim = 32;
    int mimi_layers = 2;
    int mimi_dim = 512;
    int mimi_heads = 8;
    int mimi_intermediate_size = 2048;
    int mimi_inner_dim = 32;
    int mimi_outer_dim = 512;
    int mimi_context = 250;
    int mimi_seanet_dimension = 512;
    int mimi_base_filters = 64;
    int mimi_encoder_upsample_stride = 16;
};

Descriptor descriptor_from_yaml(const io::yaml::FlattenedDocument & parsed) {
    Descriptor descriptor;
    descriptor.default_temperature =
        io::yaml::optional_float(parsed, "default_temperature").value_or(descriptor.default_temperature);
    descriptor.pad_with_spaces_for_short_inputs =
        io::yaml::optional_bool(parsed, "pad_with_spaces_for_short_inputs", false);
    descriptor.remove_semicolons = io::yaml::optional_bool(parsed, "remove_semicolons", false);
    descriptor.insert_bos_before_voice = io::yaml::optional_bool(parsed, "flow_lm.insert_bos_before_voice", false);
    descriptor.model_recommended_frames_after_eos = io::yaml::optional_int(parsed, "model_recommended_frames_after_eos");
    descriptor.sample_rate = io::yaml::require_int(parsed, "mimi.sample_rate");
    descriptor.frame_rate = io::yaml::require_float(parsed, "mimi.frame_rate");
    descriptor.flow_layers = io::yaml::require_int(parsed, "flow_lm.transformer.num_layers");
    descriptor.flow_depth = io::yaml::optional_int(parsed, "flow_lm.flow.depth").value_or(descriptor.flow_depth);
    descriptor.flow_dim = io::yaml::require_int(parsed, "flow_lm.transformer.d_model");
    descriptor.flow_heads = io::yaml::require_int(parsed, "flow_lm.transformer.num_heads");
    descriptor.flow_hidden_size = io::yaml::require_int(parsed, "flow_lm.flow.dim");
    descriptor.flow_intermediate_size =
        descriptor.flow_dim * io::yaml::require_int(parsed, "flow_lm.transformer.hidden_scale");
    descriptor.latent_dim = io::yaml::require_int(parsed, "mimi.quantizer.dimension");
    descriptor.mimi_layers = io::yaml::require_int(parsed, "mimi.transformer.num_layers");
    descriptor.mimi_dim = io::yaml::require_int(parsed, "mimi.transformer.d_model");
    descriptor.mimi_heads = io::yaml::require_int(parsed, "mimi.transformer.num_heads");
    descriptor.mimi_intermediate_size = io::yaml::require_int(parsed, "mimi.transformer.dim_feedforward");
    descriptor.mimi_inner_dim = io::yaml::require_int(parsed, "mimi.inner_dim");
    descriptor.mimi_outer_dim = io::yaml::require_int(parsed, "mimi.outer_dim");
    descriptor.mimi_context = io::yaml::require_int(parsed, "mimi.transformer.context");
    descriptor.mimi_seanet_dimension = io::yaml::require_int(parsed, "mimi.seanet.dimension");
    descriptor.mimi_base_filters = io::yaml::require_int(parsed, "mimi.seanet.n_filters");
    int hop_length = 1;
    for (const int ratio : io::yaml::require_list_int(parsed, "mimi.seanet.ratios")) {
        hop_length *= ratio;
    }
    const double encoder_frame_rate = static_cast<double>(descriptor.sample_rate) / static_cast<double>(hop_length);
    descriptor.mimi_encoder_upsample_stride =
        static_cast<int>(std::llround(encoder_frame_rate / static_cast<double>(descriptor.frame_rate)));
    return descriptor;
}

ModelConfig make_model_config(const Descriptor & descriptor) {
    ModelConfig config;
    config.default_temperature = descriptor.default_temperature;
    config.sample_rate = descriptor.sample_rate;
    config.frame_rate = descriptor.frame_rate;
    config.mimi_frame_rate = descriptor.frame_rate;
    config.flow_layers = descriptor.flow_layers;
    config.flow_depth = descriptor.flow_depth;
    config.flow_dim = descriptor.flow_dim;
    config.flow_heads = descriptor.flow_heads;
    config.flow_hidden_size = descriptor.flow_hidden_size;
    config.flow_intermediate_size = descriptor.flow_intermediate_size;
    config.latent_dim = descriptor.latent_dim;
    config.mimi_layers = descriptor.mimi_layers;
    config.mimi_dim = descriptor.mimi_dim;
    config.mimi_heads = descriptor.mimi_heads;
    config.mimi_intermediate_size = descriptor.mimi_intermediate_size;
    config.mimi_inner_dim = descriptor.mimi_inner_dim;
    config.mimi_outer_dim = descriptor.mimi_outer_dim;
    config.mimi_context = descriptor.mimi_context;
    config.mimi_seanet_dimension = descriptor.mimi_seanet_dimension;
    config.mimi_base_filters = descriptor.mimi_base_filters;
    config.mimi_encoder_upsample_stride = descriptor.mimi_encoder_upsample_stride;
    config.pad_with_spaces_for_short_inputs = descriptor.pad_with_spaces_for_short_inputs;
    config.remove_semicolons = descriptor.remove_semicolons;
    config.insert_bos_before_voice = descriptor.insert_bos_before_voice;
    config.model_recommended_frames_after_eos = descriptor.model_recommended_frames_after_eos;
    return config;
}

HostWeights load_host_weights(const assets::TensorSource & source) {
    HostWeights weights;
    weights.conditioner_embedding_table = source.require_f32_tensor("flow_lm.conditioner.embed.weight");
    weights.bos_emb = source.require_f32("flow_lm.bos_emb");
    weights.bos_before_voice = source.optional_f32_tensor("flow_lm.bos_before_voice");
    weights.emb_mean = source.require_f32("flow_lm.emb_mean");
    weights.emb_std = source.require_f32("flow_lm.emb_std");
    return weights;
}

}  // namespace

Assets load_assets_from_resources(
    assets::ResourceBundle resources,
    std::string language) {
    const auto descriptor = descriptor_from_yaml(resources.parse_flattened_yaml("config"));
    const auto tensor_source = resources.open_tensor_source("weights");

    Assets manifest;
    manifest.resources = std::move(resources);
    manifest.model_weights = tensor_source;
    manifest.voice_asset_root = tensor_source->source_path().parent_path();
    manifest.host_weights = load_host_weights(*tensor_source);
    manifest.language = std::move(language);
    manifest.tokenizer_pieces = tokenizers::load_sentencepiece_model(manifest.resources.require_file("tokenizer"));
    manifest.model_config = make_model_config(descriptor);
    return manifest;
}

std::shared_ptr<const pocket_tts_small::Assets> load_assets(const std::filesystem::path & model_path) {
    auto resources = model_spec::load_resource_bundle_for_family(model_path, "pocket_tts_small");
    auto manifest = std::make_shared<pocket_tts_small::Assets>(pocket_tts_small::load_assets_from_resources(std::move(resources), "en"));
    const auto parsed = manifest->resources.parse_flattened_yaml("config");
    auto & config = manifest->model_config;
    const auto sampler = io::yaml::require_string(parsed, "flow_lm.flow.type");
    if (sampler == "drifting") config.sampler = pocket_tts_small::FlowSamplerKind::Drifting;
    else if (sampler == "pointwise") config.sampler = pocket_tts_small::FlowSamplerKind::Pointwise;
    else throw std::runtime_error("Unsupported PocketTTS Small sampler: " + sampler);
    config.flow_intermediate_size = io::yaml::optional_int(parsed, "flow_lm.transformer.dim_feedforward")
        .value_or(config.flow_intermediate_size);
    if (io::yaml::optional_string(parsed, "flow_lm.transformer.attn_dim", "null") != "null")
        config.flow_attention_dim = io::yaml::require_int(parsed, "flow_lm.transformer.attn_dim");
    if (io::yaml::optional_string(parsed, "flow_lm.transformer.num_kv_heads", "null") != "null")
        config.flow_kv_heads = io::yaml::require_int(parsed, "flow_lm.transformer.num_kv_heads");
    config.flow_condition_dim = io::yaml::require_int(parsed, "flow_lm.z_dim");
    config.default_eos_threshold = io::yaml::require_float(parsed, "default_eos_threshold");
    for (const auto & [key, value] : parsed.scalars) {
        const std::string prefix = "replace_characters.";
        if (key.rfind(prefix, 0) == 0) config.replace_characters.emplace_back(key.substr(prefix.size()), value);
    }
    pocket_tts_small::MimiDecoderConfig decoder;
    decoder.latent_size = config.latent_dim;
    decoder.hidden_size = io::yaml::require_int(parsed, "mimi.decoder_transformer.d_model");
    decoder.num_heads = io::yaml::require_int(parsed, "mimi.decoder_transformer.num_heads");
    decoder.intermediate_size = io::yaml::require_int(parsed, "mimi.decoder_transformer.dim_feedforward");
    decoder.transformer_layers = io::yaml::require_int(parsed, "mimi.decoder_transformer.num_layers");
    decoder.encoder_upsample_stride = config.mimi_encoder_upsample_stride;
    decoder.context = io::yaml::require_int(parsed, "mimi.decoder_transformer.context");
    decoder.base_filters = io::yaml::require_int(parsed, "mimi.decoder_seanet.n_filters");
    decoder.input_kernel_size = io::yaml::require_int(parsed, "mimi.decoder_seanet.kernel_size");
    config.decoder = decoder;
    manifest->host_weights.conditioner_projection =
        manifest->model_weights->optional_f32_tensor("flow_lm.conditioner.proj.weight");
    manifest->family = "pocket_tts_small";
    manifest->model_config.min_eos_steps = 6;
    manifest->model_config.fade_in_samples = manifest->model_config.sample_rate / 200;
    manifest->default_voice_path = manifest->voice_asset_root / "embeddings/alba.safetensors";
    return manifest;
}

namespace {

namespace binding = engine::modules::binding;

PackedAttentionWeights load_backend_packed_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    const std::string in_name = prefix + "self_attn.in_proj.weight";
    const std::string out_name = prefix + "self_attn.out_proj.weight";
    return {
        binding::tensor_from_named_source(store, source, in_name, storage_type),
        binding::tensor_from_named_source(store, source, out_name, storage_type),
    };
}

TransformerLayerWeights load_backend_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    TransformerLayerWeights layer;
    layer.norm1 = binding::norm_from_named_source(store, source, prefix + "norm1.weight", prefix + "norm1.bias");
    layer.self_attn = load_backend_packed_attention(store, source, prefix, storage_type);
    layer.layer_scale_1 = binding::optional_layer_scale_from_named_source(store, source, prefix + "layer_scale_1.scale");
    layer.norm2 = binding::norm_from_named_source(store, source, prefix + "norm2.weight", prefix + "norm2.bias");
    layer.feed_forward = modules::FeedForwardWeights{
        binding::tensor_from_named_source(store, source, prefix + "linear1.weight", storage_type),
        std::nullopt,
        binding::tensor_from_named_source(store, source, prefix + "linear2.weight", storage_type),
        std::nullopt,
    };
    layer.layer_scale_2 = binding::optional_layer_scale_from_named_source(store, source, prefix + "layer_scale_2.scale");
    return layer;
}

ResidualBlockWeights load_backend_residual_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType conv_storage_type) {
    return {
        binding::conv1d_from_named_source(store, source, prefix + ".1.conv.weight", prefix + ".1.conv.bias", conv_storage_type),
        binding::conv1d_from_named_source(store, source, prefix + ".3.conv.weight", prefix + ".3.conv.bias", conv_storage_type),
    };
}

modules::ConditionedFlowMLPWeights load_backend_flow_net_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layers,
    assets::TensorStorageType storage_type) {
    modules::ConditionedFlowMLPWeights weights;
    weights.input_projection = binding::linear_from_named_source(
        store,
        source,
        "flow_lm.flow_net.input_proj.weight",
        "flow_lm.flow_net.input_proj.bias",
        storage_type);
    weights.condition_projection = binding::linear_from_named_source(
        store,
        source,
        "flow_lm.flow_net.cond_embed.weight",
        "flow_lm.flow_net.cond_embed.bias",
        storage_type);
    weights.residual_layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "flow_lm.flow_net.res_blocks." + std::to_string(layer);
        weights.residual_layers.push_back(modules::AdaLNResidualMLPWeights{
            binding::norm_from_named_source(store, source, prefix + ".in_ln.weight", prefix + ".in_ln.bias"),
            modules::AdaLNModulationWeights{
                binding::linear_from_named_source(store, source, prefix + ".adaLN_modulation.1.weight", prefix + ".adaLN_modulation.1.bias", storage_type),
            },
            binding::linear_from_named_source(store, source, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias", storage_type),
            binding::linear_from_named_source(store, source, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias", storage_type),
        });
    }
    weights.output_projection = modules::FinalAdaLNProjectionWeights{
        modules::AdaLNModulationWeights{
            binding::linear_from_named_source(
                store,
                source,
                "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight",
                "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias",
                storage_type),
        },
        binding::linear_from_named_source(
            store,
            source,
            "flow_lm.flow_net.final_layer.linear.weight",
            "flow_lm.flow_net.final_layer.linear.bias",
            storage_type),
    };
    return weights;
}

FlowWeights load_backend_flow_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ModelConfig & config,
    const HostWeights & host,
    bool has_mimi_encoder,
    assets::TensorStorageType storage_type) {
    FlowWeights weights;
    weights.input_linear = modules::LinearWeights{
        binding::tensor_from_named_source(store, source, "flow_lm.input_linear.weight", storage_type),
        std::nullopt,
    };
    if (host.input_linear_bias.has_value()) {
        if (static_cast<int64_t>(host.input_linear_bias->size()) != config.flow_dim) {
            throw std::runtime_error("PocketTTS input_linear bias must have flow_dim values");
        }
        weights.input_linear.bias = store.make_f32(core::TensorShape::from_dims({config.flow_dim}), *host.input_linear_bias);
    }
    weights.transformer_layers.reserve(static_cast<size_t>(config.flow_layers));
    for (int64_t layer = 0; layer < config.flow_layers; ++layer) {
        const std::string prefix = "flow_lm.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, storage_type));
    }
    weights.flow_net = load_backend_flow_net_weights(store, source, config.flow_depth, storage_type);
    if (config.flow_condition_dim > 0) {
        weights.z_projection = modules::LinearWeights{
            binding::tensor_from_named_source(store, source, "flow_lm.z_proj.weight", storage_type), std::nullopt};
    }
    weights.out_norm = binding::norm_from_named_source(store, source, "flow_lm.out_norm.weight", "flow_lm.out_norm.bias");
    weights.out_eos = binding::linear_from_named_source(store, source, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", storage_type);
    if (has_mimi_encoder) {
        weights.speaker_proj_weight =
            binding::tensor_from_named_source(store, source, "flow_lm.speaker_proj_weight", storage_type);
    }
    return weights;
}

MimiEncoderWeights load_backend_mimi_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    MimiEncoderWeights weights;
    weights.input_conv = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.0.conv.weight", "mimi.encoder.model.0.conv.bias", conv_storage_type);
    weights.block0 = load_backend_residual_block(store, source, "mimi.encoder.model.1.block", conv_storage_type);
    weights.downsample0 = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.3.conv.weight", "mimi.encoder.model.3.conv.bias", conv_storage_type);
    weights.block1 = load_backend_residual_block(store, source, "mimi.encoder.model.4.block", conv_storage_type);
    weights.downsample1 = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.6.conv.weight", "mimi.encoder.model.6.conv.bias", conv_storage_type);
    weights.block2 = load_backend_residual_block(store, source, "mimi.encoder.model.7.block", conv_storage_type);
    weights.downsample2 = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.9.conv.weight", "mimi.encoder.model.9.conv.bias", conv_storage_type);
    weights.output_conv = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.11.conv.weight", "mimi.encoder.model.11.conv.bias", conv_storage_type);
    weights.transformer_layers.reserve(static_cast<size_t>(config.mimi_layers));
    for (int64_t layer = 0; layer < config.mimi_layers; ++layer) {
        const std::string prefix = "mimi.encoder_transformer.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, matmul_storage_type));
    }
    weights.downsample_conv = modules::StreamingConv1dWeights{
        binding::tensor_from_named_source(store, source, "mimi.downsample.conv.conv.weight", conv_storage_type),
        std::nullopt,
    };
    return weights;
}

MimiDecoderWeights load_backend_mimi_decoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    core::BackendType backend_type) {
    const auto decoder = config.decoder;
    MimiDecoderWeights weights;
    weights.transformer_layers.reserve(static_cast<size_t>(decoder.transformer_layers));
    for (int64_t layer = 0; layer < decoder.transformer_layers; ++layer) {
        const std::string prefix = "mimi.decoder_transformer.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, matmul_storage_type));
    }
    weights.quantizer_output_proj_weight =
        binding::tensor_from_named_source(store, source, "mimi.quantizer.output_proj.weight", conv_storage_type);
    const int64_t upsample_kernel_size = decoder.encoder_upsample_stride * 2;
    auto upsample_weight = source.require_f32("mimi.upsample.convtr.convtr.weight");
    if (static_cast<int64_t>(upsample_weight.size()) != decoder.hidden_size * upsample_kernel_size) {
        throw std::runtime_error("mimi.upsample.convtr.convtr.weight has invalid depthwise conv transpose weight size");
    }
    if (core::uses_host_graph_plan(backend_type)) {
        // The depthwise module implements transpose convolution with a reversed FIR.
        for (int64_t channel = 0; channel < decoder.hidden_size; ++channel) {
            auto * kernel = upsample_weight.data() + channel * upsample_kernel_size;
            std::reverse(kernel, kernel + upsample_kernel_size);
        }
        weights.encoder_upsample_weight = store.make_f32(
            core::TensorShape::from_dims({decoder.hidden_size, 1, 1, upsample_kernel_size}),
            upsample_weight);
    } else {
        // Preserve the existing GPU GEMM lowering and its arithmetic.
        std::vector<float> dense(static_cast<size_t>(decoder.hidden_size * decoder.hidden_size * upsample_kernel_size), 0.0F);
        for (int64_t channel = 0; channel < decoder.hidden_size; ++channel) {
            std::copy_n(
                upsample_weight.data() + channel * upsample_kernel_size,
                upsample_kernel_size,
                dense.data() + (channel * decoder.hidden_size + channel) * upsample_kernel_size);
        }
        weights.encoder_upsample_weight = store.make_f32(
            core::TensorShape::from_dims({decoder.hidden_size, decoder.hidden_size, upsample_kernel_size}),
            dense);
    }
    weights.input_projection = binding::conv1d_from_named_source(store, source, "mimi.decoder.model.0.conv.weight", "mimi.decoder.model.0.conv.bias", conv_storage_type);
    weights.stage0_upsample = binding::conv_transpose1d_from_named_source(store, source, "mimi.decoder.model.2.convtr.weight", "mimi.decoder.model.2.convtr.bias", conv_storage_type);
    weights.stage0_upsample_bias_values = source.require_f32("mimi.decoder.model.2.convtr.bias");
    weights.stage0_block = load_backend_residual_block(store, source, "mimi.decoder.model.3.block", conv_storage_type);
    weights.stage1_upsample = binding::conv_transpose1d_from_named_source(store, source, "mimi.decoder.model.5.convtr.weight", "mimi.decoder.model.5.convtr.bias", conv_storage_type);
    weights.stage1_upsample_bias_values = source.require_f32("mimi.decoder.model.5.convtr.bias");
    weights.stage1_block = load_backend_residual_block(store, source, "mimi.decoder.model.6.block", conv_storage_type);
    weights.stage2_upsample = binding::conv_transpose1d_from_named_source(store, source, "mimi.decoder.model.8.convtr.weight", "mimi.decoder.model.8.convtr.bias", conv_storage_type);
    weights.stage2_upsample_bias_values = source.require_f32("mimi.decoder.model.8.convtr.bias");
    weights.stage2_block = load_backend_residual_block(store, source, "mimi.decoder.model.9.block", conv_storage_type);
    weights.output_projection = binding::conv1d_from_named_source(store, source, "mimi.decoder.model.11.conv.weight", "mimi.decoder.model.11.conv.bias", conv_storage_type);
    return weights;
}

}  // namespace

std::shared_ptr<const BackendWeights> load_pocket_tts_small_backend_weights(
    const Assets & manifest,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t flow_context_bytes,
    size_t mimi_encoder_context_bytes,
    size_t mimi_decoder_context_bytes) {
    if (manifest.model_weights == nullptr) {
        throw std::runtime_error("PocketTTS model weights source is not loaded");
    }
    auto weights = std::make_shared<BackendWeights>();
    weights->backend_type = backend_type;
    weights->host = manifest.host_weights;
    weights->flow_store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PocketTTS.flow_lm.weights",
        flow_context_bytes);
    weights->has_mimi_encoder = manifest.model_weights->has_tensor("mimi.encoder.model.0.conv.weight");
    weights->flow = load_backend_flow_weights(
        *weights->flow_store,
        *manifest.model_weights,
        manifest.model_config,
        manifest.host_weights,
        weights->has_mimi_encoder,
        matmul_storage_type);
    weights->flow_store->upload();

    if (weights->has_mimi_encoder) {
        weights->mimi_encoder_store = std::make_shared<core::BackendWeightStore>(
            backend,
            backend_type,
            "PocketTTS.mimi_encoder.weights",
            mimi_encoder_context_bytes);
        weights->mimi_encoder =
            load_backend_mimi_encoder_weights(
                *weights->mimi_encoder_store,
                *manifest.model_weights,
                manifest.model_config,
                matmul_storage_type,
                conv_storage_type);
        weights->mimi_encoder_store->upload();
    }

    weights->mimi_decoder_store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PocketTTS.mimi_decoder.weights",
        mimi_decoder_context_bytes);
    weights->mimi_decoder =
        load_backend_mimi_decoder_weights(
            *weights->mimi_decoder_store,
            *manifest.model_weights,
            manifest.model_config,
            matmul_storage_type,
            conv_storage_type,
            backend_type);
    weights->mimi_decoder_store->upload();

    manifest.model_weights->release_storage();
    return weights;
}

namespace {

void append_bytes(std::vector<unsigned char> & out, const void * data, size_t size) {
    if (size == 0) {
        return;
    }
    const auto * bytes = static_cast<const unsigned char *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

std::vector<unsigned char> make_i64_bytes(int64_t value) {
    std::vector<unsigned char> data;
    data.reserve(sizeof(int64_t));
    append_bytes(data, &value, sizeof(int64_t));
    return data;
}

size_t element_count_for_shape(const std::vector<int64_t> & shape, const std::string & name) {
    size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("tensor shape contains a negative dimension: " + name);
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

std::pair<const std::byte *, size_t> require_safetensor_data(
    const io::SafeTensorIndex & index,
    const io::BinaryBlob & bytes,
    const io::SafeTensorInfo & info) {
    const size_t byte_size = info.data_end - info.data_begin;
    // Subtractions against the total, so no term can overflow the comparison
    // and let an out-of-range span through. See the matching check in
    // framework/assets/tensor_source.cpp.
    const size_t total = bytes.size();
    if (index.header_bytes > total || byte_size > total - index.header_bytes ||
        info.data_begin > total - index.header_bytes - byte_size) {
        throw std::runtime_error("tensor data range is out of bounds: " + info.name);
    }
    const size_t data_offset = index.header_bytes + info.data_begin;
    return {bytes.data() + static_cast<std::ptrdiff_t>(data_offset), byte_size};
}

std::vector<float> read_voice_state_f32(
    const io::SafeTensorIndex & index,
    const io::BinaryBlob & bytes,
    const io::SafeTensorInfo & info) {
    if (info.dtype != "F32") {
        throw std::runtime_error("Voice state cache tensor must be F32: " + info.name);
    }
    const size_t count = element_count_for_shape(info.shape, info.name);
    const auto [data, byte_size] = require_safetensor_data(index, bytes, info);
    if (byte_size != count * sizeof(float)) {
        throw std::runtime_error("Voice state cache tensor byte size mismatch: " + info.name);
    }
    std::vector<float> values(count);
    if (!values.empty()) {
        std::memcpy(values.data(), data, byte_size);
    }
    return values;
}

int64_t read_voice_state_i64_scalar(
    const io::SafeTensorIndex & index,
    const io::BinaryBlob & bytes,
    const io::SafeTensorInfo & info) {
    if (info.dtype != "I64" || info.data_end - info.data_begin != sizeof(int64_t)) {
        throw std::runtime_error("Voice state offset tensor must be an I64 scalar: " + info.name);
    }
    const auto [data, byte_size] = require_safetensor_data(index, bytes, info);
    (void) byte_size;
    int64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

}  // namespace

VoiceStateAssets load_voice_state_assets(const std::filesystem::path & source) {
    const auto index = io::load_safetensors_index(source);
    const auto bytes = io::read_binary_blob(index.source_path);
    std::map<int64_t, VoiceAttentionCache> layers;
    const std::regex layer_pattern(R"(^transformer\.layers\.(\d+)\.self_attn$)");

    for (const auto & [name, tensor] : index.tensors) {
        const auto slash = name.rfind('/');
        if (slash == std::string::npos) {
            continue;
        }
        const std::string module_name = name.substr(0, slash);
        const std::string tensor_key = name.substr(slash + 1);
        std::smatch match;
        if (!std::regex_match(module_name, match, layer_pattern)) {
            continue;
        }
        const int64_t layer_index = std::stoll(match[1].str());
        auto & layer = layers[layer_index];

        if (tensor_key == "cache") {
            const auto values = read_voice_state_f32(index, bytes, tensor);
            if (tensor.shape.size() == 5) {
                layer.cached_steps = tensor.shape[2];
                layer.heads = tensor.shape[3];
                layer.head_dim = tensor.shape[4];
            } else if (tensor.shape.size() == 4) {
                layer.cached_steps = tensor.shape[1];
                layer.heads = tensor.shape[2];
                layer.head_dim = tensor.shape[3];
            } else {
                throw std::runtime_error("Voice state cache tensor must have rank 4 or 5: " + tensor.name);
            }
            const size_t single_size = static_cast<size_t>(layer.cached_steps * layer.heads * layer.head_dim);
            if (values.size() != single_size * 2U) {
                throw std::runtime_error("Voice state cache tensor size mismatch: " + tensor.name);
            }
            layer.key.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(single_size));
            layer.value.assign(values.begin() + static_cast<std::ptrdiff_t>(single_size), values.end());
        } else if (tensor_key == "offset") {
            layer.offset = read_voice_state_i64_scalar(index, bytes, tensor);
        } else if (tensor_key == "current_end") {
            if (tensor.shape.empty()) {
                throw std::runtime_error("Voice state current_end tensor must have shape: " + tensor.name);
            }
            layer.offset = tensor.shape[0];
        }
    }

    if (layers.empty()) {
        throw std::runtime_error("No transformer voice cache layers found in " + source.string());
    }

    VoiceStateAssets state;
    state.transformer_layers.reserve(layers.size());
    for (auto & [layer_index, layer] : layers) {
        (void) layer_index;
        if (layer.key.empty() || layer.value.empty()) {
            throw std::runtime_error("Incomplete voice state cache layer in " + source.string());
        }
        state.transformer_layers.push_back(std::move(layer));
    }
    return state;
}

std::optional<SpeakerLatents> load_speaker_latents(const std::filesystem::path & source) {
    const auto index = io::load_safetensors_index(source);
    const auto found = index.tensors.find("speaker_wavs");
    if (found == index.tensors.end()) {
        return std::nullopt;
    }
    const auto & info = found->second;
    const auto & shape = info.shape;
    if (!(shape.size() == 3 && shape[0] == 1) && shape.size() != 2) {
        throw std::runtime_error("speaker_wavs must have shape [1, channels, frames]: " + source.string());
    }
    const auto bytes = io::read_binary_blob(index.source_path);
    const auto channel_major = read_voice_state_f32(index, bytes, info);
    SpeakerLatents latents;
    latents.channels = shape[shape.size() - 2];
    latents.frames = shape[shape.size() - 1];
    latents.values.resize(channel_major.size());
    for (int64_t channel = 0; channel < latents.channels; ++channel) {
        for (int64_t frame = 0; frame < latents.frames; ++frame) {
            latents.values[static_cast<size_t>(frame * latents.channels + channel)] =
                channel_major[static_cast<size_t>(channel * latents.frames + frame)];
        }
    }
    return latents;
}

void save_voice_state_assets(
    const std::filesystem::path & destination,
    const runtime::TransformerKVState & state,
    int64_t heads,
    int64_t head_dim) {
    if (state.layers.empty()) {
        throw std::runtime_error("PocketTTS voice state export requires cache layers");
    }
    if (heads <= 0 || head_dim <= 0 || state.current_end < 0) {
        throw std::runtime_error("PocketTTS voice state export received invalid cache shape");
    }
    const int64_t step_elems = heads * head_dim;
    const size_t valid_elems = static_cast<size_t>(state.current_end * step_elems);

    std::vector<io::SafeTensorWriteEntry> entries;
    entries.reserve(state.layers.size() * 2U);
    for (size_t layer_index = 0; layer_index < state.layers.size(); ++layer_index) {
        const auto & layer = state.layers[layer_index];
        if (layer.valid_steps != state.current_end) {
            throw std::runtime_error("PocketTTS voice state export requires aligned layer offsets");
        }
        if (layer.key.size() < valid_elems || layer.value.size() < valid_elems) {
            throw std::runtime_error("PocketTTS voice state export cache is smaller than its offset");
        }

        io::SafeTensorWriteEntry cache;
        cache.name = "transformer.layers." + std::to_string(layer_index) + ".self_attn/cache";
        cache.dtype = "F32";
        cache.shape = {2, 1, state.current_end, heads, head_dim};
        cache.data.reserve(valid_elems * 2U * sizeof(float));
        append_bytes(cache.data, layer.key.data(), valid_elems * sizeof(float));
        append_bytes(cache.data, layer.value.data(), valid_elems * sizeof(float));
        entries.push_back(std::move(cache));

        io::SafeTensorWriteEntry offset;
        offset.name = "transformer.layers." + std::to_string(layer_index) + ".self_attn/offset";
        offset.dtype = "I64";
        offset.shape = {1};
        offset.data = make_i64_bytes(state.current_end);
        entries.push_back(std::move(offset));
    }

    io::write_safetensors_file(destination, entries);
}

std::filesystem::path preset_embedding_path(const std::filesystem::path & model_root, const std::string & preset_name) {
    return model_root / "embeddings" / (preset_name + ".safetensors");
}

VoiceStateAssets load_voice_assets_for_plan(const VoiceConditioningPlan & plan, const Assets &) {
    switch (plan.source) {
        case VoiceSourceKind::NamedPreset:
        case VoiceSourceKind::PreparedEmbedding:
            if (plan.asset_path.empty()) {
                throw std::runtime_error("PocketTTS preset voice conditioning requires a voice state asset path");
            }
            return load_voice_state_assets(plan.asset_path);
        case VoiceSourceKind::CloneAudio:
            throw std::runtime_error("Clone audio does not load precomputed voice state assets");
    }
    throw std::runtime_error("PocketTTS voice assets received unknown voice source");
}

modules::TransformerEncoderBlockWeights make_transformer_block_weights(
    core::ModuleBuildContext & ctx,
    const TransformerLayerWeights & weights,
    int64_t hidden_size,
    int64_t attention_size,
    int64_t kv_size) {
    return modules::TransformerEncoderBlockWeights{
        weights.norm1,
        make_packed_attention_weights(ctx, weights.self_attn, hidden_size, attention_size, kv_size),
        weights.layer_scale_1,
        weights.norm2,
        weights.feed_forward,
        weights.layer_scale_2,
    };
}

}  // namespace engine::models::pocket_tts_small
