#include "engine/models/pocket_tts/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/io/yaml.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>

namespace engine::models::pocket_tts {
namespace {

namespace binding = engine::modules::binding;

struct PocketTTSDescriptor {
    float default_temperature = 0.7F;
    bool pad_with_spaces_for_short_inputs = false;
    bool remove_semicolons = false;
    bool insert_bos_before_voice = false;
    std::optional<int> model_recommended_frames_after_eos;
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    int flow_layers = 6;
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
    const size_t data_offset = index.header_bytes + info.data_begin;
    const size_t byte_size = info.data_end - info.data_begin;
    if (data_offset + byte_size > bytes.size()) {
        throw std::runtime_error("tensor data range is out of bounds: " + info.name);
    }
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

PocketTTSDescriptor descriptor_from_yaml(const io::yaml::FlattenedDocument & parsed) {
    PocketTTSDescriptor descriptor;
    descriptor.default_temperature =
        io::yaml::optional_float(parsed, "default_temperature").value_or(descriptor.default_temperature);
    const auto pad_it = parsed.scalars.find("pad_with_spaces_for_short_inputs");
    descriptor.pad_with_spaces_for_short_inputs = io::yaml::parse_bool_scalar(
        pad_it != parsed.scalars.end() ? pad_it->second : "false",
        "pad_with_spaces_for_short_inputs");
    const auto semicolon_it = parsed.scalars.find("remove_semicolons");
    descriptor.remove_semicolons = io::yaml::parse_bool_scalar(
        semicolon_it != parsed.scalars.end() ? semicolon_it->second : "false",
        "remove_semicolons");
    const auto bos_it = parsed.scalars.find("flow_lm.insert_bos_before_voice");
    descriptor.insert_bos_before_voice = io::yaml::parse_bool_scalar(
        bos_it != parsed.scalars.end() ? bos_it->second : "false",
        "flow_lm.insert_bos_before_voice");
    descriptor.model_recommended_frames_after_eos = io::yaml::optional_int(parsed, "model_recommended_frames_after_eos");
    descriptor.sample_rate = io::yaml::require_int(parsed, "mimi.sample_rate");
    descriptor.frame_rate = io::yaml::require_float(parsed, "mimi.frame_rate");
    descriptor.flow_layers = io::yaml::require_int(parsed, "flow_lm.transformer.num_layers");
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
    const auto ratios_it = parsed.lists.find("mimi.seanet.ratios");
    if (ratios_it == parsed.lists.end() || ratios_it->second.empty()) {
        throw std::runtime_error("Missing PocketTTS config list: mimi.seanet.ratios");
    }
    int hop_length = 1;
    for (const auto & ratio : ratios_it->second) {
        hop_length *= std::stoi(ratio);
    }
    const double encoder_frame_rate = static_cast<double>(descriptor.sample_rate) / static_cast<double>(hop_length);
    descriptor.mimi_encoder_upsample_stride =
        static_cast<int>(std::llround(encoder_frame_rate / static_cast<double>(descriptor.frame_rate)));
    return descriptor;
}

PocketTTSDescriptor descriptor_from_builtin_defaults(const std::string & language) {
    PocketTTSDescriptor descriptor;
    // Upstream pocket-tts defaults the english and english_2026-04 models to
    // temperature 0.3 (human evals preferred it at equal WER/similarity).
    if (language == "english" || language == "english_2026-04") {
        descriptor.default_temperature = 0.3F;
    }
    descriptor.pad_with_spaces_for_short_inputs = (language == "english_2026-01");
    descriptor.insert_bos_before_voice = (language != "english_2026-01");
    if (language.find("french_24l") != std::string::npos) {
        descriptor.remove_semicolons = true;
        descriptor.model_recommended_frames_after_eos = 8;
    }
    descriptor.flow_layers = language.find("_24l") != std::string::npos ? 24 : 6;
    descriptor.mimi_inner_dim = (language == "english_2026-01") ? 512 : 32;
    return descriptor;
}

PocketTTSDescriptor resolve_descriptor(
    const assets::ResourceBundle & resources,
    const std::string & language) {
    if (resources.has_file("config")) {
        return descriptor_from_yaml(resources.parse_flattened_yaml("config"));
    }
    return descriptor_from_builtin_defaults(language);
}

PocketTTSModelConfig make_model_config(const PocketTTSDescriptor & descriptor) {
    PocketTTSModelConfig config;
    config.default_temperature = descriptor.default_temperature;
    config.sample_rate = descriptor.sample_rate;
    config.frame_rate = descriptor.frame_rate;
    config.mimi_frame_rate = descriptor.frame_rate;
    config.flow_layers = descriptor.flow_layers;
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

PocketTTSBackendPackedAttentionWeights load_backend_packed_attention(
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

PocketTTSBackendTransformerLayerWeights load_backend_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    PocketTTSBackendTransformerLayerWeights layer;
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

PocketTTSBackendResidualBlockWeights load_backend_residual_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType conv_storage_type) {
    return {
        binding::conv1d_from_named_source(store, source, prefix + ".1.conv.weight", prefix + ".1.conv.bias", conv_storage_type),
        binding::conv1d_from_named_source(store, source, prefix + ".3.conv.weight", prefix + ".3.conv.bias", conv_storage_type),
    };
}

modules::TimestepEmbeddingWeights load_backend_timestep_embedding(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    return modules::TimestepEmbeddingWeights{
        binding::f32_tensor_from_named_source(store, source, prefix + ".freqs"),
        binding::linear_from_named_source(store, source, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias", storage_type),
        binding::linear_from_named_source(store, source, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias", storage_type),
        binding::f32_tensor_from_named_source(store, source, prefix + ".mlp.3.alpha"),
    };
}

modules::TimedConditionedFlowMLPWeights load_backend_flow_net_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layers,
    assets::TensorStorageType storage_type) {
    modules::TimedConditionedFlowMLPWeights weights;
    weights.start_time_embedding = load_backend_timestep_embedding(store, source, "flow_lm.flow_net.time_embed.0", storage_type);
    weights.end_time_embedding = load_backend_timestep_embedding(store, source, "flow_lm.flow_net.time_embed.1", storage_type);
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

std::vector<float> expand_depthwise_convtranspose_weight(
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels,
    int64_t kernel_size) {
    const auto values = source.require_f32(name);
    if (static_cast<int64_t>(values.size()) != channels * kernel_size) {
        throw std::runtime_error(name + " has invalid depthwise conv transpose weight size");
    }
    std::vector<float> dense(static_cast<size_t>(channels * channels * kernel_size), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const size_t src_offset = static_cast<size_t>(channel * kernel_size);
        const size_t dst_offset = static_cast<size_t>(((channel * channels) + channel) * kernel_size);
        std::copy_n(
            values.begin() + static_cast<ptrdiff_t>(src_offset),
            static_cast<size_t>(kernel_size),
            dense.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
    return dense;
}

PocketTTSBackendFlowWeights load_backend_flow_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PocketTTSModelConfig & config,
    assets::TensorStorageType storage_type) {
    PocketTTSBackendFlowWeights weights;
    weights.input_linear = modules::LinearWeights{
        binding::tensor_from_named_source(store, source, "flow_lm.input_linear.weight", storage_type),
        std::nullopt,
    };
    weights.transformer_layers.reserve(static_cast<size_t>(config.flow_layers));
    for (int64_t layer = 0; layer < config.flow_layers; ++layer) {
        const std::string prefix = "flow_lm.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, storage_type));
    }
    weights.flow_net = load_backend_flow_net_weights(store, source, config.flow_layers, storage_type);
    weights.out_norm = binding::norm_from_named_source(store, source, "flow_lm.out_norm.weight", "flow_lm.out_norm.bias");
    weights.out_eos = binding::linear_from_named_source(store, source, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", storage_type);
    weights.speaker_proj_weight =
        binding::tensor_from_named_source(store, source, "flow_lm.speaker_proj_weight", storage_type);
    return weights;
}

PocketTTSBackendMimiEncoderWeights load_backend_mimi_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PocketTTSModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    PocketTTSBackendMimiEncoderWeights weights;
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

PocketTTSBackendMimiDecoderWeights load_backend_mimi_decoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PocketTTSModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    PocketTTSBackendMimiDecoderWeights weights;
    weights.transformer_layers.reserve(static_cast<size_t>(config.mimi_layers));
    for (int64_t layer = 0; layer < config.mimi_layers; ++layer) {
        const std::string prefix = "mimi.decoder_transformer.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, matmul_storage_type));
    }
    weights.quantizer_output_proj_weight =
        binding::tensor_from_named_source(store, source, "mimi.quantizer.output_proj.weight", conv_storage_type);
    const auto dense_upsample = expand_depthwise_convtranspose_weight(
        source,
        "mimi.upsample.convtr.convtr.weight",
        config.mimi_dim,
        config.mimi_encoder_upsample_stride * 2);
    weights.encoder_upsample_weight = store.make_f32(
        core::TensorShape::from_dims({
            config.mimi_dim,
            config.mimi_dim,
            config.mimi_encoder_upsample_stride * 2,
        }),
        dense_upsample);
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

PocketTTSHostWeights load_host_weights(const assets::TensorSource & source) {
    PocketTTSHostWeights weights;
    weights.conditioner_embedding_table = source.require_f32_tensor("flow_lm.conditioner.embed.weight");
    weights.bos_emb = source.require_f32("flow_lm.bos_emb");
    weights.bos_before_voice = source.optional_f32_tensor("flow_lm.bos_before_voice");
    weights.emb_mean = source.require_f32("flow_lm.emb_mean");
    weights.emb_std = source.require_f32("flow_lm.emb_std");
    return weights;
}

}  // namespace

PocketTTSAssets load_pocket_tts_assets(
    const std::filesystem::path & model_root,
    std::string language,
    const std::optional<std::filesystem::path> & config_path) {
    assets::ResourceBundle resources(model_root);
    resources.add_model_files({
        {"tokenizer", "tokenizer.model"},
        {"weights", "model.safetensors"},
    });
    if (config_path.has_value()) {
        resources.add_file("config", *config_path);
    }
    const auto descriptor = resolve_descriptor(resources, language);
    const auto tensor_source = resources.open_tensor_source("weights");

    PocketTTSAssets manifest;
    manifest.model_weights = tensor_source;
    manifest.host_weights = load_host_weights(*tensor_source);
    manifest.model_root = model_root;
    manifest.language = std::move(language);
    manifest.tokenizer_path = resources.require_file("tokenizer");
    manifest.tokenizer_pieces = tokenizers::load_sentencepiece_model(manifest.tokenizer_path);
    manifest.model_config = make_model_config(descriptor);
    return manifest;
}

std::shared_ptr<const PocketTTSBackendWeights> load_pocket_tts_backend_weights(
    const PocketTTSAssets & manifest,
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
    auto weights = std::make_shared<PocketTTSBackendWeights>();
    weights->backend_type = backend_type;
    weights->host = manifest.host_weights;
    weights->flow_store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PocketTTS.flow_lm.weights",
        flow_context_bytes);
    weights->flow =
        load_backend_flow_weights(*weights->flow_store, *manifest.model_weights, manifest.model_config, matmul_storage_type);
    weights->flow_store->upload();

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
            conv_storage_type);
    weights->mimi_decoder_store->upload();

    manifest.model_weights->release_storage();
    return weights;
}

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

VoiceStateAssets load_voice_assets_for_plan(const VoiceConditioningPlan & plan, const PocketTTSAssets &) {
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

}  // namespace engine::models::pocket_tts
