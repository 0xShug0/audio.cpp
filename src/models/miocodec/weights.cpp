#include "engine/models/miocodec/weights.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/miocodec/assets.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace engine::models::miocodec {
namespace {

bool has_suffix(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int64_t num_elements(const std::vector<int64_t> & shape) {
    if (shape.empty() || shape.size() > engine::core::kMaxTensorRank) {
        throw std::runtime_error("MioCodec weight tensor rank must be between 1 and 4");
    }
    int64_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("MioCodec weight tensor dimensions must be positive");
        }
        count *= dim;
    }
    return count;
}

engine::core::TensorShape shape_from_vector(const std::vector<int64_t> & shape) {
    switch (shape.size()) {
        case 1:
            return engine::core::TensorShape::from_dims({shape[0]});
        case 2:
            return engine::core::TensorShape::from_dims({shape[0], shape[1]});
        case 3:
            return engine::core::TensorShape::from_dims({shape[0], shape[1], shape[2]});
        case 4:
            return engine::core::TensorShape::from_dims({shape[0], shape[1], shape[2], shape[3]});
        default:
            throw std::runtime_error("MioCodec tensor rank must be between 1 and 4");
    }
}

const engine::core::TensorValue & require_loaded_tensor(
    const MioCodecWeights & weights,
    const std::string & name,
    const std::vector<int64_t> & shape) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("MioCodec missing loaded tensor: " + name);
    }
    engine::core::validate_shape(it->second, shape_from_vector(shape), name.c_str());
    return it->second;
}

bool is_transformer_qkv_weight_name(const std::string & name) {
    if (name.find(".layers.") == std::string::npos) {
        return false;
    }
    const std::string suffixes[] = {
        ".attention.wq.weight",
        ".attention.wk.weight",
        ".attention.wv.weight",
    };
    for (const auto & suffix : suffixes) {
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

bool is_norm_weight_name(const std::string & name) {
    if (!has_suffix(name, ".weight")) {
        return false;
    }
    const std::string stem = name.substr(0, name.size() - std::string(".weight").size());
    const auto pos = stem.find_last_of('.');
    const std::string leaf = pos == std::string::npos ? stem : stem.substr(pos + 1);
    return leaf == "norm" ||
           leaf == "norm1" ||
           leaf == "norm2" ||
           leaf == "attention_norm" ||
           leaf == "ffn_norm" ||
           leaf == "final_layer_norm";
}

engine::assets::TensorStorageType storage_type_for_miocodec_tensor(
    const std::string & name,
    engine::assets::TensorStorageType requested_storage_type) {
    if (name == "wave_conv_upsample.weight" ||
        has_suffix(name, ".bias") ||
        has_suffix(name, ".gamma") ||
        has_suffix(name, ".alpha") ||
        has_suffix(name, ".beta") ||
        name.find(".parametrizations.weight.") != std::string::npos ||
        is_norm_weight_name(name)) {
        return engine::assets::TensorStorageType::F32;
    }
    return requested_storage_type;
}

int64_t product(const std::vector<int> & values) {
    if (values.empty()) {
        throw std::runtime_error("MioCodec quantizer levels must not be empty");
    }
    int64_t out = 1;
    for (const int value : values) {
        if (value <= 0) {
            throw std::runtime_error("MioCodec quantizer levels must be positive");
        }
        out *= value;
    }
    return out;
}

std::vector<int64_t> quantizer_basis(const std::vector<int> & levels) {
    std::vector<int64_t> basis(levels.size(), 1);
    for (size_t index = 1; index < levels.size(); ++index) {
        basis[index] = basis[index - 1] * levels[index - 1];
    }
    return basis;
}

MioCodecContentTokenEmbeddingTable build_content_token_embedding_table(
    const engine::assets::TensorSource & source,
    const MioCodecConfig & config) {
    const int64_t quantizer_dim = static_cast<int64_t>(config.quantizer_levels.size());
    if (quantizer_dim <= 0 || config.quantizer_input_dim <= 0 || config.quantizer_output_dim <= 0) {
        throw std::runtime_error("MioCodec content token embedding dimensions are invalid");
    }
    if (config.quantizer_output_dim != config.wave_prenet_dim) {
        throw std::runtime_error("MioCodec content token embedding config is inconsistent with the wave prenet");
    }
    const int64_t codebook_size = product(config.quantizer_levels);
    const int64_t output_dim = config.quantizer_output_dim;
    const auto proj_out_weight = source.require_f32("local_quantizer.proj_out.weight", {output_dim, quantizer_dim});
    const auto proj_out_bias = source.require_f32("local_quantizer.proj_out.bias", {output_dim});
    const auto basis = quantizer_basis(config.quantizer_levels);

    MioCodecContentTokenEmbeddingTable table;
    table.codebook_size = codebook_size;
    table.dim = output_dim;
    table.values.resize(static_cast<size_t>(codebook_size * output_dim), 0.0F);
    std::vector<float> fsq_code(static_cast<size_t>(quantizer_dim), 0.0F);
    for (int64_t token = 0; token < codebook_size; ++token) {
        for (int64_t dim = 0; dim < quantizer_dim; ++dim) {
            const int64_t level = config.quantizer_levels[static_cast<size_t>(dim)];
            const int64_t half = level / 2;
            if (half <= 0) {
                throw std::runtime_error("MioCodec content token embedding level is invalid");
            }
            const int64_t component = (token / basis[static_cast<size_t>(dim)]) % level;
            fsq_code[static_cast<size_t>(dim)] =
                (static_cast<float>(component) - static_cast<float>(half)) / static_cast<float>(half);
        }
        float * dst = table.values.data() + static_cast<size_t>(token * output_dim);
        for (int64_t out = 0; out < output_dim; ++out) {
            const float * weight = proj_out_weight.data() + static_cast<size_t>(out * quantizer_dim);
            float value = proj_out_bias[static_cast<size_t>(out)];
            for (int64_t dim = 0; dim < quantizer_dim; ++dim) {
                value += fsq_code[static_cast<size_t>(dim)] * weight[dim];
            }
            dst[out] = value;
        }
    }
    return table;
}

MioCodecLinearWeights bind_linear(
    const MioCodecWeights & weights,
    const std::string & prefix,
    int64_t in_features,
    int64_t out_features,
    bool use_bias = true) {
    MioCodecLinearWeights linear;
    linear.weight = require_loaded_tensor(weights, prefix + ".weight", {out_features, in_features});
    if (use_bias) {
        linear.bias = require_loaded_tensor(weights, prefix + ".bias", {out_features});
    }
    return linear;
}

MioCodecLinearWeights bind_qkv_linear(
    MioCodecWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t dim,
    engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native) {
        const auto q = source.require_tensor(prefix + ".attention.wq.weight", storage_type, {dim, dim});
        const auto k = source.require_tensor(prefix + ".attention.wk.weight", storage_type, {dim, dim});
        const auto v = source.require_tensor(prefix + ".attention.wv.weight", storage_type, {dim, dim});
        if (q.type != k.type || q.type != v.type) {
            throw std::runtime_error("MioCodec native QKV source tensors must use the same dtype: " + prefix);
        }
        std::vector<std::byte> packed;
        packed.reserve(q.bytes.size() + k.bytes.size() + v.bytes.size());
        packed.insert(packed.end(), q.bytes.begin(), q.bytes.end());
        packed.insert(packed.end(), k.bytes.begin(), k.bytes.end());
        packed.insert(packed.end(), v.bytes.begin(), v.bytes.end());
        MioCodecLinearWeights out;
        out.weight = weights.store->make_tensor(
            engine::core::TensorShape::from_dims({3 * dim, dim}),
            q.type,
            packed.data(),
            packed.size());
        ++weights.loaded_tensor_count;
        return out;
    }
    std::vector<float> qkv(static_cast<size_t>(3 * dim * dim), 0.0F);
    const auto q = source.require_f32(prefix + ".attention.wq.weight", {dim, dim});
    const auto k = source.require_f32(prefix + ".attention.wk.weight", {dim, dim});
    const auto v = source.require_f32(prefix + ".attention.wv.weight", {dim, dim});
    const size_t block = static_cast<size_t>(dim * dim);
    std::copy(q.begin(), q.end(), qkv.begin());
    std::copy(k.begin(), k.end(), qkv.begin() + static_cast<std::ptrdiff_t>(block));
    std::copy(v.begin(), v.end(), qkv.begin() + static_cast<std::ptrdiff_t>(2 * block));
    MioCodecLinearWeights packed;
    packed.weight = weights.store->make_from_f32(
        engine::core::TensorShape::from_dims({3 * dim, dim}),
        storage_type,
        std::move(qkv));
    ++weights.loaded_tensor_count;
    return packed;
}

MioCodecConv1dWeights bind_conv1d(
    const MioCodecWeights & weights,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    bool use_bias = true) {
    MioCodecConv1dWeights conv;
    conv.config = {
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        use_bias,
    };
    conv.weights.weight = require_loaded_tensor(weights, prefix + ".weight", {out_channels, in_channels, kernel_size});
    if (use_bias) {
        conv.weights.bias = require_loaded_tensor(weights, prefix + ".bias", {out_channels});
    }
    return conv;
}

MioCodecConvTranspose1dWeights bind_conv_transpose1d(
    const MioCodecWeights & weights,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    bool use_bias = true) {
    MioCodecConvTranspose1dWeights conv;
    conv.config = {
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        use_bias,
    };
    conv.weights.weight = require_loaded_tensor(weights, prefix + ".weight", {in_channels, out_channels, kernel_size});
    if (use_bias) {
        conv.weights.bias = require_loaded_tensor(weights, prefix + ".bias", {out_channels});
    }
    return conv;
}

std::vector<float> fold_weight_norm(
    const std::vector<float> & weight_g,
    const std::vector<float> & weight_v,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2) {
    if (static_cast<int64_t>(weight_g.size()) != dim0 ||
        static_cast<int64_t>(weight_v.size()) != dim0 * dim1 * dim2) {
        throw std::runtime_error("MioCodec weight-norm tensor shape mismatch");
    }
    std::vector<float> folded(weight_v.size(), 0.0F);
    for (int64_t d0 = 0; d0 < dim0; ++d0) {
        double norm_sq = 0.0;
        const size_t base = static_cast<size_t>(d0 * dim1 * dim2);
        for (int64_t index = 0; index < dim1 * dim2; ++index) {
            const float value = weight_v[base + static_cast<size_t>(index)];
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        if (norm_sq == 0.0) {
            throw std::runtime_error("MioCodec weight-norm source tensor has zero norm");
        }
        const float scale = weight_g[static_cast<size_t>(d0)] / static_cast<float>(std::sqrt(norm_sq));
        for (int64_t index = 0; index < dim1 * dim2; ++index) {
            folded[base + static_cast<size_t>(index)] = weight_v[base + static_cast<size_t>(index)] * scale;
        }
    }
    return folded;
}

MioCodecConvTranspose1dWeights bind_weight_norm_conv_transpose1d(
    MioCodecWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int stride,
    int padding,
    engine::assets::TensorStorageType storage_type) {
    const auto weight_g = source.require_f32(prefix + ".parametrizations.weight.original0", {in_channels, 1, 1});
    const auto weight_v = source.require_f32(prefix + ".parametrizations.weight.original1", {in_channels, out_channels, kernel_size});
    MioCodecConvTranspose1dWeights conv;
    conv.config = {
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        1,
        true,
    };
    conv.weights.weight = weights.store->make_from_f32(
        engine::core::TensorShape::from_dims({in_channels, out_channels, kernel_size}),
        storage_type,
        fold_weight_norm(weight_g, weight_v, in_channels, out_channels, kernel_size));
    conv.weights.bias = require_loaded_tensor(weights, prefix + ".bias", {out_channels});
    return conv;
}

MioCodecNormWeights bind_norm(const MioCodecWeights & weights, const std::string & prefix, int64_t hidden) {
    return {
        require_loaded_tensor(weights, prefix + ".weight", {hidden}),
        require_loaded_tensor(weights, prefix + ".bias", {hidden}),
    };
}

MioCodecAdaLayerNormWeights bind_adaln(
    const MioCodecWeights & weights,
    const std::string & prefix,
    int64_t condition_dim,
    int64_t output_dim) {
    return {bind_linear(weights, prefix + ".condition_proj.1", condition_dim, output_dim)};
}

int64_t swiglu_hidden_dim(int64_t dim) {
    int64_t hidden = static_cast<int64_t>(2 * (4 * dim) / 3);
    constexpr int64_t multiple_of = 256;
    return multiple_of * ((hidden + multiple_of - 1) / multiple_of);
}

MioCodecTransformerWeights bind_transformer(
    MioCodecWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t dim,
    int64_t layers,
    int64_t heads,
    int64_t window_size,
    bool use_adaln,
    int64_t condition_dim = 0,
    std::optional<int64_t> output_dim = std::nullopt,
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::F32) {
    MioCodecTransformerWeights transformer;
    transformer.dim = dim;
    transformer.heads = heads;
    transformer.head_dim = dim / heads;
    transformer.window_size = window_size;
    transformer.intermediate_dim = swiglu_hidden_dim(dim);
    transformer.use_adaln = use_adaln;
    transformer.layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string layer_prefix = prefix + ".layers." + std::to_string(layer);
        MioCodecTransformerLayerWeights w;
        w.qkv_proj = bind_qkv_linear(weights, source, layer_prefix, dim, storage_type);
        w.out_proj = bind_linear(weights, layer_prefix + ".attention.wo", dim, dim, false);
        w.feed_forward_w1 = bind_linear(weights, layer_prefix + ".feed_forward.w1", dim, transformer.intermediate_dim, false);
        w.feed_forward_w2 = bind_linear(weights, layer_prefix + ".feed_forward.w2", transformer.intermediate_dim, dim, false);
        w.feed_forward_w3 = bind_linear(weights, layer_prefix + ".feed_forward.w3", dim, transformer.intermediate_dim, false);
        if (use_adaln) {
            w.attention_adaln = bind_adaln(weights, layer_prefix + ".attention_norm", condition_dim, 3 * dim);
            w.feed_forward_adaln = bind_adaln(weights, layer_prefix + ".ffn_norm", condition_dim, 3 * dim);
        } else {
            w.attention_norm = bind_norm(weights, layer_prefix + ".attention_norm", dim);
            w.feed_forward_norm = bind_norm(weights, layer_prefix + ".ffn_norm", dim);
        }
        transformer.layers.push_back(std::move(w));
    }
    if (use_adaln) {
        transformer.adaln_norm = bind_adaln(weights, prefix + ".norm", condition_dim, 2 * dim);
    } else {
        transformer.norm = bind_norm(weights, prefix + ".norm", dim);
    }
    if (output_dim.has_value()) {
        transformer.output_projection = bind_linear(weights, prefix + ".output_proj", dim, *output_dim);
    }
    return transformer;
}

MioCodecSnakeBetaWeights bind_snake_beta(
    MioCodecWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    std::vector<float> alpha(static_cast<size_t>(channels), 0.0F);
    std::vector<float> inv_beta(static_cast<size_t>(channels), 0.0F);
    const auto alpha_source = source.require_f32(prefix + ".alpha", {channels});
    const auto beta_source = source.require_f32(prefix + ".beta", {channels});
    for (int64_t i = 0; i < channels; ++i) {
        alpha[static_cast<size_t>(i)] = std::exp(alpha_source[static_cast<size_t>(i)]);
        inv_beta[static_cast<size_t>(i)] = 1.0F / (std::exp(beta_source[static_cast<size_t>(i)]) + 1.0e-9F);
    }
    return {
        weights.store->make_f32(engine::core::TensorShape::from_dims({channels}), std::move(alpha)),
        weights.store->make_f32(engine::core::TensorShape::from_dims({channels}), std::move(inv_beta)),
    };
}

MioCodecResNetBlockWeights bind_resnet_block(
    const MioCodecWeights & weights,
    const std::string & prefix,
    int64_t channels) {
    return {
        bind_norm(weights, prefix + ".norm1", channels),
        bind_conv1d(weights, prefix + ".conv1", channels, channels, 3, 1, 1),
        bind_norm(weights, prefix + ".norm2", channels),
        bind_conv1d(weights, prefix + ".conv2", channels, channels, 3, 1, 1),
    };
}

MioCodecResNetStackWeights bind_resnet_stack(
    const MioCodecWeights & weights,
    const std::string & prefix,
    int64_t channels,
    int64_t blocks) {
    MioCodecResNetStackWeights stack;
    stack.blocks.reserve(static_cast<size_t>(blocks));
    for (int64_t block = 0; block < blocks; ++block) {
        stack.blocks.push_back(bind_resnet_block(weights, prefix + ".blocks." + std::to_string(block), channels));
    }
    return stack;
}

MioCodecUpsamplerWeights bind_upsampler(
    MioCodecWeights & weights,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType storage_type) {
    MioCodecUpsamplerWeights upsampler;
    const int64_t factors[] = {3, 3};
    const int64_t kernels[] = {9, 9};
    int64_t in_channels = 512;
    upsampler.stages.reserve(2);
    for (int64_t stage = 0; stage < 2; ++stage) {
        const int64_t out_channels = in_channels / 2;
        const std::string stage_index = std::to_string(stage);
        MioCodecUpsamplerStageWeights item;
        item.upsample = bind_weight_norm_conv_transpose1d(
            weights,
            source,
            "wave_upsampler.upsample_layers." + stage_index,
            in_channels,
            out_channels,
            kernels[stage],
            static_cast<int>(factors[stage]),
            static_cast<int>((kernels[stage] - factors[stage]) / 2),
            storage_type);
        item.snake = bind_snake_beta(weights, source, "wave_upsampler.snake_activations." + stage_index, out_channels);
        item.resnet = bind_resnet_block(weights, "wave_upsampler.resnet_blocks." + stage_index, out_channels);
        upsampler.stages.push_back(std::move(item));
        in_channels = out_channels;
    }
    upsampler.output_projection = bind_linear(weights, "wave_upsampler.out_proj", 128, 512);
    upsampler.output_snake = bind_snake_beta(weights, source, "wave_upsampler.out_snake", 512);
    return upsampler;
}

MioCodecGlobalEncoderWeights bind_global_encoder(const MioCodecWeights & weights) {
    MioCodecGlobalEncoderWeights global;
    global.embedding = bind_conv1d(weights, "global_encoder.backbone.embed", 384, 768, 7, 1, 3);
    global.embedding_norm = bind_norm(weights, "global_encoder.backbone.norm", 384);
    global.blocks.reserve(4);
    for (int64_t layer = 0; layer < 4; ++layer) {
        const std::string prefix = "global_encoder.backbone.convnext." + std::to_string(layer);
        MioCodecConvNeXtBlockWeights block;
        block.depthwise_conv_config = {384, 7, 1, 3, 1, true};
        block.depthwise_conv = {
            require_loaded_tensor(weights, prefix + ".dwconv.weight", {384, 1, 7}),
            require_loaded_tensor(weights, prefix + ".dwconv.bias", {384}),
        };
        block.norm = bind_norm(weights, prefix + ".norm", 384);
        block.pointwise_conv1 = bind_linear(weights, prefix + ".pwconv1", 384, 1152);
        block.pointwise_conv2 = bind_linear(weights, prefix + ".pwconv2", 1152, 384);
        block.gamma = require_loaded_tensor(weights, prefix + ".gamma", {384});
        global.blocks.push_back(std::move(block));
    }
    global.final_norm = bind_norm(weights, "global_encoder.backbone.final_layer_norm", 384);
    global.attention_conv1 = bind_conv1d(weights, "global_encoder.pooling.attn.0", 128, 384, 1);
    global.attention_conv2 = bind_conv1d(weights, "global_encoder.pooling.attn.2", 384, 128, 1);
    global.pooling_projection = bind_linear(weights, "global_encoder.pooling.proj", 768, 128);
    global.pooling_norm = bind_norm(weights, "global_encoder.pooling.norm", 128);
    return global;
}

void bind_component_weights(
    MioCodecWeights & weights,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType storage_type) {
    weights.local_encoder = bind_transformer(weights, source, "local_encoder", 768, 6, 12, 125, false, 0, std::nullopt, storage_type);
    weights.local_quantizer = {
        bind_linear(weights, "local_quantizer.proj_in", 768, 5),
        bind_linear(weights, "local_quantizer.proj_out", 5, 768),
    };
    weights.conv_downsample = bind_conv1d(weights, "conv_downsample", 768, 768, 2, 2);
    weights.global_encoder = bind_global_encoder(weights);
    weights.wave_prenet = bind_transformer(weights, source, "wave_prenet", 768, 6, 12, 65, false, 0, 512, storage_type);
    weights.wave_conv_upsample = bind_conv_transpose1d(weights, "wave_conv_upsample", 512, 512, 2, 2);
    weights.wave_prior_net = bind_resnet_stack(weights, "wave_prior_net", 512, 2);
    weights.wave_decoder = bind_transformer(weights, source, "wave_decoder", 512, 8, 8, 65, true, 128, std::nullopt, storage_type);
    weights.wave_post_net = bind_resnet_stack(weights, "wave_post_net", 512, 2);
    weights.wave_upsampler = bind_upsampler(weights, source, storage_type);
    weights.istft_head = {bind_linear(weights, "istft_head.out", 512, 394)};
}

}  // namespace

std::shared_ptr<const MioCodecWeights> load_miocodec_weights(
    const engine::assets::TensorSource & source,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    const MioCodecConfig & config,
    engine::assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<MioCodecWeights>();
    weights->source_path = source.source_path();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "miocodec.weights",
        weight_context_bytes);

    const auto tensors = source.tensors();
    weights->tensors.reserve(tensors.size());
    for (const auto & tensor : tensors) {
        weights->parameter_count += num_elements(tensor.shape);
        if (is_transformer_qkv_weight_name(tensor.name)) {
            continue;
        }
        weights->tensors.emplace(
            tensor.name,
            weights->store->load_tensor(
                source,
                tensor.name,
                storage_type_for_miocodec_tensor(tensor.name, storage_type),
                tensor.shape));
        ++weights->loaded_tensor_count;
    }
    bind_component_weights(*weights, source, storage_type);
    weights->content_token_embeddings = build_content_token_embedding_table(source, config);
    weights->store->upload();
    source.release_storage();
    return weights;
}

}  // namespace engine::models::miocodec
