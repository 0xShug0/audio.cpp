#include "engine/models/niagara_asr/weights.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::niagara_asr {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;
namespace binding = engine::modules::binding;

constexpr float kBatchNormEpsilon = 1.0e-5f;

std::string lmuformer_prefix(int64_t layer) {
    if (layer == 0) {
        return "p__torch_params_encoder_lmuformer_";
    }
    return "p__torch_params_encoder_lmuformer_" + std::to_string(layer) + "_";
}

std::string state_space_basis_name(int64_t layer) {
    return "c_lifted_tensor_" + std::to_string(44 + 40 * layer);
}

std::string relative_position_name(int64_t layer) {
    return "c_lifted_tensor_" + std::to_string(34 + 40 * layer);
}

std::string indexed_layer_name(
    const std::string & prefix,
    const std::string & base,
    int64_t layer,
    bool omit_zero) {
    if (layer == 0 && omit_zero) {
        return prefix + base;
    }
    return prefix + base + "_" + std::to_string(layer);
}

std::vector<float> build_state_space_conv_kernel(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t layer,
    const NiagaraEncoderConfig & config) {
    constexpr int64_t kernel_size = 128;
    const auto basis = source.require_f32(
        state_space_basis_name(layer),
        {1, kernel_size, config.state_size});
    const auto c = source.require_f32(
        prefix + "state_space_state_space_c",
        {config.state_channels, config.state_size, 1});
    std::vector<float> kernel(static_cast<size_t>(config.state_channels * kernel_size), 0.0f);
    for (int64_t channel = 0; channel < config.state_channels; ++channel) {
        for (int64_t k = 0; k < kernel_size; ++k) {
            double sum = 0.0;
            const int64_t source_k = kernel_size - 1 - k;
            for (int64_t s = 0; s < config.state_size; ++s) {
                sum += static_cast<double>(basis[static_cast<size_t>(source_k * config.state_size + s)]) *
                    static_cast<double>(c[static_cast<size_t>(channel * config.state_size + s)]);
            }
            kernel[static_cast<size_t>(channel * kernel_size + k)] = static_cast<float>(sum);
        }
    }
    return kernel;
}

modules::Conv2dWeights load_conv2d_hwio(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t in_channels,
    int64_t out_channels,
    assets::TensorStorageType storage_type,
    bool has_bias = true) {
    const auto raw = source.require_f32(prefix + "_kernel", {kernel_height, kernel_width, in_channels, out_channels});
    std::vector<float> converted(static_cast<size_t>(out_channels * in_channels * kernel_height * kernel_width), 0.0f);
    for (int64_t out = 0; out < out_channels; ++out) {
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t kh = 0; kh < kernel_height; ++kh) {
                for (int64_t kw = 0; kw < kernel_width; ++kw) {
                    converted[static_cast<size_t>(((out * in_channels + in) * kernel_height + kh) * kernel_width + kw)] =
                        raw[static_cast<size_t>(((kh * kernel_width + kw) * in_channels + in) * out_channels + out)];
                }
            }
        }
    }
    return {
        store.make_from_f32(
            core::TensorShape::from_dims({out_channels, in_channels, kernel_height, kernel_width}),
            storage_type,
            std::move(converted)),
        has_bias ? std::optional<core::TensorValue>(store.load_f32_tensor(source, prefix + "_bias", {out_channels}))
                 : std::nullopt,
    };
}

modules::BatchNorm2dEvalWeights load_batch_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    const auto gamma = source.require_f32(prefix + "_gamma", {channels});
    const auto beta = source.require_f32(prefix + "_beta", {channels});
    const auto mean = source.require_f32(prefix + "_moving_mean", {channels});
    const auto variance = source.require_f32(prefix + "_moving_variance", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0f);
    std::vector<float> bias(static_cast<size_t>(channels), 0.0f);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const float folded_scale =
            gamma[static_cast<size_t>(channel)] / std::sqrt(variance[static_cast<size_t>(channel)] + kBatchNormEpsilon);
        scale[static_cast<size_t>(channel)] = folded_scale;
        bias[static_cast<size_t>(channel)] = beta[static_cast<size_t>(channel)] - mean[static_cast<size_t>(channel)] * folded_scale;
    }
    return {
        store.make_from_f32(core::TensorShape::from_dims({channels}), assets::TensorStorageType::F32, std::move(scale)),
        store.make_from_f32(core::TensorShape::from_dims({channels}), assets::TensorStorageType::F32, std::move(bias)),
    };
}

NiagaraFeedForwardWeights load_feed_forward(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const std::string & residual_scale_name,
    int64_t hidden_size,
    int64_t intermediate_size,
    assets::TensorStorageType storage_type) {
    const auto norm = prefix + "l1norm";
    return {
        binding::norm_from_named_source(
            store,
            source,
            norm + "_gamma",
            std::optional<std::string>{norm + "_beta"}),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "dense_1_kernel",
            std::optional<std::string>{prefix + "dense_1_bias"},
            storage_type,
            hidden_size,
            intermediate_size),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "dense_2_kernel",
            std::optional<std::string>{prefix + "dense_2_bias"},
            storage_type,
            intermediate_size,
            hidden_size),
        store.load_f32_tensor(source, residual_scale_name, {1}),
    };
}

NiagaraSelfAttentionWeights load_self_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t layer,
    int64_t hidden_size,
    assets::TensorStorageType storage_type) {
    const auto norm = indexed_layer_name(prefix, "self_attention_l1_layer_normalization", layer * 2, true);
    const auto global_attention = indexed_layer_name(prefix, "self_attention_global_attention", layer, true);
    const auto out = indexed_layer_name(prefix, "self_attention_dense", layer + 1, false);
    return {
        binding::norm_from_named_source(
            store,
            source,
            norm + "_gamma",
            std::optional<std::string>{norm + "_beta"}),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "self_attention_query_kernel",
            std::optional<std::string>{prefix + "self_attention_query_bias"},
            storage_type,
            hidden_size,
            hidden_size),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "self_attention_key_kernel",
            std::optional<std::string>{prefix + "self_attention_key_bias"},
            storage_type,
            hidden_size,
            hidden_size),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "self_attention_value_kernel",
            std::optional<std::string>{prefix + "self_attention_value_bias"},
            storage_type,
            hidden_size,
            hidden_size),
        binding::linear_from_transposed_named_source(
            store,
            source,
            global_attention + "_kernel",
            std::optional<std::string>{global_attention + "_bias"},
            storage_type,
            hidden_size,
            hidden_size),
        binding::linear_from_transposed_named_source(
            store,
            source,
            out + "_kernel",
            std::optional<std::string>{out + "_bias"},
            storage_type,
            hidden_size,
            hidden_size),
        store.load_tensor(
            source,
            relative_position_name(layer),
            assets::TensorStorageType::F32,
            {5999, hidden_size}),
        store.load_f32_tensor(source, prefix + "self_attention_u_bias_bias", {hidden_size}),
        store.load_f32_tensor(source, prefix + "self_attention_v_bias_bias", {hidden_size}),
    };
}

NiagaraStateSpaceWeights load_state_space(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t layer,
    const NiagaraEncoderConfig & config,
    assets::TensorStorageType storage_type) {
    constexpr int64_t kernel_size = 128;
    const auto norm = prefix + "state_space_l1norm";
    return {
        binding::norm_from_named_source(
            store,
            source,
            norm + "_gamma",
            std::optional<std::string>{norm + "_beta"}),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "state_space_dense_1_kernel",
            std::optional<std::string>{prefix + "state_space_dense_1_bias"},
            storage_type,
            config.hidden_size,
            config.state_channels),
        binding::linear_from_transposed_named_source(
            store,
            source,
            prefix + "state_space_dense_2_kernel",
            std::optional<std::string>{prefix + "state_space_dense_2_bias"},
            storage_type,
            config.hidden_size,
            config.hidden_size),
        store.make_from_f32(
            core::TensorShape::from_dims({config.state_channels, 1, kernel_size, 1}),
            assets::TensorStorageType::F32,
            build_state_space_conv_kernel(source, prefix, layer, config)),
    };
}

NiagaraLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer,
    const NiagaraEncoderConfig & config,
    assets::TensorStorageType storage_type) {
    const auto prefix = lmuformer_prefix(layer);
    const auto out_norm = indexed_layer_name(prefix, "l1_layer_normalization", layer * 2 + 1, false);
    return {
        load_feed_forward(
            store,
            source,
            prefix + "ffn_",
            "c_lifted_tensor_" + std::to_string(23 + 40 * layer),
            config.hidden_size,
            config.intermediate_size,
            storage_type),
        load_self_attention(store, source, prefix, layer, config.hidden_size, storage_type),
        load_state_space(store, source, prefix, layer, config, storage_type),
        load_feed_forward(
            store,
            source,
            prefix + "ffn_1_",
            "c_lifted_tensor_" + std::to_string(54 + 40 * layer),
            config.hidden_size,
            config.intermediate_size,
            storage_type),
        binding::norm_from_named_source(
            store,
            source,
            out_norm + "_gamma",
            std::optional<std::string>{out_norm + "_beta"}),
    };
}

NiagaraSubsamplingWeights load_subsampling(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const NiagaraEncoderConfig & config,
    assets::TensorStorageType storage_type) {
    const int64_t hidden = config.hidden_size;
    return {
        load_batch_norm(store, source, "p__torch_params_encoder_subsampling_batch_normalization", hidden),
        load_batch_norm(store, source, "p__torch_params_encoder_subsampling_batch_normalization_1", hidden),
        load_conv2d_hwio(
            store,
            source,
            "p__torch_params_encoder_subsampling_mask_aware_conv2d",
            1,
            1,
            1,
            4,
            storage_type,
            false),
        load_conv2d_hwio(
            store,
            source,
            "p__torch_params_encoder_subsampling_mask_aware_conv2d_1",
            config.subsampling_kernel_height,
            config.subsampling_kernel_width,
            4,
            hidden,
            storage_type),
        load_conv2d_hwio(
            store,
            source,
            "p__torch_params_encoder_subsampling_mask_aware_conv2d_2",
            config.subsampling_kernel_height,
            config.subsampling_kernel_width,
            hidden,
            hidden,
            storage_type),
        binding::linear_from_transposed_named_source(
            store,
            source,
            "p__torch_params_encoder_dense_kernel",
            std::optional<std::string>{"p__torch_params_encoder_dense_bias"},
            storage_type,
            config.dense_input_features * hidden,
            hidden),
    };
}

}  // namespace

std::shared_ptr<const NiagaraWeights> load_niagara_weights(
    const NiagaraAsrAssets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t weight_context_bytes) {
    if (assets.source == nullptr) {
        throw std::runtime_error("Niagara ASR weights require a tensor source");
    }
    auto out = std::make_shared<NiagaraWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "niagara_asr.weights",
        weight_context_bytes);
    const auto & source = *assets.source;
    const auto & config = assets.config.encoder;
    out->subsampling = load_subsampling(*out->store, source, config, storage_type);
    out->layers.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        out->layers.push_back(load_layer(*out->store, source, layer, config, storage_type));
    }
    out->ctc = binding::linear_from_transposed_named_source(
        *out->store,
        source,
        "p__torch_params_shared_dec_kernel",
        std::optional<std::string>{"p__torch_params_shared_dec_bias"},
        storage_type,
        config.hidden_size,
        assets.config.vocab_size + 1);
    out->store->upload();
    return out;
}

}  // namespace engine::models::niagara_asr
