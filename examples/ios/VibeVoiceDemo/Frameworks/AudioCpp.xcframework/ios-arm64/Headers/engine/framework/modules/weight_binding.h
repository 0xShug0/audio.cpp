#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace engine::modules::binding {

inline LinearConfig linear_config(
    int64_t in_features,
    int64_t out_features,
    bool use_bias) {
    return {in_features, out_features, use_bias, GGML_PREC_DEFAULT};
}

template <typename Store, typename Values>
core::TensorValue tensor(Store & store, const core::TensorShape & shape, const Values & values) {
    return store.make_f32(shape, values);
}

template <typename Store, typename TensorData>
core::TensorValue tensor_data(Store & store, const TensorData & data) {
    return store.make_f32(data.shape, data.values);
}

template <typename Store>
core::TensorValue tensor_data(Store &, const core::TensorValue & data) {
    return data;
}

template <typename Store>
core::TensorValue tensor_data(Store & store, const assets::TensorData & data) {
    return store.make_tensor(data.shape, data.type, data.bytes.data(), data.bytes.size());
}

inline std::vector<int64_t> tensor_shape_from_source(
    const assets::TensorSource & source,
    const std::string & name) {
    return source.require_metadata(name).shape;
}

template <typename Store>
core::TensorValue tensor_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & name,
    assets::TensorStorageType storage_type) {
    return store.load_tensor(source, name, storage_type, tensor_shape_from_source(source, name));
}

template <typename Store>
core::TensorValue f32_tensor_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & name) {
    return store.load_f32_tensor(source, name, tensor_shape_from_source(source, name));
}

template <typename Store, typename TensorData>
LinearWeights linear_data(Store & store, const TensorData & weight) {
    return {tensor_data(store, weight), std::nullopt};
}

template <typename Store, typename WeightData, typename BiasData>
LinearWeights linear_data(Store & store, const WeightData & weight, const BiasData & bias) {
    return {tensor_data(store, weight), tensor_data(store, bias)};
}

template <typename Store, typename TensorData>
LinearWeights linear_data(Store & store, const TensorData & weight, const std::optional<TensorData> & bias) {
    return bias.has_value() ? linear_data(store, weight, *bias) : linear_data(store, weight);
}

template <typename Store, typename WeightValues>
LinearWeights linear(
    Store & store,
    int64_t in_features,
    int64_t out_features,
    const WeightValues & weight) {
    return {
        tensor(store, core::TensorShape::from_dims({out_features, in_features}), weight),
        std::nullopt,
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
LinearWeights linear(
    Store & store,
    int64_t in_features,
    int64_t out_features,
    const WeightValues & weight,
    const BiasValues & bias) {
    return {
        tensor(store, core::TensorShape::from_dims({out_features, in_features}), weight),
        tensor(store, core::TensorShape::from_dims({out_features}), bias),
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
LinearWeights linear(
    Store & store,
    int64_t in_features,
    int64_t out_features,
    bool use_bias,
    const WeightValues & weight,
    const BiasValues & bias) {
    return use_bias
        ? linear(store, in_features, out_features, weight, bias)
        : linear(store, in_features, out_features, weight);
}

template <typename Store, typename TensorData>
NormWeights norm_data(Store & store, const TensorData & weight) {
    return {tensor_data(store, weight), std::nullopt};
}

template <typename Store, typename TensorData>
NormWeights norm_data(Store & store, const TensorData & weight, const TensorData & bias) {
    return {tensor_data(store, weight), tensor_data(store, bias)};
}

template <typename Store, typename WeightValues>
NormWeights norm(
    Store & store,
    const WeightValues & weight) {
    return {
        tensor(store, core::TensorShape::from_dims({static_cast<int64_t>(weight.size())}), weight),
        std::nullopt,
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
NormWeights norm(
    Store & store,
    const WeightValues & weight,
    const BiasValues & bias) {
    return {
        tensor(store, core::TensorShape::from_dims({static_cast<int64_t>(weight.size())}), weight),
        tensor(store, core::TensorShape::from_dims({static_cast<int64_t>(bias.size())}), bias),
    };
}

template <typename Store, typename TensorData>
Conv1dWeights conv1d_data(Store & store, const TensorData & weight) {
    return {tensor_data(store, weight), std::nullopt};
}

template <typename Store, typename WeightData, typename BiasData>
Conv1dWeights conv1d_data(Store & store, const WeightData & weight, const BiasData & bias) {
    return {tensor_data(store, weight), tensor_data(store, bias)};
}

template <typename Store, typename TensorData>
Conv1dWeights conv1d_data(Store & store, const TensorData & weight, const std::optional<TensorData> & bias) {
    return bias.has_value() ? conv1d_data(store, weight, *bias) : conv1d_data(store, weight);
}

template <typename Store, typename TensorData>
Conv2dWeights conv2d_data(Store & store, const TensorData & weight) {
    return {tensor_data(store, weight), std::nullopt};
}

template <typename Store, typename WeightData, typename BiasData>
Conv2dWeights conv2d_data(Store & store, const WeightData & weight, const BiasData & bias) {
    return {tensor_data(store, weight), tensor_data(store, bias)};
}

template <typename Store, typename TensorData>
Conv2dWeights conv2d_data(Store & store, const TensorData & weight, const std::optional<TensorData> & bias) {
    return bias.has_value() ? conv2d_data(store, weight, *bias) : conv2d_data(store, weight);
}

template <typename Store, typename WeightValues>
Conv1dWeights conv1d(
    Store & store,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t groups,
    const WeightValues & weight) {
    return {
        tensor(store, core::TensorShape::from_dims({out_channels, in_channels / groups, kernel}), weight),
        std::nullopt,
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
Conv1dWeights conv1d(
    Store & store,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t groups,
    const WeightValues & weight,
    const BiasValues & bias) {
    return {
        tensor(store, core::TensorShape::from_dims({out_channels, in_channels / groups, kernel}), weight),
        tensor(store, core::TensorShape::from_dims({out_channels}), bias),
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
Conv1dWeights conv1d(
    Store & store,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t groups,
    bool use_bias,
    const WeightValues & weight,
    const BiasValues & bias) {
    return use_bias
        ? conv1d(store, in_channels, out_channels, kernel, groups, weight, bias)
        : conv1d(store, in_channels, out_channels, kernel, groups, weight);
}

template <typename Store, typename WeightValues>
ConvTranspose1dWeights conv_transpose1d(
    Store & store,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    const WeightValues & weight) {
    return {
        tensor(store, core::TensorShape::from_dims({in_channels, out_channels, kernel}), weight),
        std::nullopt,
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
ConvTranspose1dWeights conv_transpose1d(
    Store & store,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    const WeightValues & weight,
    const BiasValues & bias) {
    return {
        tensor(store, core::TensorShape::from_dims({in_channels, out_channels, kernel}), weight),
        tensor(store, core::TensorShape::from_dims({out_channels}), bias),
    };
}

template <typename Store, typename WeightValues, typename BiasValues>
ConvTranspose1dWeights conv_transpose1d(
    Store & store,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    bool use_bias,
    const WeightValues & weight,
    const BiasValues & bias) {
    return use_bias
        ? conv_transpose1d(store, in_channels, out_channels, kernel, weight, bias)
        : conv_transpose1d(store, in_channels, out_channels, kernel, weight);
}

template <typename Store, typename TensorData>
ConvTranspose1dWeights conv_transpose1d_data(Store & store, const TensorData & weight) {
    return {tensor_data(store, weight), std::nullopt};
}

template <typename Store, typename WeightData, typename BiasData>
ConvTranspose1dWeights conv_transpose1d_data(Store & store, const WeightData & weight, const BiasData & bias) {
    return {tensor_data(store, weight), tensor_data(store, bias)};
}

template <typename Store, typename TensorData>
ConvTranspose1dWeights conv_transpose1d_data(Store & store, const TensorData & weight, const std::optional<TensorData> & bias) {
    return bias.has_value() ? conv_transpose1d_data(store, weight, *bias) : conv_transpose1d_data(store, weight);
}

template <typename Store>
LinearWeights linear_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

template <typename Store>
LinearWeights hf_conv1d_linear_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_features,
    int64_t out_features,
    bool use_bias) {
    const auto source_weight = source.require_f32(prefix + ".weight", {in_features, out_features});
    std::vector<float> transposed(static_cast<std::size_t>(out_features * in_features));
    for (int64_t in = 0; in < in_features; ++in) {
        for (int64_t out = 0; out < out_features; ++out) {
            transposed[static_cast<std::size_t>(out * in_features + in)] =
                source_weight[static_cast<std::size_t>(in * out_features + out)];
        }
    }
    LinearWeights weights;
    weights.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_features, in_features}),
        storage_type,
        std::move(transposed));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

template <typename Store>
NormWeights norm_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden_size}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden_size}),
    };
}

template <typename Store>
NormWeights norm_weight_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden_size}),
        std::nullopt,
    };
}

template <typename Store>
Conv1dWeights conv1d_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    bool use_bias) {
    Conv1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels, kernel_size});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

template <typename Store>
Conv2dWeights conv2d_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    bool use_bias) {
    Conv2dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels, kernel_height, kernel_width});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

template <typename Store>
DepthwiseConv1dWeights depthwise_conv1d_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels,
    int64_t kernel_size,
    bool use_bias) {
    DepthwiseConv1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {channels, 1, kernel_size});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {channels});
    }
    return weights;
}

template <typename Store>
ConvTranspose1dWeights conv_transpose1d_from_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool use_bias) {
    ConvTranspose1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {in_channels, out_channels, kernel_size});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

template <typename Store>
LinearWeights linear_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & weight_name,
    const std::optional<std::string> & bias_name,
    assets::TensorStorageType storage_type) {
    LinearWeights weights;
    weights.weight = tensor_from_named_source(store, source, weight_name, storage_type);
    if (bias_name.has_value()) {
        weights.bias = f32_tensor_from_named_source(store, source, *bias_name);
    }
    return weights;
}

template <typename Store>
NormWeights norm_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & weight_name,
    const std::optional<std::string> & bias_name) {
    NormWeights weights;
    weights.weight = f32_tensor_from_named_source(store, source, weight_name);
    if (bias_name.has_value()) {
        weights.bias = f32_tensor_from_named_source(store, source, *bias_name);
    }
    return weights;
}

template <typename Store>
Conv1dWeights conv1d_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & weight_name,
    const std::optional<std::string> & bias_name,
    assets::TensorStorageType storage_type) {
    Conv1dWeights weights;
    weights.weight = tensor_from_named_source(store, source, weight_name, storage_type);
    if (bias_name.has_value()) {
        weights.bias = f32_tensor_from_named_source(store, source, *bias_name);
    }
    return weights;
}

template <typename Store>
ConvTranspose1dWeights conv_transpose1d_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & weight_name,
    const std::optional<std::string> & bias_name,
    assets::TensorStorageType storage_type) {
    ConvTranspose1dWeights weights;
    weights.weight = tensor_from_named_source(store, source, weight_name, storage_type);
    if (bias_name.has_value()) {
        weights.bias = f32_tensor_from_named_source(store, source, *bias_name);
    }
    return weights;
}

template <typename Store>
LayerScaleWeights layer_scale_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & name) {
    return {f32_tensor_from_named_source(store, source, name)};
}

template <typename Store>
std::optional<LayerScaleWeights> optional_layer_scale_from_named_source(
    Store & store,
    const assets::TensorSource & source,
    const std::string & name) {
    if (!source.has_tensor(name)) {
        return std::nullopt;
    }
    return layer_scale_from_named_source(store, source, name);
}

template <typename Store, typename TensorData>
LayerScaleWeights layer_scale_data(Store & store, const TensorData & scale) {
    return {tensor_data(store, scale)};
}

}  // namespace engine::modules::binding
