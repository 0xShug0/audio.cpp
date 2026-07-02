#include "components/component_weights.h"

#include "engine/framework/modules/hift_vocoder.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace engine::models::chatterbox {
namespace {

using FrameworkHiFTComponent = engine::modules::HiftVocoderComponent;
using FrameworkHiFTConfig = engine::modules::HiftVocoderConfig;
using FrameworkHiFTWeights = engine::modules::HiftVocoderWeights;
using Conv1dWeights = FrameworkHiFTWeights::Conv1dWeights;
using ConvTranspose1dWeights = FrameworkHiFTWeights::ConvTranspose1dWeights;
using LinearWeights = FrameworkHiFTWeights::LinearWeights;
using ResBlockWeights = FrameworkHiFTWeights::ResBlockWeights;
using SnakeWeights = FrameworkHiFTWeights::SnakeWeights;

bool is_float_dtype(const std::string & dtype) {
    return dtype == "F32" || dtype == "F16" || dtype == "BF16";
}

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("Chatterbox HiFT tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("Chatterbox HiFT tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

std::vector<float> apply_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t leading,
    int64_t inner_size) {
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t i = 0; i < leading; ++i) {
        double norm = 0.0;
        for (int64_t j = 0; j < inner_size; ++j) {
            const float value = v[static_cast<size_t>(i * inner_size + j)];
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = g[static_cast<size_t>(i)] / std::sqrt(static_cast<float>(norm) + 1.0e-12F);
        for (int64_t j = 0; j < inner_size; ++j) {
            weight[static_cast<size_t>(i * inner_size + j)] =
                v[static_cast<size_t>(i * inner_size + j)] * scale;
        }
    }
    return weight;
}

LinearWeights load_hift_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight = components::read_f32_tensor(source, prefix + ".weight", {out_features, in_features});
    linear.weight_tensor = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_features, in_features}),
        weight_storage_type,
        linear.weight);
    if (use_bias) {
        linear.bias = components::read_f32_tensor(source, prefix + ".bias", {out_features});
        linear.bias_tensor = store.make_f32(engine::core::TensorShape::from_dims({out_features}), linear.bias);
    }
    return linear;
}

Conv1dWeights load_hift_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    conv.weight = components::read_f32_tensor(source, prefix + ".weight", {out_channels, in_channels, kernel});
    conv.weight_tensor = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        weight_storage_type,
        conv.weight);
    if (use_bias) {
        conv.bias = components::read_f32_tensor(source, prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_f32(engine::core::TensorShape::from_dims({out_channels}), conv.bias);
    }
    return conv;
}

Conv1dWeights load_hift_weight_norm_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    auto g = components::read_f32_tensor(source, prefix + ".parametrizations.weight.original0", {out_channels, 1, 1});
    auto v = components::read_f32_tensor(source, prefix + ".parametrizations.weight.original1", {out_channels, in_channels, kernel});
    Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    conv.weight = apply_weight_norm(g, v, out_channels, in_channels * kernel);
    conv.weight_tensor = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        weight_storage_type,
        conv.weight);
    if (use_bias) {
        conv.bias = components::read_f32_tensor(source, prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_f32(engine::core::TensorShape::from_dims({out_channels}), conv.bias);
    }
    return conv;
}

ConvTranspose1dWeights load_hift_weight_norm_conv_transpose1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    bool use_bias,
    engine::assets::TensorStorageType) {
    auto g = components::read_f32_tensor(source, prefix + ".parametrizations.weight.original0", {in_channels, 1, 1});
    auto v = components::read_f32_tensor(source, prefix + ".parametrizations.weight.original1", {in_channels, out_channels, kernel});
    ConvTranspose1dWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.use_bias = use_bias;
    conv.weight = apply_weight_norm(g, v, in_channels, out_channels * kernel);
    conv.weight_tensor = store.make_from_f32(
        engine::core::TensorShape::from_dims({in_channels, out_channels, kernel}),
        engine::assets::TensorStorageType::F32,
        conv.weight);
    if (use_bias) {
        conv.bias = components::read_f32_tensor(source, prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_f32(engine::core::TensorShape::from_dims({out_channels}), conv.bias);
    }
    return conv;
}

SnakeWeights load_hift_snake(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    SnakeWeights snake;
    snake.alpha = components::read_f32_tensor(source, name, {channels});
    std::vector<float> inv_alpha(snake.alpha.size(), 0.0F);
    for (size_t index = 0; index < snake.alpha.size(); ++index) {
        inv_alpha[index] = 1.0F / (snake.alpha[index] + 1.0e-9F);
    }
    snake.alpha_tensor = store.make_f32(engine::core::TensorShape::from_dims({channels, 1}), snake.alpha);
    snake.inv_alpha_tensor = store.make_f32(
        engine::core::TensorShape::from_dims({channels, 1}),
        std::move(inv_alpha));
    return snake;
}

ResBlockWeights load_hift_resblock(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    const std::vector<int64_t> & dilations,
    engine::assets::TensorStorageType weight_storage_type) {
    ResBlockWeights block;
    block.convs1.reserve(dilations.size());
    block.convs2.reserve(dilations.size());
    block.activations1.reserve(dilations.size());
    block.activations2.reserve(dilations.size());
    for (size_t index = 0; index < dilations.size(); ++index) {
        const int64_t dilation = dilations[index];
        block.convs1.push_back(load_hift_weight_norm_conv1d(
            store,
            source,
            prefix + ".convs1." + std::to_string(index),
            channels,
            channels,
            kernel,
            1,
            (kernel * dilation - dilation) / 2,
            dilation,
            true,
            weight_storage_type));
        block.convs2.push_back(load_hift_weight_norm_conv1d(
            store,
            source,
            prefix + ".convs2." + std::to_string(index),
            channels,
            channels,
            kernel,
            1,
            (kernel - 1) / 2,
            1,
            true,
            weight_storage_type));
        block.activations1.push_back(load_hift_snake(
            store,
            source,
            prefix + ".activations1." + std::to_string(index) + ".alpha",
            channels));
        block.activations2.push_back(load_hift_snake(
            store,
            source,
            prefix + ".activations2." + std::to_string(index) + ".alpha",
            channels));
    }
    return block;
}

FrameworkHiFTConfig make_chatterbox_hift_config() {
    FrameworkHiFTConfig config;
    config.in_channels = 80;
    config.base_channels = 512;
    config.nb_harmonics = 8;
    config.sampling_rate = 24000;
    config.nsf_alpha = 0.1F;
    config.nsf_sigma = 0.003F;
    config.nsf_voiced_threshold = 10.0F;
    config.upsample_rates = {8, 5, 3};
    config.upsample_kernel_sizes = {16, 11, 7};
    config.istft_n_fft = 16;
    config.istft_hop = 4;
    config.resblock_kernel_sizes = {3, 7, 11};
    config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.source_resblock_kernel_sizes = {7, 7, 11};
    config.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.lrelu_slope = 0.1F;
    config.audio_limit = 0.99F;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
    return config;
}

std::shared_ptr<const FrameworkHiFTWeights> load_chatterbox_hift_weights(
    const std::filesystem::path & checkpoint_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    const auto source = engine::assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<FrameworkHiFTWeights>();
    weights->config = make_chatterbox_hift_config();
    weights->source_path = source->source_path();
    weights->execution_context = std::shared_ptr<engine::core::ExecutionContext>(
        const_cast<engine::core::ExecutionContext *>(&execution_context),
        [](engine::core::ExecutionContext *) {});
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox.hift.weights",
        768ull * 1024ull * 1024ull);

    for (const auto & tensor : source->tensors()) {
        if (tensor.name.rfind("mel2wav.", 0) != 0) {
            continue;
        }
        if (!is_float_dtype(tensor.dtype)) {
            throw std::runtime_error("Chatterbox HiFT safetensors contains non-floating tensor: " + tensor.name);
        }
        weights->parameter_count += tensor_elements(tensor.shape);
        ++weights->loaded_tensor_count;
    }

    for (int index = 0; index < 5; ++index) {
        weights->f0_predictor.condnet.push_back(load_hift_weight_norm_conv1d(
            *weights->store,
            *source,
            "mel2wav.f0_predictor.condnet." + std::to_string(index * 2),
            512,
            index == 0 ? 80 : 512,
            3,
            1,
            1,
            1,
            true,
            weight_storage_type));
    }
    weights->f0_predictor.classifier = load_hift_linear(
        *weights->store,
        *source,
        "mel2wav.f0_predictor.classifier",
        1,
        512,
        true,
        engine::assets::TensorStorageType::F32);

    weights->conv_pre = load_hift_weight_norm_conv1d(
        *weights->store,
        *source,
        "mel2wav.conv_pre",
        512,
        80,
        7,
        1,
        3,
        1,
        true,
        weight_storage_type);
    weights->ups.push_back(load_hift_weight_norm_conv_transpose1d(
        *weights->store, *source, "mel2wav.ups.0", 512, 256, 16, 8, 4, true, weight_storage_type));
    weights->ups.push_back(load_hift_weight_norm_conv_transpose1d(
        *weights->store, *source, "mel2wav.ups.1", 256, 128, 11, 5, 3, true, weight_storage_type));
    weights->ups.push_back(load_hift_weight_norm_conv_transpose1d(
        *weights->store, *source, "mel2wav.ups.2", 128, 64, 7, 3, 2, true, weight_storage_type));

    weights->source_downs.push_back(load_hift_conv1d(
        *weights->store, *source, "mel2wav.source_downs.0", 256, 18, 30, 15, 7, 1, true, weight_storage_type));
    weights->source_downs.push_back(load_hift_conv1d(
        *weights->store, *source, "mel2wav.source_downs.1", 128, 18, 6, 3, 1, 1, true, weight_storage_type));
    weights->source_downs.push_back(load_hift_conv1d(
        *weights->store, *source, "mel2wav.source_downs.2", 64, 18, 1, 1, 0, 1, true, weight_storage_type));

    weights->source_resblocks.push_back(load_hift_resblock(
        *weights->store, *source, "mel2wav.source_resblocks.0", 256, 7, {1, 3, 5}, weight_storage_type));
    weights->source_resblocks.push_back(load_hift_resblock(
        *weights->store, *source, "mel2wav.source_resblocks.1", 128, 7, {1, 3, 5}, weight_storage_type));
    weights->source_resblocks.push_back(load_hift_resblock(
        *weights->store, *source, "mel2wav.source_resblocks.2", 64, 11, {1, 3, 5}, weight_storage_type));

    for (int up_index = 0; up_index < 3; ++up_index) {
        const int64_t channels = 256 >> up_index;
        const std::vector<int64_t> kernels = {3, 7, 11};
        for (int kernel_index = 0; kernel_index < 3; ++kernel_index) {
            const int block_index = up_index * 3 + kernel_index;
            weights->resblocks.push_back(load_hift_resblock(
                *weights->store,
                *source,
                "mel2wav.resblocks." + std::to_string(block_index),
                channels,
                kernels[static_cast<size_t>(kernel_index)],
                {1, 3, 5},
                weight_storage_type));
        }
    }

    weights->conv_post = load_hift_weight_norm_conv1d(
        *weights->store,
        *source,
        "mel2wav.conv_post",
        18,
        64,
        7,
        1,
        3,
        1,
        true,
        weight_storage_type);
    weights->source_linear = load_hift_linear(
        *weights->store,
        *source,
        "mel2wav.m_source.l_linear",
        1,
        9,
        true,
        weight_storage_type);
    weights->store->upload();
    source->release_storage();
    return weights;
}

}  // namespace

struct HiFTVocoderComponent::State {
    FrameworkHiFTComponent component;
};

HiFTVocoderComponent HiFTVocoderComponent::load_from_checkpoint(
    const std::filesystem::path & checkpoint_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto runtime_weights = load_chatterbox_hift_weights(
        checkpoint_path,
        execution_context,
        weight_storage_type);
    auto weights = std::make_shared<HiFTVocoderComponentWeights>();
    weights->runtime_weights = std::move(runtime_weights);
    HiFTVocoderComponent component(std::move(weights), execution_context);
    component.state_ = std::make_shared<State>(State{
        FrameworkHiFTComponent(component.weights_->runtime_weights, execution_context.config()),
    });
    return component;
}

HiFTVocoderComponent::HiFTVocoderComponent(
    std::shared_ptr<const HiFTVocoderComponentWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)), execution_context_(&execution_context) {}

const engine::core::BackendConfig & HiFTVocoderComponent::backend() const noexcept {
    return execution_context_->config();
}

const std::shared_ptr<const HiFTVocoderComponentWeights> & HiFTVocoderComponent::weights() const noexcept {
    return weights_;
}

HiFTVocoderOutputs HiFTVocoderComponent::infer(
    const std::vector<float> & speech_feat,
    int64_t batch,
    int64_t frames,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> & cache_source) const {
    if (batch != 1) {
        throw std::runtime_error("Chatterbox framework HiFT wrapper expects batch 1");
    }
    if (!cache_source.empty()) {
        throw std::runtime_error("Chatterbox framework HiFT wrapper does not support cache_source");
    }
    const auto result = state_->component.synthesize(speech_feat, frames, seed, prior_noise_values);
    HiFTVocoderOutputs outputs;
    outputs.waveform = result.waveform;
    outputs.samples = result.samples;
    return outputs;
}

void HiFTVocoderComponent::release_runtime_cache() const {
    if (state_ == nullptr) {
        throw std::runtime_error("Chatterbox framework HiFT wrapper is not initialized");
    }
    state_->component.release_runtime_cache();
}

}  // namespace engine::models::chatterbox
