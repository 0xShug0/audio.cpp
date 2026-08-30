#include "engine/framework/audio/flashsr.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::audio {
namespace {

constexpr int kFlashSrOutputSampleRate = 48000;
constexpr int kFlashSrChannels = 32;
constexpr int kFlashSrActivationKernel = 12;
constexpr int kFlashSrActivationRatio = 2;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct BackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

std::string shape_string(const engine::assets::TensorDataF32 & tensor) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tensor.shape.rank; ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << tensor.shape.dims[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<float> require_channel_param(
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    const std::string & name) {
    const auto tensor = source->require_f32_tensor(name);
    if (tensor.shape.rank != 3 || tensor.shape.dims[0] != 1 ||
        tensor.shape.dims[1] != kFlashSrChannels || tensor.shape.dims[2] != 1) {
        throw std::runtime_error("FlashSR tensor shape mismatch for " + name + ": " + shape_string(tensor));
    }
    return tensor.values;
}

std::string conv_weight_name(const std::string & block_id, int group, int index) {
    return "resblocks." + block_id + ".convs" + std::to_string(group) + "." +
           std::to_string(index) + ".weight";
}

std::string conv_bias_name(const std::string & block_id, int group, int index) {
    return "resblocks." + block_id + ".convs" + std::to_string(group) + "." +
           std::to_string(index) + ".bias";
}

struct Conv1dLayer {
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel = 0;
    modules::Conv1dWeights conv;
};

struct ActivationParams {
    core::TensorValue alpha;
    core::TensorValue inv_beta;
};

struct ResBlockWeights {
    Conv1dLayer convs1[3];
    Conv1dLayer convs2[3];
    ActivationParams activations[6];
};

Conv1dLayer load_conv(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & weight_name,
    const std::string & bias_name,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel) {
    Conv1dLayer layer;
    layer.out_channels = out_channels;
    layer.in_channels = in_channels;
    layer.kernel = kernel;
    layer.conv.weight = store.load_f32_tensor(source, weight_name, {out_channels, in_channels, kernel});
    if (!bias_name.empty()) {
        layer.conv.bias = store.load_f32_tensor(source, bias_name, {out_channels});
    }
    return layer;
}

ActivationParams load_activation(
    core::BackendWeightStore & store,
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    const std::string & prefix) {
    return {
        store.make_f32(core::TensorShape::from_dims({kFlashSrChannels}), require_channel_param(source, prefix + ".alpha")),
        store.make_f32(core::TensorShape::from_dims({kFlashSrChannels}), require_channel_param(source, prefix + ".inv_beta")),
    };
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    const std::string & block_id,
    int64_t kernel) {
    ResBlockWeights block;
    for (int i = 0; i < 3; ++i) {
        block.convs1[i] = load_conv(
            store,
            *source,
            conv_weight_name(block_id, 1, i),
            conv_bias_name(block_id, 1, i),
            kFlashSrChannels,
            kFlashSrChannels,
            kernel);
        block.convs2[i] = load_conv(
            store,
            *source,
            conv_weight_name(block_id, 2, i),
            conv_bias_name(block_id, 2, i),
            kFlashSrChannels,
            kFlashSrChannels,
            kernel);
    }
    for (int i = 0; i < 6; ++i) {
        block.activations[i] = load_activation(store, source, "resblocks." + block_id + ".activations." + std::to_string(i));
    }
    return block;
}

std::vector<float> make_diagonal_filter_weights(const std::vector<float> & filter, bool upsample) {
    if (filter.size() != kFlashSrActivationKernel) {
        throw std::runtime_error("FlashSR lowpass filter shape mismatch");
    }
    std::vector<float> weights(static_cast<size_t>(kFlashSrChannels * kFlashSrChannels * kFlashSrActivationKernel), 0.0f);
    for (int64_t c = 0; c < kFlashSrChannels; ++c) {
        for (int64_t k = 0; k < kFlashSrActivationKernel; ++k) {
            const size_t index = static_cast<size_t>((c * kFlashSrChannels + c) * kFlashSrActivationKernel + k);
            const int64_t source_k = upsample ? (kFlashSrActivationKernel - 1 - k) : k;
            weights[index] = filter[static_cast<size_t>(source_k)] * (upsample ? static_cast<float>(kFlashSrActivationRatio) : 1.0f);
        }
    }
    return weights;
}

}  // namespace

struct FlashSrWeights {
    std::unique_ptr<ggml_backend, BackendDeleter> backend;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::shared_ptr<core::BackendWeightStore> store;
    Conv1dLayer conv_pre;
    Conv1dLayer conv_post;
    ResBlockWeights resblock2;
    ResBlockWeights resblock0;
    ActivationParams activation_post;
    modules::Conv1dWeights downsample_filter;
    modules::ConvTranspose1dWeights upsample_filter;
};

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv1dLayer & layer,
    int64_t padding,
    int64_t dilation) {
    return modules::Conv1dModule({
        layer.in_channels,
        layer.out_channels,
        layer.kernel,
        1,
        static_cast<int>(padding),
        static_cast<int>(dilation),
        true,
    }).build(ctx, input, layer.conv);
}

core::TensorValue replicate_pad1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left,
    int64_t right) {
    if (left < 0 || right < 0) {
        throw std::runtime_error("FlashSR replicate padding must be non-negative");
    }
    auto output = input;
    constexpr int axis = 2;
    if (left > 0) {
        auto edge = modules::SliceModule({axis, 0, 1}).build(ctx, input);
        auto pad_shape = edge.shape;
        pad_shape.dims[axis] = left;
        auto pad = modules::RepeatModule({pad_shape}).build(ctx, edge);
        output = modules::ConcatModule({axis}).build(ctx, pad, output);
    }
    if (right > 0) {
        auto edge = modules::SliceModule({axis, input.shape.dims[axis] - 1, 1}).build(ctx, input);
        auto pad_shape = edge.shape;
        pad_shape.dims[axis] = right;
        auto pad = modules::RepeatModule({pad_shape}).build(ctx, edge);
        output = modules::ConcatModule({axis}).build(ctx, output, pad);
    }
    return output;
}

core::TensorValue activation1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ActivationParams & params,
    const FlashSrWeights & weights) {
    constexpr int64_t pad = kFlashSrActivationKernel / kFlashSrActivationRatio - 1;
    constexpr int64_t crop_left = pad * kFlashSrActivationRatio +
                                  (kFlashSrActivationKernel - kFlashSrActivationRatio) / 2;
    constexpr int64_t crop_right = pad * kFlashSrActivationRatio +
                                   (kFlashSrActivationKernel - kFlashSrActivationRatio + 1) / 2;
    auto padded = replicate_pad1d(ctx, input, pad, pad);
    auto up = modules::ConvTranspose1dModule({
        kFlashSrChannels,
        kFlashSrChannels,
        kFlashSrActivationKernel,
        kFlashSrActivationRatio,
        0,
        1,
        false,
    }).build(ctx, padded, weights.upsample_filter);
    up = modules::SliceModule({2, crop_left, up.shape.dims[2] - crop_left - crop_right}).build(ctx, up);

    auto alpha = core::reshape_tensor(ctx, params.alpha, core::TensorShape::from_dims({1, kFlashSrChannels, 1}));
    alpha = core::wrap_tensor(ggml_repeat(ctx.ggml, alpha.tensor, up.tensor), up.shape, GGML_TYPE_F32);
    auto inv_beta = core::reshape_tensor(ctx, params.inv_beta, core::TensorShape::from_dims({1, kFlashSrChannels, 1}));
    inv_beta = core::wrap_tensor(ggml_repeat(ctx.ggml, inv_beta.tensor, up.tensor), up.shape, GGML_TYPE_F32);
    auto periodic = core::wrap_tensor(
        ggml_sqr(ctx.ggml, ggml_sin(ctx.ggml, ggml_mul(ctx.ggml, up.tensor, alpha.tensor))),
        up.shape,
        GGML_TYPE_F32);
    auto snake = core::wrap_tensor(
        ggml_add(ctx.ggml, up.tensor, ggml_mul(ctx.ggml, periodic.tensor, inv_beta.tensor)),
        up.shape,
        GGML_TYPE_F32);

    auto down_padded = replicate_pad1d(
        ctx,
        snake,
        kFlashSrActivationKernel / 2 - 1,
        kFlashSrActivationKernel / 2);
    return modules::Conv1dModule({
        kFlashSrChannels,
        kFlashSrChannels,
        kFlashSrActivationKernel,
        kFlashSrActivationRatio,
        0,
        1,
        false,
    }).build(ctx, down_padded, weights.downsample_filter);
}

core::TensorValue resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResBlockWeights & block,
    const FlashSrWeights & weights,
    int64_t kernel) {
    auto output = input;
    constexpr int kDilations[3] = {1, 3, 5};
    for (int i = 0; i < 3; ++i) {
        auto xt = activation1d(ctx, output, block.activations[i * 2], weights);
        xt = conv1d(ctx, xt, block.convs1[i], (kernel * kDilations[i] - kDilations[i]) / 2, kDilations[i]);
        xt = activation1d(ctx, xt, block.activations[i * 2 + 1], weights);
        xt = conv1d(ctx, xt, block.convs2[i], (kernel - 1) / 2, 1);
        output = modules::AddModule().build(ctx, output, xt);
    }
    return output;
}

namespace {

float absolute_peak(const std::vector<float> & values) noexcept {
    float peak = 0.0f;
    for (const float value : values) {
        peak = std::max(peak, std::fabs(value));
    }
    return peak;
}

// Level policy for the model output. `source` is the waveform the caller handed
// in, so the input's own level can be restored instead of being discarded.
std::vector<float> apply_output_level(
    const std::vector<float> & source,
    const std::vector<float> & model_output,
    const FlashSrOptions & options) {
    const float gain = flashsr_output_gain(absolute_peak(source), absolute_peak(model_output), options);
    std::vector<float> output(model_output.size());
    for (size_t i = 0; i < model_output.size(); ++i) {
        output[i] = model_output[i] * gain;
    }
    return output;
}

}  // namespace

class FlashSrGraph {
public:
    FlashSrGraph(const FlashSrWeights & weights, int64_t input_samples)
        : weights_(weights),
          input_samples_(input_samples) {
        if (input_samples_ <= 0) {
            throw std::runtime_error("FlashSR graph input length must be positive");
        }
        ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize FlashSR GGML context");
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "flashsr", weights.backend_type};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, input_samples_}));
        input_ = input.tensor;
        auto x = conv1d(build_ctx, input, weights.conv_pre, 3, 1);
        x = modules::Interpolate1dModule({input_samples_ * 3, modules::Interpolate1dMode::Linear}).build(build_ctx, x);
        auto xs = resblock(build_ctx, x, weights.resblock2, weights, 11);
        auto xs0 = resblock(build_ctx, x, weights.resblock0, weights, 3);
        xs = core::wrap_tensor(ggml_scale(build_ctx.ggml, modules::AddModule().build(build_ctx, xs, xs0).tensor, 0.5f), xs.shape, GGML_TYPE_F32);
        xs = activation1d(build_ctx, xs, weights.activation_post, weights);
        auto output = conv1d(build_ctx, xs, weights.conv_post, 3, 1);
        output = core::wrap_tensor(ggml_tanh(build_ctx.ggml, output.tensor), output.shape, GGML_TYPE_F32);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.backend.get()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate FlashSR GGML graph");
        }
        if (core::uses_host_graph_plan(weights.backend.get())) {
            plan_ = core::create_backend_graph_plan_if_host(weights.backend.get(), graph_);
            if (plan_ == nullptr) {
                throw std::runtime_error("failed to create FlashSR graph plan");
            }
        }
    }

    ~FlashSrGraph() {
        core::release_backend_graph_resources(weights_.backend.get(), graph_);
        if (plan_ != nullptr) {
            auto * backend = weights_.backend.get();
            core::free_backend_graph_plan(backend, plan_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t input_samples) const noexcept {
        return input_samples_ == input_samples;
    }

    std::vector<float> run(const std::vector<float> & waveform) {
        if (static_cast<int64_t>(waveform.size()) != input_samples_) {
            throw std::runtime_error("FlashSR input size changed without rebuilding graph");
        }
        ggml_backend_tensor_set(input_, waveform.data(), 0, waveform.size() * sizeof(float));
        core::set_backend_threads(weights_.backend.get(), weights_.threads);
        const auto status = core::compute_backend_graph(weights_.backend.get(), graph_, plan_, "FlashSR");
        ggml_backend_synchronize(weights_.backend.get());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FlashSR GGML graph compute failed");
        }
        std::vector<float> output(ggml_nelements(output_));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
    const FlashSrWeights & weights_;
    int64_t input_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

float flashsr_output_gain(float input_peak, float output_peak, const FlashSrOptions & options) noexcept {
    if (!(output_peak > 0.0f)) {
        return 0.0f;
    }
    const float ceiling = options.peak_ceiling > 0.0f ? options.peak_ceiling : kFlashSrPeakCeiling;
    const float target = options.preserve_input_level
        ? std::min(std::max(input_peak, 0.0f), ceiling)
        : ceiling;
    return target / output_peak;
}

FlashSrModel::FlashSrModel() = default;
FlashSrModel::~FlashSrModel() = default;
FlashSrModel::FlashSrModel(FlashSrModel &&) noexcept = default;
FlashSrModel & FlashSrModel::operator=(FlashSrModel &&) noexcept = default;

FlashSrModel::FlashSrModel(std::shared_ptr<FlashSrWeights> weights) : weights_(std::move(weights)) {
    if (!weights_) {
        throw std::runtime_error("FlashSR weights are missing");
    }
}

FlashSrModel FlashSrModel::load_from_directory(const std::filesystem::path & model_dir) {
    return load_from_directory(model_dir, core::BackendConfig{});
}

FlashSrModel FlashSrModel::load_from_directory(
    const std::filesystem::path & model_dir,
    const core::BackendConfig & backend_config) {
    auto source = engine::assets::open_tensor_source(model_dir / "flashsr.safetensors");
    auto weights = std::make_shared<FlashSrWeights>();
    weights->backend.reset(core::init_backend(backend_config));
    weights->backend_type = core::backend_type(weights->backend.get());
    weights->threads = std::max(1, backend_config.threads);
    weights->store = std::make_shared<core::BackendWeightStore>(weights->backend.get(), weights->backend_type, "FlashSR", 32ull * 1024ull * 1024ull);
    auto & store = *weights->store;
    weights->conv_pre = load_conv(store, *source, "conv_pre.weight", "conv_pre.bias", kFlashSrChannels, 1, 7);
    weights->conv_post = load_conv(store, *source, "conv_post.weight", "", 1, kFlashSrChannels, 7);
    weights->resblock2 = load_resblock(store, source, "2", 11);
    weights->resblock0 = load_resblock(store, source, "0", 3);
    weights->activation_post = load_activation(store, source, "activation_post");
    const auto filter_tensor = source->require_f32_tensor("activation_filter");
    if (filter_tensor.shape.rank != 3 || filter_tensor.shape.dims[0] != 1 ||
        filter_tensor.shape.dims[1] != 1 || filter_tensor.shape.dims[2] != kFlashSrActivationKernel) {
        throw std::runtime_error("FlashSR lowpass filter shape mismatch: " + shape_string(filter_tensor));
    }
    weights->downsample_filter.weight = store.make_f32(
        core::TensorShape::from_dims({kFlashSrChannels, kFlashSrChannels, kFlashSrActivationKernel}),
        make_diagonal_filter_weights(filter_tensor.values, false));
    weights->upsample_filter.weight = store.make_f32(
        core::TensorShape::from_dims({kFlashSrChannels, kFlashSrChannels, kFlashSrActivationKernel}),
        make_diagonal_filter_weights(filter_tensor.values, true));
    store.upload();
    return FlashSrModel(std::move(weights));
}

FlashSrOutput FlashSrModel::super_resolve_mono_16k(
    const std::vector<float> & waveform,
    const FlashSrOptions & options) const {
    if (!weights_) {
        throw std::runtime_error("FlashSR model is not loaded");
    }
    if (waveform.empty()) {
        throw std::runtime_error("FlashSR input waveform is empty");
    }
    constexpr int64_t segment_samples = 16000 * 4;
    constexpr int64_t stride_samples = 16000 * 3;
    constexpr int64_t segment_threshold = segment_samples * 2;
    constexpr int64_t output_ratio = 3;
    const int64_t original_samples = static_cast<int64_t>(waveform.size());
    if (original_samples <= segment_threshold) {
        if (!graph_ || !graph_->matches(original_samples)) {
            graph_ = std::make_unique<FlashSrGraph>(*weights_, original_samples);
        }
        return FlashSrOutput{
            kFlashSrOutputSampleRate,
            apply_output_level(waveform, graph_->run(waveform), options)};
    }

    std::vector<float> padded = waveform;
    const int64_t remainder = (original_samples - segment_samples) % stride_samples;
    if (remainder != 0) {
        padded.insert(padded.end(), static_cast<size_t>(stride_samples - remainder), 0.0f);
    }
    const int64_t padded_samples = static_cast<int64_t>(padded.size());
    if (!graph_ || !graph_->matches(segment_samples)) {
        graph_ = std::make_unique<FlashSrGraph>(*weights_, segment_samples);
    }
    std::vector<float> output(static_cast<size_t>(padded_samples * output_ratio), 0.0f);
    std::vector<float> weights(static_cast<size_t>(padded_samples * output_ratio), 0.0f);
    const int64_t segment_output_samples = segment_samples * output_ratio;
    const int64_t stride_output_samples = stride_samples * output_ratio;
    const int64_t overlap_output_samples = segment_output_samples - stride_output_samples;
    for (int64_t current = 0; current + segment_samples <= padded_samples; current += stride_samples) {
        std::vector<float> segment(
            padded.begin() + static_cast<std::ptrdiff_t>(current),
            padded.begin() + static_cast<std::ptrdiff_t>(current + segment_samples));
        const auto segment_output = graph_->run(segment);
        if (static_cast<int64_t>(segment_output.size()) != segment_output_samples) {
            throw std::runtime_error("FlashSR segmented output length mismatch");
        }
        const int64_t output_offset = current * output_ratio;
        for (int64_t i = 0; i < segment_output_samples; ++i) {
            float weight = 1.0f;
            if (current > 0 && i < overlap_output_samples) {
                weight = static_cast<float>(i + 1) / static_cast<float>(overlap_output_samples + 1);
            }
            if (current + segment_samples < padded_samples && i >= stride_output_samples) {
                weight = static_cast<float>(segment_output_samples - i) / static_cast<float>(overlap_output_samples + 1);
            }
            const size_t out_index = static_cast<size_t>(output_offset + i);
            output[out_index] += segment_output[static_cast<size_t>(i)] * weight;
            weights[out_index] += weight;
        }
    }
    output.resize(static_cast<size_t>(original_samples * output_ratio));
    weights.resize(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        if (weights[i] <= 0.0f) {
            throw std::runtime_error("FlashSR segmented synthesis produced an uncovered sample");
        }
        output[i] /= weights[i];
    }
    return FlashSrOutput{kFlashSrOutputSampleRate, apply_output_level(waveform, output, options)};
}

}  // namespace engine::audio
