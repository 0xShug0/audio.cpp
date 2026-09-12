#include "engine/models/xtts_v2/speaker_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::xtts_v2 {
namespace {
namespace binding = engine::modules::binding;
namespace modules = engine::modules;

struct BlockWeights {
    modules::Conv2dWeights conv1;
    modules::BatchNorm2dEvalWeights bn1;
    modules::Conv2dWeights conv2;
    modules::BatchNorm2dEvalWeights bn2;
    modules::LinearWeights se1;
    modules::LinearWeights se2;
    std::optional<modules::Conv2dWeights> downsample;
    std::optional<modules::BatchNorm2dEvalWeights> downsample_bn;
    int64_t in_channels = 0;
    int64_t channels = 0;
    int stride = 1;
};

struct ContextDeleter { void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); } };

modules::BatchNorm2dEvalWeights load_bn(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels) {
    const auto gamma = source.require_f32(prefix + ".weight", {channels});
    const auto beta = source.require_f32(prefix + ".bias", {channels});
    const auto mean = source.require_f32(prefix + ".running_mean", {channels});
    const auto variance = source.require_f32(prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels));
    std::vector<float> bias(static_cast<size_t>(channels));
    for (int64_t i = 0; i < channels; ++i) {
        const auto j = static_cast<size_t>(i);
        scale[j] = gamma[j] / std::sqrt(variance[j] + 1.0e-5F);
        bias[j] = beta[j] - mean[j] * scale[j];
    }
    return {store.make_f32(core::TensorShape::from_dims({channels}), scale),
            store.make_f32(core::TensorShape::from_dims({channels}), bias)};
}

modules::BatchNorm1dEvalWeights load_bn1d(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels) {
    auto fused = load_bn(store, source, prefix, channels);
    return {fused.scale, fused.bias};
}

core::TensorValue block(core::ModuleBuildContext & ctx, const core::TensorValue & input, const BlockWeights & w) {
    auto x = modules::Conv2dModule({w.in_channels, w.channels, 3, 3, w.stride, w.stride, 1, 1, 1, 1, false}).build(ctx, input, w.conv1);
    x = modules::ReluModule{}.build(ctx, x);
    x = modules::BatchNorm2dEvalModule({w.channels}).build(ctx, x, w.bn1);
    x = modules::Conv2dModule({w.channels, w.channels, 3, 3, 1, 1, 1, 1, 1, 1, false}).build(ctx, x, w.conv2);
    x = modules::BatchNorm2dEvalModule({w.channels}).build(ctx, x, w.bn2);

    auto gate = modules::ReduceMeanModule({3}).build(ctx, x);
    gate = modules::ReduceMeanModule({2}).build(ctx, gate);
    gate = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, gate), core::TensorShape::from_dims({1, 1, w.channels}));
    gate = modules::LinearModule({w.channels, w.channels / 8, true}).build(ctx, gate, w.se1);
    gate = modules::ReluModule{}.build(ctx, gate);
    gate = modules::LinearModule({w.channels / 8, w.channels, true}).build(ctx, gate, w.se2);
    gate = modules::SigmoidModule{}.build(ctx, gate);
    gate = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, gate), core::TensorShape::from_dims({1, w.channels, 1, 1}));
    x = modules::MulModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, gate));

    auto residual = input;
    if (w.downsample) {
        residual = modules::Conv2dModule({w.in_channels, w.channels, 1, 1, w.stride, w.stride, 0, 0, 1, 1, false}).build(ctx, input, *w.downsample);
        residual = modules::BatchNorm2dEvalModule({w.channels}).build(ctx, residual, *w.downsample_bn);
    }
    return modules::ReluModule{}.build(ctx, modules::AddModule{}.build(ctx, x, residual));
}

}  // namespace

struct XttsV2SpeakerEncoderRuntime::Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights input;
    modules::BatchNorm2dEvalWeights input_bn;
    std::vector<BlockWeights> blocks;
    modules::Conv1dWeights attention1;
    modules::BatchNorm1dEvalWeights attention_bn;
    modules::Conv1dWeights attention2;
    modules::LinearWeights output;
};

class XttsV2SpeakerEncoderRuntime::Graph {
public:
    Graph(core::ExecutionContext & execution, std::shared_ptr<const Weights> weights, int64_t frames, size_t arena)
        : execution_(execution), weights_(std::move(weights)), frames_(frames) {
        ctx_.reset(ggml_init({arena, nullptr, true}));
        input_ctx_.reset(ggml_init({4U * 1024U * 1024U, nullptr, true}));
        if (!ctx_ || !input_ctx_) throw std::runtime_error("failed to initialize XTTS v2 speaker graph");
        core::ModuleBuildContext ctx{ctx_.get(), "xtts_v2.speaker", execution_.backend_type()};
        core::ModuleBuildContext ictx{input_ctx_.get(), "xtts_v2.speaker.input", execution_.backend_type()};
        input_ = core::make_tensor(ictx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 64, frames_})).tensor;
        ggml_set_input(input_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({1, 1, 64, frames_}), GGML_TYPE_F32);
        x = modules::Conv2dModule({1, 32, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights_->input);
        x = modules::ReluModule{}.build(ctx, x);
        x = modules::BatchNorm2dEvalModule({32}).build(ctx, x, weights_->input_bn);
        for (const auto & item : weights_->blocks) x = block(ctx, x, item);
        const int64_t time = x.shape.dims[3];
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({1, 2048, time}));
        auto w = modules::Conv1dModule({2048, 128, 1, 1, 0, 1, true}).build(ctx, x, weights_->attention1);
        w = modules::ReluModule{}.build(ctx, w);
        w = modules::BatchNorm1dEvalModule({128}).build(ctx, w, weights_->attention_bn);
        w = modules::Conv1dModule({128, 2048, 1, 1, 0, 1, true}).build(ctx, w, weights_->attention2);
        w = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, w).tensor), w.shape, GGML_TYPE_F32);
        auto weighted = modules::MulModule{}.build(ctx, x, w);
        auto mu = modules::ReduceSumModule({2}).build(ctx, weighted);
        auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
        auto second = modules::ReduceSumModule({2}).build(ctx, modules::MulModule{}.build(ctx, squared, w));
        auto mu2 = core::wrap_tensor(ggml_sqr(ctx.ggml, mu.tensor), mu.shape, GGML_TYPE_F32);
        auto variance = core::wrap_tensor(ggml_sub(ctx.ggml, second.tensor, mu2.tensor), second.shape, GGML_TYPE_F32);
        variance = core::wrap_tensor(ggml_clamp(ctx.ggml, variance.tensor, 1.0e-5F, INFINITY), variance.shape, GGML_TYPE_F32);
        auto sigma = core::wrap_tensor(ggml_sqrt(ctx.ggml, variance.tensor), variance.shape, GGML_TYPE_F32);
        auto stats = modules::ConcatModule({1}).build(ctx, mu, sigma);
        stats = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, stats), core::TensorShape::from_dims({1, 1, 4096}));
        auto output = modules::LinearModule({4096, 512, true}).build(ctx, stats, weights_->output);
        auto norm2 = modules::ReduceSumModule({2}).build(ctx, core::wrap_tensor(ggml_sqr(ctx.ggml, output.tensor), output.shape, GGML_TYPE_F32));
        auto norm = core::wrap_tensor(ggml_sqrt(ctx.ggml, norm2.tensor), norm2.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_div(ctx.ggml, output.tensor, modules::RepeatModule({output.shape}).build(ctx, norm).tensor), output.shape, GGML_TYPE_F32);
        output_ = core::ensure_backend_addressable_layout(ctx, output).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!input_buffer_ || !allocator_ || !ggml_gallocr_reserve(allocator_, graph_) || !ggml_gallocr_alloc_graph(allocator_, graph_))
            throw std::runtime_error("failed to allocate XTTS v2 speaker graph");
    }
    ~Graph() {
        if (graph_) core::release_backend_graph_resources(execution_.backend(), graph_);
        if (allocator_) ggml_gallocr_free(allocator_);
        if (input_buffer_) ggml_backend_buffer_free(input_buffer_);
    }
    bool matches(int64_t frames) const noexcept { return frames == frames_; }
    XttsV2SpeakerEmbedding run(const std::vector<float> & mel) {
        if (static_cast<int64_t>(mel.size()) != 64 * frames_) throw std::runtime_error("XTTS v2 speaker mel shape mismatch");
        ggml_backend_tensor_set(input_, mel.data(), 0, mel.size() * sizeof(float));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("XTTS v2 speaker compute failed");
        ggml_backend_synchronize(execution_.backend());
        XttsV2SpeakerEmbedding result; result.values.resize(512);
        ggml_backend_tensor_get(output_, result.values.data(), 0, result.values.size() * sizeof(float));
        return result;
    }
private:
    core::ExecutionContext & execution_;
    std::shared_ptr<const Weights> weights_;
    int64_t frames_;
    std::unique_ptr<ggml_context, ContextDeleter> ctx_, input_ctx_;
    ggml_tensor * input_ = nullptr; ggml_tensor * output_ = nullptr; ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr; ggml_backend_buffer_t input_buffer_ = nullptr;
};

XttsV2SpeakerEncoderRuntime::XttsV2SpeakerEncoderRuntime(
    const XttsV2Assets & assets, core::ExecutionContext & execution,
    size_t weight_context_bytes, size_t graph_context_bytes,
    assets::TensorStorageType matmul_type, assets::TensorStorageType conv_type)
    : execution_(execution), graph_context_bytes_(graph_context_bytes) {
    auto out = std::make_shared<Weights>();
    out->store = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(), "xtts_v2.speaker.weights", weight_context_bytes);
    const auto & source = *assets.speaker_encoder;
    out->input = binding::conv2d_from_source(*out->store, source, "conv1", conv_type, 32, 1, 3, 3, true);
    out->input_bn = load_bn(*out->store, source, "bn1", 32);
    constexpr std::array<int, 4> counts{3, 4, 6, 3};
    constexpr std::array<int64_t, 4> channels{32, 64, 128, 256};
    int64_t in_channels = 32;
    for (size_t stage = 0; stage < counts.size(); ++stage) for (int index = 0; index < counts[stage]; ++index) {
        const int stride = stage > 0 && index == 0 ? 2 : 1;
        const std::string p = "layer" + std::to_string(stage + 1) + "." + std::to_string(index);
        BlockWeights item; item.in_channels = in_channels; item.channels = channels[stage]; item.stride = stride;
        item.conv1 = binding::conv2d_from_source(*out->store, source, p + ".conv1", conv_type, item.channels, in_channels, 3, 3, false);
        item.bn1 = load_bn(*out->store, source, p + ".bn1", item.channels);
        item.conv2 = binding::conv2d_from_source(*out->store, source, p + ".conv2", conv_type, item.channels, item.channels, 3, 3, false);
        item.bn2 = load_bn(*out->store, source, p + ".bn2", item.channels);
        item.se1 = binding::linear_from_source(*out->store, source, p + ".se.fc.0", matmul_type, item.channels / 8, item.channels, true);
        item.se2 = binding::linear_from_source(*out->store, source, p + ".se.fc.2", matmul_type, item.channels, item.channels / 8, true);
        if (stride != 1 || in_channels != item.channels) {
            item.downsample = binding::conv2d_from_source(*out->store, source, p + ".downsample.0", conv_type, item.channels, in_channels, 1, 1, false);
            item.downsample_bn = load_bn(*out->store, source, p + ".downsample.1", item.channels);
        }
        out->blocks.push_back(std::move(item)); in_channels = channels[stage];
    }
    out->attention1 = binding::conv1d_from_source(*out->store, source, "attention.0", conv_type, 128, 2048, 1, true);
    out->attention_bn = load_bn1d(*out->store, source, "attention.2", 128);
    out->attention2 = binding::conv1d_from_source(*out->store, source, "attention.3", conv_type, 2048, 128, 1, true);
    out->output = binding::linear_from_source(*out->store, source, "fc", matmul_type, 512, 4096, true);
    out->store->upload(); weights_ = std::move(out);
}

XttsV2SpeakerEncoderRuntime::~XttsV2SpeakerEncoderRuntime() = default;
XttsV2SpeakerEmbedding XttsV2SpeakerEncoderRuntime::encode(const XttsV2MelFeatures & mel) {
    if (mel.channels != 64 || mel.frames <= 0) throw std::runtime_error("XTTS v2 speaker encoder expects [64, frames]");
    if (!graph_ || !graph_->matches(mel.frames)) graph_ = std::make_unique<Graph>(execution_, weights_, mel.frames, graph_context_bytes_);
    return graph_->run(mel.values);
}

}  // namespace engine::models::xtts_v2
