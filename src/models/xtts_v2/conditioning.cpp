#include "engine/models/xtts_v2/conditioning.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::xtts_v2 {
namespace {

namespace binding = engine::modules::binding;
namespace modules = engine::modules;

constexpr int64_t kDim = 1024;
constexpr int64_t kHeads = 16;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kPerceiverHeads = 8;
constexpr int64_t kPerceiverInner = 512;
constexpr int64_t kPerceiverFf = 2730;
constexpr int64_t kLatents = 32;

struct AttentionWeights {
    modules::NormWeights norm;
    modules::Conv1dWeights qkv;
    modules::Conv1dWeights out;
};

struct PerceiverLayerWeights {
    modules::LinearWeights q;
    modules::LinearWeights kv;
    modules::LinearWeights out;
    modules::LinearWeights ff_in;
    modules::LinearWeights ff_out;
};

struct ContextDeleter {
    void operator()(ggml_context * ctx) const noexcept { if (ctx != nullptr) ggml_free(ctx); }
};

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t heads, int64_t dim) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const AttentionWeights & weights) {
    auto x = modules::GroupNormModule({kDim, 32, 1.0e-5F, true, true}).build(ctx, input_bct, weights.norm);
    auto qkv = modules::Conv1dModule({kDim, 3 * kDim, 1, 1, 0, 1, true}).build(ctx, x, weights.qkv);
    auto q = modules::SliceModule({1, 0, kDim}).build(ctx, qkv);
    auto k = modules::SliceModule({1, kDim, kDim}).build(ctx, qkv);
    auto v = modules::SliceModule({1, 2 * kDim, kDim}).build(ctx, qkv);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, q), kHeads, kHeadDim));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, k), kHeads, kHeadDim));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, v), kHeads, kHeadDim));
    auto scores = modules::MatMulModule{}.build(ctx, q, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(kHeadDim))), scores.shape, GGML_TYPE_F32);
    auto probs = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto mixed = modules::MatMulModule{}.build(ctx, probs, v);
    mixed = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, mixed);
    mixed = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mixed), core::TensorShape::from_dims({1, input_bct.shape.dims[2], kDim}));
    mixed = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, mixed);
    mixed = modules::Conv1dModule({kDim, kDim, 1, 1, 0, 1, true}).build(ctx, mixed, weights.out);
    return modules::AddModule{}.build(ctx, x, mixed);
}

core::TensorValue perceiver_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latents,
    const core::TensorValue & context,
    const PerceiverLayerWeights & weights) {
    const auto full = modules::ConcatModule({1}).build(ctx, latents, context);
    auto q = modules::LinearModule({kDim, kPerceiverInner, false}).build(ctx, latents, weights.q);
    auto kv = modules::LinearModule({kDim, 2 * kPerceiverInner, false}).build(ctx, full, weights.kv);
    auto k = modules::SliceModule({2, 0, kPerceiverInner}).build(ctx, kv);
    auto v = modules::SliceModule({2, kPerceiverInner, kPerceiverInner}).build(ctx, kv);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, kPerceiverHeads, kHeadDim));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, k, kPerceiverHeads, kHeadDim));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, v, kPerceiverHeads, kHeadDim));
    auto scores = modules::MatMulModule{}.build(ctx, q, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0F / 8.0F), scores.shape, GGML_TYPE_F32);
    auto probs = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto mixed = modules::MatMulModule{}.build(ctx, probs, v);
    mixed = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, mixed);
    mixed = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mixed), core::TensorShape::from_dims({1, kLatents, kPerceiverInner}));
    return modules::LinearModule({kPerceiverInner, kDim, false}).build(ctx, mixed, weights.out);
}

core::TensorValue geglu(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto value = modules::SliceModule({2, 0, kPerceiverFf}).build(ctx, input);
    auto gate = modules::SliceModule({2, kPerceiverFf, kPerceiverFf}).build(ctx, input);
    gate = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, gate);
    return modules::MulModule{}.build(ctx, value, gate);
}

core::TensorValue rms_norm(core::ModuleBuildContext & ctx, const core::TensorValue & input, const core::TensorValue & gamma) {
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({2}).build(ctx, squared);
    sum = core::wrap_tensor(ggml_sqrt(ctx.ggml, sum.tensor), sum.shape, GGML_TYPE_F32);
    auto divisor = modules::RepeatModule({input.shape}).build(ctx, sum);
    auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, input.tensor, divisor.tensor), input.shape, GGML_TYPE_F32);
    normalized = core::wrap_tensor(ggml_scale(ctx.ggml, normalized.tensor, std::sqrt(static_cast<float>(kDim))), normalized.shape, GGML_TYPE_F32);
    auto gamma_view = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({1, 1, kDim}));
    return modules::MulModule{}.build(ctx, normalized, modules::RepeatModule({input.shape}).build(ctx, gamma_view));
}

}  // namespace

struct XttsV2ConditioningRuntime::Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights input;
    std::vector<AttentionWeights> attention;
    core::TensorValue latents;
    std::vector<PerceiverLayerWeights> perceiver;
    core::TensorValue final_gamma;
    std::vector<float> mel_stats;
};

class XttsV2ConditioningRuntime::Graph {
public:
    Graph(core::ExecutionContext & execution, std::shared_ptr<const Weights> weights, int64_t frames, size_t arena)
        : execution_(execution), weights_(std::move(weights)), frames_(frames) {
        ggml_init_params params{arena, nullptr, true};
        ctx_.reset(ggml_init(params));
        ggml_init_params input_params{4U * 1024U * 1024U, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (!ctx_ || !input_ctx_) throw std::runtime_error("failed to initialize XTTS v2 conditioning graph contexts");
        core::ModuleBuildContext ctx{ctx_.get(), "xtts_v2.conditioning", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "xtts_v2.conditioning.input", execution_.backend_type()};
        input_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 80, frames_})).tensor;
        ggml_set_input(input_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({1, 80, frames_}), GGML_TYPE_F32);
        x = modules::Conv1dModule({80, kDim, 1, 1, 0, 1, true}).build(ctx, x, weights_->input);
        for (const auto & layer : weights_->attention) x = attention(ctx, x, layer);
        auto context = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        auto latents = core::reshape_tensor(ctx, weights_->latents, core::TensorShape::from_dims({1, kLatents, kDim}));
        latents = modules::RepeatModule({core::TensorShape::from_dims({1, kLatents, kDim})}).build(ctx, latents);
        for (const auto & layer : weights_->perceiver) {
            latents = modules::AddModule{}.build(ctx, latents, perceiver_attention(ctx, latents, context, layer));
            auto ff = modules::LinearModule({kDim, 2 * kPerceiverFf, true}).build(ctx, latents, layer.ff_in);
            ff = geglu(ctx, ff);
            ff = modules::LinearModule({kPerceiverFf, kDim, true}).build(ctx, ff, layer.ff_out);
            latents = modules::AddModule{}.build(ctx, latents, ff);
        }
        auto output = rms_norm(ctx, latents, weights_->final_gamma);
        output_ = core::ensure_backend_addressable_layout(ctx, output).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!input_buffer_ || !allocator_ || !ggml_gallocr_reserve(allocator_, graph_) || !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("failed to allocate XTTS v2 conditioning graph");
        }
    }

    ~Graph() {
        if (graph_) core::release_backend_graph_resources(execution_.backend(), graph_);
        if (allocator_) ggml_gallocr_free(allocator_);
        if (input_buffer_) ggml_backend_buffer_free(input_buffer_);
    }

    bool matches(int64_t frames) const noexcept { return frames_ == frames; }

    XttsV2ConditioningLatent run(const std::vector<float> & mel) {
        if (static_cast<int64_t>(mel.size()) != 80 * frames_) throw std::runtime_error("XTTS v2 conditioning mel shape mismatch");
        ggml_backend_tensor_set(input_, mel.data(), 0, mel.size() * sizeof(float));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("XTTS v2 conditioning compute failed");
        ggml_backend_synchronize(execution_.backend());
        XttsV2ConditioningLatent out;
        out.values.resize(static_cast<size_t>(kLatents * kDim));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    core::ExecutionContext & execution_;
    std::shared_ptr<const Weights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, ContextDeleter> ctx_;
    std::unique_ptr<ggml_context, ContextDeleter> input_ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

XttsV2ConditioningRuntime::XttsV2ConditioningRuntime(
    const XttsV2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type)
    : execution_(execution), graph_context_bytes_(graph_context_bytes) {
    auto weights = std::make_shared<Weights>();
    weights->store = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(), "xtts_v2.conditioning.weights", weight_context_bytes);
    const auto & source = *assets.gpt;
    weights->input = binding::conv1d_from_source(*weights->store, source, "conditioning_encoder.init", conv_storage_type, kDim, 80, 1, true);
    weights->attention.reserve(6);
    for (int64_t i = 0; i < 6; ++i) {
        const std::string prefix = "conditioning_encoder.attn." + std::to_string(i);
        AttentionWeights layer;
        layer.norm = binding::norm_from_source(*weights->store, source, prefix + ".norm", kDim);
        layer.qkv = binding::conv1d_from_source(*weights->store, source, prefix + ".qkv", conv_storage_type, 3 * kDim, kDim, 1, true);
        layer.out = binding::conv1d_from_source(*weights->store, source, prefix + ".proj_out", conv_storage_type, kDim, kDim, 1, true);
        weights->attention.push_back(std::move(layer));
    }
    weights->latents = weights->store->load_f32_tensor(source, "conditioning_perceiver.latents", {kLatents, kDim});
    weights->perceiver.reserve(2);
    for (int64_t i = 0; i < 2; ++i) {
        const std::string prefix = "conditioning_perceiver.layers." + std::to_string(i);
        PerceiverLayerWeights layer;
        layer.q = binding::linear_from_source(*weights->store, source, prefix + ".0.to_q", matmul_storage_type, kPerceiverInner, kDim, false);
        layer.kv = binding::linear_from_source(*weights->store, source, prefix + ".0.to_kv", matmul_storage_type, 2 * kPerceiverInner, kDim, false);
        layer.out = binding::linear_from_source(*weights->store, source, prefix + ".0.to_out", matmul_storage_type, kDim, kPerceiverInner, false);
        layer.ff_in = binding::linear_from_source(*weights->store, source, prefix + ".1.0", matmul_storage_type, 2 * kPerceiverFf, kDim, true);
        layer.ff_out = binding::linear_from_source(*weights->store, source, prefix + ".1.2", matmul_storage_type, kDim, kPerceiverFf, true);
        weights->perceiver.push_back(std::move(layer));
    }
    weights->final_gamma = weights->store->load_f32_tensor(source, "conditioning_perceiver.norm.gamma", {kDim});
    weights->mel_stats = source.require_f32("mel_stats", {80});
    weights->store->upload();
    weights_ = std::move(weights);
}

XttsV2ConditioningRuntime::~XttsV2ConditioningRuntime() = default;

XttsV2ConditioningLatent XttsV2ConditioningRuntime::encode(const XttsV2MelFeatures & mel) {
    if (mel.channels != 80 || mel.frames <= 0) throw std::runtime_error("XTTS v2 conditioning expects [80, frames] mel input");
    if (!graph_ || !graph_->matches(mel.frames)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(execution_, weights_, mel.frames, graph_context_bytes_);
    }
    return graph_->run(mel.values);
}

const std::vector<float> & XttsV2ConditioningRuntime::mel_stats() const noexcept { return weights_->mel_stats; }

}  // namespace engine::models::xtts_v2
