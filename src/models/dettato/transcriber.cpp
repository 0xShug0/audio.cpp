#include "transcriber.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "gru.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include <cmath>
#include <numeric>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

LaneTranscriber::LaneTranscriber(const assets::TensorSource &source, const ExecutionContext &execution)
    : execution_(execution),
      store_(execution.backend(), execution.backend_type(), "dettato.transcriber.weights", 8 * 1024 * 1024) {
    for (const auto &metadata : source.tensors()) {
        const auto &name = metadata.name;
        if (name.rfind("front.", 0) == 0)
            continue;
        weights_.emplace(name, store_.load_tensor(source, name, assets::TensorStorageType::Native, metadata.shape));
        const std::string relative_suffix = ".rel_bias";
        if (name.size() > relative_suffix.size() &&
            name.compare(name.size() - relative_suffix.size(), relative_suffix.size(), relative_suffix) == 0)
            host_relative_bias_.emplace(name.substr(0, name.size() - relative_suffix.size()), source.require_f32(name));
        const std::string suffix = ".running_var";
        if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const auto prefix = name.substr(0, name.size() - suffix.size());
            const auto folded = binding::batch_norm_eval_from_source(store_, source, prefix, metadata.shape[0], 1e-5f);
            weights_.emplace(prefix + ".eval_scale", folded.scale);
            weights_.emplace(prefix + ".eval_bias", folded.bias);
        }
        const std::string taps_suffix = ".pmix.taps";
        if (name.size() > taps_suffix.size() &&
            name.compare(name.size() - taps_suffix.size(), taps_suffix.size(), taps_suffix) == 0) {
            // Materialize the upstream banded pitch operator once, not per call.
            constexpr int lanes = 88;
            const std::array<int, 14> offsets{-2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31};
            const auto taps = source.require_f32(name);
            const int64_t channels = metadata.shape[0];
            std::vector<float> matrix(static_cast<size_t>(channels * lanes * lanes));
            for (int64_t c = 0; c < channels; ++c)
                for (int i = 0; i < lanes; ++i)
                    for (size_t k = 0; k < offsets.size(); ++k) {
                        const int j = i + offsets[k];
                        if (j >= 0 && j < lanes)
                            matrix[static_cast<size_t>((c * lanes + i) * lanes + j)] =
                                taps[static_cast<size_t>(c * offsets.size() + k)];
                    }
            weights_.emplace(name + ".matrix",
                             store_.make_f32(TensorShape::from_dims({channels, lanes, lanes}), matrix));
        }
    }
    store_.upload();
}

TensorValue LaneTranscriber::conv(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name, int stride,
                                  int dilation, bool depthwise) {
    const auto w = weights_.at(name + ".weight");
    const Conv2dWeights parameters{w, weights_.at(name + ".bias")};
    const int64_t kh = w.shape.dims[2], kw = w.shape.dims[3];
    if (depthwise) {
        DepthwiseConv2dConfig cfg{
            w.shape.dims[0],          kh,       kw, stride, 1, static_cast<int>((kh / 2) * dilation),
            static_cast<int>(kw / 2), dilation, 1,  true};
        return DepthwiseConv2dModule(cfg).build(ctx, x, parameters);
    }
    Conv2dConfig cfg{w.shape.dims[1],          w.shape.dims[0],         kh, kw, stride, name == "stem.down" ? 3 : 1,
                     static_cast<int>(kh / 2), static_cast<int>(kw / 2)};
    return Conv2dModule(cfg).build(ctx, x, parameters);
}

TensorValue LaneTranscriber::norm(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name) {
    if (weights_.count(name + ".eval_scale")) {
        const auto shape = TensorShape::from_dims({1, x.shape.dims[1], 1, 1});
        const auto scale = reshape_tensor(ctx, weights_.at(name + ".eval_scale"), shape);
        const auto bias = reshape_tensor(ctx, weights_.at(name + ".eval_bias"), shape);
        return AddModule{false, true}.build(ctx, MulModule{true, true}.build(ctx, x, scale), bias);
    }
    const auto normalized = LayerNormModule({x.shape.last_dim(), 1e-5f, false, false}).build(ctx, x, {});
    return AddModule{false, true}.build(ctx,
        MulModule{true, true}.build(ctx, normalized, weights_.at(name + ".weight")), weights_.at(name + ".bias"));
}

TensorValue LaneTranscriber::linear(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name) {
    auto w = weights_.at(name + ".weight");
    const auto bias = weights_.at(name + ".bias");
    const bool pointwise = w.shape.rank == 4;
    if (pointwise) w = reshape_tensor(ctx, w, TensorShape::from_dims({w.shape.dims[0], w.shape.dims[1]}));
    return LinearModule({w.shape.dims[1], w.shape.dims[0], true, GGML_PREC_F32}).build(ctx, x, {w, bias});
}

TensorValue LaneTranscriber::lane(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name,
                                  int dilation) {
    const auto canonical = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, x);
    auto y = conv(ctx, canonical, name + ".dw_t", 1, dilation, true);
    // Keep native [pitch, time, channel, batch] storage. GGML broadcasts the
    // per-channel pitch matrix across windows; merging batch/time would first
    // copy the whole activation to exchange its channel and batch dimensions.
    auto *matrix = weights_.at(name + ".pmix.taps.matrix").tensor;
    // Vulkan matmul does not broadcast its fourth dimension.
    if (execution_.backend_type() == BackendType::Vulkan && y.tensor->ne[3] > 1)
        matrix = ggml_repeat_4d(ctx.ggml, matrix, matrix->ne[0], matrix->ne[1], matrix->ne[2], y.tensor->ne[3]);
    auto *mixed = ggml_mul_mat(ctx.ggml, matrix, y.tensor);
    ggml_mul_mat_set_prec(mixed, GGML_PREC_F32);
    const auto pitch = wrap_tensor(mixed, y.shape);
    y = norm(ctx, AddModule{}.build(ctx, y, pitch), name + ".norm");
    y = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, y);
    y = linear(ctx, GeluModule{}.build(ctx, linear(ctx, y, name + ".pw1")), name + ".pw2");
    return AddModule{}.build(ctx, x, MulModule{true}.build(ctx, y, weights_.at(name + ".gamma")));
}

TensorValue LaneTranscriber::attention(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name,
                                       bool time) {
    const auto B = x.shape.dims[0], T = x.shape.dims[1], P = x.shape.dims[2], C = x.shape.dims[3];
    const int64_t N = time ? B * P : B * T, L = time ? T : P, H = 8, D = C / H;
    auto s = time ? TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x) : x;
    s = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, s), TensorShape::from_dims({N, L, C}));
    const auto packed = linear(ctx, norm(ctx, s, name + ".ln"), name + ".qkv");
    std::array<TensorValue, 3> heads;
    for (int i = 0; i < 3; ++i) {
        auto part = SliceModule({2, i * C, C}).build(ctx, packed);
        part = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, part), TensorShape::from_dims({N, L, H, D}));
        heads[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, part);
    }
    std::optional<TensorValue> mask;
    if (time) {
        if (!rotary_positions_.valid()) {
            rotary_positions_ = core::make_tensor(graph_->persistent_build_context(), GGML_TYPE_I32,
                TensorShape::from_dims({N, L}));
            ggml_set_input(rotary_positions_.tensor);
        }
        for (int i = 0; i < 2; ++i) {
            auto rows = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, heads[i]);
            rows = RoPEModule({D, GGML_ROPE_TYPE_NEOX}).build_batched(ctx, rows, rotary_positions_);
            heads[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, rows);
        }
    } else {
        auto bias =
            core::make_tensor(graph_->persistent_build_context(), execution_.backend_type() == BackendType::Cpu ? GGML_TYPE_F32 : GGML_TYPE_F16, TensorShape::from_dims({1, H, L, L}));
        ggml_set_input(bias.tensor);
        const auto values = host_relative_bias_.at(name);
        std::vector<float> table(static_cast<size_t>(H * L * L));
        for (int64_t h = 0; h < H; ++h)
            for (int64_t i = 0; i < L; ++i)
                for (int64_t j = 0; j < L; ++j)
                    table[static_cast<size_t>((h * L + i) * L + j)] =
                        values[static_cast<size_t>(h * 255 + j - i + 127)];
        constants_.push_back({bias, std::move(table)});
        mask = bias;
    }
    TensorValue y;
    if (execution_.backend_type() == BackendType::Cpu) {
        y = ScaledDotProductAttentionModule({D, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
            .build(ctx, heads[0], heads[1], heads[2], mask);
    } else {
        const int64_t alignment = core::uses_ggml_cuda_or_hip_backend(execution_.backend_type()) ? 64 : 8;
        const int64_t width = (D + alignment - 1) / alignment * alignment;
        if (width != D) for (auto & head : heads) {
            head = wrap_tensor(ggml_pad(ctx.ggml, head.tensor, width - D, 0, 0, 0),
                TensorShape::from_dims({N, H, L, width}), GGML_TYPE_F32);
        }
        ScaledDotProductAttentionConfig cfg{width, ScaledDotProductAttentionLowering::Flash, GGML_PREC_F32};
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(D));
        y = ScaledDotProductAttentionModule(cfg).build(ctx, heads[0], heads[1], heads[2], mask);
        if (width != D) y = SliceModule({3, 0, D}).build(ctx, y);
    }
    y = linear(ctx, reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, y), TensorShape::from_dims({N, L, C})),
               name + ".out");
    y = MulModule{true}.build(ctx, y, weights_.at(name + ".gamma"));
    y = reshape_tensor(ctx, y, time ? TensorShape::from_dims({B, P, T, C}) : TensorShape::from_dims({B, T, P, C}));
    if (time) y = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, y);
    return AddModule{}.build(ctx, x, y);
}

TensorValue LaneTranscriber::recurrent(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name) {
    const auto B = x.shape.dims[0], T = x.shape.dims[1], P = x.shape.dims[2], C = x.shape.dims[3];
    auto s = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    s = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, s), TensorShape::from_dims({B * P, T, C}));
    auto normalized = TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, norm(ctx, s, name + ".ln"));
    auto initial = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({B * P, 192}));
    ggml_set_input(initial.tensor);
    constants_.push_back({initial, std::vector<float>(static_cast<size_t>(B * P * 192), 0)});
    std::array<GRUWeights,2> parameters;
    for(int d=0;d<2;++d) {
        const auto suffix=d ? "_l0_reverse" : "_l0";
        parameters[d]={weights_.at(name+".gru.weight_ih"+suffix),weights_.at(name+".gru.weight_hh"+suffix),
            weights_.at(name+".gru.bias_ih"+suffix),weights_.at(name+".gru.bias_hh"+suffix)};
    }
    auto h = BidirectionalGRU({C, 192}).build(
        ctx, normalized, initial, initial, parameters[0], parameters[1]).sequence;
    h = TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, h);
    h = linear(ctx, h, name + ".out");
    h = reshape_tensor(ctx, h, TensorShape::from_dims({B, P, T, C}));
    return AddModule{}.build(ctx, x, TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, h));
}

void LaneTranscriber::prepare(int64_t batch, int64_t frames) {
    if (graph_ && batch == batch_ && frames == frames_)
        return;
    graph_.reset();
    constants_.clear();
    rotary_positions_ = {};
    batch_ = batch;
    frames_ = frames;
    graph_ = std::make_unique<PreparedGraph>(
        execution_, PreparedGraphConfig{"dettato.transcriber", 32 * 1024 * 1024, 65536, 1024 * 1024});
    auto &ctx = graph_->build_context();
    input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 9, frames, 288}));
    ggml_set_input(input_.tensor);
    auto h = norm(ctx, input_, "stem.inorm");
    h = GeluModule{}.build(ctx, norm(ctx, conv(ctx, h, "stem.c1"), "stem.n1"));
    h = GeluModule{}.build(ctx, norm(ctx, conv(ctx, h, "stem.c2"), "stem.n2"));
    stem_ = norm(ctx, conv(ctx, h, "stem.down", 2), "stem.n3");
    h = SliceModule({3, 0, 88}).build(ctx, stem_);
    const auto skip = h;
    h = conv(ctx, h, "down", 2, 1, true);
    h = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, h);
    int index = 0;
    for (int layer = 0; layer < 15; ++layer) {
        h = lane(ctx, h, "layers." + std::to_string(index++), 1 << (layer % 4));
        if ((layer + 1) % 5 == 0) {
            h = attention(ctx, h, "layers." + std::to_string(index++), false);
            h = attention(ctx, h, "layers." + std::to_string(index++), true);
            h = recurrent(ctx, h, "layers." + std::to_string(index++));
        }
    }
    // Repeat each frame twice without changing its lane/channel ordering.
    h = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, h);
    h = NearestUpsample2dModule({h.shape.dims[2] * 2, h.shape.dims[3]}).build(ctx, h);
    h = SliceModule({2, 0, skip.shape.dims[2]}).build(ctx, h);
    h = norm(ctx, AddModule{}.build(ctx, h, skip), "up_norm");
    h = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, h);
    for (int layer = 0; layer < 2; ++layer)
        h = lane(ctx, h, "refine." + std::to_string(layer), 1);
    hidden_ = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, h);
    hidden_ = wrap_tensor(ggml_cont(ctx.ggml, hidden_.tensor), hidden_.shape);
    logits_ = conv(ctx, hidden_, "head");
    for (const auto &output : {stem_, hidden_, logits_}) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    debug::timing_log_context_reservation("dettato.transcriber.graph", ctx.ggml);
    debug::timing_log_scalar("dettato.transcriber.graph_nodes", ggml_graph_n_nodes(graph_->graph()));
    graph_->allocate_persistent();
    graph_->allocate();
    for (const auto &constant : constants_)
        write_tensor_float(constant.first, constant.second);
    if (rotary_positions_.valid()) {
        std::vector<int32_t> positions(static_cast<size_t>(rotary_positions_.shape.num_elements()));
        for (size_t i = 0; i < positions.size(); ++i)
            positions[i] = static_cast<int32_t>(i % rotary_positions_.shape.dims[1]);
        write_tensor_i32(rotary_positions_, positions);
    }
}

LaneOutputs LaneTranscriber::run(const std::vector<float> &hcqt, int64_t batch, int64_t frames) {
    const auto outputs = run_device(hcqt, batch, frames);
    return {read_tensor_f32(outputs[0].tensor), read_tensor_f32(outputs[1].tensor), read_tensor_f32(outputs[2].tensor),
            batch, stem_.shape.dims[2]};
}

std::array<TensorValue, 3> LaneTranscriber::run_device(const std::vector<float> &hcqt, int64_t batch, int64_t frames) {
    prepare(batch, frames);
    const auto start = std::chrono::steady_clock::now();
    write_tensor_f32(input_, hcqt);
    graph_->compute();
    debug::timing_log_scalar("dettato.transcriber.forward_ms", debug::elapsed_ms(start));
    return {logits_, stem_, hidden_};
}
} // namespace engine::models::dettato::detail
