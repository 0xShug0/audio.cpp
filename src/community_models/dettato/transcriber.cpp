#include "engine/community_models/dettato/transcriber.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/community_models/dettato/gru.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>

namespace engine::community_models::dettato::detail {

using namespace core;
using namespace modules;

LaneTranscriber::LaneTranscriber(TranscriberWeights weights, const ExecutionContext &execution)
    : execution_(execution), weights_(std::move(weights)) {
    if (execution_.backend_type() == BackendType::Cuda) {
        const auto device = ggml_backend_get_device(execution_.backend());
        set_fusions_ = reinterpret_cast<ggml_backend_set_graph_fusions_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_set_graph_fusions"));
    }
}

LaneTranscriber::~LaneTranscriber() {
    if (graph_ && set_fusions_) {
        ggml_backend_synchronize(execution_.backend());
        set_fusions_(execution_.backend(), graph_->graph(), 0);
    }
}

TensorValue LaneTranscriber::conv(ModuleBuildContext &ctx, const TensorValue &x, const Conv2dWeights &parameters,
                                  int stride, int dilation, bool depthwise, int frequency_stride) {
    const auto w = parameters.weight;
    const int64_t kh = w.shape.dims[2], kw = w.shape.dims[3];
    if (depthwise) {
        DepthwiseConv2dConfig cfg{
            w.shape.dims[0],          kh,       kw, stride, 1, static_cast<int>((kh / 2) * dilation),
            static_cast<int>(kw / 2), dilation, 1,  true};
        if (ctx.backend_type == BackendType::Cuda && kw == 1) {
            cfg.use_bias = false;
            const auto output = DepthwiseConv2dModule(cfg).build(ctx, x, parameters);
            ggml_conv_2d_dw_set_lowering(output.tensor, GGML_CONV_2D_DW_LOWERING_CUDA_ROWS);
            const auto bias = reshape_tensor(ctx, *parameters.bias, TensorShape::from_dims({1, cfg.channels, 1, 1}));
            return AddModule{false, true}.build(ctx, output, bias);
        }
        return DepthwiseConv2dModule(cfg).build(ctx, x, parameters);
    }
    Conv2dConfig cfg{w.shape.dims[1],          w.shape.dims[0],         kh, kw, stride, frequency_stride,
                     static_cast<int>(kh / 2), static_cast<int>(kw / 2)};
    if (ctx.backend_type == BackendType::Cuda || ctx.backend_type == BackendType::Vulkan) {
        cfg.use_bias = false;
        TensorValue output;
        if (kh == 3 && kw == 3 && cfg.stride_height == 1 && cfg.stride_width == 1) {
            const auto input = ensure_backend_addressable_layout(ctx, x);
            const auto weight = ensure_backend_addressable_layout(ctx, w);
            auto *columns = ggml_im2col(ctx.ggml, weight.tensor, input.tensor, 1, 1, 1, 1, 1, 1, true, weight.type);
            if (ctx.backend_type == BackendType::Cuda)
                ggml_im2col_2d_set_lowering(columns, GGML_IM2COL_2D_LOWERING_CUDA_N_K3_PAD1_X8);
            auto *projected = ggml_mul_mat(
                ctx.ggml,
                ggml_reshape_2d(ctx.ggml, columns, columns->ne[0], columns->ne[1] * columns->ne[2] * columns->ne[3]),
                ggml_reshape_2d(ctx.ggml, weight.tensor,
                                weight.tensor->ne[0] * weight.tensor->ne[1] * weight.tensor->ne[2],
                                weight.tensor->ne[3]));
            projected = ggml_reshape_4d(ctx.ggml, projected, columns->ne[1], columns->ne[2], columns->ne[3],
                                        weight.tensor->ne[3]);
            projected = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, projected, 0, 1, 3, 2));
            output = wrap_tensor(projected, TensorShape::from_dims(
                                                {x.shape.dims[0], cfg.out_channels, x.shape.dims[2], x.shape.dims[3]}));
        } else {
            output = Conv2dModule(cfg).build(ctx, x, parameters);
        }
        const auto bias = reshape_tensor(ctx, *parameters.bias, TensorShape::from_dims({1, cfg.out_channels, 1, 1}));
        return AddModule{false, true}.build(ctx, output, bias);
    }
    return Conv2dModule(cfg).build(ctx, x, parameters);
}

TensorValue LaneTranscriber::norm(ModuleBuildContext &ctx, const TensorValue &x,
                                  const TranscriberNormWeights &weights) {
    const auto &affine = weights.affine;
    if (weights.channel_affine) {
        const auto shape = TensorShape::from_dims({1, x.shape.dims[1], 1, 1});
        if (ctx.backend_type == BackendType::Cuda) {
            const auto input = ensure_backend_addressable_layout(ctx, x);
            return wrap_tensor(ggml_affine(ctx.ggml, input.tensor, affine.scale.tensor, affine.bias.tensor, 2),
                               x.shape);
        }
        return AddModule{false, true}.build(
            ctx, MulModule{true, true}.build(ctx, x, reshape_tensor(ctx, affine.scale, shape)),
            reshape_tensor(ctx, affine.bias, shape));
    }
    if (ctx.backend_type != BackendType::Cuda) {
        const auto normalized = LayerNormModule({x.shape.last_dim(), 1e-5f, false, false}).build(ctx, x, {});
        return AddModule{false, true}.build(ctx, MulModule{true, true}.build(ctx, normalized, affine.scale), affine.bias);
    }
    // Prepare parameter conversions before NORM so its private MUL/ADD chain
    // can use the existing opt-in backend fusion.
    auto scale = affine.scale;
    auto bias = affine.bias;
    if (ctx.backend_type == BackendType::Cuda) {
        if (scale.type != GGML_TYPE_F32)
            scale = wrap_tensor(ggml_cast(ctx.ggml, scale.tensor, GGML_TYPE_F32), scale.shape);
        if (bias.type != GGML_TYPE_F32)
            bias = wrap_tensor(ggml_cast(ctx.ggml, bias.tensor, GGML_TYPE_F32), bias.shape);
    }
    const auto normalized = LayerNormModule({x.shape.last_dim(), 1e-5f, false, false}).build(ctx, x, {});
    return AddModule{false, true}.build(ctx, MulModule{true, true}.build(ctx, normalized, scale), bias);
}

TensorValue LaneTranscriber::linear(ModuleBuildContext &ctx, const TensorValue &x, const LinearWeights &weights,
                                    bool use_bias) {
    auto w = weights.weight;
    const auto bias = weights.bias;
    return LinearModule({w.shape.dims[1], w.shape.dims[0], use_bias, GGML_PREC_F32}).build(ctx, x, {w, bias});
}

TensorValue LaneTranscriber::lane(ModuleBuildContext &ctx, const TensorValue &x, const LaneBlockWeights &weights,
                                  int dilation) {
    const auto canonical = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, x);
    auto y = conv(ctx, canonical, weights.depthwise, 1, dilation, true);
    // Keep native [pitch, time, channel, batch] storage. GGML broadcasts the
    // per-channel pitch matrix across windows; merging batch/time would first
    // copy the whole activation to exchange its channel and batch dimensions.
    TensorValue pitch;
    if (ctx.backend_type == BackendType::Cuda) {
        pitch = wrap_tensor(
            ggml_conv_2d_dw_sparse(ctx.ggml, weights.pitch.coefficients.tensor, y.tensor, weights.pitch.offsets.tensor),
            y.shape);
    } else {
        auto *matrix = weights.pitch.matrix.tensor;
        // Vulkan matmul does not broadcast its fourth dimension.
        if (execution_.backend_type() == BackendType::Vulkan && y.tensor->ne[3] > 1)
            matrix = ggml_repeat_4d(ctx.ggml, matrix, matrix->ne[0], matrix->ne[1], matrix->ne[2], y.tensor->ne[3]);
        auto *mixed = ggml_mul_mat(ctx.ggml, matrix, y.tensor);
        ggml_mul_mat_set_prec(mixed, GGML_PREC_F32);
        pitch = wrap_tensor(mixed, y.shape);
    }
    y = norm(ctx, AddModule{}.build(ctx, y, pitch), TranscriberNormWeights{weights.norm, true});
    y = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, y);
    const bool fused = ctx.backend_type == BackendType::Cuda;
    y = linear(ctx, GeluModule{}.build(ctx, linear(ctx, y, weights.ffn_in)), weights.ffn_out, !fused);
    if (fused) {
        const auto residual = ensure_backend_addressable_layout(ctx, x);
        y = AddModule{false, true}.build(ctx, y, *weights.ffn_out.bias);
        y = MulModule{true, true}.build(ctx, y, weights.gamma);
        return AddModule{}.build(ctx, y, residual);
    }
    return AddModule{}.build(ctx, x, MulModule{true}.build(ctx, y, weights.gamma));
}

TensorValue LaneTranscriber::attention(ModuleBuildContext &ctx, const TensorValue &x,
                                       const LaneAttentionWeights &weights, bool time) {
    const auto B = x.shape.dims[0], T = x.shape.dims[1], P = x.shape.dims[2], C = x.shape.dims[3];
    const int64_t N = time ? B * P : B * T, L = time ? T : P, H = kAttentionHeads, D = C / H;
    const int64_t projection_alignment = core::uses_ggml_cuda_or_hip_backend(ctx.backend_type) ? 64 : 8;
    const int64_t projection_width = (ctx.backend_type == BackendType::Cuda || ctx.backend_type == BackendType::Vulkan)
                                         ? (D + projection_alignment - 1) / projection_alignment * projection_alignment
                                         : D;
    auto s = time ? TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x) : x;
    s = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, s), TensorShape::from_dims({N, L, C}));
    const auto packed = linear(ctx, norm(ctx, s, weights.norm), weights.qkv);
    std::array<TensorValue, 3> heads;
    for (int i = 0; i < 3; ++i) {
        TensorValue part;
        if (ctx.backend_type == BackendType::Cuda || ctx.backend_type == BackendType::Vulkan) {
            core::validate_last_dim(packed, 3 * H * projection_width, "packed attention width");
            part = wrap_tensor(ggml_view_4d(ctx.ggml, packed.tensor, projection_width, H, L, N,
                                            projection_width * packed.tensor->nb[0], packed.tensor->nb[1],
                                            packed.tensor->nb[2], i * H * projection_width * packed.tensor->nb[0]),
                               TensorShape::from_dims({N, L, H, projection_width}));
        } else {
            part = SliceModule({2, i * C, C}).build(ctx, packed);
            part =
                reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, part), TensorShape::from_dims({N, L, H, D}));
        }
        heads[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, part);
    }
    std::optional<TensorValue> mask;
    if (time) {
        if (!rotary_positions_.valid()) {
            rotary_positions_ =
                core::make_tensor(graph_->persistent_build_context(), GGML_TYPE_I32,
                                  (ctx.backend_type == BackendType::Cuda || ctx.backend_type == BackendType::Vulkan)
                                      ? TensorShape::from_dims({L})
                                      : TensorShape::from_dims({N, L}));
            ggml_set_input(rotary_positions_.tensor);
        }
        for (int i = 0; i < 2; ++i) {
            auto rows = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, heads[i]);
            const RoPEModule rope({D, GGML_ROPE_TYPE_NEOX});
            rows = (ctx.backend_type == BackendType::Cuda || ctx.backend_type == BackendType::Vulkan)
                       ? rope.build(ctx, rows, rotary_positions_)
                       : rope.build_batched(ctx, rows, rotary_positions_);
            heads[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, rows);
        }
    } else {
        auto bias = core::make_tensor(graph_->persistent_build_context(),
                                      execution_.backend_type() == BackendType::Cpu ? GGML_TYPE_F32 : GGML_TYPE_F16,
                                      TensorShape::from_dims({1, H, L, L}));
        ggml_set_input(bias.tensor);
        constants_.push_back({bias, weights.relative_bias});
        mask = bias;
    }
    TensorValue y;
    if (execution_.backend_type() == BackendType::Cpu) {
        y = ScaledDotProductAttentionModule({D, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
                .build(ctx, heads[0], heads[1], heads[2], mask);
    } else {
        const int64_t alignment = core::uses_ggml_cuda_or_hip_backend(execution_.backend_type()) ? 64 : 8;
        const int64_t width = (D + alignment - 1) / alignment * alignment;
        if (width != projection_width)
            for (auto &head : heads) {
                head = wrap_tensor(ggml_pad(ctx.ggml, head.tensor, width - projection_width, 0, 0, 0),
                                   TensorShape::from_dims({N, H, L, width}), GGML_TYPE_F32);
            }
        ScaledDotProductAttentionConfig cfg{
            width,
            (ctx.backend_type == BackendType::Cuda || ctx.backend_type == BackendType::Vulkan)
                ? ScaledDotProductAttentionLowering::FlashPreserveViews
                : ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32};
        cfg.scale = 1.0f / std::sqrt(static_cast<float>(D));
        y = ScaledDotProductAttentionModule(cfg).build(ctx, heads[0], heads[1], heads[2], mask);
        if (width != D)
            y = SliceModule({3, 0, D}).build(ctx, y);
    }
    const bool fused = ctx.backend_type == BackendType::Cuda;
    y = linear(ctx, reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, y), TensorShape::from_dims({N, L, C})),
               weights.output, !fused);
    if (!fused)
        y = MulModule{true}.build(ctx, y, weights.gamma);
    y = reshape_tensor(ctx, y, time ? TensorShape::from_dims({B, P, T, C}) : TensorShape::from_dims({B, T, P, C}));
    if (time)
        y = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, y);
    if (fused) {
        y = ensure_backend_addressable_layout(ctx, y);
        const auto residual = ensure_backend_addressable_layout(ctx, x);
        y = AddModule{false, true}.build(ctx, y, *weights.output.bias);
        y = MulModule{true, true}.build(ctx, y, weights.gamma);
        return AddModule{}.build(ctx, y, residual);
    }
    return AddModule{}.build(ctx, x, y);
}

TensorValue LaneTranscriber::recurrent(ModuleBuildContext &ctx, const TensorValue &x,
                                       const LaneRecurrentWeights &weights) {
    const auto B = x.shape.dims[0], T = x.shape.dims[1], P = x.shape.dims[2], C = x.shape.dims[3];
    auto s = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    s = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, s), TensorShape::from_dims({B * P, T, C}));
    auto normalized = TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, norm(ctx, s, weights.norm));
    auto initial = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({B * P, 192}));
    ggml_set_input(initial.tensor);
    constants_.push_back({initial, std::vector<float>(static_cast<size_t>(B * P * 192), 0)});
    const auto &parameters = weights.gru;
    auto h = BidirectionalGRU({C, 192,
                               ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceCell
                                                                     : GRURecurrenceLowering::Unrolled})
                 .build(ctx, normalized, initial, initial, parameters[0], parameters[1])
                 .sequence;
    h = TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, h);
    h = linear(ctx, h, weights.output);
    h = reshape_tensor(ctx, h, TensorShape::from_dims({B, P, T, C}));
    return AddModule{}.build(ctx, x, TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, h));
}

void LaneTranscriber::prepare(int64_t batch, int64_t frames) {
    if (graph_ && batch == batch_ && frames == frames_)
        return;
    if (graph_ && set_fusions_) {
        ggml_backend_synchronize(execution_.backend());
        set_fusions_(execution_.backend(), graph_->graph(), 0);
    }
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
    auto h = norm(ctx, input_, weights_.input_norm);
    h = GeluModule{}.build(ctx, norm(ctx, conv(ctx, h, weights_.stem1), weights_.stem_norm1));
    h = GeluModule{}.build(ctx, norm(ctx, conv(ctx, h, weights_.stem2), weights_.stem_norm2));
    stem_ = norm(ctx, conv(ctx, h, weights_.stem_down, 2, 1, false, 3), weights_.stem_norm3);
    h = SliceModule({3, 0, 88}).build(ctx, stem_);
    const auto skip = h;
    h = conv(ctx, h, weights_.down, 2, 1, true);
    h = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, h);
    int layer = 0;
    for (const auto &stage : weights_.stages) {
        for (const auto &block : stage.lanes)
            h = lane(ctx, h, block, 1 << (layer++ % 4));
        h = attention(ctx, h, stage.pitch_attention, false);
        h = attention(ctx, h, stage.time_attention, true);
        h = recurrent(ctx, h, stage.recurrent);
    }
    // Repeat each frame twice without changing its lane/channel ordering.
    h = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, h);
    h = NearestUpsample2dModule({h.shape.dims[2] * 2, h.shape.dims[3]}).build(ctx, h);
    h = SliceModule({2, 0, skip.shape.dims[2]}).build(ctx, h);
    h = norm(ctx, AddModule{}.build(ctx, h, skip), weights_.up_norm);
    h = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, h);
    for (int layer = 0; layer < 2; ++layer)
        h = lane(ctx, h, weights_.refine[layer], 1);
    hidden_ = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, h);
    hidden_ = wrap_tensor(ggml_cont(ctx.ggml, hidden_.tensor), hidden_.shape);
    logits_ = conv(ctx, hidden_, weights_.head);
    for (const auto &output : {stem_, hidden_, logits_}) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    debug::timing_log_context_reservation("dettato.transcriber.graph", ctx.ggml);
    debug::timing_log_scalar("dettato.transcriber.graph_nodes", ggml_graph_n_nodes(graph_->graph()));
    if (execution_.backend_type() == BackendType::Cuda) {
        // Keep copy selection local to these explicitly opted-in nodes.
        for (int i = 0; i < ggml_graph_n_nodes(graph_->graph()); ++i) {
            auto *node = ggml_graph_node(graph_->graph(), i);
            if (node->op == GGML_OP_CONT && node->type == GGML_TYPE_F32)
                ggml_cont_set_lowering(node, GGML_CONT_LOWERING_CUDA_TILED);
        }
    }
    if (set_fusions_) {
        const auto enabled = set_fusions_(execution_.backend(), graph_->graph(),
            GGML_BACKEND_GRAPH_FUSION_NORM_AFFINE | GGML_BACKEND_GRAPH_FUSION_GATED_RESIDUAL);
        debug::trace_log_scalar("dettato.transcriber.graph_fusions", enabled);
    }
    graph_->allocate_persistent();
    graph_->allocate();
    for (const auto &constant : constants_)
        write_tensor_float(constant.first, constant.second);
    if (rotary_positions_.valid()) {
        std::vector<int32_t> positions(static_cast<size_t>(rotary_positions_.shape.num_elements()));
        for (size_t i = 0; i < positions.size(); ++i)
            positions[i] = static_cast<int32_t>(i % rotary_positions_.shape.last_dim());
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

struct WindowTranscriber::Impl {
    const ExecutionContext &execution;
    HarmonicFrontend frontend;
    LaneTranscriber network;
    std::unique_ptr<PreparedGraph> average;
    TensorValue final_indices;
    std::vector<int32_t> final_rows;
    std::vector<float> map_buffer;
    std::array<std::vector<float>, 12> host_maps;
    std::array<TensorValue, 3> raw;
    std::array<TensorValue, 2> indices;
    TensorValue taper;
    // Full maps/hidden/weight, then subgroup maps/hidden/stem/weight.
    std::array<TensorValue, 7> banks;
    std::array<TensorValue, 5> results;
    std::array<std::shared_ptr<std::vector<ggml_fp16_t>>, 3> host_taps{std::make_shared<std::vector<ggml_fp16_t>>(),
                                                                       std::make_shared<std::vector<ggml_fp16_t>>(),
                                                                       std::make_shared<std::vector<ggml_fp16_t>>()};
    std::array<std::shared_ptr<std::vector<ggml_fp16_t>>, 2> compact_taps{std::make_shared<std::vector<ggml_fp16_t>>(),
                                                                          std::make_shared<std::vector<ggml_fp16_t>>()};
    static constexpr int64_t ring_frames = 800;
    int64_t frames = 0;
    const int64_t hop_samples;
    bool ready = false;
    bool full_grouping_ready = true;
    std::chrono::steady_clock::time_point window_start;

    Impl(TranscriberWeights weights, const ExecutionContext &context, int window_hop_sec)
        : execution(context), frontend(std::move(weights.frontend), context.config().threads),
          network(std::move(weights), context), hop_samples(window_hop_sec * 16000) {
        if (window_hop_sec != 2 && window_hop_sec != 4)
            throw std::invalid_argument("Dettato window_hop_sec must be 2 or 4");
        if (window_hop_sec == 4)
            host_taps[1] = host_taps[0];
    }

    void prepare() {
        if (average)
            return;
        average = std::make_unique<PreparedGraph>(
            execution, PreparedGraphConfig{"dettato.window_average", 1024 * 1024, 1024, 1024 * 1024});
        auto &persistent = average->persistent_build_context();
        raw[0] = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2, 6, 400, 88}));
        raw[1] = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2, 368, 400, 96}));
        raw[2] = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2, 368, 400, 88}));
        taper = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({400, 1}));
        for (auto &value : indices) {
            value = make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({400}));
            ggml_set_input(value.tensor);
        }
        const std::array<int64_t, 7> widths{528, 88 * 368, 1, 528, 88 * 368, 88 * 368, 1};
        for (size_t i = 0; i < banks.size(); ++i) {
            if (hop_samples == 64000 && (i == 3 || i == 4 || i == 6)) {
                banks[i] = banks[i == 3 ? 0 : i == 4 ? 1 : 2];
                continue;
            }
            banks[i] = make_tensor(persistent, (i == 1 || i == 4 || i == 5) ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                   TensorShape::from_dims({ring_frames, widths[i]}));
        }
        const auto second = average->add_graph("odd_window");
        const auto finish = average->add_graph("finish");
        auto &ctx = average->build_context();
        for (int batch = 0; batch < 2; ++batch) {
            auto probabilities = SliceModule({0, batch, 1}).build(ctx, raw[0]);
            probabilities = SigmoidModule{}.build(ctx, probabilities);
            probabilities = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, probabilities);
            probabilities = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, probabilities),
                                           TensorShape::from_dims({400, 528}));
            probabilities = MulModule{true}.build(ctx, probabilities, taper);
            const auto weighted = [&](const TensorValue &input) {
                auto value = SliceModule({0, batch, 1}).build(ctx, input);
                value = SliceModule({3, 0, 88}).build(ctx, value);
                value = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, value);
                value = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, value),
                                       TensorShape::from_dims({400, 88 * 368}));
                value = MulModule{true}.build(ctx, value, taper);
                // The contribution rounds before addition to the half bank.
                value = wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F16), value.shape, GGML_TYPE_F16);
                return wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape);
            };
            const auto hidden = weighted(raw[2]);
            const auto graph = batch ? second : PreparedGraph::GraphId{0};
            const auto accumulate = [&](size_t index, const TensorValue &contribution) {
                auto previous = wrap_tensor(ggml_get_rows(ctx.ggml, banks[index].tensor, indices[batch].tensor),
                                            contribution.shape);
                const auto sum = AddModule{}.build(ctx, previous, contribution);
                auto *update = ggml_set_rows(ctx.ggml, banks[index].tensor, sum.tensor, indices[batch].tensor);
                ggml_set_output(update);
                ggml_build_forward_expand(average->graph(graph), update);
            };
            accumulate(0, probabilities);
            accumulate(1, hidden);
            accumulate(2, taper);
            if (batch == 0 || hop_samples == 64000) {
                if (hop_samples != 64000) {
                    accumulate(3, probabilities);
                    accumulate(4, hidden);
                    accumulate(6, taper);
                }
                accumulate(5, weighted(raw[1]));
            }
        }
        const int64_t final_frames = 2 * hop_samples / 320;
        final_indices = make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({final_frames}));
        ggml_set_input(final_indices.tensor);
        final_rows.resize(static_cast<size_t>(final_frames));
        map_buffer.resize(static_cast<size_t>(final_frames * 528));
        const std::array<size_t, 5> source{0, 3, 1, 4, 5};
        for (size_t i = 0; i < source.size(); ++i) {
            if (hop_samples == 64000 && (i == 1 || i == 3)) {
                results[i] = results[i - 1];
                continue;
            }
            auto value = wrap_tensor(ggml_get_rows(ctx.ggml, banks[source[i]].tensor, final_indices.tensor),
                                     TensorShape::from_dims({final_frames, banks[source[i]].shape.dims[1]}));
            const size_t weight_index = (i == 0 || i == 2) ? 2 : 6;
            auto weight = wrap_tensor(ggml_get_rows(ctx.ggml, banks[weight_index].tensor, final_indices.tensor),
                                      TensorShape::from_dims({final_frames, 1}));
            weight = wrap_tensor(ggml_clamp(ctx.ggml, weight.tensor, 1e-6f, INFINITY), weight.shape);
            if (i >= 2) {
                weight = wrap_tensor(ggml_cast(ctx.ggml, weight.tensor, GGML_TYPE_F16), weight.shape, GGML_TYPE_F16);
                weight = wrap_tensor(ggml_cast(ctx.ggml, weight.tensor, GGML_TYPE_F32), weight.shape);
                value = wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape);
            }
            value = wrap_tensor(ggml_div(ctx.ggml, value.tensor, weight.tensor), value.shape);
            if (i >= 2)
                value = wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F16), value.shape, GGML_TYPE_F16);
            results[i] = value;
            ggml_set_output(value.tensor);
            ggml_build_forward_expand(average->graph(finish), value.tensor);
        }
        average->allocate_persistent();
        average->allocate();
        std::vector<float> hann(400);
        for (size_t i = 0; i < hann.size(); ++i)
            hann[i] = static_cast<float>(0.5 - 0.5 * std::cos(2 * 3.14159265358979323846 * (i + 1) / 401));
        write_tensor_f32(taper, hann);
    }

    void run_device(const std::vector<float> &audio, bool full_grouping) {
        ready = false;
        if (audio.empty())
            throw std::runtime_error("Dettato needs nonempty audio");
        window_start = std::chrono::steady_clock::now();
        const auto samples = static_cast<int64_t>(audio.size());
        frames = (samples + 319) / 320;
        for (auto &map : host_maps)
            map.resize(static_cast<size_t>(frames * 88));
        for (size_t i = 0; i < host_taps.size(); ++i) {
            if (i == 1 && hop_samples == 64000)
                continue;
            host_taps[i]->resize(static_cast<size_t>(frames * 88 * 368));
        }
        double host_bytes = 0;
        for (const auto &map : host_maps)
            host_bytes += map.capacity() * sizeof(float);
        for (size_t i = 0; i < host_taps.size(); ++i)
            if (i != 1 || hop_samples != 64000)
                host_bytes += host_taps[i]->capacity() * sizeof(ggml_fp16_t);
        debug::timing_log_scalar("dettato.window_average.host_buffer_mb", host_bytes / (1024 * 1024));
        double frontend_ms = 0, averaging_ms = 0, finish_ms = 0;
        const auto prepare_features = [&](int64_t first) {
            const auto frontend_start = std::chrono::steady_clock::now();
            std::vector<float> windows(256000, 0);
            for (int64_t batch = 0; batch < 2; ++batch) {
                const auto origin = first + batch * hop_samples - 64000;
                const auto lo = std::max<int64_t>(origin, 0), hi = std::min<int64_t>(origin + 128000, samples);
                if (hi > lo)
                    std::copy(audio.begin() + lo, audio.begin() + hi, windows.begin() + batch * 128000 + lo - origin);
            }
            auto features = frontend.extract(windows, 2);
            return std::make_pair(std::move(features), debug::elapsed_ms(frontend_start));
        };
        // Only one frontend job is outstanding. Its private next-window data
        // overlaps GPU execution; no graph or FFT workspace is used concurrently.
        const bool prefetch = execution.backend_type() != BackendType::Cpu;
        std::future<std::pair<HarmonicFeatures, double>> next_features;
        std::vector<int32_t> rows(400);
        for (int64_t first = 0; first < samples + 64000; first += 2 * hop_samples) {
            auto prepared = next_features.valid() ? next_features.get() : prepare_features(first);
            if (prefetch && first + 2 * hop_samples < samples + 64000)
                next_features = std::async(std::launch::async, prepare_features, first + 2 * hop_samples);
            frontend_ms += prepared.second;
            const auto &features = prepared.first;
            const auto outputs = network.run_device(features.values, 2, 800);
            const auto average_start = std::chrono::steady_clock::now();
            prepare();
            for (size_t i = 0; i < raw.size(); ++i)
                ggml_backend_tensor_copy(outputs[i].tensor, raw[i].tensor);
            if (first == 0) {
                for (size_t i = 0; i < banks.size(); ++i)
                    if (hop_samples != 64000 || (i != 3 && i != 4 && i != 6))
                        ggml_backend_tensor_memset(banks[i].tensor, 0, 0, ggml_nbytes(banks[i].tensor));
            }
            for (int64_t batch = 0; batch < 2 && first + batch * hop_samples < samples + 64000; ++batch) {
                for (size_t i = 0; i < rows.size(); ++i)
                    rows[i] = static_cast<int32_t>(((first + batch * hop_samples) / 320 + i) % ring_frames);
                write_tensor_i32(indices[batch], rows);
                average->compute(static_cast<size_t>(batch));
            }
            averaging_ms += debug::elapsed_ms(average_start);
            // No future window touches rows before the next pair's start.
            // Normalize and download them, then reuse the bounded ring slots.
            const auto finish_start = std::chrono::steady_clock::now();
            const int64_t begin = first / 320, count = static_cast<int64_t>(final_rows.size());
            for (size_t i = 0; i < final_rows.size(); ++i)
                final_rows[i] = static_cast<int32_t>((begin + i) % ring_frames);
            write_tensor_i32(final_indices, final_rows);
            average->compute(2);
            const int64_t lo = std::max<int64_t>(begin, 200), hi = std::min(begin + count, frames + 200);
            if (hi > lo) {
                for (size_t group = 0; group < 2; ++group) {
                    ggml_backend_tensor_get(results[group].tensor, map_buffer.data(),
                                            static_cast<size_t>((lo - begin) * 528) * sizeof(float),
                                            static_cast<size_t>((hi - lo) * 528) * sizeof(float));
                    for (size_t head = 0; head < 6; ++head)
                        for (int64_t t = lo; t < hi; ++t)
                            std::copy_n(map_buffer.data() + (t - lo) * 528 + head * 88, 88,
                                        host_maps[group * 6 + head].data() + (t - 200) * 88);
                }
                for (size_t i = 0; i < host_taps.size(); ++i) {
                    if (i == 1 && hop_samples == 64000)
                        continue;
                    ggml_backend_tensor_get(results[i + 2].tensor, host_taps[i]->data() + (lo - 200) * 88 * 368,
                                            static_cast<size_t>((lo - begin) * 88 * 368) * sizeof(ggml_fp16_t),
                                            static_cast<size_t>((hi - lo) * 88 * 368) * sizeof(ggml_fp16_t));
                }
            }
            for (size_t i = 0; i < banks.size(); ++i) {
                if (hop_samples == 64000 && (i == 3 || i == 4 || i == 6))
                    continue;
                const auto slot = begin % ring_frames;
                const auto first_count = std::min(count, ring_frames - slot);
                const size_t row_bytes = banks[i].tensor->nb[1];
                ggml_backend_tensor_memset(banks[i].tensor, 0, static_cast<size_t>(slot) * row_bytes,
                                           static_cast<size_t>(first_count) * row_bytes);
                if (first_count < count)
                    ggml_backend_tensor_memset(banks[i].tensor, 0, 0,
                                               static_cast<size_t>(count - first_count) * row_bytes);
            }
            finish_ms += debug::elapsed_ms(finish_start);
        }
        debug::timing_log_scalar("dettato.transcriber.average_ms", averaging_ms);
        debug::timing_log_scalar("dettato.transcriber.finish_ms", finish_ms);
        debug::timing_log_scalar("dettato.transcriber.frontend_ms", frontend_ms);
        full_grouping_ready = full_grouping;
        ready = true;
    }

    WindowTranscription readback() {
        if (!ready)
            throw std::runtime_error("Dettato windows require a completed run_device() before readback");
        const auto readback_start = std::chrono::steady_clock::now();
        WindowTranscription out;
        out.frames = frames;
        out.hidden = host_taps[0];
        if (full_grouping_ready) {
            out.grouping_hidden = host_taps[1];
            out.grouping_stem = host_taps[2];
        }
        for (size_t head = 0; head < 6; ++head) {
            out.maps[head] = host_maps[head];
            out.grouping_maps[head] = host_maps[head + 6];
        }
        debug::timing_log_scalar("dettato.transcriber.readback_ms", debug::elapsed_ms(readback_start));
        debug::timing_log_scalar("dettato.transcriber.windows_ms", debug::elapsed_ms(window_start));
        return out;
    }
};

WindowTranscriber::WindowTranscriber(TranscriberWeights weights, const core::ExecutionContext &execution,
                                     int window_hop_sec)
    : impl_(std::make_unique<Impl>(std::move(weights), execution, window_hop_sec)) {}
WindowTranscriber::~WindowTranscriber() = default;
void WindowTranscriber::run_device(const std::vector<float> &audio16k, bool full_grouping) {
    impl_->run_device(audio16k, full_grouping);
}
WindowTranscription WindowTranscriber::readback() {
    return impl_->readback();
}

void WindowTranscriber::readback_grouping(WindowTranscription &out, const std::vector<Note> &notes) {
    auto &s = *impl_;
    if (!s.ready || out.frames != s.frames)
        throw std::runtime_error("Dettato grouping readback requires the current completed transcription");
    const auto start = std::chrono::steady_clock::now();
    out.compact_grouping = true;
    auto &rows = out.grouping_rows;
    rows.clear();
    for (const auto &note : notes) {
        const auto a = std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint(note.onset / .02)), 0, s.frames - 1);
        const auto end = std::min<int64_t>(
            std::max<int64_t>(static_cast<int64_t>(std::ceil(std::max(note.offset, note.onset + .02) / .02)), a + 1),
            s.frames);
        const auto lane = std::clamp(note.pitch - 21, 0, 87);
        for (auto t = a; t < std::min(end, a + 50); ++t)
            rows.push_back(static_cast<int32_t>(t * 88 + lane));
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    out.grouping_hidden = s.compact_taps[0];
    out.grouping_stem = s.compact_taps[1];
    for (size_t i = 0; i < s.compact_taps.size(); ++i) {
        auto &target = *s.compact_taps[i];
        target.resize(rows.size() * 368);
        const auto &source = *s.host_taps[i + 1];
        for (size_t j = 0; j < rows.size(); ++j)
            std::copy_n(source.data() + static_cast<size_t>(rows[j]) * 368, 368, target.data() + j * 368);
    }
    debug::timing_log_scalar("dettato.grouping_readback.rows", rows.size());
    debug::timing_log_scalar("dettato.grouping_readback.wall_ms", debug::elapsed_ms(start));
}
} // namespace engine::community_models::dettato::detail
