// TTSZipformer graph construction for the ZipVoice community port.
//
// Reference: zipvoice/models/modules/zipformer.py (k2-fsa/ZipVoice,
// Apache-2.0) evaluated in inference mode. Balancer / Whiten / Identity /
// dropout modules are no-ops; BypassModule reduces to a per-channel learned
// residual scale; ScaledLinear checkpoints already contain folded scales.
//
// Physical layouts (ne0 fastest):
//   activations:  [C (ne0), T (ne1), B (ne2)]
//   attention:    [src (ne0), tgt (ne1), H (ne2), B (ne3)]
// Softmax always runs over ne0 (keys), matching torch softmax(dim=-1) on
// (H, B, tgt, src).
//
// The compact relative-position attention lowers torch's overlapping
// as_strided view into an explicit strided ggml view over the contiguous
// relative scores R [2T-1 (ne0), T (ne1), H, B]:
//   abs[s, t] = R_flat[(T - 1) + t * (2T - 2) + s]
// where R_flat[m + t * (2T - 1)] = p_t . w_m. Validated against the
// moonlight fixture (tests/zipvoice).

#include "engine/community_models/zipvoice/zipformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace engine::models::zipvoice {
ZipVoiceGraphResources::~ZipVoiceGraphResources() {
    if (backend && graph) core::release_backend_graph_resources(backend, graph, true);
    if (gallocr) ggml_gallocr_free(static_cast<ggml_gallocr_t>(gallocr));
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    if (tensor_ctx) ggml_free(tensor_ctx);
}

namespace {

constexpr float kPadBias = -1000.0F;

// ---------------------------------------------------------------------------
// small helpers (raw ggml, channel-fastest layout)

// Physical views remain explicit for Zipformer's overlapping relative shift;
// ordinary projections, activation, normalization and convolution use modules.
struct GraphBuilder : core::ModuleBuildContext {
    GraphBuilder(ggml_context * c, core::BackendType type) { ggml = c; backend_type = type; }
    operator ggml_context *() const { return ggml; }
};
core::TensorValue value(ggml_tensor * t) {
    return core::wrap_tensor(t, core::TensorShape::from_dims({t->ne[3], t->ne[2], t->ne[1], t->ne[0]}), t->type);
}
ggml_tensor * t_cont(GraphBuilder & ctx, ggml_tensor * t) {
    return core::ensure_backend_addressable_layout(ctx, value(t)).tensor;
}
ggml_tensor * t_permute(GraphBuilder & ctx, ggml_tensor * t, int p0, int p1, int p2, int p3) {
    const int physical[4] = {p0, p1, p2, p3};
    std::array<int, 4> axes;
    for (int source = 0; source < 4; ++source) axes[3 - physical[source]] = 3 - source;
    return modules::TransposeModule({axes, 4}).build(ctx, value(t)).tensor;
}
ggml_tensor * t_reshape_2d(GraphBuilder & ctx, ggml_tensor * t, int64_t n0, int64_t n1) {
    return core::reshape_tensor(ctx, value(t), core::TensorShape::from_dims({n1, n0})).tensor;
}
ggml_tensor * t_reshape_3d(GraphBuilder & ctx, ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2) {
    return core::reshape_tensor(ctx, value(t), core::TensorShape::from_dims({n2, n1, n0})).tensor;
}
ggml_tensor * t_reshape_4d(GraphBuilder & ctx, ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    return core::reshape_tensor(ctx, value(t), core::TensorShape::from_dims({n3, n2, n1, n0})).tensor;
}
ggml_tensor * t_rows(GraphBuilder & ctx, ggml_tensor * table, ggml_tensor * indices) {
    return modules::EmbeddingModule({table->ne[1], table->ne[0]}).build(ctx,
        core::wrap_tensor(indices, core::TensorShape::from_dims({indices->ne[0]}), GGML_TYPE_I32),
        core::wrap_tensor(table, core::TensorShape::from_dims({table->ne[1], table->ne[0]}), table->type)).tensor;
}
ggml_tensor * t_matmul(GraphBuilder & ctx, ggml_tensor * a, ggml_tensor * b) {
    if (a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3]) {
        // MatMulModule requires equal batch dimensions. Relative-position
        // attention and weighted downsampling rely on ggml's batch broadcast.
        return ggml_mul_mat(ctx, t_cont(ctx, a), t_cont(ctx, b));
    }
    const auto rhs = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, value(a));
    return modules::MatMulModule().build(ctx, value(b), rhs).tensor;
}
ggml_tensor * t_add(GraphBuilder & ctx, ggml_tensor * a, ggml_tensor * b) {
    auto rhs = modules::RepeatModule({value(a).shape}).build(ctx, value(b));
    return modules::AddModule().build(ctx, value(a), rhs).tensor;
}
ggml_tensor * t_mul(GraphBuilder & ctx, ggml_tensor * a, ggml_tensor * b) {
    auto rhs = modules::RepeatModule({value(a).shape}).build(ctx, value(b));
    return modules::MulModule().build(ctx, value(a), rhs).tensor;
}
ggml_tensor * t_linear(GraphBuilder & ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    modules::LinearWeights weights{core::wrap_tensor(w, core::TensorShape::from_dims({w->ne[1], w->ne[0]}), w->type), std::nullopt};
    if (b) weights.bias = core::wrap_tensor(b, core::TensorShape::from_dims({b->ne[0]}), b->type);
    return modules::LinearModule({w->ne[0], w->ne[1], b != nullptr}).build(ctx, value(x), weights).tensor;
}
ggml_tensor * swoosh_l(GraphBuilder & ctx, ggml_tensor * x) {
    return modules::SwooshLModule().build(ctx, value(x)).tensor;
}
ggml_tensor * swoosh_r(GraphBuilder & ctx, ggml_tensor * x) {
    return modules::SwooshRModule().build(ctx, value(x)).tensor;
}
ggml_tensor * bias_norm(GraphBuilder & ctx, ggml_tensor * x, ggml_tensor * bias, float log_scale) {
    return modules::BiasNormModule({x->ne[0]}).build(ctx, value(x), {
        core::wrap_tensor(bias, core::TensorShape::from_dims({bias->ne[0]})), log_scale}).tensor;
}

std::vector<float> compact_relative_position(int64_t seq, int64_t pos_dim) {
    constexpr float pi = 3.14159265358979323846F;
    const int64_t length = 2 * seq - 1;
    std::vector<float> table(static_cast<size_t>(length * pos_dim), 0.0F);
    const float compression = std::sqrt(static_cast<float>(pos_dim));
    const float length_scale = static_cast<float>(pos_dim) / (2.0F * pi);
    for (int64_t row = 0; row < length; ++row) {
        const float x = static_cast<float>(row - (seq - 1));
        const float sign = x < 0.0F ? -1.0F : (x > 0.0F ? 1.0F : 0.0F);
        const float compressed = compression * sign *
            (std::log(std::fabs(x) + compression) - std::log(compression));
        const float angle = std::atan(compressed / length_scale);
        for (int64_t dim = 0; dim < pos_dim / 2; ++dim) {
            const float frequency = static_cast<float>(dim + 1);
            table[static_cast<size_t>(row * pos_dim + 2 * dim)] = std::cos(angle * frequency);
            table[static_cast<size_t>(row * pos_dim + 2 * dim + 1)] = std::sin(angle * frequency);
        }
        table[static_cast<size_t>(row * pos_dim + pos_dim - 1)] = 1.0F;
    }
    return table;
}

// Host constant factory. On CPU (inline ggml ctx) values are written
// directly; on CUDA (no_alloc ctx) they are staged and uploaded later.
struct StagedConst {
    ggml_tensor * tensor;
    std::vector<uint8_t> bytes;
};

struct Consts {
    // Tensors are created in the dedicated tensor context (never the graph
    // context) so ggml_backend_alloc_ctx_tensors can give them a private
    // backend buffer that the graph arena never aliases.
    ggml_context * ctx = nullptr;
    std::vector<StagedConst> * staged = nullptr;
    int index = 0;

    template <typename T>
    ggml_tensor * make(const T * data, size_t count, ggml_type type,
                       std::initializer_list<int64_t> dims) {
        std::vector<int64_t> d(dims);
        ggml_tensor * t = nullptr;
        switch (d.size()) {
            case 1: t = ggml_new_tensor_1d(ctx, type, d[0]); break;
            case 2: t = ggml_new_tensor_2d(ctx, type, d[0], d[1]); break;
            case 3: t = ggml_new_tensor_3d(ctx, type, d[0], d[1], d[2]); break;
            case 4: t = ggml_new_tensor_4d(ctx, type, d[0], d[1], d[2], d[3]); break;
            default: throw std::runtime_error("zipvoice: bad const rank");
        }
        ggml_set_name(t, ("zv_c" + std::to_string(index++)).c_str());
        const size_t bytes = count * sizeof(T);
        if (t->data != nullptr) {
            std::memcpy(t->data, data, bytes);
        } else if (staged != nullptr) {
            const auto * raw = reinterpret_cast<const uint8_t *>(data);
            staged->push_back({t, std::vector<uint8_t>(raw, raw + bytes)});
        } else {
            throw std::runtime_error("zipvoice: no home for graph constant");
        }
        return t;
    }

    ggml_tensor * f32(std::vector<float> values, std::initializer_list<int64_t> dims) {
        return make(values.data(), values.size(), GGML_TYPE_F32, dims);
    }
    ggml_tensor * i32(std::vector<int32_t> values, std::initializer_list<int64_t> dims) {
        return make(values.data(), values.size(), GGML_TYPE_I32, dims);
    }
};

// ---------------------------------------------------------------------------
// attention pieces

// RelPositionMultiheadAttentionWeights (inference path).
// src [C, T, B] -> weights [src(ne0), tgt(ne1), H(ne2), B(ne3)].
ggml_tensor * attention_weights(
    GraphBuilder & ctx,
    Consts & consts,
    const ZipLayerWeights & w,
    ggml_tensor * src,
    ggml_tensor * attn_bias,  // [T, 1, 1, 1] additive
    int64_t H,
    int64_t qh,
    int64_t ph,
    int64_t pos_dim,
    int64_t T,
    int64_t B,
    ggml_tensor ** tap_qtb = nullptr,
    ggml_tensor ** tap_ktb = nullptr,
    ggml_tensor ** tap_inproj = nullptr,
    ggml_tensor ** tap_scores = nullptr,
    ggml_tensor ** tap_pos = nullptr) {
    const int64_t q_total = H * qh;
    const int64_t M = 2 * T - 1;

    auto * projected = t_linear(ctx, src, w.attn_in_proj_w.tensor, w.attn_in_proj_b.tensor);
    if (tap_inproj != nullptr) *tap_inproj = projected;
    // [C_total, T, B]: slice channels, reshape (d, H), permute to (d, T, H, B)
    // slice [H*per] channels at `offset` (per-head split, head fastest after
    // the per-channel dim) -> [per, T, H, B]
    const auto qktb = [&](int64_t offset, int64_t total, int64_t per) {
        (void)total;
        // channel slice as a 4D view [per, H, T, B]: strides for T/B come
        // from the SOURCE (frame stride = full channel width), reshape would
        // assert on this non-contiguous layout
        auto * four = ggml_view_4d(ctx, projected, per, H, T, B,
                                   per * sizeof(float), projected->nb[1],
                                   projected->nb[2], offset * sizeof(float));
        return t_cont(ctx, t_permute(ctx, four, 0, 2, 1, 3));  // [per, T, H, B]
    };
    auto * q_tb = qktb(0, q_total, qh);
    auto * k_tb = qktb(q_total, q_total, qh);
    auto * p_tb = qktb(2 * q_total, H * ph, ph);
    if (tap_qtb != nullptr) *tap_qtb = q_tb;
    if (tap_ktb != nullptr) *tap_ktb = k_tb;

    // Raw mul_mat is retained for batched contractions: MatMulModule
    // requires equal batch dimensions and cannot express the position broadcast.
    // scores[src, tgt, H, B] = k^T q
    auto * scores = t_matmul(ctx, k_tb, q_tb);
    if (tap_scores != nullptr) *tap_scores = scores;

    // relative position scores: R[m, t] = p_t . w_m as [M, T, H, B]
    // (mul_mat broadcasts the degenerate batch dims of the pos operand)
    auto * pos_emb = consts.f32(compact_relative_position(T, pos_dim), {pos_dim, M});
    auto * pos_proj = t_matmul(ctx, w.linear_pos_w.tensor, pos_emb);  // [H*ph, M]
    auto * pos_4d = t_reshape_4d(ctx, pos_proj, ph, H, M, 1);
    auto * pos_tb = t_cont(ctx, t_permute(ctx, pos_4d, 0, 2, 1, 3));  // [ph, M, H, 1]
    auto * rel_t = t_matmul(ctx, pos_tb, p_tb);  // [M, T, H, B] contiguous

    // overlapping strided view (torch as_strided lowering):
    // abs[s, t] = flat[(T - 1) + t * (2T - 2) + s]
    const size_t nb0 = sizeof(float);
    const size_t nb1 = static_cast<size_t>(2 * T - 2) * nb0;
    const size_t nb2 = static_cast<size_t>(M) * static_cast<size_t>(T) * nb0;
    const size_t nb3 = nb2 * static_cast<size_t>(H);
    auto * pos_abs = ggml_view_4d(ctx, rel_t, T, T, H, B, nb1, nb2, nb3,
                                  static_cast<size_t>(T - 1) * nb0);
    auto * pos_final = t_cont(ctx, pos_abs);  // [src, tgt, H, B]

    auto * with_pos = t_add(ctx, scores, pos_final);
    if (tap_pos != nullptr) *tap_pos = with_pos;
    auto * biased = t_add(ctx, with_pos, attn_bias);
    return modules::SoftmaxModule().build(ctx, value(biased)).tensor;
}

// SelfAttention (value path). weights [src, tgt, H, B]; x [C, T, B].
ggml_tensor * self_attention(
    GraphBuilder & ctx,
    const ZipLayerWeights & w,
    int which,  // 1 or 2
    ggml_tensor * x,
    ggml_tensor * attn_weights,
    int64_t H,
    int64_t vh,
    int64_t T,
    int64_t B) {
    const auto & in_w = which == 1 ? w.sa1_in_w : w.sa2_in_w;
    const auto & in_b = which == 1 ? w.sa1_in_b : w.sa2_in_b;
    const auto & out_w = which == 1 ? w.sa1_out_w : w.sa2_out_w;
    const auto & out_b = which == 1 ? w.sa1_out_b : w.sa2_out_b;

    auto * v = t_linear(ctx, x, in_w.tensor, in_b.tensor);  // [H*vh, T, B]
    auto * v4 = t_reshape_4d(ctx, v, vh, H, T, B);
    // [T(src), vh, H, B] for the contraction over src
    auto * v_tb = t_cont(ctx, t_permute(ctx, v4, 1, 2, 0, 3));
    auto * attended = t_matmul(ctx, attn_weights, v_tb);  // [tgt, vh, H, B]
    // back to channel-fastest [H*vh, T, B]
    auto * ch = t_cont(ctx, t_permute(ctx, attended, 2, 0, 1, 3));  // [vh, H, T, B]
    auto * ch2 = t_reshape_2d(ctx, ch, H * vh, T * B);
    auto * out = t_linear(ctx, ch2, out_w.tensor, out_b.tensor);  // [C, T*B]
    return t_reshape_4d(ctx, out, out->ne[0], T, B, 1);
}

// NonlinAttention (uses head-0 weights only).
ggml_tensor * nonlin_attention(
    GraphBuilder & ctx,
    const ZipLayerWeights & w,
    ggml_tensor * x,
    ggml_tensor * attn_weights,  // [src, tgt, H, B]
    int64_t C,
    int64_t H,
    int64_t T,
    int64_t B,
    ggml_tensor ** tap_gated = nullptr,
    ggml_tensor ** tap_attended = nullptr,
    ggml_tensor ** tap_gtb = nullptr,
    ggml_tensor ** tap_attnrep = nullptr,
    ggml_tensor ** tap_mul = nullptr,
    ggml_tensor ** tap_h = nullptr,
    ggml_tensor ** tap_y2 = nullptr) {
    const int64_t hidden = C * 3 / 4;
    auto * h = t_linear(ctx, x, w.na_in_w.tensor, w.na_in_b.tensor);  // [3*hidden, T, B]
    if (tap_h != nullptr) *tap_h = h;
    const size_t frame_stride = h->nb[1];   // 3*hidden floats per frame
    const size_t batch_stride = h->nb[2];
    auto * s = ggml_view_3d(ctx, h, hidden, T, B,
                            frame_stride, batch_stride, 0);
    auto * xx = ggml_view_3d(ctx, h, hidden, T, B,
                             frame_stride, batch_stride, hidden * sizeof(float));
    auto * y = ggml_view_3d(ctx, h, hidden, T, B,
                            frame_stride, batch_stride, 2 * hidden * sizeof(float));
    auto * gated = t_mul(ctx, xx, modules::TanhModule().build(ctx, value(s)).tensor);  // [hidden, T, B]
    if (tap_gated != nullptr) *tap_gated = gated;

    // All heads share head-0 weights, so the per-head matmul collapses to a
    // single 2D contraction: out[t, c] = sum_s attn0[s, t] * gatedT[s, c],
    // with gated channels already in torch (h*hd + d) order.
    // out[t, c] = sum_s attn0[s, t] * gated_T[s, c]; gated channels are
    // already in torch (h*hd + d) order.
    auto * attn0_2d = t_cont(ctx, ggml_view_3d(ctx, attn_weights, T, T, B,
                                               attn_weights->nb[1], attn_weights->nb[3], 0));
    if (tap_attnrep != nullptr) *tap_attnrep = attn0_2d;
    ggml_tensor * gated_T;
    if (B == 1) {
        gated_T = t_cont(ctx, t_permute(ctx, gated, 1, 0, 2, 3));  // [s, c]
    } else {
        auto * g3 = t_reshape_4d(ctx, gated, hidden, T, B, 1);
        gated_T = t_cont(ctx, t_permute(ctx, g3, 1, 0, 2, 3));
    }
    if (tap_gtb != nullptr) *tap_gtb = gated_T;
    auto * attended = t_matmul(ctx, attn0_2d, gated_T);  // [t (ne0), c (ne1)]
    if (tap_attended != nullptr) *tap_attended = attended;
    ggml_tensor * ch2;
    if (B == 1) {
        ch2 = t_cont(ctx, t_permute(ctx, attended, 1, 0, 2, 3));  // [c, t]
    } else {
        auto * a3 = t_reshape_4d(ctx, attended, attended->ne[0], attended->ne[1], B, 1);
        ch2 = t_cont(ctx, t_permute(ctx, a3, 1, 0, 2, 3));
    }
    // y is a strided view; reshape would assert. Slice h directly as 2D:
    // element (c, r) with r = t + b*T sits at 2*hidden + c + r*3*hidden.
    auto * y2 = t_cont(ctx, ggml_view_2d(ctx, h, hidden, T * B,
                             h->nb[1], 2 * hidden * sizeof(float)));
    if (tap_y2 != nullptr) *tap_y2 = y2;
    auto * na_mul = t_mul(ctx, t_reshape_2d(ctx, ch2, hidden, T * B), y2);
    if (tap_mul != nullptr) *tap_mul = na_mul;
    auto * out = t_linear(ctx, na_mul, w.na_out_w.tensor, w.na_out_b.tensor);
    return t_reshape_4d(ctx, out, out->ne[0], T, B, 1);
}

// ConvolutionModule (non-causal, symmetric padding). conv_gate zeroes
// padded frames before the depthwise convolution (torch masked_fill).
ggml_tensor * conv_module(
    GraphBuilder & ctx,
    const ZipLayerWeights & w,
    int which,
    ggml_tensor * x,          // [C, T, B]
    ggml_tensor * conv_gate,  // [1, T, 1] (0/1)
    int64_t C,
    int64_t T,
    int64_t B) {
    const auto & in_w = which == 1 ? w.cm1_in_w : w.cm2_in_w;
    const auto & in_b = which == 1 ? w.cm1_in_b : w.cm2_in_b;
    const auto & conv_w = which == 1 ? w.cm1_conv_w : w.cm2_conv_w;
    const auto & conv_b = which == 1 ? w.cm1_conv_b : w.cm2_conv_b;
    const auto & out_w = which == 1 ? w.cm1_out_w : w.cm2_out_w;
    const auto & out_b = which == 1 ? w.cm1_out_b : w.cm2_out_b;
    const int64_t kernel = conv_w.tensor->ne[0];

    auto * h = t_linear(ctx, x, in_w.tensor, in_b.tensor);  // [2C, T, B]
    const size_t conv_frame_stride = h->nb[1];  // 2C floats per frame
    const size_t conv_batch_stride = h->nb[2];
    auto * xv = ggml_view_3d(ctx, h, C, T, B,
                             conv_frame_stride, conv_batch_stride, 0);
    auto * sv = ggml_view_3d(ctx, h, C, T, B,
                             conv_frame_stride, conv_batch_stride,
                             C * sizeof(float));
    auto * gated = t_mul(ctx, xv, modules::SigmoidModule().build(ctx, value(sv)).tensor);
    gated = t_mul(ctx, gated, conv_gate);  // zero padded frames

    auto input = core::wrap_tensor(gated, core::TensorShape::from_dims({B, T, C}));
    input = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    auto conv = modules::DepthwiseConv1dModule({C, kernel, 1, int(kernel / 2), 1, true})
        .build(ctx, input, {conv_w, conv_b});
    conv = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, conv);
    auto * with_bias = core::ensure_backend_addressable_layout(ctx, conv).tensor;

    // out_proj = SwooshR + linear
    auto * activated = swoosh_r(ctx, with_bias);
    auto * out = t_linear(ctx, activated, out_w.tensor, out_b.tensor);
    return t_reshape_4d(ctx, out, out->ne[0], T, B, 1);
}

// FeedforwardModule: linear -> SwooshL -> linear.
ggml_tensor * feedforward(
    GraphBuilder & ctx,
    const ZipLayerWeights & w,
    int which,
    ggml_tensor * x,
    int64_t T,
    int64_t B,
    ggml_tensor ** inproj_out = nullptr,
    ggml_tensor ** activated_out = nullptr) {
    const auto & in_w = which == 1 ? w.ff1_in_w : which == 2 ? w.ff2_in_w : w.ff3_in_w;
    const auto & in_b = which == 1 ? w.ff1_in_b : which == 2 ? w.ff2_in_b : w.ff3_in_b;
    const auto & out_w = which == 1 ? w.ff1_out_w : which == 2 ? w.ff2_out_w : w.ff3_out_w;
    const auto & out_b = which == 1 ? w.ff1_out_b : which == 2 ? w.ff2_out_b : w.ff3_out_b;
    auto * h = t_linear(ctx, x, in_w.tensor, in_b.tensor);
    if (inproj_out != nullptr) *inproj_out = h;
    auto * activated = swoosh_l(ctx, h);
    if (activated_out != nullptr) *activated_out = activated;
    auto * out = t_linear(ctx, activated, out_w.tensor, out_b.tensor);
    return t_reshape_4d(ctx, out, out->ne[0], T, B, 1);
}

// Per-submodule parity taps (first text-encoder layer only). ggml tensors
// mirror torch module outputs; transpose on readback.
struct LayerTaps {
    ggml_tensor * layer_in = nullptr;
    ggml_tensor * attn_inproj = nullptr;   // attention in_proj output
    ggml_tensor * attn_qtb = nullptr;      // q after permute [qh, T, H, B]
    ggml_tensor * attn_ktb = nullptr;      // k after permute [qh, T, H, B]
    ggml_tensor * attn_scores = nullptr;   // q@k before pos/bias
    ggml_tensor * attn_pos = nullptr;      // after pos add, before bias
    ggml_tensor * attn_weights = nullptr;
    ggml_tensor * ff1_inproj = nullptr;
    ggml_tensor * ff1 = nullptr;
    ggml_tensor * ff1_act = nullptr;
    ggml_tensor * na = nullptr;
    ggml_tensor * na_gated = nullptr;    // after x*tanh(s)
    ggml_tensor * na_gtb = nullptr;      // permuted gated [T, hd, H, B]
    ggml_tensor * na_attnrep = nullptr;  // repeated head-0 weights
    ggml_tensor * na_attended = nullptr; // after head-0 weighted matmul
    ggml_tensor * na_mul = nullptr;      // ch2 * y2 before out_proj
    ggml_tensor * na_y2 = nullptr;       // the y chunk [144, T, B]
    ggml_tensor * na_h = nullptr;        // in_proj output [3h, T, B]
    ggml_tensor * sa1 = nullptr;
    ggml_tensor * cm1_inproj = nullptr;
    ggml_tensor * conv1 = nullptr;
    ggml_tensor * ff2 = nullptr;
    ggml_tensor * bypass_mid = nullptr;
    ggml_tensor * sa2 = nullptr;
    ggml_tensor * conv2 = nullptr;
    ggml_tensor * ff3 = nullptr;
    ggml_tensor * norm = nullptr;
};

// BypassModule: src_orig + (src - src_orig) * bypass_scale.
ggml_tensor * bypass(GraphBuilder & ctx, ggml_tensor * scale_t, ggml_tensor * src_orig, ggml_tensor * src) {
    return modules::ScaledBypassModule().build(ctx, value(src_orig), value(src),
        core::wrap_tensor(scale_t, core::TensorShape::from_dims({scale_t->ne[0]}))).tensor;
}

// ---------------------------------------------------------------------------
// encoder layer / stack / zipformer

ggml_tensor * encoder_layer(
    GraphBuilder & ctx,
    Consts & consts,
    const ZipLayerWeights & w,
    ggml_tensor * src,          // [C, T, B]
    ggml_tensor * attn_bias,    // [T, 1, 1, 1]
    ggml_tensor * conv_gate,    // [1, T, 1]
    ggml_tensor * time_row,     // [C, 1, 1] or null
    const ZipVoiceConfig & config,
    int64_t kernel,
    int64_t T,
    int64_t B,
    bool fm,
    LayerTaps * taps = nullptr) {
    const int64_t C = src->ne[0];
    const int64_t H = fm ? config.fm_num_heads : config.text_num_heads;

    auto * src_orig = src;
    if (taps != nullptr) taps->layer_in = src;
    ggml_tensor * attn_qtb = nullptr;
    ggml_tensor * attn_ktb = nullptr;
    ggml_tensor * attn_inproj = nullptr;
    ggml_tensor * attn_scores = nullptr;
    ggml_tensor * attn_pos = nullptr;
    auto * attn_weights = attention_weights(
        ctx, consts, w, src, attn_bias, H, config.query_head_dim,
        config.pos_head_dim, config.pos_dim, T, B,
        taps != nullptr ? &attn_qtb : nullptr,
        taps != nullptr ? &attn_ktb : nullptr,
        taps != nullptr ? &attn_inproj : nullptr,
        taps != nullptr ? &attn_scores : nullptr,
        taps != nullptr ? &attn_pos : nullptr);
    if (taps != nullptr) {
        taps->attn_qtb = attn_qtb;
        taps->attn_ktb = attn_ktb;
        taps->attn_inproj = attn_inproj;
        taps->attn_scores = attn_scores;
        taps->attn_pos = attn_pos;
        taps->attn_weights = attn_weights;
    }

    auto * x = src;
    if (time_row != nullptr) {
        x = t_add(ctx, x, time_row);
    }
    ggml_tensor * ff1_inproj = nullptr;
    ggml_tensor * ff1_act = nullptr;
    auto * ff1_out = feedforward(ctx, w, 1, x, T, B, &ff1_inproj, &ff1_act);
    if (taps != nullptr) { taps->ff1_inproj = ff1_inproj; taps->ff1_act = ff1_act; taps->ff1 = ff1_out; }
    x = t_add(ctx, x, ff1_out);

    // nonlin attention uses only head 0
    ggml_tensor * na_gated = nullptr;
    ggml_tensor * na_attended = nullptr;
    ggml_tensor * na_gtb = nullptr;
    ggml_tensor * na_attnrep = nullptr;
    ggml_tensor * na_mul = nullptr;
    ggml_tensor * na_h = nullptr;
    ggml_tensor * na_y2 = nullptr;
    auto * na_out = nonlin_attention(ctx, w, x, attn_weights, C, H, T, B,
                                     &na_gated, &na_attended, &na_gtb, &na_attnrep,
                                     &na_mul, &na_h, &na_y2);
    if (taps != nullptr) {
        taps->na = na_out;
        taps->na_gated = na_gated;
        taps->na_gtb = na_gtb;
        taps->na_attnrep = na_attnrep;
        taps->na_attended = na_attended;
        taps->na_mul = na_mul;
        taps->na_y2 = na_y2;
        taps->na_h = na_h;
    }
    x = t_add(ctx, x, na_out);

    auto * sa1_out = self_attention(ctx, w, 1, x, attn_weights, H, config.value_head_dim, T, B);
    if (taps != nullptr) taps->sa1 = sa1_out;
    x = t_add(ctx, x, sa1_out);

    if (time_row != nullptr) {
        x = t_add(ctx, x, time_row);
    }
    // conv in_proj runs on the pre-conv activation (torch cm1_inproj input)
    auto * cm1_out = conv_module(ctx, w, 1, x, conv_gate, C, T, B);
    if (taps != nullptr) taps->conv1 = cm1_out;
    x = t_add(ctx, x, cm1_out);
    auto * ff2_out = feedforward(ctx, w, 2, x, T, B);
    if (taps != nullptr) taps->ff2 = ff2_out;
    x = t_add(ctx, x, ff2_out);

    x = bypass(ctx, w.bypass_mid_scale.tensor, src_orig, x);
    if (taps != nullptr) taps->bypass_mid = x;

    auto * sa2_out = self_attention(ctx, w, 2, x, attn_weights, H, config.value_head_dim, T, B);
    if (taps != nullptr) taps->sa2 = sa2_out;
    x = t_add(ctx, x, sa2_out);

    if (time_row != nullptr) {
        x = t_add(ctx, x, time_row);
    }
    auto * conv2_out = conv_module(ctx, w, 2, x, conv_gate, C, T, B);
    if (taps != nullptr) taps->conv2 = conv2_out;
    x = t_add(ctx, x, conv2_out);
    auto * ff3_out = feedforward(ctx, w, 3, x, T, B);
    if (taps != nullptr) taps->ff3 = ff3_out;
    x = t_add(ctx, x, ff3_out);

    x = bias_norm(ctx, x, w.norm_bias.tensor, w.norm_log_scale);
    if (taps != nullptr) taps->norm = x;
    x = bypass(ctx, w.bypass_scale.tensor, src_orig, x);
    return x;
}

// SimpleDownsample: softmax-weighted sum over ds frames; pads by repeating
// the last frame. Input [C, T, B] -> [C, ceil(T/ds), B].
ggml_tensor * downsample(
    GraphBuilder & ctx,
    Consts & consts,
    const std::vector<float> & softmax_bias,
    ggml_tensor * x,
    int64_t T,
    int64_t B) {
    const int64_t ds = static_cast<int64_t>(softmax_bias.size());
    const int64_t Td = (T + ds - 1) / ds;
    const int64_t Tp = Td * ds;
    const int64_t C = x->ne[0];

    auto * x2d = t_reshape_2d(ctx, x, C, T * B);  // rows (t, b) t-fastest
    std::vector<int32_t> idx(static_cast<size_t>(Tp * B));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t j = 0; j < Tp; ++j) {
            idx[static_cast<size_t>(j + b * Tp)] =
                static_cast<int32_t>(std::min(j, T - 1) + b * T);
        }
    }
    auto * gather = consts.i32(idx, {Tp * B});
    auto * padded = t_rows(ctx, x2d, gather);  // [C, Tp*B]

    // weighted sum over the ds sub-frame axis
    auto * windowed = t_reshape_4d(ctx, padded, C, ds, Td, B);
    auto * w_frame = t_permute(ctx, windowed, 3, 0, 1, 2);  // [ds, Td, B, C]
    auto * w_frame_c = t_cont(ctx, w_frame);
    auto * w2 = t_reshape_3d(ctx, w_frame_c, ds, Td * B, C);
    auto * weights2 = consts.f32(softmax_bias, {ds, 1});
    auto * summed = t_matmul(ctx, weights2, w2);  // [1, Td*B, C]
    auto * as_c = t_permute(ctx, summed, 1, 2, 0, 3);  // [C, 1, Td*B, 1]
    auto * as_c2 = t_cont(ctx, as_c);
    auto * out = t_reshape_3d(ctx, as_c2, C, Td, B);
    return out;
}

// SimpleUpsample: repeat each frame ds times, then truncate to T.
ggml_tensor * upsample(
    GraphBuilder & ctx,
    Consts & consts,
    int64_t ds,
    int64_t T,
    ggml_tensor * x,  // [C, Td, B]
    int64_t B) {
    const int64_t Td = x->ne[1];
    const int64_t C = x->ne[0];
    auto * x2d = t_reshape_2d(ctx, x, C, Td * B);
    const int64_t Tp = Td * ds;
    std::vector<int32_t> idx(static_cast<size_t>(Tp * B));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < Tp; ++i) {
            idx[static_cast<size_t>(i + b * Tp)] =
                static_cast<int32_t>(i / ds + b * Td);
        }
    }
    auto * gather = consts.i32(idx, {Tp * B});
    auto * out = t_rows(ctx, x2d, gather);  // [C, Tp*B] contiguous
    if (Tp == T) {
        return t_reshape_3d(ctx, out, C, T, B);
    }
    // truncate to T frames: rows are (frame, batch) frame-fastest
    return ggml_view_3d(ctx, out, C, T, B,
                        C * sizeof(float), static_cast<size_t>(Tp) * C * sizeof(float), 0);
}

ggml_tensor * zipformer_forward(
    GraphBuilder & ctx,
    Consts & consts,
    const TTSZipformerWeights & w,
    const ZipVoiceConfig & config,
    ggml_tensor * input,         // [in_dim, T, B]
    ggml_tensor * time_emb,      // [time_embed_dim] or null
    ggml_tensor * const * pad_bias,
    ggml_tensor * const * conv_gate,
    bool fm,
    int64_t B,
    std::vector<ggml_tensor *> * layer_taps = nullptr,
    LayerTaps * first_layer_taps = nullptr) {
    const bool with_time = fm;
    auto * x = t_linear(ctx, input, w.in_proj_w.tensor, w.in_proj_b.tensor);  // [C, T, B]

    ggml_tensor * outer_time = nullptr;
    if (with_time && time_emb != nullptr) {
        auto * h = t_linear(ctx, time_emb, w.time_mlp0_w.tensor, w.time_mlp0_b.tensor);
        outer_time = t_linear(ctx, swoosh_r(ctx, h), w.time_mlp2_w.tensor, w.time_mlp2_b.tensor);
    }

    const int64_t T = x->ne[1];
    for (size_t s = 0; s < w.stacks.size(); ++s) {
        auto & stack = w.stacks[s];
        const int64_t ds = fm ? config.fm_downsampling_factor[s] : 1;
        const int64_t kernel = fm ? config.fm_cnn_kernel[s] : config.text_cnn_kernel;

        ggml_tensor * time_row = nullptr;
        if (with_time && stack.time_proj_w.tensor != nullptr) {
            // torch: encoders[i].time_emb = Sequential(SwooshR, Linear)
            auto * activated = swoosh_r(ctx, outer_time);
            auto * projected = t_linear(ctx, activated, stack.time_proj_w.tensor, stack.time_proj_b.tensor);
            time_row = t_reshape_3d(ctx, projected, projected->ne[0], 1, 1);
        }

        ggml_tensor * stack_in = x;
        if (ds > 1) {
            stack_in = downsample(ctx, consts, stack.downsample_bias, x, T, B);
        }
        const int64_t Ts = stack_in->ne[1];

        auto * h = stack_in;
        for (size_t li = 0; li < stack.layers.size(); ++li) {
            h = encoder_layer(
                ctx, consts, stack.layers[li], h, pad_bias[s], conv_gate[s], time_row,
                config, kernel, Ts, B, fm,
                first_layer_taps != nullptr && s == 0 && li == 0 ? first_layer_taps : nullptr);
            if (layer_taps != nullptr) {
                layer_taps->push_back(h);
            }
        }

        if (ds > 1) {
            auto * up = upsample(ctx, consts, ds, T, h, B);
            x = bypass(ctx, stack.out_combiner_scale.tensor, x, up);
        } else {
            x = h;
        }
    }
    auto * out = t_linear(ctx, x, w.out_proj_w.tensor, w.out_proj_b.tensor);
    return t_reshape_4d(ctx, out, out->ne[0], out->ne[1], B, 1);
}

struct GraphBuildResult {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
};

GraphBuildResult finish_graph(
    ggml_context * ctx,
    ggml_context * tensor_ctx,
    ggml_tensor * output,
    const std::vector<ggml_tensor *> & extra_roots,
    std::vector<StagedConst> & staged,
    size_t node_budget,
    ggml_backend_t backend,
    void ** gallocr_out,
    ggml_backend_buffer_t * buffer_out) {
    GraphBuildResult result;
    result.ctx = ctx;
    result.graph = ggml_new_graph_custom(ctx, node_budget, false);
    ggml_build_forward_expand(result.graph, output);
    // ALL extra outputs must be expanded BEFORE the arena reservation:
    // nodes appended after ggml_gallocr_alloc_graph never get memory.
    for (ggml_tensor * tap : extra_roots) {
        if (tap != nullptr) {
            ggml_build_forward_expand(result.graph, tap);
        }
    }
    if (backend == nullptr) {
        throw std::runtime_error("zipvoice: graph backend is required");
    }
    // Half-staged matrix products accumulate audible error across the flow
    // solver steps. Keep the CPU oracle unchanged and request F32 on GPUs.
    if (core::backend_type(backend) != core::BackendType::Cpu) {
        for (int i = 0; i < ggml_graph_n_nodes(result.graph); ++i) {
            auto * node = ggml_graph_node(result.graph, i);
            if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
    }
    core::validate_backend_graph_supported(backend, result.graph, "zipvoice");
    // private storage for leaves + constants FIRST: with a buffer already
    // assigned, gallocr skips these tensors instead of arena-aliasing them
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(tensor_ctx, backend);
    if (buffer == nullptr) {
        throw std::runtime_error("zipvoice: tensor buffer allocation failed");
    }
    *buffer_out = buffer;
    auto * gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    *gallocr_out = gallocr;
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, result.graph) ||
        !ggml_gallocr_alloc_graph(gallocr, result.graph)) {
        throw std::runtime_error("zipvoice: graph allocation failed");
    }
    for (auto & item : staged) {
        ggml_backend_tensor_set(item.tensor, item.bytes.data(), 0, item.bytes.size());
    }
    staged.clear();
    return result;
}

}  // namespace

FmDecoderGraph build_fm_decoder_graph(
    const ZipVoiceWeights & weights,
    const ZipVoiceConfig & config,
    int64_t T,
    int64_t B,
    bool with_guidance,
    bool cuda_backend,
    ggml_backend_t backend) {
    (void)cuda_backend;
    const size_t ctx_bytes = 512ULL * 1024ULL * 1024ULL;
    ggml_context * ctx = ggml_init({ctx_bytes, nullptr, true});
    // leaves + constants get their own context and backend buffer so the
    // graph arena (gallocr) never aliases them (see skill: gallocr aliasing)
    ggml_context * tensor_ctx = ggml_init({64ULL * 1024ULL * 1024ULL, nullptr, true});
    FmDecoderGraph g;
    g.ctx = ctx;
    g.tensor_ctx = tensor_ctx;
    g.T = T;
    g.B = B;
    g.cuda = false;
    g.backend = backend;

    std::vector<StagedConst> staged;
    Consts consts{tensor_ctx, &staged};

    auto * x_cat = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, config.feat_dim * 3, T, B);
    ggml_set_input(x_cat);
    g.x_cat = x_cat;

    auto * time_emb = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, config.time_embed_dim);
    ggml_set_input(time_emb);
    g.time_emb = time_emb;

    if (with_guidance) {
        auto * guidance = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, config.time_embed_dim);
        ggml_set_input(guidance);
        g.guidance_emb = guidance;
    }

    // per-stack masks. Downsampling factors are relative to the full frame
    // rate (U-Net style), so stack s runs at ceil(T / ds[s]) frames.
    for (size_t s = 0; s < weights.fm_decoder.stacks.size(); ++s) {
        const int64_t ds = config.fm_downsampling_factor[s];
        const int64_t T_s = (T + ds - 1) / ds;
        auto * bias = ggml_new_tensor_4d(tensor_ctx, GGML_TYPE_F32, T_s, 1, 1, 1);
        ggml_set_input(bias);
        g.pad_bias[s] = bias;
        auto * gate = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 1, T_s, 1);
        ggml_set_input(gate);
        g.conv_gate[s] = gate;
    }

    GraphBuilder builder(ctx, core::backend_type(backend));
    ggml_tensor * time_input = g.time_emb;
    if (with_guidance && weights.fm_decoder.guidance_embed_w.tensor != nullptr) {
        auto * guidance_proj = t_linear(
            builder, g.guidance_emb, weights.fm_decoder.guidance_embed_w.tensor, nullptr);
        time_input = t_add(builder, g.time_emb, guidance_proj);
    }

    auto * out = zipformer_forward(
        builder, consts, weights.fm_decoder, config, x_cat, time_input,
        g.pad_bias, g.conv_gate, true, B);
    ggml_set_output(out);
    g.output = out;

    auto built = finish_graph(ctx, tensor_ctx, out, {}, staged, 262144, backend, &g.gallocr, &g.buffer);
    g.graph = built.graph;
    return g;
}

TextEncoderGraph build_text_encoder_graph(
    const ZipVoiceWeights & weights,
    const ZipVoiceConfig & config,
    int64_t S,
    bool cuda_backend,
    ggml_backend_t backend) {
    (void)cuda_backend;
    const size_t ctx_bytes = 256ULL * 1024ULL * 1024ULL;
    ggml_context * ctx = ggml_init({ctx_bytes, nullptr, true});
    ggml_context * tensor_ctx = ggml_init({16ULL * 1024ULL * 1024ULL, nullptr, true});
    TextEncoderGraph g;
    g.ctx = ctx;
    g.tensor_ctx = tensor_ctx;
    g.S = S;
    g.cuda = false;
    g.backend = backend;

    std::vector<StagedConst> staged;
    Consts consts{tensor_ctx, &staged};

    auto * ids = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_I32, S);
    ggml_set_input(ids);
    g.token_ids = ids;

    auto * bias = ggml_new_tensor_4d(tensor_ctx, GGML_TYPE_F32, S, 1, 1, 1);
    ggml_set_input(bias);
    g.pad_bias[0] = bias;
    auto * gate = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, 1, S, 1);
    ggml_set_input(gate);
    g.conv_gate[0] = gate;

    GraphBuilder builder(ctx, core::backend_type(backend));
    auto * embedded = t_rows(builder, weights.text_encoder.embed_w.tensor, ids);
    std::vector<ggml_tensor *> layer_taps;
    LayerTaps first_layer;
    auto * out = zipformer_forward(
        builder, consts, weights.text_encoder, config, embedded, nullptr,
        g.pad_bias, g.conv_gate, false, 1, &layer_taps, &first_layer);
    ggml_set_output(out);
    g.output = out;
    std::vector<ggml_tensor *> extra_roots;
    const auto mark_output = [](ggml_tensor * tap) {
        ggml_set_output(tap);
        // reshape results are VIEWS: the gallocr frees/reuses the underlying
        // node unless it is flagged too (graph outputs are never freed).
        if (tap->view_src != nullptr) {
            ggml_set_output(tap->view_src);
        }
    };
    for (auto * tap : layer_taps) {
        mark_output(tap);
        g.layer_taps.push_back(tap);
        extra_roots.push_back(tap);
    }
    for (ggml_tensor * tap : {
                 first_layer.layer_in, first_layer.attn_qtb, first_layer.attn_ktb,
                 first_layer.attn_inproj, first_layer.attn_scores,
                 first_layer.attn_pos, first_layer.attn_weights, first_layer.ff1_inproj,
                 first_layer.ff1_act, first_layer.ff1, first_layer.na, first_layer.na_gated,
                 first_layer.na_gtb, first_layer.na_attnrep, first_layer.na_attended, first_layer.na_mul, first_layer.na_y2, first_layer.na_h, first_layer.sa1, first_layer.cm1_inproj,
                 first_layer.conv1, first_layer.ff2, first_layer.bypass_mid,
                 first_layer.sa2, first_layer.conv2, first_layer.ff3, first_layer.norm}) {
        if (tap != nullptr) {
            mark_output(tap);
            extra_roots.push_back(tap);
        }
    }

    auto built = finish_graph(ctx, tensor_ctx, out, extra_roots, staged, 262144, backend, &g.gallocr, &g.buffer);
    g.stage_taps.clear();
    for (ggml_tensor * tap : {
                 first_layer.layer_in, first_layer.attn_qtb, first_layer.attn_ktb,
                 first_layer.attn_inproj, first_layer.attn_scores,
                 first_layer.attn_pos, first_layer.attn_weights, first_layer.ff1_inproj,
                 first_layer.ff1_act, first_layer.ff1, first_layer.na, first_layer.na_gated,
                 first_layer.na_gtb, first_layer.na_attnrep, first_layer.na_attended, first_layer.na_mul, first_layer.na_y2, first_layer.na_h, first_layer.sa1, first_layer.cm1_inproj,
                 first_layer.conv1, first_layer.ff2, first_layer.bypass_mid,
                 first_layer.sa2, first_layer.conv2, first_layer.ff3, first_layer.norm}) {
        if (tap != nullptr) g.stage_taps.push_back(tap);
    }
    g.graph = built.graph;
    return g;
}

}  // namespace engine::models::zipvoice
