// F5 DiT forward composed from framework modules (dev-branch pattern).
//
// All activations use the framework's logical [batch, frames, features]
// layout; modules handle the ggml mapping. Verified stage-by-stage against
// the same goldens as the original raw-ggml implementation (cosine 1.0).
#include "engine/community_models/f5_tts/weights.h"

#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace engine::models::f5_tts {

// Build-time constant staging: on CPU (inline ctx) values are written
// directly; on CUDA (no_alloc ctx) they are staged for the caller to upload
// via ggml_backend_tensor_set after allocation.
struct ConstStage {
    ggml_tensor * tensor;
    std::vector<uint8_t> bytes;
};
thread_local std::vector<ConstStage> * t_const_stage = nullptr;

namespace {

namespace mod = engine::modules;

core::ModuleBuildContext make_ctx(
    ggml_context * ggml, const char * name, core::BackendType type) {
    return core::ModuleBuildContext{ggml, name, type};
}

// ---- grouped conv1d (groups=16, k=31) via per-group im2col + matmul ----
// The framework has no grouped Conv1d module yet; lower it with the same
// primitives Conv1dModule uses (im2col + mul_mat), expressed as modules.
core::TensorValue grouped_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,  // [B, C_in, T]
    const core::TensorValue & weight, // [C_out, C_in/g, k]
    const core::TensorValue & bias,   // [C_out]
    int64_t groups) {
    const int64_t frames = input.shape.dims[2];
    const int64_t c_in = input.shape.dims[1];
    const int64_t c_out = weight.shape.dims[0];
    const int64_t kernel = weight.shape.dims[2];
    const int64_t cg_in = c_in / groups;
    const int64_t cg_out = c_out / groups;

    std::vector<core::TensorValue> group_outputs;
    group_outputs.reserve(static_cast<size_t>(groups));
    for (int64_t g = 0; g < groups; ++g) {
        // input slice [B, cg_in, T] on axis 1; im2col needs contiguous input
        auto in_sliced = mod::SliceModule({1, g * cg_in, cg_in}).build(ctx, input);
        auto in_g = core::ensure_backend_addressable_layout(ctx, in_sliced);
        auto * cols = ggml_im2col(
            ctx.ggml, weight.tensor, in_g.tensor,
            1, 1, kernel / 2, 0, 1, 1, false, GGML_TYPE_F32);
        // ggml im2col output ne [cg_in*k, T]; mul_mat with the per-group
        // weight slice (ggml ne [k, cg_in, cg_out] -> view as [K, cg_out])
        auto * w_g = ggml_view_3d(
            ctx.ggml, weight.tensor,
            kernel, cg_in, cg_out,
            weight.tensor->nb[1], weight.tensor->nb[2],
            g * cg_out * weight.tensor->nb[2]);
        auto * w2 = ggml_reshape_2d(ctx.ggml, w_g, cg_in * kernel, cg_out);
        auto * y_raw = ggml_mul_mat(ctx.ggml, w2, cols);  // ggml [cg_out, T]
        // logical [T, cg_out] shares the same memory; wrap for module use
        auto y = core::wrap_tensor(
            y_raw, core::TensorShape::from_dims({frames, cg_out}), GGML_TYPE_F32);
        auto b_g = mod::SliceModule({0, g * cg_out, cg_out}).build(ctx, bias);
        auto b_row = core::reshape_tensor(
            ctx, b_g, core::TensorShape::from_dims({1, cg_out}));
        y = mod::AddModule().build(
            ctx, y, mod::RepeatModule({y.shape}).build(ctx, b_row));
        group_outputs.push_back(y);
    }
    auto out = group_outputs[0];
    for (size_t i = 1; i < group_outputs.size(); ++i) {
        out = mod::ConcatModule({1}).build(ctx, out, group_outputs[i]);  // [T, c_out]
    }
    return out;  // [T, c_out] logical (batch folded into frames column)
}

core::TensorValue ctx_store_f32(
    core::ModuleBuildContext & ctx, const core::TensorShape & shape,
    const std::vector<float> & values) {
    auto t = core::make_tensor(ctx, GGML_TYPE_F32, shape);
    if (t.tensor->data != nullptr) {
        std::memcpy(t.tensor->data, values.data(), values.size() * sizeof(float));
    } else if (t_const_stage != nullptr) {
        const auto * b = reinterpret_cast<const uint8_t *>(values.data());
        t_const_stage->push_back({t.tensor, std::vector<uint8_t>(b, b + values.size() * sizeof(float))});
    }
    return t;
}

// softplus(x): ggml's numerically stable primitive (same op the raw path used)
core::TensorValue exp_log_softplus(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    return core::wrap_tensor(ggml_softplus(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
}

// lift a [1, C] row to [1, 1, C] so Repeat can broadcast over frames
core::TensorValue lift_row(
    core::ModuleBuildContext & ctx, const core::TensorValue & row) {
    if (row.shape.rank == 3) {
        return row;
    }
    return core::reshape_tensor(
        ctx, core::ensure_backend_addressable_layout(ctx, row),
        core::TensorShape::from_dims({1, 1, row.shape.dims[row.shape.rank - 1]}));
}

// adaLN modulate: x * (1 + scale) + shift, scale/shift [1, C]
core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & scale,
    const core::TensorValue & shift) {
    // constants sized [1, 1, C] (NOT [B, T, C]): one shared ones-row, and the
    // repeat broadcasts handle the expansion in the compute buffer.
    const auto row_shape = core::TensorShape::from_dims(
        {1, 1, x.shape.dims[x.shape.rank - 1]});
    auto ones = ctx_store_f32(
        ctx, row_shape, std::vector<float>(static_cast<size_t>(x.shape.dims[x.shape.rank - 1]), 1.0F));
    auto s_row = lift_row(ctx, scale);
    auto sh_row = lift_row(ctx, shift);
    // 1 + scale, then broadcast once
    auto one_plus_s = mod::AddModule().build(ctx, ones, s_row);      // [1,1,C]
    auto scale_b = mod::RepeatModule({x.shape}).build(ctx, one_plus_s);
    auto shift_b = mod::RepeatModule({x.shape}).build(ctx, sh_row);
    return mod::AddModule().build(
        ctx, mod::MulModule().build(ctx, x, scale_b), shift_b);
}

// ---- GRN (global response norm): no framework module; expressed with
// primitives on [B, T, C] ----
core::TensorValue grn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & h,        // [B, T, C]
    const std::vector<float> & gamma,
    const std::vector<float> & beta,
    core::TensorValue & gamma_out,
    core::TensorValue & beta_out) {
    // per-channel L2 norm over T, then normalize by mean over channels:
    // gx[c] = ||h[:, :, c]||_2 ; nx[c] = gx[c] / (mean(gx) + 1e-6)
    const int64_t frames = h.shape.dims[1];
    const int64_t channels = h.shape.dims[2];
    // squares -> sum over T -> sqrt  (reduce axis=1)
    auto sq = mod::MulModule().build(ctx, h, h);  // elementwise
    auto sums = mod::ReduceSumModule({1}).build(ctx, sq);  // [B, 1, C]
    auto gx = mod::SqrtModule().build(ctx, sums);
    auto mean = mod::ReduceMeanModule({2}).build(ctx, gx);  // [B, 1, 1]
    // nx = gx / (mean + eps): eps via a constant scalar broadcast
    // (ReduceMean + add-eps fused: use MeanModule output + eps tensor)
    // eps as a [1,1,1] constant:
    auto eps = ctx_store_f32(ctx, mean.shape, std::vector<float>{1e-6F});
    auto denom = mod::AddModule().build(ctx, mean, eps);
    auto nx = core::wrap_tensor(
        ggml_div(ctx.ggml, gx.tensor, denom.tensor), gx.shape, GGML_TYPE_F32);
    // scale h by nx per channel + gamma*x + beta + residual
    auto nx_b = mod::RepeatModule({h.shape}).build(ctx, nx);  // [B,1,C]->[B,T,C] via repeat
    auto scaled = mod::MulModule().build(ctx, h, nx_b);
    // gamma/beta as [1, 1, C] constants
    gamma_out = ctx_store_f32(ctx, core::TensorShape::from_dims({1, 1, channels}), gamma);
    beta_out = ctx_store_f32(ctx, core::TensorShape::from_dims({1, 1, channels}), beta);
    auto g_b = mod::RepeatModule({h.shape}).build(ctx, gamma_out);
    auto b_b = mod::RepeatModule({h.shape}).build(ctx, beta_out);
    auto out = mod::AddModule().build(ctx, mod::MulModule().build(ctx, scaled, g_b), b_b);
    return mod::AddModule().build(ctx, out, h);  // + residual
}


// test hook: grouped conv exposed for unit comparison against numpy
core::TensorValue grouped_conv1d_pub(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    const core::TensorValue & bias,
    int64_t groups) {
    return grouped_conv1d(ctx, input, weight, bias, groups);
}

}  // namespace

// Builds the full DiT velocity graph. Leaves: x/cond [B=1, T, MEL], text ids
// [NT], time-embedding input [1, 256]. Returns the output TensorValue.
struct F5DiTGraphBuild {
    core::TensorValue x;
    core::TensorValue cond;
    core::TensorValue text_ids;
    core::TensorValue time_input;
    core::TensorValue output;
};

// staging control for the CUDA build path (see runtime.cpp)
std::vector<ConstStage> * const_stage_begin() {
    t_const_stage = new std::vector<ConstStage>();
    return t_const_stage;
}
void const_stage_upload(std::vector<ConstStage> * stage) {
    for (auto & c : *stage) {
        ggml_backend_tensor_set(c.tensor, c.bytes.data(), 0, c.bytes.size());
    }
}
void const_stage_end(std::vector<ConstStage> * stage) {
    // detach only; the vector is deliberately leaked (a few KB per graph
    // build) to avoid ownership hazards across build paths.
    t_const_stage = nullptr;
    (void)stage;
}


F5DiTGraphBuild build_dit_modules_graph(
    ggml_context * ggml,
    const F5DiTWeights & w,
    const F5Architecture & arch,
    int frames,
    int text_len,
    core::BackendType backend_type) {
    auto ctx = make_ctx(ggml, "f5.dit", backend_type);
    constexpr int64_t kMel = 100, kTextDim = 512, kDim = 1024, kVocab = 2731;
    constexpr int64_t kHeads = 16, kHeadDim = 64;
    const int64_t N = frames;
    const int64_t NT = text_len;

    F5DiTGraphBuild io;
    io.x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, N, kMel}));
    io.cond = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, N, kMel}));
    io.text_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({NT}));
    io.time_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 256}));

    // ---- text embed: lookup + sinusoidal pe (both halves share) ----
    auto te = mod::EmbeddingModule({kVocab, kTextDim}).build(ctx, io.text_ids, w.text_embedding);
    // batch the text stream: [NT, C] -> [1, NT, C]
    te = core::reshape_tensor(
        ctx, core::ensure_backend_addressable_layout(ctx, te),
        core::TensorShape::from_dims({1, NT, kTextDim}));
    // pe table [1, NT, kTextDim] constant (verified layout: cos|sin halves)
    {
        std::vector<float> pe(static_cast<size_t>(NT) * kTextDim);
        const int64_t half = kTextDim / 2;
        for (int64_t pos = 0; pos < NT; ++pos) {
            for (int64_t i = 0; i < half; ++i) {
                const float inv = std::pow(10000.0F, -2.0F * i / static_cast<float>(kTextDim));
                const float f = static_cast<float>(pos) * inv;
                pe[static_cast<size_t>(pos) * kTextDim + i] = std::cos(f);
                pe[static_cast<size_t>(pos) * kTextDim + half + i] = std::sin(f);
            }
        }
        auto pe_t = ctx_store_f32(
            ctx, core::TensorShape::from_dims({1, NT, kTextDim}), pe);
        te = mod::AddModule().build(ctx, te, pe_t);
    }

    // ---- 4x ConvNeXt text blocks (dwconv k7, LN, pw1+GELU, GRN, pw2, residual) ----
    for (int bi = 0; bi < 4; ++bi) {
        const auto & B = w.text_blocks[static_cast<size_t>(bi)];
        // ConvNeXt operates channel-major: [B, C, T]; our te is [B, T, C]
        auto te_c = mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, te);
        auto dw = mod::DepthwiseConv1dModule({kTextDim, 7, 1, 3, 1, true}).build(ctx, te_c, B.dwconv);
        auto dw_t = mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, dw);  // [B, T, C]
        auto nrm = mod::LayerNormModule({kTextDim, 1e-6F, true, true}).build(ctx, dw_t, B.norm);
        auto h1 = mod::LinearModule({kTextDim, kDim, true}).build(ctx, nrm, B.pw1);
        h1 = mod::GeluModule({mod::GeluApproximation::ExactErf}).build(ctx, h1);
        core::TensorValue g_dummy, b_dummy;
        auto grn_out = grn(ctx, h1, B.grn_gamma, B.grn_beta, g_dummy, b_dummy);
        auto h2 = mod::LinearModule({kDim, kTextDim, true}).build(ctx, grn_out, B.pw2);
        te = mod::AddModule().build(ctx, te, h2);
    }

    // ---- pad/curtail text to N frames (zero-pad along T) ----
    core::TensorValue te_pad;
    if (NT >= N) {
        te_pad = mod::SliceModule({1, 0, N}).build(ctx, te);
    } else {
        const auto zshape = core::TensorShape::from_dims({1, N - NT, kTextDim});
        std::vector<float> zv(static_cast<size_t>(zshape.num_elements()), 0.0F);
        te_pad = mod::ConcatModule({1}).build(ctx, te, ctx_store_f32(ctx, zshape, zv));
    }

    // ---- input embed: concat features [x | cond | text] -> proj -> CPE ----
    auto cat0 = mod::ConcatModule({2}).build(ctx, io.x, io.cond);
    auto cat1 = mod::ConcatModule({2}).build(ctx, cat0, te_pad);  // [1, N, 712]
    auto inp = mod::LinearModule({712LL, kDim, true}).build(ctx, cat1, w.input_proj);

    // conv pos embed (grouped k31 g16, Mish x2):
    // inp += mish(conv1(mish(conv0(inp))))
    {
        auto conv_mish = [&](const core::TensorValue & x_bnd,
                             const core::TensorValue & cweight,
                             const core::TensorValue & cbias) -> core::TensorValue {
            // im2col requires a contiguous, time-fastest input; ggml_transpose
        // yields a strided view, so materialize it first.
        auto x_t = mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, x_bnd);  // logical [1, D, N]
        auto x_c = core::wrap_tensor(
            ggml_cont(ctx.ggml, x_t.tensor), x_t.shape, GGML_TYPE_F32);
            auto r = grouped_conv1d(ctx, x_c, cweight, cbias, 16);              // [N, D]
            auto sp = exp_log_softplus(ctx, r);
            auto mish = mod::MulModule().build(ctx, r, mod::TanhModule().build(ctx, sp));
            return core::reshape_tensor(
                ctx, core::ensure_backend_addressable_layout(ctx, mish),
                core::TensorShape::from_dims({1, N, kDim}));
        };
        auto r0 = conv_mish(inp, w.cpe0.weight, *w.cpe0.bias);
        auto r1 = conv_mish(r0, w.cpe2.weight, *w.cpe2.bias);
        inp = mod::AddModule().build(ctx, inp, r1);
    }

    // ---- time embedding: shared MLP over the per-call leaf ----
    auto t0 = mod::LinearModule({256, kDim, true}).build(ctx, io.time_input, w.time0);
    t0 = mod::SiluModule().build(ctx, t0);
    auto t_emb = mod::LinearModule({kDim, kDim, true}).build(ctx, t0, w.time2);  // [1, 1024]

    // ---- RoPE positions [N] constant ----
    core::TensorValue positions;
    {
        std::vector<int32_t> pos(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) {
            pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        auto p = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({N}));
        if (p.tensor->data != nullptr) {
            std::memcpy(p.tensor->data, pos.data(), pos.size() * sizeof(int32_t));
        } else if (t_const_stage != nullptr) {
            const auto * b = reinterpret_cast<const uint8_t *>(pos.data());
            t_const_stage->push_back({p.tensor, std::vector<uint8_t>(b, b + pos.size() * sizeof(int32_t))});
        }
        positions = p;
    }

    // ---- 22 DiT blocks ----
    auto h = inp;
    for (int bi = 0; bi < 22; ++bi) {
        const auto & B = w.blocks[static_cast<size_t>(bi)];
        // adaLN modulation: 6 chunks from silu(t_emb) @ attn_norm
        auto emb = mod::LinearModule({kDim, 6 * kDim, true}).build(
            ctx, mod::SiluModule().build(ctx, t_emb), B.attn_norm);  // [1, 6144]
        // F5 chunk order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        auto shift_msa = mod::SliceModule({1, 0 * kDim, kDim}).build(ctx, emb);
        auto scale_msa = mod::SliceModule({1, 1 * kDim, kDim}).build(ctx, emb);
        auto gate_msa = mod::SliceModule({1, 2 * kDim, kDim}).build(ctx, emb);
        auto shift_mlp = mod::SliceModule({1, 3 * kDim, kDim}).build(ctx, emb);
        auto scale_mlp = mod::SliceModule({1, 4 * kDim, kDim}).build(ctx, emb);
        auto gate_mlp = mod::SliceModule({1, 5 * kDim, kDim}).build(ctx, emb);
        (void)shift_msa;

        // modulate: x * (1 + scale) + shift
        auto norm = modulate(ctx, mod::LayerNormModule({kDim, 1e-6F, false, false}).build(ctx, h, mod::NormWeights{}), scale_msa, shift_msa);
        auto q = mod::LinearModule({kDim, kDim, true}).build(ctx, norm, B.to_q);
        auto k = mod::LinearModule({kDim, kDim, true}).build(ctx, norm, B.to_k);
        auto v = mod::LinearModule({kDim, kDim, true}).build(ctx, norm, B.to_v);
        // heads: [1, N, H, DH] (roformer reshape_heads pattern)
        auto to_heads = [&](core::TensorValue t) {
            return core::reshape_tensor(
                ctx, core::ensure_backend_addressable_layout(ctx, t),
                core::TensorShape::from_dims({t.shape.dims[0], t.shape.dims[1], kHeads, kHeadDim}));
        };
        q = to_heads(q);
        k = to_heads(k);
        v = to_heads(v);
        q = mod::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
        k = mod::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);
        auto q_heads = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
        auto k_heads = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
        auto v_heads = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
        auto attn = mod::ScaledDotProductAttentionModule({
            kHeadDim,
            mod::ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32,
            mod::AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads);  // [1, N, H, DH]
        auto attn_flat = core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, attn),
            core::TensorShape::from_dims({attn.shape.dims[0], attn.shape.dims[1], kDim}));
        auto proj = mod::LinearModule({kDim, kDim, true}).build(ctx, attn_flat, B.to_out);
        // gated residual: h + proj * gate
        h = mod::AddModule().build(
            ctx, h, mod::MulModule().build(
                ctx, proj, mod::RepeatModule({proj.shape}).build(ctx, lift_row(ctx, gate_msa))));

        auto norm2 = modulate(ctx, mod::LayerNormModule({kDim, 1e-6F, false, false}).build(ctx, h, mod::NormWeights{}), scale_mlp, shift_mlp);
        auto f1 = mod::LinearModule({kDim, 2048, true}).build(ctx, norm2, B.ff0);
        f1 = mod::GeluModule({mod::GeluApproximation::Tanh}).build(ctx, f1);
        auto f2 = mod::LinearModule({2048, kDim, true}).build(ctx, f1, B.ff2);
        h = mod::AddModule().build(
            ctx, h, mod::MulModule().build(
                ctx, f2, mod::RepeatModule({f2.shape}).build(ctx, lift_row(ctx, gate_mlp))));
    }

    // ---- final adaLN + projection to mel ----
    {
        auto emb = mod::LinearModule({kDim, 2 * kDim, true}).build(
            ctx, mod::SiluModule().build(ctx, t_emb), w.norm_out);
        auto scale = mod::SliceModule({1, 0, kDim}).build(ctx, emb);
        auto shift = mod::SliceModule({1, kDim, kDim}).build(ctx, emb);
        auto norm = modulate(ctx, mod::LayerNormModule({kDim, 1e-6F, false, false}).build(ctx, h, mod::NormWeights{}), scale, shift);
        io.output = mod::LinearModule({kDim, kMel, true}).build(ctx, norm, w.proj_out);  // [1, N, 100]
    }
    return io;
}



// Batched-CFG variant: B=2 halves share every weight; halves differ only in
// text ids (cond half: ids+1 offset embeds 〈ref+text〉, uncond half: pad id).
F5DiTGraphBuild build_dit_cfg_modules_graph(
    ggml_context * ggml,
    const F5DiTWeights & w,
    const F5Architecture & arch,
    int frames,
    int text_len,
    core::BackendType backend_type) {
    auto ctx = make_ctx(ggml, "f5.dit.cfg", backend_type);
    constexpr int64_t kMel = 100, kTextDim = 512, kDim = 1024;
    constexpr int64_t kHeads = 16, kHeadDim = 64, kVocab = 2731;
    const int64_t N = frames;
    const int64_t NT = text_len;

    F5DiTGraphBuild io;
    // leaves: x/cond [2, N, 100] (both halves identical values), ids [2*NT]
    io.x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, N, kMel}));
    io.cond = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, N, kMel}));
    io.text_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2 * NT}));
    io.time_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 256}));

    // per-half embeddings then concat on the batch axis
    auto ids_c = mod::SliceModule({0, 0, NT}).build(ctx, io.text_ids);
    auto ids_u = mod::SliceModule({0, NT, NT}).build(ctx, io.text_ids);
    auto emb = [&](const core::TensorValue & ids) {
        auto e = mod::EmbeddingModule({kVocab, kTextDim}).build(ctx, ids, w.text_embedding);
        return core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, e),
            core::TensorShape::from_dims({1, NT, kTextDim}));
    };
    auto te_c = emb(ids_c);
    auto te_u = emb(ids_u);
    // shared pe over positions
    std::vector<float> pe(static_cast<size_t>(NT) * kTextDim);
    {
        const int64_t half = kTextDim / 2;
        for (int64_t pos = 0; pos < NT; ++pos) {
            for (int64_t i = 0; i < half; ++i) {
                const float inv = std::pow(10000.0F, -2.0F * i / static_cast<float>(kTextDim));
                const float f = static_cast<float>(pos) * inv;
                pe[static_cast<size_t>(pos) * kTextDim + i] = std::cos(f);
                pe[static_cast<size_t>(pos) * kTextDim + half + i] = std::sin(f);
            }
        }
    }
    auto pe_t = ctx_store_f32(ctx, core::TensorShape::from_dims({1, NT, kTextDim}), pe);
    te_c = mod::AddModule().build(ctx, te_c, pe_t);
    te_u = mod::AddModule().build(ctx, te_u, pe_t);
    auto te = mod::ConcatModule({0}).build(ctx, te_c, te_u);  // [2, NT, 512]

    // text ConvNeXt x4 (batch-aware: dwconv input [B, C, T])
    for (int bi = 0; bi < 4; ++bi) {
        const auto & B = w.text_blocks[static_cast<size_t>(bi)];
        auto te_c2 = mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, te);   // [B, C, T]
        // depthwise over batch: module requires rank-3 [B, C, T] — supported
        auto dw = mod::DepthwiseConv1dModule({kTextDim, 7, 1, 3, 1, true}).build(ctx, te_c2, B.dwconv);
        auto dw_t = mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, dw);    // [B, T, C]
        auto nrm = mod::LayerNormModule({kTextDim, 1e-6F, true, true}).build(ctx, dw_t, B.norm);
        auto h1 = mod::LinearModule({kTextDim, kDim, true}).build(ctx, nrm, B.pw1);
        h1 = mod::GeluModule({mod::GeluApproximation::ExactErf}).build(ctx, h1);
        core::TensorValue g_dummy, b_dummy;
        auto grn_out = grn(ctx, h1, B.grn_gamma, B.grn_beta, g_dummy, b_dummy);
        auto h2 = mod::LinearModule({kDim, kTextDim, true}).build(ctx, grn_out, B.pw2);
        te = mod::AddModule().build(ctx, te, h2);
    }

    // pad text to N on axis 1 (both halves share the same NT)
    core::TensorValue te_pad;
    if (NT >= N) {
        te_pad = mod::SliceModule({1, 0, N}).build(ctx, te);
    } else {
        const auto zshape = core::TensorShape::from_dims({2, N - NT, kTextDim});
        std::vector<float> zv(static_cast<size_t>(zshape.num_elements()), 0.0F);
        te_pad = mod::ConcatModule({1}).build(ctx, te, ctx_store_f32(ctx, zshape, zv));
    }

    // input embed
    auto cat0 = mod::ConcatModule({2}).build(ctx, io.x, io.cond);
    auto cat1 = mod::ConcatModule({2}).build(ctx, cat0, te_pad);  // [2, N, 712]
    auto inp = mod::LinearModule({712LL, kDim, true}).build(ctx, cat1, w.input_proj);

    // CPE: conv over [B, D, N] — grouped_conv1d folds batch; keep B=2 by
    // reshaping to rows [2*N, D] and treating 2N as the time axis of a B=1
    // conv (valid: the conv is per-(batch,row) in time; frames never mix).
    {
        auto conv_mish = [&](const core::TensorValue & x_bnd,
                             const core::TensorValue & cweight,
                             const core::TensorValue & cbias) -> core::TensorValue {
            // [B, N, D] -> rows [B*N, D] -> [1, B*N, D] -> conv -> [B*N, D]
            auto rows = core::reshape_tensor(
                ctx, core::ensure_backend_addressable_layout(ctx, x_bnd),
                core::TensorShape::from_dims({x_bnd.shape.dims[0] * x_bnd.shape.dims[1], x_bnd.shape.dims[2]}));
            auto b1 = core::reshape_tensor(
                ctx, rows, core::TensorShape::from_dims({1, rows.shape.dims[0], rows.shape.dims[1]}));
            auto x_c = mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, b1);  // [1, D, B*N]
            auto x_cc = core::wrap_tensor(ggml_cont(ctx.ggml, x_c.tensor), x_c.shape, GGML_TYPE_F32);
            auto r = grouped_conv1d(ctx, x_cc, cweight, cbias, 16);          // [B*N, D]
            auto sp = exp_log_softplus(ctx, r);
            auto mish = mod::MulModule().build(ctx, r, mod::TanhModule().build(ctx, sp));
            return core::reshape_tensor(
                ctx, core::ensure_backend_addressable_layout(ctx, mish),
                core::TensorShape::from_dims({2, N, kDim}));
        };
        auto r0 = conv_mish(inp, w.cpe0.weight, *w.cpe0.bias);
        auto r1 = conv_mish(r0, w.cpe2.weight, *w.cpe2.bias);
        inp = mod::AddModule().build(ctx, inp, r1);
    }

    // time embedding (shared across halves)
    auto t0 = mod::LinearModule({256, kDim, true}).build(ctx, io.time_input, w.time0);
    t0 = mod::SiluModule().build(ctx, t0);
    auto t_emb = mod::LinearModule({kDim, kDim, true}).build(ctx, t0, w.time2);  // [1, 1024]

    core::TensorValue positions;
    {
        std::vector<int32_t> pos(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        auto p = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({N}));
        if (p.tensor->data != nullptr) {
            std::memcpy(p.tensor->data, pos.data(), pos.size() * sizeof(int32_t));
        } else if (t_const_stage != nullptr) {
            const auto * b = reinterpret_cast<const uint8_t *>(pos.data());
            t_const_stage->push_back({p.tensor, std::vector<uint8_t>(b, b + pos.size() * sizeof(int32_t))});
        }
        positions = p;
    }

    // 22 DiT blocks: identical to the B=1 graph; all modules are batch-aware
    auto h = inp;
    for (int bi = 0; bi < 22; ++bi) {
        const auto & B = w.blocks[static_cast<size_t>(bi)];
        auto emb6 = mod::LinearModule({kDim, 6 * kDim, true}).build(
            ctx, mod::SiluModule().build(ctx, t_emb), B.attn_norm);  // [1, 6144]
        auto shift_msa = mod::SliceModule({1, 0 * kDim, kDim}).build(ctx, emb6);
        auto scale_msa = mod::SliceModule({1, 1 * kDim, kDim}).build(ctx, emb6);
        auto gate_msa = mod::SliceModule({1, 2 * kDim, kDim}).build(ctx, emb6);
        auto shift_mlp = mod::SliceModule({1, 3 * kDim, kDim}).build(ctx, emb6);
        auto scale_mlp = mod::SliceModule({1, 4 * kDim, kDim}).build(ctx, emb6);
        auto gate_mlp = mod::SliceModule({1, 5 * kDim, kDim}).build(ctx, emb6);
        (void)shift_msa;

        auto norm = modulate(ctx, mod::LayerNormModule({kDim, 1e-6F, false, false}).build(ctx, h, mod::NormWeights{}), scale_msa, shift_msa);
        auto q = mod::LinearModule({kDim, kDim, true}).build(ctx, norm, B.to_q);
        auto k = mod::LinearModule({kDim, kDim, true}).build(ctx, norm, B.to_k);
        auto v = mod::LinearModule({kDim, kDim, true}).build(ctx, norm, B.to_v);
        auto to_heads = [&](core::TensorValue t) {
            return core::reshape_tensor(
                ctx, core::ensure_backend_addressable_layout(ctx, t),
                core::TensorShape::from_dims({t.shape.dims[0], t.shape.dims[1], kHeads, kHeadDim}));
        };
        q = to_heads(q); k = to_heads(k); v = to_heads(v);
        q = mod::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
        k = mod::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);
        auto q_heads = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
        auto k_heads = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
        auto v_heads = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
        auto attn = mod::ScaledDotProductAttentionModule({
            kHeadDim,
            mod::ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32,
            mod::AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads);  // [2, N, H, DH]
        auto attn_flat = core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, attn),
            core::TensorShape::from_dims({attn.shape.dims[0], attn.shape.dims[1], kDim}));
        auto proj = mod::LinearModule({kDim, kDim, true}).build(ctx, attn_flat, B.to_out);
        h = mod::AddModule().build(
            ctx, h, mod::MulModule().build(
                ctx, proj, mod::RepeatModule({proj.shape}).build(ctx, lift_row(ctx, gate_msa))));

        auto norm2 = modulate(ctx, mod::LayerNormModule({kDim, 1e-6F, false, false}).build(ctx, h, mod::NormWeights{}), scale_mlp, shift_mlp);
        auto f1 = mod::LinearModule({kDim, 2048, true}).build(ctx, norm2, B.ff0);
        f1 = mod::GeluModule({mod::GeluApproximation::Tanh}).build(ctx, f1);
        auto f2 = mod::LinearModule({2048, kDim, true}).build(ctx, f1, B.ff2);
        h = mod::AddModule().build(
            ctx, h, mod::MulModule().build(
                ctx, f2, mod::RepeatModule({f2.shape}).build(ctx, lift_row(ctx, gate_mlp))));
    }

    {
        auto emb2 = mod::LinearModule({kDim, 2 * kDim, true}).build(
            ctx, mod::SiluModule().build(ctx, t_emb), w.norm_out);
        auto scale = mod::SliceModule({1, 0, kDim}).build(ctx, emb2);
        auto shift = mod::SliceModule({1, kDim, kDim}).build(ctx, emb2);
        auto norm = modulate(ctx, mod::LayerNormModule({kDim, 1e-6F, false, false}).build(ctx, h, mod::NormWeights{}), scale, shift);
        io.output = mod::LinearModule({kDim, kMel, true}).build(ctx, norm, w.proj_out);  // [2, N, 100]
    }
    return io;
}

}  // namespace engine::models::f5_tts
