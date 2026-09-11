#include "engine/community_models/echo_tts/dit.h"

#include "engine/community_models/echo_tts/sampler.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::echo_tts {
namespace {

constexpr size_t kWeightContextBytes = 6144ull * 1024ull * 1024ull;
constexpr size_t kGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr float kRopeTheta = 10000.0F;

// cond, text-uncond, speaker-uncond. The KV cache is sized to this so the
// three-lane graph never has to broadcast it at sampling time.
constexpr int64_t kMaxCfgLanes = 3;

// Additive mask value for disallowed keys. -INFINITY would be exact but
// produces NaN when an entire row is masked, which happens for the
// speaker-uncond CFG lane; a large finite penalty is the standard ggml
// workaround and leaves the softmax well defined.
constexpr float kMaskedBias = -1.0e9F;

// The denoiser mask is F16 for the flash-attention path, whose maximum
// magnitude is 65504, so -1e9 would become -inf on conversion. A large finite
// penalty keeps the softmax well defined even if a row were ever fully masked.
constexpr float kMaskedBiasF16 = -65000.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};
using GgmlContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

// Opt-out for the flash-attention lowering in the DiT's joint attention, so the
// explicit path can be compared against it without a rebuild.
bool echo_flash_disabled() {
    static const bool disabled = [] {
        const char * value = std::getenv("AUDIOCPP_ECHO_TTS_NO_FLASH");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}

// Pre-broadcasting the conditioning KV cache across the CFG lanes moves a
// per-layer, per-step RepeatModule out of the denoiser graph and into the
// once-per-request conditioning graph. At 24 layers x 40 steps the repeat it
// removes is tens of gigabytes of pure copy traffic on a long reference.
// Set AUDIOCPP_ECHO_TTS_NO_KV_EXPAND=1 to fall back to the batch-1 cache.
bool echo_kv_expand_disabled() {
    static const bool disabled = [] {
        const char * value = std::getenv("AUDIOCPP_ECHO_TTS_NO_KV_EXPAND");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    assets::TensorStorageType storage,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, name + ".weight", storage, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, name + ".bias", {out_features});
    }
    return weights;
}

EchoRmsNormWeights load_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    std::initializer_list<int64_t> shape) {
    return EchoRmsNormWeights{store.load_f32_tensor(source, name + ".weight", shape)};
}

EchoMlpWeights load_mlp(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t dim,
    int64_t inter) {
    EchoMlpWeights weights;
    weights.w1 = load_linear(store, source, prefix + ".w1", storage, inter, dim, false);
    weights.w3 = load_linear(store, source, prefix + ".w3", storage, inter, dim, false);
    weights.w2 = load_linear(store, source, prefix + ".w2", storage, dim, inter, false);
    return weights;
}

EchoEncoderBlockWeights load_encoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t dim,
    int64_t inter,
    int64_t heads) {
    const int64_t head_dim = dim / heads;
    EchoEncoderBlockWeights block;
    auto & attn = block.attention;
    attn.wq = load_linear(store, source, prefix + ".attention.wq", storage, dim, dim, false);
    attn.wk = load_linear(store, source, prefix + ".attention.wk", storage, dim, dim, false);
    attn.wv = load_linear(store, source, prefix + ".attention.wv", storage, dim, dim, false);
    attn.wo = load_linear(store, source, prefix + ".attention.wo", storage, dim, dim, false);
    attn.gate = load_linear(store, source, prefix + ".attention.gate", storage, dim, dim, false);
    attn.q_norm = load_norm(store, source, prefix + ".attention.q_norm", {heads, head_dim});
    attn.k_norm = load_norm(store, source, prefix + ".attention.k_norm", {heads, head_dim});
    block.mlp = load_mlp(store, source, prefix + ".mlp", storage, dim, inter);
    block.attention_norm = load_norm(store, source, prefix + ".attention_norm", {dim});
    block.mlp_norm = load_norm(store, source, prefix + ".mlp_norm", {dim});
    return block;
}

EchoAdaLnWeights load_adaln(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t dim,
    int64_t rank) {
    EchoAdaLnWeights weights;
    weights.shift_down = load_linear(store, source, prefix + ".shift_down", storage, rank, dim, false);
    weights.scale_down = load_linear(store, source, prefix + ".scale_down", storage, rank, dim, false);
    weights.gate_down = load_linear(store, source, prefix + ".gate_down", storage, rank, dim, false);
    weights.shift_up = load_linear(store, source, prefix + ".shift_up", storage, dim, rank, true);
    weights.scale_up = load_linear(store, source, prefix + ".scale_up", storage, dim, rank, true);
    weights.gate_up = load_linear(store, source, prefix + ".gate_up", storage, dim, rank, true);
    return weights;
}

EchoDitWeights load_dit_weights(
    const EchoTtsConfig & config,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & p,
    assets::TensorStorageType storage) {
    EchoDitWeights w;
    const int64_t D = config.model_size;
    const int64_t TD = config.text_model_size;
    const int64_t SD = config.speaker_model_size;

    w.text_encoder.text_embedding = store.load_tensor(
        source, p + "text_encoder.text_embedding.weight", storage,
        {config.text_vocab_size, TD});
    for (int64_t i = 0; i < config.text_num_layers; ++i) {
        w.text_encoder.blocks.push_back(load_encoder_block(
            store, source, p + "text_encoder.blocks." + std::to_string(i), storage,
            TD, config.text_intermediate_size, config.text_num_heads));
    }

    w.speaker_encoder.in_proj = load_linear(
        store, source, p + "speaker_encoder.in_proj", storage,
        SD, config.latent_size * config.speaker_patch_size, true);
    for (int64_t i = 0; i < config.speaker_num_layers; ++i) {
        w.speaker_encoder.blocks.push_back(load_encoder_block(
            store, source, p + "speaker_encoder.blocks." + std::to_string(i), storage,
            SD, config.speaker_intermediate_size, config.speaker_num_heads));
    }

    w.text_norm = load_norm(store, source, p + "text_norm", {TD});
    w.speaker_norm = load_norm(store, source, p + "speaker_norm", {SD});

    w.cond_0 = load_linear(store, source, p + "cond_module.0", storage, D, config.timestep_embed_size, false);
    w.cond_2 = load_linear(store, source, p + "cond_module.2", storage, D, D, false);
    w.cond_4 = load_linear(store, source, p + "cond_module.4", storage, D * 3, D, false);

    w.in_proj = load_linear(store, source, p + "in_proj", storage, D, config.latent_size, true);

    const int64_t head_dim = config.head_dim();
    for (int64_t i = 0; i < config.num_layers; ++i) {
        const std::string prefix = p + "blocks." + std::to_string(i);
        EchoDitBlockWeights block;
        auto & a = block.attention;
        a.wq = load_linear(store, source, prefix + ".attention.wq", storage, D, D, false);
        a.wk = load_linear(store, source, prefix + ".attention.wk", storage, D, D, false);
        a.wv = load_linear(store, source, prefix + ".attention.wv", storage, D, D, false);
        a.wk_text = load_linear(store, source, prefix + ".attention.wk_text", storage, D, TD, false);
        a.wv_text = load_linear(store, source, prefix + ".attention.wv_text", storage, D, TD, false);
        a.wk_speaker = load_linear(store, source, prefix + ".attention.wk_speaker", storage, D, SD, false);
        a.wv_speaker = load_linear(store, source, prefix + ".attention.wv_speaker", storage, D, SD, false);
        a.gate = load_linear(store, source, prefix + ".attention.gate", storage, D, D, false);
        a.wo = load_linear(store, source, prefix + ".attention.wo", storage, D, D, false);
        a.q_norm = load_norm(store, source, prefix + ".attention.q_norm", {config.num_heads, head_dim});
        a.k_norm = load_norm(store, source, prefix + ".attention.k_norm", {config.num_heads, head_dim});
        block.mlp = load_mlp(store, source, prefix + ".mlp", storage, D, config.intermediate_size);
        block.attention_adaln = load_adaln(store, source, prefix + ".attention_adaln", storage, D, config.adaln_rank);
        block.mlp_adaln = load_adaln(store, source, prefix + ".mlp_adaln", storage, D, config.adaln_rank);
        w.blocks.push_back(std::move(block));
    }

    w.out_norm = load_norm(store, source, p + "out_norm", {D});
    w.out_proj = load_linear(store, source, p + "out_proj", storage, config.latent_size, D, true);
    return w;
}

// --- graph building blocks ---------------------------------------------

core::TensorValue rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    float eps) {
    return modules::RMSNormModule({input.shape.last_dim(), eps, true, false})
        .build(ctx, input, {weight, std::nullopt});
}

// Elementwise multiply where the right operand is broadcast over one or more
// leading axes. MulModule requires identical shapes, but adaLN produces a
// per-sequence-position-invariant gate of shape (lanes, 1, dim) that has to
// scale a (lanes, seq, dim) activation. ggml_mul itself broadcasts whenever
// every rhs dimension divides the lhs, which is exactly this case.
core::TensorValue broadcast_mul(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous(ctx, lhs).tensor, contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

// Per-head RMS normalisation for q_norm / k_norm.
//
// Echo's q_norm and k_norm carry a (num_heads, head_dim) weight: each head has
// its own learned scale. RMSNormModule takes a 1-D weight sized to the last
// dimension, so it cannot express this -- the in-tree rf_dit.cpp uses that
// module because its q_norm really is 1-D {head_dim}, which is a different
// architecture, not a different spelling of the same one.
//
// The reduction is over head_dim only. In ggml layout the activation is
// (head_dim, heads, seq, batch) and the weight is (head_dim, heads, 1, 1), so a
// plain multiply broadcasts the per-head scales across sequence and batch.
core::TensorValue head_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    float eps) {
    auto * normed = ggml_rms_norm(ctx.ggml, contiguous(ctx, input).tensor, eps);
    return core::wrap_tensor(
        ggml_mul(ctx.ggml, normed, contiguous(ctx, weight).tensor),
        input.shape,
        GGML_TYPE_F32);
}

// RMS normalisation with no learned scale, used inside LowRankAdaLN where the
// scale arrives from the conditioning path instead.
core::TensorValue rms_norm_bare(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float eps) {
    return core::wrap_tensor(
        ggml_rms_norm(ctx.ggml, contiguous(ctx, input).tensor, eps),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        contiguous(ctx, input),
        core::TensorShape::from_dims(
            {input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue to_bhsd(core::ModuleBuildContext & ctx, const core::TensorValue & bshd) {
    return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, bshd);
}

core::TensorValue mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EchoMlpWeights & weights,
    int64_t dim,
    int64_t inter) {
    auto gate = modules::LinearModule({dim, inter, false, GGML_PREC_F32})
                    .build(ctx, input, weights.w1);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({dim, inter, false, GGML_PREC_F32})
                  .build(ctx, input, weights.w3);
    auto hidden = modules::MulModule{}.build(ctx, gate, up);
    return modules::LinearModule({inter, dim, false, GGML_PREC_F32})
        .build(ctx, hidden, weights.w2);
}

// Applies output * sigmoid(gate) before the output projection, as every
// attention block in this model does.
core::TensorValue apply_attention_gate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & context,
    const core::TensorValue & input,
    const modules::LinearWeights & gate_weights,
    int64_t dim) {
    auto gate = modules::LinearModule({dim, dim, false, GGML_PREC_F32})
                    .build(ctx, input, gate_weights);
    gate = modules::SigmoidModule{}.build(ctx, gate);
    return modules::MulModule{}.build(ctx, context, gate);
}

// SelfAttention shared by the text and speaker encoders. RoPE covers all heads
// here, unlike the DiT's joint attention.
core::TensorValue encoder_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const std::optional<core::TensorValue> & mask,
    const EchoSelfAttentionWeights & weights,
    int64_t dim,
    int64_t heads,
    float eps,
    bool causal) {
    const int64_t head_dim = dim / heads;
    auto q = modules::LinearModule({dim, dim, false, GGML_PREC_F32}).build(ctx, input, weights.wq);
    auto k = modules::LinearModule({dim, dim, false, GGML_PREC_F32}).build(ctx, input, weights.wk);
    auto v = modules::LinearModule({dim, dim, false, GGML_PREC_F32}).build(ctx, input, weights.wv);

    q = head_rms_norm(ctx, reshape_heads(ctx, q, heads, head_dim), weights.q_norm.weight, eps);
    k = head_rms_norm(ctx, reshape_heads(ctx, k, heads, head_dim), weights.k_norm.weight, eps);
    v = reshape_heads(ctx, v, heads, head_dim);

    // Upstream builds freqs_cis by viewing adjacent pairs as complex, which is
    // the interleaved convention -- GGML_ROPE_TYPE_NORMAL, not NEOX.
    const modules::RoPEModule rope({head_dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta});
    q = rope.build(ctx, q, positions);
    k = rope.build(ctx, k, positions);

    modules::ScaledDotProductAttentionConfig attn_config;
    attn_config.head_dim = head_dim;
    attn_config.lowering = modules::ScaledDotProductAttentionLowering::Explicit;
    attn_config.precision = GGML_PREC_F32;
    attn_config.causality = causal ? modules::AttentionCausality::Causal
                                   : modules::AttentionCausality::NonCausal;
    auto context = modules::ScaledDotProductAttentionModule(attn_config)
                       .build(ctx, to_bhsd(ctx, q), to_bhsd(ctx, k), to_bhsd(ctx, v), mask);

    context = core::reshape_tensor(
        ctx,
        contiguous(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], dim}));
    context = apply_attention_gate(ctx, context, input, weights.gate, dim);
    return modules::LinearModule({dim, dim, false, GGML_PREC_F32}).build(ctx, context, weights.wo);
}

core::TensorValue encoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const std::optional<core::TensorValue> & mask,
    const EchoEncoderBlockWeights & weights,
    int64_t dim,
    int64_t inter,
    int64_t heads,
    float eps,
    bool causal) {
    auto normed = rms_norm(ctx, input, weights.attention_norm.weight, eps);
    auto attn = encoder_self_attention(ctx, normed, positions, mask, weights.attention, dim, heads, eps, causal);
    auto hidden = modules::AddModule{}.build(ctx, input, attn);

    auto mlp_normed = rms_norm(ctx, hidden, weights.mlp_norm.weight, eps);
    auto mlp_out = mlp(ctx, mlp_normed, weights.mlp, dim, inter);
    return modules::AddModule{}.build(ctx, hidden, mlp_out);
}

// LowRankAdaLN. Returns the normalised activation and the tanh-bounded gate.
struct AdaLnResult {
    core::TensorValue normed;
    core::TensorValue gate;
};

AdaLnResult adaln(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & cond_embed,
    const EchoAdaLnWeights & weights,
    int64_t dim,
    int64_t rank,
    float eps) {
    // cond_embed is (batch, 1, dim * 3); chunk into shift, scale, gate.
    auto shift = modules::SliceModule({2, 0, dim}).build(ctx, cond_embed);
    auto scale = modules::SliceModule({2, dim, dim}).build(ctx, cond_embed);
    auto gate = modules::SliceModule({2, 2 * dim, dim}).build(ctx, cond_embed);

    auto refine = [&](const core::TensorValue & value,
                      const modules::LinearWeights & down,
                      const modules::LinearWeights & up) {
        auto hidden = modules::SiluModule{}.build(ctx, value);
        hidden = modules::LinearModule({dim, rank, false, GGML_PREC_F32}).build(ctx, hidden, down);
        hidden = modules::LinearModule({rank, dim, true, GGML_PREC_F32}).build(ctx, hidden, up);
        return modules::AddModule{}.build(ctx, hidden, value);
    };

    shift = refine(shift, weights.shift_down, weights.shift_up);
    scale = refine(scale, weights.scale_down, weights.scale_up);
    gate = refine(gate, weights.gate_down, weights.gate_up);

    auto normed = rms_norm_bare(ctx, x, eps);
    // x * (scale + 1) + shift, with the conditioning broadcast over sequence.
    auto scale_plus_one = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, contiguous(ctx, scale).tensor, 1.0F, 1.0F),
        scale.shape,
        GGML_TYPE_F32);
    normed = core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous(ctx, normed).tensor, contiguous(ctx, scale_plus_one).tensor),
        normed.shape,
        GGML_TYPE_F32);
    normed = core::wrap_tensor(
        ggml_add(ctx.ggml, normed.tensor, contiguous(ctx, shift).tensor),
        normed.shape,
        GGML_TYPE_F32);

    gate = modules::TanhModule{}.build(ctx, gate);
    return AdaLnResult{normed, gate};
}

}  // namespace
}  // namespace engine::models::echo_tts
