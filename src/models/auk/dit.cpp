#include "engine/models/auk/dit.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace engine::models::auk {
namespace {

namespace mod = engine::modules;

constexpr size_t kGraphContextBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kGraphNodeCapacity = 262144;

struct DoubleBlockWeights {
    mod::LinearWeights norm_c;      // adaLN: dim -> 6 * dim
    mod::LinearWeights norm_x;
    mod::LinearWeights qkv_x;       // dim -> 3 * dim
    mod::LinearWeights qkv_c;
    core::TensorValue q_norm, k_norm, c_q_norm, c_k_norm;   // RMSNorm over head_dim
    mod::LinearWeights out_x;
    mod::LinearWeights out_c;
    mod::LinearWeights ff_x_in, ff_x_out, ff_c_in, ff_c_out;
};

struct SingleBlockWeights {
    mod::LinearWeights norm;        // adaLN: dim -> 6 * dim
    mod::LinearWeights qkv;
    core::TensorValue q_norm, k_norm;
    mod::LinearWeights out;
    mod::LinearWeights ff_in, ff_out;
};

struct DitWeights {
    mod::LinearWeights time_mlp_in;
    mod::LinearWeights time_mlp_out;
    mod::LinearWeights txt_proj;
    core::TensorValue txt_norm;
    mod::LinearWeights audio_linear;
    mod::LinearWeights conv_pos_0;   // weight [dim, dim/groups, k], bias [dim]
    mod::LinearWeights conv_pos_1;
    std::vector<DoubleBlockWeights> double_blocks;
    std::vector<SingleBlockWeights> single_blocks;
    mod::LinearWeights norm_out;    // dim -> 2 * dim (scale, shift)
    mod::LinearWeights proj_out;    // dim -> latent_dim
    core::TensorValue inv_freq;     // [head_dim / 2], stored rather than recomputed
    int64_t loaded = 0;
};

// mish(x) = x * tanh(softplus(x)). ggml has softplus and tanh but no mish; sopro_tts
// writes it out the same way (acoustic.cpp:138).
core::TensorValue mish(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    auto * inner = ggml_tanh(ctx.ggml, ggml_softplus(ctx.ggml, x.tensor));
    return core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, inner), x.shape, GGML_TYPE_F32);
}

// The framework has no grouped Conv1d module; lowered the same way
// f5_tts/dit_modules.cpp:50 does it, since this is the same ConvPositionEmbedding.
core::TensorValue grouped_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,    // [frames, channels]
    const core::TensorValue & weight,   // [c_out, c_in/groups, kernel]
    const core::TensorValue & bias,     // [c_out]
    int64_t groups) {
    const int64_t frames = input.shape.dims[0];
    const int64_t c_out = weight.shape.dims[0];
    const int64_t cg_in = weight.shape.dims[1];
    const int64_t kernel = weight.shape.dims[2];
    const int64_t cg_out = c_out / groups;

    std::vector<core::TensorValue> outputs;
    outputs.reserve(static_cast<size_t>(groups));
    for (int64_t group = 0; group < groups; ++group) {
        auto slice = mod::SliceModule({1, group * cg_in, cg_in}).build(ctx, input);
        auto contiguous = core::ensure_backend_addressable_layout(ctx, slice);
        auto * channel_major = ggml_cont(
            ctx.ggml, ggml_transpose(ctx.ggml, contiguous.tensor));
        auto * input3 = ggml_reshape_3d(ctx.ggml, channel_major, frames, cg_in, 1);
        auto * weight_group = ggml_view_3d(
            ctx.ggml, weight.tensor, kernel, cg_in, cg_out,
            weight.tensor->nb[1], weight.tensor->nb[2],
            group * cg_out * weight.tensor->nb[2]);
        auto * cols = ggml_im2col(
            ctx.ggml, weight_group, input3, 1, 1, kernel / 2, 0, 1, 1, false, GGML_TYPE_F32);
        auto * flat = ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, weight_group), cg_in * kernel, cg_out);
        auto * product = ggml_mul_mat(ctx.ggml, flat, cols);
        auto value = core::wrap_tensor(
            product, core::TensorShape::from_dims({frames, cg_out}), GGML_TYPE_F32);
        auto bias_slice = mod::SliceModule({0, group * cg_out, cg_out}).build(ctx, bias);
        auto bias_row = core::reshape_tensor(ctx, bias_slice, core::TensorShape::from_dims({1, cg_out}));
        value = mod::AddModule().build(ctx, value, mod::RepeatModule({value.shape}).build(ctx, bias_row));
        outputs.push_back(value);
    }
    auto out = outputs.front();
    for (size_t index = 1; index < outputs.size(); ++index) {
        out = mod::ConcatModule({1}).build(ctx, out, outputs[index]);
    }
    return out;
}


// ---- block helpers ---------------------------------------------------------------

// A [1, C] row broadcast over frames, which is how every modulation parameter is
// applied: one vector per batch element, the same for every position.
core::TensorValue broadcast_row(
    core::ModuleBuildContext & ctx, const core::TensorValue & row, const core::TensorShape & shape) {
    auto lifted = core::reshape_tensor(ctx, row, core::TensorShape::from_dims({1, row.shape.dims[1]}));
    return mod::RepeatModule({shape}).build(ctx, lifted);
}

// LayerNorm(x) * (1 + scale) + shift, eps 1e-6 and no affine parameters -- the
// modulation is the affine part.
core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & scale,
    const core::TensorValue & shift,
    int64_t dim) {
    auto normed = mod::LayerNormModule({dim, 1e-6F, false, false}).build(ctx, x, mod::NormWeights{});
    auto scaled = mod::MulModule().build(ctx, normed, broadcast_row(ctx, scale, normed.shape));
    return mod::AddModule().build(
        ctx, mod::AddModule().build(ctx, normed, scaled), broadcast_row(ctx, shift, normed.shape));
}

// SwiGLU: linear_in projects to 2 * inner, then silu(first half) * second half.
// Both linears are bias-free (SwiGLUFeedForward, modules.py:135).
core::TensorValue swiglu_ff(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const mod::LinearWeights & linear_in,
    const mod::LinearWeights & linear_out,
    int64_t dim,
    int64_t inner) {
    auto projected = mod::LinearModule({dim, inner * 2, false}).build(ctx, x, linear_in);
    auto gate = mod::SliceModule({1, 0, inner}).build(ctx, projected);
    auto value = mod::SliceModule({1, inner, inner}).build(ctx, projected);
    auto activated = core::wrap_tensor(
        ggml_silu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, gate).tensor),
        gate.shape, GGML_TYPE_F32);
    auto gated = mod::MulModule().build(ctx, activated, value);
    return mod::LinearModule({inner, dim, false}).build(ctx, gated, linear_out);
}

struct AttentionStream {
    core::TensorValue query;
    core::TensorValue key;
    core::TensorValue value;
};

// to_qkv -> per-head split -> RMSNorm on q and k -> RoPE.
//
// ⚠ GGML_ROPE_TYPE_NORMAL, not NEOX. x_transformers' rotate_half pairs ADJACENT
// elements ('... (d r) -> ... d r', r=2) and its frequencies are duplicated
// adjacently (stack((freqs, freqs), -1)), which is the GPT-J convention. NeoX pairs
// i with i + d/2 and would rotate the wrong components -- audio that is plausible
// and wrong, with no error.
AttentionStream project_stream(
    core::ModuleBuildContext & ctx,
    const AukDitConfig & config,
    const core::TensorValue & x,
    const mod::LinearWeights & qkv_weights,
    const core::TensorValue & q_norm,
    const core::TensorValue & k_norm,
    const core::TensorValue & positions) {
    const int64_t frames = x.shape.dims[0];
    auto packed = mod::LinearModule({config.dim, config.dim * 3, true}).build(ctx, x, qkv_weights);
    auto take = [&](int64_t offset) {
        auto part = mod::SliceModule({1, offset * config.dim, config.dim}).build(ctx, packed);
        auto dense = core::ensure_backend_addressable_layout(ctx, part);
        return core::reshape_tensor(
            ctx, dense, core::TensorShape::from_dims({1, frames, config.heads, config.head_dim}));
    };
    AttentionStream stream{take(0), take(1), take(2)};

    // RMSNorm over head_dim, affine, applied per head before RoPE.
    const mod::RMSNormModule head_norm({config.head_dim, config.rms_norm_eps, true, false});
    stream.query = head_norm.build(ctx, stream.query, {q_norm, std::nullopt});
    stream.key = head_norm.build(ctx, stream.key, {k_norm, std::nullopt});

    // ggml_rope_ext on a strided view corrupts arena neighbours (f5_tts:429 records
    // the same bug), so materialize before rotating.
    auto dense = [&](const core::TensorValue & t) {
        return core::wrap_tensor(ggml_cont(ctx.ggml, t.tensor), t.shape, GGML_TYPE_F32);
    };
    stream.query = dense(stream.query);
    stream.key = dense(stream.key);
    const mod::RoPEModule rope({config.head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_theta});
    stream.query = rope.build(ctx, stream.query, positions);
    stream.key = rope.build(ctx, stream.key, positions);
    return stream;
}

core::TensorValue joint_attention(
    core::ModuleBuildContext & ctx,
    const AukDitConfig & config,
    const AttentionStream & audio,
    const AttentionStream & text) {
    // ⚠ [audio, text] here. The single-stream phase concatenates the other way round
    // ([text, audio], flux2_edit.py:330), and swapping either one silently mixes the
    // streams' positions.
    auto join = [&](const core::TensorValue & lhs, const core::TensorValue & rhs) {
        return mod::ConcatModule({1}).build(ctx, lhs, rhs);
    };
    auto query = join(audio.query, text.query);
    auto key = join(audio.key, text.key);
    auto value = join(audio.value, text.value);

    auto heads_first = [&](const core::TensorValue & t) {
        auto transposed = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, t);
        return core::wrap_tensor(ggml_cont(ctx.ggml, transposed.tensor), transposed.shape, GGML_TYPE_F32);
    };
    auto attended = mod::ScaledDotProductAttentionModule({
        config.head_dim,
        mod::ScaledDotProductAttentionLowering::Flash,
        GGML_PREC_F32,
        mod::AttentionCausality::NonCausal,
    }).build(ctx, heads_first(query), heads_first(key), heads_first(value));
    auto dense = core::ensure_backend_addressable_layout(ctx, attended);
    return core::reshape_tensor(
        ctx, dense, core::TensorShape::from_dims({dense.shape.dims[1], config.dim}));
}


// Six modulation parameters from one adaLN linear, in the reference's chunk order:
// shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp (modules.py:89).
struct Modulation {
    core::TensorValue shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp;
};

Modulation build_modulation(
    core::ModuleBuildContext & ctx,
    const AukDitConfig & config,
    const core::TensorValue & time,
    const mod::LinearWeights & weights) {
    auto activated = core::wrap_tensor(ggml_silu(ctx.ggml, time.tensor), time.shape, GGML_TYPE_F32);
    auto packed = mod::LinearModule({config.dim, config.dim * 6, true}).build(ctx, activated, weights);
    auto part = [&](int64_t index) {
        return mod::SliceModule({1, index * config.dim, config.dim}).build(ctx, packed);
    };
    return {part(0), part(1), part(2), part(3), part(4), part(5)};
}

// gated residual: h + gate * value, with the gate broadcast over frames.
core::TensorValue gated_add(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & h,
    const core::TensorValue & value,
    const core::TensorValue & gate) {
    auto scaled = mod::MulModule().build(ctx, value, broadcast_row(ctx, gate, value.shape));
    return mod::AddModule().build(ctx, h, scaled);
}

struct DoubleBlockOutputs {
    core::TensorValue context;
    core::TensorValue audio;
};

DoubleBlockOutputs build_double_block(
    core::ModuleBuildContext & ctx,
    const AukDitConfig & config,
    const DoubleBlockWeights & weights,
    const core::TensorValue & audio,
    const core::TensorValue & context,
    const core::TensorValue & time,
    const core::TensorValue & audio_positions,
    const core::TensorValue & text_positions,
    int64_t inner) {
    const auto mod_c = build_modulation(ctx, config, time, weights.norm_c);
    const auto mod_x = build_modulation(ctx, config, time, weights.norm_x);
    auto norm_c = modulate(ctx, context, mod_c.scale_msa, mod_c.shift_msa, config.dim);
    auto norm_x = modulate(ctx, audio, mod_x.scale_msa, mod_x.shift_msa, config.dim);

    const auto stream_x = project_stream(
        ctx, config, norm_x, weights.qkv_x, weights.q_norm, weights.k_norm, audio_positions);
    const auto stream_c = project_stream(
        ctx, config, norm_c, weights.qkv_c, weights.c_q_norm, weights.c_k_norm, text_positions);
    auto attended = joint_attention(ctx, config, stream_x, stream_c);

    const int64_t audio_frames = audio.shape.dims[0];
    const int64_t text_frames = context.shape.dims[0];
    auto audio_part = mod::SliceModule({0, 0, audio_frames}).build(ctx, attended);
    auto text_part = mod::SliceModule({0, audio_frames, text_frames}).build(ctx, attended);
    audio_part = mod::LinearModule({config.dim, config.dim, true}).build(ctx, audio_part, weights.out_x);
    text_part = mod::LinearModule({config.dim, config.dim, true}).build(ctx, text_part, weights.out_c);

    auto out_c = gated_add(ctx, context, text_part, mod_c.gate_msa);
    auto ff_c = swiglu_ff(
        ctx, modulate(ctx, out_c, mod_c.scale_mlp, mod_c.shift_mlp, config.dim),
        weights.ff_c_in, weights.ff_c_out, config.dim, inner);
    out_c = gated_add(ctx, out_c, ff_c, mod_c.gate_mlp);

    auto out_x = gated_add(ctx, audio, audio_part, mod_x.gate_msa);
    auto ff_x = swiglu_ff(
        ctx, modulate(ctx, out_x, mod_x.scale_mlp, mod_x.shift_mlp, config.dim),
        weights.ff_x_in, weights.ff_x_out, config.dim, inner);
    out_x = gated_add(ctx, out_x, ff_x, mod_x.gate_mlp);

    return {out_c, out_x};
}

core::TensorValue build_single_block(
    core::ModuleBuildContext & ctx,
    const AukDitConfig & config,
    const SingleBlockWeights & weights,
    const core::TensorValue & x,
    const core::TensorValue & time,
    const core::TensorValue & positions,
    int64_t inner) {
    const auto modulation = build_modulation(ctx, config, time, weights.norm);
    auto normed = modulate(ctx, x, modulation.scale_msa, modulation.shift_msa, config.dim);
    const auto stream = project_stream(
        ctx, config, normed, weights.qkv, weights.q_norm, weights.k_norm, positions);

    auto heads_first = [&](const core::TensorValue & t) {
        auto transposed = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, t);
        return core::wrap_tensor(ggml_cont(ctx.ggml, transposed.tensor), transposed.shape, GGML_TYPE_F32);
    };
    auto attended = mod::ScaledDotProductAttentionModule({
        config.head_dim,
        mod::ScaledDotProductAttentionLowering::Flash,
        GGML_PREC_F32,
        mod::AttentionCausality::NonCausal,
    }).build(ctx, heads_first(stream.query), heads_first(stream.key), heads_first(stream.value));
    auto dense = core::ensure_backend_addressable_layout(ctx, attended);
    auto flat = core::reshape_tensor(
        ctx, dense, core::TensorShape::from_dims({dense.shape.dims[1], config.dim}));
    auto projected = mod::LinearModule({config.dim, config.dim, true}).build(ctx, flat, weights.out);

    auto out = gated_add(ctx, x, projected, modulation.gate_msa);
    auto ff = swiglu_ff(
        ctx, modulate(ctx, out, modulation.scale_mlp, modulation.shift_mlp, config.dim),
        weights.ff_in, weights.ff_out, config.dim, inner);
    return gated_add(ctx, out, ff, modulation.gate_mlp);
}

// SinusPositionEmbedding(dim=256) at scale 1000, then Linear -> SiLU -> Linear.
core::TensorValue build_time_embedding(
    core::ModuleBuildContext & ctx,
    const AukDitConfig & config,
    const DitWeights & weights,
    float timestep) {
    const int64_t half = config.freq_embed_dim / 2;
    std::vector<float> values(static_cast<size_t>(config.freq_embed_dim), 0.0F);
    const double step = std::log(10000.0) / static_cast<double>(half - 1);
    for (int64_t index = 0; index < half; ++index) {
        const double frequency =
            static_cast<double>(config.time_scale) * static_cast<double>(timestep) *
            std::exp(static_cast<double>(index) * -step);
        values[static_cast<size_t>(index)] = static_cast<float>(std::sin(frequency));
        values[static_cast<size_t>(half + index)] = static_cast<float>(std::cos(frequency));
    }
    // Host-side because the timestep is a scalar known when the graph is built: the
    // sinusoids are constants, not something to recompute on the backend.
    auto embedded = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.freq_embed_dim}));
    ggml_set_input(embedded.tensor);
    ggml_set_name(embedded.tensor, "auk.dit.time_sinusoid");
    auto hidden = mod::LinearModule({config.freq_embed_dim, config.dim, true})
                      .build(ctx, embedded, weights.time_mlp_in);
    hidden = core::wrap_tensor(ggml_silu(ctx.ggml, hidden.tensor), hidden.shape, GGML_TYPE_F32);
    return mod::LinearModule({config.dim, config.dim, true}).build(ctx, hidden, weights.time_mlp_out);
}

}  // namespace

void AukDitConfig::validate() const {
    if (dim <= 0 || heads <= 0 || head_dim <= 0 || latent_dim <= 0) {
        throw std::runtime_error("AuK DiT config has non-positive dimensions");
    }
    if (heads * head_dim != dim) {
        throw std::runtime_error("AuK DiT heads * head_dim must equal dim");
    }
    if (conv_pos_kernel % 2 == 0) {
        throw std::runtime_error("AuK DiT conv position embedding needs an odd kernel");
    }
}

struct AukDit::Impl {
    AukDitConfig config;
    core::ExecutionContext * execution = nullptr;
    std::unique_ptr<core::BackendWeightStore> store;
    DitWeights weights;

    std::mutex mutex;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    ggml_tensor * sinusoid = nullptr;
    ggml_tensor * text_input = nullptr;
    ggml_tensor * audio_input = nullptr;
    ggml_tensor * reference_input = nullptr;
    ggml_tensor * audio_positions_tensor = nullptr;
    ggml_tensor * text_positions_tensor = nullptr;
    ggml_tensor * joint_positions_tensor = nullptr;
    ggml_tensor * text_gate_tensor = nullptr;
    ggml_tensor * first_double_output = nullptr;
    ggml_tensor * first_single_output = nullptr;
    ggml_tensor * velocity_output = nullptr;
    int64_t ref_frames = 0;
    ggml_tensor * time_output = nullptr;
    ggml_tensor * text_output = nullptr;
    ggml_tensor * audio_output = nullptr;
    int64_t text_tokens = 0;
    int64_t gen_frames = 0;
    float timestep = 0.0F;

    void release() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution->backend(), graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
        plan.reset();
        graph = nullptr;
        sinusoid = text_input = audio_input = reference_input = nullptr;
        audio_positions_tensor = text_positions_tensor = joint_positions_tensor = nullptr;
        text_gate_tensor = nullptr;
        time_output = text_output = audio_output = nullptr;
        first_double_output = first_single_output = velocity_output = nullptr;
        ref_frames = 0;
        text_tokens = gen_frames = 0;
    }

    void ensure_graph(const AukDitInputs & inputs) {
        // ⚠ NOT keyed on the timestep. The sinusoids are an input tensor, so one graph
        // serves every step; keying on it would rebuild a 30-block graph per Euler step.
        if (ggml != nullptr && text_tokens == inputs.text_tokens && gen_frames == inputs.gen_frames &&
            ref_frames == inputs.ref_frames) {
            return;
        }
        release();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize AuK DiT graph context");
        }
        core::ModuleBuildContext ctx{ggml, "auk.dit", execution->backend_type()};

        auto time_value = build_time_embedding(ctx, config, weights, inputs.timestep);
        sinusoid = ggml_get_tensor(ggml, "auk.dit.time_sinusoid");
        time_output = core::ensure_backend_addressable_layout(ctx, time_value).tensor;
        ggml_set_output(time_output);

        // project_text: Linear -> RMSNorm, in that order (flux2_edit.py:207).
        auto text = core::make_tensor(
            ctx, GGML_TYPE_F32, core::TensorShape::from_dims({inputs.text_tokens, config.text_hidden_dim}));
        text_input = text.tensor;
        ggml_set_input(text_input);
        auto projected = mod::LinearModule({config.text_hidden_dim, config.dim, true})
                             .build(ctx, text, weights.txt_proj);
        projected = mod::RMSNormModule({config.dim, config.rms_norm_eps, true, false})
                        .build(ctx, projected, {weights.txt_norm, std::nullopt});
        // drop_text zeroes the context AFTER projection (flux2_edit.py:208), so it is a
        // runtime multiplier rather than a second graph.
        text_gate_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_F32, 1);
        ggml_set_input(text_gate_tensor);
        auto gate_row = core::wrap_tensor(
            text_gate_tensor, core::TensorShape::from_dims({1, 1}), GGML_TYPE_F32);
        projected = mod::MulModule().build(
            ctx, projected, mod::RepeatModule({projected.shape}).build(ctx, gate_row));
        text_output = core::ensure_backend_addressable_layout(ctx, projected).tensor;
        ggml_set_output(text_output);

        // _embed: linear, then conv_pos_embed(x) + x -- a residual, not a replacement.
        auto audio = core::make_tensor(
            ctx, GGML_TYPE_F32, core::TensorShape::from_dims({inputs.gen_frames, config.latent_dim}));
        audio_input = audio.tensor;
        ggml_set_input(audio_input);
        auto embedded = mod::LinearModule({config.latent_dim, config.dim, true})
                            .build(ctx, audio, weights.audio_linear);
        auto positional = grouped_conv1d(
            ctx, embedded, weights.conv_pos_0.weight, *weights.conv_pos_0.bias, config.conv_pos_groups);
        positional = mish(ctx, positional);
        positional = grouped_conv1d(
            ctx, positional, weights.conv_pos_1.weight, *weights.conv_pos_1.bias, config.conv_pos_groups);
        positional = mish(ctx, positional);
        auto target = mod::AddModule().build(ctx, embedded, positional);
        audio_output = core::ensure_backend_addressable_layout(ctx, target).tensor;
        ggml_set_output(audio_output);

        // The reference latent goes through the SAME _embed and is prepended, so the
        // audio stream is [ref | target] and the target is recovered at the end by
        // dropping text_len + prompt_len (flux2_edit.py:344).
        auto audio_stream = target;
        if (inputs.ref_frames > 0) {
            auto reference = core::make_tensor(
                ctx, GGML_TYPE_F32, core::TensorShape::from_dims({inputs.ref_frames, config.latent_dim}));
            reference_input = reference.tensor;
            ggml_set_input(reference_input);
            auto ref_embedded = mod::LinearModule({config.latent_dim, config.dim, true})
                                    .build(ctx, reference, weights.audio_linear);
            auto ref_positional = grouped_conv1d(
                ctx, ref_embedded, weights.conv_pos_0.weight, *weights.conv_pos_0.bias, config.conv_pos_groups);
            ref_positional = mish(ctx, ref_positional);
            ref_positional = grouped_conv1d(
                ctx, ref_positional, weights.conv_pos_1.weight, *weights.conv_pos_1.bias, config.conv_pos_groups);
            ref_positional = mish(ctx, ref_positional);
            auto ref_stream = mod::AddModule().build(ctx, ref_embedded, ref_positional);
            audio_stream = mod::ConcatModule({0}).build(ctx, ref_stream, audio_stream);
        }

        const int64_t audio_frames = inputs.ref_frames + inputs.gen_frames;
        const int64_t joint_frames = inputs.text_tokens + audio_frames;
        const int64_t inner = static_cast<int64_t>(config.ff_mult * static_cast<double>(config.dim));

        // Independent position ranges: each stream is rotated from its own zero.
        audio_positions_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, audio_frames);
        ggml_set_input(audio_positions_tensor);
        text_positions_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, inputs.text_tokens);
        ggml_set_input(text_positions_tensor);
        joint_positions_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, joint_frames);
        ggml_set_input(joint_positions_tensor);
        auto audio_positions = core::wrap_tensor(
            audio_positions_tensor, core::TensorShape::from_dims({audio_frames}), GGML_TYPE_I32);
        auto text_positions = core::wrap_tensor(
            text_positions_tensor, core::TensorShape::from_dims({inputs.text_tokens}), GGML_TYPE_I32);
        auto joint_positions = core::wrap_tensor(
            joint_positions_tensor, core::TensorShape::from_dims({joint_frames}), GGML_TYPE_I32);

        auto context = projected;
        for (int64_t index = 0; index < config.double_layers; ++index) {
            auto out = build_double_block(
                ctx, config, weights.double_blocks[static_cast<size_t>(index)],
                audio_stream, context, time_value, audio_positions, text_positions, inner);
            context = out.context;
            audio_stream = out.audio;
            if (index == 0) {
                first_double_output = core::ensure_backend_addressable_layout(ctx, context).tensor;
                ggml_set_output(first_double_output);
            }
        }

        // ⚠ [text, audio] for the single-stream phase -- the opposite order to the
        // joint attention inside the double blocks.
        auto joined = mod::ConcatModule({0}).build(ctx, context, audio_stream);
        for (int64_t index = 0; index < config.single_layers; ++index) {
            joined = build_single_block(
                ctx, config, weights.single_blocks[static_cast<size_t>(index)],
                joined, time_value, joint_positions, inner);
            if (index == 0) {
                first_single_output = core::ensure_backend_addressable_layout(ctx, joined).tensor;
                ggml_set_output(first_single_output);
            }
        }

        // Drop the text prefix and the reference prompt; what remains is the target.
        auto tail = mod::SliceModule({0, inputs.text_tokens + inputs.ref_frames, inputs.gen_frames})
                        .build(ctx, joined);

        // AdaLayerNorm_Final: chunk order is scale THEN shift, the reverse of the
        // six-way chunk in every other block (modules.py:108).
        auto final_activated = core::wrap_tensor(ggml_silu(ctx.ggml, time_value.tensor), time_value.shape, GGML_TYPE_F32);
        auto final_packed = mod::LinearModule({config.dim, config.dim * 2, true})
                                .build(ctx, final_activated, weights.norm_out);
        auto final_scale = mod::SliceModule({1, 0, config.dim}).build(ctx, final_packed);
        auto final_shift = mod::SliceModule({1, config.dim, config.dim}).build(ctx, final_packed);
        auto normed_tail = modulate(ctx, tail, final_scale, final_shift, config.dim);
        auto velocity = mod::LinearModule({config.dim, config.latent_dim, true})
                            .build(ctx, normed_tail, weights.proj_out);
        velocity_output = core::ensure_backend_addressable_layout(ctx, velocity).tensor;
        ggml_set_output(velocity_output);

        graph = ggml_new_graph_custom(ggml, kGraphNodeCapacity, false);
        ggml_build_forward_expand(graph, time_output);
        ggml_build_forward_expand(graph, text_output);
        ggml_build_forward_expand(graph, audio_output);
        ggml_build_forward_expand(graph, first_double_output);
        ggml_build_forward_expand(graph, first_single_output);
        ggml_build_forward_expand(graph, velocity_output);
        core::validate_backend_graph_supported(execution->backend(), graph, "auk.dit");
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution->backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            release();
            throw std::runtime_error("failed to allocate AuK DiT graph memory");
        }
        core::prepare_host_graph_plan(*execution, graph, plan);
        text_tokens = inputs.text_tokens;
        gen_frames = inputs.gen_frames;
        ref_frames = inputs.ref_frames;
        timestep = inputs.timestep;

    }

    void fill_sinusoid(float timestep) {
        sinusoid_values.assign(static_cast<size_t>(config.freq_embed_dim), 0.0F);
        const int64_t half = config.freq_embed_dim / 2;
        const double step = std::log(10000.0) / static_cast<double>(half - 1);
        for (int64_t index = 0; index < half; ++index) {
            const double frequency = static_cast<double>(config.time_scale) *
                                     static_cast<double>(timestep) *
                                     std::exp(static_cast<double>(index) * -step);
            sinusoid_values[static_cast<size_t>(index)] = static_cast<float>(std::sin(frequency));
            sinusoid_values[static_cast<size_t>(half + index)] = static_cast<float>(std::cos(frequency));
        }
    }

    std::vector<float> sinusoid_values;
};

AukDit::AukDit(
    AukDitConfig config,
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    std::string prefix)
    : impl_(std::make_unique<Impl>()) {
    config.validate();
    impl_->config = std::move(config);
    impl_->execution = &execution;
    impl_->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "auk.dit.weights",
        8ull * 1024ull * 1024ull * 1024ull);

    auto & store = *impl_->store;
    const auto & cfg = impl_->config;
    const auto storage = assets::TensorStorageType::Native;
    auto & weights = impl_->weights;
    const std::string base = prefix + ".";

    weights.time_mlp_in.weight = store.load_tensor(source, base + "time_embed.time_mlp.0.weight", storage, {cfg.dim, cfg.freq_embed_dim});
    weights.time_mlp_in.bias = store.load_tensor(source, base + "time_embed.time_mlp.0.bias", assets::TensorStorageType::F32, {cfg.dim});
    weights.time_mlp_out.weight = store.load_tensor(source, base + "time_embed.time_mlp.2.weight", storage, {cfg.dim, cfg.dim});
    weights.time_mlp_out.bias = store.load_tensor(source, base + "time_embed.time_mlp.2.bias", assets::TensorStorageType::F32, {cfg.dim});
    weights.txt_proj.weight = store.load_tensor(source, base + "txt_proj.weight", storage, {cfg.dim, cfg.text_hidden_dim});
    weights.txt_proj.bias = store.load_tensor(source, base + "txt_proj.bias", assets::TensorStorageType::F32, {cfg.dim});
    weights.txt_norm = store.load_tensor(source, base + "txt_norm.weight", assets::TensorStorageType::F32, {cfg.dim});
    weights.audio_linear.weight = store.load_tensor(source, base + "audio_embed.linear.weight", storage, {cfg.dim, cfg.latent_dim});
    weights.audio_linear.bias = store.load_tensor(source, base + "audio_embed.linear.bias", assets::TensorStorageType::F32, {cfg.dim});

    const int64_t grouped_in = cfg.dim / cfg.conv_pos_groups;
    weights.conv_pos_0.weight = store.load_tensor(
        source, base + "audio_embed.conv_pos_embed.conv1d.0.weight", assets::TensorStorageType::F32,
        {cfg.dim, grouped_in, cfg.conv_pos_kernel});
    weights.conv_pos_0.bias = store.load_tensor(
        source, base + "audio_embed.conv_pos_embed.conv1d.0.bias", assets::TensorStorageType::F32, {cfg.dim});
    weights.conv_pos_1.weight = store.load_tensor(
        source, base + "audio_embed.conv_pos_embed.conv1d.2.weight", assets::TensorStorageType::F32,
        {cfg.dim, grouped_in, cfg.conv_pos_kernel});
    weights.conv_pos_1.bias = store.load_tensor(
        source, base + "audio_embed.conv_pos_embed.conv1d.2.bias", assets::TensorStorageType::F32, {cfg.dim});
    weights.inv_freq = store.load_tensor(
        source, base + "rotary_embed.inv_freq", assets::TensorStorageType::F32, {cfg.head_dim / 2});
    weights.norm_out.weight = store.load_tensor(source, base + "norm_out.linear.weight", storage, {cfg.dim * 2, cfg.dim});
    weights.norm_out.bias = store.load_tensor(source, base + "norm_out.linear.bias", assets::TensorStorageType::F32, {cfg.dim * 2});
    weights.proj_out.weight = store.load_tensor(source, base + "proj_out.weight", storage, {cfg.latent_dim, cfg.dim});
    weights.proj_out.bias = store.load_tensor(source, base + "proj_out.bias", assets::TensorStorageType::F32, {cfg.latent_dim});
    weights.loaded = 18;

    const int64_t inner = static_cast<int64_t>(cfg.ff_mult * static_cast<double>(cfg.dim));
    auto linear = [&](const std::string & name, int64_t out_dim, int64_t in_dim, bool bias) {
        mod::LinearWeights value;
        value.weight = store.load_tensor(source, name + ".weight", storage, {out_dim, in_dim});
        if (bias) {
            value.bias = store.load_tensor(source, name + ".bias", assets::TensorStorageType::F32, {out_dim});
        }
        weights.loaded += bias ? 2 : 1;
        return value;
    };
    auto head_norm = [&](const std::string & name) {
        weights.loaded += 1;
        return store.load_tensor(source, name + ".weight", assets::TensorStorageType::F32, {cfg.head_dim});
    };

    weights.double_blocks.reserve(static_cast<size_t>(cfg.double_layers));
    for (int64_t index = 0; index < cfg.double_layers; ++index) {
        const std::string block = base + "transformer_blocks." + std::to_string(index) + ".";
        DoubleBlockWeights value;
        value.norm_c = linear(block + "attn_norm_c.linear", cfg.dim * 6, cfg.dim, true);
        value.norm_x = linear(block + "attn_norm_x.linear", cfg.dim * 6, cfg.dim, true);
        value.qkv_x = linear(block + "attn.to_qkv", cfg.dim * 3, cfg.dim, true);
        value.qkv_c = linear(block + "attn.to_qkv_c", cfg.dim * 3, cfg.dim, true);
        value.q_norm = head_norm(block + "attn.q_norm");
        value.k_norm = head_norm(block + "attn.k_norm");
        value.c_q_norm = head_norm(block + "attn.c_q_norm");
        value.c_k_norm = head_norm(block + "attn.c_k_norm");
        value.out_x = linear(block + "attn.to_out.0", cfg.dim, cfg.dim, true);
        value.out_c = linear(block + "attn.to_out_c", cfg.dim, cfg.dim, true);
        value.ff_x_in = linear(block + "ff_x.linear_in", inner * 2, cfg.dim, false);
        value.ff_x_out = linear(block + "ff_x.linear_out", cfg.dim, inner, false);
        value.ff_c_in = linear(block + "ff_c.linear_in", inner * 2, cfg.dim, false);
        value.ff_c_out = linear(block + "ff_c.linear_out", cfg.dim, inner, false);
        weights.double_blocks.push_back(std::move(value));
    }

    weights.single_blocks.reserve(static_cast<size_t>(cfg.single_layers));
    for (int64_t index = 0; index < cfg.single_layers; ++index) {
        const std::string block = base + "single_transformer_blocks." + std::to_string(index) + ".";
        SingleBlockWeights value;
        value.norm = linear(block + "attn_norm.linear", cfg.dim * 6, cfg.dim, true);
        value.qkv = linear(block + "attn.to_qkv", cfg.dim * 3, cfg.dim, true);
        value.q_norm = head_norm(block + "attn.q_norm");
        value.k_norm = head_norm(block + "attn.k_norm");
        value.out = linear(block + "attn.to_out.0", cfg.dim, cfg.dim, true);
        value.ff_in = linear(block + "ff.linear_in", inner * 2, cfg.dim, false);
        value.ff_out = linear(block + "ff.linear_out", cfg.dim, inner, false);
        weights.single_blocks.push_back(std::move(value));
    }
    store.upload();
}

AukDit::~AukDit() {
    if (impl_ != nullptr) {
        impl_->release();
    }
}

AukDitStageOutputs AukDit::forward(const AukDitInputs & inputs) {
    if (inputs.text_tokens <= 0 || inputs.gen_frames <= 0) {
        throw std::runtime_error("AuK DiT received an empty text or audio stream");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->ensure_graph(inputs);
    impl_->fill_sinusoid(inputs.timestep);
    ggml_backend_tensor_set(
        impl_->sinusoid, impl_->sinusoid_values.data(), 0, impl_->sinusoid_values.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->text_input, inputs.text.data(), 0, inputs.text.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->audio_input, inputs.noised.data(), 0, inputs.noised.size() * sizeof(float));
    if (impl_->reference_input != nullptr) {
        ggml_backend_tensor_set(
            impl_->reference_input, inputs.reference.data(), 0, inputs.reference.size() * sizeof(float));
    }
    auto ramp = [](ggml_tensor * tensor, int64_t count) {
        std::vector<int32_t> values(static_cast<size_t>(count));
        for (int64_t index = 0; index < count; ++index) {
            values[static_cast<size_t>(index)] = static_cast<int32_t>(index);
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(int32_t));
    };
    const float text_gate = inputs.drop_text ? 0.0F : 1.0F;
    ggml_backend_tensor_set(impl_->text_gate_tensor, &text_gate, 0, sizeof(float));
    ramp(impl_->audio_positions_tensor, inputs.ref_frames + inputs.gen_frames);
    ramp(impl_->text_positions_tensor, inputs.text_tokens);
    ramp(impl_->joint_positions_tensor, inputs.text_tokens + inputs.ref_frames + inputs.gen_frames);
    if (core::compute_graph(*impl_->execution, impl_->graph, impl_->plan, "auk.dit") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK DiT graph compute failed");
    }
    AukDitStageOutputs out;
    out.time_embedding = core::read_tensor_f32(impl_->time_output);
    out.projected_text = core::read_tensor_f32(impl_->text_output);
    out.embedded_audio = core::read_tensor_f32(impl_->audio_output);
    out.first_double = core::read_tensor_f32(impl_->first_double_output);
    out.first_single = core::read_tensor_f32(impl_->first_single_output);
    out.velocity = core::read_tensor_f32(impl_->velocity_output);
    return out;
}

int64_t AukDit::loaded_tensor_count() const noexcept { return impl_->weights.loaded; }

}  // namespace engine::models::auk
