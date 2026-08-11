#include "engine/community_models/minimax_h3/dit_denoiser.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace engine::models::minimax_h3 {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kVideoTag = 0;
constexpr int64_t kTextTag = 1;
constexpr int64_t kAudioTag = 2;
constexpr int64_t kModalityCount = 3;

bool matches_dit_weight_filter(
    std::string_view name,
    const std::vector<std::string> & required_names,
    const std::vector<std::string> & prefix_filters) {
    for (const auto & required : required_names) {
        if (name == required) {
            return true;
        }
    }
    for (const auto & prefix : prefix_filters) {
        if (name.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<float> load_dit_adaln_table(const assets::TensorSource & source) {
    if (!source.has_tensor("adaln_t_table")) {
        return {};
    }
    const auto meta = source.require_metadata("adaln_t_table");
    const auto table = source.require_tensor("adaln_t_table", assets::TensorStorageType::F32, meta.shape);
    if (table.type != GGML_TYPE_F32 || table.shape.rank != 2) {
        throw std::runtime_error("MiniMax-H3 adaln_t_table must be stored as F32 [grid, rank]");
    }
    std::vector<float> values(static_cast<size_t>(table.shape.num_elements()));
    std::memcpy(values.data(), table.bytes.data(), table.bytes.size());
    return values;
}

MiniMaxH3DitWeightStore::MiniMaxH3DitWeightStore(
    core::ExecutionContext & execution_context,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    size_t weight_context_bytes)
    : execution(execution_context),
      source_(std::move(tensor_source)),
      store_(execution.backend(), execution.backend_type(), "minimax_h3.dit.weights", weight_context_bytes) {
    if (source_ == nullptr) {
        throw std::runtime_error("MiniMax-H3 DiT tensor source is missing");
    }
    adaln_curve_table = load_dit_adaln_table(*source_);
    for (const auto & meta : source_->tensors()) {
        if (meta.name == "adaln_t_table") {
            continue;
        }
        weights_.emplace(
            meta.name,
            store_.load_tensor(*source_, meta.name, assets::TensorStorageType::Native, meta.shape));
    }
    store_.upload();
    source_->release_storage();
}

MiniMaxH3DitWeightStore::MiniMaxH3DitWeightStore(
    core::ExecutionContext & execution_context,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    size_t weight_context_bytes,
    const std::vector<std::string> & required_names,
    const std::vector<std::string> & prefix_filters,
    bool load_adaln_table)
    : execution(execution_context),
      source_(std::move(tensor_source)),
      store_(execution.backend(), execution.backend_type(), "minimax_h3.dit.layerwise.weights", weight_context_bytes) {
    if (source_ == nullptr) {
        throw std::runtime_error("MiniMax-H3 DiT tensor source is missing");
    }
    if (load_adaln_table) {
        adaln_curve_table = load_dit_adaln_table(*source_);
    }
    for (const auto & meta : source_->tensors()) {
        if (meta.name == "adaln_t_table" || !matches_dit_weight_filter(meta.name, required_names, prefix_filters)) {
            continue;
        }
        weights_.emplace(
            meta.name,
            store_.load_tensor(*source_, meta.name, assets::TensorStorageType::Native, meta.shape));
    }
    for (const auto & required : required_names) {
        if (weights_.find(required) == weights_.end()) {
            throw std::runtime_error("missing MiniMax-H3 DiT layerwise tensor: " + required);
        }
    }
    store_.upload();
    source_->release_storage();
}

const core::TensorValue & MiniMaxH3DitWeightStore::require(std::string_view name) const {
    const auto it = weights_.find(std::string(name));
    if (it == weights_.end()) {
        throw std::runtime_error("missing MiniMax-H3 DiT tensor: " + std::string(name));
    }
    return it->second;
}

const core::TensorValue * MiniMaxH3DitWeightStore::find(std::string_view name) const {
    const auto it = weights_.find(std::string(name));
    return it == weights_.end() ? nullptr : &it->second;
}

ggml_prec dit_linear_precision(const std::string & prefix, ggml_type weight_type) {
    if (!ggml_is_quantized(weight_type)) {
        return GGML_PREC_F32;
    }
    if (prefix.find(".mlp.fc2") != std::string::npos) {
        return GGML_PREC_F32;
    }
    if (prefix.find(".attn.out_proj") != std::string::npos) {
        return GGML_PREC_F32;
    }
    return GGML_PREC_DEFAULT;
}

core::TensorValue dit_f32_view_hnd(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & src,
    int64_t head_dim,
    int64_t seq,
    int64_t heads,
    size_t offset_f32,
    size_t seq_stride_f32,
    size_t head_stride_f32) {
    return core::wrap_tensor(
        ggml_view_3d(
            ctx.ggml,
            src.tensor,
            head_dim,
            seq,
            heads,
            seq_stride_f32 * sizeof(float),
            head_stride_f32 * sizeof(float),
            offset_f32 * sizeof(float)),
        core::TensorShape::from_dims({heads, seq, head_dim}),
        GGML_TYPE_F32);
}

core::TensorValue dit_linear_projection(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const std::string & prefix,
    const core::TensorValue & x,
    const core::TensorValue & weight,
    const core::TensorValue * bias,
    int64_t in_features,
    int64_t out_features) {
    const ggml_prec precision = dit_linear_precision(prefix, weight.tensor->type);
    if (weight.tensor->type == GGML_TYPE_I8) {
        const core::TensorValue * scale = weights.find(prefix + ".weight_scale");
        if (scale == nullptr) {
            throw std::runtime_error("MiniMax-H3 ConvRot INT8 tensor is missing scale: " + prefix + ".weight_scale");
        }
        if (ctx.backend_type != core::BackendType::Cuda) {
            throw std::runtime_error("MiniMax-H3 ConvRot INT8 DiT requires CUDA backend");
        }
        core::validate_rank_between(x, 1, core::kMaxTensorRank, "input");
        core::validate_last_dim(x, in_features, "input");
        core::validate_shape(weight, core::TensorShape::from_dims({out_features, in_features}), "weight");
        core::validate_shape(*scale, core::TensorShape::from_dims({out_features, 1}), "weight_scale");

        auto input = core::ensure_backend_addressable_layout(ctx, x);
        if (input.tensor->type != GGML_TYPE_F32) {
            input = core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
            input = core::ensure_backend_addressable_layout(ctx, input);
        }
        const core::TensorShape matrix_shape = input.shape.rank == 1
            ? core::TensorShape::from_dims({1, input.shape.last_dim()})
            : core::TensorShape::from_dims({input.shape.prefix_elements(), input.shape.last_dim()});
        auto matrix_input = core::reshape_tensor(ctx, input, matrix_shape);
        const core::TensorValue * bias_value = bias;
        std::optional<core::TensorValue> f32_bias;
        if (bias_value != nullptr && bias_value->tensor->type != GGML_TYPE_F32) {
            f32_bias = core::wrap_tensor(ggml_cast(ctx.ggml, bias_value->tensor, GGML_TYPE_F32), bias_value->shape, GGML_TYPE_F32);
            bias_value = &*f32_bias;
        }
        ggml_tensor * raw = ggml_convrot_linear(
            ctx.ggml,
            weight.tensor,
            matrix_input.tensor,
            scale->tensor,
            bias_value == nullptr ? nullptr : bias_value->tensor,
            256);
        auto projected = core::wrap_tensor(
            raw,
            core::TensorShape::from_dims({matrix_shape.at(0), out_features}),
            GGML_TYPE_F32);
        return core::reshape_tensor(ctx, projected, x.shape.with_last_dim(out_features));
    }
    if (bias != nullptr) {
        auto bias_value = *bias;
        if (bias_value.type != GGML_TYPE_F32) {
            bias_value = core::wrap_tensor(ggml_cast(ctx.ggml, bias_value.tensor, GGML_TYPE_F32), bias_value.shape, GGML_TYPE_F32);
        }
        return modules::LinearModule({in_features, out_features, true, precision}).build(
            ctx,
            x,
            {weight, bias_value});
    }
    modules::LinearWeights params{weight, std::nullopt};
    const bool use_fast_projection =
        ctx.backend_type == core::BackendType::Cuda && out_features % 4 == 0;
    return
        use_fast_projection
            ? modules::FastPackedProjection4Module({in_features, out_features, precision}).build(ctx, x, params)
            : modules::LinearModule({in_features, out_features, false, precision}).build(ctx, x, params);
}

core::TensorValue apply_dit_rope_hnd(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    const MiniMaxH3Config & cfg,
    int64_t seq) {
    const int64_t rot_half = cfg.rope_inv_freq_len * 3;
    const int64_t pass_dim = cfg.head_dim - rot_half * 2;
    if (rot_half <= 0 || pass_dim < 0) {
        throw std::runtime_error("MiniMax-H3 invalid DiT RoPE dimensions");
    }
    auto x_f32 = x.tensor->type == GGML_TYPE_F32
        ? x
        : core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32);
    auto x1 = dit_f32_view_hnd(ctx, x_f32, rot_half, seq, cfg.heads, 0, cfg.head_dim, cfg.head_dim * seq);
    auto x2 = dit_f32_view_hnd(ctx, x_f32, rot_half, seq, cfg.heads, static_cast<size_t>(rot_half), cfg.head_dim, cfg.head_dim * seq);
    x1 = core::wrap_tensor(ggml_cont_3d(ctx.ggml, x1.tensor, rot_half, seq, cfg.heads), x1.shape, GGML_TYPE_F32);
    x2 = core::wrap_tensor(ggml_cont_3d(ctx.ggml, x2.tensor, rot_half, seq, cfg.heads), x2.shape, GGML_TYPE_F32);
    auto cos_full = core::wrap_tensor(ggml_repeat(ctx.ggml, cos.tensor, x1.tensor), x1.shape, GGML_TYPE_F32);
    auto sin_full = core::wrap_tensor(ggml_repeat(ctx.ggml, sin.tensor, x1.tensor), x1.shape, GGML_TYPE_F32);
    auto x1_cos = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x1),
        core::ensure_backend_addressable_layout(ctx, cos_full));
    auto x2_sin = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x2),
        core::ensure_backend_addressable_layout(ctx, sin_full));
    auto out1 = core::wrap_tensor(ggml_sub(ctx.ggml, x1_cos.tensor, x2_sin.tensor), x1.shape, GGML_TYPE_F32);
    auto x2_cos = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x2),
        core::ensure_backend_addressable_layout(ctx, cos_full));
    auto x1_sin = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x1),
        core::ensure_backend_addressable_layout(ctx, sin_full));
    auto out2 = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x2_cos),
        core::ensure_backend_addressable_layout(ctx, x1_sin));
    auto rot = core::wrap_tensor(
        ggml_concat(ctx.ggml, out1.tensor, out2.tensor, 0),
        core::TensorShape::from_dims({cfg.heads, seq, rot_half * 2}),
        GGML_TYPE_F32);
    if (pass_dim == 0) {
        return rot;
    }
    auto pass = dit_f32_view_hnd(
        ctx,
        x_f32,
        pass_dim,
        seq,
        cfg.heads,
        static_cast<size_t>(rot_half * 2),
        cfg.head_dim,
        cfg.head_dim * seq);
    pass = core::wrap_tensor(ggml_cont_3d(ctx.ggml, pass.tensor, pass_dim, seq, cfg.heads), pass.shape, GGML_TYPE_F32);
    return core::wrap_tensor(
        ggml_concat(ctx.ggml, rot.tensor, pass.tensor, 0),
        core::TensorShape::from_dims({cfg.heads, seq, cfg.head_dim}),
        GGML_TYPE_F32);
}

std::vector<float> h3_rope_half_values(const MiniMaxH3Config & cfg, int64_t text_len, int64_t inv_freq_len, bool cosine) {
    const int64_t audio_rows = cfg.audio_steps * cfg.audio_channels;
    const int64_t total = text_len + audio_rows + cfg.video_patches;
    const int64_t rot_half = inv_freq_len * 3;
    std::vector<float> out(static_cast<size_t>(total * rot_half));
    const int64_t ph = cfg.video_latent_h / 2;
    const int64_t pw = cfg.video_latent_w / 2;
    const double sqrt_area = std::sqrt(static_cast<double>(cfg.video_latent_h * cfg.video_latent_w));
    const double h_ratio = static_cast<double>(cfg.video_latent_h) / sqrt_area;
    const double w_ratio = static_cast<double>(cfg.video_latent_w) / sqrt_area;
    const double h_left = (1.0 - h_ratio) * 0.5;
    const double w_left = (1.0 - w_ratio) * 0.5;
    const double h_step = ph > 0 ? h_ratio / static_cast<double>(ph) : 0.0;
    const double w_step = pw > 0 ? w_ratio / static_cast<double>(pw) : 0.0;
    for (int64_t row = 0; row < total; ++row) {
        double pos[3] = {0.0, 0.0, 0.0};
        if (row < text_len) {
            pos[0] = static_cast<double>(row);
        } else if (row < text_len + audio_rows) {
            const int64_t audio_row = row - text_len;
            const int64_t t = audio_row % cfg.audio_steps;
            const int64_t channel = audio_row / cfg.audio_steps;
            pos[0] = static_cast<double>(text_len + t);
            pos[2] = (w_left + (channel == 0 ? 0.0 : w_ratio - w_step)) * 32.0;
        } else {
            const int64_t v = row - text_len - audio_rows;
            const int64_t t = v / std::max<int64_t>(ph * pw, 1);
            const int64_t rem = v % std::max<int64_t>(ph * pw, 1);
            const int64_t h = rem / std::max<int64_t>(pw, 1);
            const int64_t w = rem % std::max<int64_t>(pw, 1);
            static constexpr double kFrameRescale = 5.0 / 3.0;
            static constexpr int64_t kFramePerToken[5] = {1, 4, 4, 4, 4};
            double t_pos = static_cast<double>(text_len);
            for (int64_t i = 0; i < t; ++i) {
                t_pos += kFrameRescale * static_cast<double>(kFramePerToken[i % 5]);
            }
            pos[0] = t_pos;
            pos[1] = (h_left + static_cast<double>(h) * h_step) * 32.0;
            pos[2] = (w_left + static_cast<double>(w) * w_step) * 32.0;
        }
        for (int axis = 0; axis < 3; ++axis) {
            for (int64_t i = 0; i < inv_freq_len; ++i) {
                const float inv = std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(inv_freq_len));
                const float v = cosine ? std::cos(static_cast<float>(pos[axis]) * inv) : std::sin(static_cast<float>(pos[axis]) * inv);
                out[static_cast<size_t>(row * rot_half + axis * inv_freq_len + i)] = v;
            }
        }
    }
    return out;
}


core::TensorValue h3_mlp(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x,
    const std::string & prefix) {
    auto fc1 = dit_linear_projection(ctx, weights, prefix + ".fc1", x, weights.require(prefix + ".fc1.weight"), nullptr, cfg.hidden, cfg.ffn * 2);
    auto hidden = core::wrap_tensor(
        ggml_cont_2d(ctx.ggml, ggml_swiglu(ctx.ggml, fc1.tensor), cfg.ffn, x.shape.dims[0]),
        x.shape.with_last_dim(cfg.ffn),
        GGML_TYPE_F32);
    return dit_linear_projection(ctx, weights, prefix + ".fc2", hidden, weights.require(prefix + ".fc2.weight"), nullptr, cfg.ffn, cfg.hidden);
}

core::TensorValue h3_attention(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x,
    const core::TensorValue * cos,
    const core::TensorValue * sin,
    const std::string & prefix) {
    const int64_t tokens = x.shape.dims[0];
    auto qkv = dit_linear_projection(ctx, weights, prefix + ".qkv_proj", x, weights.require(prefix + ".qkv_proj.weight"), nullptr, cfg.hidden, cfg.heads * cfg.head_dim * 3);
    qkv = core::wrap_tensor(
        ggml_reshape_4d(ctx.ggml, qkv.tensor, cfg.head_dim, 3, cfg.heads, tokens),
        core::TensorShape::from_dims({tokens, cfg.heads, 3, cfg.head_dim}),
        GGML_TYPE_F32);
    auto q = dit_f32_view_hnd(
        ctx,
        qkv,
        cfg.head_dim,
        tokens,
        cfg.heads,
        0,
        static_cast<size_t>(3 * cfg.head_dim * cfg.heads),
        static_cast<size_t>(3 * cfg.head_dim));
    auto k = dit_f32_view_hnd(
        ctx,
        qkv,
        cfg.head_dim,
        tokens,
        cfg.heads,
        static_cast<size_t>(cfg.head_dim),
        static_cast<size_t>(3 * cfg.head_dim * cfg.heads),
        static_cast<size_t>(3 * cfg.head_dim));
    auto v = dit_f32_view_hnd(
        ctx,
        qkv,
        cfg.head_dim,
        tokens,
        cfg.heads,
        static_cast<size_t>(2 * cfg.head_dim),
        static_cast<size_t>(3 * cfg.head_dim * cfg.heads),
        static_cast<size_t>(3 * cfg.head_dim));
    q = modules::RMSNormModule({cfg.head_dim, cfg.qk_norm_eps, true, false}).build(
        ctx, q, {weights.require(prefix + ".q_norm.weight"), std::nullopt});
    k = modules::RMSNormModule({cfg.head_dim, cfg.qk_norm_eps, true, false}).build(
        ctx, k, {weights.require(prefix + ".k_norm.weight"), std::nullopt});
    if (cos != nullptr && sin != nullptr) {
        q = apply_dit_rope_hnd(ctx, q, *cos, *sin, cfg, tokens);
        k = apply_dit_rope_hnd(ctx, k, *cos, *sin, cfg, tokens);
    }
    ggml_tensor * attn = nullptr;
    const float scale = 1.0F / std::sqrt(static_cast<float>(cfg.head_dim));
    if (ctx.backend_type == core::BackendType::Cuda && prefix.rfind("blocks.", 0) == 0) {
        auto q_f16 = core::wrap_tensor(ggml_cont_3d(ctx.ggml, ggml_cast(ctx.ggml, q.tensor, GGML_TYPE_F16), cfg.head_dim, tokens, cfg.heads), q.shape, GGML_TYPE_F16);
        auto k_f16 = core::wrap_tensor(ggml_cont_3d(ctx.ggml, ggml_cast(ctx.ggml, k.tensor, GGML_TYPE_F16), cfg.head_dim, tokens, cfg.heads), k.shape, GGML_TYPE_F16);
        auto v_f16 = core::wrap_tensor(ggml_cont_3d(ctx.ggml, ggml_cast(ctx.ggml, v.tensor, GGML_TYPE_F16), cfg.head_dim, tokens, cfg.heads), v.shape, GGML_TYPE_F16);
        ggml_tensor * sage_attn = ggml_sage_attn2(ctx.ggml, q_f16.tensor, k_f16.tensor, v_f16.tensor, scale, false);
        if (ggml_backend_supports_op(weights.execution.backend(), sage_attn)) {
            attn = sage_attn;
        }
    }
    if (attn == nullptr) {
        v = core::wrap_tensor(ggml_cont_3d(ctx.ggml, v.tensor, cfg.head_dim, tokens, cfg.heads), v.shape, GGML_TYPE_F32);
        attn = ggml_flash_attn_ext(ctx.ggml, q.tensor, k.tensor, v.tensor, nullptr, scale, 0.0F, 0.0F);
    }
    auto h = core::wrap_tensor(
        ggml_cont_2d(ctx.ggml, attn, cfg.heads * cfg.head_dim, tokens),
        core::TensorShape::from_dims({tokens, cfg.heads * cfg.head_dim}),
        GGML_TYPE_F32);
    return dit_linear_projection(ctx, weights, prefix + ".out_proj", h, weights.require(prefix + ".out_proj.weight"), nullptr, cfg.heads * cfg.head_dim, cfg.hidden);
}

core::TensorValue token_refiner(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & prompt) {
    auto x = prompt;
    const std::string prefix;
    for (int64_t layer = 0; layer < cfg.token_refiner_layers; ++layer) {
        const std::string p = prefix + "token_refiner.blocks." + std::to_string(layer) + ".";
        auto h = modules::RMSNormModule({cfg.hidden, cfg.norm_eps, true, false}).build(
            ctx, x, {weights.require(p + "norm1.weight"), std::nullopt});
        auto attn = h3_attention(ctx, weights, cfg, h, nullptr, nullptr, p + "attn");
        x = modules::AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, attn));
        h = modules::RMSNormModule({cfg.hidden, cfg.norm_eps, true, false}).build(
            ctx, x, {weights.require(p + "norm2.weight"), std::nullopt});
        auto mlp = h3_mlp(ctx, weights, cfg, h, p + "mlp");
        x = modules::AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, mlp));
    }
    return modules::RMSNormModule({cfg.hidden, cfg.final_norm_eps, true, false}).build(
        ctx, x, {weights.require(prefix + "token_refiner.final_norm.weight"), std::nullopt});
}

std::vector<core::TensorValue> adaln_chunks(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const core::TensorValue & t_emb,
    const std::string & prefix,
    int64_t time_dim,
    int64_t hidden,
    int64_t chunks,
    int64_t modality_count,
    bool apply_silu) {
    auto x = apply_silu ? modules::SiluModule{}.build(ctx, t_emb) : t_emb;
    x = dit_linear_projection(ctx, weights,
        prefix + ".linear",
        x,
        weights.require(prefix + ".linear.weight"),
        &weights.require(prefix + ".linear.bias"),
        time_dim,
        chunks * hidden * modality_count);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({t_emb.shape.dims[0] * modality_count, chunks * hidden}));
    std::vector<core::TensorValue> out;
    for (int64_t i = 0; i < chunks; ++i) {
        out.push_back(core::ensure_backend_addressable_layout(
            ctx,
            modules::SliceModule({1, i * hidden, hidden}).build(ctx, x)));
    }
    return out;
}

core::TensorValue select_rows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & table,
    const core::TensorValue & indices) {
    auto rows = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, table.tensor, indices.tensor),
        core::TensorShape::from_dims({indices.shape.dims[0], table.shape.dims[1]}),
        GGML_TYPE_F32);
    return core::ensure_backend_addressable_layout(ctx, rows);
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & shift,
    const core::TensorValue & scale_values,
    const core::TensorValue & indices) {
    auto shift_rows = select_rows(ctx, shift, indices);
    auto scale_rows = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, select_rows(ctx, scale_values, indices).tensor, 1.0F, 1.0F),
        x.shape,
        GGML_TYPE_F32);
    auto scaled = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::ensure_backend_addressable_layout(ctx, scale_rows));
    return modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, scaled),
        core::ensure_backend_addressable_layout(ctx, shift_rows));
}

core::TensorValue gated_residual(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & gate,
    const core::TensorValue & other,
    const core::TensorValue & indices) {
    auto gate_rows = select_rows(ctx, gate, indices);
    auto gated = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, gate_rows),
        core::ensure_backend_addressable_layout(ctx, other));
    return modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::ensure_backend_addressable_layout(ctx, gated));
}

core::TensorValue dit_block(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & combined_indices,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer) {
    const std::string p = "blocks." + std::to_string(layer) + ".";
    auto chunks = adaln_chunks(
        ctx,
        weights,
        t_emb,
        p + "adaln_proj",
        cfg.time_embed_dim,
        cfg.hidden,
        6,
        kModalityCount,
        cfg.adaln_curve_grid == 0);
    auto h = modules::RMSNormModule({cfg.hidden, cfg.norm_eps, true, false}).build(
        ctx, x_in, {weights.require(p + "norm1.weight"), std::nullopt});
    h = modulate(ctx, h, chunks[0], chunks[1], combined_indices);
    auto x = gated_residual(ctx, x_in, chunks[2], h3_attention(ctx, weights, cfg, h, &cos, &sin, p + "attn"), combined_indices);
    h = modules::RMSNormModule({cfg.hidden, cfg.norm_eps, true, false}).build(
        ctx, x, {weights.require(p + "norm2.weight"), std::nullopt});
    h = modulate(ctx, h, chunks[3], chunks[4], combined_indices);
    return gated_residual(ctx, x, chunks[5], h3_mlp(ctx, weights, cfg, h, p + "mlp"), combined_indices);
}

core::TensorValue dit_block_attention_part(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & combined_indices,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer) {
    const std::string p = "blocks." + std::to_string(layer) + ".";
    auto chunks = adaln_chunks(
        ctx,
        weights,
        t_emb,
        p + "adaln_proj",
        cfg.time_embed_dim,
        cfg.hidden,
        6,
        kModalityCount,
        cfg.adaln_curve_grid == 0);
    auto h = modules::RMSNormModule({cfg.hidden, cfg.norm_eps, true, false}).build(
        ctx, x_in, {weights.require(p + "norm1.weight"), std::nullopt});
    h = modulate(ctx, h, chunks[0], chunks[1], combined_indices);
    return gated_residual(ctx, x_in, chunks[2], h3_attention(ctx, weights, cfg, h, &cos, &sin, p + "attn"), combined_indices);
}

core::TensorValue dit_block_mlp_part(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & combined_indices,
    int64_t layer) {
    const std::string p = "blocks." + std::to_string(layer) + ".";
    auto chunks = adaln_chunks(
        ctx,
        weights,
        t_emb,
        p + "adaln_proj",
        cfg.time_embed_dim,
        cfg.hidden,
        6,
        kModalityCount,
        cfg.adaln_curve_grid == 0);
    auto h = modules::RMSNormModule({cfg.hidden, cfg.norm_eps, true, false}).build(
        ctx, x_in, {weights.require(p + "norm2.weight"), std::nullopt});
    h = modulate(ctx, h, chunks[3], chunks[4], combined_indices);
    return gated_residual(ctx, x_in, chunks[5], h3_mlp(ctx, weights, cfg, h, p + "mlp"), combined_indices);
}

core::TensorValue build_time_embed(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & features) {
    const std::string prefix;
    auto h = dit_linear_projection(ctx, weights,
        prefix + "time_embedder.proj_in",
        features,
        weights.require(prefix + "time_embedder.proj_in.weight"),
        &weights.require(prefix + "time_embedder.proj_in.bias"),
        cfg.timestep_input_dim,
        cfg.time_embed_hidden);
    h = modules::SiluModule{}.build(ctx, h);
    return dit_linear_projection(ctx, weights,
        prefix + "time_embedder.proj_out",
        h,
        weights.require(prefix + "time_embedder.proj_out.weight"),
        &weights.require(prefix + "time_embedder.proj_out.bias"),
        cfg.time_embed_hidden,
        cfg.time_embed_dim);
}


MiniMaxH3DitGraph::DitOutput build_dit(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
    const core::TensorValue & prompt,
    const core::TensorValue & audio_full,
    const core::TensorValue & video_full,
    const core::TensorValue & audio_mask,
    const core::TensorValue & video_mask,
    const core::TensorValue & text_positions,
    const core::TensorValue & timestep_features_value,
    const core::TensorValue & sigma_delta_audio,
    const core::TensorValue & sigma_delta_video,
    const core::TensorValue & combined_indices,
    const core::TensorValue & inverse_indices,
    const core::TensorValue & cos,
    const core::TensorValue & sin) {
    const std::string prefix;
    auto text = dit_linear_projection(ctx, weights,
        prefix + "condition_proj",
        prompt,
        weights.require(prefix + "condition_proj.weight"),
        &weights.require(prefix + "condition_proj.bias"),
        cfg.text_dim,
        cfg.hidden);
    text = token_refiner(ctx, weights, cfg, text);
    auto audio_projected = dit_linear_projection(ctx, weights,
        prefix + "audio_patch_proj",
        audio_full,
        weights.require(prefix + "audio_patch_proj.weight"),
        &weights.require(prefix + "audio_patch_proj.bias"),
        cfg.audio_latents_dim,
        cfg.hidden);
    auto video_projected = dit_linear_projection(ctx, weights,
        prefix + "video_patch_proj",
        video_full,
        weights.require(prefix + "video_patch_proj.weight"),
        &weights.require(prefix + "video_patch_proj.bias"),
        cfg.video_latents_dim * 4,
        cfg.hidden);
    auto audio = core::wrap_tensor(ggml_mul(ctx.ggml, audio_projected.tensor, audio_mask.tensor), audio_projected.shape, GGML_TYPE_F32);
    auto video = core::wrap_tensor(ggml_mul(ctx.ggml, video_projected.tensor, video_mask.tensor), video_projected.shape, GGML_TYPE_F32);
    auto audio_video = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, video),
        core::ensure_backend_addressable_layout(ctx, audio));
    auto text_zero_base = core::wrap_tensor(ggml_scale(ctx.ggml, audio_video.tensor, 0.0F), audio_video.shape, GGML_TYPE_F32);
    auto text_embed = core::wrap_tensor(
        ggml_set_rows(ctx.ggml, text_zero_base.tensor, text.tensor, text_positions.tensor),
        core::TensorShape::from_dims({layout.total, cfg.hidden}),
        GGML_TYPE_F32);
    auto hidden = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, audio_video),
        core::ensure_backend_addressable_layout(ctx, text_embed));
    auto t_emb = cfg.adaln_curve_grid > 0 ? timestep_features_value : build_time_embed(ctx, weights, cfg, timestep_features_value);
    for (int64_t layer = 0; layer < cfg.dit_layers; ++layer) {
        hidden = dit_block(ctx, weights, cfg, hidden, t_emb, combined_indices, cos, sin, layer);
    }
    auto target_hidden = hidden;
    auto chunks = adaln_chunks(
        ctx,
        weights,
        t_emb,
        prefix + "final_layer.adaln_proj",
        cfg.time_embed_dim,
        cfg.hidden,
        2,
        1,
        cfg.adaln_curve_grid == 0);
    hidden = modules::RMSNormModule({cfg.hidden, cfg.final_norm_eps, true, false}).build(
        ctx, hidden, {weights.require(prefix + "final_layer.norm.weight"), std::nullopt});
    hidden = modulate(ctx, hidden, chunks[0], chunks[1], inverse_indices);
    auto video_full_logits = dit_linear_projection(ctx, weights,
        prefix + "final_layer.video_out",
        hidden,
        weights.require(prefix + "final_layer.video_out.weight"),
        &weights.require(prefix + "final_layer.video_out.bias"),
        cfg.hidden,
        cfg.video_latents_dim * 4);
    auto audio_full_logits = dit_linear_projection(ctx, weights,
        prefix + "final_layer.audio_out",
        hidden,
        weights.require(prefix + "final_layer.audio_out.weight"),
        &weights.require(prefix + "final_layer.audio_out.bias"),
        cfg.hidden,
        cfg.audio_latents_dim);
    auto video_logits = modules::SliceModule({0, layout.video_start, layout.video_rows}).build(ctx, video_full_logits);
    auto audio_logits = modules::SliceModule({0, layout.audio_start, layout.audio_rows}).build(ctx, audio_full_logits);
    auto current_video = modules::SliceModule({0, layout.video_start, layout.video_rows}).build(ctx, video_full);
    auto current_audio = modules::SliceModule({0, layout.audio_start, layout.audio_rows}).build(ctx, audio_full);
    auto video_delta = core::wrap_tensor(
        ggml_mul(ctx.ggml, video_logits.tensor, sigma_delta_video.tensor),
        video_logits.shape,
        GGML_TYPE_F32);
    auto next_video = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, current_video),
        core::ensure_backend_addressable_layout(ctx, video_delta));
    auto audio_delta = core::wrap_tensor(
        ggml_mul(ctx.ggml, audio_logits.tensor, sigma_delta_audio.tensor),
        audio_logits.shape,
        GGML_TYPE_F32);
    auto next_audio = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, current_audio),
        core::ensure_backend_addressable_layout(ctx, audio_delta));
    return {
        video_logits,
        audio_logits,
        target_hidden,
        next_video,
        next_audio,
    };
}

MiniMaxH3DitGraph::DitOutput build_dit_cfg(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const MiniMaxH3DitGraph::PackedSequenceLayout & positive_layout,
    const MiniMaxH3DitGraph::PackedSequenceLayout & negative_layout,
    const core::TensorValue & positive_prompt,
    const core::TensorValue & positive_audio_full,
    const core::TensorValue & positive_video_full,
    const core::TensorValue & positive_audio_mask,
    const core::TensorValue & positive_video_mask,
    const core::TensorValue & positive_text_positions,
    const core::TensorValue & negative_prompt,
    const core::TensorValue & negative_audio_full,
    const core::TensorValue & negative_video_full,
    const core::TensorValue & negative_audio_mask,
    const core::TensorValue & negative_video_mask,
    const core::TensorValue & negative_text_positions,
    const core::TensorValue & timestep_features_value,
    const core::TensorValue & guidance_scale,
    const core::TensorValue & sigma_delta_audio,
    const core::TensorValue & sigma_delta_video,
    const core::TensorValue & positive_combined_indices,
    const core::TensorValue & positive_inverse_indices,
    const core::TensorValue & positive_cos,
    const core::TensorValue & positive_sin,
    const core::TensorValue & negative_combined_indices,
    const core::TensorValue & negative_inverse_indices,
    const core::TensorValue & negative_cos,
    const core::TensorValue & negative_sin) {
    auto positive = build_dit(
        ctx,
        weights,
        cfg,
        positive_layout,
        positive_prompt,
        positive_audio_full,
        positive_video_full,
        positive_audio_mask,
        positive_video_mask,
        positive_text_positions,
        timestep_features_value,
        sigma_delta_audio,
        sigma_delta_video,
        positive_combined_indices,
        positive_inverse_indices,
        positive_cos,
        positive_sin);
    auto negative = build_dit(
        ctx,
        weights,
        cfg,
        negative_layout,
        negative_prompt,
        negative_audio_full,
        negative_video_full,
        negative_audio_mask,
        negative_video_mask,
        negative_text_positions,
        timestep_features_value,
        sigma_delta_audio,
        sigma_delta_video,
        negative_combined_indices,
        negative_inverse_indices,
        negative_cos,
        negative_sin);
    auto video_delta_logits = core::wrap_tensor(
        ggml_mul(ctx.ggml, ggml_sub(ctx.ggml, positive.video_logits.tensor, negative.video_logits.tensor), guidance_scale.tensor),
        positive.video_logits.shape,
        GGML_TYPE_F32);
    auto video_logits = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, negative.video_logits),
        core::ensure_backend_addressable_layout(ctx, video_delta_logits));
    auto audio_delta_logits = core::wrap_tensor(
        ggml_mul(ctx.ggml, ggml_sub(ctx.ggml, positive.audio_logits.tensor, negative.audio_logits.tensor), guidance_scale.tensor),
        positive.audio_logits.shape,
        GGML_TYPE_F32);
    auto audio_logits = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, negative.audio_logits),
        core::ensure_backend_addressable_layout(ctx, audio_delta_logits));
    auto current_video = modules::SliceModule({0, positive_layout.video_start, positive_layout.video_rows}).build(ctx, positive_video_full);
    auto current_audio = modules::SliceModule({0, positive_layout.audio_start, positive_layout.audio_rows}).build(ctx, positive_audio_full);
    auto next_video = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, current_video),
        core::ensure_backend_addressable_layout(
            ctx,
            core::wrap_tensor(ggml_mul(ctx.ggml, video_logits.tensor, sigma_delta_video.tensor), video_logits.shape, GGML_TYPE_F32)));
    auto next_audio = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, current_audio),
        core::ensure_backend_addressable_layout(
            ctx,
            core::wrap_tensor(ggml_mul(ctx.ggml, audio_logits.tensor, sigma_delta_audio.tensor), audio_logits.shape, GGML_TYPE_F32)));
    auto hidden_delta = core::wrap_tensor(
        ggml_mul(ctx.ggml, ggml_sub(ctx.ggml, positive.hidden.tensor, negative.hidden.tensor), guidance_scale.tensor),
        positive.hidden.shape,
        GGML_TYPE_F32);
    auto hidden = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, negative.hidden),
        core::ensure_backend_addressable_layout(ctx, hidden_delta));
    return {video_logits, audio_logits, hidden, next_video, next_audio};
}

struct LayerwisePreludeOutput {
    core::TensorValue hidden;
    core::TensorValue t_emb;
};

LayerwisePreludeOutput build_dit_prelude(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
    const core::TensorValue & prompt,
    const core::TensorValue & audio_full,
    const core::TensorValue & video_full,
    const core::TensorValue & audio_mask,
    const core::TensorValue & video_mask,
    const core::TensorValue & text_positions,
    const core::TensorValue & timestep_features_value) {
    const std::string prefix;
    auto text = dit_linear_projection(ctx, weights,
        prefix + "condition_proj",
        prompt,
        weights.require(prefix + "condition_proj.weight"),
        &weights.require(prefix + "condition_proj.bias"),
        cfg.text_dim,
        cfg.hidden);
    text = token_refiner(ctx, weights, cfg, text);
    auto audio_projected = dit_linear_projection(ctx, weights,
        prefix + "audio_patch_proj",
        audio_full,
        weights.require(prefix + "audio_patch_proj.weight"),
        &weights.require(prefix + "audio_patch_proj.bias"),
        cfg.audio_latents_dim,
        cfg.hidden);
    auto video_projected = dit_linear_projection(ctx, weights,
        prefix + "video_patch_proj",
        video_full,
        weights.require(prefix + "video_patch_proj.weight"),
        &weights.require(prefix + "video_patch_proj.bias"),
        cfg.video_latents_dim * 4,
        cfg.hidden);
    auto audio = core::wrap_tensor(ggml_mul(ctx.ggml, audio_projected.tensor, audio_mask.tensor), audio_projected.shape, GGML_TYPE_F32);
    auto video = core::wrap_tensor(ggml_mul(ctx.ggml, video_projected.tensor, video_mask.tensor), video_projected.shape, GGML_TYPE_F32);
    auto audio_video = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, video),
        core::ensure_backend_addressable_layout(ctx, audio));
    auto text_zero_base = core::wrap_tensor(ggml_scale(ctx.ggml, audio_video.tensor, 0.0F), audio_video.shape, GGML_TYPE_F32);
    auto text_embed = core::wrap_tensor(
        ggml_set_rows(ctx.ggml, text_zero_base.tensor, text.tensor, text_positions.tensor),
        core::TensorShape::from_dims({layout.total, cfg.hidden}),
        GGML_TYPE_F32);
    auto hidden = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, audio_video),
        core::ensure_backend_addressable_layout(ctx, text_embed));
    auto t_emb = cfg.adaln_curve_grid > 0 ? timestep_features_value : build_time_embed(ctx, weights, cfg, timestep_features_value);
    return {hidden, t_emb};
}

core::TensorValue build_dit_layer_group(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & hidden_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & combined_indices,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer_begin,
    int64_t layer_end) {
    auto hidden = hidden_in;
    for (int64_t layer = layer_begin; layer < layer_end; ++layer) {
        hidden = dit_block(ctx, weights, cfg, hidden, t_emb, combined_indices, cos, sin, layer);
    }
    return hidden;
}

core::TensorValue build_dit_layer_chunked_mlp(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & hidden_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & combined_indices,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer,
    int64_t chunk_rows) {
    auto hidden = dit_block_attention_part(ctx, weights, cfg, hidden_in, t_emb, combined_indices, cos, sin, layer);
    const int64_t total_rows = hidden.shape.dims[0];
    if (chunk_rows <= 0 || chunk_rows >= total_rows) {
        return dit_block_mlp_part(ctx, weights, cfg, hidden, t_emb, combined_indices, layer);
    }

    core::TensorValue output;
    for (int64_t row = 0; row < total_rows; row += chunk_rows) {
        const int64_t rows = std::min<int64_t>(chunk_rows, total_rows - row);
        auto hidden_chunk = modules::SliceModule({0, row, rows}).build(ctx, hidden);
        auto index_chunk = modules::SliceModule({0, row, rows}).build(ctx, combined_indices);
        auto chunk_output = dit_block_mlp_part(ctx, weights, cfg, hidden_chunk, t_emb, index_chunk, layer);
        output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, chunk_output) : chunk_output;
    }
    return output;
}

core::TensorValue build_dit_chunked_layer_group(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & hidden_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & combined_indices,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer_begin,
    int64_t layer_end,
    int64_t chunk_rows) {
    auto hidden = hidden_in;
    for (int64_t layer = layer_begin; layer < layer_end; ++layer) {
        hidden = build_dit_layer_chunked_mlp(ctx, weights, cfg, hidden, t_emb, combined_indices, cos, sin, layer, chunk_rows);
    }
    return hidden;
}

MiniMaxH3DitGraph::DitOutput build_dit_final(
    core::ModuleBuildContext & ctx,
    const MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
    const core::TensorValue & hidden_in,
    const core::TensorValue & t_emb,
    const core::TensorValue & inverse_indices,
    const core::TensorValue & audio_full,
    const core::TensorValue & video_full,
    const core::TensorValue & sigma_delta_audio,
    const core::TensorValue & sigma_delta_video) {
    const std::string prefix;
    auto chunks = adaln_chunks(
        ctx,
        weights,
        t_emb,
        prefix + "final_layer.adaln_proj",
        cfg.time_embed_dim,
        cfg.hidden,
        2,
        1,
        cfg.adaln_curve_grid == 0);
    auto hidden = modules::RMSNormModule({cfg.hidden, cfg.final_norm_eps, true, false}).build(
        ctx, hidden_in, {weights.require(prefix + "final_layer.norm.weight"), std::nullopt});
    hidden = modulate(ctx, hidden, chunks[0], chunks[1], inverse_indices);
    auto video_full_logits = dit_linear_projection(ctx, weights,
        prefix + "final_layer.video_out",
        hidden,
        weights.require(prefix + "final_layer.video_out.weight"),
        &weights.require(prefix + "final_layer.video_out.bias"),
        cfg.hidden,
        cfg.video_latents_dim * 4);
    auto audio_full_logits = dit_linear_projection(ctx, weights,
        prefix + "final_layer.audio_out",
        hidden,
        weights.require(prefix + "final_layer.audio_out.weight"),
        &weights.require(prefix + "final_layer.audio_out.bias"),
        cfg.hidden,
        cfg.audio_latents_dim);
    auto video_logits = modules::SliceModule({0, layout.video_start, layout.video_rows}).build(ctx, video_full_logits);
    auto audio_logits = modules::SliceModule({0, layout.audio_start, layout.audio_rows}).build(ctx, audio_full_logits);
    auto current_video = modules::SliceModule({0, layout.video_start, layout.video_rows}).build(ctx, video_full);
    auto current_audio = modules::SliceModule({0, layout.audio_start, layout.audio_rows}).build(ctx, audio_full);
    auto video_delta = core::wrap_tensor(
        ggml_mul(ctx.ggml, video_logits.tensor, sigma_delta_video.tensor),
        video_logits.shape,
        GGML_TYPE_F32);
    auto next_video = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, current_video),
        core::ensure_backend_addressable_layout(ctx, video_delta));
    auto audio_delta = core::wrap_tensor(
        ggml_mul(ctx.ggml, audio_logits.tensor, sigma_delta_audio.tensor),
        audio_logits.shape,
        GGML_TYPE_F32);
    auto next_audio = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, current_audio),
        core::ensure_backend_addressable_layout(ctx, audio_delta));
    return {video_logits, audio_logits, core::TensorValue{}, next_video, next_audio};
}

void build_dit_index_pattern(
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
    bool split_audio_timestep,
    std::vector<int32_t> & combined,
    std::vector<int32_t> & inverse) {
    combined.assign(static_cast<size_t>(layout.total), 0);
    inverse.assign(static_cast<size_t>(layout.total), 0);
    for (int64_t i = 0; i < layout.text_len; ++i) {
        combined[static_cast<size_t>(i)] = kTextTag;
    }
    for (int64_t i = 0; i < layout.audio_rows; ++i) {
        const int64_t row = layout.audio_start + i;
        inverse[static_cast<size_t>(row)] = split_audio_timestep ? 1 : 0;
        combined[static_cast<size_t>(row)] = (split_audio_timestep ? kModalityCount : 0) + kAudioTag;
    }
    for (int64_t i = 0; i < layout.video_rows; ++i) {
        combined[static_cast<size_t>(layout.video_start + i)] = kVideoTag;
    }
}

MiniMaxH3DitGraph::PackedSequenceLayout make_dit_layout(
    int64_t text_len,
    const MiniMaxH3Config & cfg) {
    const int64_t audio_rows = cfg.audio_steps * cfg.audio_channels;
    return {
        text_len,
        text_len,
        audio_rows,
        text_len + audio_rows,
        cfg.video_patches,
        text_len + audio_rows + cfg.video_patches};
}

struct DitLayerwiseGgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

class DitLayerwisePreludeGraph {
public:
    DitLayerwisePreludeGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
        const std::vector<float> & prompt)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT layerwise prelude graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.layerwise.prelude.inputs", execution_.backend_type()};
        prompt_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.text_len, cfg_.text_dim}));
        audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.audio_latents_dim}));
        video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.video_latents_dim * 4}));
        audio_mask_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, 1}));
        video_mask_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, 1}));
        text_positions_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.text_len}));
        time_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.adaln_curve_grid > 0 ? cfg_.time_embed_dim : cfg_.timestep_input_dim}));
        for (auto * input :
             {prompt_t_.tensor, audio_t_.tensor, video_t_.tensor, audio_mask_t_.tensor, video_mask_t_.tensor, text_positions_t_.tensor, time_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.layerwise.prelude", execution_.backend_type()};
        out_ = build_dit_prelude(
            build_ctx,
            weights,
            cfg_,
            layout_,
            prompt_t_,
            audio_t_,
            video_t_,
            audio_mask_t_,
            video_mask_t_,
            text_positions_t_,
            time_t_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : {out_.hidden.tensor, out_.t_emb.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT layerwise prelude graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<float> audio_mask(static_cast<size_t>(layout_.total), 0.0F);
        std::vector<float> video_mask(static_cast<size_t>(layout_.total), 0.0F);
        for (int64_t i = 0; i < layout_.audio_rows; ++i) {
            audio_mask[static_cast<size_t>(layout_.audio_start + i)] = 1.0F;
        }
        for (int64_t i = 0; i < layout_.video_rows; ++i) {
            video_mask[static_cast<size_t>(layout_.video_start + i)] = 1.0F;
        }
        core::write_tensor_f32(prompt_t_, prompt);
        core::write_tensor_f32(audio_mask_t_, audio_mask);
        core::write_tensor_f32(video_mask_t_, video_mask);
        std::vector<int32_t> text_positions(static_cast<size_t>(layout_.text_len));
        for (int64_t i = 0; i < layout_.text_len; ++i) {
            text_positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(text_positions_t_, text_positions);
        std::vector<float> audio_zeros(static_cast<size_t>(layout_.total * cfg_.audio_latents_dim), 0.0F);
        std::vector<float> video_zeros(static_cast<size_t>(layout_.total * cfg_.video_latents_dim * 4), 0.0F);
        core::write_tensor_f32(audio_t_, audio_zeros);
        core::write_tensor_f32(video_t_, video_zeros);
    }

    ~DitLayerwisePreludeGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        std::vector<float> & hidden,
        std::vector<float> & t_emb,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        const int64_t video_dim = cfg_.video_latents_dim * 4;
        core::write_tensor_f32_slice(audio_t_, static_cast<size_t>(layout_.audio_start * cfg_.audio_latents_dim), audio_rows.data(), audio_rows.size());
        core::write_tensor_f32_slice(video_t_, static_cast<size_t>(layout_.video_start * video_dim), video_rows.data(), video_rows.size());
        core::write_tensor_f32(time_t_, timestep_values, timestep_count);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.layerwise.prelude");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT layerwise prelude graph compute failed");
        }
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(out_.hidden.tensor, hidden);
        core::read_tensor_f32_into(out_.t_emb.tensor, t_emb);
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue prompt_t_;
    core::TensorValue audio_t_;
    core::TensorValue video_t_;
    core::TensorValue audio_mask_t_;
    core::TensorValue video_mask_t_;
    core::TensorValue text_positions_t_;
    core::TensorValue time_t_;
    LayerwisePreludeOutput out_;
};

class DitLayerwiseBlockGroupGraph {
public:
    DitLayerwiseBlockGroupGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
        int64_t layer_begin,
        int64_t layer_end)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT layerwise block graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.layerwise.block.inputs", execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        t_emb_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.time_embed_dim}));
        combined_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        cos_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        sin_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        for (auto * input : {hidden_.tensor, t_emb_.tensor, combined_t_.tensor, cos_t_.tensor, sin_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.layerwise.block", execution_.backend_type()};
        output_ = build_dit_layer_group(build_ctx, weights, cfg_, hidden_, t_emb_, combined_t_, cos_t_, sin_t_, layer_begin, layer_end);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT layerwise block graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        core::write_tensor_f32(cos_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, true));
        core::write_tensor_f32(sin_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, false));
    }

    ~DitLayerwiseBlockGroupGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<float> & hidden_in,
        const std::vector<float> & t_emb,
        const std::vector<int32_t> & combined,
        std::vector<float> & hidden_out,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        core::write_tensor_f32(hidden_, hidden_in);
        core::write_tensor_f32(t_emb_, t_emb);
        core::write_tensor_i32(combined_t_, combined);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.layerwise.block");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT layerwise block graph compute failed");
        }
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(output_.tensor, hidden_out);
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue t_emb_;
    core::TensorValue combined_t_;
    core::TensorValue cos_t_;
    core::TensorValue sin_t_;
    core::TensorValue output_;
};

class DitLayerwiseChunkedBlockGroupGraph {
public:
    DitLayerwiseChunkedBlockGroupGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
        int64_t layer_begin,
        int64_t layer_end,
        int64_t chunk_rows)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT layerwise chunked block graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.layerwise.chunked_block.inputs", execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        t_emb_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.time_embed_dim}));
        combined_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        cos_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        sin_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        for (auto * input : {hidden_.tensor, t_emb_.tensor, combined_t_.tensor, cos_t_.tensor, sin_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.layerwise.chunked_block", execution_.backend_type()};
        output_ = build_dit_chunked_layer_group(
            build_ctx,
            weights,
            cfg_,
            hidden_,
            t_emb_,
            combined_t_,
            cos_t_,
            sin_t_,
            layer_begin,
            layer_end,
            chunk_rows);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT layerwise chunked block graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        core::write_tensor_f32(cos_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, true));
        core::write_tensor_f32(sin_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, false));
    }

    ~DitLayerwiseChunkedBlockGroupGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<float> & hidden_in,
        const std::vector<float> & t_emb,
        const std::vector<int32_t> & combined,
        std::vector<float> & hidden_out,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        core::write_tensor_f32(hidden_, hidden_in);
        core::write_tensor_f32(t_emb_, t_emb);
        core::write_tensor_i32(combined_t_, combined);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.layerwise.chunked_block");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT layerwise chunked block graph compute failed");
        }
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(output_.tensor, hidden_out);
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue t_emb_;
    core::TensorValue combined_t_;
    core::TensorValue cos_t_;
    core::TensorValue sin_t_;
    core::TensorValue output_;
};

class DitLayerwiseFinalGraph {
public:
    DitLayerwiseFinalGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT layerwise final graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.layerwise.final.inputs", execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        t_emb_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.time_embed_dim}));
        inverse_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.audio_latents_dim}));
        video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.video_latents_dim * 4}));
        sigma_delta_audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        sigma_delta_video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        for (auto * input : {hidden_.tensor, t_emb_.tensor, inverse_t_.tensor, audio_t_.tensor, video_t_.tensor,
                             sigma_delta_audio_t_.tensor, sigma_delta_video_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.layerwise.final", execution_.backend_type()};
        out_ = build_dit_final(build_ctx, weights, cfg_, layout_, hidden_, t_emb_, inverse_t_, audio_t_, video_t_, sigma_delta_audio_t_, sigma_delta_video_t_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : {out_.video_logits.tensor, out_.audio_logits.tensor, out_.next_video.tensor, out_.next_audio.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT layerwise final graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<float> audio_zeros(static_cast<size_t>(layout_.total * cfg_.audio_latents_dim), 0.0F);
        std::vector<float> video_zeros(static_cast<size_t>(layout_.total * cfg_.video_latents_dim * 4), 0.0F);
        core::write_tensor_f32(audio_t_, audio_zeros);
        core::write_tensor_f32(video_t_, video_zeros);
    }

    ~DitLayerwiseFinalGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<float> & hidden,
        const std::vector<float> & t_emb,
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const std::vector<int32_t> & inverse,
        float sigma_delta_audio,
        float sigma_delta_video,
        bool read_logits,
        bool read_next_rows,
        DitGraphResult & result,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        const int64_t video_dim = cfg_.video_latents_dim * 4;
        core::write_tensor_f32(hidden_, hidden);
        core::write_tensor_f32(t_emb_, t_emb);
        core::write_tensor_i32(inverse_t_, inverse);
        core::write_tensor_f32_slice(audio_t_, static_cast<size_t>(layout_.audio_start * cfg_.audio_latents_dim), audio_rows.data(), audio_rows.size());
        core::write_tensor_f32_slice(video_t_, static_cast<size_t>(layout_.video_start * video_dim), video_rows.data(), video_rows.size());
        core::write_tensor_f32(sigma_delta_audio_t_, &sigma_delta_audio, 1);
        core::write_tensor_f32(sigma_delta_video_t_, &sigma_delta_video, 1);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.layerwise.final");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT layerwise final graph compute failed");
        }
        const auto read_start = Clock::now();
        if (read_logits) {
            core::read_tensor_f32_into(out_.video_logits.tensor, result.video);
            core::read_tensor_f32_into(out_.audio_logits.tensor, result.audio);
        }
        if (read_next_rows) {
            core::read_tensor_f32_into(out_.next_video.tensor, result.next_video);
            core::read_tensor_f32_into(out_.next_audio.tensor, result.next_audio);
        }
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue t_emb_;
    core::TensorValue inverse_t_;
    core::TensorValue audio_t_;
    core::TensorValue video_t_;
    core::TensorValue sigma_delta_audio_t_;
    core::TensorValue sigma_delta_video_t_;
    MiniMaxH3DitGraph::DitOutput out_;
};

struct FirstBlockGraphOutput {
    core::TensorValue first_hidden;
    core::TensorValue first_residual;
    core::TensorValue residual_diff;
    core::TensorValue t_emb;
};

struct FirstBlockDeviceState {
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx;
    ggml_backend_buffer_t buffer = nullptr;
    core::TensorValue first_hidden;
    core::TensorValue t_emb;
    core::TensorValue first_residual;
    core::TensorValue previous_first_residual;
    core::TensorValue tail_residual;

    ~FirstBlockDeviceState() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    void init(core::ExecutionContext & execution, const MiniMaxH3Config & cfg, const MiniMaxH3DitGraph::PackedSequenceLayout & layout) {
        ctx.reset(ggml_init({64 * 1024 * 1024, nullptr, true}));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 first-block device state context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "minimax_h3.dit.first_block.state", execution.backend_type()};
        const auto hidden_shape = core::TensorShape::from_dims({layout.total, cfg.hidden});
        first_hidden = core::make_tensor(build_ctx, GGML_TYPE_F32, hidden_shape);
        t_emb = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg.time_embed_dim}));
        first_residual = core::make_tensor(build_ctx, GGML_TYPE_F32, hidden_shape);
        previous_first_residual = core::make_tensor(build_ctx, GGML_TYPE_F32, hidden_shape);
        tail_residual = core::make_tensor(build_ctx, GGML_TYPE_F32, hidden_shape);
        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend());
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-H3 first-block device state");
        }
        const std::vector<float> zeros(static_cast<size_t>(layout.total * cfg.hidden), 0.0F);
        core::write_tensor_f32(previous_first_residual, zeros);
        core::write_tensor_f32(tail_residual, zeros);
    }
};

class DitFirstBlockPreludeGraph {
public:
    DitFirstBlockPreludeGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout,
        const std::vector<float> & prompt)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT first-block graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.first_block.inputs", execution_.backend_type()};
        prompt_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.text_len, cfg_.text_dim}));
        audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.audio_latents_dim}));
        video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.video_latents_dim * 4}));
        audio_mask_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, 1}));
        video_mask_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, 1}));
        text_positions_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.text_len}));
        time_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.adaln_curve_grid > 0 ? cfg_.time_embed_dim : cfg_.timestep_input_dim}));
        combined_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        previous_residual_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        cos_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        sin_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        for (auto * input : {prompt_t_.tensor, audio_t_.tensor, video_t_.tensor, audio_mask_t_.tensor, video_mask_t_.tensor,
                             text_positions_t_.tensor, time_t_.tensor, combined_t_.tensor, previous_residual_t_.tensor,
                             cos_t_.tensor, sin_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.first_block", execution_.backend_type()};
        auto prelude = build_dit_prelude(
            build_ctx,
            weights,
            cfg_,
            layout_,
            prompt_t_,
            audio_t_,
            video_t_,
            audio_mask_t_,
            video_mask_t_,
            text_positions_t_,
            time_t_);
        out_.t_emb = prelude.t_emb;
        out_.first_hidden = build_dit_layer_group(
            build_ctx,
            weights,
            cfg_,
            prelude.hidden,
            prelude.t_emb,
            combined_t_,
            cos_t_,
            sin_t_,
            0,
            1);
        out_.first_residual = core::wrap_tensor(
            ggml_sub(build_ctx.ggml, out_.first_hidden.tensor, prelude.hidden.tensor),
            prelude.hidden.shape,
            GGML_TYPE_F32);
        auto diff_abs = core::wrap_tensor(
            ggml_abs(build_ctx.ggml, ggml_sub(build_ctx.ggml, out_.first_residual.tensor, previous_residual_t_.tensor)),
            out_.first_residual.shape,
            GGML_TYPE_F32);
        auto previous_abs = core::wrap_tensor(
            ggml_abs(build_ctx.ggml, previous_residual_t_.tensor),
            previous_residual_t_.shape,
            GGML_TYPE_F32);
        out_.residual_diff = core::wrap_tensor(
            ggml_div(build_ctx.ggml, ggml_sum(build_ctx.ggml, diff_abs.tensor), ggml_sum(build_ctx.ggml, previous_abs.tensor)),
            core::TensorShape::from_dims({1}),
            GGML_TYPE_F32);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : {out_.first_hidden.tensor, out_.first_residual.tensor, out_.residual_diff.tensor, out_.t_emb.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT first-block graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<float> audio_mask(static_cast<size_t>(layout_.total), 0.0F);
        std::vector<float> video_mask(static_cast<size_t>(layout_.total), 0.0F);
        for (int64_t i = 0; i < layout_.audio_rows; ++i) {
            audio_mask[static_cast<size_t>(layout_.audio_start + i)] = 1.0F;
        }
        for (int64_t i = 0; i < layout_.video_rows; ++i) {
            video_mask[static_cast<size_t>(layout_.video_start + i)] = 1.0F;
        }
        std::vector<int32_t> text_positions(static_cast<size_t>(layout_.text_len));
        for (int64_t i = 0; i < layout_.text_len; ++i) {
            text_positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        std::vector<float> audio_zeros(static_cast<size_t>(layout_.total * cfg_.audio_latents_dim), 0.0F);
        std::vector<float> video_zeros(static_cast<size_t>(layout_.total * cfg_.video_latents_dim * 4), 0.0F);
        core::write_tensor_f32(prompt_t_, prompt);
        core::write_tensor_f32(audio_mask_t_, audio_mask);
        core::write_tensor_f32(video_mask_t_, video_mask);
        core::write_tensor_i32(text_positions_t_, text_positions);
        core::write_tensor_f32(audio_t_, audio_zeros);
        core::write_tensor_f32(video_t_, video_zeros);
        core::write_tensor_f32(cos_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, true));
        core::write_tensor_f32(sin_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, false));
    }

    ~DitFirstBlockPreludeGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        const std::vector<int32_t> & combined,
        float & residual_diff,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        const int64_t video_dim = cfg_.video_latents_dim * 4;
        core::write_tensor_f32_slice(audio_t_, static_cast<size_t>(layout_.audio_start * cfg_.audio_latents_dim), audio_rows.data(), audio_rows.size());
        core::write_tensor_f32_slice(video_t_, static_cast<size_t>(layout_.video_start * video_dim), video_rows.data(), video_rows.size());
        core::write_tensor_f32(time_t_, timestep_values, timestep_count);
        core::write_tensor_i32(combined_t_, combined);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.first_block");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT first-block graph compute failed");
        }
        const auto read_start = Clock::now();
        std::vector<float> residual_diff_values;
        core::read_tensor_f32_into(out_.residual_diff.tensor, residual_diff_values);
        residual_diff = residual_diff_values.empty() ? -1.0F : residual_diff_values.front();
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

    ggml_tensor * previous_residual_input() const {
        return previous_residual_t_.tensor;
    }

    ggml_tensor * first_hidden_output() const {
        return out_.first_hidden.tensor;
    }

    ggml_tensor * first_residual_output() const {
        return out_.first_residual.tensor;
    }

    ggml_tensor * t_emb_output() const {
        return out_.t_emb.tensor;
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue prompt_t_;
    core::TensorValue audio_t_;
    core::TensorValue video_t_;
    core::TensorValue audio_mask_t_;
    core::TensorValue video_mask_t_;
    core::TensorValue text_positions_t_;
    core::TensorValue time_t_;
    core::TensorValue combined_t_;
    core::TensorValue previous_residual_t_;
    core::TensorValue cos_t_;
    core::TensorValue sin_t_;
    FirstBlockGraphOutput out_;
};

class DitFirstBlockTailFinalGraph {
public:
    DitFirstBlockTailFinalGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({64 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT first-block tail/final graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.first_block.tail_final.inputs", execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        t_emb_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.time_embed_dim}));
        combined_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        inverse_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.audio_latents_dim}));
        video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.video_latents_dim * 4}));
        sigma_delta_audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        sigma_delta_video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        cos_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        sin_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
        for (auto * input : {hidden_.tensor, t_emb_.tensor, combined_t_.tensor, inverse_t_.tensor, audio_t_.tensor, video_t_.tensor,
                             sigma_delta_audio_t_.tensor, sigma_delta_video_t_.tensor, cos_t_.tensor, sin_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.first_block.tail_final", execution_.backend_type()};
        final_hidden_ = build_dit_layer_group(build_ctx, weights, cfg_, hidden_, t_emb_, combined_t_, cos_t_, sin_t_, 1, cfg_.dit_layers);
        tail_residual_ = core::wrap_tensor(
            ggml_sub(build_ctx.ggml, final_hidden_.tensor, hidden_.tensor),
            final_hidden_.shape,
            GGML_TYPE_F32);
        out_ = build_dit_final(
            build_ctx,
            weights,
            cfg_,
            layout_,
            final_hidden_,
            t_emb_,
            inverse_t_,
            audio_t_,
            video_t_,
            sigma_delta_audio_t_,
            sigma_delta_video_t_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : {tail_residual_.tensor, out_.next_video.tensor, out_.next_audio.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT first-block tail/final graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<float> audio_zeros(static_cast<size_t>(layout_.total * cfg_.audio_latents_dim), 0.0F);
        std::vector<float> video_zeros(static_cast<size_t>(layout_.total * cfg_.video_latents_dim * 4), 0.0F);
        core::write_tensor_f32(audio_t_, audio_zeros);
        core::write_tensor_f32(video_t_, video_zeros);
        core::write_tensor_f32(cos_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, true));
        core::write_tensor_f32(sin_t_, h3_rope_half_values(cfg_, layout_.text_len, cfg_.rope_inv_freq_len, false));
    }

    ~DitFirstBlockTailFinalGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<int32_t> & combined,
        const std::vector<int32_t> & inverse,
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        float sigma_delta_audio,
        float sigma_delta_video,
        DitGraphResult & result,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        const int64_t video_dim = cfg_.video_latents_dim * 4;
        core::write_tensor_i32(combined_t_, combined);
        core::write_tensor_i32(inverse_t_, inverse);
        core::write_tensor_f32_slice(audio_t_, static_cast<size_t>(layout_.audio_start * cfg_.audio_latents_dim), audio_rows.data(), audio_rows.size());
        core::write_tensor_f32_slice(video_t_, static_cast<size_t>(layout_.video_start * video_dim), video_rows.data(), video_rows.size());
        core::write_tensor_f32(sigma_delta_audio_t_, &sigma_delta_audio, 1);
        core::write_tensor_f32(sigma_delta_video_t_, &sigma_delta_video, 1);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.first_block.tail_final");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT first-block tail/final graph compute failed");
        }
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(out_.next_video.tensor, result.next_video);
        core::read_tensor_f32_into(out_.next_audio.tensor, result.next_audio);
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

    ggml_tensor * hidden_input() const {
        return hidden_.tensor;
    }

    ggml_tensor * t_emb_input() const {
        return t_emb_.tensor;
    }

    ggml_tensor * tail_residual_output() const {
        return tail_residual_.tensor;
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue t_emb_;
    core::TensorValue combined_t_;
    core::TensorValue inverse_t_;
    core::TensorValue audio_t_;
    core::TensorValue video_t_;
    core::TensorValue sigma_delta_audio_t_;
    core::TensorValue sigma_delta_video_t_;
    core::TensorValue cos_t_;
    core::TensorValue sin_t_;
    core::TensorValue final_hidden_;
    core::TensorValue tail_residual_;
    MiniMaxH3DitGraph::DitOutput out_;
};

class DitFirstBlockCachedFinalGraph {
public:
    DitFirstBlockCachedFinalGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout)
        : execution_(weights.execution),
          cfg_(cfg),
          layout_(layout) {
        ctx_.reset(ggml_init({512 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({64 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT first-block cached final graph context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.dit.first_block.cached_final.inputs", execution_.backend_type()};
        first_hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        tail_residual_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.hidden}));
        t_emb_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.time_embed_dim}));
        inverse_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
        audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.audio_latents_dim}));
        video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.video_latents_dim * 4}));
        sigma_delta_audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        sigma_delta_video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        for (auto * input : {first_hidden_.tensor, tail_residual_.tensor, t_emb_.tensor, inverse_t_.tensor, audio_t_.tensor, video_t_.tensor,
                             sigma_delta_audio_t_.tensor, sigma_delta_video_t_.tensor}) {
            ggml_set_input(input);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.dit.first_block.cached_final", execution_.backend_type()};
        auto hidden = core::wrap_tensor(
            ggml_add(build_ctx.ggml, first_hidden_.tensor, tail_residual_.tensor),
            first_hidden_.shape,
            GGML_TYPE_F32);
        out_ = build_dit_final(build_ctx, weights, cfg_, layout_, hidden, t_emb_, inverse_t_, audio_t_, video_t_, sigma_delta_audio_t_, sigma_delta_video_t_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : {out_.next_video.tensor, out_.next_audio.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT first-block cached final graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<float> audio_zeros(static_cast<size_t>(layout_.total * cfg_.audio_latents_dim), 0.0F);
        std::vector<float> video_zeros(static_cast<size_t>(layout_.total * cfg_.video_latents_dim * 4), 0.0F);
        core::write_tensor_f32(audio_t_, audio_zeros);
        core::write_tensor_f32(video_t_, video_zeros);
    }

    ~DitFirstBlockCachedFinalGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const std::vector<int32_t> & inverse,
        float sigma_delta_audio,
        float sigma_delta_video,
        DitGraphResult & result,
        double & input_ms,
        double & output_ms) {
        const auto input_start = Clock::now();
        const int64_t video_dim = cfg_.video_latents_dim * 4;
        core::write_tensor_i32(inverse_t_, inverse);
        core::write_tensor_f32_slice(audio_t_, static_cast<size_t>(layout_.audio_start * cfg_.audio_latents_dim), audio_rows.data(), audio_rows.size());
        core::write_tensor_f32_slice(video_t_, static_cast<size_t>(layout_.video_start * video_dim), video_rows.data(), video_rows.size());
        core::write_tensor_f32(sigma_delta_audio_t_, &sigma_delta_audio, 1);
        core::write_tensor_f32(sigma_delta_video_t_, &sigma_delta_video, 1);
        input_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.dit.first_block.cached_final");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT first-block cached final graph compute failed");
        }
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(out_.next_video.tensor, result.next_video);
        core::read_tensor_f32_into(out_.next_audio.tensor, result.next_audio);
        output_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }

    ggml_tensor * first_hidden_input() const {
        return first_hidden_.tensor;
    }

    ggml_tensor * tail_residual_input() const {
        return tail_residual_.tensor;
    }

    ggml_tensor * t_emb_input() const {
        return t_emb_.tensor;
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    MiniMaxH3DitGraph::PackedSequenceLayout layout_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, DitLayerwiseGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue first_hidden_;
    core::TensorValue tail_residual_;
    core::TensorValue t_emb_;
    core::TensorValue inverse_t_;
    core::TensorValue audio_t_;
    core::TensorValue video_t_;
    core::TensorValue sigma_delta_audio_t_;
    core::TensorValue sigma_delta_video_t_;
    MiniMaxH3DitGraph::DitOutput out_;
};

struct MiniMaxH3DitFirstBlockCacheRuntime::Impl {
    MiniMaxH3DitWeightStore * weights = nullptr;
    MiniMaxH3Config cfg;
    std::vector<float> prompt;
    MiniMaxH3DitGraph::PackedSequenceLayout layout;
    FirstBlockDeviceState state;
    std::unique_ptr<DitFirstBlockPreludeGraph> first_graph;
    std::unique_ptr<DitFirstBlockTailFinalGraph> tail_final_graph;
    std::unique_ptr<DitFirstBlockCachedFinalGraph> cached_final_graph;
    std::vector<int32_t> combined_shared;
    std::vector<int32_t> inverse_shared;
    std::vector<int32_t> combined_split;
    std::vector<int32_t> inverse_split;
    float threshold = 0.10F;
    float start_sigma = 0.95F;
    float end_sigma = 0.10F;
    int64_t max_consecutive_hits = 2;
    int64_t full_steps = 0;
    int64_t cached_steps = 0;
    int64_t consecutive_hits = 0;
    bool has_cache = false;
    double input_upload_ms = 0.0;
    double output_read_ms = 0.0;
};

MiniMaxH3DitFirstBlockCacheRuntime::MiniMaxH3DitFirstBlockCacheRuntime(
    MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const MiniMaxH3GenerateRequest & request,
    const std::vector<float> & prompt)
    : impl_(std::make_unique<Impl>()) {
    if (cfg.dit_layers < 2) {
        throw std::runtime_error("MiniMax-H3 first-block cache requires at least two DiT blocks");
    }
    impl_->cfg = cfg;
    impl_->weights = &weights;
    impl_->prompt = prompt;
    impl_->threshold = request.first_block_cache_threshold;
    impl_->start_sigma = request.first_block_cache_start_sigma;
    impl_->end_sigma = request.first_block_cache_end_sigma;
    impl_->max_consecutive_hits = request.first_block_cache_max_consecutive;
    if (!(impl_->threshold >= 0.0F) || !std::isfinite(impl_->threshold) ||
        !std::isfinite(impl_->start_sigma) || !std::isfinite(impl_->end_sigma) ||
        impl_->start_sigma < impl_->end_sigma || impl_->max_consecutive_hits <= 0) {
        throw std::runtime_error("MiniMax-H3 first-block cache configuration is invalid");
    }
    const int64_t text_len = static_cast<int64_t>(prompt.size()) / cfg.text_dim;
    impl_->layout = {
        text_len,
        text_len,
        cfg.audio_steps * cfg.audio_channels,
        text_len + cfg.audio_steps * cfg.audio_channels,
        cfg.video_patches,
        text_len + cfg.audio_steps * cfg.audio_channels + cfg.video_patches};
    build_dit_index_pattern(impl_->layout, false, impl_->combined_shared, impl_->inverse_shared);
    build_dit_index_pattern(impl_->layout, true, impl_->combined_split, impl_->inverse_split);
    impl_->state.init(weights.execution, cfg, impl_->layout);
    const auto build_start = Clock::now();
    impl_->first_graph = std::make_unique<DitFirstBlockPreludeGraph>(weights, cfg, impl_->layout, prompt);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
}

MiniMaxH3DitFirstBlockCacheRuntime::~MiniMaxH3DitFirstBlockCacheRuntime() = default;

void MiniMaxH3DitFirstBlockCacheRuntime::run(
    const std::vector<float> & audio_rows,
    const std::vector<float> & video_rows,
    const float * timestep_values,
    size_t timestep_count,
    float sigma,
    float sigma_delta_audio,
    float sigma_delta_video,
    bool split_audio_timestep,
    DitGraphResult & result) {
    const auto & combined = split_audio_timestep ? impl_->combined_split : impl_->combined_shared;
    const auto & inverse = split_audio_timestep ? impl_->inverse_split : impl_->inverse_shared;
    if (impl_->first_graph == nullptr) {
        const auto build_start = Clock::now();
        impl_->first_graph = std::make_unique<DitFirstBlockPreludeGraph>(*impl_->weights, impl_->cfg, impl_->layout, impl_->prompt);
        engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.first_graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
    }
    ggml_backend_tensor_copy(impl_->state.previous_first_residual.tensor, impl_->first_graph->previous_residual_input());
    float residual_diff = -1.0F;
    const auto first_start = Clock::now();
    impl_->first_graph->run(
        audio_rows,
        video_rows,
        timestep_values,
        timestep_count,
        combined,
        residual_diff,
        impl_->input_upload_ms,
        impl_->output_read_ms);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.first_graph_ms", engine::debug::elapsed_ms(first_start, Clock::now()));
    const auto state_copy_start = Clock::now();
    ggml_backend_tensor_copy(impl_->first_graph->first_hidden_output(), impl_->state.first_hidden.tensor);
    ggml_backend_tensor_copy(impl_->first_graph->first_residual_output(), impl_->state.first_residual.tensor);
    ggml_backend_tensor_copy(impl_->first_graph->t_emb_output(), impl_->state.t_emb.tensor);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.state_copy_ms", engine::debug::elapsed_ms(state_copy_start, Clock::now()));

    const auto decision_start = Clock::now();
    bool compared = false;
    bool use_cache = false;
    const bool within_window = sigma <= impl_->start_sigma && sigma >= impl_->end_sigma;
    if (impl_->has_cache && within_window) {
        compared = true;
        use_cache = std::isfinite(residual_diff) &&
                    residual_diff <= impl_->threshold &&
                    impl_->consecutive_hits < impl_->max_consecutive_hits;
    }
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.residual_diff", residual_diff);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.cache_hit", use_cache ? 1.0 : 0.0);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.compared", compared ? 1.0 : 0.0);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.decision_ms", engine::debug::elapsed_ms(decision_start, Clock::now()));

    if (use_cache) {
        if (impl_->cached_final_graph == nullptr) {
            const auto build_start = Clock::now();
            impl_->cached_final_graph = std::make_unique<DitFirstBlockCachedFinalGraph>(*impl_->weights, impl_->cfg, impl_->layout);
            engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.final_graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        }
        const auto reconstruct_start = Clock::now();
        ggml_backend_tensor_copy(impl_->state.first_hidden.tensor, impl_->cached_final_graph->first_hidden_input());
        ggml_backend_tensor_copy(impl_->state.tail_residual.tensor, impl_->cached_final_graph->tail_residual_input());
        ggml_backend_tensor_copy(impl_->state.t_emb.tensor, impl_->cached_final_graph->t_emb_input());
        ++impl_->cached_steps;
        ++impl_->consecutive_hits;
        engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.reconstruct_ms", engine::debug::elapsed_ms(reconstruct_start, Clock::now()));
    } else {
        if (impl_->tail_final_graph == nullptr) {
            const auto build_start = Clock::now();
            impl_->tail_final_graph = std::make_unique<DitFirstBlockTailFinalGraph>(*impl_->weights, impl_->cfg, impl_->layout);
            engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.tail_final_graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        }
        const auto copy_start = Clock::now();
        ggml_backend_tensor_copy(impl_->state.first_hidden.tensor, impl_->tail_final_graph->hidden_input());
        ggml_backend_tensor_copy(impl_->state.t_emb.tensor, impl_->tail_final_graph->t_emb_input());
        engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.tail_input_copy_ms", engine::debug::elapsed_ms(copy_start, Clock::now()));
        const auto tail_start = Clock::now();
        impl_->tail_final_graph->run(
            combined,
            inverse,
            audio_rows,
            video_rows,
            sigma_delta_audio,
            sigma_delta_video,
            result,
            impl_->input_upload_ms,
            impl_->output_read_ms);
        engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.tail_final_ms", engine::debug::elapsed_ms(tail_start, Clock::now()));
        const auto update_start = Clock::now();
        ggml_backend_tensor_copy(impl_->tail_final_graph->tail_residual_output(), impl_->state.tail_residual.tensor);
        ggml_backend_tensor_copy(impl_->state.first_residual.tensor, impl_->state.previous_first_residual.tensor);
        impl_->has_cache = true;
        impl_->consecutive_hits = 0;
        ++impl_->full_steps;
        engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.cache_update_ms", engine::debug::elapsed_ms(update_start, Clock::now()));
    }
    if (!use_cache) {
        return;
    }
    const auto final_start = Clock::now();
    impl_->cached_final_graph->run(
        audio_rows,
        video_rows,
        inverse,
        sigma_delta_audio,
        sigma_delta_video,
        result,
        impl_->input_upload_ms,
        impl_->output_read_ms);
    engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.final_ms", engine::debug::elapsed_ms(final_start, Clock::now()));
}

double MiniMaxH3DitFirstBlockCacheRuntime::input_upload_ms() const {
    return impl_->input_upload_ms;
}

double MiniMaxH3DitFirstBlockCacheRuntime::output_read_ms() const {
    return impl_->output_read_ms;
}

int64_t MiniMaxH3DitFirstBlockCacheRuntime::full_steps() const {
    return impl_->full_steps;
}

int64_t MiniMaxH3DitFirstBlockCacheRuntime::cached_steps() const {
    return impl_->cached_steps;
}

MiniMaxH3DitLayerwiseRuntime::MiniMaxH3DitLayerwiseRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & prompt,
    size_t weight_context_bytes,
    int64_t layer_batch,
    int64_t mlp_chunk_tokens)
    : execution_(execution),
      tensor_source_(std::move(tensor_source)),
      cfg_(cfg),
      prompt_(prompt),
      weight_context_bytes_(weight_context_bytes),
      layer_batch_(std::max<int64_t>(1, layer_batch)),
      mlp_chunk_tokens_(std::max<int64_t>(0, mlp_chunk_tokens)) {
    if (tensor_source_ == nullptr) {
        throw std::runtime_error("MiniMax-H3 DiT layerwise tensor source is missing");
    }
    adaln_curve_table_ = load_dit_adaln_table(*tensor_source_);
    tensor_source_->release_storage();
}

void MiniMaxH3DitLayerwiseRuntime::run(
    const std::vector<float> & audio_rows,
    const std::vector<float> & video_rows,
    const float * timestep_values,
    size_t timestep_count,
    float sigma_delta_audio,
    float sigma_delta_video,
    bool split_audio_timestep,
    bool read_logits,
    bool read_next_rows,
    DitGraphResult & result) {
    const int64_t text_len = static_cast<int64_t>(prompt_.size()) / cfg_.text_dim;
    const MiniMaxH3DitGraph::PackedSequenceLayout layout = make_dit_layout(text_len, cfg_);
    std::vector<int32_t> combined;
    std::vector<int32_t> inverse;
    build_dit_index_pattern(layout, split_audio_timestep, combined, inverse);
    std::vector<float> hidden;
    std::vector<float> t_emb;
    {
        std::vector<std::string> prefixes{
            "condition_proj.",
            "audio_patch_proj.",
            "video_patch_proj.",
            "token_refiner."};
        if (cfg_.adaln_curve_grid == 0) {
            prefixes.push_back("time_embedder.");
        }
        MiniMaxH3DitWeightStore weights(execution_, tensor_source_, weight_context_bytes_, {}, prefixes, false);
        DitLayerwisePreludeGraph graph(weights, cfg_, layout, prompt_);
        const auto compute_start = Clock::now();
        graph.run(audio_rows, video_rows, timestep_values, timestep_count, hidden, t_emb, input_upload_ms_, output_read_ms_);
        engine::debug::timing_log_scalar("minimax_h3.dit.layerwise.prelude_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    }
    if (mlp_chunk_tokens_ > 0) {
        for (int64_t layer = 0; layer < cfg_.dit_layers; layer += layer_batch_) {
            const int64_t end = std::min<int64_t>(layer + layer_batch_, cfg_.dit_layers);
            std::vector<std::string> prefixes;
            for (int64_t i = layer; i < end; ++i) {
                prefixes.push_back("blocks." + std::to_string(i) + ".");
            }
            MiniMaxH3DitWeightStore weights(execution_, tensor_source_, weight_context_bytes_, {}, prefixes, false);
            DitLayerwiseChunkedBlockGroupGraph graph(weights, cfg_, layout, layer, end, mlp_chunk_tokens_);
            std::vector<float> next_hidden;
            const auto compute_start = Clock::now();
            graph.run(hidden, t_emb, combined, next_hidden, input_upload_ms_, output_read_ms_);
            engine::debug::timing_log_scalar("minimax_h3.dit.layerwise.chunked_block_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
            hidden = std::move(next_hidden);
        }
    } else {
        for (int64_t layer = 0; layer < cfg_.dit_layers; layer += layer_batch_) {
            const int64_t end = std::min<int64_t>(layer + layer_batch_, cfg_.dit_layers);
            std::vector<std::string> prefixes;
            for (int64_t i = layer; i < end; ++i) {
                prefixes.push_back("blocks." + std::to_string(i) + ".");
            }
            MiniMaxH3DitWeightStore weights(execution_, tensor_source_, weight_context_bytes_, {}, prefixes, false);
            DitLayerwiseBlockGroupGraph graph(weights, cfg_, layout, layer, end);
            std::vector<float> next_hidden;
            const auto compute_start = Clock::now();
            graph.run(hidden, t_emb, combined, next_hidden, input_upload_ms_, output_read_ms_);
            engine::debug::timing_log_scalar("minimax_h3.dit.layerwise.block_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
            hidden = std::move(next_hidden);
        }
    }
    {
        MiniMaxH3DitWeightStore weights(execution_, tensor_source_, weight_context_bytes_, {}, {"final_layer."}, false);
        DitLayerwiseFinalGraph graph(weights, cfg_, layout);
        const auto compute_start = Clock::now();
        graph.run(
            hidden,
            t_emb,
            audio_rows,
            video_rows,
            inverse,
            sigma_delta_audio,
            sigma_delta_video,
            read_logits,
            read_next_rows,
            result,
            input_upload_ms_,
            output_read_ms_);
        engine::debug::timing_log_scalar("minimax_h3.dit.layerwise.final_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    }
}

double MiniMaxH3DitLayerwiseRuntime::input_upload_ms() const {
    return input_upload_ms_;
}

double MiniMaxH3DitLayerwiseRuntime::output_read_ms() const {
    return output_read_ms_;
}

const std::vector<float> & MiniMaxH3DitLayerwiseRuntime::adaln_curve_table() const {
    return adaln_curve_table_;
}

MiniMaxH3DitGraph::MiniMaxH3DitGraph(
    MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & prompt,
    bool include_sampler_update)
    : weights_(weights),
      cfg_(cfg),
      text_len_(static_cast<int64_t>(prompt.size()) / cfg.text_dim),
      layout_(make_dit_layout(text_len_, cfg)),
      include_sampler_update_(include_sampler_update) {
    ctx_ = ggml_init({1024 * 1024 * 1024, nullptr, true});
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize MiniMax-H3 DiT graph context");
    }
    input_ctx_ = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (input_ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize MiniMax-H3 DiT input context");
    }
    const auto build_start = Clock::now();
    core::ModuleBuildContext input_ctx{input_ctx_, "minimax_h3.dit.inputs", weights_.execution.backend_type()};
    prompt_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({text_len_, cfg_.text_dim}));
    audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.audio_latents_dim}));
    video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.video_latents_dim * 4}));
    audio_mask_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, 1}));
    video_mask_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, 1}));
    text_positions_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({text_len_}));
    time_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.adaln_curve_grid > 0 ? cfg_.time_embed_dim : cfg_.timestep_input_dim}));
    sigma_delta_audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    sigma_delta_video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    combined_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
    inverse_t_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({layout_.total}));
    cos_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
    sin_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({layout_.total, cfg_.rope_inv_freq_len * 3}));
    for (auto * input : {prompt_t_.tensor, audio_t_.tensor, video_t_.tensor, audio_mask_t_.tensor, video_mask_t_.tensor,
                         text_positions_t_.tensor, time_t_.tensor, sigma_delta_audio_t_.tensor, sigma_delta_video_t_.tensor, combined_t_.tensor,
                         inverse_t_.tensor, cos_t_.tensor, sin_t_.tensor}) {
        ggml_set_input(input);
    }
    core::ModuleBuildContext build_ctx{ctx_, "minimax_h3.dit", weights_.execution.backend_type()};
    out_ = build_dit(
        build_ctx,
        weights_,
        cfg_,
        layout_,
        prompt_t_,
        audio_t_,
        video_t_,
        audio_mask_t_,
        video_mask_t_,
        text_positions_t_,
        time_t_,
        sigma_delta_audio_t_,
        sigma_delta_video_t_,
        combined_t_,
        inverse_t_,
        cos_t_,
        sin_t_);
    graph_ = ggml_new_graph_custom(ctx_, 262144, false);
    for (auto * output : {out_.video_logits.tensor, out_.audio_logits.tensor, out_.hidden.tensor}) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph_, output);
    }
    if (include_sampler_update_) {
        for (auto * output : {out_.next_video.tensor, out_.next_audio.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
    }
    input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_, weights_.execution.backend());
    if (input_buffer_ == nullptr) {
        throw std::runtime_error("failed to allocate MiniMax-H3 DiT inputs");
    }
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_.execution.backend()));
    if (gallocr_ == nullptr ||
        !ggml_gallocr_reserve(gallocr_, graph_) ||
        !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("failed to allocate MiniMax-H3 DiT graph");
    }
    core::prepare_host_graph_plan(weights_.execution, graph_, plan_);
    engine::debug::timing_log_scalar("minimax_h3.dit.graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
    std::vector<float> audio_mask(static_cast<size_t>(layout_.total), 0.0F);
    std::vector<float> video_mask(static_cast<size_t>(layout_.total), 0.0F);
    for (int64_t i = 0; i < layout_.audio_rows; ++i) {
        audio_mask[static_cast<size_t>(layout_.audio_start + i)] = 1.0F;
    }
    for (int64_t i = 0; i < layout_.video_rows; ++i) {
        video_mask[static_cast<size_t>(layout_.video_start + i)] = 1.0F;
    }
    core::write_tensor_f32(prompt_t_, prompt);
    core::write_tensor_f32(audio_mask_t_, audio_mask);
    core::write_tensor_f32(video_mask_t_, video_mask);
    std::vector<int32_t> text_positions(static_cast<size_t>(text_len_));
    for (int64_t i = 0; i < text_len_; ++i) {
        text_positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    core::write_tensor_i32(text_positions_t_, text_positions);
    core::write_tensor_f32(cos_t_, h3_rope_half_values(cfg_, text_len_, cfg_.rope_inv_freq_len, true));
    core::write_tensor_f32(sin_t_, h3_rope_half_values(cfg_, text_len_, cfg_.rope_inv_freq_len, false));
    std::vector<float> audio_zeros(static_cast<size_t>(layout_.total * cfg_.audio_latents_dim), 0.0F);
    std::vector<float> video_zeros(static_cast<size_t>(layout_.total * cfg_.video_latents_dim * 4), 0.0F);
    core::write_tensor_f32(audio_t_, audio_zeros);
    core::write_tensor_f32(video_t_, video_zeros);
    build_dit_index_pattern(layout_, false, combined_shared_timestep_, inverse_shared_timestep_);
    build_dit_index_pattern(layout_, true, combined_split_timestep_, inverse_split_timestep_);
}

MiniMaxH3DitGraph::~MiniMaxH3DitGraph() {
    plan_.reset();
    if (graph_ != nullptr) {
        core::release_backend_graph_resources(weights_.execution.backend(), graph_);
    }
    if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
    }
    if (input_buffer_ != nullptr) {
        ggml_backend_buffer_free(input_buffer_);
    }
    if (input_ctx_ != nullptr) {
        ggml_free(input_ctx_);
    }
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

void MiniMaxH3DitGraph::run(
    const std::vector<float> & audio_rows,
    const std::vector<float> & video_rows,
    const float * timestep_values,
    size_t timestep_count,
    float sigma_delta_audio,
    float sigma_delta_video,
    bool split_audio_timestep,
    bool read_logits,
    bool read_hidden,
    bool read_next_rows,
    DitGraphResult & result) {
    const auto input_start = Clock::now();
    const int64_t video_dim = cfg_.video_latents_dim * 4;
    core::write_tensor_f32_slice(
        audio_t_,
        static_cast<size_t>(layout_.audio_start * cfg_.audio_latents_dim),
        audio_rows.data(),
        audio_rows.size());
    core::write_tensor_f32_slice(
        video_t_,
        static_cast<size_t>(layout_.video_start * video_dim),
        video_rows.data(),
        video_rows.size());
    core::write_tensor_f32(time_t_, timestep_values, timestep_count);
    core::write_tensor_f32(sigma_delta_audio_t_, &sigma_delta_audio, 1);
    core::write_tensor_f32(sigma_delta_video_t_, &sigma_delta_video, 1);
    if (!indices_uploaded_ || split_audio_timestep != indices_are_split_) {
        core::write_tensor_i32(combined_t_, split_audio_timestep ? combined_split_timestep_ : combined_shared_timestep_);
        core::write_tensor_i32(inverse_t_, split_audio_timestep ? inverse_split_timestep_ : inverse_shared_timestep_);
        indices_uploaded_ = true;
        indices_are_split_ = split_audio_timestep;
    }
    input_upload_ms_ += engine::debug::elapsed_ms(input_start, Clock::now());
    const auto compute_start = Clock::now();
    core::set_backend_threads(weights_.execution.backend(), 8);
    const ggml_status status = core::compute_graph(weights_.execution, graph_, plan_, "minimax_h3.dit");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MiniMax-H3 DiT graph compute failed");
    }
    ggml_backend_synchronize(weights_.execution.backend());
    engine::debug::timing_log_scalar("minimax_h3.dit.graph_compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    const auto read_start = Clock::now();
    if (read_logits) {
        core::read_tensor_f32_into(out_.video_logits.tensor, result.video);
        core::read_tensor_f32_into(out_.audio_logits.tensor, result.audio);
    }
    if (read_hidden) {
        core::read_tensor_f32_into(out_.hidden.tensor, result.hidden);
    }
    if (read_next_rows) {
        if (!include_sampler_update_) {
            throw std::runtime_error("MiniMax-H3 DiT graph was built without sampler update outputs");
        }
        core::read_tensor_f32_into(out_.next_video.tensor, result.next_video);
        core::read_tensor_f32_into(out_.next_audio.tensor, result.next_audio);
    }
    output_read_ms_ += engine::debug::elapsed_ms(read_start, Clock::now());
}

double MiniMaxH3DitGraph::input_upload_ms() const {
    return input_upload_ms_;
}

double MiniMaxH3DitGraph::output_read_ms() const {
    return output_read_ms_;
}

const MiniMaxH3DitGraph::PackedSequenceLayout & MiniMaxH3DitGraph::layout() const {
    return layout_;
}

ggml_tensor * MiniMaxH3DitGraph::video_state_tensor() const {
    return video_t_.tensor;
}

ggml_tensor * MiniMaxH3DitGraph::audio_state_tensor() const {
    return audio_t_.tensor;
}

ggml_tensor * MiniMaxH3DitGraph::video_logits_tensor() const {
    return out_.video_logits.tensor;
}

ggml_tensor * MiniMaxH3DitGraph::audio_logits_tensor() const {
    return out_.audio_logits.tensor;
}

struct MiniMaxH3DitFinalGraph::Impl {
    MiniMaxH3DitWeightStore & weights;
    MiniMaxH3Config cfg;
    MiniMaxH3DitGraph::PackedSequenceLayout layout;
    ggml_context * ctx = nullptr;
    ggml_context * input_ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue hidden;
    core::TensorValue time;
    core::TensorValue inverse;
    core::TensorValue audio;
    core::TensorValue video;
    core::TensorValue sigma_delta_audio;
    core::TensorValue sigma_delta_video;
    MiniMaxH3DitGraph::DitOutput out;
    std::vector<int32_t> inverse_shared_timestep;
    std::vector<int32_t> inverse_split_timestep;
    bool indices_uploaded = false;
    bool indices_are_split = false;
    double input_upload_ms = 0.0;
    double output_read_ms = 0.0;

    Impl(
        MiniMaxH3DitWeightStore & weights_in,
        const MiniMaxH3Config & cfg_in,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout_in)
        : weights(weights_in),
          cfg(cfg_in),
          layout(layout_in) {
        ctx = ggml_init({512 * 1024 * 1024, nullptr, true});
        input_ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
        if (ctx == nullptr || input_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 DiT final graph context");
        }
        core::ModuleBuildContext input_build{input_ctx, "minimax_h3.dit.final.inputs", weights.execution.backend_type()};
        hidden = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({layout.total, cfg.hidden}));
        time = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg.adaln_curve_grid > 0 ? cfg.time_embed_dim : cfg.timestep_input_dim}));
        inverse = core::make_tensor(input_build, GGML_TYPE_I32, core::TensorShape::from_dims({layout.total}));
        audio = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({layout.total, cfg.audio_latents_dim}));
        video = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({layout.total, cfg.video_latents_dim * 4}));
        sigma_delta_audio = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        sigma_delta_video = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        for (auto * input : {hidden.tensor, time.tensor, inverse.tensor, audio.tensor, video.tensor,
                             sigma_delta_audio.tensor, sigma_delta_video.tensor}) {
            ggml_set_input(input);
        }

        core::ModuleBuildContext build_ctx{ctx, "minimax_h3.dit.final", weights.execution.backend_type()};
        auto t_emb = cfg.adaln_curve_grid > 0 ? time : build_time_embed(build_ctx, weights, cfg, time);
        out = build_dit_final(
            build_ctx,
            weights,
            cfg,
            layout,
            hidden,
            t_emb,
            inverse,
            audio,
            video,
            sigma_delta_audio,
            sigma_delta_video);
        graph = ggml_new_graph_custom(ctx, 262144, false);
        for (auto * output : {out.next_video.tensor, out.next_audio.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
        }
        input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx, weights.execution.backend());
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.execution.backend()));
        if (input_buffer == nullptr || gallocr == nullptr ||
            !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 DiT final graph");
        }
        core::prepare_host_graph_plan(weights.execution, graph, plan);
        std::vector<float> audio_zeros(static_cast<size_t>(layout.total * cfg.audio_latents_dim), 0.0F);
        std::vector<float> video_zeros(static_cast<size_t>(layout.total * cfg.video_latents_dim * 4), 0.0F);
        core::write_tensor_f32(audio, audio_zeros);
        core::write_tensor_f32(video, video_zeros);
        std::vector<int32_t> combined_unused;
        build_dit_index_pattern(layout, false, combined_unused, inverse_shared_timestep);
        build_dit_index_pattern(layout, true, combined_unused, inverse_split_timestep);
    }

    ~Impl() {
        plan.reset();
        if (graph != nullptr) {
            core::release_backend_graph_resources(weights.execution.backend(), graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
        }
        if (input_ctx != nullptr) {
            ggml_free(input_ctx);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    void run(
        const std::vector<float> & hidden_rows,
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        float sigma_delta_audio_value,
        float sigma_delta_video_value,
        bool split_audio_timestep,
        DitGraphResult & result) {
        const auto input_start = Clock::now();
        if (hidden_rows.size() != static_cast<size_t>(layout.total * cfg.hidden)) {
            throw std::runtime_error("MiniMax-H3 DiT final graph hidden input shape mismatch");
        }
        const int64_t video_dim = cfg.video_latents_dim * 4;
        core::write_tensor_f32(hidden, hidden_rows);
        core::write_tensor_f32_slice(audio, static_cast<size_t>(layout.audio_start * cfg.audio_latents_dim), audio_rows.data(), audio_rows.size());
        core::write_tensor_f32_slice(video, static_cast<size_t>(layout.video_start * video_dim), video_rows.data(), video_rows.size());
        core::write_tensor_f32(time, timestep_values, timestep_count);
        core::write_tensor_f32(sigma_delta_audio, &sigma_delta_audio_value, 1);
        core::write_tensor_f32(sigma_delta_video, &sigma_delta_video_value, 1);
        if (!indices_uploaded || split_audio_timestep != indices_are_split) {
            core::write_tensor_i32(inverse, split_audio_timestep ? inverse_split_timestep : inverse_shared_timestep);
            indices_uploaded = true;
            indices_are_split = split_audio_timestep;
        }
        input_upload_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        const auto compute_start = Clock::now();
        core::set_backend_threads(weights.execution.backend(), 8);
        const ggml_status status = core::compute_graph(weights.execution, graph, plan, "minimax_h3.dit.final");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 DiT final graph compute failed");
        }
        ggml_backend_synchronize(weights.execution.backend());
        engine::debug::timing_log_scalar("minimax_h3.dit.final.graph_compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(out.next_video.tensor, result.next_video);
        core::read_tensor_f32_into(out.next_audio.tensor, result.next_audio);
        output_read_ms += engine::debug::elapsed_ms(read_start, Clock::now());
    }
};

MiniMaxH3DitFinalGraph::MiniMaxH3DitFinalGraph(
    MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout)
    : impl_(std::make_unique<Impl>(weights, cfg, layout)) {}

MiniMaxH3DitFinalGraph::~MiniMaxH3DitFinalGraph() = default;

void MiniMaxH3DitFinalGraph::run(
    const std::vector<float> & hidden,
    const std::vector<float> & audio_rows,
    const std::vector<float> & video_rows,
    const float * timestep_values,
    size_t timestep_count,
    float sigma_delta_audio,
    float sigma_delta_video,
    bool split_audio_timestep,
    DitGraphResult & result) {
    impl_->run(hidden, audio_rows, video_rows, timestep_values, timestep_count, sigma_delta_audio, sigma_delta_video, split_audio_timestep, result);
}

double MiniMaxH3DitFinalGraph::input_upload_ms() const {
    return impl_->input_upload_ms;
}

double MiniMaxH3DitFinalGraph::output_read_ms() const {
    return impl_->output_read_ms;
}

MiniMaxH3DitCfgGraph::MiniMaxH3DitCfgGraph(
    MiniMaxH3DitWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & positive_prompt,
    const std::vector<float> & negative_prompt,
    bool include_sampler_update)
    : weights_(weights),
      cfg_(cfg),
      include_sampler_update_(include_sampler_update) {
    positive_.text_len = static_cast<int64_t>(positive_prompt.size()) / cfg_.text_dim;
    negative_.text_len = static_cast<int64_t>(negative_prompt.size()) / cfg_.text_dim;
    positive_.layout = make_dit_layout(positive_.text_len, cfg_);
    negative_.layout = make_dit_layout(negative_.text_len, cfg_);
    ctx_ = ggml_init({1024 * 1024 * 1024, nullptr, true});
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize MiniMax-H3 CFG DiT graph context");
    }
    input_ctx_ = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (input_ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize MiniMax-H3 CFG DiT input context");
    }
    const auto build_start = Clock::now();
    const int64_t video_dim = cfg_.video_latents_dim * 4;
    core::ModuleBuildContext input_ctx{input_ctx_, "minimax_h3.dit.cfg.inputs", weights_.execution.backend_type()};
    positive_.prompt = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.text_len, cfg_.text_dim}));
    positive_.audio = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.layout.total, cfg_.audio_latents_dim}));
    positive_.video = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.layout.total, video_dim}));
    positive_.audio_mask = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.layout.total, 1}));
    positive_.video_mask = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.layout.total, 1}));
    positive_.text_positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({positive_.text_len}));
    negative_.prompt = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.text_len, cfg_.text_dim}));
    negative_.audio = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.layout.total, cfg_.audio_latents_dim}));
    negative_.video = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.layout.total, video_dim}));
    negative_.audio_mask = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.layout.total, 1}));
    negative_.video_mask = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.layout.total, 1}));
    negative_.text_positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({negative_.text_len}));
    time_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, cfg_.adaln_curve_grid > 0 ? cfg_.time_embed_dim : cfg_.timestep_input_dim}));
    guidance_scale_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    sigma_delta_audio_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    sigma_delta_video_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    positive_.combined = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({positive_.layout.total}));
    positive_.inverse = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({positive_.layout.total}));
    positive_.cos = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.layout.total, cfg_.rope_inv_freq_len * 3}));
    positive_.sin = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({positive_.layout.total, cfg_.rope_inv_freq_len * 3}));
    negative_.combined = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({negative_.layout.total}));
    negative_.inverse = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({negative_.layout.total}));
    negative_.cos = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.layout.total, cfg_.rope_inv_freq_len * 3}));
    negative_.sin = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({negative_.layout.total, cfg_.rope_inv_freq_len * 3}));
    for (auto * input : {
             positive_.prompt.tensor,
             positive_.audio.tensor,
             positive_.video.tensor,
             positive_.audio_mask.tensor,
             positive_.video_mask.tensor,
             positive_.text_positions.tensor,
             negative_.prompt.tensor,
             negative_.audio.tensor,
             negative_.video.tensor,
             negative_.audio_mask.tensor,
             negative_.video_mask.tensor,
             negative_.text_positions.tensor,
             time_t_.tensor,
             guidance_scale_t_.tensor,
             sigma_delta_audio_t_.tensor,
             sigma_delta_video_t_.tensor,
             positive_.combined.tensor,
             positive_.inverse.tensor,
             positive_.cos.tensor,
             positive_.sin.tensor,
             negative_.combined.tensor,
             negative_.inverse.tensor,
             negative_.cos.tensor,
             negative_.sin.tensor}) {
        ggml_set_input(input);
    }
    core::ModuleBuildContext build_ctx{ctx_, "minimax_h3.dit.cfg", weights_.execution.backend_type()};
    out_ = build_dit_cfg(
        build_ctx,
        weights_,
        cfg_,
        positive_.layout,
        negative_.layout,
        positive_.prompt,
        positive_.audio,
        positive_.video,
        positive_.audio_mask,
        positive_.video_mask,
        positive_.text_positions,
        negative_.prompt,
        negative_.audio,
        negative_.video,
        negative_.audio_mask,
        negative_.video_mask,
        negative_.text_positions,
        time_t_,
        guidance_scale_t_,
        sigma_delta_audio_t_,
        sigma_delta_video_t_,
        positive_.combined,
        positive_.inverse,
        positive_.cos,
        positive_.sin,
        negative_.combined,
        negative_.inverse,
        negative_.cos,
        negative_.sin);
    graph_ = ggml_new_graph_custom(ctx_, 262144, false);
    for (auto * output : {out_.video_logits.tensor, out_.audio_logits.tensor, out_.hidden.tensor}) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph_, output);
    }
    if (include_sampler_update_) {
        for (auto * output : {out_.next_video.tensor, out_.next_audio.tensor}) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
    }
    input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_, weights_.execution.backend());
    if (input_buffer_ == nullptr) {
        throw std::runtime_error("failed to allocate MiniMax-H3 CFG DiT inputs");
    }
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_.execution.backend()));
    if (gallocr_ == nullptr ||
        !ggml_gallocr_reserve(gallocr_, graph_) ||
        !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("failed to allocate MiniMax-H3 CFG DiT graph");
    }
    core::prepare_host_graph_plan(weights_.execution, graph_, plan_);
    engine::debug::timing_log_scalar("minimax_h3.dit.cfg.graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
    core::write_tensor_f32(positive_.prompt, positive_prompt);
    core::write_tensor_f32(negative_.prompt, negative_prompt);
    std::vector<float> positive_audio_mask(static_cast<size_t>(positive_.layout.total), 0.0F);
    std::vector<float> positive_video_mask(static_cast<size_t>(positive_.layout.total), 0.0F);
    std::vector<float> negative_audio_mask(static_cast<size_t>(negative_.layout.total), 0.0F);
    std::vector<float> negative_video_mask(static_cast<size_t>(negative_.layout.total), 0.0F);
    for (int64_t i = 0; i < positive_.layout.audio_rows; ++i) {
        positive_audio_mask[static_cast<size_t>(positive_.layout.audio_start + i)] = 1.0F;
    }
    for (int64_t i = 0; i < positive_.layout.video_rows; ++i) {
        positive_video_mask[static_cast<size_t>(positive_.layout.video_start + i)] = 1.0F;
    }
    for (int64_t i = 0; i < negative_.layout.audio_rows; ++i) {
        negative_audio_mask[static_cast<size_t>(negative_.layout.audio_start + i)] = 1.0F;
    }
    for (int64_t i = 0; i < negative_.layout.video_rows; ++i) {
        negative_video_mask[static_cast<size_t>(negative_.layout.video_start + i)] = 1.0F;
    }
    core::write_tensor_f32(positive_.audio_mask, positive_audio_mask);
    core::write_tensor_f32(positive_.video_mask, positive_video_mask);
    core::write_tensor_f32(negative_.audio_mask, negative_audio_mask);
    core::write_tensor_f32(negative_.video_mask, negative_video_mask);
    std::vector<int32_t> positive_text_positions(static_cast<size_t>(positive_.text_len));
    std::vector<int32_t> negative_text_positions(static_cast<size_t>(negative_.text_len));
    for (int64_t i = 0; i < positive_.text_len; ++i) {
        positive_text_positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    for (int64_t i = 0; i < negative_.text_len; ++i) {
        negative_text_positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    core::write_tensor_i32(positive_.text_positions, positive_text_positions);
    core::write_tensor_i32(negative_.text_positions, negative_text_positions);
    core::write_tensor_f32(positive_.cos, h3_rope_half_values(cfg_, positive_.text_len, cfg_.rope_inv_freq_len, true));
    core::write_tensor_f32(positive_.sin, h3_rope_half_values(cfg_, positive_.text_len, cfg_.rope_inv_freq_len, false));
    core::write_tensor_f32(negative_.cos, h3_rope_half_values(cfg_, negative_.text_len, cfg_.rope_inv_freq_len, true));
    core::write_tensor_f32(negative_.sin, h3_rope_half_values(cfg_, negative_.text_len, cfg_.rope_inv_freq_len, false));
    std::vector<float> positive_audio_zeros(static_cast<size_t>(positive_.layout.total * cfg_.audio_latents_dim), 0.0F);
    std::vector<float> negative_audio_zeros(static_cast<size_t>(negative_.layout.total * cfg_.audio_latents_dim), 0.0F);
    std::vector<float> positive_video_zeros(static_cast<size_t>(positive_.layout.total * video_dim), 0.0F);
    std::vector<float> negative_video_zeros(static_cast<size_t>(negative_.layout.total * video_dim), 0.0F);
    core::write_tensor_f32(positive_.audio, positive_audio_zeros);
    core::write_tensor_f32(negative_.audio, negative_audio_zeros);
    core::write_tensor_f32(positive_.video, positive_video_zeros);
    core::write_tensor_f32(negative_.video, negative_video_zeros);
    build_dit_index_pattern(positive_.layout, false, positive_.combined_shared_timestep, positive_.inverse_shared_timestep);
    build_dit_index_pattern(positive_.layout, true, positive_.combined_split_timestep, positive_.inverse_split_timestep);
    build_dit_index_pattern(negative_.layout, false, negative_.combined_shared_timestep, negative_.inverse_shared_timestep);
    build_dit_index_pattern(negative_.layout, true, negative_.combined_split_timestep, negative_.inverse_split_timestep);
}

MiniMaxH3DitCfgGraph::~MiniMaxH3DitCfgGraph() {
    plan_.reset();
    if (graph_ != nullptr) {
        core::release_backend_graph_resources(weights_.execution.backend(), graph_);
    }
    if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
    }
    if (input_buffer_ != nullptr) {
        ggml_backend_buffer_free(input_buffer_);
    }
    if (input_ctx_ != nullptr) {
        ggml_free(input_ctx_);
    }
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

void MiniMaxH3DitCfgGraph::run(
    const std::vector<float> & audio_rows,
    const std::vector<float> & video_rows,
    const float * timestep_values,
    size_t timestep_count,
    float sigma_delta_audio,
    float sigma_delta_video,
    bool split_audio_timestep,
    float guidance_scale,
    bool read_logits,
    bool read_hidden,
    bool read_next_rows,
    DitGraphResult & result) {
    const auto input_start = Clock::now();
    const int64_t video_dim = cfg_.video_latents_dim * 4;
    core::write_tensor_f32_slice(
        positive_.audio,
        static_cast<size_t>(positive_.layout.audio_start * cfg_.audio_latents_dim),
        audio_rows.data(),
        audio_rows.size());
    core::write_tensor_f32_slice(
        negative_.audio,
        static_cast<size_t>(negative_.layout.audio_start * cfg_.audio_latents_dim),
        audio_rows.data(),
        audio_rows.size());
    core::write_tensor_f32_slice(
        positive_.video,
        static_cast<size_t>(positive_.layout.video_start * video_dim),
        video_rows.data(),
        video_rows.size());
    core::write_tensor_f32_slice(
        negative_.video,
        static_cast<size_t>(negative_.layout.video_start * video_dim),
        video_rows.data(),
        video_rows.size());
    core::write_tensor_f32(time_t_, timestep_values, timestep_count);
    core::write_tensor_f32(guidance_scale_t_, &guidance_scale, 1);
    core::write_tensor_f32(sigma_delta_audio_t_, &sigma_delta_audio, 1);
    core::write_tensor_f32(sigma_delta_video_t_, &sigma_delta_video, 1);
    if (!indices_uploaded_ || split_audio_timestep != indices_are_split_) {
        core::write_tensor_i32(
            positive_.combined,
            split_audio_timestep ? positive_.combined_split_timestep : positive_.combined_shared_timestep);
        core::write_tensor_i32(
            positive_.inverse,
            split_audio_timestep ? positive_.inverse_split_timestep : positive_.inverse_shared_timestep);
        core::write_tensor_i32(
            negative_.combined,
            split_audio_timestep ? negative_.combined_split_timestep : negative_.combined_shared_timestep);
        core::write_tensor_i32(
            negative_.inverse,
            split_audio_timestep ? negative_.inverse_split_timestep : negative_.inverse_shared_timestep);
        indices_uploaded_ = true;
        indices_are_split_ = split_audio_timestep;
    }
    input_upload_ms_ += engine::debug::elapsed_ms(input_start, Clock::now());
    const auto compute_start = Clock::now();
    core::set_backend_threads(weights_.execution.backend(), 8);
    const ggml_status status = core::compute_graph(weights_.execution, graph_, plan_, "minimax_h3.dit.cfg");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MiniMax-H3 CFG DiT graph compute failed");
    }
    ggml_backend_synchronize(weights_.execution.backend());
    engine::debug::timing_log_scalar("minimax_h3.dit.cfg.graph_compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    const auto read_start = Clock::now();
    if (read_logits) {
        core::read_tensor_f32_into(out_.video_logits.tensor, result.video);
        core::read_tensor_f32_into(out_.audio_logits.tensor, result.audio);
    }
    if (read_hidden) {
        core::read_tensor_f32_into(out_.hidden.tensor, result.hidden);
    }
    if (read_next_rows) {
        if (!include_sampler_update_) {
            throw std::runtime_error("MiniMax-H3 CFG DiT graph was built without sampler update outputs");
        }
        core::read_tensor_f32_into(out_.next_video.tensor, result.next_video);
        core::read_tensor_f32_into(out_.next_audio.tensor, result.next_audio);
    }
    output_read_ms_ += engine::debug::elapsed_ms(read_start, Clock::now());
}

double MiniMaxH3DitCfgGraph::input_upload_ms() const {
    return input_upload_ms_;
}

double MiniMaxH3DitCfgGraph::output_read_ms() const {
    return output_read_ms_;
}

const MiniMaxH3DitGraph::PackedSequenceLayout & MiniMaxH3DitCfgGraph::layout() const {
    return positive_.layout;
}

ggml_tensor * MiniMaxH3DitCfgGraph::video_state_tensor() const {
    return positive_.video.tensor;
}

ggml_tensor * MiniMaxH3DitCfgGraph::audio_state_tensor() const {
    return positive_.audio.tensor;
}

ggml_tensor * MiniMaxH3DitCfgGraph::video_logits_tensor() const {
    return out_.video_logits.tensor;
}

ggml_tensor * MiniMaxH3DitCfgGraph::audio_logits_tensor() const {
    return out_.audio_logits.tensor;
}

}  // namespace engine::models::minimax_h3
