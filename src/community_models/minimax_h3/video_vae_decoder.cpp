#include "engine/community_models/minimax_h3/video_vae_decoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace engine::models::minimax_h3 {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;
using H3Config = MiniMaxH3Config;

struct VideoVaeGgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue add_scaled_hidden(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & residual,
    const core::TensorValue & update,
    const core::TensorValue & scale) {
    return core::wrap_tensor(
        ggml_add(ctx.ggml, residual.tensor, ggml_mul(ctx.ggml, update.tensor, scale.tensor)),
        residual.shape,
        GGML_TYPE_F32);
}

core::TensorValue f32_view_bhtd(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & src,
    int64_t head_dim,
    int64_t seq,
    int64_t heads,
    size_t qkv_offset) {
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            src.tensor,
            head_dim,
            seq,
            heads,
            1,
            static_cast<size_t>(3 * head_dim * heads) * sizeof(float),
            static_cast<size_t>(3 * head_dim) * sizeof(float),
            static_cast<size_t>(3 * head_dim * heads * seq) * sizeof(float),
            qkv_offset * sizeof(float)),
        core::TensorShape::from_dims({1, heads, seq, head_dim}),
        GGML_TYPE_F32);
}

core::TensorValue f32_view_hnd(
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

int64_t positive_mod(int64_t value, int64_t divisor) {
    const int64_t mod = value % divisor;
    return mod < 0 ? mod + divisor : mod;
}

std::vector<float> unpack_video_rows_to_latents(const H3Config & cfg, const std::vector<float> & rows) {
    static constexpr float kVideoLatentsMean[] = {
        0.858090341091156F, -0.9606591463088989F, 1.0661640167236328F, -0.5090325474739075F,
        -0.2727581858634949F, -1.3675414323806763F, -0.2553254961967468F, -0.26907554268836975F,
        -0.5376840829849243F, -0.0464097298681736F, 0.6657370328903198F, 0.19690127670764923F,
        -0.5460608005523682F, -0.4035342037677765F, -0.23683024942874908F, 0.25928452610969543F,
        -0.30133944749832153F, 0.211341992020607F, -1.1206848621368408F, 0.3581933379173279F,
        -0.04225143790245056F, 0.2604829967021942F, 0.22864092886447906F, 0.7056031823158264F,
    };
    static constexpr float kVideoLatentsStd[] = {
        1.2223774194717407F, 1.2767263650894165F, 1.6831774711608887F, 1.7549455165863037F,
        1.5636216402053833F, 2.194143533706665F, 0.9653137922286987F, 1.0569885969161987F,
        0.841948926448822F, 0.7729952931404114F, 1.8955937623977661F, 0.946841835975647F,
        0.7996809482574463F, 0.44988900423049927F, 0.7197399735450745F, 0.6936293244361877F,
        2.961095094680786F, 2.7694199085235596F, 3.0496184825897217F, 2.1088054180145264F,
        3.276226282119751F, 3.1627357006073F, 2.2816812992095947F, 2.6127843856811523F,
    };
    if (cfg.video_vae_latent_channels != 24 || cfg.video_latents_dim != 24) {
        throw std::runtime_error("MiniMax-H3 video VAE latent statistics require 24 channels");
    }
    std::vector<float> latents(static_cast<size_t>(cfg.video_latents_dim * cfg.video_latent_t * cfg.video_latent_h * cfg.video_latent_w));
    const int64_t ph = cfg.video_latent_h / 2;
    const int64_t pw = cfg.video_latent_w / 2;
    for (int64_t t = 0; t < cfg.video_latent_t; ++t) {
        for (int64_t hp = 0; hp < ph; ++hp) {
            for (int64_t wp = 0; wp < pw; ++wp) {
                const int64_t row = (t * ph + hp) * pw + wp;
                for (int64_t c = 0; c < cfg.video_latents_dim; ++c) {
                    for (int64_t ih = 0; ih < 2; ++ih) {
                        for (int64_t iw = 0; iw < 2; ++iw) {
                            const int64_t src_col = ((c * 2 + ih) * 2 + iw);
                            const int64_t dst = ((c * cfg.video_latent_t + t) * cfg.video_latent_h + hp * 2 + ih) * cfg.video_latent_w + wp * 2 + iw;
                            latents[static_cast<size_t>(dst)] =
                                rows[static_cast<size_t>(row * cfg.video_latents_dim * 4 + src_col)] *
                                    kVideoLatentsStd[static_cast<size_t>(c)] +
                                kVideoLatentsMean[static_cast<size_t>(c)];
                        }
                    }
                }
            }
        }
    }
    return latents;
}

std::vector<float> video_vae_rope_values(
    int64_t heads,
    int64_t tokens,
    int64_t latent_t,
    int64_t latent_h,
    int64_t latent_w,
    bool cosine) {
    constexpr int64_t kAxes = 3;
    constexpr int64_t kRotHalf = 24;
    constexpr int64_t kInvPerAxis = kRotHalf / kAxes;
    constexpr float kTheta = 100.0F;
    constexpr float kAngleScale = 6.2831853071795864769F;
    std::vector<float> out(static_cast<size_t>(heads * tokens * kRotHalf), cosine ? 1.0F : 0.0F);
    for (int64_t t = 0; t < latent_t; ++t) {
        const float pt = 2.0F * ((static_cast<float>(t) + 0.5F) / static_cast<float>(latent_t)) - 1.0F;
        for (int64_t h = 0; h < latent_h; ++h) {
            const float ph = 2.0F * ((static_cast<float>(h) + 0.5F) / static_cast<float>(latent_h)) - 1.0F;
            for (int64_t w = 0; w < latent_w; ++w) {
                const float pw = 2.0F * ((static_cast<float>(w) + 0.5F) / static_cast<float>(latent_w)) - 1.0F;
                const int64_t row = (t * latent_h + h) * latent_w + w;
                const float pos[3] = {pt, ph, pw};
                for (int axis = 0; axis < 3; ++axis) {
                    for (int64_t i = 0; i < kInvPerAxis; ++i) {
                        const float inv = 1.0F / std::pow(kTheta, static_cast<float>(i) / static_cast<float>(kInvPerAxis));
                        const float v = cosine ? std::cos(kAngleScale * pos[axis] * inv) : std::sin(kAngleScale * pos[axis] * inv);
                        for (int64_t head = 0; head < heads; ++head) {
                            out[static_cast<size_t>((head * tokens + row) * kRotHalf + axis * kInvPerAxis + i)] = v;
                        }
                    }
                }
            }
        }
    }
    return out;
}

struct VideoTensor4D {
    int64_t channels = 3;
    int64_t frames = 0;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<float> values;

    float & at(int64_t c, int64_t t, int64_t h, int64_t w) {
        return values[static_cast<size_t>(((c * frames + t) * height + h) * width + w)];
    }
    const float & at(int64_t c, int64_t t, int64_t h, int64_t w) const {
        return values[static_cast<size_t>(((c * frames + t) * height + h) * width + w)];
    }
};

void blend_height_from_previous(const VideoTensor4D & previous, VideoTensor4D & current, int64_t extent) {
    extent = std::min<int64_t>({extent, previous.height, current.height});
    for (int64_t y = 0; y < extent; ++y) {
        const float wgt = static_cast<float>(y) / static_cast<float>(extent);
        const int64_t prev_y = previous.height - extent + y;
        for (int64_t c = 0; c < current.channels; ++c) {
            for (int64_t t = 0; t < current.frames; ++t) {
                for (int64_t x = 0; x < current.width; ++x) {
                    current.at(c, t, y, x) = previous.at(c, t, prev_y, x) * (1.0F - wgt) + current.at(c, t, y, x) * wgt;
                }
            }
        }
    }
}

void blend_width_from_previous(const VideoTensor4D & previous, VideoTensor4D & current, int64_t extent) {
    extent = std::min<int64_t>({extent, previous.width, current.width});
    for (int64_t x = 0; x < extent; ++x) {
        const float wgt = static_cast<float>(x) / static_cast<float>(extent);
        const int64_t prev_x = previous.width - extent + x;
        for (int64_t c = 0; c < current.channels; ++c) {
            for (int64_t t = 0; t < current.frames; ++t) {
                for (int64_t y = 0; y < current.height; ++y) {
                    current.at(c, t, y, x) = previous.at(c, t, y, prev_x) * (1.0F - wgt) + current.at(c, t, y, x) * wgt;
                }
            }
        }
    }
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> split_video_tiles(int64_t input_len, int64_t tile_size, int64_t min_overlap, int64_t ratio) {
    if (tile_size >= input_len) {
        return {{0}, {input_len}};
    }
    int64_t count = (input_len + tile_size - min_overlap - 1) / (tile_size - min_overlap);
    while (tile_size * count - min_overlap * (count - 1) < input_len) {
        ++count;
    }
    std::vector<int64_t> overlaps(static_cast<size_t>(count - 1), min_overlap);
    int64_t remaining = tile_size * count - min_overlap * (count - 1) - input_len;
    for (int64_t i = 0; i < remaining / ratio; ++i) {
        overlaps[static_cast<size_t>(i % (count - 1))] += ratio;
    }
    std::vector<int64_t> starts{0};
    for (int64_t i = 0; i + 1 < count; ++i) {
        starts.push_back(starts.back() + tile_size - overlaps[static_cast<size_t>(i)]);
    }
    return {starts, overlaps};
}

core::TensorValue video_linear(
    core::ModuleBuildContext & ctx,
    const VideoVaeWeightStore & weights,
    const std::string & prefix,
    const core::TensorValue & x,
    int64_t in_features,
    int64_t out_features) {
    modules::LinearWeights params = weights.linear(prefix);
    const ggml_prec precision = GGML_PREC_DEFAULT;
    const bool use_fast_projection = ctx.backend_type == core::BackendType::Cuda && out_features % 4 == 0;
    if (!use_fast_projection) {
        return modules::LinearModule({in_features, out_features, true, precision}).build(ctx, x, params);
    }
    auto projected = modules::FastPackedProjection4Module({in_features, out_features, precision})
                         .build(ctx, x, {params.weight, std::nullopt});
    return core::wrap_tensor(
        ggml_add(ctx.ggml, projected.tensor, params.bias->tensor),
        projected.shape,
        GGML_TYPE_F32);
}

core::TensorValue video_apply_rope(
    core::ModuleBuildContext & ctx,
    const H3Config & cfg,
    const core::TensorValue & x,
    const core::TensorValue & cos,
    const core::TensorValue & sin) {
    const int64_t rot_dim = static_cast<int64_t>(static_cast<float>(cfg.video_vae_head_dim) * cfg.video_vae_rope_dim_ratio);
    const int64_t rot_half = rot_dim / 2;
    auto x1 = modules::SliceModule({2, 0, rot_half}).build(ctx, x);
    auto x2 = modules::SliceModule({2, rot_half, rot_half}).build(ctx, x);
    auto out1 = core::wrap_tensor(
        ggml_sub(ctx.ggml, ggml_mul(ctx.ggml, x1.tensor, cos.tensor), ggml_mul(ctx.ggml, x2.tensor, sin.tensor)),
        x1.shape,
        GGML_TYPE_F32);
    auto out2 = core::wrap_tensor(
        ggml_add(ctx.ggml, ggml_mul(ctx.ggml, x2.tensor, cos.tensor), ggml_mul(ctx.ggml, x1.tensor, sin.tensor)),
        x1.shape,
        GGML_TYPE_F32);
    auto rotated = modules::ConcatModule({2}).build(ctx, out1, out2);
    if (rot_dim == cfg.video_vae_head_dim) {
        return rotated;
    }
    return modules::ConcatModule({2}).build(
        ctx,
        rotated,
        modules::SliceModule({2, rot_dim, cfg.video_vae_head_dim - rot_dim}).build(ctx, x));
}

core::TensorValue build_video_vae_attention(
    core::ModuleBuildContext & ctx,
    const VideoVaeWeightStore & weights,
    const H3Config & cfg,
    const core::TensorValue & x,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    const std::string & prefix) {
    const int64_t seq = x.shape.dims[0];
    auto qkv = video_linear(ctx, weights, prefix + ".to_qkv", x, cfg.video_vae_hidden, cfg.video_vae_hidden * 3);
    qkv = core::wrap_tensor(
        ggml_reshape_4d(ctx.ggml, qkv.tensor, cfg.video_vae_head_dim, 3, cfg.video_vae_heads, seq),
        core::TensorShape::from_dims({seq, cfg.video_vae_heads, 3, cfg.video_vae_head_dim}),
        GGML_TYPE_F32);
    auto q = f32_view_hnd(ctx, qkv, cfg.video_vae_head_dim, seq, cfg.video_vae_heads, 0, static_cast<size_t>(3 * cfg.video_vae_head_dim * cfg.video_vae_heads), static_cast<size_t>(3 * cfg.video_vae_head_dim));
    auto k = f32_view_hnd(ctx, qkv, cfg.video_vae_head_dim, seq, cfg.video_vae_heads, static_cast<size_t>(cfg.video_vae_head_dim), static_cast<size_t>(3 * cfg.video_vae_head_dim * cfg.video_vae_heads), static_cast<size_t>(3 * cfg.video_vae_head_dim));
    auto v = f32_view_bhtd(ctx, qkv, cfg.video_vae_head_dim, seq, cfg.video_vae_heads, static_cast<size_t>(2 * cfg.video_vae_head_dim));
    q = modules::RMSNormModule({cfg.video_vae_head_dim, 1.0e-5F, false, false}).build(ctx, q, {});
    k = modules::RMSNormModule({cfg.video_vae_head_dim, 1.0e-5F, false, false}).build(ctx, k, {});
    q = video_apply_rope(ctx, cfg, q, cos, sin);
    k = video_apply_rope(ctx, cfg, k, cos, sin);
    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({1, cfg.video_vae_heads, seq, cfg.video_vae_head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({1, cfg.video_vae_heads, seq, cfg.video_vae_head_dim}));
    auto h = modules::ScaledDotProductAttentionModule({
        cfg.video_vae_head_dim,
        modules::ScaledDotProductAttentionLowering::FlashPreserveViews,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal}).build(ctx, q, k, v);
    h = core::reshape_tensor(ctx, h, core::TensorShape::from_dims({seq, cfg.video_vae_hidden}));
    return video_linear(ctx, weights, prefix + ".to_out", h, cfg.video_vae_hidden, cfg.video_vae_hidden);
}

core::TensorValue build_video_vae_decoder_graph(
    core::ModuleBuildContext & ctx,
    const VideoVaeWeightStore & weights,
    const H3Config & cfg,
    const core::TensorValue & latents,
    const core::TensorValue & cos,
    const core::TensorValue & sin) {
    const int64_t patch_tokens = latents.shape.dims[0];
    auto z = video_linear(ctx, weights, "post_quant_conv", latents, cfg.video_vae_latent_channels, cfg.video_vae_latent_channels);
    auto hidden = video_linear(ctx, weights, "decoder.x_embedder", z, cfg.video_vae_latent_channels, cfg.video_vae_hidden);
    auto register_tokens = weights.require("decoder.register_tokens");
    hidden = modules::ConcatModule({0}).build(ctx, hidden, register_tokens);
    auto zero = core::wrap_tensor(
        ggml_scale(ctx.ggml, modules::SliceModule({0, 0, 1}).build(ctx, hidden).tensor, 0.0F),
        core::TensorShape::from_dims({1, cfg.video_vae_hidden}),
        GGML_TYPE_F32);
    hidden = modules::ConcatModule({0}).build(ctx, hidden, zero);
    for (int64_t layer = 0; layer < cfg.video_vae_layers; ++layer) {
        const std::string p = "decoder.transformer_blocks." + std::to_string(layer) + ".";
        auto h = modules::RMSNormModule({cfg.video_vae_hidden, 1.0e-5F, true, false}).build(
            ctx,
            hidden,
            {weights.require(p + "norm1.weight"), std::nullopt});
        h = build_video_vae_attention(ctx, weights, cfg, h, cos, sin, p + "attn");
        hidden = add_scaled_hidden(ctx, hidden, h, weights.require(p + "scale1"));
        h = modules::RMSNormModule({cfg.video_vae_hidden, 1.0e-5F, true, false}).build(
            ctx,
            hidden,
            {weights.require(p + "norm2.weight"), std::nullopt});
        auto ff = video_linear(ctx, weights, p + "ff.w1", h, cfg.video_vae_hidden, cfg.video_vae_hidden * 8);
        auto gate = modules::SliceModule({1, 0, cfg.video_vae_hidden * 4}).build(ctx, ff);
        auto up = modules::SliceModule({1, cfg.video_vae_hidden * 4, cfg.video_vae_hidden * 4}).build(ctx, ff);
        auto gated = core::wrap_tensor(ggml_swiglu_split(ctx.ggml, gate.tensor, up.tensor), gate.shape, gate.type);
        ff = video_linear(
            ctx,
            weights,
            p + "ff.w2",
            gated,
            cfg.video_vae_hidden * 4,
            cfg.video_vae_hidden);
        hidden = add_scaled_hidden(ctx, hidden, ff, weights.require(p + "scale2"));
    }
    hidden = modules::LayerNormModule({cfg.video_vae_hidden, 1.0e-5F, true, true}).build(
        ctx,
        hidden,
        {weights.require("decoder.norm_out.weight"), weights.require("decoder.norm_out.bias")});
    auto output = video_linear(ctx, weights, "decoder.proj_out", hidden, cfg.video_vae_hidden, 3 * cfg.video_vae_patch_size_t * cfg.video_vae_patch_size * cfg.video_vae_patch_size);
    return modules::SliceModule({0, 0, patch_tokens}).build(ctx, output);
}

}  // namespace

class VideoVaeTileGraph {
public:
    VideoVaeTileGraph(VideoVaeWeightStore & weights, const H3Config & cfg, int64_t latent_t, int64_t latent_h, int64_t latent_w)
        : weights_(weights),
          cfg_(cfg),
          latent_t_(latent_t),
          latent_h_(latent_h),
          latent_w_(latent_w) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 video VAE graph context");
        }
        input_ctx_.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 video VAE input context");
        }
        const auto build_start = Clock::now();
        const int64_t tokens = latent_t_ * latent_h_ * latent_w_;
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.video_vae.inputs", weights_.execution.backend_type()};
        latent_tvalue_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({tokens, cfg_.video_vae_latent_channels}));
        cos_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({cfg_.video_vae_heads, tokens + cfg_.video_vae_register_tokens + 1, 24}));
        sin_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({cfg_.video_vae_heads, tokens + cfg_.video_vae_register_tokens + 1, 24}));
        ggml_set_input(latent_tvalue_.tensor);
        ggml_set_input(cos_t_.tensor);
        ggml_set_input(sin_t_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.video_vae_decode", weights_.execution.backend_type()};
        output_ = build_video_vae_decoder_graph(build_ctx, weights_, cfg_, latent_tvalue_, cos_t_, sin_t_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), weights_.execution.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-H3 video VAE inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_.execution.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 video VAE graph");
        }
        core::prepare_host_graph_plan(weights_.execution, graph_, plan_);
        core::write_tensor_f32(
            cos_t_,
            video_vae_rope_values(cfg_.video_vae_heads, tokens + cfg_.video_vae_register_tokens + 1, latent_t_, latent_h_, latent_w_, true));
        core::write_tensor_f32(
            sin_t_,
            video_vae_rope_values(cfg_.video_vae_heads, tokens + cfg_.video_vae_register_tokens + 1, latent_t_, latent_h_, latent_w_, false));
        engine::debug::timing_log_scalar("minimax_h3.video_vae.graph_build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~VideoVaeTileGraph() {
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
    }

    VideoTensor4D run(const std::vector<float> & latents) {
        const int64_t tokens = latent_t_ * latent_h_ * latent_w_;
        if (static_cast<int64_t>(latents.size()) != tokens * cfg_.video_vae_latent_channels) {
            throw std::runtime_error("MiniMax-H3 video VAE tile latent size mismatch");
        }
        const auto input_start = Clock::now();
        core::write_tensor_f32(latent_tvalue_, latents);
        engine::debug::timing_log_scalar("minimax_h3.video_vae.input_upload_ms", engine::debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        core::set_backend_threads(weights_.execution.backend(), 8);
        const ggml_status status = core::compute_graph(weights_.execution, graph_, plan_, "minimax_h3.video_vae");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 video VAE graph compute failed");
        }
        ggml_backend_synchronize(weights_.execution.backend());
        engine::debug::timing_log_scalar("minimax_h3.video_vae.graph_compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
        const auto read_start = Clock::now();
        auto packed = core::read_tensor_f32(output_.tensor);
        engine::debug::timing_log_scalar("minimax_h3.video_vae.output_read_ms", engine::debug::elapsed_ms(read_start, Clock::now()));
        VideoTensor4D out;
        out.frames = latent_t_ * cfg_.video_vae_patch_size_t;
        out.height = latent_h_ * cfg_.video_vae_patch_size;
        out.width = latent_w_ * cfg_.video_vae_patch_size;
        out.values.resize(static_cast<size_t>(out.channels * out.frames * out.height * out.width));
        for (int64_t tp = 0; tp < latent_t_; ++tp) {
            for (int64_t hp = 0; hp < latent_h_; ++hp) {
                for (int64_t wp = 0; wp < latent_w_; ++wp) {
                    const int64_t row = (tp * latent_h_ + hp) * latent_w_ + wp;
                    for (int64_t c = 0; c < 3; ++c) {
                        for (int64_t dt = 0; dt < cfg_.video_vae_patch_size_t; ++dt) {
                            for (int64_t dh = 0; dh < cfg_.video_vae_patch_size; ++dh) {
                                for (int64_t dw = 0; dw < cfg_.video_vae_patch_size; ++dw) {
                                    const int64_t col = (((c * cfg_.video_vae_patch_size_t + dt) * cfg_.video_vae_patch_size + dh) * cfg_.video_vae_patch_size + dw);
                                    out.at(c, tp * cfg_.video_vae_patch_size_t + dt, hp * cfg_.video_vae_patch_size + dh, wp * cfg_.video_vae_patch_size + dw) =
                                        packed[static_cast<size_t>(row * 3 * cfg_.video_vae_patch_size_t * cfg_.video_vae_patch_size * cfg_.video_vae_patch_size + col)];
                                }
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

private:
    VideoVaeWeightStore & weights_;
    const H3Config cfg_;
    int64_t latent_t_ = 0;
    int64_t latent_h_ = 0;
    int64_t latent_w_ = 0;
    std::unique_ptr<ggml_context, VideoVaeGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, VideoVaeGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue latent_tvalue_;
    core::TensorValue cos_t_;
    core::TensorValue sin_t_;
    core::TensorValue output_;
};

namespace {

std::vector<float> gather_video_tile_latents(
    const std::vector<float> & latents,
    const H3Config & cfg,
    int64_t t_start,
    int64_t t_len,
    int64_t h_start,
    int64_t h_len,
    int64_t w_start,
    int64_t w_len) {
    std::vector<float> out(static_cast<size_t>(t_len * h_len * w_len * cfg.video_vae_latent_channels));
    for (int64_t t = 0; t < t_len; ++t) {
        for (int64_t h = 0; h < h_len; ++h) {
            for (int64_t w = 0; w < w_len; ++w) {
                const int64_t row = (t * h_len + h) * w_len + w;
                for (int64_t c = 0; c < cfg.video_vae_latent_channels; ++c) {
                    const int64_t src = ((c * cfg.video_latent_t + t_start + t) * cfg.video_latent_h + h_start + h) * cfg.video_latent_w + w_start + w;
                    out[static_cast<size_t>(row * cfg.video_vae_latent_channels + c)] = latents[static_cast<size_t>(src)];
                }
            }
        }
    }
    return out;
}

VideoTensor4D decode_video_spatial_tile_set(
    VideoVaeWeightStore & weights,
    const H3Config & cfg,
    VideoVaeDecodeCache & cache,
    const std::vector<float> & latents,
    int64_t t_start,
    int64_t t_len) {
    const auto [y_starts, y_overlaps] = split_video_tiles(cfg.height, cfg.video_vae_tile_size, cfg.video_vae_tile_overlap, cfg.video_vae_patch_size);
    const auto [x_starts, x_overlaps] = split_video_tiles(cfg.width, cfg.video_vae_tile_size, cfg.video_vae_tile_overlap, cfg.video_vae_patch_size);
    VideoTensor4D out;
    out.frames = t_len * cfg.video_vae_patch_size_t;
    out.height = cfg.height;
    out.width = cfg.width;
    out.values.resize(static_cast<size_t>(out.channels * out.frames * out.height * out.width));
    std::vector<VideoTensor4D> previous_raw_row;
    previous_raw_row.reserve(x_starts.size());
    int64_t y_out = 0;
    for (size_t yi = 0; yi < y_starts.size(); ++yi) {
        std::vector<VideoTensor4D> current_raw_row;
        current_raw_row.reserve(x_starts.size());
        int64_t x_out = 0;
        for (size_t xi = 0; xi < x_starts.size(); ++xi) {
            const int64_t h_start = y_starts[yi] / cfg.video_vae_patch_size;
            const int64_t w_start = x_starts[xi] / cfg.video_vae_patch_size;
            const int64_t h_len = std::min<int64_t>(cfg.video_vae_tile_size, cfg.height - y_starts[yi]) / cfg.video_vae_patch_size;
            const int64_t w_len = std::min<int64_t>(cfg.video_vae_tile_size, cfg.width - x_starts[xi]) / cfg.video_vae_patch_size;
            auto raw = cache.graph(weights, cfg, t_len, h_len, w_len)
                           .run(gather_video_tile_latents(latents, cfg, t_start, t_len, h_start, h_len, w_start, w_len));
            auto tile = raw;
            if (yi > 0) {
                blend_height_from_previous(previous_raw_row[xi], tile, y_overlaps[yi - 1]);
            }
            if (xi > 0) {
                blend_width_from_previous(current_raw_row[xi - 1], tile, x_overlaps[xi - 1]);
            }
            const int64_t keep_h = yi + 1 < y_starts.size() ? tile.height - y_overlaps[yi] : tile.height;
            const int64_t keep_w = xi + 1 < x_starts.size() ? tile.width - x_overlaps[xi] : tile.width;
            for (int64_t c = 0; c < tile.channels; ++c) {
                for (int64_t t = 0; t < tile.frames; ++t) {
                    for (int64_t h = 0; h < keep_h; ++h) {
                        for (int64_t w = 0; w < keep_w; ++w) {
                            out.at(c, t, y_out + h, x_out + w) = tile.at(c, t, h, w);
                        }
                    }
                }
            }
            x_out += keep_w;
            current_raw_row.push_back(std::move(raw));
        }
        y_out += yi + 1 < y_starts.size() ? current_raw_row.front().height - y_overlaps[yi] : current_raw_row.front().height;
        previous_raw_row = std::move(current_raw_row);
    }
    return out;
}

MiniMaxH3VideoFrames run_video_vae_decode_graph_impl(
    VideoVaeWeightStore & weights,
    const H3Config & cfg,
    const std::vector<float> & video_rows,
    VideoVaeDecodeCache & cache) {
    const auto decode_start = Clock::now();
    const auto latents = unpack_video_rows_to_latents(cfg, video_rows);
    const int64_t pseudo_total_base = cfg.video_latent_t + cfg.video_vae_token_drop;
    const int64_t tokens_chunk = (cfg.video_vae_clip_length + cfg.video_vae_patch_size_t - 1) / cfg.video_vae_patch_size_t;
    const int64_t token_overlap = positive_mod(-cfg.video_vae_token_drop, tokens_chunk);
    const int64_t frame_pre_padding = positive_mod(-cfg.video_vae_clip_length, cfg.video_vae_patch_size_t);
    const int64_t frame_overlap = std::max<int64_t>(token_overlap * cfg.video_vae_patch_size_t - frame_pre_padding, 0);
    const int64_t pad_tokens = (tokens_chunk - pseudo_total_base % tokens_chunk) % tokens_chunk;
    const int64_t pseudo_total = pseudo_total_base + pad_tokens;
    const int64_t chunks = pseudo_total / tokens_chunk - (cfg.video_vae_token_drop > 0 ? 1 : 0);
    VideoTensor4D video;
    video.frames = cfg.num_frames;
    video.height = cfg.height;
    video.width = cfg.width;
    video.values.resize(static_cast<size_t>(video.channels * video.frames * video.height * video.width));
    std::optional<VideoTensor4D> overlap;
    int64_t output_t = 0;
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int64_t t_start = chunk * tokens_chunk;
        const int64_t t_len = std::min<int64_t>(tokens_chunk + token_overlap, cfg.video_latent_t - std::min<int64_t>(t_start, cfg.video_latent_t));
        auto decoded = decode_video_spatial_tile_set(weights, cfg, cache, latents, t_start, t_len);
        const int64_t chunk_dec = tokens_chunk * cfg.video_vae_patch_size_t;
        const int64_t split_count = cfg.video_vae_token_drop > 0 ? 2 : 1;
        for (int64_t split = 0; split < split_count; ++split) {
            const int64_t frame_start = split * chunk_dec;
            const int64_t available = std::max<int64_t>(0, std::min(frame_start + chunk_dec, decoded.frames) - frame_start - frame_pre_padding);
            if (available <= 0) {
                continue;
            }
            if (split == 0) {
                const int64_t blend_extent = overlap.has_value() ? std::min<int64_t>({frame_overlap, overlap->frames, available}) : 0;
                for (int64_t c = 0; c < decoded.channels; ++c) {
                    for (int64_t t = 0; t < available && output_t + t < video.frames; ++t) {
                        const int64_t src_t = frame_start + frame_pre_padding + t;
                        const float wgt = blend_extent > 0 && t < blend_extent ? static_cast<float>(t) / static_cast<float>(blend_extent) : 1.0F;
                        const int64_t prev_t = blend_extent > 0 && t < blend_extent ? overlap->frames - blend_extent + t : 0;
                        for (int64_t h = 0; h < decoded.height; ++h) {
                            for (int64_t w = 0; w < decoded.width; ++w) {
                                float value = decoded.at(c, src_t, h, w);
                                if (t < blend_extent) {
                                    value = overlap->at(c, prev_t, h, w) * (1.0F - wgt) + value * wgt;
                                }
                                video.at(c, output_t + t, h, w) = value;
                            }
                        }
                    }
                }
                const int64_t remaining = std::max<int64_t>(0, video.frames - output_t);
                output_t += std::min<int64_t>(available, remaining);
                overlap.reset();
            } else {
                VideoTensor4D next_overlap;
                next_overlap.frames = available;
                next_overlap.height = decoded.height;
                next_overlap.width = decoded.width;
                next_overlap.values.resize(static_cast<size_t>(next_overlap.channels * next_overlap.frames * next_overlap.height * next_overlap.width));
                for (int64_t c = 0; c < next_overlap.channels; ++c) {
                    for (int64_t t = 0; t < next_overlap.frames; ++t) {
                        const int64_t src_t = frame_start + frame_pre_padding + t;
                        for (int64_t h = 0; h < next_overlap.height; ++h) {
                            for (int64_t w = 0; w < next_overlap.width; ++w) {
                                next_overlap.at(c, t, h, w) = decoded.at(c, src_t, h, w);
                            }
                        }
                    }
                }
                overlap = std::move(next_overlap);
            }
        }
        if (chunk + 1 == chunks && overlap.has_value()) {
            for (int64_t c = 0; c < overlap->channels; ++c) {
                for (int64_t t = 0; t < overlap->frames && output_t + t < video.frames; ++t) {
                    for (int64_t h = 0; h < overlap->height; ++h) {
                        for (int64_t w = 0; w < overlap->width; ++w) {
                            video.at(c, output_t + t, h, w) = overlap->at(c, t, h, w);
                        }
                    }
                }
            }
            const int64_t remaining = std::max<int64_t>(0, video.frames - output_t);
            output_t += std::min<int64_t>(overlap->frames, remaining);
            overlap.reset();
        }
    }
    MiniMaxH3VideoFrames out;
    out.width = static_cast<int>(video.width);
    out.height = static_cast<int>(video.height);
    out.frames = static_cast<int>(video.frames);
    out.fps = 24;
    out.rgb24.resize(static_cast<size_t>(video.frames * video.height * video.width * 3));
    constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
    constexpr float stdv[3] = {0.229F, 0.224F, 0.225F};
    for (int64_t t = 0; t < video.frames; ++t) {
        for (int64_t h = 0; h < video.height; ++h) {
            for (int64_t w = 0; w < video.width; ++w) {
                for (int64_t c = 0; c < 3; ++c) {
                    const float v = std::clamp(video.at(c, t, h, w) * stdv[c] + mean[c], 0.0F, 1.0F);
                    out.rgb24[static_cast<size_t>(((t * video.height + h) * video.width + w) * 3 + c)] =
                        static_cast<std::byte>(std::clamp<int>(static_cast<int>(std::lround(v * 255.0F)), 0, 255));
                }
            }
        }
    }
    engine::debug::timing_log_scalar("minimax_h3.video_decode_ms", engine::debug::elapsed_ms(decode_start, Clock::now()));
    return out;
}



}  // namespace

struct VideoVaeDecodeCache::Impl {
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::unique_ptr<VideoVaeTileGraph>> graphs;
};

VideoVaeWeightStore::VideoVaeWeightStore(
    core::ExecutionContext & execution_context,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    const MiniMaxH3Config & cfg,
    size_t weight_context_bytes)
    : execution(execution_context),
      source_(std::move(tensor_source)),
      store_(execution.backend(), execution.backend_type(), "minimax_h3.video_vae", weight_context_bytes) {
    if (source_ == nullptr) {
        throw std::runtime_error("MiniMax-H3 video VAE tensor source is missing");
    }
    auto load = [&](const std::string & name, assets::TensorStorageType storage, const std::vector<int64_t> & shape) {
        weights_.emplace(name, store_.load_tensor(*source_, name, storage, shape));
    };
    auto load_as = [&](const std::string & name, assets::TensorStorageType storage, const std::vector<int64_t> & source_shape, const core::TensorShape & shape) {
        weights_.emplace(name, store_.load_tensor_as_shape(*source_, name, storage, source_shape, shape));
    };
    auto load_linear = [&](const std::string & prefix, int64_t out_features, int64_t in_features) {
        modules::LinearWeights linear;
        linear.weight = store_.load_tensor(
            *source_,
            prefix + ".weight",
            assets::TensorStorageType::Native,
            {out_features, in_features});
        linear.bias = store_.load_tensor(
            *source_,
            prefix + ".bias",
            assets::TensorStorageType::F32,
            {out_features});
        linear_weights_.emplace(prefix, std::move(linear));
    };
    load_linear("post_quant_conv", cfg.video_vae_latent_channels, cfg.video_vae_latent_channels);
    load_linear("decoder.x_embedder", cfg.video_vae_hidden, cfg.video_vae_latent_channels);
    load_as("decoder.register_tokens", assets::TensorStorageType::F32, {1, cfg.video_vae_register_tokens, cfg.video_vae_hidden}, core::TensorShape::from_dims({cfg.video_vae_register_tokens, cfg.video_vae_hidden}));
    for (int64_t layer = 0; layer < cfg.video_vae_layers; ++layer) {
        const std::string p = "decoder.transformer_blocks." + std::to_string(layer) + ".";
        load(p + "norm1.weight", assets::TensorStorageType::Native, {cfg.video_vae_hidden});
        load(p + "norm2.weight", assets::TensorStorageType::Native, {cfg.video_vae_hidden});
        load(p + "scale1", assets::TensorStorageType::F32, {cfg.video_vae_hidden});
        load(p + "scale2", assets::TensorStorageType::F32, {cfg.video_vae_hidden});
        load_linear(p + "attn.to_qkv", cfg.video_vae_hidden * 3, cfg.video_vae_hidden);
        load_linear(p + "attn.to_out", cfg.video_vae_hidden, cfg.video_vae_hidden);
        load_linear(p + "ff.w1", cfg.video_vae_hidden * 8, cfg.video_vae_hidden);
        load_linear(p + "ff.w2", cfg.video_vae_hidden, cfg.video_vae_hidden * 4);
    }
    load("decoder.norm_out.weight", assets::TensorStorageType::Native, {cfg.video_vae_hidden});
    load("decoder.norm_out.bias", assets::TensorStorageType::F32, {cfg.video_vae_hidden});
    load_linear(
        "decoder.proj_out",
        3 * cfg.video_vae_patch_size_t * cfg.video_vae_patch_size * cfg.video_vae_patch_size,
        cfg.video_vae_hidden);
    store_.upload();
    source_->release_storage();
}

const core::TensorValue & VideoVaeWeightStore::require(std::string_view name) const {
    const auto it = weights_.find(std::string(name));
    if (it == weights_.end()) {
        throw std::runtime_error("missing MiniMax-H3 video VAE tensor: " + std::string(name));
    }
    return it->second;
}

const modules::LinearWeights & VideoVaeWeightStore::linear(std::string_view name) const {
    const auto it = linear_weights_.find(std::string(name));
    if (it == linear_weights_.end()) {
        throw std::runtime_error("missing MiniMax-H3 video VAE linear weights: " + std::string(name));
    }
    return it->second;
}

VideoVaeDecodeCache::VideoVaeDecodeCache()
    : impl_(std::make_unique<Impl>()) {
}

VideoVaeDecodeCache::~VideoVaeDecodeCache() = default;

VideoVaeTileGraph & VideoVaeDecodeCache::graph(
    VideoVaeWeightStore & weights,
    const MiniMaxH3Config & cfg,
    int64_t latent_t,
    int64_t latent_h,
    int64_t latent_w) {
    const auto key = std::make_tuple(latent_t, latent_h, latent_w);
    auto & cached = impl_->graphs[key];
    if (cached == nullptr) {
        cached = std::make_unique<VideoVaeTileGraph>(weights, cfg, latent_t, latent_h, latent_w);
    }
    return *cached;
}

MiniMaxH3VideoFrames run_video_vae_decode_graph(
    VideoVaeWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & video_rows,
    VideoVaeDecodeCache & cache) {
    return run_video_vae_decode_graph_impl(weights, cfg, video_rows, cache);
}

}  // namespace engine::models::minimax_h3
