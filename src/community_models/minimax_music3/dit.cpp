#include "engine/community_models/minimax_music3/dit.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/sampling/diffusion_math.h"

#include <ggml.h>
#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::minimax_music3 {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;

constexpr int64_t kBatch = 2;  // conditional + unconditional CFG branches

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct DitBlockWeights {
    core::TensorValue norm1_weight;
    core::TensorValue norm1_bias;
    core::TensorValue to_q;
    core::TensorValue to_k;
    core::TensorValue to_v;
    core::TensorValue to_out;
    core::TensorValue norm2_weight;
    core::TensorValue norm2_bias;
    core::TensorValue ff_in_weight;
    core::TensorValue ff_in_bias;
    core::TensorValue ff_out_weight;
    core::TensorValue ff_out_bias;
};

ggml_tensor * layer_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight), bias);
}

}  // namespace

struct MiniMaxMusic3DitRuntime::Impl {
    core::ExecutionContext & execution;
    MiniMaxMusic3Config config;
    core::BackendWeightStore store;

    core::TensorValue preprocess_conv;
    core::TensorValue proj_in;
    core::TensorValue time_proj;
    core::TensorValue time_linear1_weight;
    core::TensorValue time_linear1_bias;
    core::TensorValue time_linear2_weight;
    core::TensorValue time_linear2_bias;
    std::vector<DitBlockWeights> blocks;
    core::TensorValue proj_out;
    core::TensorValue postprocess_conv;

    // Per-chunk graph state.
    int64_t chunk_length = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_tensor * in_latent = nullptr;     // f32 [length, in_channels] (channel-major rows)
    ggml_tensor * in_condition = nullptr;  // f32 [length, condition_dim, batch]
    ggml_tensor * in_time = nullptr;       // f32 [1]
    ggml_tensor * in_positions = nullptr;  // i32 [length + 1]
    ggml_tensor * out_velocity = nullptr;  // f32 [length, in_channels, batch]
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;

    Impl(
        core::ExecutionContext & execution_context,
        const assets::TensorSource & source,
        const MiniMaxMusic3Config & cfg,
        size_t weight_context_bytes)
        : execution(execution_context),
          config(cfg),
          store(execution.backend(), execution.backend_type(), "minimax_music3.dit", weight_context_bytes) {
        const int64_t inner = config.dit_heads * config.dit_head_dim;
        const int64_t concat_channels = 2 * config.dit_in_channels + config.dit_condition_dim;
        const auto native = assets::TensorStorageType::Native;
        preprocess_conv = store.load_tensor(source, "preprocess_conv.weight", native, {concat_channels, concat_channels, 1});
        proj_in = store.load_tensor(source, "proj_in.weight", native, {inner, concat_channels});
        time_proj = store.load_tensor(source, "time_proj.weight", assets::TensorStorageType::F32, {config.dit_fourier_dim / 2, 1});
        time_linear1_weight = store.load_tensor(source, "time_embed.linear_1.weight", native, {inner, config.dit_fourier_dim});
        time_linear1_bias = store.load_tensor(source, "time_embed.linear_1.bias", assets::TensorStorageType::F32, {inner});
        time_linear2_weight = store.load_tensor(source, "time_embed.linear_2.weight", native, {inner, inner});
        time_linear2_bias = store.load_tensor(source, "time_embed.linear_2.bias", assets::TensorStorageType::F32, {inner});
        blocks.reserve(static_cast<size_t>(config.dit_layers));
        for (int64_t layer = 0; layer < config.dit_layers; ++layer) {
            const std::string prefix = "transformer_blocks." + std::to_string(layer) + ".";
            DitBlockWeights out;
            out.norm1_weight = store.load_tensor(source, prefix + "norm1.weight", assets::TensorStorageType::F32, {inner});
            out.norm1_bias = store.load_tensor(source, prefix + "norm1.bias", assets::TensorStorageType::F32, {inner});
            out.to_q = store.load_tensor(source, prefix + "attn.to_q.weight", native, {inner, inner});
            out.to_k = store.load_tensor(source, prefix + "attn.to_k.weight", native, {inner, inner});
            out.to_v = store.load_tensor(source, prefix + "attn.to_v.weight", native, {inner, inner});
            out.to_out = store.load_tensor(source, prefix + "attn.to_out.0.weight", native, {inner, inner});
            out.norm2_weight = store.load_tensor(source, prefix + "norm2.weight", assets::TensorStorageType::F32, {inner});
            out.norm2_bias = store.load_tensor(source, prefix + "norm2.bias", assets::TensorStorageType::F32, {inner});
            out.ff_in_weight = store.load_tensor(source, prefix + "ff_in.weight", native, {2 * config.dit_ff_inner, inner});
            out.ff_in_bias = store.load_tensor(source, prefix + "ff_in.bias", assets::TensorStorageType::F32, {2 * config.dit_ff_inner});
            out.ff_out_weight = store.load_tensor(source, prefix + "ff_out.weight", native, {inner, config.dit_ff_inner});
            out.ff_out_bias = store.load_tensor(source, prefix + "ff_out.bias", assets::TensorStorageType::F32, {inner});
            blocks.push_back(std::move(out));
        }
        proj_out = store.load_tensor(source, "proj_out.weight", native, {config.dit_in_channels, inner});
        postprocess_conv = store.load_tensor(source, "postprocess_conv.weight", native, {config.dit_in_channels, config.dit_in_channels, 1});
        store.upload();
        source.release_storage();
    }

    ~Impl() {
        release_chunk();
    }

    void release_chunk() {
        plan.reset();
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph);
            graph = nullptr;
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
        }
        graph_ctx.reset();
        input_ctx.reset();
        chunk_length = 0;
    }

    void build_chunk_graph(int64_t length) {
        release_chunk();
        const int64_t in_channels = config.dit_in_channels;
        const int64_t cond_dim = config.dit_condition_dim;
        const int64_t inner = config.dit_heads * config.dit_head_dim;
        const int64_t heads = config.dit_heads;
        const int64_t head_dim = config.dit_head_dim;
        const int64_t seq = length + 1;  // time token at position 0
        const int64_t tokens = kBatch * seq;
        const int64_t concat_channels = 2 * in_channels + cond_dim;

        graph_ctx.reset(ggml_init({1536 * 1024 * 1024, nullptr, true}));
        input_ctx.reset(ggml_init({8 * 1024 * 1024, nullptr, true}));
        if (graph_ctx == nullptr || input_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-Music3 DiT graph context");
        }
        ggml_context * ictx = input_ctx.get();
        ggml_context * ctx = graph_ctx.get();

        in_latent = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, length, in_channels);
        in_condition = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, length, cond_dim, kBatch);
        in_time = ggml_new_tensor_1d(ictx, GGML_TYPE_F32, 1);
        in_positions = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, seq);
        for (ggml_tensor * tensor : {in_latent, in_condition, in_time}) {
            ggml_set_input(tensor);
        }
        ggml_set_input(in_positions);

        // Assemble [latent, zeros, condition] channels for both batches.
        ggml_tensor * latent3 = ggml_reshape_3d(ctx, in_latent, length, in_channels, 1);
        ggml_tensor * latent_pair = ggml_concat(ctx, latent3, latent3, 2);  // [length, in_ch, 2]
        ggml_tensor * zeros_pair = ggml_scale(ctx, latent_pair, 0.0F);
        ggml_tensor * xc = ggml_concat(ctx, ggml_concat(ctx, latent_pair, zeros_pair, 1), in_condition, 1);
        // Residual 1x1 conv over channels: as a linear over the channel dim.
        {
            ggml_tensor * pre_w = ggml_reshape_2d(ctx, preprocess_conv.tensor, concat_channels, concat_channels);
            ggml_tensor * x_t = ggml_cont(ctx, ggml_permute(ctx, xc, 1, 0, 2, 3));  // [C, length, 2]
            ggml_tensor * pre = ggml_mul_mat(ctx, pre_w, x_t);
            xc = ggml_add(ctx, pre, x_t);  // [C, length, 2] token-major now
        }
        // Activations are cast to F16 ahead of the large matmuls to match the F16 weight
        // storage (uniform half gemms); norms and residuals stay F32.
        const bool cast_activations = execution.backend_type() == core::BackendType::Cuda;
        auto gemm_input = [&](ggml_tensor * t) {
            return cast_activations ? ggml_cast(ctx, t, GGML_TYPE_F16) : t;
        };
        ggml_tensor * x = ggml_reshape_2d(ctx, xc, concat_channels, kBatch * length);
        x = ggml_mul_mat(ctx, proj_in.tensor, gemm_input(x));  // [inner, batch * length]

        // Time embedding token.
        ggml_tensor * angles = ggml_scale(
            ctx,
            ggml_mul_mat(ctx, ggml_reshape_2d(ctx, time_proj.tensor, 1, config.dit_fourier_dim / 2), ggml_reshape_2d(ctx, in_time, 1, 1)),
            2.0F * static_cast<float>(M_PI));  // [fourier/2, 1]
        ggml_tensor * fourier = ggml_concat(ctx, ggml_cos(ctx, angles), ggml_sin(ctx, angles), 0);  // [fourier, 1]
        ggml_tensor * temb = ggml_add(
            ctx,
            ggml_mul_mat(ctx, time_linear1_weight.tensor, fourier),
            time_linear1_bias.tensor);
        temb = ggml_silu(ctx, temb);
        temb = ggml_add(ctx, ggml_mul_mat(ctx, time_linear2_weight.tensor, temb), time_linear2_bias.tensor);  // [inner, 1]

        // Prepend the time token per batch: tokens are batch-major [b][s].
        ggml_tensor * x3 = ggml_reshape_3d(ctx, x, inner, length, kBatch);
        ggml_tensor * temb3 = ggml_reshape_3d(ctx, temb, inner, 1, 1);
        ggml_tensor * temb_pair = ggml_concat(ctx, temb3, temb3, 2);  // [inner, 1, 2]
        x3 = ggml_concat(ctx, temb_pair, x3, 1);  // [inner, seq, 2]
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, x3), inner, tokens);

        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        for (const auto & block : blocks) {
            ggml_tensor * h = layer_norm(ctx, x, block.norm1_weight.tensor, block.norm1_bias.tensor, 1.0e-5F);
            h = gemm_input(h);
            ggml_tensor * q = ggml_mul_mat(ctx, block.to_q.tensor, h);
            ggml_tensor * k = ggml_mul_mat(ctx, block.to_k.tensor, h);
            ggml_tensor * v = ggml_mul_mat(ctx, block.to_v.tensor, h);
            auto rope_heads = [&](ggml_tensor * t) {
                // [inner, tokens] -> [head_dim, heads, seq, batch] -> partial NEOX rope on
                // the first rotary_dim dims -> contiguous f16 [head_dim, seq, heads, batch]
                t = ggml_reshape_4d(ctx, t, head_dim, heads, seq, kBatch);
                t = ggml_rope_ext(
                    ctx,
                    t,
                    in_positions,
                    nullptr,
                    static_cast<int>(config.dit_rotary_dim),
                    GGML_ROPE_TYPE_NEOX,
                    0,
                    config.dit_rope_theta,
                    1.0F,
                    0.0F,
                    1.0F,
                    0.0F,
                    0.0F);
                t = ggml_permute(ctx, t, 0, 2, 1, 3);  // [head_dim, seq, heads, batch]
                return ggml_cont(ctx, t);
            };
            // Flash attention wants Q in F32 and K/V in F16.
            q = rope_heads(q);
            k = ggml_cast(ctx, rope_heads(k), GGML_TYPE_F16);
            ggml_tensor * v4 = ggml_reshape_4d(ctx, v, head_dim, heads, seq, kBatch);
            v4 = ggml_cont(ctx, ggml_cast(ctx, ggml_permute(ctx, v4, 0, 2, 1, 3), GGML_TYPE_F16));

            ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v4, nullptr, scale, 0.0F, 0.0F);
            // Result is [head_dim, heads, seq, batch]; rows are already [heads][head_dim]
            // per token in batch-major order.
            attn = ggml_reshape_2d(ctx, ggml_cont(ctx, attn), inner, tokens);
            x = ggml_add(ctx, x, ggml_mul_mat(ctx, block.to_out.tensor, gemm_input(attn)));

            ggml_tensor * h2 = layer_norm(ctx, x, block.norm2_weight.tensor, block.norm2_bias.tensor, 1.0e-5F);
            ggml_tensor * ff = ggml_add(
                ctx, ggml_mul_mat(ctx, block.ff_in_weight.tensor, gemm_input(h2)), block.ff_in_bias.tensor);
            ggml_tensor * gate_states = ggml_view_2d(ctx, ff, config.dit_ff_inner, tokens, ff->nb[1], 0);
            ggml_tensor * gate = ggml_view_2d(
                ctx, ff, config.dit_ff_inner, tokens, ff->nb[1], static_cast<size_t>(config.dit_ff_inner) * ff->nb[0]);
            ggml_tensor * act = ggml_mul(ctx, ggml_cont(ctx, gate_states), ggml_silu(ctx, ggml_cont(ctx, gate)));
            ggml_tensor * down = ggml_add(
                ctx, ggml_mul_mat(ctx, block.ff_out_weight.tensor, gemm_input(act)), block.ff_out_bias.tensor);
            x = ggml_add(ctx, x, down);
        }

        // Drop the time token, project out, and apply the residual 1x1 conv.
        ggml_tensor * x_body = ggml_reshape_3d(ctx, x, inner, seq, kBatch);
        x_body = ggml_view_3d(
            ctx, x_body, inner, length, kBatch, x_body->nb[1], x_body->nb[2], x_body->nb[1]);
        x_body = ggml_cont(ctx, x_body);
        ggml_tensor * out = ggml_mul_mat(
            ctx,
            proj_out.tensor,
            gemm_input(ggml_reshape_2d(ctx, x_body, inner, kBatch * length)));  // [in_ch, b*length]
        {
            ggml_tensor * post_w = ggml_reshape_2d(ctx, postprocess_conv.tensor, in_channels, in_channels);
            out = ggml_add(ctx, ggml_mul_mat(ctx, post_w, out), out);
        }
        // [in_ch, length, batch] token-major -> channel-major [length, in_ch, batch].
        out = ggml_reshape_3d(ctx, out, in_channels, length, kBatch);
        out_velocity = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));  // [length, in_ch, batch]

        graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_set_output(out_velocity);
        ggml_build_forward_expand(graph, out_velocity);
        input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), execution.backend());
        if (input_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 DiT inputs");
        }
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 DiT graph");
        }
        core::prepare_host_graph_plan(execution, graph, plan);

        std::vector<int32_t> positions(static_cast<size_t>(seq));
        for (int64_t index = 0; index < seq; ++index) {
            positions[static_cast<size_t>(index)] = static_cast<int32_t>(index);
        }
        ggml_backend_tensor_set(in_positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        chunk_length = length;
    }
};

MiniMaxMusic3DitRuntime::MiniMaxMusic3DitRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source,
    const MiniMaxMusic3Config & config,
    size_t weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("MiniMax-Music3 DiT tensor source is missing");
    }
    impl_ = std::make_unique<Impl>(execution, *source, config, weight_context_bytes);
}

MiniMaxMusic3DitRuntime::~MiniMaxMusic3DitRuntime() = default;

void MiniMaxMusic3DitRuntime::begin_chunk(const std::vector<float> & condition, int64_t length) {
    auto & impl = *impl_;
    const int64_t cond_dim = impl.config.dit_condition_dim;
    if (length <= 0 || condition.size() != static_cast<size_t>(cond_dim * length)) {
        throw std::runtime_error("MiniMax-Music3 DiT condition size mismatch");
    }
    if (impl.chunk_length != length) {
        impl.build_chunk_graph(length);
    }
    // Conditional branch gets the condition; the unconditional branch conditions on zeros.
    std::vector<float> both(static_cast<size_t>(2 * cond_dim * length), 0.0F);
    std::copy(condition.begin(), condition.end(), both.begin());
    ggml_backend_tensor_set(impl.in_condition, both.data(), 0, both.size() * sizeof(float));
}

std::vector<float> MiniMaxMusic3DitRuntime::guided_velocity(
    const std::vector<float> & latent,
    float t,
    float guidance_scale) {
    auto & impl = *impl_;
    const int64_t length = impl.chunk_length;
    const int64_t in_channels = impl.config.dit_in_channels;
    if (length <= 0) {
        throw std::runtime_error("MiniMax-Music3 DiT chunk is not prepared");
    }
    if (latent.size() != static_cast<size_t>(in_channels * length)) {
        throw std::runtime_error("MiniMax-Music3 DiT latent size mismatch");
    }
    ggml_backend_tensor_set(impl.in_latent, latent.data(), 0, latent.size() * sizeof(float));
    ggml_backend_tensor_set(impl.in_time, &t, 0, sizeof(float));
    const ggml_status status = core::compute_graph(impl.execution, impl.graph, impl.plan, "minimax_music3.dit");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MiniMax-Music3 DiT graph compute failed");
    }
    const size_t branch = latent.size();
    std::vector<float> both(2 * branch);
    ggml_backend_tensor_get(impl.out_velocity, both.data(), 0, both.size() * sizeof(float));
    std::vector<float> cond(both.begin(), both.begin() + static_cast<int64_t>(branch));
    std::vector<float> uncond(both.begin() + static_cast<int64_t>(branch), both.end());
    return engine::sampling::cfg_guidance(cond, uncond, guidance_scale);
}

void MiniMaxMusic3DitRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_chunk();
    }
}

}  // namespace engine::models::minimax_music3
