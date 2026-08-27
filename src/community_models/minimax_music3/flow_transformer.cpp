#include "engine/community_models/minimax_music3/flow_transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

MiniMaxMusic3FlowWeights load_flow_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.flow;
    const auto & source = *assets.transformer_weights;
    MiniMaxMusic3FlowWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.flow.weights",
        weight_context_bytes);
    const int64_t concat_channels = 2 * config.in_channels + config.condition_dim;
    out.preprocess_conv = binding::conv1d_from_source(
        *out.store,
        source,
        "preprocess_conv",
        conv_safe_storage_type(source, "preprocess_conv", storage_type),
        concat_channels,
        concat_channels,
        1,
        false);
    out.proj_in = binding::linear_from_source(*out.store, source, "proj_in", storage_type, config.attention_heads * config.head_dim, concat_channels, false);
    out.time_proj_weight = out.store->load_f32_tensor(source, "time_proj.weight", {config.fourier_embedding_dim / 2, 1});
    out.time_linear_1 = binding::linear_from_source(*out.store, source, "time_embed.linear_1", storage_type, config.attention_heads * config.head_dim, config.fourier_embedding_dim, true);
    out.time_linear_2 = binding::linear_from_source(*out.store, source, "time_embed.linear_2", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, true);
    out.blocks.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        const std::string prefix = "transformer_blocks." + std::to_string(layer);
        MiniMaxMusic3FlowBlockWeights block;
        block.norm1 = binding::norm_from_source(*out.store, source, prefix + ".norm1", config.attention_heads * config.head_dim);
        block.q = binding::linear_from_source(*out.store, source, prefix + ".attn.to_q", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.k = binding::linear_from_source(*out.store, source, prefix + ".attn.to_k", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.v = binding::linear_from_source(*out.store, source, prefix + ".attn.to_v", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.out = binding::linear_from_source(*out.store, source, prefix + ".attn.to_out.0", storage_type, config.attention_heads * config.head_dim, config.attention_heads * config.head_dim, false);
        block.norm2 = binding::norm_from_source(*out.store, source, prefix + ".norm2", config.attention_heads * config.head_dim);
        block.ff_in = binding::linear_from_source(*out.store, source, prefix + ".ff_in", storage_type, 2 * config.ff_inner_dim, config.attention_heads * config.head_dim, true);
        block.ff_out = binding::linear_from_source(*out.store, source, prefix + ".ff_out", storage_type, config.attention_heads * config.head_dim, config.ff_inner_dim, true);
        out.blocks.push_back(std::move(block));
    }
    out.proj_out = binding::linear_from_source(*out.store, source, "proj_out", storage_type, config.in_channels, config.attention_heads * config.head_dim, false);
    out.postprocess_conv = binding::conv1d_from_source(
        *out.store,
        source,
        "postprocess_conv",
        conv_safe_storage_type(source, "postprocess_conv", storage_type),
        config.in_channels,
        config.in_channels,
        1,
        false);
    out.store->upload();
    assets.transformer_weights->release_storage();
    return out;
}

std::vector<float> fourier_embedding(const std::vector<float> & weights, int64_t dim, float timestep) {
    constexpr float kTwoPi = 6.2831853071795864769F;
    std::vector<float> out(static_cast<size_t>(dim), 0.0F);
    const int64_t half = dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float angle = kTwoPi * timestep * weights[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = std::cos(angle);
        out[static_cast<size_t>(half + i)] = std::sin(angle);
    }
    return out;
}

std::vector<float> split_rope_table(int64_t steps, int64_t heads, int64_t rotary_dim, bool cosine) {
    std::vector<float> out(static_cast<size_t>(steps * heads * rotary_dim / 2), 0.0F);
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t i = 0; i < rotary_dim / 2; ++i) {
            const float inv_freq = std::pow(10000.0F, -static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
            const float value = cosine ? std::cos(static_cast<float>(step) * inv_freq) : std::sin(static_cast<float>(step) * inv_freq);
            for (int64_t head = 0; head < heads; ++head) {
                out[static_cast<size_t>((step * heads + head) * (rotary_dim / 2) + i)] = value;
            }
        }
    }
    return out;
}

core::TensorValue apply_partial_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t rotary_dim) {
    auto rotary = modules::SliceModule({3, 0, rotary_dim}).build(ctx, input);
    auto rest = modules::SliceModule({3, rotary_dim, input.shape.dims[3] - rotary_dim}).build(ctx, input);
    auto cos_used = cos;
    auto sin_used = sin;
    if (input.shape.dims[0] == 1) {
        // Break every view chain feeding the rope elementwise ops: at batch 1
        // the graph allocator produced buffer aliasing along these views
        // (bitwise-reproducible call-to-call divergence, absent at batch 2
        // where ggml_repeat materializes the chain).
        rotary = core::wrap_tensor(ggml_cont(ctx.ggml, rotary.tensor), rotary.shape, GGML_TYPE_F32);
        rest = core::wrap_tensor(ggml_cont(ctx.ggml, rest.tensor), rest.shape, GGML_TYPE_F32);
        // At batch 1 the rope tables match the sliced shape exactly, so the
        // generic rope helper would feed the graph-input tensors straight
        // into the elementwise chain (its repeat short-circuit). That path
        // produced nondeterministic outputs on CUDA; materializing a copy
        // restores the batch>1 behavior where ggml_repeat isolates the
        // inputs from downstream scheduling.
        cos_used = core::wrap_tensor(ggml_cont(ctx.ggml, cos.tensor), cos.shape, GGML_TYPE_F32);
        sin_used = core::wrap_tensor(ggml_cont(ctx.ggml, sin.tensor), sin.shape, GGML_TYPE_F32);
    }
    const auto rotated = modules::SplitRoPEModule({rotary_dim}).build(ctx, rotary, cos_used, sin_used);
    return modules::ConcatModule({3}).build(ctx, rotated, rest);
}

}  // namespace

struct MiniMaxMusic3FlowTransformerRuntime::Impl {
    // Two independently cached graphs share the weights: slot 0 evaluates the
    // usual cond+uncond batch, slot 1 evaluates the cond branch alone for
    // guidance-delta reuse steps. Keeping both alive avoids rebuilding when a
    // sampling schedule alternates between them.
    struct GraphSlot {
        int64_t batch = 0;
        int64_t latent_frames = 0;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue latents;
        core::TensorValue condition;
        core::TensorValue time_features;
        core::TensorValue rope_cos;
        core::TensorValue rope_sin;
        core::TensorValue rope_positions;
        ggml_tensor * output = nullptr;
    };

    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::ExecutionContext & input_execution,
        size_t input_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool input_evict_cuda_graph_cache_on_release)
        : assets(std::move(input_assets)),
          execution(input_execution),
          graph_arena_bytes(input_graph_arena_bytes),
          evict_cuda_graph_cache_on_release(input_evict_cuda_graph_cache_on_release),
          weights(load_flow_weights(*assets, execution, weight_context_bytes, storage_type)) {
        time_proj = assets->transformer_weights->require_f32(
            "time_proj.weight",
            {assets->config.flow.fourier_embedding_dim / 2, 1});
    
        if (getenv("MM3_DC_B1_SELFTEST") != nullptr) {
            run_b1_selftest();
        }
    }

    void run_b1_selftest() {
        const auto & config = assets->config.flow;
        const int64_t frames = 344;
        std::vector<float> latents(static_cast<size_t>(config.in_channels * frames));
        std::vector<float> condition(static_cast<size_t>(config.condition_dim * frames));
        for (size_t i = 0; i < latents.size(); ++i) {
            latents[i] = std::sin(0.001F * static_cast<float>(i));
        }
        for (size_t i = 0; i < condition.size(); ++i) {
            condition[i] = std::cos(0.0007F * static_cast<float>(i));
        }
        prepare_chunk_condition(condition, frames);
        std::vector<float> batch2_cond;
        {
            // batch-2 determinism control at full depth
            std::vector<float> runs2[2];
            for (int r = 0; r < 2; ++r) {
                runs2[r] = predict_velocity(latents, condition, frames, 0.85F, 2);
            }
            float d2 = 0.0F;
            for (size_t i = 0; i < runs2[0].size(); ++i) {
                d2 = std::max(d2, std::fabs(runs2[0][i] - runs2[1][i]));
            }
            fprintf(stderr, "MM3_B1_SELFTEST batch2_control rep01=%.6f\n", d2);
            batch2_cond.assign(runs2[0].begin(), runs2[0].begin() + static_cast<std::ptrdiff_t>(runs2[0].size() / 2));
        }
        for (const char * skip : {"", "rope", "attn", "ffn", "rope,attn", "attn,ffn"}) {
#ifndef _WIN32
            if (skip[0] != '\0') setenv("MM3_DC_B1_SKIP", skip, 1); else unsetenv("MM3_DC_B1_SKIP");
#endif
            release_slot(slots[1]);
            std::vector<float> runsk[2];
            for (int r = 0; r < 2; ++r) {
                runsk[r] = predict_velocity(latents, condition, frames, 0.85F, 1);
            }
            float dk = 0.0F;
            for (size_t i = 0; i < runsk[0].size(); ++i) {
                dk = std::max(dk, std::fabs(runsk[0][i] - runsk[1][i]));
            }
            float db2 = 0.0F;
            if (skip[0] == '\0' && batch2_cond.size() == runsk[0].size()) {
                for (size_t i = 0; i < runsk[0].size(); ++i) {
                    db2 = std::max(db2, std::fabs(runsk[0][i] - batch2_cond[i]));
                }
            }
            fprintf(stderr, "MM3_B1_SELFTEST skip='%s' rep01=%.6f b1_vs_b2cond=%.6f\n", skip, dk, db2);
        }
#ifndef _WIN32
        unsetenv("MM3_DC_B1_SKIP");
#endif
        const char * caps_env = getenv("MM3_DC_B1_SELFTEST");
        std::string caps = caps_env != nullptr && std::string(caps_env) != "1" ? caps_env : "1,2,4,8,16,36";
        size_t pos = 0;
        while (pos <= caps.size()) {
            size_t comma = caps.find(',', pos);
            if (comma == std::string::npos) comma = caps.size();
            const std::string tok = caps.substr(pos, comma - pos);
            pos = comma + 1;
            if (tok.empty()) continue;
#ifndef _WIN32
            setenv("MM3_DC_B1_LAYERS", tok.c_str(), 1);
#endif
            release_slot(slots[1]);
            std::vector<float> runs[3];
            for (int r = 0; r < 3; ++r) {
                runs[r] = predict_velocity(latents, condition, frames, 0.85F, 1);
            }
            float d01 = 0.0F, d12 = 0.0F, mag = 0.0F;
            for (size_t i = 0; i < runs[0].size(); ++i) {
                d01 = std::max(d01, std::fabs(runs[0][i] - runs[1][i]));
                d12 = std::max(d12, std::fabs(runs[1][i] - runs[2][i]));
                mag = std::max(mag, std::fabs(runs[2][i]));
            }
            fprintf(stderr, "MM3_B1_SELFTEST layers=%s rep01=%.6f rep12=%.6f maxabs=%.6f\n",
                    tok.c_str(), d01, d12, mag);
        }
#ifndef _WIN32
        unsetenv("MM3_DC_B1_LAYERS");
#endif
        release_slot(slots[1]);
    }

    ~Impl() {
        release_runtime_graphs();
    }

    void release_slot(GraphSlot & slot) {
        if (slot.graph != nullptr) {
            core::release_backend_graph_resources(
                execution.backend(), slot.graph, evict_cuda_graph_cache_on_release);
        }
        slot = {};
    }

    void release_runtime_graphs() {
        release_slot(slots[0]);
        release_slot(slots[1]);
        condition_frames = 0;
        rope_cos_table.clear();
        rope_sin_table.clear();
    }

    GraphSlot & ensure_graph(int64_t frames, int64_t batch) {
        GraphSlot & slot = slots[batch == 2 ? 0 : 1];
        if (slot.graph != nullptr && slot.latent_frames == frames && slot.batch == batch) {
            return slot;
        }
        release_slot(slot);
        const auto & config = assets->config.flow;
        const int64_t inner = config.attention_heads * config.head_dim;
        const int64_t steps = frames + 1;
        const int64_t concat_channels = 2 * config.in_channels + config.condition_dim;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        slot.ggml.reset(ggml_init(params));
        if (slot.ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 flow graph context");
        }
        int64_t layer_cap = static_cast<int64_t>(weights.blocks.size());
        bool skip_rope = false;
        bool skip_attn = false;
        bool skip_ffn = false;
        if (batch == 1) {
            if (const char * cap_env = getenv("MM3_DC_B1_LAYERS")) {
                layer_cap = std::min<int64_t>(layer_cap, std::max<int64_t>(0, atoll(cap_env)));
            }
            if (const char * skip_env = getenv("MM3_DC_B1_SKIP")) {
                const std::string skips = skip_env;
                skip_rope = skips.find("rope") != std::string::npos;
                skip_attn = skips.find("attn") != std::string::npos;
                skip_ffn = skips.find("ffn") != std::string::npos;
            }
        }
        core::ModuleBuildContext ctx{slot.ggml.get(), "minimax_music3.flow", execution.backend_type()};
        slot.latents = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.in_channels, frames}));
        slot.condition = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.condition_dim, frames}));
        slot.time_features = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.fourier_embedding_dim}));
        slot.rope_cos = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.attention_heads, config.rotary_dim / 2}));
        slot.rope_sin = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.attention_heads, config.rotary_dim / 2}));
        slot.rope_positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(slot.latents.tensor);
        ggml_set_input(slot.condition.tensor);
        ggml_set_input(slot.time_features.tensor);
        ggml_set_input(slot.rope_cos.tensor);
        ggml_set_input(slot.rope_sin.tensor);
        ggml_set_input(slot.rope_positions.tensor);

        auto zeros = core::wrap_tensor(ggml_scale(ctx.ggml, slot.latents.tensor, 0.0F), slot.latents.shape, GGML_TYPE_F32);
        auto x = modules::ConcatModule({1}).build(ctx, slot.latents, zeros);
        x = modules::ConcatModule({1}).build(ctx, x, slot.condition);
        auto conv = modules::Conv1dModule({concat_channels, concat_channels, 1, 1, 0, 1, false}).build(
            ctx,
            x,
            weights.preprocess_conv);
        x = core::wrap_tensor(ggml_add(ctx.ggml, conv.tensor, x.tensor), conv.shape, GGML_TYPE_F32);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        x = modules::LinearModule({concat_channels, inner, false}).build(ctx, x, weights.proj_in);

        auto temb = modules::LinearModule({config.fourier_embedding_dim, inner, true}).build(
            ctx,
            slot.time_features,
            weights.time_linear_1);
        temb = modules::SiluModule().build(ctx, temb);
        temb = modules::LinearModule({inner, inner, true}).build(ctx, temb, weights.time_linear_2);
        temb = core::reshape_tensor(ctx, temb, core::TensorShape::from_dims({batch, 1, inner}));
        x = modules::ConcatModule({1}).build(ctx, temb, x);

        int64_t built_layers = 0;
        for (const auto & block : weights.blocks) {
            if (built_layers++ >= layer_cap) {
                break;
            }
            auto normed = modules::LayerNormModule({inner, 1.0e-5F, true, true, false}).build(ctx, x, block.norm1);
            auto q = modules::LinearModule({inner, inner, false}).build(ctx, normed, block.q);
            auto k = modules::LinearModule({inner, inner, false}).build(ctx, normed, block.k);
            auto v = modules::LinearModule({inner, inner, false}).build(ctx, normed, block.v);
            q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({batch, steps, config.attention_heads, config.head_dim}));
            k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({batch, steps, config.attention_heads, config.head_dim}));
            v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({batch, steps, config.attention_heads, config.head_dim}));
            if (!skip_rope) {
                if (batch == 1) {
                    // The framework split-rope lowering aliases buffers at
                    // batch 1 (bitwise-reproducible call-to-call divergence);
                    // the native partial NEOX rope computes the identical
                    // rotation from positions and sidesteps that path.
                    q = core::wrap_tensor(
                        ggml_rope_ext(ctx.ggml, q.tensor, slot.rope_positions.tensor, nullptr,
                                      static_cast<int>(config.rotary_dim), GGML_ROPE_TYPE_NEOX, 0,
                                      10000.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F),
                        q.shape, GGML_TYPE_F32);
                    k = core::wrap_tensor(
                        ggml_rope_ext(ctx.ggml, k.tensor, slot.rope_positions.tensor, nullptr,
                                      static_cast<int>(config.rotary_dim), GGML_ROPE_TYPE_NEOX, 0,
                                      10000.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F),
                        k.shape, GGML_TYPE_F32);
                } else {
                    q = apply_partial_rope(ctx, q, slot.rope_cos, slot.rope_sin, config.rotary_dim);
                    k = apply_partial_rope(ctx, k, slot.rope_cos, slot.rope_sin, config.rotary_dim);
                }
            }
            q = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
            const bool flash_ok =
                core::uses_ggml_cuda_or_hip_backend(execution.backend_type()) &&
                !(batch == 1 && getenv("MM3_DC_B1_EXPLICIT_ATTN") != nullptr);
            auto attn = modules::ScaledDotProductAttentionModule({
                config.head_dim,
                flash_ok
                    ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
                    : modules::ScaledDotProductAttentionLowering::Explicit,
                GGML_PREC_F32,
                modules::AttentionCausality::NonCausal,
            }).build(ctx, q, k, v);
            attn = core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(ctx, attn),
                core::TensorShape::from_dims({batch, steps, inner}));
            attn = modules::LinearModule({inner, inner, false}).build(ctx, attn, block.out);
            if (!skip_attn) {
                x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, attn.tensor), x.shape, GGML_TYPE_F32);
            }

            auto ffn = modules::LayerNormModule({inner, 1.0e-5F, true, true, false}).build(ctx, x, block.norm2);
            ffn = modules::LinearModule({inner, 2 * config.ff_inner_dim, true}).build(ctx, ffn, block.ff_in);
            const auto gate_states = modules::SliceModule({2, 0, config.ff_inner_dim}).build(ctx, ffn);
            auto gate = modules::SliceModule({2, config.ff_inner_dim, config.ff_inner_dim}).build(ctx, ffn);
            gate = modules::SiluModule().build(ctx, gate);
            auto gated = core::wrap_tensor(
                ggml_mul(ctx.ggml, gate_states.tensor, gate.tensor),
                gate_states.shape,
                GGML_TYPE_F32);
            gated = modules::LinearModule({config.ff_inner_dim, inner, true}).build(ctx, gated, block.ff_out);
            if (!skip_ffn) {
                x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, gated.tensor), x.shape, GGML_TYPE_F32);
            }
        }

        x = modules::SliceModule({1, 1, frames}).build(ctx, x);
        x = modules::LinearModule({inner, config.in_channels, false}).build(ctx, x, weights.proj_out);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        auto post = modules::Conv1dModule({config.in_channels, config.in_channels, 1, 1, 0, 1, false}).build(
            ctx,
            x,
            weights.postprocess_conv);
        x = core::wrap_tensor(ggml_add(ctx.ggml, post.tensor, x.tensor), post.shape, GGML_TYPE_F32);
        slot.output = x.tensor;
        slot.graph = ggml_new_graph_custom(slot.ggml.get(), 524288, false);
        ggml_build_forward_expand(slot.graph, slot.output);
        slot.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (slot.gallocr == nullptr || !ggml_gallocr_reserve(slot.gallocr.get(), slot.graph) ||
            !ggml_gallocr_alloc_graph(slot.gallocr.get(), slot.graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 flow graph");
        }
        core::prepare_host_graph_plan(execution, slot.graph, slot.plan);
        slot.latent_frames = frames;
        slot.batch = batch;
        if (rope_cos_table.size() != static_cast<size_t>(steps * config.attention_heads * config.rotary_dim / 2)) {
            rope_cos_table = split_rope_table(steps, config.attention_heads, config.rotary_dim, true);
            rope_sin_table = split_rope_table(steps, config.attention_heads, config.rotary_dim, false);
        }
        static const bool trace = getenv("MM3_DC_B1_TRACE") != nullptr;
        if (trace) {
            int rope_as_src = 0;
            int rope_as_leaf = 0;
            for (int n = 0; n < ggml_graph_n_nodes(slot.graph); ++n) {
                ggml_tensor * node = ggml_graph_node(slot.graph, n);
                for (int si = 0; si < GGML_MAX_SRC; ++si) {
                    if (node->src[si] == slot.rope_cos.tensor || node->src[si] == slot.rope_sin.tensor) {
                        ++rope_as_src;
                    }
                }
            }
            fprintf(stderr, "MM3_B1_TRACE build batch=%lld frames=%lld nodes=%d rope_src=%d rope_leaf=%d rope_buf=%p rope_flags=%d latents_buf=%p\n",
                    (long long) batch, (long long) frames,
                    ggml_graph_n_nodes(slot.graph), rope_as_src, rope_as_leaf,
                    (void *) slot.rope_cos.tensor->buffer, slot.rope_cos.tensor->flags,
                    (void *) slot.latents.tensor->buffer);
        }
        if (slot.rope_cos.tensor->buffer != nullptr) {
            core::write_tensor_f32(slot.rope_cos, rope_cos_table);
            core::write_tensor_f32(slot.rope_sin, rope_sin_table);
        }
        if (slot.rope_positions.tensor->buffer != nullptr) {
            std::vector<int32_t> positions(static_cast<size_t>(steps));
            for (int64_t i = 0; i < steps; ++i) {
                positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            core::write_tensor_i32(slot.rope_positions, positions);
        }
        latent_batch.resize(static_cast<size_t>(2 * config.in_channels * frames));
        time_batch.resize(static_cast<size_t>(2 * config.fourier_embedding_dim));
        return slot;
    }

    void prepare_chunk_condition(
        const std::vector<float> & input_condition,
        int64_t frames) {
        const auto & config = assets->config.flow;
        if (static_cast<int64_t>(input_condition.size()) != config.condition_dim * frames) {
            throw std::runtime_error("MiniMax Music 3 flow condition shape mismatch");
        }
        condition_batch.assign(static_cast<size_t>(2 * config.condition_dim * frames), 0.0F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < config.condition_dim; ++channel) {
                condition_batch[static_cast<size_t>(channel * frames + frame)] =
                    input_condition[static_cast<size_t>(frame * config.condition_dim + channel)];
            }
        }
        condition_frames = frames;
    }

    std::vector<float> predict_velocity(
        const std::vector<float> & input_latents,
        const std::vector<float> & input_condition,
        int64_t frames,
        float timestep,
        int64_t batch) {
        static const bool b1_as_batch2 = getenv("MM3_DC_B1_BATCH2") != nullptr;
        if (batch == 1 && b1_as_batch2) {
            auto both = predict_velocity(input_latents, input_condition, frames, timestep, 2);
            both.resize(both.size() / 2);
            return both;
        }
        const auto & config = assets->config.flow;
        if (static_cast<int64_t>(input_latents.size()) != config.in_channels * frames ||
            static_cast<int64_t>(input_condition.size()) != config.condition_dim * frames) {
            throw std::runtime_error("MiniMax Music 3 flow input shape mismatch");
        }
        auto & slot = ensure_graph(frames, batch);
        if (condition_frames != frames ||
            condition_batch.size() != static_cast<size_t>(2 * config.condition_dim * frames)) {
            throw std::runtime_error("MiniMax Music 3 flow condition was not prepared for this chunk");
        }
        std::copy(input_latents.begin(), input_latents.end(), latent_batch.begin());
        if (batch == 2) {
            std::copy(input_latents.begin(), input_latents.end(), latent_batch.begin() + input_latents.size());
        }
        const auto time_feature = fourier_embedding(time_proj, config.fourier_embedding_dim, timestep);
        std::copy(time_feature.begin(), time_feature.end(), time_batch.begin());
        if (batch == 2) {
            std::copy(time_feature.begin(), time_feature.end(), time_batch.begin() + time_feature.size());
        }
        core::write_tensor_f32(slot.latents, latent_batch.data(), static_cast<size_t>(batch * config.in_channels * frames));
        core::write_tensor_f32(slot.condition, condition_batch.data(), static_cast<size_t>(batch * config.condition_dim * frames));
        core::write_tensor_f32(slot.time_features, time_batch.data(), static_cast<size_t>(batch * config.fourier_embedding_dim));
        if (core::compute_graph(execution, slot.graph, slot.plan, "minimax_music3.flow") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 flow graph compute failed");
        }
        return core::read_tensor_f32(slot.output);
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    bool evict_cuda_graph_cache_on_release = false;
    MiniMaxMusic3FlowWeights weights;
    std::vector<float> time_proj;
    int64_t condition_frames = 0;
    std::vector<float> latent_batch;
    std::vector<float> condition_batch;
    std::vector<float> time_batch;
    std::vector<float> rope_cos_table;
    std::vector<float> rope_sin_table;
    GraphSlot slots[2];
};

MiniMaxMusic3FlowTransformerRuntime::MiniMaxMusic3FlowTransformerRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type,
    bool evict_cuda_graph_cache_on_release)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          evict_cuda_graph_cache_on_release)) {}

MiniMaxMusic3FlowTransformerRuntime::~MiniMaxMusic3FlowTransformerRuntime() = default;

std::vector<float> MiniMaxMusic3FlowTransformerRuntime::predict_velocity_branches(
    const std::vector<float> & latents,
    const std::vector<float> & condition,
    int64_t latent_frames,
    float timestep) {
    return impl_->predict_velocity(latents, condition, latent_frames, timestep, 2);
}

std::vector<float> MiniMaxMusic3FlowTransformerRuntime::predict_velocity_cond(
    const std::vector<float> & latents,
    const std::vector<float> & condition,
    int64_t latent_frames,
    float timestep) {
    return impl_->predict_velocity(latents, condition, latent_frames, timestep, 1);
}

void MiniMaxMusic3FlowTransformerRuntime::prepare_chunk_condition(
    const std::vector<float> & condition,
    int64_t latent_frames) {
    impl_->prepare_chunk_condition(condition, latent_frames);
}

void MiniMaxMusic3FlowTransformerRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
