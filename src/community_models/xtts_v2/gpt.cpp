// xtts_v2 GPT engine -- GPT-2 AR + ConditioningEncoder + Perceiver resampler
// MSVC's <cmath> only exposes M_PI with _USE_MATH_DEFINES; define it first.
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "engine/community_models/xtts_v2/gpt.h"
#include "engine/community_models/xtts_v2/frontend.h"
#include "engine/community_models/xtts_v2/assets.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {
namespace {

struct GptBlock {
    modules::NormWeights attention_norm, feedforward_norm;
    modules::LinearWeights qkv, attention_output;
    modules::FeedForwardWeights feedforward;
};
struct GptWeights {
    core::TensorValue text_embedding, mel_embedding, text_position, mel_position;
    std::vector<GptBlock> blocks;
    modules::NormWeights decoder_norm, final_norm;
    modules::LinearWeights mel_head;
};
static GptWeights bind_gpt_weights(const std::map<std::string, core::TensorValue> & tensors, const XttSv2HyperParams & hp) {
    GptWeights weights;
    weights.text_embedding = tensors.at("gpt.text_embedding.weight");
    weights.mel_embedding = tensors.at("gpt.mel_embedding.weight");
    weights.text_position = tensors.at("gpt.text_pos.weight");
    weights.mel_position = tensors.at("gpt.mel_pos.weight");
    weights.blocks.resize(hp.n_layers);
    for (uint32_t layer = 0; layer < hp.n_layers; ++layer) {
        const auto prefix = "gpt.gpt.h." + std::to_string(layer) + ".";
        auto & block = weights.blocks[layer];
        block.attention_norm = {tensors.at(prefix + "ln_1.weight"), tensors.at(prefix + "ln_1.bias")};
        block.qkv = {tensors.at(prefix + "attn.c_attn.weight"), tensors.at(prefix + "attn.c_attn.bias")};
        block.attention_output = {tensors.at(prefix + "attn.c_proj.weight"), tensors.at(prefix + "attn.c_proj.bias")};
        block.feedforward_norm = {tensors.at(prefix + "ln_2.weight"), tensors.at(prefix + "ln_2.bias")};
        block.feedforward = {tensors.at(prefix + "mlp.c_fc.weight"), tensors.at(prefix + "mlp.c_fc.bias"),
            tensors.at(prefix + "mlp.c_proj.weight"), tensors.at(prefix + "mlp.c_proj.bias")};
    }
    weights.decoder_norm = {tensors.at("gpt.gpt.ln_f.weight"), tensors.at("gpt.gpt.ln_f.bias")};
    weights.final_norm = {tensors.at("gpt.final_norm.weight"), tensors.at("gpt.final_norm.bias")};
    weights.mel_head = {tensors.at("gpt.mel_head.weight"), tensors.at("gpt.mel_head.bias")};
    return weights;
}
// ---------------------------------------------------------------------------
// GGML-native GPT-2 autoregressive decode.
//
//  * Weights are used in-place from the backend buffer in whatever type the GGUF
//    stores (F32/F16/Q8_0/...); no dequantized host copies.  ggml's mul_mat
//    kernels do the GEMM/GEMV (vectorized, quantized vec_dot, multithreaded).
//  * The KV cache is a persistent F16 backend tensor per layer/K/V, laid out
//    [HD, n_ctx, H] so flash_attn_ext reads it directly.  New K/V rows are
//    written with ggml_set_rows (position index is a graph *input*, so the graph
//    does not change from step to step).
//  * Attention runs over a window that is padded up to a multiple of KV_BUCKET
//    and masked with an additive F16 mask (-inf outside the causal range).  The
//    graph is only rebuilt when the window crosses a bucket boundary (every 256
//    steps), not per token.
//  * Intermediates are planned by ggml_gallocr (buffers reused across layers).
//  * The lm_head (ln_f -> final_norm -> mel_head) is fused into the graph and,
//    for the prefix, only evaluated on the last position.
// ---------------------------------------------------------------------------
static constexpr int KV_BUCKET = 256;
static int round_up_i(int x, int m) { return (x + m - 1) / m * m; }

// Persistent KV cache. Kept across gpt_gen calls: the buffer is zeroed once at
// allocation, afterwards it only ever holds finite values from earlier chunks,
// and anything outside the causal window is masked to -inf, so no re-zeroing.
// The conditioning encoder and perceiver form one session-owned graph. The
// convolution kernels are size one, so Linear preserves their exact layout.
class ConditioningGraph {
public:
    ConditioningGraph(core::ExecutionContext & execution,
        const std::map<std::string, core::TensorValue> & weights, const XttSv2HyperParams & hp, int frames)
        : execution_(execution), frames_(frames) {
        ctx_ = ggml_init({16 * 1024 * 1024, nullptr, true});
        context_owner_.reset(ctx_);
        if (!ctx_) throw std::runtime_error("XTTS v2 conditioning graph context allocation failed");
        core::ModuleBuildContext build{ctx_, "xtts_v2.conditioning", execution.backend_type()};
        const int D = hp.d_model, H = hp.cond_n_heads, HD = D / H, N = hp.perceiver_n_latents;
        auto parameter = [&](const std::string & name, const core::TensorShape & shape) {
            const auto & tensor = weights.at(name);
            if (tensor.shape.num_elements() != shape.num_elements()) {
                throw std::runtime_error("XTTS v2 conditioning tensor shape mismatch: " + name);
            }
            return core::reshape_tensor(build, tensor, shape);
        };
        auto linear = [&](const core::TensorValue & input, const std::string & prefix, int out, bool bias, bool preserve_f32_inputs = false) {
            modules::LinearWeights projection;
            projection.weight = parameter(prefix + ".weight", core::TensorShape::from_dims({out, input.shape.last_dim()}));
            if (bias) projection.bias = parameter(prefix + ".bias", core::TensorShape::from_dims({out}));
            modules::LinearConfig config{input.shape.last_dim(), out, bias, GGML_PREC_F32};
            config.vulkan_f32_activation_lowering = execution.backend_type() == core::BackendType::Vulkan && preserve_f32_inputs;
            if (execution.backend_type() == core::BackendType::Metal && preserve_f32_inputs) {
                // Metal's matrix kernel narrows operands to FP16. Batches <= 8
                // use the vector kernel, preserving the Perceiver's FP32 range.
                core::TensorValue result;
                for (int64_t row = 0; row < input.shape.at(0); row += 8) {
                    auto rows = modules::SliceModule({0, row, std::min<int64_t>(8, input.shape.at(0) - row)}).build(build, input);
                    auto projected = modules::LinearModule(config).build(build, rows, projection);
                    result = row == 0 ? projected : modules::ConcatModule({0}).build(build, result, projected);
                }
                return result;
            }
            return modules::LinearModule(config).build(build, input, projection);
        };
        input_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 80, frames);
        ggml_set_input(input_);
        auto x = linear(core::wrap_tensor(input_, core::TensorShape::from_dims({frames, 80})), "gpt.cond_enc.init", D, true);
        const modules::ScaledDotProductAttentionModule attention({HD, modules::ScaledDotProductAttentionLowering::Explicit});
        for (uint32_t layer = 0; layer < hp.cond_n_layers; ++layer) {
            const auto prefix = "gpt.cond_enc.attn." + std::to_string(layer);
            auto channels = modules::TransposeModule({{1, 0}, 2}).build(build, x);
            channels = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, channels),
                core::TensorShape::from_dims({1, D, frames}));
            auto normalized = modules::GroupNormModule({D, 32, 1e-5f}).build(build, channels,
                {parameter(prefix + ".norm.weight", core::TensorShape::from_dims({D})),
                 parameter(prefix + ".norm.bias", core::TensorShape::from_dims({D}))});
            normalized = core::reshape_tensor(build, normalized, core::TensorShape::from_dims({D, frames}));
            x = modules::TransposeModule({{1, 0}, 2}).build(build, normalized);
            auto qkv = linear(x, prefix + ".qkv", 3 * D, true);
            qkv = core::reshape_tensor(build, qkv, core::TensorShape::from_dims({frames, H, 3, HD}));
            // Tortoise packs Q/K/V inside each head, unlike GPT-2's Q/K/V blocks.
            auto q = modules::SliceModule({2, 0, 1}).build(build, qkv);
            auto k = modules::SliceModule({2, 1, 1}).build(build, qkv);
            auto v = modules::SliceModule({2, 2, 1}).build(build, qkv);
            const auto head_shape = core::TensorShape::from_dims({1, frames, H, HD});
            q = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, q), head_shape);
            k = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, k), head_shape);
            v = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, v), head_shape);
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, v);
            auto context = attention.build(build, q, k, v);
            auto merged = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, context),
                core::TensorShape::from_dims({frames, D}));
            x = modules::ResidualAddModule().build(build, x, linear(merged, prefix + ".proj_out", D, true));
        }
        encoder_output_ = x.tensor;
        ggml_set_output(encoder_output_);
        // The unnormalized Perceiver residuals exceed FP16 range. Preserve FP32
        // inputs through its projections and attention, not only accumulation.
        auto latents = parameter("gpt.perc.latents", core::TensorShape::from_dims({N, D}));
        for (uint32_t layer = 0; layer < hp.perceiver_n_layers; ++layer) {
            const auto prefix = "gpt.perc.layers." + std::to_string(layer);
            const int inner = weights.at(prefix + ".0.to_q.weight").shape.at(0);
            const int heads = 8, dim_head = inner / heads;
            auto kv_input = modules::ConcatModule({0}).build(build, latents, x);
            auto q = linear(latents, prefix + ".0.to_q", inner, false, true);
            auto kv = linear(kv_input,
                prefix + ".0.to_kv", 2 * inner, false, true);
            auto k = modules::SliceModule({1, 0, inner}).build(build, kv);
            auto v = modules::SliceModule({1, inner, inner}).build(build, kv);
            q = core::reshape_tensor(build, q, core::TensorShape::from_dims({1, N, heads, dim_head}));
            k = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, k),
                core::TensorShape::from_dims({1, N + frames, heads, dim_head}));
            v = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, v),
                core::TensorShape::from_dims({1, N + frames, heads, dim_head}));
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, v);
            core::TensorValue context;
            if (execution.backend_type() == core::BackendType::Vulkan) {
                // XTTS's unnormalized Perceiver needs full-range FP32 operands.
                // Vulkan otherwise narrows strided matmul inputs to FP16, even
                // when accumulation is FP32. Materialize both matrix layouts.
                q = core::ensure_backend_addressable_layout(build, q);
                k = core::ensure_backend_addressable_layout(build, k);
                auto scores = core::wrap_tensor(ggml_mul_mat(ctx_, k.tensor, q.tensor),
                    core::TensorShape::from_dims({1, heads, N, N + frames}));
                ggml_mul_mat_set_prec(scores.tensor, GGML_PREC_F32);
                ggml_mul_mat_set_lowering(scores.tensor, GGML_MUL_MAT_LOWERING_VULKAN_F32_INPUTS);
                scores = modules::ScaleBiasModule({1.f / std::sqrt(float(dim_head)), 0.f}).build(build, scores);
                const auto probabilities = modules::SoftmaxModule().build(build, scores);
                auto values = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(build, v);
                values = core::ensure_backend_addressable_layout(build, values);
                context = core::wrap_tensor(ggml_mul_mat(ctx_, values.tensor, probabilities.tensor),
                    core::TensorShape::from_dims({1, heads, N, dim_head}));
                ggml_mul_mat_set_prec(context.tensor, GGML_PREC_F32);
                ggml_mul_mat_set_lowering(context.tensor, GGML_MUL_MAT_LOWERING_VULKAN_F32_INPUTS);
                context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, context);
            } else if (execution.backend_type() == core::BackendType::Metal) {
                // Each query is independent; keep both attention products on
                // Metal's full-range vector path without changing softmax.
                const modules::ScaledDotProductAttentionModule attention({dim_head,
                    modules::ScaledDotProductAttentionLowering::Explicit});
                for (int row = 0; row < N; row += 8) {
                    auto queries = modules::SliceModule({2, row, std::min(8, N - row)}).build(build, q);
                    auto values = attention.build(build, queries, k, v);
                    context = row == 0 ? values : modules::ConcatModule({1}).build(build, context, values);
                }
            } else {
                context = modules::ScaledDotProductAttentionModule({dim_head,
                    modules::ScaledDotProductAttentionLowering::Explicit}).build(build, q, k, v);
            }
            auto merged = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, context),
                core::TensorShape::from_dims({N, inner}));
            latents = modules::ResidualAddModule().build(build, latents, linear(merged, prefix + ".0.to_out", D, false, true));
            const int width = weights.at(prefix + ".1.0.weight").shape.at(0) / 2;
            auto ff = linear(latents, prefix + ".1.0", width * 2, true, true);
            auto gated = modules::GegluModule().build(build, ff);
            latents = modules::ResidualAddModule().build(build, latents, linear(gated, prefix + ".1.2", D, true, true));
        }
        // Upstream normalizes each latent with F.normalize, then applies sqrt(D)
        // and learned gamma. Additive-epsilon RMSNorm has different semantics.
        auto normalized = core::wrap_tensor(ggml_l2_norm(ctx_, latents.tensor, 1e-12f), latents.shape);
        normalized = modules::ScaleBiasModule({std::sqrt(float(D)), 0.f}).build(build, normalized);
        auto gamma = core::reshape_tensor(build, weights.at("gpt.perc.norm.gamma"), core::TensorShape::from_dims({1, D}));
        output_ = modules::MulModule().build(build, normalized,
            modules::RepeatModule({normalized.shape}).build(build, gamma)).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_, 8192, false);
        ggml_build_forward_expand(graph_, output_);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        allocator_owner_.reset(allocator_);
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("XTTS v2 conditioning graph allocation failed");
        }
    }
    ~ConditioningGraph() {
        core::release_backend_graph_resources(execution_.backend(), graph_);

    }
    int frames() const { return frames_; }
    std::vector<float> run(const std::vector<float> & mel) {
        ggml_backend_tensor_set(input_, mel.data(), 0, mel.size() * sizeof(float));
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("XTTS v2 conditioning compute failed");
        }
        std::vector<float> result(ggml_nelements(output_));
        ggml_backend_tensor_get(output_, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
    std::vector<float> encoder_output() const {
        std::vector<float> result(ggml_nelements(encoder_output_));
        ggml_backend_tensor_get(encoder_output_, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
private:
    core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_owner_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_owner_{nullptr, ggml_gallocr_free};
    int frames_;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * encoder_output_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

struct GptKV {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::unique_ptr<runtime::TransformerKVCache> state;
    int n_ctx = 0;
    void release() {
        state.reset();
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        n_ctx = 0;
    }
    ~GptKV() { release(); }
    bool ensure(ggml_backend_t be, const XttSv2HyperParams& hp, int need) {
        if (n_ctx >= need && buf) return true;
        release();
        const int nc = round_up_i(need, KV_BUCKET);
        ggml_init_params ip = { ggml_tensor_overhead() * (size_t)(2 * hp.n_layers + 8), nullptr, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        core::ModuleBuildContext build{ctx, "xtts_v2.kv"};
        const auto shape = core::TensorShape::from_dims({1, nc, hp.n_heads, hp.head_dim});
        std::vector<core::TensorValue> keys, values;
        for (uint32_t layer = 0; layer < hp.n_layers; ++layer) {
            keys.push_back(core::make_tensor(build, GGML_TYPE_F16, shape));
            values.push_back(core::make_tensor(build, GGML_TYPE_F16, shape));
        }
        runtime::TransformerKVCacheOptions options;
        options.allow_f16_storage = true;
        options.lazy_import_scratch = true;
        state = std::make_unique<runtime::TransformerKVCache>(nc, hp.d_model, std::move(keys), std::move(values), options);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { release(); return false; }
        ggml_backend_buffer_clear(buf, 0);
        n_ctx = nc;
        return true;
    }
};

// One compiled graph for a fixed (n_tok, n_kv) shape.
struct GptGraph {
    std::vector<uint8_t> meta;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_tensor * conditioning = nullptr, * text_ids = nullptr, * text_positions = nullptr;
    ggml_tensor * mel_id = nullptr, * mel_position = nullptr;
    ggml_tensor * idx = nullptr, * mask = nullptr, * logits = nullptr;
    ggml_tensor *hidden = nullptr;   // post ln_f+final_norm last-position state (pre-logits)
    std::vector<ggml_fp16_t> mask_scratch;
    int n_tok = 0, n_kv = 0;
    void release() {
        if (gf != nullptr && backend != nullptr) core::release_backend_graph_resources(backend, gf);
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        gf = nullptr; conditioning = text_ids = text_positions = mel_id = mel_position = nullptr;
        idx = mask = logits = hidden = nullptr; n_tok = n_kv = 0;
    }
    ~GptGraph() { release(); }
};

static bool gpt_build_graph(GptGraph& g, ggml_backend_t be, const GptWeights& m, const XttSv2HyperParams& hp,
                            const core::BackendConfig & backend_config, GptKV& kv, int n_tok, int n_kv) {
    g.release();
    g.backend = be;
    const int D = (int)hp.d_model, H = (int)hp.n_heads, HD = (int)hp.head_dim, L = (int)hp.n_layers;
    const size_t max_nodes = 4096;
    g.meta.resize(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false) + (1u << 20));
    ggml_init_params ip = { g.meta.size(), g.meta.data(), true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    ggml_context* c = g.ctx;
    core::ModuleBuildContext build{c, "xtts_v2.gpt", backend_config.type};
    const auto activation_shape = core::TensorShape::from_dims({n_tok, D});
    const modules::LayerNormModule normalization({D, 1e-5f, true, true, true});
    const modules::ScaledDotProductAttentionModule attention({HD,
        modules::ScaledDotProductAttentionLowering::FlashPreserveViews, GGML_PREC_DEFAULT});
    g.gf = ggml_new_graph_custom(c, max_nodes, false);
    g.n_tok = n_tok; g.n_kv = n_kv;

    g.mel_id = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    g.mel_position = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_input(g.mel_id);
    ggml_set_input(g.mel_position);
    auto mel = modules::EmbeddingModule({hp.mel_vocab_size, D}).build(build,
        core::wrap_tensor(g.mel_id, core::TensorShape::from_dims({1}), GGML_TYPE_I32),
        m.mel_embedding);
    auto mel_pos = modules::EmbeddingModule({hp.mel_pos_size, D}).build(build,
        core::wrap_tensor(g.mel_position, core::TensorShape::from_dims({1}), GGML_TYPE_I32),
        m.mel_position);
    auto input = modules::AddModule().build(build, mel, mel_pos);
    if (n_tok > 1) {
        const int conditioning_steps = hp.perceiver_n_latents;
        const int text_steps = n_tok - conditioning_steps - 1;
        g.conditioning = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, conditioning_steps);
        g.text_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, text_steps);
        g.text_positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, text_steps);
        ggml_set_input(g.conditioning);
        ggml_set_input(g.text_ids);
        ggml_set_input(g.text_positions);
        auto text = modules::EmbeddingModule({hp.text_vocab_size, D}).build(build,
            core::wrap_tensor(g.text_ids, core::TensorShape::from_dims({text_steps}), GGML_TYPE_I32),
            m.text_embedding);
        auto text_pos = modules::EmbeddingModule({hp.text_pos_size, D}).build(build,
            core::wrap_tensor(g.text_positions, core::TensorShape::from_dims({text_steps}), GGML_TYPE_I32),
            m.text_position);
        text = modules::AddModule().build(build, text, text_pos);
        auto prefix = modules::ConcatModule({0}).build(build,
            core::wrap_tensor(g.conditioning, core::TensorShape::from_dims({conditioning_steps, D})), text);
        input = modules::ConcatModule({0}).build(build, prefix, input);
    }
    g.idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);      ggml_set_name(g.idx, "kv_idx");   ggml_set_input(g.idx);
    g.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok); ggml_set_name(g.mask, "kq_mask"); ggml_set_input(g.mask);

    ggml_tensor* x = input.tensor;
    for (int l = 0; l < L; l++) {
        const GptBlock& b = m.blocks[l];
        ggml_tensor* cur = normalization.build(build, core::wrap_tensor(x, activation_shape),
            b.attention_norm).tensor;
        auto qkv = modules::LinearModule({D, 3 * D, true}).build(build,
            core::wrap_tensor(cur, activation_shape),
            b.qkv);
        const auto heads = core::TensorShape::from_dims({1, n_tok, H, HD});
        auto q = modules::SliceModule({1, 0, D}).build(build, qkv);
        auto k = modules::SliceModule({1, D, D}).build(build, qkv);
        auto v = modules::SliceModule({1, 2 * D, D}).build(build, qkv);
        q = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, q), heads);
        k = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, k), heads);
        v = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, v), heads);
        const auto indices = core::wrap_tensor(g.idx, core::TensorShape::from_dims({n_tok}), GGML_TYPE_I32);
        const modules::FastKVSetRowsModule append({modules::FastKVSetRowsMode::BackendViewOptimized});
        auto cached_key = append.build_block(build, kv.state->key_tensor(l), k, indices);
        auto cached_value = append.build_block(build, kv.state->value_tensor(l), v, indices);
        cached_key = modules::SliceModule({1, 0, n_kv}).build(build, cached_key);
        cached_value = modules::SliceModule({1, 0, n_kv}).build(build, cached_value);
        q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, q);
        cached_key = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, cached_key);
        cached_value = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, cached_value);
        auto context = attention.build(build, q, cached_key, cached_value,
            core::wrap_tensor(g.mask, core::TensorShape::from_dims({n_tok, n_kv}), GGML_TYPE_F16));
        auto merged = core::reshape_tensor(build, context, activation_shape);
        ggml_tensor * a = merged.tensor;
        a = modules::LinearModule({D, D, true}).build(build, core::wrap_tensor(a, activation_shape),
            b.attention_output).tensor;
        x = modules::ResidualAddModule().build(build, core::wrap_tensor(x, activation_shape), core::wrap_tensor(a, activation_shape)).tensor;

        auto normalized = normalization.build(build, core::wrap_tensor(x, activation_shape),
            b.feedforward_norm);
        cur = modules::FeedForwardModule({D, hp.ff_dim, true, modules::GeluApproximation::Tanh}).build(build, normalized,
            b.feedforward).tensor;
        x = modules::ResidualAddModule().build(build, core::wrap_tensor(x, activation_shape), core::wrap_tensor(cur, activation_shape)).tensor;
    }
    // Only the last position feeds the sampler. ln_f then final_norm (HF GPT2Model
    // applies ln_f itself, then the XTTS head applies final_norm again), then mel_head.
    ggml_tensor * last = modules::SliceModule({0, n_tok - 1, 1}).build(build, core::wrap_tensor(x, activation_shape)).tensor;
    const auto last_shape = core::TensorShape::from_dims({1, D});
    last = normalization.build(build, core::wrap_tensor(last, last_shape),
        m.decoder_norm).tensor;
    last = normalization.build(build, core::wrap_tensor(last, last_shape),
        m.final_norm).tensor;
    g.hidden = last; ggml_set_name(last, "mel_hidden"); ggml_set_output(last);
    ggml_tensor * lg = modules::LinearModule({D, hp.mel_vocab_size, true}).build(build,
        core::wrap_tensor(last, last_shape),
        m.mel_head).tensor;
    g.logits = lg; ggml_set_name(lg, "mel_logits"); ggml_set_output(lg);
    ggml_build_forward_expand(g.gf, lg);

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!g.galloc || !ggml_gallocr_alloc_graph(g.galloc, g.gf)) { g.release(); return false; }
    return true;
}

// Causal mask for query rows at absolute positions pos0..pos0+n_tok-1 over n_kv keys.
static void gpt_fill_mask(GptGraph& g, int pos0) {
    if (g.n_tok == 1) {
        modules::write_decoder_cached_step_mask(g.mask, g.mask_scratch, g.n_kv, pos0, pos0);
        return;
    }
    const auto compact = modules::causal_suffix_mask_values(1, g.n_tok, pos0);
    g.mask_scratch.assign(static_cast<size_t>(g.n_kv) * g.n_tok, ggml_fp32_to_fp16(-INFINITY));
    const int width = pos0 + g.n_tok;
    for (int row = 0; row < g.n_tok; ++row) {
        std::copy_n(compact.data() + row * width, width, g.mask_scratch.data() + row * g.n_kv);
    }
    ggml_backend_tensor_set(g.mask, g.mask_scratch.data(), 0, g.mask_scratch.size() * sizeof(ggml_fp16_t));
}

// GPT-2 AR generation. Sequence layout:
//   [cond_latents (no pos emb)] -> [text tokens + text_pos] -> [start_mel + mel_pos] -> [mel codes + mel_pos]
// Returns generated mel codes (including the stop token if it was sampled, as before).
static std::vector<int> gpt_gen(ggml_backend_t be, GptKV& kv, GptGraph & prefill_graph,
        std::map<int, std::unique_ptr<GptGraph>> & decode_graphs, const GptWeights& m, const XttSv2HyperParams& hp,
        const std::vector<int>& tids, const std::vector<float>& cl, float temp, int top_k, float top_p, float rep_pen,
        int max_n, const core::BackendConfig & backend_config, std::mt19937 & rng, std::vector<float>* hiddens_out) {
    std::vector<int> codes;
    const int D = (int)hp.d_model, V = (int)hp.mel_vocab_size;

    if (tids.size() > hp.text_pos_size || max_n > static_cast<int>(hp.mel_pos_size)) {
        throw std::runtime_error("XTTS v2 token sequence exceeds positional embedding capacity");
    }
    // --- Prefix embeddings: [cond_latents | text+pos | start_mel+mel_pos0] ---
    const int ncl = (int)(cl.size() / hp.d_model);
    const int Tpre = ncl + (int)tids.size() + 1;
    if (Tpre + max_n + 1 > kv.n_ctx) {
        prefill_graph.release();
        decode_graphs.clear();
    }
    if (!kv.ensure(be, hp, Tpre + max_n + 1)) throw std::runtime_error("XTTS v2 KV cache allocation failed");
    kv.state->retain_prefix(0);
    std::vector<float> logits(V, 0.f);
    auto run = [&](GptGraph& g, int32_t token, int32_t mel_position, int pos0, std::vector<float>* hid_out = nullptr) -> bool {
        const auto idx = modules::decoder_position_ids(g.n_tok, pos0);
        ggml_backend_tensor_set(g.mel_id, &token, 0, sizeof(token));
        ggml_backend_tensor_set(g.mel_position, &mel_position, 0, sizeof(mel_position));
        ggml_backend_tensor_set(g.idx, idx.data(), 0, idx.size() * sizeof(int32_t));
        if (core::compute_backend_graph(be, g.gf) != GGML_STATUS_SUCCESS) return false;
        kv.state->advance_after_direct_append(g.n_tok);
        ggml_backend_tensor_get(g.logits, logits.data(), 0, (size_t)V * sizeof(float));
        if (hid_out) {
            const size_t prev = hid_out->size();
            hid_out->resize(prev + D);
            ggml_backend_tensor_get(g.hidden, hid_out->data() + prev, 0, (size_t)D * sizeof(float));
        }
        return true;
    };

    // --- Prefill (one pass over the whole prefix) ---
    {
        auto & pg = prefill_graph;
        if ((pg.gf == nullptr || pg.n_tok != Tpre || pg.n_kv != round_up_i(Tpre, KV_BUCKET)) &&
            !gpt_build_graph(pg, be, m, hp, backend_config, kv, Tpre, round_up_i(Tpre, KV_BUCKET))) {
            throw std::runtime_error("XTTS v2 prefill graph build failed");
        }
        std::vector<int32_t> text_ids(tids.begin(), tids.end()), positions(tids.size());
        std::iota(positions.begin(), positions.end(), 0);
        ggml_backend_tensor_set(pg.conditioning, cl.data(), 0, cl.size() * sizeof(float));
        ggml_backend_tensor_set(pg.text_ids, text_ids.data(), 0, text_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pg.text_positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        gpt_fill_mask(pg, 0);
        if (!run(pg, hp.start_mel_token, 0, 0, hiddens_out)) throw std::runtime_error("XTTS v2 prefill compute failed");
    }

    // --- Decode ---
    int mel_pos = 1;                // start_mel consumed mel position 0 during prefill
    sampling::HfSampler sampler;
    sampling::HfSamplerScratch scratch;
    sampling::HfSamplingOptions sampling_options;
    sampling_options.do_sample = temp > 0;
    sampling_options.temperature = temp > 0 ? temp : 1.f;
    sampling_options.top_k = top_k;
    sampling_options.top_p = top_p;
    sampling_options.repetition_penalty = rep_pen;
    for (int g = 0; g < max_n; g++) {
        if ((int)codes.size() >= max_n) break;
        // ---- sample ----
        const int st = sampler.sample(logits, codes, sampling_options, scratch, rng, nullptr, "XTTS v2 mel decoder");
        codes.push_back(st);
        if (st == static_cast<int>(hp.stop_mel_token) || g + 1 == max_n) break;

        const int seq_pos = static_cast<int>(kv.state->current_end());
        const int want_kv = std::min(kv.n_ctx, round_up_i(seq_pos + 1, KV_BUCKET));
        auto & graph = decode_graphs[want_kv];
        if (!graph) {
            graph = std::make_unique<GptGraph>();
            if (!gpt_build_graph(*graph, be, m, hp, backend_config, kv, 1, want_kv)) {
                throw std::runtime_error("XTTS v2 decode graph build failed");
            }
        }
        auto & dg = *graph;
        // Refill the whole mask every step: gallocr may reassign the mask tensor's
        // buffer region between computes, so incremental single-element updates are
        // not durable. The refill is O(n_kv) F16 bytes (~512 B) - negligible.
        gpt_fill_mask(dg, seq_pos);
        if (!run(dg, st, mel_pos++, seq_pos, hiddens_out)) throw std::runtime_error("XTTS v2 decode compute failed");
    }
    return codes;
}
}  // anonymous namespace

struct XttSv2GptEngine::Impl {
    std::shared_ptr<const assets::TensorSource> source;
    std::shared_ptr<core::BackendWeightStore> weight_store;
    std::map<std::string, core::TensorValue> tensors;
    GptWeights model;
    XttSv2HyperParams hp;
    core::BackendConfig backend_config;
    ggml_backend_t backend = nullptr;
    GptKV kv;
    GptGraph prefill_graph;
    std::map<int, std::unique_ptr<GptGraph>> decode_graphs;
    std::vector<float> mel_norms;
    MelFrontend frontend{MelPurpose::Conditioning};
    core::ExecutionContext * execution = nullptr;
    std::unique_ptr<ConditioningGraph> conditioning_graph;
    ~Impl() {
        conditioning_graph.reset();
        prefill_graph.release();
        decode_graphs.clear();
        kv.release();
        weight_store.reset();
    }
};

XttSv2GptEngine::XttSv2GptEngine() = default;
XttSv2GptEngine::~XttSv2GptEngine() = default;

bool XttSv2GptEngine::load(const XttSv2Assets & assets, core::ExecutionContext & execution) {
    impl_ = std::make_unique<Impl>();
    impl_->execution = &execution;
    impl_->backend_config = execution.config();
    impl_->backend = execution.backend();
    impl_->hp = assets.hyper_params;
    impl_->mel_norms = assets.mel_stats;
    impl_->source = assets.gpt_source;
    impl_->weight_store = std::make_shared<core::BackendWeightStore>(
        impl_->backend, execution.backend_type(), "xtts_v2.gpt.weights", 8 * 1024 * 1024);
    for (const auto & tensor : impl_->source->tensors()) {
        auto value = impl_->weight_store->load_tensor(
            *impl_->source, tensor.name,
            tensor.name.rfind("gpt.cond_enc.", 0) == 0 || tensor.name.rfind("gpt.perc.", 0) == 0
                ? assets::TensorStorageType::F32 : assets::TensorStorageType::Native, tensor.shape);
        ggml_set_name(value.tensor, tensor.name.c_str());
        impl_->tensors.emplace(tensor.name, value);
    }
    impl_->weight_store->upload();
    impl_->model = bind_gpt_weights(impl_->tensors, impl_->hp);
    return true;
}

bool XttSv2GptEngine::compute_conditioning(const float* ref_pcm, int ref_ns_in, std::vector<float>& out) {
    if (!impl_) throw std::runtime_error("XTTS v2 GPT weights are not loaded");
    auto& hp = impl_->hp;
    // Reference get_gpt_cond_latents(audio, sr, length=gpt_cond_len=30, chunk_length=gpt_cond_chunk_len=6):
    //   audio = audio[:, :22050*length]           (truncate to 30 s)
    //   for each 6-second chunk (skip chunks < 0.33 s):
    //       mel_chunk = wav_to_mel_cloning(chunk, n_fft=2048, hop=256, win=1024)
    //       style_emb = conditioning_encoder + perceiver  -> [32, 1024]
    //   cond_latent = mean(style_embs)
    int ref_ns = ref_ns_in;
    int max_ns = 22050 * 30;   // gpt_cond_len = 30
    if (ref_ns > max_ns) ref_ns = max_ns;
    const int chunk_ns = 22050 * 6;    // gpt_cond_chunk_len = 6 s
    const int min_chunk_ns = (int)(22050 * 0.33f);
    const std::vector<float>& mel_norms = impl_->mel_norms;
    // Process 6-second chunks: mel -> cond encoder -> perceiver -> [32,1024]; average.
    // Chunks reuse the session-owned graph sequentially and are averaged.
    struct ChunkRange { int cs, cn; };
    std::vector<ChunkRange> ranges;
    for (int cs = 0; cs < ref_ns; cs += chunk_ns) {
        int cn = std::min(chunk_ns, ref_ns - cs);
        if (cn < min_chunk_ns) continue;    // reference skips chunks < 0.33 s
        ranges.push_back({cs, cn});
    }
    if (ranges.empty()) throw std::runtime_error("XTTS v2 reference must contain at least 0.33 seconds");
    std::vector<std::vector<float>> chunk_latents(ranges.size());
    const int d = hp.d_model, nlat = hp.perceiver_n_latents > 0 ? hp.perceiver_n_latents : 32;
    for (int ci = 0; ci < (int)ranges.size(); ci++) {
        const int cs = ranges[ci].cs, cn = ranges[ci].cn;
        std::vector<float> mel = impl_->frontend.compute(
            std::vector<float>(ref_pcm + cs, ref_pcm + cs + cn), impl_->backend_config.threads);
        engine::debug::trace_log_f32("xtts_v2.conditioning.mel", {80, static_cast<int64_t>(mel.size() / 80)}, mel);
        const int T = (int)mel.size() / 80;
        for (int mch = 0; mch < 80; mch++) for (int t = 0; t < T; t++)
                mel[(size_t)mch * T + t] /= mel_norms[mch];
        // The frontend returns feature-major [80,T]; the graph input is [T,80].
        std::vector<float> melT((size_t)T * 80);
        for (int mch = 0; mch < 80; mch++) for (int t = 0; t < T; t++)
            melT[(size_t)t * 80 + mch] = mel[(size_t)mch * T + t];
        if (!impl_->conditioning_graph || impl_->conditioning_graph->frames() != T) {
            impl_->conditioning_graph = std::make_unique<ConditioningGraph>(*impl_->execution, impl_->tensors, hp, T);
        }
        auto pl = impl_->conditioning_graph->run(melT);
        engine::debug::trace_log_f32("xtts_v2.conditioning.encoder", {T, hp.d_model}, impl_->conditioning_graph->encoder_output());
        engine::debug::trace_log_f32("xtts_v2.conditioning.latents", {hp.perceiver_n_latents, hp.d_model}, pl);
        chunk_latents[ci] = std::move(pl);
    }
    std::vector<float> acc((size_t)nlat * d, 0.0f);
    const int n_chunks = static_cast<int>(chunk_latents.size());
    for (auto& pl : chunk_latents) {
        for (size_t i = 0; i < acc.size(); i++) acc[i] += pl[i];
    }
    for (auto& v : acc) v /= (float)n_chunks;
    out = std::move(acc);
    return true;
}
std::vector<int32_t> XttSv2GptEngine::generate_mel_codes(const std::vector<int32_t>& text_tokens, const std::vector<float>& cond_latents,
        float temperature, int top_k, float top_p, float rep_penalty, int max_tokens,
        std::mt19937 & rng, std::vector<float>* decode_hiddens) {
    if (!impl_) throw std::runtime_error("XTTS v2 GPT weights are not loaded");
    auto codes = gpt_gen(impl_->backend, impl_->kv, impl_->prefill_graph, impl_->decode_graphs, impl_->model, impl_->hp,
        std::vector<int>(text_tokens.begin(), text_tokens.end()), cond_latents,
        temperature, top_k, top_p, rep_penalty, max_tokens,
        impl_->backend_config, rng, decode_hiddens);
    return std::vector<int32_t>(codes.begin(), codes.end());
}
}  // namespace engine::models::xtts_v2
