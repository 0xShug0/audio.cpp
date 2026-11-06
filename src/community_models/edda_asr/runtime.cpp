#include "engine/community_models/edda_asr/model.h"

#include "engine/framework/debug/trace.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::community_models::edda_asr {
namespace {

using core::TensorShape;
using core::TensorValue;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int64_t kMaxBeams = 16;

struct Graph {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    core::HostGraphPlan plan;

    explicit Graph(core::ExecutionContext & execution) : execution(execution) {
        context.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!context) {
            throw std::runtime_error("Edda graph context allocation failed");
        }
        graph = ggml_new_graph_custom(context.get(), 16384, false);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
    }
    ~Graph() {
        plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
    void allocate() {
        core::validate_backend_graph_supported(execution.backend(), graph, "Edda");
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("Edda graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan, "Edda") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Edda graph execution failed");
        }
    }
    core::ModuleBuildContext build_context(const char * name) const {
        core::ModuleBuildContext ctx{};
        ctx.ggml = context.get();
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = name;
        return ctx;
    }
};

TensorValue wrap(ggml_tensor * tensor, std::initializer_list<int64_t> dims) {
    return core::wrap_tensor(tensor, TensorShape::from_dims(dims), tensor->type);
}

TensorValue linear(core::ModuleBuildContext & ctx, const TensorValue & x, const TensorValue & weight,
    const std::optional<TensorValue> & bias = std::nullopt) {
    const auto in = weight.shape.dims[1], out = weight.shape.dims[0];
    return modules::LinearModule({in, out, bias.has_value()}).build(ctx, x, {weight, bias});
}

// [1, T, d] rows -> flat [T, d] for row scatter into a [rows, d] cache view.
ggml_tensor * rows_2d(core::ModuleBuildContext & ctx, const TensorValue & x) {
    return ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, x.tensor), x.shape.dims[2], x.shape.dims[1]);
}

// Multi-head attention of [1, T, d] queries over a [1, rows, H, hd] F16 cache with an
// additive F16 [T, rows] mask; returns [1, T, d].
TensorValue cached_attention(core::ModuleBuildContext & ctx, const TensorValue & q, ggml_tensor * key,
    ggml_tensor * value, ggml_tensor * mask, int64_t heads) {
    const int64_t t = q.shape.dims[1], d = q.shape.dims[2], hd = d / heads;
    auto * qh = ggml_permute(ctx.ggml, ggml_reshape_4d(ctx.ggml, ggml_cont(ctx.ggml, q.tensor), hd, heads, t, 1), 0, 2, 1, 3);
    auto * kh = ggml_permute(ctx.ggml, key, 0, 2, 1, 3);
    auto * vh = ggml_permute(ctx.ggml, value, 0, 2, 1, 3);
    auto * out = ggml_flash_attn_ext(ctx.ggml, qh, kh, vh, mask, 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    return wrap(ggml_reshape_3d(ctx.ggml, out, d, t, 1), {1, t, d});
}

// Whisper encoder with packed QKV projections and flash attention over F16 keys and values.
TensorValue build_encoder(core::ModuleBuildContext & ctx, const TensorValue & features, const EncoderWeights & w,
    const modules::WhisperEmbeddingConfig & config) {
    const int64_t d = config.n_audio_state, h = config.n_audio_head, hd = d / h;
    auto x = modules::Conv1dModule({config.n_mels, d, 3, 1, 1, 1, true}).build(ctx, features, w.conv1);
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
    x = modules::Conv1dModule({d, d, 3, 2, 1, 1, true}).build(ctx, x, w.conv2);
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
    const int64_t frames = x.shape.dims[2];
    x = wrap(ggml_add(ctx.ggml, ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, ggml_reshape_2d(ctx.ggml, x.tensor, frames, d))),
        w.positions.tensor), {1, frames, d});
    const modules::LayerNormModule norm({d, config.layer_norm_eps, true, true});
    const modules::FeedForwardModule ffn({d, 4 * d, true, modules::GeluApproximation::ExactErf});
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (const auto & layer : w.layers) {
        auto qkv = linear(ctx, norm.build(ctx, x, layer.attention_norm), layer.qkv_weight, layer.qkv_bias);
        const auto lane = [&](int64_t index, ggml_type type) {
            auto * view = ggml_view_2d(ctx.ggml, qkv.tensor, d, frames, qkv.tensor->nb[1],
                static_cast<size_t>(index * d) * ggml_element_size(qkv.tensor));
            auto * heads = type == GGML_TYPE_F32 ? ggml_cont(ctx.ggml, view) : ggml_cast(ctx.ggml, view, type);
            return ggml_permute(ctx.ggml, ggml_reshape_4d(ctx.ggml, heads, hd, h, frames, 1), 0, 2, 1, 3);
        };
        auto * attended = ggml_flash_attn_ext(ctx.ggml, lane(0, GGML_TYPE_F32), lane(1, GGML_TYPE_F16),
            lane(2, GGML_TYPE_F16), nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        auto out = linear(ctx, wrap(ggml_reshape_3d(ctx.ggml, attended, d, frames, 1), {1, frames, d}),
            layer.out_weight, layer.out_bias);
        x = modules::AddModule().build(ctx, x, out);
        x = modules::AddModule().build(ctx, x, ffn.build(ctx, norm.build(ctx, x, layer.mlp_norm), layer.mlp));
    }
    return norm.build(ctx, x, w.final_norm);
}

// Target self-attention cache. Rows are KV slots; a query's mask row selects the slots it attends to.
struct KVCache {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::vector<TensorValue> keys, values;
    int64_t rows = 0;
};

std::unique_ptr<KVCache> make_cache(core::ExecutionContext & execution, int64_t layers, int64_t rows, int64_t heads,
    int64_t head_dim) {
    auto cache = std::make_unique<KVCache>();
    cache->rows = rows;
    cache->context.reset(ggml_init({static_cast<size_t>(2 * layers + 8) * ggml_tensor_overhead(), nullptr, true}));
    core::ModuleBuildContext ctx{};
    ctx.ggml = cache->context.get();
    ctx.backend_type = execution.backend_type();
    for (int64_t i = 0; i < layers; ++i) {
        cache->keys.push_back(core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, rows, heads, head_dim})));
        cache->values.push_back(core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, rows, heads, head_dim})));
    }
    cache->buffer.reset(ggml_backend_alloc_ctx_tensors(cache->context.get(), execution.backend()));
    if (!cache->buffer) {
        throw std::runtime_error("Edda KV cache allocation failed");
    }
    ggml_backend_buffer_clear(cache->buffer.get(), 0);
    return cache;
}

}  // namespace

struct EddaRuntime::State {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    int64_t positions = 0, frames = 0, block = 0, context_rows = 0;
    TensorValue features, memory_mask, cross_mask, suppress_bias, staging;
    std::unique_ptr<KVCache> cache, beam_cache;
    std::vector<modules::CrossAttentionKeyValue> cross, draft_cross;
    std::vector<TensorValue> draft_keys, draft_values;
    Graph encoder;
    bool flash_cross = false;

    explicit State(core::ExecutionContext & execution) : encoder(execution) {}
};

struct EddaRuntime::TargetGraph {
    Graph graph;
    int64_t tokens = 0;
    TensorValue token, position, slot, mask, logits, argmax;
    explicit TargetGraph(core::ExecutionContext & execution) : graph(execution) {}
};

struct EddaRuntime::DrafterGraph {
    Graph graph;
    int64_t block = 0;
    TensorValue anchor, position, context_slot, block_slot, mask, argmax, top_ids, top_logits, projected;
    explicit DrafterGraph(core::ExecutionContext & execution) : graph(execution) {}
};

EddaRuntime::EddaRuntime(const EddaAssets & assets, const EddaWeights & weights, core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution),
      frontend_({16000, 400, 160, assets.encoder.n_mels, audio::STFTFamily::Kokoro}) {
    state_ = std::make_unique<State>(execution);
    auto & s = *state_;
    const int64_t d = assets.encoder.n_audio_state, h = assets.decoder_heads, hd = d / h;
    s.positions = assets.max_positions;
    s.frames = assets.encoder.n_audio_ctx;
    s.block = weights.drafter ? assets.drafter->block : 1;
    // Draft cache rows: target positions, then the block being drafted, then one discard row.
    s.context_rows = ((s.positions + s.block + 1 + 63) / 64) * 64;
    const auto backend = execution.backend_type();
    s.flash_cross = backend == core::BackendType::Cuda || backend == core::BackendType::Hip;
    s.context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
    core::ModuleBuildContext ctx{};
    ctx.ggml = s.context.get();
    ctx.backend_type = backend;
    const int64_t staging_rows = std::max<int64_t>(s.block, 8);
    const int64_t query_rows = std::max<int64_t>(s.block, kMaxBeams);
    s.features = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, assets.encoder.n_mels, 2 * s.frames}));
    s.memory_mask = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, s.frames}));
    s.cross_mask = core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, 1, query_rows, s.frames}));
    s.suppress_bias = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({assets.vocabulary_size}));
    s.staging = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({staging_rows, d}));
    s.cache = make_cache(execution, assets.decoder_layers, s.positions, h, hd);
    for (int64_t i = 0; i < assets.decoder_layers; ++i) {
        s.cross.push_back({core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, h, s.frames, hd})),
                           core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, h, s.frames, hd}))});
    }
    if (weights.drafter) {
        for (size_t i = 0; i < weights.drafter->layers.size(); ++i) {
            s.draft_keys.push_back(core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, s.context_rows, h, hd})));
            s.draft_values.push_back(core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, s.context_rows, h, hd})));
            s.draft_cross.push_back({core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, h, s.frames, hd})),
                                     core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, h, s.frames, hd}))});
        }
    }
    s.buffer.reset(ggml_backend_alloc_ctx_tensors(s.context.get(), execution.backend()));
    if (!s.buffer) {
        throw std::runtime_error("Edda state allocation failed");
    }
    ggml_backend_buffer_clear(s.buffer.get(), 0);
    core::write_tensor_i32(s.memory_mask, std::vector<int32_t>(static_cast<size_t>(s.frames), 1));
    core::write_tensor_f16(s.cross_mask, std::vector<float>(static_cast<size_t>(query_rows * s.frames), 0.0f));
    std::vector<float> bias(static_cast<size_t>(assets.vocabulary_size), 0.0f);
    for (const auto id : assets.suppress) {
        bias[static_cast<size_t>(id)] = kNegInf;
    }
    core::write_tensor_f32(s.suppress_bias, bias);

    auto & g = s.encoder;
    auto ectx = g.build_context("edda.encoder");
    ggml_set_input(s.features.tensor);
    const auto encoded = build_encoder(ectx, s.features, weights.encoder, assets.encoder);
    modules::AttentionConfig cross_config{d, h, true};
    cross_config.use_packed_kv = true;
    const auto project = [&](const modules::AttentionWeights & w, const modules::CrossAttentionKeyValue & dst) {
        const auto kv = modules::CrossAttentionModule(cross_config).build_key_value(ectx, encoded, w);
        ggml_build_forward_expand(g.graph, ggml_cpy(ectx.ggml, kv.key.tensor, dst.key.tensor));
        ggml_build_forward_expand(g.graph, ggml_cpy(ectx.ggml, kv.value.tensor, dst.value.tensor));
    };
    for (size_t i = 0; i < weights.decoder.size(); ++i) {
        project(weights.decoder[i].cross_attention, s.cross[i]);
    }
    if (weights.drafter) {
        for (size_t i = 0; i < weights.drafter->layers.size(); ++i) {
            project(weights.drafter->layers[i].cross_attention, s.draft_cross[i]);
        }
    }
    g.allocate();
}

EddaRuntime::~EddaRuntime() {
    ggml_backend_synchronize(execution_.backend());
}

EddaRuntime::TargetGraph & EddaRuntime::target_graph(int64_t tokens, bool beam) {
    auto & graphs = beam ? beam_graphs_ : target_graphs_;
    if (static_cast<int64_t>(graphs.size()) <= tokens) {
        graphs.resize(static_cast<size_t>(tokens + 1));
    }
    auto & slot = graphs[static_cast<size_t>(tokens)];
    if (slot) {
        return *slot;
    }
    const auto build_start = std::chrono::steady_clock::now();
    auto & s = *state_;
    const auto & cache = beam ? *s.beam_cache : *s.cache;
    const int64_t rows = cache.rows;
    auto out = std::make_unique<TargetGraph>(execution_);
    auto & tg = *out;
    tg.tokens = tokens;
    auto ctx = tg.graph.build_context("edda.decoder");
    const int64_t d = assets_.encoder.n_audio_state, h = assets_.decoder_heads, t = tokens;
    tg.token = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, t}));
    tg.position = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, t}));
    tg.slot = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({t}));
    tg.mask = core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({t, rows}));
    for (auto input : {tg.token, tg.position, tg.slot, tg.mask}) {
        ggml_set_input(input.tensor);
    }
    auto x = wrap(ggml_add(ctx.ggml, ggml_get_rows(ctx.ggml, weights_.embedding.tensor, tg.token.tensor),
        ggml_get_rows(ctx.ggml, weights_.positions.tensor, tg.position.tensor)), {1, t, d});
    const modules::LayerNormModule norm({d, 1e-5f, true, true});
    const modules::FeedForwardModule ffn({d, assets_.decoder_ffn, true, modules::GeluApproximation::ExactErf});
    modules::AttentionConfig cross_config{d, h, true};
    cross_config.use_packed_kv = true;
    const modules::CrossAttentionModule cross(cross_config);
    ggml_tensor * cross_mask = s.flash_cross ? ggml_view_4d(ctx.ggml, s.cross_mask.tensor, s.frames, t, 1, 1,
        s.cross_mask.tensor->nb[1], s.cross_mask.tensor->nb[2], s.cross_mask.tensor->nb[3], 0) : nullptr;
    std::vector<ggml_tensor *> taps;
    const auto & tap_ids = weights_.drafter && !beam ? assets_.drafter->target_layer_ids : std::vector<int64_t>{};
    for (size_t i = 0; i < weights_.decoder.size(); ++i) {
        const auto & w = weights_.decoder[i];
        auto qkv = linear(ctx, norm.build(ctx, x, w.attention_norm), w.qkv_weight, w.qkv_bias);
        const auto lane = [&](int64_t index) {
            return wrap(ggml_view_3d(ctx.ggml, qkv.tensor, d, t, 1, qkv.tensor->nb[1], qkv.tensor->nb[2],
                static_cast<size_t>(index * d) * ggml_element_size(qkv.tensor)), {1, t, d});
        };
        auto * key2d = ggml_reshape_2d(ctx.ggml, cache.keys[i].tensor, d, rows);
        auto * value2d = ggml_reshape_2d(ctx.ggml, cache.values[i].tensor, d, rows);
        auto * key = ggml_set_rows(ctx.ggml, key2d, rows_2d(ctx, lane(1)), tg.slot.tensor);
        auto * value = ggml_set_rows(ctx.ggml, value2d, rows_2d(ctx, lane(2)), tg.slot.tensor);
        const auto hd = d / h;
        auto attended = cached_attention(ctx, lane(0), ggml_reshape_4d(ctx.ggml, key, hd, h, rows, 1),
            ggml_reshape_4d(ctx.ggml, value, hd, h, rows, 1), tg.mask.tensor, h);
        x = modules::AddModule().build(ctx, x, linear(ctx, attended, w.out_weight, w.out_bias));
        const auto y = norm.build(ctx, x, w.cross_norm);
        x = modules::AddModule().build(ctx, x, s.flash_cross
            ? cross.build_cached_flash(ctx, y, s.cross[i], w.cross_attention,
                  wrap(cross_mask, {1, 1, t, s.frames}))
            : cross.build_cached(ctx, y, s.cross[i], w.cross_attention, s.memory_mask));
        x = modules::AddModule().build(ctx, x, ffn.build(ctx, norm.build(ctx, x, w.mlp_norm), w.mlp));
        if (std::find(tap_ids.begin(), tap_ids.end(), static_cast<int64_t>(i)) != tap_ids.end()) {
            taps.push_back(ggml_cont(ctx.ggml, x.tensor));
        }
    }
    x = norm.build(ctx, x, weights_.decoder_norm);
    auto * logits = ggml_mul_mat(ctx.ggml, weights_.embedding.tensor, ggml_reshape_2d(ctx.ggml, x.tensor, d, t));
    // Raw logits for the host (beam search normalizes before suppressing); greedy argmax applies suppress_tokens.
    tg.logits = wrap(logits, {t, assets_.vocabulary_size});
    tg.argmax = wrap(ggml_argmax(ctx.ggml, ggml_add(ctx.ggml, logits, s.suppress_bias.tensor)), {t});
    ggml_set_output(tg.logits.tensor);
    ggml_set_output(tg.argmax.tensor);
    ggml_build_forward_expand(tg.graph.graph, tg.argmax.tensor);
    if (!taps.empty()) {
        // Draft context: concatenated target hidden states -> fc -> RMSNorm, staged for the drafter.
        auto * features = taps.front();
        for (size_t i = 1; i < taps.size(); ++i) {
            features = ggml_concat(ctx.ggml, features, taps[i], 0);
        }
        const auto & dw = *weights_.drafter;
        auto context = linear(ctx, wrap(features, {1, t, static_cast<int64_t>(taps.size()) * d}), dw.context_fc);
        context = modules::RMSNormModule({d, 1e-6f, true, false}).build(ctx, context, dw.context_norm);
        auto * staged = ggml_view_2d(ctx.ggml, s.staging.tensor, d, t, s.staging.tensor->nb[1], 0);
        ggml_build_forward_expand(tg.graph.graph, ggml_cpy(ctx.ggml, rows_2d(ctx, context), staged));
    }
    tg.graph.allocate();
    debug::timing_log_scalar("edda.target_graph_build_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count());
    slot = std::move(out);
    return *slot;
}

EddaRuntime::DrafterGraph & EddaRuntime::drafter_graph(int64_t block) {
    if (static_cast<int64_t>(drafter_graphs_.size()) <= block) {
        drafter_graphs_.resize(static_cast<size_t>(block + 1));
    }
    auto & slot = drafter_graphs_[static_cast<size_t>(block)];
    if (slot) {
        return *slot;
    }
    auto & s = *state_;
    const auto & config = *assets_.drafter;
    const auto & dw = *weights_.drafter;
    auto out = std::make_unique<DrafterGraph>(execution_);
    auto & dg = *out;
    dg.block = block;
    auto ctx = dg.graph.build_context("edda.drafter");
    const int64_t d = assets_.encoder.n_audio_state, h = assets_.decoder_heads, hd = d / h, b = block;
    const int64_t staging_rows = s.staging.shape.dims[0], rows = s.context_rows;
    const int64_t kernel = config.conv_kernel, groups = d / config.conv_group;
    dg.anchor = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
    dg.position = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({b}));
    dg.context_slot = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({staging_rows}));
    dg.block_slot = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({b}));
    dg.mask = core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({b, rows}));
    for (auto input : {dg.anchor, dg.position, dg.context_slot, dg.block_slot, dg.mask}) {
        ggml_set_input(input.tensor);
    }
    // Block input: the verified anchor token, then mask slots, plus target positions.
    auto * anchor = ggml_get_rows(ctx.ggml, weights_.embedding.tensor, dg.anchor.tensor);
    auto * masked = ggml_repeat(ctx.ggml, ggml_reshape_2d(ctx.ggml, dw.mask_embedding.tensor, d, 1),
        ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, d, b - 1));
    auto * embedded = ggml_add(ctx.ggml, ggml_concat(ctx.ggml, anchor, masked, 1),
        ggml_get_rows(ctx.ggml, weights_.positions.tensor, dg.position.tensor));
    auto hidden = wrap(embedded, {1, b, d});
    const auto staged = wrap(s.staging.tensor, {1, staging_rows, d});
    const modules::LayerNormModule norm({d, 1e-5f, true, true});
    const modules::FeedForwardModule ffn({d, assets_.decoder_ffn, true, modules::GeluApproximation::ExactErf});
    modules::AttentionConfig cross_config{d, h, true};
    cross_config.use_packed_kv = true;
    const modules::CrossAttentionModule cross(cross_config);
    ggml_tensor * cross_mask = s.flash_cross ? ggml_view_4d(ctx.ggml, s.cross_mask.tensor, s.frames, b, 1, 1,
        s.cross_mask.tensor->nb[1], s.cross_mask.tensor->nb[2], s.cross_mask.tensor->nb[3], 0) : nullptr;

    // DFlash2 grouped two-tap dynamic causal convolution within the block:
    // out[t, c] = sum_k (base[k, c] + dynamic[t, k, c / group]) * x[t - k, c].
    const auto conv = [&](const TensorValue & x, ggml_tensor * dynamic, const DrafterConvWeights & w, int64_t which) {
        ggml_tensor * acc = nullptr;
        auto * flat = ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, x.tensor), d, b);
        for (int64_t k = 0; k < kernel && k < b; ++k) {
            auto * shifted = k == 0 ? flat : ggml_pad_ext(ctx.ggml,
                ggml_view_2d(ctx.ggml, flat, d, b - k, flat->nb[1], 0), 0, 0, static_cast<int>(k), 0, 0, 0, 0, 0);
            auto * base = ggml_view_1d(ctx.ggml, w.base.tensor, d,
                static_cast<size_t>((which * kernel + k) * d) * sizeof(float));
            // The per-group dynamic tap broadcasts over the group's channels.
            auto * tap = ggml_view_3d(ctx.ggml, dynamic, 1, groups, b, dynamic->nb[0], dynamic->nb[3],
                static_cast<size_t>(k) * dynamic->nb[1] + static_cast<size_t>(which) * dynamic->nb[2]);
            auto * grouped = ggml_mul(ctx.ggml, ggml_reshape_3d(ctx.ggml, shifted, config.conv_group, groups, b), tap);
            auto * term = ggml_add(ctx.ggml, ggml_mul(ctx.ggml, shifted, base), ggml_reshape_2d(ctx.ggml, grouped, d, b));
            acc = acc ? ggml_add(ctx.ggml, acc, term) : term;
        }
        return wrap(acc, {1, b, d});
    };
    const auto dynamic = [&](const TensorValue & x, const DrafterConvWeights & w) {
        auto projected = linear(ctx, x, w.projection);
        return ggml_reshape_4d(ctx.ggml, ggml_cont(ctx.ggml, projected.tensor), groups, kernel, 2, b);
    };
    for (size_t i = 0; i < dw.layers.size(); ++i) {
        const auto & w = dw.layers[i];
        auto * key2d = ggml_reshape_2d(ctx.ggml, s.draft_keys[i].tensor, d, rows);
        auto * value2d = ggml_reshape_2d(ctx.ggml, s.draft_values[i].tensor, d, rows);
        // Append newly verified target context, then the block's own keys and values.
        const auto kv_weight = wrap(ggml_view_2d(ctx.ggml, w.qkv_weight.tensor, d, 2 * d, w.qkv_weight.tensor->nb[1],
            static_cast<size_t>(d) * w.qkv_weight.tensor->nb[1]), {2 * d, d});
        const auto kv_bias = wrap(ggml_view_1d(ctx.ggml, w.qkv_bias.tensor, 2 * d,
            static_cast<size_t>(d) * sizeof(float)), {2 * d});
        auto context = linear(ctx, staged, kv_weight, kv_bias);
        const auto split = [&](const TensorValue & packed, int64_t index, int64_t n) {
            return wrap(ggml_view_3d(ctx.ggml, packed.tensor, d, n, 1, packed.tensor->nb[1], packed.tensor->nb[2],
                static_cast<size_t>(index * d) * ggml_element_size(packed.tensor)), {1, n, d});
        };
        key2d = ggml_set_rows(ctx.ggml, key2d, rows_2d(ctx, split(context, 0, staging_rows)), dg.context_slot.tensor);
        value2d = ggml_set_rows(ctx.ggml, value2d, rows_2d(ctx, split(context, 1, staging_rows)),
            dg.context_slot.tensor);
        auto x = norm.build(ctx, hidden, w.attention_norm);
        auto * dyn = dynamic(x, w.attention_conv);
        x = conv(x, dyn, w.attention_conv, 0);
        const auto qkv = linear(ctx, x, w.qkv_weight, w.qkv_bias);
        key2d = ggml_set_rows(ctx.ggml, key2d, rows_2d(ctx, split(qkv, 1, b)), dg.block_slot.tensor);
        value2d = ggml_set_rows(ctx.ggml, value2d, rows_2d(ctx, split(qkv, 2, b)), dg.block_slot.tensor);
        auto attended = cached_attention(ctx, split(qkv, 0, b),
            ggml_reshape_4d(ctx.ggml, key2d, hd, h, rows, 1), ggml_reshape_4d(ctx.ggml, value2d, hd, h, rows, 1),
            dg.mask.tensor, h);
        attended = linear(ctx, attended, w.out_weight, w.out_bias);
        hidden = modules::AddModule().build(ctx, hidden, conv(attended, dyn, w.attention_conv, 1));
        const auto query = norm.build(ctx, hidden, w.cross_norm);
        hidden = modules::AddModule().build(ctx, hidden, s.flash_cross
            ? cross.build_cached_flash(ctx, query, s.draft_cross[i], w.cross_attention, wrap(cross_mask, {1, 1, b, s.frames}))
            : cross.build_cached(ctx, query, s.draft_cross[i], w.cross_attention, s.memory_mask));
        x = norm.build(ctx, hidden, w.mlp_norm);
        dyn = dynamic(x, w.mlp_conv);
        x = ffn.build(ctx, conv(x, dyn, w.mlp_conv, 0), w.mlp);
        hidden = modules::AddModule().build(ctx, hidden, conv(x, dyn, w.mlp_conv, 1));
    }
    hidden = norm.build(ctx, hidden, dw.final_norm);
    auto * drafted = ggml_cont(ctx.ggml, ggml_view_2d(ctx.ggml, hidden.tensor, d, b - 1, hidden.tensor->nb[1],
        hidden.tensor->nb[1]));
    auto * logits = ggml_mul_mat(ctx.ggml, weights_.embedding.tensor, drafted);
    dg.argmax = wrap(ggml_argmax(ctx.ggml, logits), {b - 1});
    ggml_set_output(dg.argmax.tensor);
    ggml_build_forward_expand(dg.graph.graph, dg.argmax.tensor);
    const auto top_k = std::min<int64_t>(config.selector_top_k, assets_.vocabulary_size);
    auto * ids = ggml_cont(ctx.ggml, ggml_argsort_top_k(ctx.ggml, logits, static_cast<int>(top_k)));
    auto * scores = ggml_get_rows(ctx.ggml, ggml_reshape_3d(ctx.ggml, logits, 1, assets_.vocabulary_size, b - 1), ids);
    dg.top_ids = wrap(ids, {b - 1, top_k});
    dg.top_logits = wrap(scores, {b - 1, top_k, 1});
    dg.projected = linear(ctx, wrap(drafted, {1, b - 1, d}), dw.selector_projection);
    for (auto output : {dg.top_ids, dg.top_logits, dg.projected}) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(dg.graph.graph, output.tensor);
    }
    dg.graph.allocate();
    slot = std::move(out);
    return *slot;
}

std::vector<int32_t> EddaRuntime::select_path(const DrafterGraph & graph, int32_t anchor) const {
    // DFlash2 candidate selector: greedy walk where each slot scores its top-k candidates
    // by drafter logit + <P(previous) * H(h_t), S(candidate)>.
    const auto & dw = *weights_.drafter;
    const auto rank = assets_.drafter->selector_rank;
    std::vector<int32_t> ids;
    std::vector<float> unary, projected;
    core::read_tensor_i32_into(graph.top_ids.tensor, ids);
    core::read_tensor_f32_into(graph.top_logits.tensor, unary);
    core::read_tensor_f32_into(graph.projected.tensor, projected);
    const auto slots = graph.block - 1, top_k = graph.top_ids.shape.dims[1];
    const auto row = [&](const std::vector<uint16_t> & codebook, int32_t id) {
        return reinterpret_cast<const ggml_fp16_t *>(codebook.data()) + static_cast<size_t>(id) * rank;
    };
    std::vector<float> query(static_cast<size_t>(rank));
    std::vector<int32_t> path;
    int32_t previous = anchor;
    for (int64_t t = 0; t < slots; ++t) {
        const auto * predecessor = row(dw.predecessor_codebook, previous);
        for (int64_t r = 0; r < rank; ++r) {
            query[r] = ggml_fp16_to_fp32(predecessor[r]) * projected[static_cast<size_t>(t * rank + r)];
        }
        float best = kNegInf;
        int32_t choice = ids[static_cast<size_t>(t * top_k)];
        for (int64_t k = 0; k < top_k; ++k) {
            const auto candidate = ids[static_cast<size_t>(t * top_k + k)];
            const auto * successor = row(dw.successor_codebook, candidate);
            float score = unary[static_cast<size_t>(t * top_k + k)];
            for (int64_t r = 0; r < rank; ++r) {
                score += query[r] * ggml_fp16_to_fp32(successor[r]);
            }
            if (score > best) {
                best = score;
                choice = candidate;
            }
        }
        path.push_back(choice);
        previous = choice;
    }
    return path;
}

void EddaRuntime::ensure_beam_cache(int64_t rows) {
    auto & s = *state_;
    if (s.beam_cache && s.beam_cache->rows >= rows) {
        return;
    }
    beam_graphs_.clear();
    s.beam_cache.reset();
    const int64_t d = assets_.encoder.n_audio_state, h = assets_.decoder_heads;
    s.beam_cache = make_cache(execution_, assets_.decoder_layers, ((rows + 255) / 256) * 256, h, d / h);
}

// Transformers beam search (length_penalty, early_stopping=False) over a token tree: every beam's
// newest token takes a fresh KV slot and each query row attends to its own ancestors only, so
// beams share their common prefix and no cache is ever reordered.
std::vector<int32_t> EddaRuntime::beam_search(const std::vector<int32_t> & prompt, const DecodeOptions & options,
    DecodeResult & result) {
    struct Beam {
        std::vector<int32_t> tokens, slots;  // slots hold the KV rows of the tokens fed so far
        double score = 0.0;
    };
    struct Finished {
        std::vector<int32_t> tokens;
        double score = 0.0;
    };
    struct Candidate {
        double score;
        int64_t beam;
        int32_t token;
    };
    const int64_t beams = options.num_beams, keep = 2 * beams, vocab = assets_.vocabulary_size;
    const int64_t prompt_len = static_cast<int64_t>(prompt.size());
    const int64_t max_len = std::min<int64_t>(state_->positions, prompt_len + options.max_tokens);
    ensure_beam_cache(prompt_len + beams * (max_len - prompt_len));
    const int64_t rows = state_->beam_cache->rows;
    const double lp = options.length_penalty;

    std::vector<float> logits, mask;
    std::vector<double> log_probs(static_cast<size_t>(vocab));
    std::vector<int32_t> ids(static_cast<size_t>(vocab));
    std::vector<int32_t> tokens, positions, slots;
    const auto run = [&](int64_t t) -> TargetGraph & {
        auto & tg = target_graph(t, true);
        core::write_tensor_i32(tg.token, tokens);
        core::write_tensor_i32(tg.position, positions);
        core::write_tensor_i32(tg.slot, slots);
        core::write_tensor_f16(tg.mask, mask);
        tg.graph.compute();
        ++result.target_passes;
        core::read_tensor_f32_into(tg.logits.tensor, logits);
        return tg;
    };
    // Prefill the prompt causally into slots 0..prompt_len-1.
    tokens = prompt;
    positions.resize(static_cast<size_t>(prompt_len));
    slots.resize(static_cast<size_t>(prompt_len));
    mask.assign(static_cast<size_t>(prompt_len * rows), kNegInf);
    for (int64_t i = 0; i < prompt_len; ++i) {
        positions[static_cast<size_t>(i)] = slots[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        std::fill_n(mask.begin() + i * rows, i + 1, 0.0f);
    }
    run(prompt_len);
    std::vector<Beam> running(1);
    running[0].tokens = prompt;
    running[0].slots = slots;
    int64_t logits_row = prompt_len - 1;
    int64_t next_slot = prompt_len;
    std::vector<Finished> finished;
    bool improvable = true;
    std::vector<Candidate> candidates;
    for (int64_t cur_len = prompt_len; cur_len < max_len;) {
        candidates.clear();
        for (size_t b = 0; b < running.size(); ++b) {
            const float * row = logits.data() + static_cast<size_t>(logits_row + static_cast<int64_t>(b)) * vocab;
            const float peak = *std::max_element(row, row + vocab);
            double sum = 0.0;
            for (int64_t v = 0; v < vocab; ++v) {
                sum += std::exp(static_cast<double>(row[v] - peak));
            }
            const double norm = peak + std::log(sum);
            for (int64_t v = 0; v < vocab; ++v) {
                log_probs[static_cast<size_t>(v)] = row[v] - norm;
            }
            // Logits processors run on log-probabilities, after normalization.
            for (const auto id : assets_.suppress) {
                log_probs[static_cast<size_t>(id)] = -std::numeric_limits<double>::infinity();
            }
            if (cur_len == prompt_len) {
                for (const auto id : assets_.begin_suppress) {
                    log_probs[static_cast<size_t>(id)] = -std::numeric_limits<double>::infinity();
                }
            }
            std::iota(ids.begin(), ids.end(), 0);
            const auto better = [&](int32_t a, int32_t c) {
                return log_probs[static_cast<size_t>(a)] > log_probs[static_cast<size_t>(c)] ||
                    (log_probs[static_cast<size_t>(a)] == log_probs[static_cast<size_t>(c)] && a < c);
            };
            std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(), better);
            for (int64_t k = 0; k < keep; ++k) {
                const auto id = ids[static_cast<size_t>(k)];
                candidates.push_back({running[b].score + log_probs[static_cast<size_t>(id)], static_cast<int64_t>(b), id});
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(),
            [](const Candidate & a, const Candidate & c) { return a.score > c.score; });
        candidates.resize(static_cast<size_t>(std::min<int64_t>(keep, static_cast<int64_t>(candidates.size()))));
        const auto hits = [&](const Candidate & c) { return c.token == assets_.eos || cur_len + 1 >= max_len; };
        // Only the best `beams` continuations may finish; the rest only keep enough beams alive.
        if (improvable) {
            for (int64_t k = 0; k < std::min<int64_t>(beams, static_cast<int64_t>(candidates.size())); ++k) {
                const auto & c = candidates[static_cast<size_t>(k)];
                if (!hits(c)) {
                    continue;
                }
                Finished done{running[static_cast<size_t>(c.beam)].tokens,
                    c.score / std::pow(static_cast<double>(cur_len + 1 - prompt_len), lp)};
                done.tokens.push_back(c.token);
                finished.push_back(std::move(done));
            }
            std::stable_sort(finished.begin(), finished.end(),
                [](const Finished & a, const Finished & c) { return a.score > c.score; });
            if (static_cast<int64_t>(finished.size()) > beams) {
                finished.resize(static_cast<size_t>(beams));
            }
        }
        std::vector<Candidate> alive;
        for (const auto & c : candidates) {
            if (!hits(c)) {
                alive.push_back(c);
            }
        }
        if (alive.empty()) {
            break;
        }
        std::vector<Beam> next;
        for (size_t k = 0; k < alive.size() && static_cast<int64_t>(next.size()) < beams; ++k) {
            Beam beam = running[static_cast<size_t>(alive[k].beam)];
            beam.tokens.push_back(alive[k].token);
            beam.score = alive[k].score;
            next.push_back(std::move(beam));
        }
        running = std::move(next);
        ++cur_len;
        if (static_cast<int64_t>(finished.size()) == beams) {
            const double best = running.front().score / std::pow(static_cast<double>(cur_len - prompt_len), lp);
            improvable = improvable && best > finished.back().score;
        }
        if (!improvable) {
            break;
        }
        // Feed each running beam's newest token into a fresh slot that attends to its ancestors.
        const auto t = static_cast<int64_t>(running.size());
        tokens.resize(static_cast<size_t>(t));
        positions.assign(static_cast<size_t>(t), static_cast<int32_t>(cur_len - 1));
        slots.resize(static_cast<size_t>(t));
        mask.assign(static_cast<size_t>(t * rows), kNegInf);
        for (int64_t b = 0; b < t; ++b) {
            auto & beam = running[static_cast<size_t>(b)];
            beam.slots.push_back(static_cast<int32_t>(next_slot++));
            tokens[static_cast<size_t>(b)] = beam.tokens.back();
            slots[static_cast<size_t>(b)] = beam.slots.back();
            for (const auto slot : beam.slots) {
                mask[static_cast<size_t>(b * rows + slot)] = 0.0f;
            }
        }
        run(t);
        logits_row = 0;
    }
    const auto & best = finished.empty() ? running.front().tokens : finished.front().tokens;
    std::vector<int32_t> out(best.begin() + prompt_len, best.end());
    result.reached_eos = !out.empty() && out.back() == assets_.eos;
    if (result.reached_eos) {
        out.pop_back();
    }
    return out;
}

DecodeResult EddaRuntime::transcribe(const std::vector<float> & samples, const std::vector<int32_t> & prompt,
    const DecodeOptions & options) {
    auto & s = *state_;
    const auto max_samples = static_cast<size_t>(s.frames * 320);
    if (samples.empty() || samples.size() > max_samples) {
        throw std::runtime_error("Edda encoder input must be non-empty and at most 30 seconds");
    }
    const auto staging_rows = s.staging.shape.dims[0];
    if (prompt.empty() || static_cast<int64_t>(prompt.size()) > staging_rows || options.max_tokens < 1) {
        throw std::runtime_error("Edda invalid decoder prompt or token limit");
    }
    const bool speculative = options.speculative && weights_.drafter.has_value();
    const int64_t block = std::clamp<int64_t>(options.block, 2, s.block);
    DecodeResult result;
    const auto clock = std::chrono::steady_clock::now;
    auto started = clock();
    std::vector<float> padded = samples;
    padded.resize(max_samples, 0.0f);
    const auto features = frontend_.compute(padded, execution_.config().threads);
    core::write_tensor_f32(s.features, features.values);
    s.encoder.compute();
    result.encoder_ms = std::chrono::duration<double, std::milli>(clock() - started).count();
    started = clock();
    if (options.num_beams > 1) {
        if (options.num_beams > kMaxBeams) {
            throw std::runtime_error("Edda num_beams must be at most " + std::to_string(kMaxBeams));
        }
        result.tokens = beam_search(prompt, options, result);
        result.decoder_ms = std::chrono::duration<double, std::milli>(clock() - started).count();
        return result;
    }

    const int64_t max_len = std::min<int64_t>(s.positions, static_cast<int64_t>(prompt.size()) + options.max_tokens);
    std::vector<int32_t> seq = prompt;
    std::vector<float> mask, logits;
    std::vector<int32_t> picked;
    const auto run_target = [&](const std::vector<int32_t> & tokens, int64_t start) -> TargetGraph & {
        const auto t = static_cast<int64_t>(tokens.size());
        auto & tg = target_graph(t);
        std::vector<int32_t> positions(static_cast<size_t>(t));
        for (int64_t i = 0; i < t; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(start + i);
        }
        mask.assign(static_cast<size_t>(t * s.positions), kNegInf);
        for (int64_t i = 0; i < t; ++i) {
            std::fill_n(mask.begin() + i * s.positions, start + i + 1, 0.0f);
        }
        core::write_tensor_i32(tg.token, tokens);
        core::write_tensor_i32(tg.position, positions);
        core::write_tensor_i32(tg.slot, positions);
        core::write_tensor_f16(tg.mask, mask);
        tg.graph.compute();
        ++result.target_passes;
        return tg;
    };

    // Prefill the prompt; the first generated token also applies begin_suppress_tokens.
    {
        auto & tg = run_target(prompt, 0);
        core::read_tensor_f32_into(tg.logits.tensor, logits);
        const auto * last = logits.data() + (prompt.size() - 1) * static_cast<size_t>(assets_.vocabulary_size);
        std::vector<float> row(last, last + assets_.vocabulary_size);
        for (const auto & ids : {assets_.suppress, assets_.begin_suppress}) {
            for (const auto id : ids) {
                row[static_cast<size_t>(id)] = kNegInf;
            }
        }
        seq.push_back(static_cast<int32_t>(std::max_element(row.begin(), row.end()) - row.begin()));
    }
    int64_t context_start = 0, context_valid = static_cast<int64_t>(prompt.size());
    std::vector<int32_t> context_slots(static_cast<size_t>(staging_rows)), block_slots, draft_positions, verify;
    while (seq.back() != assets_.eos && static_cast<int64_t>(seq.size()) < max_len) {
        const int64_t p = static_cast<int64_t>(seq.size()) - 1;
        const int64_t b = speculative ? std::min(block, max_len - p) : 1;
        std::vector<int32_t> draft;
        if (b >= 2) {
            auto & dg = drafter_graph(b);
            const int64_t discard = s.positions + s.block;
            for (int64_t i = 0; i < staging_rows; ++i) {
                context_slots[static_cast<size_t>(i)] = static_cast<int32_t>(i < context_valid ? context_start + i : discard);
            }
            block_slots.resize(static_cast<size_t>(b));
            draft_positions.resize(static_cast<size_t>(b));
            for (int64_t i = 0; i < b; ++i) {
                block_slots[static_cast<size_t>(i)] = static_cast<int32_t>(s.positions + i);
                draft_positions[static_cast<size_t>(i)] = static_cast<int32_t>(std::min(p + i, s.positions - 1));
            }
            mask.assign(static_cast<size_t>(b * s.context_rows), kNegInf);
            for (int64_t i = 0; i < b; ++i) {
                std::fill_n(mask.begin() + i * s.context_rows, p, 0.0f);
                std::fill_n(mask.begin() + i * s.context_rows + s.positions, b, 0.0f);
            }
            core::write_tensor_i32(dg.anchor, {seq.back()});
            core::write_tensor_i32(dg.position, draft_positions);
            core::write_tensor_i32(dg.context_slot, context_slots);
            core::write_tensor_i32(dg.block_slot, block_slots);
            core::write_tensor_f16(dg.mask, mask);
            const auto drafting = clock();
            dg.graph.compute();
            result.drafter_ms += std::chrono::duration<double, std::milli>(clock() - drafting).count();
            if (options.use_selector) {
                draft = select_path(dg, seq.back());
            } else {
                core::read_tensor_i32_into(dg.argmax.tensor, draft);
            }
        }
        verify.assign(1, seq.back());
        verify.insert(verify.end(), draft.begin(), draft.end());
        auto & tg = run_target(verify, p);
        core::read_tensor_i32_into(tg.argmax.tensor, picked);
        size_t accepted = 0;
        while (accepted < draft.size() && draft[accepted] == picked[accepted]) {
            ++accepted;
        }
        std::vector<int32_t> fresh(draft.begin(), draft.begin() + static_cast<std::ptrdiff_t>(accepted));
        fresh.push_back(picked[accepted]);
        if (const auto eos = std::find(fresh.begin(), fresh.end(), assets_.eos); eos != fresh.end()) {
            fresh.erase(eos + 1, fresh.end());
        }
        fresh.resize(std::min<size_t>(fresh.size(), static_cast<size_t>(max_len - static_cast<int64_t>(seq.size()))));
        result.drafted += static_cast<int64_t>(draft.size());
        result.accepted += static_cast<int64_t>(std::min(accepted, fresh.size()));
        seq.insert(seq.end(), fresh.begin(), fresh.end());
        context_start = p;
        context_valid = static_cast<int64_t>(fresh.size());
    }
    result.decoder_ms = std::chrono::duration<double, std::milli>(clock() - started).count();
    result.reached_eos = seq.back() == assets_.eos;
    result.tokens.assign(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
    if (result.reached_eos) {
        result.tokens.pop_back();
    }
    return result;
}

}  // namespace engine::community_models::edda_asr
