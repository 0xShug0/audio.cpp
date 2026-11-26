// Reverb-TTS masked-diffusion generator, ported from reverb_tts/sampler.py and reverb_tts/model.py
// (https://github.com/software-mansion-labs/Reverb-TTS).
//
// Reverb is a pruned, retrained OmniVoice (Qwen3 backbone over text tokens and 8-codebook audio
// tokens). Unlike OmniVoice it has no classifier-free guidance batch: the prefix (prompt text, then
// the reference frames) never attends to the target, so it runs once and its keys and values are
// kept. Every later pass runs over the target frames alone and attends to [prefix | target].

#include "engine/community_models/reverb/generator.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::reverb {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};
using GgmlContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

struct LayerWeights {
    ggml_tensor * input_norm = nullptr;
    ggml_tensor * q_proj = nullptr;
    ggml_tensor * k_proj = nullptr;
    ggml_tensor * v_proj = nullptr;
    ggml_tensor * o_proj = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * post_norm = nullptr;
    ggml_tensor * gate_proj = nullptr;
    ggml_tensor * up_proj = nullptr;
    ggml_tensor * down_proj = nullptr;
};

struct Weights {
    std::unique_ptr<core::BackendWeightStore> store;
    ggml_tensor * text_table = nullptr;    // [vocab, rank] (or [vocab, hidden] when not factorized)
    ggml_tensor * text_project = nullptr;  // [hidden, rank], or null
    ggml_tensor * audio_embeddings = nullptr;
    ggml_tensor * audio_heads = nullptr;
    ggml_tensor * norm = nullptr;
    std::vector<LayerWeights> layers;
};

// A graph with its own input/compute allocations, built for one request's shapes.
struct Graph {
    GgmlContextPtr ctx;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_t backend = nullptr;

    Graph() = default;
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;
    ~Graph() { reset(); }

    void reset() {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        graph = nullptr;
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        ctx.reset();
    }
};

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

// Sum of the 8 codebook embeddings of every frame. ids: [codebooks * frames], codebook-major, with
// the codebook offset (c * codes) already added.
ggml_tensor * audio_frame_embeddings(
    ggml_context * ctx,
    ggml_tensor * table,
    ggml_tensor * ids,
    int64_t codebooks,
    int64_t frames) {
    ggml_tensor * rows = ggml_get_rows(ctx, table, ids);  // [hidden, codebooks * frames]
    ggml_tensor * sum = nullptr;
    for (int64_t c = 0; c < codebooks; ++c) {
        ggml_tensor * part = ggml_view_2d(ctx, rows, rows->ne[0], frames, rows->nb[1],
            static_cast<size_t>(c * frames) * rows->nb[1]);
        sum = sum == nullptr ? ggml_cont(ctx, part) : ggml_add(ctx, sum, part);
    }
    return sum;
}

std::vector<double> time_steps_impl(int64_t steps, float t_shift) {
    // torch.linspace(0, 1, steps + 1) in float32, then t_shift * t / (1 - (1 - t_shift) * t), the
    // Python scalars rounded to float32 by the tensor ops, then .tolist().
    const int64_t n = steps + 1;
    std::vector<double> out(static_cast<size_t>(n));
    const float step = n > 1 ? 1.0F / static_cast<float>(n - 1) : 0.0F;
    const int64_t halfway = n / 2;
    const float shift = t_shift;
    const float one_minus = static_cast<float>(1.0 - static_cast<double>(t_shift));
    for (int64_t i = 0; i < n; ++i) {
        float t = i < halfway ? step * static_cast<float>(i) : 1.0F - step * static_cast<float>(n - i - 1);
        const float numerator = shift * t;
        const float denominator = 1.0F - one_minus * t;
        out[static_cast<size_t>(i)] = static_cast<double>(numerator / denominator);
    }
    return out;
}

}  // namespace

std::vector<double> reverb_time_steps(int64_t steps, float t_shift) {
    return time_steps_impl(steps, t_shift);
}

struct ReverbGenerator::Impl {
    std::shared_ptr<const assets::TensorSource> source;
    ReverbArchitecture arch;
    core::ExecutionContext * execution = nullptr;
    ggml_backend_t backend = nullptr;
    int threads = 1;
    Weights weights;
    ReverbGeneratorStats stats;

    // Per-request prefix keys and values: K as [dim, prefix, kv_heads], V as [prefix, dim, kv_heads].
    GgmlContextPtr cache_ctx;
    ggml_backend_buffer_t cache_buffer = nullptr;
    std::vector<ggml_tensor *> cache_k;
    std::vector<ggml_tensor *> cache_v;
    int64_t cache_tokens = 0;

    ~Impl() { free_cache(); }

    void free_cache() {
        if (cache_buffer != nullptr) {
            ggml_backend_buffer_free(cache_buffer);
            cache_buffer = nullptr;
        }
        cache_ctx.reset();
        cache_k.clear();
        cache_v.clear();
        cache_tokens = 0;
    }

    void load(assets::TensorStorageType storage_type) {
        const auto & a = arch;
        const auto & src = *source;
        const int64_t dim = a.head_dim;
        const size_t tensor_count = static_cast<size_t>(8 + 11 * a.num_hidden_layers);
        weights.store = std::make_unique<core::BackendWeightStore>(
            backend, execution->backend_type(), "reverb.generator.weights",
            (tensor_count + 16) * ggml_tensor_overhead());
        auto & store = *weights.store;
        const auto load = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            return store.load_tensor(src, name, storage_type, shape).tensor;
        };
        const auto load_f32 = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            return store.load_f32_tensor(src, name, shape).tensor;
        };
        if (a.embedding_rank > 0) {
            weights.text_table = load("llm.embed_tokens.table.weight", {a.vocab_size, a.embedding_rank});
            weights.text_project = load("llm.embed_tokens.project.weight", {a.hidden_size, a.embedding_rank});
        } else {
            weights.text_table = load("llm.embed_tokens.weight", {a.vocab_size, a.hidden_size});
        }
        weights.audio_embeddings = load("audio_embeddings.weight", {a.codebooks * a.codes, a.hidden_size});
        weights.audio_heads = load("audio_heads.weight", {a.codebooks * a.codes, a.hidden_size});
        weights.norm = load_f32("llm.norm.weight", {a.hidden_size});
        weights.layers.resize(static_cast<size_t>(a.num_hidden_layers));
        for (int64_t i = 0; i < a.num_hidden_layers; ++i) {
            const std::string p = "llm.layers." + std::to_string(i);
            auto & w = weights.layers[static_cast<size_t>(i)];
            w.input_norm = load_f32(p + ".input_layernorm.weight", {a.hidden_size});
            w.post_norm = load_f32(p + ".post_attention_layernorm.weight", {a.hidden_size});
            w.q_norm = load_f32(p + ".self_attn.q_norm.weight", {dim});
            w.k_norm = load_f32(p + ".self_attn.k_norm.weight", {dim});
            w.q_proj = load(p + ".self_attn.q_proj.weight", {a.num_attention_heads * dim, a.hidden_size});
            w.k_proj = load(p + ".self_attn.k_proj.weight", {a.num_key_value_heads * dim, a.hidden_size});
            w.v_proj = load(p + ".self_attn.v_proj.weight", {a.num_key_value_heads * dim, a.hidden_size});
            w.o_proj = load(p + ".self_attn.o_proj.weight", {a.hidden_size, a.num_attention_heads * dim});
            w.gate_proj = load(p + ".mlp.gate_proj.weight", {a.intermediate_size, a.hidden_size});
            w.up_proj = load(p + ".mlp.up_proj.weight", {a.intermediate_size, a.hidden_size});
            w.down_proj = load(p + ".mlp.down_proj.weight", {a.hidden_size, a.intermediate_size});
        }
        store.upload();
        source->release_storage();
    }

    // One Qwen3 decoder layer over x [hidden, tokens]. In prefix mode the layer's keys and values
    // are copied into the cache and attention covers the prefix alone; in target mode attention
    // covers [cached prefix | these tokens] with the additive mask [prefix + tokens, tokens].
    ggml_tensor * layer(
        ggml_context * ctx,
        ggml_cgraph * graph,
        ggml_tensor * x,
        ggml_tensor * positions,
        size_t index,
        ggml_tensor * mask,
        bool prefix) {
        const auto & a = arch;
        const auto & w = weights.layers[index];
        const int64_t dim = a.head_dim;
        const int64_t tokens = x->ne[1];
        ggml_tensor * h = rms_norm(ctx, x, w.input_norm, a.rms_norm_eps);
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.q_proj, h), dim, a.num_attention_heads, tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.k_proj, h), dim, a.num_key_value_heads, tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.v_proj, h), dim, a.num_key_value_heads, tokens);
        q = rms_norm(ctx, q, w.q_norm, a.rms_norm_eps);
        k = rms_norm(ctx, k, w.k_norm, a.rms_norm_eps);
        q = ggml_rope_ext(ctx, q, positions, nullptr, static_cast<int>(dim), GGML_ROPE_TYPE_NEOX, 0,
            a.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        k = ggml_rope_ext(ctx, k, positions, nullptr, static_cast<int>(dim), GGML_ROPE_TYPE_NEOX, 0,
            a.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);

        ggml_tensor * keys = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));    // [dim, tokens, kv_heads]
        ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [tokens, dim, kv_heads]
        if (prefix) {
            ggml_build_forward_expand(graph, ggml_cpy(ctx, keys, cache_k[index]));
            ggml_build_forward_expand(graph, ggml_cpy(ctx, values, cache_v[index]));
        } else {
            keys = ggml_concat(ctx, cache_k[index], keys, 1);
            values = ggml_concat(ctx, cache_v[index], values, 0);
        }
        ggml_tensor * queries = ggml_permute(ctx, q, 0, 2, 1, 3);  // [dim, tokens, heads]
        // K/V heads broadcast over their query-head groups (head h reads kv head h / group).
        ggml_tensor * scores = ggml_mul_mat(ctx, keys, queries);  // [keys, tokens, heads]
        scores = ggml_soft_max_ext(ctx, scores, mask, 1.0F / std::sqrt(static_cast<float>(dim)), 0.0F);
        ggml_tensor * context = ggml_mul_mat(ctx, values, scores);  // [dim, tokens, heads]
        context = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));  // [dim, heads, tokens]
        context = ggml_reshape_2d(ctx, context, dim * a.num_attention_heads, tokens);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, w.o_proj, context));

        ggml_tensor * f = rms_norm(ctx, x, w.post_norm, a.rms_norm_eps);
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w.gate_proj, f));
        ggml_tensor * up = ggml_mul_mat(ctx, w.up_proj, f);
        return ggml_add(ctx, x, ggml_mul_mat(ctx, w.down_proj, ggml_mul(ctx, gate, up)));
    }

    void allocate_cache(int64_t prefix_tokens) {
        free_cache();
        const auto & a = arch;
        const size_t layers = static_cast<size_t>(a.num_hidden_layers);
        ggml_init_params params{(2 * layers + 4) * ggml_tensor_overhead(), nullptr, true};
        cache_ctx.reset(ggml_init(params));
        if (cache_ctx == nullptr) {
            throw std::runtime_error("failed to initialize Reverb prefix cache context");
        }
        for (size_t i = 0; i < layers; ++i) {
            cache_k.push_back(ggml_new_tensor_3d(
                cache_ctx.get(), GGML_TYPE_F32, a.head_dim, prefix_tokens, a.num_key_value_heads));
            cache_v.push_back(ggml_new_tensor_3d(
                cache_ctx.get(), GGML_TYPE_F32, prefix_tokens, a.head_dim, a.num_key_value_heads));
        }
        cache_buffer = ggml_backend_alloc_ctx_tensors(cache_ctx.get(), backend);
        if (cache_buffer == nullptr) {
            throw std::runtime_error("failed to allocate the Reverb prefix cache");
        }
        cache_tokens = prefix_tokens;
    }

    static void allocate(Graph & g, ggml_backend_t backend) {
        g.backend = backend;
        g.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (g.allocator == nullptr || !ggml_gallocr_alloc_graph(g.allocator, g.graph)) {
            throw std::runtime_error("failed to allocate a Reverb generator graph");
        }
    }

    void compute(Graph & g, const char * label) {
        core::set_backend_threads(backend, threads);
        if (core::compute_backend_graph(backend, g.graph, nullptr, label) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("Reverb generator graph compute failed: ") + label);
        }
        ggml_backend_synchronize(backend);
    }

    static size_t graph_bytes(int64_t layers) {
        return static_cast<size_t>(64 * layers + 256) * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);
    }

    void run_prefix(const std::vector<int32_t> & text_ids, const std::vector<int32_t> & reference_codes,
        int64_t reference_frames) {
        const auto & a = arch;
        const int64_t text_tokens = static_cast<int64_t>(text_ids.size());
        const int64_t prefix_tokens = text_tokens + reference_frames;
        allocate_cache(prefix_tokens);

        Graph g;
        ggml_init_params params{graph_bytes(a.num_hidden_layers), nullptr, true};
        g.ctx.reset(ggml_init(params));
        ggml_context * ctx = g.ctx.get();
        g.graph = ggml_new_graph_custom(ctx, 16384, false);

        ggml_tensor * text = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, text_tokens);
        ggml_set_input(text);
        ggml_tensor * audio = nullptr;
        if (reference_frames > 0) {
            audio = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a.codebooks * reference_frames);
            ggml_set_input(audio);
        }
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefix_tokens);
        ggml_set_input(positions);

        ggml_tensor * x = ggml_get_rows(ctx, weights.text_table, text);
        if (weights.text_project != nullptr) {
            x = ggml_mul_mat(ctx, weights.text_project, x);
        }
        if (audio != nullptr) {
            x = ggml_concat(ctx, x, audio_frame_embeddings(ctx, weights.audio_embeddings, audio, a.codebooks,
                reference_frames), 1);
        }
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            x = layer(ctx, g.graph, x, positions, i, nullptr, true);
        }
        // The prefix's own outputs are not needed: only its cached keys and values.
        allocate(g, backend);

        ggml_backend_tensor_set(text, text_ids.data(), 0, text_ids.size() * sizeof(int32_t));
        if (audio != nullptr) {
            std::vector<int32_t> ids(reference_codes.size());
            for (int64_t c = 0; c < a.codebooks; ++c) {
                for (int64_t f = 0; f < reference_frames; ++f) {
                    const size_t at = static_cast<size_t>(c * reference_frames + f);
                    ids[at] = reference_codes[at] + static_cast<int32_t>(c * a.codes);
                }
            }
            ggml_backend_tensor_set(audio, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        std::vector<int32_t> pos(static_cast<size_t>(prefix_tokens));
        std::iota(pos.begin(), pos.end(), 0);
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        compute(g, "reverb.prefix");
    }

    // The target-pass graph: tokens in, all codebooks' logits [codebooks * codes, frames] out.
    struct TargetGraph {
        // Inputs live in their own buffer: the graph allocator may reuse a graph tensor's memory
        // once its last reader has run, and positions and the mask must survive across passes.
        GgmlContextPtr input_ctx;
        ggml_backend_buffer_t input_buffer = nullptr;
        Graph g;
        ~TargetGraph() {
            g.reset();
            if (input_buffer != nullptr) {
                ggml_backend_buffer_free(input_buffer);
            }
        }
        ggml_tensor * ids = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * mask = nullptr;
        ggml_tensor * logits = nullptr;
    };

    std::unique_ptr<TargetGraph> build_target(int64_t frames) {
        const auto & a = arch;
        auto t = std::make_unique<TargetGraph>();
        ggml_init_params params{graph_bytes(a.num_hidden_layers), nullptr, true};
        t->g.ctx.reset(ggml_init(params));
        ggml_context * ctx = t->g.ctx.get();
        t->g.graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_init_params input_params{4 * ggml_tensor_overhead(), nullptr, true};
        t->input_ctx.reset(ggml_init(input_params));
        if (t->input_ctx == nullptr) {
            throw std::runtime_error("failed to initialize the Reverb target input context");
        }
        ggml_context * inputs = t->input_ctx.get();
        t->ids = ggml_new_tensor_1d(inputs, GGML_TYPE_I32, a.codebooks * frames);
        t->positions = ggml_new_tensor_1d(inputs, GGML_TYPE_I32, frames);
        t->mask = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, cache_tokens + frames, frames);
        t->input_buffer = ggml_backend_alloc_ctx_tensors(inputs, backend);
        if (t->input_buffer == nullptr) {
            throw std::runtime_error("failed to allocate the Reverb target inputs");
        }
        ggml_tensor * x = audio_frame_embeddings(ctx, weights.audio_embeddings, t->ids, a.codebooks, frames);
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            x = layer(ctx, t->g.graph, x, t->positions, i, t->mask, false);
        }
        x = rms_norm(ctx, x, weights.norm, a.rms_norm_eps);
        t->logits = ggml_mul_mat(ctx, weights.audio_heads, x);  // [codebooks * codes, frames]
        ggml_set_output(t->logits);
        ggml_build_forward_expand(t->g.graph, t->logits);
        allocate(t->g, backend);
        std::vector<int32_t> pos(static_cast<size_t>(frames));
        std::iota(pos.begin(), pos.end(), static_cast<int32_t>(cache_tokens));
        ggml_backend_tensor_set(t->positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        return t;
    }
};

ReverbGenerator::ReverbGenerator(
    std::shared_ptr<const assets::TensorSource> weights,
    const ReverbArchitecture & architecture,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>()) {
    if (weights == nullptr) {
        throw std::runtime_error("Reverb generator requires weights");
    }
    const auto & a = architecture;
    if (a.hidden_size <= 0 || a.num_hidden_layers <= 0 || a.head_dim <= 0 || a.num_attention_heads <= 0 ||
        a.num_key_value_heads <= 0 || a.num_attention_heads % a.num_key_value_heads != 0 || a.codebooks != 8 ||
        a.codes <= a.mask_id) {
        throw std::runtime_error("Reverb generator configuration is invalid");
    }
    impl_->source = std::move(weights);
    impl_->arch = architecture;
    impl_->execution = &execution_context;
    impl_->backend = execution_context.backend();
    impl_->threads = std::max(1, execution_context.config().threads);
    impl_->load(storage_type);
}

ReverbGenerator::~ReverbGenerator() = default;

const ReverbArchitecture & ReverbGenerator::architecture() const noexcept {
    return impl_->arch;
}

const ReverbGeneratorStats & ReverbGenerator::last_stats() const noexcept {
    return impl_->stats;
}

namespace {

// Each entry's rank within `score`, sorted descending (best = 0), or ascending.
std::vector<int64_t> ranks(const std::vector<float> & score, bool descending) {
    std::vector<int64_t> order(score.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t l, int64_t r) {
        return descending ? score[static_cast<size_t>(l)] > score[static_cast<size_t>(r)]
                          : score[static_cast<size_t>(l)] < score[static_cast<size_t>(r)];
    });
    std::vector<int64_t> out(score.size());
    for (size_t i = 0; i < order.size(); ++i) {
        out[static_cast<size_t>(order[i])] = static_cast<int64_t>(i);
    }
    return out;
}

// log_softmax over all `codes` entries of one row of logits, written for the first `kept` (the
// mask id is the last entry and is never a prediction).
void log_softmax_row(const float * logits, int64_t codes, int64_t kept, float * out) {
    float peak = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < codes; ++i) {
        peak = std::max(peak, logits[i]);
    }
    double sum = 0.0;
    for (int64_t i = 0; i < codes; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - peak));
    }
    const float lse = peak + static_cast<float>(std::log(sum));
    for (int64_t i = 0; i < kept; ++i) {
        out[i] = logits[i] - lse;
    }
}

}  // namespace

std::vector<int32_t> ReverbGenerator::generate(
    const std::vector<int32_t> & text_ids,
    const std::vector<int32_t> & reference_codes,
    int64_t reference_frames,
    int64_t target_frames,
    uint32_t seed,
    const ReverbSamplerConfig & config) {
    auto & impl = *impl_;
    const auto & a = impl.arch;
    if (text_ids.empty() || target_frames <= 0 || reference_frames < 0 ||
        static_cast<int64_t>(reference_codes.size()) != a.codebooks * reference_frames) {
        throw std::runtime_error("Reverb generator request is invalid");
    }
    if (config.steps < 1 || !(config.remask_ratio >= 0.0F && config.remask_ratio < 1.0F) ||
        !(config.t_shift > 0.0F && config.t_shift <= 1.0F) || !(config.position_temperature > 0.0F)) {
        throw std::runtime_error("Reverb sampler options are invalid");
    }
    for (const int32_t id : text_ids) {
        if (id < 0 || id >= a.vocab_size) {
            throw std::runtime_error("Reverb prompt token is outside the text vocabulary");
        }
    }
    for (const int32_t code : reference_codes) {
        if (code < 0 || code >= a.mask_id) {
            throw std::runtime_error("Reverb reference code is out of range");
        }
    }
    impl.stats = {};
    const auto prefix_start = Clock::now();
    impl.run_prefix(text_ids, reference_codes, reference_frames);
    const int64_t P = impl.cache_tokens;
    const int64_t F = target_frames;
    const int64_t C = a.codebooks;
    const int64_t V = a.codes;
    const int64_t K = a.mask_id;  // predictable codes per codebook
    auto target = impl.build_target(F);
    impl.stats.prefix_ms = engine::debug::elapsed_ms(prefix_start, Clock::now());

    // Masks: plain attends to everything; polish adds polish_bias on the reference frames' keys.
    const int64_t keys = P + F;
    std::vector<float> plain(static_cast<size_t>(keys * F), 0.0F);
    std::vector<float> polish = plain;
    for (int64_t q = 0; q < F; ++q) {
        for (int64_t k = P - reference_frames; k < P; ++k) {
            polish[static_cast<size_t>(q * keys + k)] = config.polish_bias;
        }
    }

    std::vector<int32_t> tokens(static_cast<size_t>(C * F), a.mask_id);  // [C, F]
    std::vector<float> logits(static_cast<size_t>(C * V * F));
    bool mask_is_polish = true;  // forces the first upload
    const auto pass = [&](bool use_polish) {
        const auto start = Clock::now();
        if (use_polish != mask_is_polish) {
            const auto & m = use_polish ? polish : plain;
            ggml_backend_tensor_set(target->mask, m.data(), 0, m.size() * sizeof(float));
            mask_is_polish = use_polish;
        }
        std::vector<int32_t> ids(tokens.size());
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t f = 0; f < F; ++f) {
                const size_t at = static_cast<size_t>(c * F + f);
                ids[at] = tokens[at] + static_cast<int32_t>(c * V);
            }
        }
        ggml_backend_tensor_set(target->ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        impl.compute(target->g, "reverb.target");
        ggml_backend_tensor_get(target->logits, logits.data(), 0, logits.size() * sizeof(float));
        impl.stats.passes_ms += engine::debug::elapsed_ms(start, Clock::now());
        ++impl.stats.passes;
    };
    // log-probabilities of codebook c at frame f, the mask excluded: lp[(slot) * K + code]
    std::vector<float> lp;
    const auto log_probs = [&](int64_t first_codebook, int64_t count) {
        lp.assign(static_cast<size_t>(count * F * K), 0.0F);
        for (int64_t c = 0; c < count; ++c) {
            for (int64_t f = 0; f < F; ++f) {
                const float * row = logits.data() + static_cast<size_t>(f * C * V + (first_codebook + c) * V);
                log_softmax_row(row, V, K, lp.data() + static_cast<size_t>((c * F + f) * K));
            }
        }
    };
    const auto argmax = [&](const float * row) {
        int32_t best = 0;
        for (int32_t i = 1; i < K; ++i) {
            if (row[i] > row[best]) {
                best = i;
            }
        }
        return best;
    };

    std::mt19937 rng(seed);
    const auto uniform = [&]() {
        return static_cast<float>(static_cast<double>(rng() & ((1u << 24) - 1)) * std::ldexp(1.0, -24));
    };
    const auto t = time_steps_impl(config.steps, config.t_shift);
    const int64_t slots = 2 * F;
    std::vector<float> confidence(static_cast<size_t>(slots));
    std::vector<int32_t> prediction(static_cast<size_t>(slots));
    std::vector<float> score(static_cast<size_t>(slots));
    std::vector<float> likely(static_cast<size_t>(slots));
    const float inf = std::numeric_limits<float>::infinity();

    for (int64_t step = 0; step < config.steps; ++step) {
        pass(false);
        const auto sampling_start = Clock::now();
        log_probs(0, 2);
        int64_t left = 0;
        int64_t revealed = 0;
        for (int64_t s = 0; s < slots; ++s) {
            const float * row = lp.data() + static_cast<size_t>(s * K);
            prediction[static_cast<size_t>(s)] = argmax(row);
            confidence[static_cast<size_t>(s)] = row[prediction[static_cast<size_t>(s)]];
            if (tokens[static_cast<size_t>(s)] == a.mask_id) {
                ++left;
            } else {
                ++revealed;
            }
        }
        const bool last = step == config.steps - 1;
        const int64_t quota = static_cast<int64_t>(std::ceil(
            2.0 * static_cast<double>(F) * (t[static_cast<size_t>(step + 1)] - t[static_cast<size_t>(step)])));
        const int64_t k = last ? left : std::min(left, quota);
        // Noise for this pass: [2, F] uniforms drawn in row-major order.
        for (int64_t s = 0; s < slots; ++s) {
            const float noise = uniform();
            const float gumbel = -std::log(-std::log(noise + 1.0e-10F) + 1.0e-10F);
            const int64_t codebook = s / F;
            const float penalty = config.codebook_penalty * static_cast<float>(codebook);
            score[static_cast<size_t>(s)] = tokens[static_cast<size_t>(s)] == a.mask_id
                ? (confidence[static_cast<size_t>(s)] - penalty) / config.position_temperature + gumbel
                : -inf;
        }
        const auto score_rank = ranks(score, true);
        std::vector<int32_t> updated(tokens.begin(), tokens.begin() + slots);
        for (int64_t s = 0; s < slots; ++s) {
            if (score_rank[static_cast<size_t>(s)] < k && std::isfinite(score[static_cast<size_t>(s)])) {
                updated[static_cast<size_t>(s)] = prediction[static_cast<size_t>(s)];
            }
        }
        if (config.remasking && !last) {
            for (int64_t s = 0; s < slots; ++s) {
                const int32_t held = tokens[static_cast<size_t>(s)];
                likely[static_cast<size_t>(s)] = held != a.mask_id
                    ? lp[static_cast<size_t>(s * K + std::min<int32_t>(held, static_cast<int32_t>(K - 1)))]
                    : inf;
            }
            const int64_t requested = static_cast<int64_t>(std::nearbyint(static_cast<double>(config.remask_ratio) *
                static_cast<double>(k)));
            const int64_t count = std::min(requested, revealed);
            const auto doubt_rank = ranks(likely, false);
            for (int64_t s = 0; s < slots; ++s) {
                if (tokens[static_cast<size_t>(s)] != a.mask_id && doubt_rank[static_cast<size_t>(s)] < count) {
                    updated[static_cast<size_t>(s)] = a.mask_id;
                }
            }
        }
        std::copy(updated.begin(), updated.end(), tokens.begin());
        impl.stats.sampling_ms += engine::debug::elapsed_ms(sampling_start, Clock::now());
    }

    // Polish: codebooks 2-3 from the first pass, then 4-7 guided against the first pass's view.
    pass(true);
    log_probs(2, 6);
    std::vector<float> first = lp;  // codebooks 2..7
    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t f = 0; f < F; ++f) {
            tokens[static_cast<size_t>((2 + c) * F + f)] = argmax(first.data() + static_cast<size_t>((c * F + f) * K));
        }
    }
    pass(true);
    log_probs(4, 4);
    std::vector<float> guided(static_cast<size_t>(K));
    for (int64_t c = 0; c < 4; ++c) {
        for (int64_t f = 0; f < F; ++f) {
            const float * second = lp.data() + static_cast<size_t>((c * F + f) * K);
            const float * earlier = first.data() + static_cast<size_t>(((2 + c) * F + f) * K);
            for (int64_t i = 0; i < K; ++i) {
                guided[static_cast<size_t>(i)] = second[i] + config.polish_guidance * (second[i] - earlier[i]);
            }
            tokens[static_cast<size_t>((4 + c) * F + f)] = argmax(guided.data());
        }
    }
    impl.free_cache();
    return tokens;
}

}  // namespace engine::community_models::reverb
