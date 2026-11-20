#include "engine/models/voxmel_tts/ar.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/cache_slots.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace engine::models::voxmel_tts {

class VoxMelQwenRuntime::Impl {
public:
    const ModelAssets & a;
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    core::TensorValue embedding;
    modules::LinearWeights projection;
    std::vector<float> stop;
    std::unique_ptr<modules::CausalDecoderRuntime> decoder;
    int64_t branches = 0;
    struct ProjectionGraph {
        std::unique_ptr<core::PreparedGraph> owner;
        core::TensorValue input, output;
    };
    runtime::CacheSlots<int64_t, ProjectionGraph> text_graphs{1}, coarse_graphs{2};

    Impl(const ModelAssets & assets, core::ExecutionContext & e,
        assets::TensorStorageType storage, core::AttentionPreference attention)
        : a(assets), execution(e), store(e.backend(), e.backend_type(), "voxmel_tts.qwen.weights", 2 * 1024 * 1024) {
        namespace binding = modules::binding;
        const auto & source = *a.model;
        const std::string prefix = "llm.model.model";
        modules::CausalDecoderRuntimeWeights w;
        embedding = store.load_tensor(source, prefix + ".embed_tokens.weight", storage, {a.vocabulary, a.hidden});
        w.token_embedding = embedding;
        for (int64_t i = 0; i < a.layers; ++i) {
            const auto n = prefix + ".layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store, source, n + ".input_layernorm", a.hidden);
            layer.post_norm = binding::norm_weight_from_source(store, source, n + ".post_attention_layernorm", a.hidden);
            const int64_t kv = a.kv_heads * (a.hidden / a.heads);
            const auto packed = modules::PackedLinearWeightsBuilder({a.hidden, {
                {n + ".self_attn.q_proj.weight", n + ".self_attn.q_proj.bias", a.hidden},
                {n + ".self_attn.k_proj.weight", n + ".self_attn.k_proj.bias", kv},
                {n + ".self_attn.v_proj.weight", n + ".self_attn.v_proj.bias", kv}}, true}).build(store, source, storage);
            layer.self_attention.qkv_weight = packed.weight;
            layer.self_attention.qkv_bias = packed.bias;
            layer.self_attention.out_weight = store.load_tensor(source, n + ".self_attn.o_proj.weight", storage, {a.hidden, a.hidden});
            layer.mlp.gate_up_proj = modules::PackedLinearWeightsBuilder({a.hidden, {
                {n + ".mlp.gate_proj.weight", std::nullopt, a.intermediate},
                {n + ".mlp.up_proj.weight", std::nullopt, a.intermediate}}, false}).build(store, source, storage);
            layer.mlp.down_proj = binding::linear_from_source(store, source, n + ".mlp.down_proj", storage,
                a.hidden, a.intermediate, false);
            w.stack.layers.push_back(std::move(layer));
        }
        w.final_norm = binding::norm_weight_from_source(store, source, prefix + ".norm", a.hidden);
        projection = binding::linear_from_source(store, source, "coarse_frame_proj", storage, a.hidden, a.mels / 2, true);
        stop = source.require_f32("stop_token_head.weight", {1, a.hidden});
        store.upload();
        modules::CausalDecoderRuntimeConfig c;
        c.trace_name = "voxmel_tts.qwen";
        c.prefill_graph_arena_bytes = c.decode_graph_arena_bytes = 8 * 1024 * 1024;
        c.output_mode = modules::CausalDecoderOutputMode::Hidden;
        c.return_hidden = true;
        c.decoder.static_cache_type = GGML_TYPE_F32;
        auto & s = c.decoder.stack;
        s.hidden_size = a.hidden; s.intermediate_size = a.intermediate; s.layers = a.layers;
        s.num_attention_heads = a.heads; s.num_key_value_heads = a.kv_heads; s.head_dim = a.hidden / a.heads;
        s.rms_norm_eps = a.norm_eps; s.rope_theta = a.rope_theta; s.use_qk_norm = false;
        s.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
        s.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
        const bool flash = core::resolve_flash_attention(e.backend(), s.head_dim, attention);
        s.runtime.attention.prefill_mode = s.runtime.attention.static_mode = flash
            ? modules::DecoderAttentionMode::FlashGroupedViewKV : modules::DecoderAttentionMode::ManualRepeat;
        s.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        c.batched_attention_alignment = 32;
        c.reuse_batched_decode_scratch = true;
        decoder = std::make_unique<modules::CausalDecoderRuntime>(e, c, std::move(w));
    }

    std::vector<float> text_embeddings(const std::vector<int32_t> & ids) {
        if (ids.empty() || std::any_of(ids.begin(), ids.end(), [&](int32_t id) { return id < 0 || id >= a.vocabulary; }))
            throw std::runtime_error("Invalid VoxMel text token IDs");
        int64_t capacity = 32;
        while (capacity < static_cast<int64_t>(ids.size())) capacity *= 2;
        auto * g = text_graphs.find(capacity);
        if (!g) {
            ProjectionGraph next;
            next.owner = std::make_unique<core::PreparedGraph>(execution,
                core::PreparedGraphConfig{"voxmel_tts.text_embedding", 256 * 1024, 64});
            auto & ctx = next.owner->build_context();
            next.input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({capacity}));
            ggml_set_input(next.input.tensor);
            next.output = modules::EmbeddingModule({a.vocabulary, a.hidden}).build(ctx, next.input, embedding);
            ggml_set_output(next.output.tensor);
            ggml_build_forward_expand(next.owner->graph(), next.output.tensor);
            next.owner->allocate();
            text_graphs.put(capacity, std::move(next));
            g = text_graphs.find(capacity);
        }
        std::vector<int32_t> padded(static_cast<size_t>(capacity));
        std::copy(ids.begin(), ids.end(), padded.begin());
        ggml_backend_tensor_set(g->input.tensor, padded.data(), 0, padded.size() * sizeof(int32_t));
        g->owner->compute();
        auto out = core::read_tensor_f32(g->output.tensor);
        out.resize(ids.size() * static_cast<size_t>(a.hidden));
        return out;
    }

    std::vector<float> coarse_embeddings(const std::vector<float> & coarse) {
        const int64_t frames = static_cast<int64_t>(coarse.size()) / (a.mels / 2);
        if (frames <= 0 || coarse.size() != static_cast<size_t>(frames * (a.mels / 2)))
            throw std::runtime_error("Invalid VoxMel coarse frames");
        auto * g = coarse_graphs.find(frames);
        if (!g) {
            ProjectionGraph next;
            next.owner = std::make_unique<core::PreparedGraph>(execution,
                core::PreparedGraphConfig{"voxmel_tts.coarse_embedding", 256 * 1024, 64});
            auto & ctx = next.owner->build_context();
            next.input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({frames, a.mels / 2}));
            ggml_set_input(next.input.tensor);
            next.output = modules::LinearModule({a.mels / 2, a.hidden, true}).build(ctx, next.input, projection);
            ggml_set_output(next.output.tensor);
            ggml_build_forward_expand(next.owner->graph(), next.output.tensor);
            next.owner->allocate();
            coarse_graphs.put(frames, std::move(next));
            g = coarse_graphs.find(frames);
        }
        core::write_tensor_f32(g->input, coarse);
        g->owner->compute();
        return core::read_tensor_f32(g->output.tensor);
    }

    QwenState begin(const std::vector<std::vector<float>> & prefixes, int64_t max_frames) {
        if (prefixes.empty() || prefixes.size() > 3 || max_frames <= 0)
            throw std::runtime_error("Invalid VoxMel prefix batch or generation budget");
        const int64_t steps = static_cast<int64_t>(prefixes.front().size()) / a.hidden;
        if (steps <= 0 || steps + max_frames > a.max_positions ||
            std::any_of(prefixes.begin(), prefixes.end(), [&](const auto & p) { return p.size() != static_cast<size_t>(steps * a.hidden); }))
            throw std::runtime_error("VoxMel input and generation budget exceed the context");
        branches = static_cast<int64_t>(prefixes.size());
        const int64_t cache = ((steps + max_frames + 31) / 32) * 32;
        const auto states = decoder->prefill_embeddings_into_batched_decode_cache(prefixes, cache);
        QwenState result;
        for (const auto & s : states) result.hidden.insert(result.hidden.end(), s.hidden.begin(), s.hidden.end());
        result.branches = branches;
        result.stop_logit = std::inner_product(result.hidden.begin(), result.hidden.begin() + a.hidden, stop.begin(), 0.0f);
        return result;
    }

    QwenState advance(const std::vector<float> & coarse) {
        auto embedding = coarse_embeddings(coarse);
        if (embedding.size() != static_cast<size_t>(a.hidden)) throw std::runtime_error("VoxMel AR step requires one frame");
        std::vector<float> batch;
        for (int64_t i = 0; i < branches; ++i) batch.insert(batch.end(), embedding.begin(), embedding.end());
        auto state = decoder->decode_embeddings_batched(batch, branches);
        return {std::move(state.hidden), 0, branches};
    }
};

VoxMelQwenRuntime::VoxMelQwenRuntime(const ModelAssets & a, core::ExecutionContext & e,
    assets::TensorStorageType s, core::AttentionPreference p) : impl_(std::make_unique<Impl>(a, e, s, p)) {}
VoxMelQwenRuntime::~VoxMelQwenRuntime() = default;
std::vector<float> VoxMelQwenRuntime::text_embeddings(const std::vector<int32_t> & x) { return impl_->text_embeddings(x); }
std::vector<float> VoxMelQwenRuntime::coarse_embeddings(const std::vector<float> & x) { return impl_->coarse_embeddings(x); }
QwenState VoxMelQwenRuntime::begin(const std::vector<std::vector<float>> & p, int64_t n) { return impl_->begin(p, n); }
QwenState VoxMelQwenRuntime::advance(const std::vector<float> & x) {
    auto state = impl_->advance(x);
    state.stop_logit = std::inner_product(state.hidden.begin(), state.hidden.begin() + impl_->a.hidden, impl_->stop.begin(), 0.0f);
    return state;
}

} // namespace engine::models::voxmel_tts
