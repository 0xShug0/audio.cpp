#include "engine/models/hviske_asr_v6/decoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::hviske_asr_v6 {
namespace {
constexpr size_t kContextBytes = 4 * 1024 * 1024;
struct ContextDeleter {
    void operator()(ggml_context * p) const { ggml_free(p); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t p) const { ggml_gallocr_free(p); }
};

class HviskeV6Qwen3BeamGraph {
public:
    HviskeV6Qwen3BeamGraph(core::ExecutionContext & execution, const modules::CausalDecoderConfig & config,
        const modules::CausalDecoderRuntimeWeights & weights, int64_t capacity, int64_t batch)
        : execution_(execution), capacity_(capacity), layers_(config.stack.layers), vocab_(config.logits_size), batch_(batch) {
        context_.reset(ggml_init({kContextBytes, nullptr, true}));
        core::ModuleBuildContext ctx{context_.get(), "hviske_asr_v6.decoder", execution.backend_type()};
        token_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch, 1}));
        position_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        slot_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({batch, 1, 1, capacity}));
        ggml_set_input(token_.tensor);
        ggml_set_input(position_.tensor);
        ggml_set_input(slot_.tensor);
        ggml_set_input(mask_.tensor);
        auto x = modules::EmbeddingModule({vocab_, config.stack.hidden_size}).build(ctx, token_, weights.token_embedding);
        graph_ = ggml_new_graph_custom(ctx.ggml, 8192, false);
        std::vector<std::vector<core::TensorValue>> keys(batch), values(batch);
        const modules::DecoderLayerModule layer(modules::decoder_layer_config_from_stack(config.stack));
        for (int64_t i = 0; i < layers_; ++i) {
            const auto shape = core::TensorShape::from_dims(
                {batch, capacity, config.stack.num_key_value_heads, config.stack.head_dim});
            auto key = core::make_tensor(ctx, config.static_cache_type, shape);
            auto value = core::make_tensor(ctx, config.static_cache_type, shape);
            ggml_set_input(key.tensor);
            ggml_set_output(key.tensor);
            ggml_set_input(value.tensor);
            ggml_set_output(value.tensor);
            x = layer.build_with_static_cache_tail_batched(ctx, graph_, x, position_, weights.stack.layers[i],
                key, value, slot_, mask_).output;
            for (int64_t b = 0; b < batch; ++b) {
                keys[b].push_back(modules::SliceModule({0, b, 1}).build(ctx, key));
                values[b].push_back(modules::SliceModule({0, b, 1}).build(ctx, value));
            }
        }
        runtime::TransformerKVCacheOptions cache_options;
        cache_options.allow_f16_storage = config.static_cache_type == GGML_TYPE_F16;
        cache_options.allow_bf16_storage = config.static_cache_type == GGML_TYPE_BF16;
        cache_options.lazy_import_scratch = true;
        for (int64_t b = 0; b < batch; ++b) {
            caches_.emplace_back(capacity, config.stack.num_key_value_heads * config.stack.head_dim,
                std::move(keys[b]), std::move(values[b]), cache_options);
        }
        x = modules::RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false})
            .build(ctx, x, weights.final_norm);
        output_ = modules::LinearModule({config.stack.hidden_size, vocab_, false}).build(ctx, x, *weights.lm_head);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("Hviske v6 beam graph allocation failed");
        }
        for (auto & cache : caches_) {
            for (int64_t i = 0; i < layers_; ++i) {
                if (ggml_backend_view_init(cache.key_tensor(i).tensor) != GGML_STATUS_SUCCESS ||
                    ggml_backend_view_init(cache.value_tensor(i).tensor) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Hviske v6 beam cache view initialization failed");
                }
            }
        }
        core::prepare_host_graph_plan(execution, graph_, plan_);
    }
    ~HviskeV6Qwen3BeamGraph() {
        core::release_backend_graph_resources(execution_.backend(), prefill_graph_);
        prefill_plan_.reset();
        core::release_backend_graph_resources(execution_.backend(), graph_);
        plan_.reset();
    }
    int64_t capacity() const { return capacity_; }
    int64_t batch_size() const { return batch_; }
    std::vector<float> prefill(const std::vector<float> & input, int64_t steps,
        const modules::CausalDecoderConfig & config, const modules::CausalDecoderRuntimeWeights & weights) {
        if (!prefill_graph_ || prefill_steps_ != steps) {
            core::release_backend_graph_resources(execution_.backend(), prefill_graph_);
            prefill_plan_.reset();
            prefill_graph_ = nullptr;
            prefill_allocator_.reset();
            prefill_context_.reset(ggml_init({kContextBytes, nullptr, true}));
            core::ModuleBuildContext ctx{prefill_context_.get(), "hviske_asr_v6.decoder.prefill", execution_.backend_type()};
            prefill_input_ = core::make_tensor(ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({1, steps, config.stack.hidden_size}));
            prefill_positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
            prefill_mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, steps}));
            ggml_set_input(prefill_input_.tensor);
            ggml_set_input(prefill_positions_.tensor);
            ggml_set_input(prefill_mask_.tensor);
            const auto out = modules::CausalDecoderModule(config).build(ctx, prefill_input_, prefill_positions_,
                {weights.stack, weights.final_norm, *weights.lm_head}, std::nullopt, prefill_mask_);
            prefill_graph_ = ggml_new_graph_custom(ctx.ggml, 8192, false);
            const modules::FastKVSetRowsModule write_cache({modules::FastKVSetRowsMode::BackendViewOptimized});
            // Populate the first beam directly; beam selection forks it on the backend.
            for (int64_t i = 0; i < layers_; ++i) {
                const auto & state = out.state.layers[i];
                auto key = write_cache.build_block(ctx, caches_.front().key_tensor(i), *state.key, prefill_positions_);
                auto value = write_cache.build_block(ctx, caches_.front().value_tensor(i), *state.value, prefill_positions_);
                ggml_build_forward_expand(prefill_graph_, key.tensor);
                ggml_build_forward_expand(prefill_graph_, value.tensor);
            }
            prefill_output_ = out.logits;
            ggml_set_output(prefill_output_.tensor);
            ggml_build_forward_expand(prefill_graph_, prefill_output_.tensor);
            prefill_allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
            if (!ggml_gallocr_alloc_graph(prefill_allocator_.get(), prefill_graph_)) {
                throw std::runtime_error("Hviske v6 prefill graph allocation failed");
            }
            core::prepare_host_graph_plan(execution_, prefill_graph_, prefill_plan_);
            prefill_steps_ = steps;
        }
        caches_.front().clear_on_backend();
        std::vector<int32_t> positions(steps);
        std::iota(positions.begin(), positions.end(), 0);
        const auto mask = modules::causal_prefill_mask_values(1, steps);
        ggml_backend_tensor_set(prefill_input_.tensor, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(prefill_positions_.tensor, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(prefill_mask_.tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        if (core::compute_graph(execution_, prefill_graph_, prefill_plan_, "hviske_asr_v6.decoder.prefill") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske v6 prefill execution failed");
        }
        std::vector<float> logits(vocab_);
        ggml_backend_tensor_get(prefill_output_.tensor, logits.data(), 0, logits.size() * sizeof(float));
        return logits;
    }
    void copy_from(size_t destination, size_t source) {
        for (int64_t i = 0; i < layers_; ++i) {
            ggml_backend_tensor_copy(caches_[source].key_tensor(i).tensor, caches_[destination].key_tensor(i).tensor);
            ggml_backend_tensor_copy(caches_[source].value_tensor(i).tensor, caches_[destination].value_tensor(i).tensor);
        }
    }
    const std::vector<float> & step(const std::vector<int32_t> & tokens, int32_t position) {
        ggml_backend_tensor_set(token_.tensor, tokens.data(), 0, tokens.size() * sizeof(int32_t));
        ggml_backend_tensor_set(position_.tensor, &position, 0, sizeof(position));
        slots_.resize(batch_);
        for (int64_t b = 0; b < batch_; ++b) slots_[b] = static_cast<int32_t>(b * capacity_ + position);
        ggml_backend_tensor_set(slot_.tensor, slots_.data(), 0, slots_.size() * sizeof(int32_t));
        modules::write_decoder_batched_cached_step_mask(mask_.tensor, mask_values_, batch_, capacity_, position, position);
        if (core::compute_graph(execution_, graph_, plan_, "hviske_asr_v6.decoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske v6 decoder execution failed");
        }
        logits_.resize(static_cast<size_t>(vocab_ * batch_));
        ggml_backend_tensor_get(output_.tensor, logits_.data(), 0, logits_.size() * sizeof(float));
        return logits_;
    }

private:
    core::ExecutionContext & execution_;
    int64_t capacity_, layers_, vocab_, batch_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue token_, position_, slot_, mask_, output_;
    std::vector<runtime::TransformerKVCache> caches_;
    std::vector<int32_t> slots_;
    std::vector<ggml_fp16_t> mask_values_;
    std::vector<float> logits_;
    std::unique_ptr<ggml_context, ContextDeleter> prefill_context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> prefill_allocator_;
    core::HostGraphPlan prefill_plan_;
    ggml_cgraph * prefill_graph_ = nullptr;
    core::TensorValue prefill_input_, prefill_positions_, prefill_mask_, prefill_output_;
    int64_t prefill_steps_ = 0;
};
}  // namespace

struct HviskeV6Qwen3DecoderRuntime::Impl {
    Impl(std::shared_ptr<const HviskeV6Assets> assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "hviske_asr_v6.decoder.weights", kContextBytes) {
        const auto & c = assets_->config.decoder;
        const auto & source = *assets_->weights;
        bool packed_qkv = true, packed_gate_up = true;
        if (storage == assets::TensorStorageType::Native) {
            for (int64_t i = 0; i < c.layers; ++i) {
                const auto p = "llm.model.layers." + std::to_string(i);
                const auto q_type = source.require_metadata(p + ".self_attn.q_proj.weight").dtype;
                packed_qkv = packed_qkv && q_type == source.require_metadata(p + ".self_attn.k_proj.weight").dtype &&
                    q_type == source.require_metadata(p + ".self_attn.v_proj.weight").dtype;
                packed_gate_up = packed_gate_up && source.require_metadata(p + ".mlp.gate_proj.weight").dtype ==
                    source.require_metadata(p + ".mlp.up_proj.weight").dtype;
            }
        }
        weights_.token_embedding = store_.load_tensor(source, "llm.model.embed_tokens.weight", storage,
            {c.vocab_size, c.hidden_size});
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto p = "llm.model.layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = modules::binding::norm_weight_from_source(store_, source, p + ".input_layernorm", c.hidden_size);
            layer.post_norm = modules::binding::norm_weight_from_source(store_, source, p + ".post_attention_layernorm", c.hidden_size);
            layer.q_norm = modules::binding::norm_weight_from_source(store_, source, p + ".self_attn.q_norm", c.head_dim);
            layer.k_norm = modules::binding::norm_weight_from_source(store_, source, p + ".self_attn.k_norm", c.head_dim);
            if (packed_qkv) {
                layer.self_attention.qkv_weight = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                    {p + ".self_attn.q_proj.weight", std::nullopt, c.heads * c.head_dim},
                    {p + ".self_attn.k_proj.weight", std::nullopt, c.kv_heads * c.head_dim},
                    {p + ".self_attn.v_proj.weight", std::nullopt, c.kv_heads * c.head_dim}}, false})
                    .build(store_, source, storage).weight;
            } else {
                layer.self_attention.q_weight = store_.load_tensor(source, p + ".self_attn.q_proj.weight", storage,
                    {c.heads * c.head_dim, c.hidden_size});
                layer.self_attention.k_weight = store_.load_tensor(source, p + ".self_attn.k_proj.weight", storage,
                    {c.kv_heads * c.head_dim, c.hidden_size});
                layer.self_attention.v_weight = store_.load_tensor(source, p + ".self_attn.v_proj.weight", storage,
                    {c.kv_heads * c.head_dim, c.hidden_size});
            }
            layer.self_attention.out_weight = store_.load_tensor(source, p + ".self_attn.o_proj.weight", storage,
                {c.hidden_size, c.heads * c.head_dim});
            if (packed_gate_up) {
                layer.mlp.gate_up_proj = modules::PackedLinearWeightsBuilder({c.hidden_size, {
                    {p + ".mlp.gate_proj.weight", std::nullopt, c.intermediate_size},
                    {p + ".mlp.up_proj.weight", std::nullopt, c.intermediate_size}}, false})
                    .build(store_, source, storage);
            } else {
                layer.mlp.gate_proj = modules::binding::linear_from_source(store_, source, p + ".mlp.gate_proj", storage,
                    c.intermediate_size, c.hidden_size, false);
                layer.mlp.up_proj = modules::binding::linear_from_source(store_, source, p + ".mlp.up_proj", storage,
                    c.intermediate_size, c.hidden_size, false);
            }
            layer.mlp.down_proj = modules::binding::linear_from_source(store_, source, p + ".mlp.down_proj", storage,
                c.hidden_size, c.intermediate_size, false);
            weights_.stack.layers.push_back(std::move(layer));
        }
        weights_.final_norm = modules::binding::norm_weight_from_source(store_, source, "llm.model.norm", c.hidden_size);
        weights_.lm_head = modules::LinearWeights{weights_.token_embedding, std::nullopt};
        store_.upload();
        auto & d = config_;
        d.logits_size = c.vocab_size;
        d.static_cache_type = GGML_TYPE_F16;
        auto & s = d.stack;
        s.hidden_size = c.hidden_size;
        s.num_attention_heads = c.heads;
        s.num_key_value_heads = c.kv_heads;
        s.head_dim = c.head_dim;
        s.intermediate_size = c.intermediate_size;
        s.layers = c.layers;
        s.rms_norm_eps = c.rms_norm_eps;
        s.rope_theta = c.rope_theta;
        s.use_qk_norm = true;
        s.qkv_layout = packed_qkv ? modules::DecoderQKVLayout::PackedQKV : modules::DecoderQKVLayout::Separate;
        s.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        s.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        s.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        s.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        s.runtime.mlp.mode = packed_gate_up ? modules::DecoderMLPMode::PackedGateUp : modules::DecoderMLPMode::FusedSwiGLU;

        std::unique_ptr<ggml_context, ContextDeleter> context(ggml_init({kContextBytes, nullptr, true}));
        core::ModuleBuildContext ctx{context.get(), "hviske_asr_v6.embedding", execution.backend_type()};
        const auto & mc = assets_->config;
        special_ids_ = {mc.bos_token_id, mc.cased_token_id, mc.nocase_token_id,
            mc.punctuation_token_id, mc.no_punctuation_token_id};
        auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({5}));
        auto embeddings = modules::EmbeddingModule({c.vocab_size, c.hidden_size}).build(ctx, ids, weights_.token_embedding);
        auto * graph = ggml_new_graph_custom(ctx.ggml, 32, false);
        ggml_build_forward_expand(graph, embeddings.tensor);
        std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("Hviske v6 control embedding allocation failed");
        }
        ggml_backend_tensor_set(ids.tensor, special_ids_.data(), 0, sizeof(special_ids_));
        if (core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske v6 control embedding lookup failed");
        }
        special_embeddings_.resize(static_cast<size_t>(5 * c.hidden_size));
        ggml_backend_tensor_get(embeddings.tensor, special_embeddings_.data(), 0, special_embeddings_.size() * sizeof(float));
        core::release_backend_graph_resources(execution.backend(), graph);
    }

    std::vector<int32_t> generate(const std::vector<float> & audio, const HviskeV6DecodingOptions & options) {
        const auto & c = assets_->config;
        auto input = audio;
        auto controls = options.control_tokens;
        controls.push_back(c.bos_token_id);
        for (const auto id : controls) {
            const auto found = std::find(special_ids_.begin(), special_ids_.end(), id);
            if (found == special_ids_.end()) throw std::runtime_error("Invalid Hviske v6 control token");
            const auto offset = std::distance(special_ids_.begin(), found) * c.decoder.hidden_size;
            input.insert(input.end(), special_embeddings_.begin() + offset,
                special_embeddings_.begin() + offset + c.decoder.hidden_size);
        }
        const int64_t steps = input.size() / c.decoder.hidden_size;
        const int64_t capacity = steps + options.max_tokens;
        if (capacity > c.decoder.max_positions) throw std::runtime_error("Hviske v6 decoder context exceeded");
        if (!beam_graph_ || beam_graph_->batch_size() != options.num_beams || beam_graph_->capacity() != capacity) {
            beam_graph_.reset();
            beam_graph_ = std::make_unique<HviskeV6Qwen3BeamGraph>(
                execution_, config_, weights_, capacity, options.num_beams);
        }
        auto started = std::chrono::steady_clock::now();
        auto logits = beam_graph_->prefill(input, steps, config_, weights_);
        debug::timing_log_scalar("hviske_asr_v6.decoder.prefill_ms", debug::elapsed_ms(started));
        started = std::chrono::steady_clock::now();
        struct Beam {
            std::vector<int32_t> tokens;
            float score = 0;
            size_t slot = 0;
        };
        struct Candidate { float score; size_t parent; int32_t token; };
        struct Finished { float score; std::vector<int32_t> tokens; };
        const std::vector<float> * current_logits = &logits;
        std::vector<Beam> active{{{}, 0.0f, 0}};
        std::vector<Finished> finished;
        double search_ms = 0, cache_copy_ms = 0, graph_ms = 0;
        for (int64_t step = 0; step < options.max_tokens; ++step) {
            const auto search_started = std::chrono::steady_clock::now();
            std::vector<Candidate> candidates;
            const auto candidate_limit = static_cast<size_t>(2 * options.num_beams);
            candidates.reserve(candidate_limit + 1);
            for (size_t b = 0; b < active.size(); ++b) {
                const auto first = current_logits->begin() + active[b].slot * c.decoder.vocab_size;
                const auto maximum = *std::max_element(first, first + c.decoder.vocab_size);
                double sum = 0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+:sum) num_threads(std::max(1, execution_.config().threads))
#endif
                for (int64_t token = 0; token < c.decoder.vocab_size; ++token) {
                    sum += std::exp(static_cast<double>(first[token] - maximum));
                }
                const auto norm = maximum + std::log(sum);
                for (int32_t token = 0; token < c.decoder.vocab_size; ++token) {
                    if (token == c.cased_token_id || token == c.nocase_token_id ||
                        token == c.punctuation_token_id || token == c.no_punctuation_token_id) continue;
                    const Candidate candidate{active[b].score + static_cast<float>(first[token] - norm), b, token};
                    if (candidates.size() == candidate_limit && candidate.score <= candidates.back().score) continue;
                    const auto at = std::lower_bound(candidates.begin(), candidates.end(), candidate,
                        [](const auto & a, const auto & b) { return a.score > b.score; });
                    candidates.insert(at, candidate);
                    if (candidates.size() > candidate_limit) candidates.pop_back();
                }
            }
            const auto top = candidates.size();
            if (options.num_beams == 1 && candidates.front().token == c.eos_token_id) {
                finished.push_back({0.0f, active.front().tokens});
                active.clear();
                search_ms += debug::elapsed_ms(search_started);
                break;
            }
            std::vector<Beam> next;
            std::vector<size_t> parents;
            for (size_t rank = 0; rank < top && next.size() < static_cast<size_t>(options.num_beams); ++rank) {
                const auto & item = candidates[rank];
                auto tokens = active[item.parent].tokens;
                if (item.token == c.eos_token_id) {
                    if (rank < static_cast<size_t>(options.num_beams)) {
                        finished.push_back({item.score / std::pow(static_cast<float>(step + 1), options.length_penalty), tokens});
                    }
                    continue;
                }
                tokens.push_back(item.token);
                next.push_back({std::move(tokens), item.score, active[item.parent].slot});
                parents.push_back(active[item.parent].slot);
            }
            std::sort(finished.begin(), finished.end(), [](const auto & a, const auto & b) { return a.score > b.score; });
            if (finished.size() > static_cast<size_t>(options.num_beams)) finished.resize(options.num_beams);
            search_ms += debug::elapsed_ms(search_started);
            if (next.empty()) break;
            if (finished.size() == static_cast<size_t>(options.num_beams) && finished.back().score >=
                candidates.front().score / std::pow(static_cast<float>(step + 1), options.length_penalty)) {
                active.clear();
                break;
            }
            // Unique parents keep their physical cache; only a fork copies K/V, on the backend.
            const auto copy_started = std::chrono::steady_clock::now();
            std::vector<bool> used(options.num_beams, false), parent_used(options.num_beams, false);
            for (const auto parent : parents) used[parent] = true;
            for (size_t i = 0; i < next.size(); ++i) {
                if (parent_used[parents[i]]) {
                    const auto free = std::find(used.begin(), used.end(), false);
                    const auto slot = static_cast<size_t>(std::distance(used.begin(), free));
                    beam_graph_->copy_from(slot, parents[i]);
                    used[slot] = true;
                    next[i].slot = slot;
                }
                parent_used[parents[i]] = true;
            }
            cache_copy_ms += debug::elapsed_ms(copy_started);
            active = std::move(next);
            if (step + 1 == options.max_tokens) break;
            const auto graph_started = std::chrono::steady_clock::now();
            std::vector<int32_t> tokens(options.num_beams, c.pad_token_id);
            for (const auto & beam : active) tokens[beam.slot] = beam.tokens.back();
            current_logits = &beam_graph_->step(tokens, steps + step);
            graph_ms += debug::elapsed_ms(graph_started);
        }
        for (const auto & beam : active) {
            finished.push_back({beam.score / std::pow(static_cast<float>(std::max<size_t>(1, beam.tokens.size())),
                options.length_penalty), beam.tokens});
        }
        if (finished.empty()) throw std::runtime_error("Hviske v6 beam search returned no hypothesis");
        const auto best = std::max_element(finished.begin(), finished.end(),
            [](const auto & a, const auto & b) { return a.score < b.score; });
        debug::timing_log_scalar("hviske_asr_v6.decoder.decode_ms", debug::elapsed_ms(started));
        debug::timing_log_scalar("hviske_asr_v6.decoder.search_ms", search_ms);
        debug::timing_log_scalar("hviske_asr_v6.decoder.cache_copy_ms", cache_copy_ms);
        debug::timing_log_scalar("hviske_asr_v6.decoder.graph_ms", graph_ms);
        return best->tokens;
    }

    std::shared_ptr<const HviskeV6Assets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    modules::CausalDecoderRuntimeWeights weights_;
    modules::CausalDecoderConfig config_;
    std::unique_ptr<HviskeV6Qwen3BeamGraph> beam_graph_;
    std::array<int32_t, 5> special_ids_;
    std::vector<float> special_embeddings_;
};

HviskeV6Qwen3DecoderRuntime::HviskeV6Qwen3DecoderRuntime(std::shared_ptr<const HviskeV6Assets> assets,
    core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}
HviskeV6Qwen3DecoderRuntime::~HviskeV6Qwen3DecoderRuntime() = default;
std::vector<int32_t> HviskeV6Qwen3DecoderRuntime::generate(
    const std::vector<float> & audio, const HviskeV6DecodingOptions & options) {
    return impl_->generate(audio, options);
}

}  // namespace engine::models::hviske_asr_v6
