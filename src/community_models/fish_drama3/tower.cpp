#include "engine/community_models/fish_drama3/tower.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <list>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::fish_drama3 {
namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const { if (context) ggml_free(context); }
};

struct LayerWeights {
    ggml_tensor * embedding = nullptr;
    ggml_tensor * enorm = nullptr;
    ggml_tensor * hnorm = nullptr;
    ggml_tensor * eh = nullptr;
    ggml_tensor * input_norm = nullptr;
    ggml_tensor * qkv = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * post_norm = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * final_norm = nullptr;
    ggml_tensor * head = nullptr;
    ggml_tensor * key_cache = nullptr;
    ggml_tensor * value_cache = nullptr;
};

struct LayerGraph {
    std::unique_ptr<ggml_context, ContextDeleter> context;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * codes = nullptr;
    ggml_tensor * label_mask = nullptr;
    ggml_tensor * extra = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * logits = nullptr;
    std::vector<ggml_tensor *> book_codes;
    ggml_backend_t backend = nullptr;

    ~LayerGraph() {
        if (graph && backend) core::release_backend_graph_resources(backend, graph, true);
        if (allocator) ggml_gallocr_free(allocator);
    }
};

ggml_tensor * gemma_norm(ggml_context * context, ggml_tensor * input,
                         ggml_tensor * weight, float epsilon) {
    input = ggml_cont(context, input);
    if (weight->type != GGML_TYPE_F32) weight = ggml_cast(context, weight, GGML_TYPE_F32);
    return ggml_mul(context, ggml_rms_norm(context, input, epsilon),
                    ggml_scale_bias(context, weight, 1.0F, 1.0F));
}

ggml_tensor * columns(ggml_context * context, ggml_tensor * input,
                      int64_t begin, int64_t count) {
    return ggml_cont(context, ggml_view_2d(context, input, count, input->ne[1],
                                          input->nb[1], begin * input->nb[0]));
}

ggml_tensor * partial_rope(ggml_context * context, ggml_tensor * input,
                           ggml_tensor * positions, const Drama3TowerConfig & config) {
    const int64_t rotary = static_cast<int64_t>(config.head_dim * config.rotary_fraction);
    if (rotary == config.head_dim) {
        return ggml_rope_ext(context, input, positions, nullptr, rotary,
                             GGML_ROPE_TYPE_NORMAL, 0, config.rope_theta, 1, 0, 1, 0, 0);
    }
    auto * rotating = ggml_view_3d(context, input, rotary, input->ne[1], input->ne[2],
                                   input->nb[1], input->nb[2], 0);
    auto * remainder = ggml_view_3d(context, input, config.head_dim - rotary,
                                    input->ne[1], input->ne[2], input->nb[1], input->nb[2],
                                    rotary * input->nb[0]);
    rotating = ggml_rope_ext(context, ggml_cont(context, rotating), positions, nullptr,
                             rotary, GGML_ROPE_TYPE_NORMAL, 0, config.rope_theta, 1, 0, 1, 0, 0);
    return ggml_concat(context, rotating, remainder, 0);
}

bool semantic(const Drama3Frame & frame, const Drama3TowerConfig & config) {
    return frame[0] >= config.semantic_begin &&
        frame[0] < config.semantic_begin + config.codebook_size;
}

}  // namespace

class Drama3Tower::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, Drama3TowerConfig config,
         core::ExecutionContext & execution, int64_t capacity, size_t prefill_cache_bytes)
        : config_(std::move(config)), execution_(execution), capacity_(capacity), prefill_cache_budget_(prefill_cache_bytes) {
        const auto & c = config_;
        if (!source || capacity <= 0 || c.hidden_size <= 0 || c.heads <= 0 || c.kv_heads <= 0 ||
            c.heads % c.kv_heads || c.shared_layers <= 0 || c.residual_layers != 9 ||
            c.backbone_codebooks < 0 || c.backbone_codebooks > 9 ||
            static_cast<int64_t>(c.head_dim * c.rotary_fraction) % 2) {
            throw std::runtime_error("invalid Drama 3 tower configuration");
        }
        code_embeddings_ = source->require_f32("audio_decoder.codebook_embeddings.weight",
                                               {10 * c.codebook_size, c.hidden_size});
        store_ = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
                                                          "drama3.tower", 4 * 1024 * 1024);
        const auto load = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            return store_->load_tensor(*source, name, assets::TensorStorageType::Native, shape).tensor;
        };
        for (int64_t index = 0; index < c.shared_layers + c.residual_layers; ++index) {
            const bool shared = index < c.shared_layers;
            const int64_t local_index = shared ? index : index - c.shared_layers;
            const std::string prefix = std::string(shared ? "audio_decoder_mtp_shared.layers." : "audio_decoder_mtp.layers.") +
                std::to_string(local_index);
            LayerWeights w;
            w.embedding = load(prefix + (shared ? ".semantic_embedding.weight" : ".codebook_embedding.weight"),
                               {c.codebook_size, c.hidden_size});
            w.enorm = load(prefix + ".enorm.weight", {c.hidden_size});
            w.hnorm = load(prefix + ".hnorm.weight", {c.hidden_size});
            w.eh = load(prefix + ".eh_proj.weight", {c.hidden_size, 2 * c.hidden_size});
            w.final_norm = load(prefix + ".final_layernorm.weight", {c.hidden_size});
            const std::string transformer = prefix + ".transformer_layer";
            w.input_norm = load(transformer + ".attention_norm.weight", {c.hidden_size});
            w.post_norm = load(transformer + ".ffn_norm.weight", {c.hidden_size});
            const int64_t query_width = c.heads * c.head_dim;
            const int64_t kv_width = c.kv_heads * c.head_dim;
            w.qkv = load(transformer + ".attention.wqkv.weight", {2 * query_width + 2 * kv_width, c.hidden_size});
            w.q_norm = load(transformer + ".attention.q_norm.weight", {c.head_dim});
            w.k_norm = load(transformer + ".attention.k_norm.weight", {c.head_dim});
            w.output = load(transformer + ".attention.wo.weight", {c.hidden_size, query_width});
            w.gate = load(transformer + ".feed_forward.w1.weight", {c.intermediate_size, c.hidden_size});
            w.up = load(transformer + ".feed_forward.w3.weight", {c.intermediate_size, c.hidden_size});
            w.down = load(transformer + ".feed_forward.w2.weight", {c.hidden_size, c.intermediate_size});
            if (!shared) w.head = load("audio_decoder_mtp.heads." + std::to_string(local_index) + ".weight",
                                      {c.codebook_size, c.hidden_size});
            layers_.push_back(w);
        }
        store_->upload();
        ggml_init_params params{1024 * 1024, nullptr, true};
        cache_context_.reset(ggml_init(params));
        for (auto & layer : layers_) {
            layer.key_cache = ggml_new_tensor_2d(cache_context_.get(), GGML_TYPE_F16,
                                                 c.kv_heads * c.head_dim, capacity);
            layer.value_cache = ggml_new_tensor_2d(cache_context_.get(), GGML_TYPE_F16,
                                                   c.kv_heads * c.head_dim, capacity);
        }
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_context_.get(), execution.backend());
        if (!cache_buffer_) throw std::runtime_error("failed to allocate Drama 3 tower cache");
        graphs_.resize(layers_.size());
        reset();
    }

    ~Impl() {
        prefill_graphs_.clear();
        graphs_.clear();
        if (cache_buffer_) ggml_backend_buffer_free(cache_buffer_);
    }

    void reset() { ggml_backend_buffer_clear(cache_buffer_, 0); }

    void release_prefill_graphs() {
        prefill_graphs_.clear();
        prefill_cached_bytes_ = 0;
    }

    void mix_backbone_embeddings(std::vector<float> & values, const std::vector<Drama3Frame> & frames) const {
        if (values.size() != frames.size() * static_cast<size_t>(config_.hidden_size)) {
            throw std::runtime_error("Drama 3 frame embedding size mismatch");
        }
        const float scale = 1.0F / std::sqrt(1.0F + config_.backbone_codebooks);
        for (size_t row = 0; row < frames.size(); ++row) {
            if (!semantic(frames[row], config_)) continue;
            auto * output = values.data() + row * config_.hidden_size;
            add_codes(output, frames[row], 1, 1 + config_.backbone_codebooks);
            for (int64_t column = 0; column < config_.hidden_size; ++column) output[column] *= scale;
        }
    }

    std::vector<float> shared_input(const std::vector<Drama3Frame> & frames) const {
        std::vector<float> result(frames.size() * config_.hidden_size, 0.0F);
        for (size_t row = 0; row < frames.size(); ++row) {
            if (semantic(frames[row], config_)) {
                add_codes(result.data() + row * config_.hidden_size, frames[row],
                          1 + config_.backbone_codebooks, 10);
            }
        }
        return result;
    }

    void prefill(const std::vector<float> & input, const std::vector<Drama3Frame> & frames) {
        if (frames.empty() || input.size() != frames.size() * static_cast<size_t>(config_.hidden_size)) {
            throw std::runtime_error("Drama 3 tower prompt hidden size mismatch");
        }
        // The last prompt row predicts the first generated frame. Its tower
        // cache entry is filled by generation, after the semantic draw.
        const int64_t steps = static_cast<int64_t>(frames.size()) - 1;
        if (steps == 0) return;
        if (steps > capacity_) throw std::runtime_error("Drama 3 tower prefill exceeds cache length");
        std::vector<float> hidden(input.begin(), input.begin() + steps * config_.hidden_size);
        std::vector<Drama3Frame> current(frames.begin(), frames.end() - 1);
        std::vector<float> masks(static_cast<size_t>(steps));
        std::vector<int32_t> codes(static_cast<size_t>(steps));
        for (int64_t row = 0; row < steps; ++row) masks[row] = semantic(frames[row + 1], config_) ? 1.0F : 0.0F;
        const auto extra = shared_input(current);
        std::unique_ptr<LayerGraph> temporary;
        auto * graph = prefill_graph(steps, temporary);
        ggml_backend_tensor_set(graph->hidden, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(graph->label_mask, masks.data(), 0, masks.size() * sizeof(float));
        ggml_backend_tensor_set(graph->extra, extra.data(), 0, extra.size() * sizeof(float));
        std::vector<int32_t> positions(static_cast<size_t>(steps));
        for (int64_t row = 0; row < steps; ++row) positions[row] = static_cast<int32_t>(row);
        ggml_backend_tensor_set(graph->positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        std::vector<ggml_fp16_t> attention_mask(static_cast<size_t>(capacity_ * steps));
        const auto negative_infinity = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        for (int64_t row = 0; row < steps; ++row) for (int64_t key = 0; key < capacity_; ++key) {
            attention_mask[row * capacity_ + key] = key <= row ? ggml_fp32_to_fp16(0) : negative_infinity;
        }
        ggml_backend_tensor_set(graph->attention_mask, attention_mask.data(), 0, attention_mask.size() * sizeof(ggml_fp16_t));
        for (size_t book = 0; book < graph->book_codes.size(); ++book) {
            for (int64_t row = 0; row < steps; ++row) {
                codes[row] = frames[row + 1][1 + book];
                if (codes[row] < 0 || codes[row] >= config_.codebook_size) throw std::runtime_error("Drama 3 code is outside the codebook");
            }
            ggml_backend_tensor_set(graph->book_codes[book], codes.data(), 0, codes.size() * sizeof(int32_t));
        }
        if (core::compute_backend_graph(execution_.backend(), graph->graph, nullptr, "drama3.tower.prefill") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Drama 3 tower prefill computation failed");
        }
        // Prefill only populates the layer caches. The last backbone row, not
        // this hidden output, seeds generation. Keep all intermediates on GPU.
        ggml_backend_synchronize(execution_.backend());
    }

    Drama3TowerOutput run_layer(int64_t layer, const std::vector<float> & hidden,
                               const std::vector<int32_t> & codes, const std::vector<float> & masks,
                               const std::vector<float> & extra, int64_t position, int64_t steps) {
        if (layer < 0 || layer >= static_cast<int64_t>(layers_.size()) || steps <= 0 || position < 0 ||
            position + steps > capacity_ || hidden.size() != static_cast<size_t>(steps * config_.hidden_size) ||
            codes.size() != static_cast<size_t>(steps) || masks.size() != static_cast<size_t>(steps) ||
            (!extra.empty() && extra.size() != hidden.size())) {
            throw std::runtime_error("invalid Drama 3 tower layer input");
        }
        for (int32_t code : codes) {
            if (code < 0 || code >= config_.codebook_size) throw std::runtime_error("Drama 3 code is outside the codebook");
        }
        std::unique_ptr<LayerGraph> temporary;
        LayerGraph * graph = nullptr;
        if (steps == 1) {
            if (!graphs_[layer]) graphs_[layer] = build_graph(layer, steps);
            graph = graphs_[layer].get();
        } else {
            temporary = build_graph(layer, steps);
            graph = temporary.get();
        }
        ggml_backend_tensor_set(graph->hidden, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(graph->codes, codes.data(), 0, codes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(graph->label_mask, masks.data(), 0, masks.size() * sizeof(float));
        if (graph->extra) {
            std::vector<float> zero;
            if (extra.empty()) zero.resize(hidden.size(), 0.0F);
            const auto & values = extra.empty() ? zero : extra;
            ggml_backend_tensor_set(graph->extra, values.data(), 0, values.size() * sizeof(float));
        }
        std::vector<int32_t> positions(static_cast<size_t>(steps));
        for (int64_t row = 0; row < steps; ++row) positions[row] = static_cast<int32_t>(position + row);
        ggml_backend_tensor_set(graph->positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        std::vector<ggml_fp16_t> attention_mask(static_cast<size_t>(capacity_ * steps));
        const auto negative_infinity = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        for (int64_t row = 0; row < steps; ++row) {
            for (int64_t key = 0; key < capacity_; ++key) {
                attention_mask[row * capacity_ + key] = key <= position + row ? ggml_fp32_to_fp16(0) : negative_infinity;
            }
        }
        ggml_backend_tensor_set(graph->attention_mask, attention_mask.data(), 0, attention_mask.size() * sizeof(ggml_fp16_t));
        if (core::compute_backend_graph(execution_.backend(), graph->graph, nullptr, "drama3.tower") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Drama 3 tower computation failed");
        }
        Drama3TowerOutput result;
        result.hidden = core::read_tensor_f32(graph->output);
        if (graph->logits) result.logits = core::read_tensor_f32(graph->logits);
        return result;
    }

    Drama3TowerConfig config_;

private:
    struct PrefillGraphs {
        int64_t steps;
        size_t bytes = 0;
        std::unique_ptr<LayerGraph> graph;
    };

    LayerGraph * prefill_graph(int64_t steps, std::unique_ptr<LayerGraph> & temporary) {
        if (prefill_cache_budget_ == 0 || uncached_prefill_shapes_.count(steps)) {
            temporary = build_prefill_graph(steps);
            return temporary.get();
        }
        const auto found = std::find_if(prefill_graphs_.begin(), prefill_graphs_.end(),
            [steps](const auto & entry) { return entry.steps == steps; });
        if (found != prefill_graphs_.end()) {
            prefill_graphs_.splice(prefill_graphs_.begin(), prefill_graphs_, found);
            return prefill_graphs_.front().graph.get();
        }
        auto first = build_prefill_graph(steps);
        const size_t estimate = ggml_gallocr_get_buffer_size(first->allocator, 0);
        if (estimate > prefill_cache_budget_) {
            if (uncached_prefill_shapes_.size() >= 32) uncached_prefill_shapes_.erase(uncached_prefill_shapes_.begin());
            uncached_prefill_shapes_.insert(steps);
            temporary = std::move(first);
            return temporary.get();
        }
        while (!prefill_graphs_.empty() &&
               (prefill_cached_bytes_ + estimate > prefill_cache_budget_ || prefill_graphs_.size() >= 8)) {
            prefill_cached_bytes_ -= prefill_graphs_.back().bytes;
            prefill_graphs_.pop_back();
        }
        PrefillGraphs entry{steps};
        entry.graph = std::move(first);
        entry.bytes = estimate;
        prefill_cached_bytes_ += entry.bytes;
        prefill_graphs_.push_front(std::move(entry));
        return prefill_graphs_.front().graph.get();
    }

    void add_codes(float * output, const Drama3Frame & frame, int64_t begin, int64_t end) const {
        for (int64_t book = begin; book < end; ++book) {
            const int32_t code = frame[book + 1];
            if (code < 0 || code >= config_.codebook_size) throw std::runtime_error("invalid Drama 3 frame code");
            const auto * embedding = code_embeddings_.data() + (book * config_.codebook_size + code) * config_.hidden_size;
            for (int64_t column = 0; column < config_.hidden_size; ++column) output[column] += embedding[column];
        }
    }

    ggml_tensor * build_layer(ggml_context * ctx, int64_t layer, int64_t steps,
                             ggml_tensor * hidden, ggml_tensor * codes, ggml_tensor * label_mask,
                             ggml_tensor * extra, ggml_tensor * positions, ggml_tensor * attention_mask) {
        const auto & c = config_;
        const auto & w = layers_[layer];
        auto * embedding = ggml_get_rows(ctx, w.embedding, codes);
        embedding = ggml_mul(ctx, embedding, label_mask);
        if (extra) embedding = ggml_add(ctx, embedding, extra);
        auto * mixed = ggml_mul_mat(ctx, w.eh, ggml_concat(ctx,
            gemma_norm(ctx, embedding, w.enorm, c.norm_eps),
            gemma_norm(ctx, hidden, w.hnorm, c.norm_eps), 0));
        auto * qkv = ggml_mul_mat(ctx, w.qkv, gemma_norm(ctx, mixed, w.input_norm, c.norm_eps));
        const int64_t query_width = c.heads * c.head_dim;
        const int64_t kv_width = c.kv_heads * c.head_dim;
        auto * query = ggml_reshape_3d(ctx, columns(ctx, qkv, 0, query_width), c.head_dim, c.heads, steps);
        auto * gate = columns(ctx, qkv, query_width, query_width);
        auto * key = ggml_reshape_3d(ctx, columns(ctx, qkv, 2 * query_width, kv_width), c.head_dim, c.kv_heads, steps);
        auto * value = columns(ctx, qkv, 2 * query_width + kv_width, kv_width);
        query = partial_rope(ctx, gemma_norm(ctx, query, w.q_norm, c.norm_eps), positions, c);
        key = partial_rope(ctx, gemma_norm(ctx, key, w.k_norm, c.norm_eps), positions, c);
        key = ggml_reshape_2d(ctx, ggml_cont(ctx, key), kv_width, steps);
        auto * updated_key = ggml_set_rows(ctx, w.key_cache, key, positions);
        auto * updated_value = ggml_set_rows(ctx, w.value_cache, value, positions);
        auto * keys = ggml_permute(ctx, ggml_reshape_3d(ctx, updated_key, c.head_dim, c.kv_heads, capacity_), 0, 2, 1, 3);
        auto * values = ggml_permute(ctx, ggml_reshape_3d(ctx, updated_value, c.head_dim, c.kv_heads, capacity_), 0, 2, 1, 3);
        query = ggml_cont(ctx, ggml_permute(ctx, query, 0, 2, 1, 3));
        auto * attended = ggml_flash_attn_ext(ctx, query, keys, values, attention_mask,
                                             1.0F / std::sqrt(static_cast<float>(c.head_dim)), 0, 0);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        attended = ggml_reshape_2d(ctx, attended, query_width, steps);
        attended = ggml_mul(ctx, attended, ggml_sigmoid(ctx, gate));
        auto * x = ggml_add(ctx, mixed, ggml_mul_mat(ctx, w.output, attended));
        auto * normed = gemma_norm(ctx, x, w.post_norm, c.norm_eps);
        auto * ff = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, w.gate, normed)), ggml_mul_mat(ctx, w.up, normed));
        return gemma_norm(ctx, ggml_add(ctx, x, ggml_mul_mat(ctx, w.down, ff)), w.final_norm, c.norm_eps);
    }

    std::unique_ptr<LayerGraph> build_prefill_graph(int64_t steps) {
        auto result = std::make_unique<LayerGraph>();
        result->backend = execution_.backend();
        ggml_init_params params{8 * 1024 * 1024, nullptr, true};
        result->context.reset(ggml_init(params));
        auto * ctx = result->context.get();
        result->graph = ggml_new_graph_custom(ctx, 8192, false);
        const auto input = [&](ggml_type type, int64_t columns, int64_t rows) {
            auto * tensor = ggml_new_tensor_2d(ctx, type, columns, rows);
            ggml_set_input(tensor);
            return tensor;
        };
        result->hidden = input(GGML_TYPE_F32, config_.hidden_size, steps);
        result->extra = input(GGML_TYPE_F32, config_.hidden_size, steps);
        result->label_mask = input(GGML_TYPE_F32, 1, steps);
        result->positions = input(GGML_TYPE_I32, steps, 1);
        result->attention_mask = input(GGML_TYPE_F16, capacity_, steps);
        result->book_codes.resize(static_cast<size_t>(std::max<int64_t>(1, config_.residual_layers)));
        for (auto & codes : result->book_codes) codes = input(GGML_TYPE_I32, steps, 1);
        auto * hidden = result->hidden;
        for (size_t layer = 0; layer < layers_.size(); ++layer) {
            const size_t book = layer < static_cast<size_t>(config_.shared_layers) ? 0 : layer - config_.shared_layers;
            hidden = build_layer(ctx, layer, steps, hidden, result->book_codes[book], result->label_mask,
                layer == 0 ? result->extra : nullptr, result->positions, result->attention_mask);
        }
        result->output = hidden;
        ggml_set_output(hidden);
        ggml_build_forward_expand(result->graph, hidden);
        result->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!result->allocator || !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
            throw std::runtime_error("failed to allocate Drama 3 tower prefill graph");
        }
        return result;
    }

    std::unique_ptr<LayerGraph> build_graph(int64_t layer, int64_t steps) {
        const auto & c = config_;
        const auto & w = layers_[layer];
        auto result = std::make_unique<LayerGraph>();
        result->backend = execution_.backend();
        ggml_init_params params{4 * 1024 * 1024, nullptr, true};
        result->context.reset(ggml_init(params));
        auto * ctx = result->context.get();
        result->graph = ggml_new_graph_custom(ctx, 4096, false);
        const auto input = [&](ggml_type type, int64_t columns, int64_t rows) {
            auto * tensor = ggml_new_tensor_2d(ctx, type, columns, rows);
            ggml_set_input(tensor);
            return tensor;
        };
        result->hidden = input(GGML_TYPE_F32, c.hidden_size, steps);
        result->codes = input(GGML_TYPE_I32, steps, 1);
        result->label_mask = input(GGML_TYPE_F32, 1, steps);
        result->positions = input(GGML_TYPE_I32, steps, 1);
        result->attention_mask = input(GGML_TYPE_F16, capacity_, steps);
        if (layer == 0) {
            result->extra = input(GGML_TYPE_F32, c.hidden_size, steps);
        }
        result->output = build_layer(ctx, layer, steps, result->hidden, result->codes, result->label_mask,
            result->extra, result->positions, result->attention_mask);
        ggml_set_output(result->output);
        ggml_build_forward_expand(result->graph, result->output);
        // Prompt prefill needs the states, but does not sample codebook heads.
        if (w.head && steps == 1) {
            result->logits = ggml_mul_mat(ctx, w.head, result->output);
            ggml_set_output(result->logits);
            ggml_build_forward_expand(result->graph, result->logits);
        }
        result->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!result->allocator || !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
            throw std::runtime_error("failed to allocate Drama 3 tower graph");
        }
        return result;
    }

    core::ExecutionContext & execution_;
    int64_t capacity_;
    std::vector<float> code_embeddings_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::vector<LayerWeights> layers_;
    std::unique_ptr<ggml_context, ContextDeleter> cache_context_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::vector<std::unique_ptr<LayerGraph>> graphs_;
    size_t prefill_cache_budget_;
    size_t prefill_cached_bytes_ = 0;
    std::list<PrefillGraphs> prefill_graphs_;
    std::set<int64_t> uncached_prefill_shapes_;
};

Drama3Tower::Drama3Tower(std::shared_ptr<const assets::TensorSource> weights,
                         Drama3TowerConfig config, core::ExecutionContext & execution, int64_t cache_steps, size_t prefill_cache_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), std::move(config), execution, cache_steps, prefill_cache_bytes)) {}
Drama3Tower::~Drama3Tower() = default;
void Drama3Tower::reset() { impl_->reset(); }
void Drama3Tower::release_prefill_graphs() { impl_->release_prefill_graphs(); }
void Drama3Tower::mix_backbone_embeddings(std::vector<float> & embeddings, const std::vector<Drama3Frame> & frames) const {
    impl_->mix_backbone_embeddings(embeddings, frames);
}
std::vector<float> Drama3Tower::shared_input(const std::vector<Drama3Frame> & frames) const { return impl_->shared_input(frames); }
void Drama3Tower::prefill(const std::vector<float> & hidden, const std::vector<Drama3Frame> & frames) { impl_->prefill(hidden, frames); }
Drama3TowerOutput Drama3Tower::run_layer(int64_t layer, const std::vector<float> & hidden,
                                        const std::vector<int32_t> & codes, const std::vector<float> & semantic_mask,
                                        const std::vector<float> & extra_input, int64_t position, int64_t steps) {
    return impl_->run_layer(layer, hidden, codes, semantic_mask, extra_input, position, steps);
}
const Drama3TowerConfig & Drama3Tower::config() const noexcept { return impl_->config_; }

}  // namespace engine::models::fish_drama3
