#include "engine/community_models/zonos3/backbone.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/moe_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/staged_moe_runtime.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos3 {
namespace {

constexpr int kPrefillChunk = 256;
constexpr int kGraphNodes = 8192;

using core::TensorValue;

struct LayerWeights {
    bool gdn = false;
    int top_k = 0;  // 0 = dense FFN
    TensorValue attn_norm, ffn_norm;
    TensorValue wq, wk, wv, wo, attn_temp, attn_gate;           // global attention
    TensorValue gdn_in, gdn_conv, gdn_norm, gdn_a_log, gdn_dt;  // gated DeltaNet
    TensorValue ffn_up, ffn_gate, ffn_down;                     // dense SwiGLU
    TensorValue up_exps, gate_exps, down_exps, latent_down, latent_up, router, router_norm, neg_router_bias;
    size_t disk_group = 0;
    int state_index = -1;  // index into the KV caches (global layers) or GDN states
};

struct GraphShape {
    int rows_per_slot = 0;  // T
    int slots = 0;          // B
    int first_slot = 0;
    bool prefill = false;
    int rows() const { return rows_per_slot * slots; }
};

// Persistent graph inputs; one set per prepared graph.
struct GraphInputs {
    ggml_tensor * ids = nullptr;      // I32 [N, n_codebooks + 1], codebook-major
    ggml_tensor * pos = nullptr;      // I32 [N]
    ggml_tensor * kv_rows = nullptr;  // I32 [n_head_kv * N]
    ggml_tensor * mask = nullptr;     // F16 [cap, T, 1, B]
    ggml_tensor * keep = nullptr;     // F32 [1, N] (prefill)
    ggml_tensor * sel = nullptr;      // I32 [N] (prefill)
    ggml_tensor * speaker = nullptr;  // F32 [speaker_dim, 1] (prefill)
    ggml_tensor * zero = nullptr;     // F32 [n_embd, 1] (prefill)
    ggml_tensor * buckets = nullptr;  // I32 [n_cond] (prefill)
};

ggml_tensor * mul_mat_f32(ggml_context * g, ggml_tensor * a, ggml_tensor * b) {
    auto * out = ggml_mul_mat(g, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

}  // namespace

class Zonos3Backbone::Impl {
public:
    Impl(std::shared_ptr<const Zonos3Assets> assets, core::ExecutionContext & execution, Zonos3BackboneOptions options)
        : assets_(std::move(assets)), hp_(assets_->hparams), execution_(execution), options_(options) {
        load_weights();
    }

    ~Impl() {
        decode_.clear();
        prefill_.clear();
        offload_.clear();
    }

    void begin(int slots, int capacity_rows) {
        if (slots < 1 || slots > 2) throw std::runtime_error("zonos3: the backbone runs one or two slots");
        if (capacity_rows <= 0 || capacity_rows > hp_.n_ctx_train) {
            throw std::runtime_error("zonos3: backbone capacity must be 1.." + std::to_string(hp_.n_ctx_train));
        }
        const int capacity = std::min(hp_.n_ctx_train, (capacity_rows + 255) / 256 * 256);
        if (capacity > capacity_ || slots > slots_) {
            decode_.clear();
            prefill_.clear();
            offload_.clear();
            state_buffer_.reset();
            state_context_.reset();
            capacity_ = std::max(capacity, capacity_);
            slots_ = std::max(slots, slots_);
            allocate_state();
        }
        ggml_backend_buffer_clear(state_buffer_.get(), 0);
        n_past_.assign(2, 0);
        if (disk_store_) disk_store_->invalidate_cache();
    }

    void prefill(int slot, const Zonos3Prompt & prompt, std::vector<float> & logits) {
        if (slot < 0 || slot >= slots_) throw std::runtime_error("zonos3: prefill slot is not allocated");
        const int width = hp_.n_codebooks + 1;
        if (prompt.n_rows <= 0 || prompt.n_rows > capacity_ ||
            prompt.ids.size() != static_cast<size_t>(prompt.n_rows) * static_cast<size_t>(width)) {
            throw std::runtime_error("zonos3: prompt does not fit the backbone capacity");
        }
        for (int position = 0; position < prompt.n_rows; position += kPrefillChunk) {
            const int rows = std::min(kPrefillChunk, prompt.n_rows - position);
            const GraphShape shape{rows, 1, slot, true};
            if (disk_store_) {
                auto & graph = offload_graph(shape);
                set_prefill_inputs(graph.inputs, shape, prompt, position);
                graph.run();
                if (position + rows == prompt.n_rows) read_logits(graph.logits, 1, logits);
            } else {
                auto & graph = forward_graph(prefill_, shape);
                set_prefill_inputs(graph.inputs, shape, prompt, position);
                graph.prepared->compute();
                if (position + rows == prompt.n_rows) read_logits(graph.logits, 1, logits);
            }
        }
        n_past_[static_cast<size_t>(slot)] = prompt.n_rows;
    }

    void step(const std::array<const int32_t *, 2> & rows, int slots, std::vector<float> & logits) {
        if (slots < 1 || slots > slots_) throw std::runtime_error("zonos3: decode slots are not allocated");
        for (int b = 0; b < slots; ++b) {
            if (n_past_[static_cast<size_t>(b)] >= capacity_) throw std::runtime_error("zonos3: backbone capacity exceeded");
        }
        const GraphShape shape{1, slots, 0, false};
        auto & graph = forward_graph(decode_, shape);
        const int width = hp_.n_codebooks + 1;
        step_ids_.resize(static_cast<size_t>(width * slots));
        for (int c = 0; c < width; ++c) {
            for (int b = 0; b < slots; ++b) step_ids_[static_cast<size_t>(c * slots + b)] = rows[static_cast<size_t>(b)][c];
        }
        step_positions_.resize(static_cast<size_t>(slots));
        for (int b = 0; b < slots; ++b) step_positions_[static_cast<size_t>(b)] = n_past_[static_cast<size_t>(b)];
        set_common_inputs(graph.inputs, shape, step_ids_, step_positions_);
        graph.prepared->compute();
        read_logits(graph.logits, slots, logits);
        for (int b = 0; b < slots; ++b) ++n_past_[static_cast<size_t>(b)];
    }

private:
    struct ForwardGraph {
        std::unique_ptr<core::PreparedGraph> prepared;
        GraphShape shape;
        GraphInputs inputs;
        ggml_tensor * logits = nullptr;
    };

    // Expert-cache prefill: a 256-row chunk routes more experts than the per-layer cache holds, so every MoE layer
    // runs through StagedMoeRuntime between per-layer graphs. Full AttnRes sources cross graph boundaries through
    // persistent stacks.
    struct OffloadGraph {
        GraphShape shape;
        GraphInputs inputs;
        std::shared_ptr<core::GraphWorkspace> workspace;
        std::unique_ptr<core::PreparedGraph> prepared;
        std::unique_ptr<modules::StagedMoeRuntime> staged;
        std::vector<core::PreparedGraph::GraphId> layer_graphs, finish_graphs;
        std::vector<int64_t> staged_layers;
        core::PreparedGraph::GraphId final_graph = 0;
        ggml_tensor * logits = nullptr;

        void run() {
            prepared->compute(0);
            uint64_t rows = 0;
            for (size_t layer = 0; layer < layer_graphs.size(); ++layer) {
                prepared->compute(layer_graphs[layer]);
                if (staged_layers[layer] < 0) continue;
                rows += staged->run(static_cast<size_t>(staged_layers[layer]));
                prepared->compute(finish_graphs[layer]);
            }
            prepared->compute(final_graph);
            debug::timing_log_scalar("zonos3.offload_prefill_expert_rows", rows);
        }
    };

    // Full AttnRes: every source is scored once against all 2L+1 queries as it enters the bank; a route softmaxes
    // one query row over the sources so far and sums the weighted sources in source order.
    class AttnRes {
    public:
        AttnRes(Impl & model, core::ModuleBuildContext & ctx, int rows) : m_(model), ctx_(ctx), rows_(rows) {}

        ggml_tensor * score(ggml_tensor * source) const {
            auto * g = ctx_.ggml;
            auto * sc = mul_mat_f32(g, m_.attnres_q_.tensor, ggml_rms_norm(g, source, m_.hp_.rms_eps));
            return ggml_reshape_3d(g, sc, m_.queries(), 1, rows_);
        }

        // Appends `source` (in-graph) and routes query `qi` over every source.
        ggml_tensor * route(ggml_tensor * source, int qi) {
            auto * g = ctx_.ggml;
            auto * column = ggml_reshape_3d(g, source, m_.hp_.n_embd, 1, rows_);
            auto * sc = score(source);
            scores_ = scores_ ? ggml_concat(g, scores_, sc, 1) : sc;
            sources_ = sources_ ? ggml_concat(g, sources_, column, 1) : column;
            if (++count_ == 1) return source;
            return mix(scores_, sources_, count_, qi);
        }

        // Routes over explicit stacks: scores [Q, n, rows], sources [D, n, rows].
        ggml_tensor * mix(ggml_tensor * scores, ggml_tensor * sources, int n, int qi) const {
            auto * g = ctx_.ggml;
            auto * row = ggml_view_3d(g, scores, 1, n, rows_, scores->nb[1], scores->nb[2],
                                      static_cast<size_t>(qi) * scores->nb[0]);
            auto * weights = ggml_soft_max(g, ggml_reshape_2d(g, ggml_cont(g, row), n, rows_));
            auto * weighted = ggml_mul(g, sources, ggml_reshape_3d(g, weights, 1, n, rows_));
            auto slice = [&](int s) {
                return ggml_view_2d(g, weighted, m_.hp_.n_embd, rows_, weighted->nb[2], static_cast<size_t>(s) * weighted->nb[1]);
            };
            auto * out = slice(0);
            for (int s = 1; s < n; ++s) out = ggml_add(g, out, slice(s));
            return out;
        }

    private:
        Impl & m_;
        core::ModuleBuildContext & ctx_;
        int rows_;
        ggml_tensor * scores_ = nullptr;
        ggml_tensor * sources_ = nullptr;
        int count_ = 0;
    };

    struct MoeInputs {
        modules::MoeRouting routing;
        ggml_tensor * latent = nullptr;  // [latent, rows]
    };

    int queries() const { return 2 * hp_.n_layer + 1; }
    static TensorValue matrix(ggml_tensor * x) { return core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[1], x->ne[0]})); }

    ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & weight) const {
        const auto out_features = weight.shape.dims[0];
        const auto in_features = weight.shape.dims[1];
        return modules::LinearModule({in_features, out_features, false, GGML_PREC_F32})
            .build(ctx, matrix(x), {weight, std::nullopt}).tensor;
    }

    ggml_tensor * rms_weighted(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & weight, float eps) const {
        return modules::RMSNormModule({weight.shape.dims[0], eps, true, false}).build(ctx, matrix(x), {weight, std::nullopt}).tensor;
    }

    core::BackendWeightStore & extra_store(ggml_backend_buffer_type_t buffer_type) {
        for (auto & entry : extra_stores_) {
            if (entry.first == buffer_type) return *entry.second;
        }
        extra_stores_.emplace_back(buffer_type, std::make_unique<core::BackendWeightStore>(
            execution_.backend(), execution_.backend_type(),
            std::string("zonos3.backbone.") + ggml_backend_buft_name(buffer_type), 16 * 1024 * 1024, buffer_type));
        return *extra_stores_.back().second;
    }

    void load_weights() {
        const auto & source = *assets_->backbone;
        const int D = hp_.n_embd;
        const int H = hp_.gdn_heads;
        const int S = hp_.gdn_head_dim;
        const int I = H * S;
        if (options_.moe.pin_host_weights && options_.moe.residency != core::MoeExpertResidency::Staged) {
            throw std::runtime_error("zonos3.pin_experts requires zonos3.disk_experts=true");
        }
        int top_k = 0;
        int min_experts = std::numeric_limits<int>::max();
        for (int layer = 1; layer < hp_.n_layer; ++layer) {
            const int used = hp_.expert_used[static_cast<size_t>(layer)];
            if (top_k != 0 && used != top_k && (options_.moe.residency == core::MoeExpertResidency::Staged)) {
                throw std::runtime_error("zonos3 disk_experts requires the same top-k in every MoE layer");
            }
            top_k = used;
            min_experts = std::min(min_experts, static_cast<int>(
                source.require_metadata("blk." + std::to_string(layer) + ".ffn_up_exps.weight").shape[0]));
        }
        top_k_ = top_k;
        if (options_.moe.residency == core::MoeExpertResidency::Staged) {
            if (options_.moe.cache_slots < 2 * top_k || options_.moe.cache_slots > min_experts) {
                throw std::runtime_error("zonos3.expert_cache_size must be between " + std::to_string(2 * top_k) + " and " +
                                         std::to_string(min_experts));
            }
        }
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos3.backbone.weights", 16 * 1024 * 1024);
        extra_buffer_types_ = core::cpu_extra_buffer_types(execution_.backend());
        core::BackendWeightStore * expert_weights = store_.get();
        if (options_.moe.residency == core::MoeExpertResidency::Backend && options_.repack_experts) {
            for (auto * buffer_type : extra_buffer_types_) {
                bool supported = true;
                for (int layer = 1; layer < hp_.n_layer && supported; ++layer) {
                    for (const auto * projection : {"ffn_up_exps.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight"}) {
                        const auto & metadata = source.require_metadata("blk." + std::to_string(layer) + "." + projection);
                        const auto type = assets::ggml_type_for_tensor_storage(assets::tensor_storage_type_for_dtype(metadata.dtype));
                        // Expert banks need MUL_MAT_ID support; the framework's 2-D
                        // matmul probe does not establish that for every CPU buffer type.
                        std::unique_ptr<ggml_context, decltype(&ggml_free)> probe(
                            ggml_init({7 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
                        if (!probe) throw std::runtime_error("zonos3: cannot allocate expert buffer probe metadata");
                        auto * bank = ggml_new_tensor_3d(probe.get(), type, metadata.shape.at(2),
                                                        metadata.shape.at(1), metadata.shape.at(0));
                        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                            ggml_backend_buft_alloc_buffer(buffer_type, 0), ggml_backend_buffer_free);
                        bank->buffer = buffer.get();
                        for (const int rows : {1, kPrefillChunk}) {
                            auto * input = ggml_new_tensor_3d(probe.get(), GGML_TYPE_F32, bank->ne[0], 1, rows);
                            auto * ids = ggml_new_tensor_2d(probe.get(), GGML_TYPE_I32,
                                                           hp_.expert_used[static_cast<size_t>(layer)], rows);
                            if (!ggml_backend_dev_supports_op(ggml_backend_get_device(execution_.backend()),
                                                               ggml_mul_mat_id(probe.get(), bank, input, ids))) {
                                supported = false;
                                break;
                            }
                        }
                        if (!supported) break;
                    }
                }
                if (supported) {
                    expert_weights = &extra_store(buffer_type);
                    break;
                }
            }
        }
        resources_ = std::make_unique<core::MoeResources>(execution_, source, *expert_weights, options_.moe,
                                                        "zonos3.backbone.experts", 16 * 1024 * 1024);
        disk_store_ = resources_->disk_store();
        auto native = [&](const std::string & name, std::vector<int64_t> shape) {
            consumed_.insert(name);
            return store_->load_tensor(source, name, assets::TensorStorageType::Native, shape);
        };
        auto f32 = [&](const std::string & name, std::vector<int64_t> shape) {
            consumed_.insert(name);
            return store_->load_f32_tensor(source, name, shape);
        };
        // Quantized projection weights live in the CPU backend's repacked buffers when it has kernels for them, as
        // in ZONOS3's runtime: the repacked GEMMs round differently from the plain ones.
        auto matmul = [&](const std::string & name, std::vector<int64_t> shape,
                          assets::TensorStorageType storage = assets::TensorStorageType::Native) {
            consumed_.insert(name);
            const auto resolved = storage == assets::TensorStorageType::Native
                ? assets::tensor_storage_type_for_dtype(source.require_metadata(name).dtype) : storage;
            const ggml_type type = assets::ggml_type_for_tensor_storage(resolved);
            if (ggml_is_quantized(type)) {
                for (auto * buffer_type : extra_buffer_types_) {
                    if (!core::buffer_type_supports_matmul_weight(
                            execution_.backend(), buffer_type, type, shape[1], shape[0])) continue;
                    return extra_store(buffer_type).load_tensor(source, name, storage, shape);
                }
            }
            return store_->load_tensor(source, name, storage, shape);
        };
        for (int c = 0; c < hp_.n_codebooks; ++c) {
            audio_embd_.push_back(native("audio_embd." + std::to_string(c) + ".weight", {hp_.audio_vocab, D}));
        }
        text_embd_ = native("text_embd.weight", {hp_.text_vocab + 1 + hp_.text_extra_tokens, D});
        output_norm_ = f32("output_norm.weight", {D});
        consumed_.insert("output.weight");
        // Quantized checkpoints run their F16 logits head as Q8_0, quantized at load (ZONOS3's runtime does the same).
        output_ = matmul("output.weight", {static_cast<int64_t>(hp_.audio_vocab) * hp_.n_codebooks, D},
                         hp_.file_type > 1 ? assets::TensorStorageType::Q8_0 : assets::TensorStorageType::Native);
        attnres_q_ = native("attnres_q.weight", {queries(), D});
        speaker_proj_ = native("speaker_proj.weight", {D, hp_.speaker_dim});
        speaker_bias_ = f32("speaker_proj.bias", {D});
        for (int i = 0; i < hp_.n_cond; ++i) {
            cond_embd_.push_back(native("cond_embd." + std::to_string(i) + ".weight",
                                        {hp_.cond_bucket_counts[static_cast<size_t>(i)] + 1, D}));
        }
        int attention_layers = 0;
        int gdn_layers = 0;
        for (int layer = 0; layer < hp_.n_layer; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            LayerWeights w;
            w.attn_norm = f32(p + "attn_norm.weight", {D});
            w.ffn_norm = f32(p + "ffn_norm.weight", {D});
            w.gdn = (layer + 1) % 4 != 0;
            if (w.gdn) {
                w.state_index = gdn_layers++;
                w.gdn_in = matmul(p + "gdn_in.weight", {4 * I + 2 * H, D});
                w.gdn_conv = f32(p + "gdn_conv.weight", {3 * I, hp_.gdn_conv_kernel});
                w.gdn_norm = f32(p + "gdn_norm.weight", {S});
                w.gdn_a_log = f32(p + "gdn_A_log", {H});
                w.gdn_dt = f32(p + "gdn_dt_bias", {H});
                w.wo = matmul(p + "attn_output.weight", {D, I});
            } else {
                w.state_index = attention_layers++;
                w.wq = matmul(p + "attn_q.weight", {hp_.head_dim * hp_.n_head, D});
                w.wk = matmul(p + "attn_k.weight", {hp_.head_dim * hp_.n_head_kv, D});
                w.wv = matmul(p + "attn_v.weight", {hp_.head_dim * hp_.n_head_kv, D});
                w.wo = matmul(p + "attn_output.weight", {D, hp_.head_dim * hp_.n_head});
                w.attn_temp = f32(p + "attn_temp", {hp_.n_head});
                w.attn_gate = matmul(p + "attn_gate.weight", {hp_.n_head, D});
            }
            w.top_k = hp_.expert_used[static_cast<size_t>(layer)];
            if (w.top_k == 0) {
                w.ffn_up = matmul(p + "ffn_up.weight", {hp_.n_ff, D});
                w.ffn_gate = matmul(p + "ffn_gate.weight", {hp_.n_ff, D});
                w.ffn_down = matmul(p + "ffn_down.weight", {D, hp_.n_ff});
            } else {
                const auto bank = source.require_metadata(p + "ffn_up_exps.weight").shape;  // {E, hidden, latent}
                if (bank.size() != 3 || bank[0] < w.top_k || bank[0] > hp_.n_expert) {
                    throw std::runtime_error("zonos3: backbone GGUF: " + p + "ffn_up_exps.weight has an invalid shape");
                }
                const int64_t experts = bank[0];
                const int64_t hidden = bank[1];
                const int64_t latent = bank[2];
                const std::vector<std::string> names{p + "ffn_up_exps.weight", p + "ffn_gate_exps.weight",
                                                     p + "ffn_down_exps.weight"};
                std::vector<core::MoeExpertTensorBinding> bindings;
                const std::vector<std::vector<int64_t>> shapes{bank, bank, {experts, latent, hidden}};
                for (size_t i = 0; i < names.size(); ++i) {
                    consumed_.insert(names[i]);
                    bindings.push_back({names[i], names[i], shapes[i], assets::TensorStorageType::Native});
                }
                const auto & experts_layer = resources_->bind_layer(static_cast<size_t>(layer - 1),
                    static_cast<size_t>(layer), bindings, w.top_k);
                w.disk_group = experts_layer.disk_group;
                w.up_exps = experts_layer.weights[0];
                w.gate_exps = experts_layer.weights[1];
                w.down_exps = experts_layer.weights[2];
                w.latent_down = matmul(p + "ffn_latent_down.weight", {latent, D});
                w.latent_up = matmul(p + "ffn_latent_up.weight", {D, latent});
                w.router = matmul(p + "router.weight", {experts, D});
                w.router_norm = f32(p + "router_norm.weight", {D});
                consumed_.insert(p + "router_bias");
                auto bias = source.require_f32(p + "router_bias", {experts});
                for (auto & value : bias) value = -value;  // routing ranks logits - router_bias
                w.neg_router_bias = store_->make_f32(core::TensorShape::from_dims({experts}), std::move(bias));
                latent_ = static_cast<int>(latent);
            }
            layers_.push_back(std::move(w));
        }
        attention_layers_ = attention_layers;
        gdn_layers_ = gdn_layers;
        for (const auto & tensor : source.tensors()) {
            if (!consumed_.count(tensor.name)) {
                throw std::runtime_error("zonos3: backbone GGUF: unmapped tensor '" + tensor.name + "'");
            }
        }
        store_->upload();
        for (auto & extra : extra_stores_) extra.second->upload();
        resources_->finalize_weights();
        assets_->backbone->release_storage();
        if (disk_store_) {
            const auto stats = disk_store_->statistics();
            debug::timing_log_scalar("zonos3.expert_cache_mb", stats.buffer_bytes / (1024.0 * 1024.0));
            debug::timing_log_scalar("zonos3.host_experts_mb", stats.host_buffer_bytes / (1024.0 * 1024.0));
        }
    }

    void allocate_state() {
        const size_t tensors = static_cast<size_t>(2 * attention_layers_ + 2 * gdn_layers_ + 4);
        state_context_.reset(ggml_init({tensors * ggml_tensor_overhead(), nullptr, true}));
        if (!state_context_) throw std::runtime_error("zonos3: cannot allocate backbone state metadata");
        auto * g = state_context_.get();
        const int I = hp_.gdn_heads * hp_.gdn_head_dim;
        keys_.clear();
        values_.clear();
        gdn_states_.clear();
        gdn_history_.clear();
        for (int i = 0; i < attention_layers_; ++i) {
            keys_.push_back(ggml_new_tensor_4d(g, GGML_TYPE_F32, hp_.head_dim, capacity_, hp_.n_head_kv, slots_));
            values_.push_back(ggml_new_tensor_4d(g, GGML_TYPE_F32, hp_.head_dim, capacity_, hp_.n_head_kv, slots_));
        }
        for (int i = 0; i < gdn_layers_; ++i) {
            gdn_states_.push_back(ggml_new_tensor_3d(
                g, GGML_TYPE_F32, static_cast<int64_t>(hp_.gdn_head_dim) * hp_.gdn_head_dim * hp_.gdn_heads, 1, slots_));
            gdn_history_.push_back(ggml_new_tensor_3d(g, GGML_TYPE_F32, hp_.gdn_conv_kernel - 1, 3 * I, slots_));
        }
        state_buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution_.backend()));
        if (!state_buffer_) throw std::runtime_error("zonos3: cannot allocate backbone state");
        debug::timing_log_scalar("zonos3.state_mb", ggml_backend_buffer_get_size(state_buffer_.get()) / (1024.0 * 1024.0));
    }

    GraphInputs make_inputs(core::PreparedGraph & prepared, const GraphShape & shape) const {
        auto * g = prepared.persistent_build_context().ggml;
        const int N = shape.rows();
        GraphInputs in;
        in.ids = ggml_new_tensor_2d(g, GGML_TYPE_I32, N, hp_.n_codebooks + 1);
        in.pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, N);
        in.kv_rows = ggml_new_tensor_1d(g, GGML_TYPE_I32, static_cast<int64_t>(hp_.n_head_kv) * N);
        in.mask = ggml_new_tensor_4d(g, GGML_TYPE_F16, capacity_, shape.rows_per_slot, 1, shape.slots);
        if (shape.prefill) {
            in.keep = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, N);
            in.sel = ggml_new_tensor_1d(g, GGML_TYPE_I32, N);
            in.speaker = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp_.speaker_dim, 1);
            in.zero = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp_.n_embd, 1);
            in.buckets = ggml_new_tensor_1d(g, GGML_TYPE_I32, hp_.n_cond);
        }
        for (auto * t : {in.ids, in.pos, in.kv_rows, in.mask, in.keep, in.sel, in.speaker, in.zero, in.buckets}) {
            if (t) ggml_set_input(t);
        }
        return in;
    }

    // Embedding sum, speaker / condition row replacement (prefill), weightless emb_norm.
    ggml_tensor * embed(core::ModuleBuildContext & ctx, const GraphInputs & in, int rows) const {
        auto * g = ctx.ggml;
        ggml_tensor * emb = nullptr;
        for (int c = 0; c <= hp_.n_codebooks; ++c) {
            auto * ids = ggml_view_1d(g, in.ids, rows, static_cast<size_t>(c) * in.ids->nb[1]);
            ids = core::ensure_backend_addressable_layout(
                ctx, core::wrap_tensor(ids, core::TensorShape::from_dims({rows}), GGML_TYPE_I32)).tensor;
            const auto & table = c < hp_.n_codebooks ? audio_embd_[static_cast<size_t>(c)] : text_embd_;
            auto * e = ggml_get_rows(g, table.tensor, ids);
            emb = emb ? ggml_add(g, emb, e) : e;
        }
        if (in.keep) {
            auto speaker = modules::LinearModule({hp_.speaker_dim, hp_.n_embd, true, GGML_PREC_F32})
                               .build(ctx, matrix(in.speaker), {speaker_proj_, speaker_bias_});
            auto * table = ggml_concat(g, in.zero, speaker.tensor, 1);
            for (int i = 0; i < hp_.n_cond; ++i) {
                auto * bucket = ggml_view_1d(g, in.buckets, 1, static_cast<size_t>(i) * sizeof(int32_t));
                bucket = core::ensure_backend_addressable_layout(
                    ctx, core::wrap_tensor(bucket, core::TensorShape::from_dims({1}), GGML_TYPE_I32)).tensor;
                table = ggml_concat(g, table, ggml_get_rows(g, cond_embd_[static_cast<size_t>(i)].tensor, bucket), 1);
            }
            emb = ggml_add(g, ggml_mul(g, emb, in.keep), ggml_get_rows(g, table, in.sel));
        }
        return ggml_rms_norm(g, emb, hp_.rms_eps);
    }

    ggml_tensor * attention(core::ModuleBuildContext & ctx, const LayerWeights & w, ggml_tensor * x,
                            const GraphInputs & in, const GraphShape & shape) const {
        auto * g = ctx.ggml;
        const int hd = hp_.head_dim;
        const int nh = hp_.n_head;
        const int nkv = hp_.n_head_kv;
        const int T = shape.rows_per_slot;
        const int B = shape.slots;
        const int N = shape.rows();
        auto * gate = modules::SigmoidModule().build(ctx, matrix(linear(ctx, x, w.attn_gate))).tensor;
        auto * q = ggml_reshape_3d(g, linear(ctx, x, w.wq), hd, nh, N);
        auto * k = ggml_reshape_3d(g, linear(ctx, x, w.wk), hd, nkv, N);
        auto * v = ggml_reshape_3d(g, linear(ctx, x, w.wv), hd, nkv, N);
        q = ggml_mul(g, ggml_rms_norm(g, q, hp_.qk_norm_eps), ggml_reshape_3d(g, w.attn_temp.tensor, 1, nh, 1));
        k = ggml_rms_norm(g, k, hp_.qk_norm_eps);
        const modules::RoPEModule rope({hd, GGML_ROPE_TYPE_NORMAL, hp_.rope_freq_base});
        const auto positions = core::wrap_tensor(in.pos, core::TensorShape::from_dims({N}), GGML_TYPE_I32);
        q = rope.build(ctx, core::wrap_tensor(q, core::TensorShape::from_dims({1, N, nh, hd})), positions).tensor;
        k = rope.build(ctx, core::wrap_tensor(k, core::TensorShape::from_dims({1, N, nkv, hd})), positions).tensor;
        auto * kc = keys_[static_cast<size_t>(w.state_index)];
        auto * vc = values_[static_cast<size_t>(w.state_index)];
        const int64_t cache_rows = static_cast<int64_t>(capacity_) * nkv * slots_;
        // The attention views hang off the set_rows results, so ggml orders the reads after the writes.
        auto * kset = ggml_set_rows(g, ggml_reshape_2d(g, kc, hd, cache_rows), ggml_reshape_2d(g, k, hd, nkv * N), in.kv_rows);
        auto * vset = ggml_set_rows(g, ggml_reshape_2d(g, vc, hd, cache_rows), ggml_reshape_2d(g, v, hd, nkv * N), in.kv_rows);
        const size_t offset = static_cast<size_t>(shape.first_slot) * kc->nb[3];
        auto * k4 = ggml_view_4d(g, kset, hd, capacity_, nkv, B, kc->nb[1], kc->nb[2], kc->nb[3], offset);
        auto * v4 = ggml_view_4d(g, vset, hd, capacity_, nkv, B, vc->nb[1], vc->nb[2], vc->nb[3], offset);
        auto * q4 = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, q, hd, nh, T, B), 0, 2, 1, 3));  // [hd, T, nh, B]
        auto * scores = mul_mat_f32(g, k4, q4);                                                     // [cap, T, nh, B]
        scores = ggml_soft_max_ext(g, scores, in.mask, 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);
        auto * vt = ggml_cont(g, ggml_permute(g, v4, 1, 0, 2, 3));                                  // [cap, hd, nkv, B]
        auto * out = mul_mat_f32(g, vt, scores);                                                    // [hd, T, nh, B]
        out = ggml_reshape_3d(g, ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)), hd, nh, N);
        out = ggml_mul(g, out, ggml_reshape_3d(g, gate, 1, nh, N));
        return linear(ctx, ggml_reshape_2d(g, out, hd * nh, N), w.wo);
    }

    ggml_tensor * gated_delta_net(core::ModuleBuildContext & ctx, ggml_cgraph * graph, const LayerWeights & w,
                                  ggml_tensor * x, const GraphShape & shape) const {
        auto * g = ctx.ggml;
        const int H = hp_.gdn_heads;
        const int S = hp_.gdn_head_dim;
        const int I = H * S;
        const int R = hp_.gdn_conv_kernel - 1;
        const int T = shape.rows_per_slot;
        const int B = shape.slots;
        const int N = shape.rows();
        auto * mixed = linear(ctx, x, w.gdn_in);  // [q k v | z | a | b] x N
        auto rows = [&](int first, int count) {
            return ggml_view_2d(g, mixed, count, N, mixed->nb[1], static_cast<size_t>(first) * mixed->nb[0]);
        };
        auto * hs = gdn_history_[static_cast<size_t>(w.state_index)];
        auto * ss = gdn_states_[static_cast<size_t>(w.state_index)];
        auto * history = ggml_view_3d(g, hs, R, 3 * I, B, hs->nb[1], hs->nb[2], static_cast<size_t>(shape.first_slot) * hs->nb[2]);
        auto * state = ggml_view_3d(g, ss, ss->ne[0], 1, B, ss->nb[1], ss->nb[2], static_cast<size_t>(shape.first_slot) * ss->nb[2]);
        auto * time_major = T == 1
            ? ggml_view_3d(g, mixed, 1, 3 * I, B, mixed->nb[0], mixed->nb[1], 0)
            : ggml_reshape_3d(g, ggml_cont(g, ggml_transpose(g, rows(0, 3 * I))), T, 3 * I, 1);
        auto * window = ggml_concat(g, history, time_major, 0);  // [R + T, 3I, B]
        auto * conv = ggml_ssm_conv(g, window, w.gdn_conv.tensor);
        conv = modules::SiluModule().build(
            ctx, core::wrap_tensor(conv, core::TensorShape::from_dims({B, T, 3 * I}))).tensor;
        conv = ggml_reshape_2d(g, conv, 3 * I, N);
        auto heads = [&](int first) {
            return ggml_view_4d(g, conv, S, H, T, B, static_cast<size_t>(S) * conv->nb[0], conv->nb[1],
                                static_cast<size_t>(T) * conv->nb[1], static_cast<size_t>(first) * conv->nb[0]);
        };
        // FLA's q / sqrt(sum q^2 + eps), unlike ggml_l2_norm's max(norm, eps).
        auto l2 = [&](ggml_tensor * t) {
            return ggml_scale(g, ggml_rms_norm(g, t, 1e-6f / static_cast<float>(S)), 1.0f / std::sqrt(static_cast<float>(S)));
        };
        auto * q = l2(heads(0));
        auto * k = l2(heads(I));
        auto * v = heads(2 * I);
        auto * decay = ggml_neg(g, ggml_mul(g, ggml_softplus(g, ggml_add(g, rows(4 * I, H), w.gdn_dt.tensor)),
                                            ggml_exp(g, w.gdn_a_log.tensor)));
        auto * beta = modules::SigmoidModule().build(ctx, matrix(rows(4 * I + H, H))).tensor;
        decay = ggml_reshape_4d(g, decay, 1, H, T, B);
        beta = ggml_reshape_4d(g, beta, 1, H, T, B);
        auto * packed = ggml_gated_delta_net(g, q, k, v, decay, beta, state);  // [I, N + S*B]
        auto * output = ggml_view_2d(g, packed, I, N, static_cast<size_t>(I) * sizeof(float), 0);
        auto * next_state = ggml_view_1d(g, packed, ggml_nelements(state), static_cast<size_t>(I) * N * sizeof(float));
        auto * next_history = ggml_cont(g, ggml_view_3d(g, window, R, 3 * I, B, window->nb[1], window->nb[2],
                                                        static_cast<size_t>(T) * window->nb[0]));
        ggml_build_forward_expand(graph, ggml_cpy(g, next_state, state));
        ggml_build_forward_expand(graph, ggml_cpy(g, next_history, history));
        auto * z = ggml_view_3d(g, mixed, S, H, N, static_cast<size_t>(S) * mixed->nb[0], mixed->nb[1],
                                static_cast<size_t>(3 * I) * mixed->nb[0]);
        auto * normed = ggml_mul(g, ggml_rms_norm(g, ggml_reshape_3d(g, output, S, H, N), 1e-5f), w.gdn_norm.tensor);
        auto * activated_z = modules::SiluModule().build(
            ctx, core::wrap_tensor(z, core::TensorShape::from_dims({N, H, S}))).tensor;
        auto * gated = ggml_mul(g, normed, activated_z);
        return linear(ctx, ggml_reshape_2d(g, gated, I, N), w.wo);
    }

    ggml_tensor * dense_ffn(core::ModuleBuildContext & ctx, const LayerWeights & w, ggml_tensor * x) const {
        modules::GatedFeedForwardConfig config;
        config.hidden_size = hp_.n_embd;
        config.intermediate_size = hp_.n_ff;
        config.use_bias = false;
        config.activation = modules::GatedFeedForwardActivation::Silu;
        config.projection_precision = GGML_PREC_F32;
        return modules::GatedFeedForwardModule(config)
            .build(ctx, matrix(x), {{w.ffn_gate, std::nullopt}, {w.ffn_up, std::nullopt}, {w.ffn_down, std::nullopt}})
            .tensor;
    }

    // Weighted RMSNorm -> router; experts rank logits - router_bias and weigh the softmax of their raw logits. The
    // experts read the latent projection of the FFN input, not the router norm.
    MoeInputs build_moe_inputs(core::ModuleBuildContext & ctx, ggml_tensor * x, const LayerWeights & w) const {
        auto * normed = rms_weighted(ctx, x, w.router_norm, hp_.rms_eps);
        modules::MoeRoutingConfig config;
        config.experts = w.router.tensor->ne[1];
        config.selected = w.top_k;
        config.scoring = modules::MoeScoring::SelectedSoftmax;
        config.normalization = modules::MoeNormalization::None;
        config.precision = GGML_PREC_F32;
        config.contiguous_ids = disk_store_ != nullptr;
        MoeInputs out;
        out.routing = modules::MoeModule::build_routing(ctx, normed, w.router.tensor, w.neg_router_bias.tensor, config);
        out.latent = linear(ctx, x, w.latent_down);
        return out;
    }

    ggml_tensor * moe(core::ModuleBuildContext & ctx, ggml_tensor * x, const LayerWeights & w) const {
        const auto inputs = build_moe_inputs(ctx, x, w);
        auto * bank_ids = disk_store_ ? disk_store_->route(ctx, inputs.routing.ids, w.disk_group) : nullptr;
        auto * y = modules::MoeModule({modules::MoeActivation::SwiGlu}).build(
            ctx, inputs.latent, inputs.routing, {w.up_exps.tensor, w.gate_exps.tensor, w.down_exps.tensor}, bank_ids);
        return linear(ctx, y, w.latent_up);
    }

    // output_norm, the 10 x 4098 head and the logit softcap over the head rows.
    ggml_tensor * head(core::ModuleBuildContext & ctx, ggml_tensor * mixed, int rows, int head_rows) const {
        auto * g = ctx.ggml;
        auto * last = rows == head_rows ? mixed
            : ggml_view_2d(g, mixed, hp_.n_embd, head_rows, mixed->nb[1], static_cast<size_t>(rows - head_rows) * mixed->nb[1]);
        auto * h = rms_weighted(ctx, last, output_norm_, hp_.rms_eps);
        auto * logits = ggml_reshape_3d(g, mul_mat_f32(g, output_.tensor, h), hp_.audio_vocab, hp_.n_codebooks, head_rows);
        if (hp_.logit_softcap > 0) {
            logits = ggml_scale(g, ggml_tanh(g, ggml_scale(g, logits, 1.0f / hp_.logit_softcap)), hp_.logit_softcap);
        }
        return logits;
    }

    void configure_expert_scheduler(core::PreparedGraph & prepared) {
        resources_->configure_scheduler(prepared);
    }

    ForwardGraph & forward_graph(std::vector<std::unique_ptr<ForwardGraph>> & cache, const GraphShape & shape) {
        // Keep the full-chunk graph and at most one shorter (tail) graph per slot.
        for (auto & graph : cache) {
            const auto & s = graph->shape;
            if (s.rows_per_slot == shape.rows_per_slot && s.slots == shape.slots && s.first_slot == shape.first_slot) return *graph;
        }
        if (shape.prefill && shape.rows_per_slot != kPrefillChunk) {
            cache.erase(std::remove_if(cache.begin(), cache.end(), [&](const auto & graph) {
                return graph->shape.first_slot == shape.first_slot && graph->shape.rows_per_slot != kPrefillChunk;
            }), cache.end());
        }
        auto out = std::make_unique<ForwardGraph>();
        out->shape = shape;
        const size_t metadata = ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead();
        out->prepared = std::make_unique<core::PreparedGraph>(execution_, core::PreparedGraphConfig{
            shape.prefill ? "zonos3.prefill" : "zonos3.decode", metadata, kGraphNodes, 64 * ggml_tensor_overhead()});
        out->inputs = make_inputs(*out->prepared, shape);
        out->prepared->allocate_persistent();
        out->prepared->clear_persistent();
        auto & ctx = out->prepared->build_context();
        auto * graph = out->prepared->graph();
        const int N = shape.rows();
        AttnRes bank(*this, ctx, N);
        auto * x = embed(ctx, out->inputs, N);
        for (int layer = 0; layer < hp_.n_layer; ++layer) {
            const auto & w = layers_[static_cast<size_t>(layer)];
            auto * input = rms_weighted(ctx, bank.route(x, 2 * layer), w.attn_norm, hp_.rms_eps);
            auto * mixer = w.gdn ? gated_delta_net(ctx, graph, w, input, shape) : attention(ctx, w, input, out->inputs, shape);
            input = rms_weighted(ctx, bank.route(mixer, 2 * layer + 1), w.ffn_norm, hp_.rms_eps);
            x = w.top_k == 0 ? dense_ffn(ctx, w, input) : moe(ctx, input, w);
        }
        out->logits = head(ctx, bank.route(x, 2 * hp_.n_layer), N, shape.prefill ? 1 : shape.slots);
        ggml_set_output(out->logits);
        ggml_build_forward_expand(graph, out->logits);
        if (disk_store_ || resources_->has_cpu_layers()) configure_expert_scheduler(*out->prepared);
        out->prepared->allocate();
        cache.push_back(std::move(out));
        return *cache.back();
    }

    OffloadGraph & offload_graph(const GraphShape & shape) {
        for (auto & graph : offload_) {
            if (graph->shape.rows_per_slot == shape.rows_per_slot && graph->shape.first_slot == shape.first_slot) return *graph;
        }
        if (shape.rows_per_slot != kPrefillChunk) {
            offload_.erase(std::remove_if(offload_.begin(), offload_.end(), [&](const auto & graph) {
                return graph->shape.first_slot == shape.first_slot && graph->shape.rows_per_slot != kPrefillChunk;
            }), offload_.end());
        }
        offload_.push_back(build_offload(shape));
        return *offload_.back();
    }

    std::unique_ptr<OffloadGraph> build_offload(const GraphShape & shape) {
        auto out = std::make_unique<OffloadGraph>();
        out->shape = shape;
        const int T = shape.rows_per_slot;
        const int Q = queries();
        const int D = hp_.n_embd;
        int64_t bucket = std::min<int64_t>(T, 32);
        while ((bucket & (bucket - 1)) != 0) ++bucket;
        modules::StagedMoeConfig staged;
        staged.hidden_size = latent_;
        staged.tokens = T;
        staged.selected = top_k_;
        staged.experts = hp_.n_expert;
        staged.cache_slots = options_.moe.cache_slots;
        for (int64_t count = 1; count <= bucket; count *= 2) staged.rows_per_expert.push_back(count);
        staged.scatter_index_type = GGML_TYPE_I32;
        out->workspace = std::make_shared<core::GraphWorkspace>(execution_.backend());
        out->staged = std::make_unique<modules::StagedMoeRuntime>(execution_, std::move(staged), out->workspace, resources_->cpu_execution());
        const size_t nodes = 2048;
        const size_t graphs = static_cast<size_t>(2 * hp_.n_layer + 2);
        out->prepared = std::make_unique<core::PreparedGraph>(execution_, core::PreparedGraphConfig{
            "zonos3.offload_prefill", graphs * (ggml_graph_overhead_custom(nodes, false) + 1024 * ggml_tensor_overhead()),
            nodes, 4 * 1024 * 1024});
        out->inputs = make_inputs(*out->prepared, shape);
        auto * persistent = out->prepared->persistent_build_context().ggml;
        auto * sources = ggml_new_tensor_3d(persistent, GGML_TYPE_F32, D, Q, T);  // Full AttnRes sources
        auto * scores = ggml_new_tensor_3d(persistent, GGML_TYPE_F32, Q, Q, T);   // their query scores
        out->prepared->allocate_persistent();
        out->prepared->clear_persistent();
        auto & ctx = out->prepared->build_context();
        auto * g = ctx.ggml;
        AttnRes bank(*this, ctx, T);
        auto stack_view = [&](ggml_tensor * stack, int first, int count) {
            return ggml_view_3d(g, stack, stack->ne[0], count, T, stack->nb[1], stack->nb[2], static_cast<size_t>(first) * stack->nb[1]);
        };
        // Score `source` and copy it and its scores into slot s of the persistent stacks.
        auto append = [&](ggml_cgraph * graph, ggml_tensor * source, int s) {
            auto * sc = bank.score(source);
            ggml_build_forward_expand(graph, ggml_cpy(g, sc, stack_view(scores, s, 1)));
            ggml_build_forward_expand(graph, ggml_cpy(g, ggml_reshape_3d(g, source, D, 1, T), stack_view(sources, s, 1)));
            return sc;
        };
        // Route query qi over the stacked sources 0..n-1.
        auto route_stacked = [&](int n, int qi) {
            if (n == 1) return ggml_view_2d(g, sources, D, T, sources->nb[2], 0);
            return bank.mix(stack_view(scores, 0, n), stack_view(sources, 0, n), n, qi);
        };
        // Route query qi over the stacked sources 0..n-2 and `source` (source n-1, appended in this graph).
        auto route_with = [&](ggml_cgraph * graph, ggml_tensor * source, int n, int qi) {
            auto * sc = append(graph, source, n - 1);
            auto * all_scores = ggml_concat(g, stack_view(scores, 0, n - 1), sc, 1);
            auto * all_sources = ggml_concat(g, stack_view(sources, 0, n - 1), ggml_reshape_3d(g, source, D, 1, T), 1);
            return bank.mix(all_scores, all_sources, n, qi);
        };
        auto * embed_graph = out->prepared->graph(0);
        append(embed_graph, embed(ctx, out->inputs, T), 0);
        for (int layer = 0; layer < hp_.n_layer; ++layer) {
            const auto & w = layers_[static_cast<size_t>(layer)];
            const auto id = out->prepared->add_graph("layer_" + std::to_string(layer), nodes);
            auto * graph = out->prepared->graph(id);
            auto * input = rms_weighted(ctx, route_stacked(2 * layer + 1, 2 * layer), w.attn_norm, hp_.rms_eps);
            auto * mixer = w.gdn ? gated_delta_net(ctx, graph, w, input, shape) : attention(ctx, w, input, out->inputs, shape);
            input = rms_weighted(ctx, route_with(graph, mixer, 2 * layer + 2, 2 * layer + 1), w.ffn_norm, hp_.rms_eps);
            out->layer_graphs.push_back(id);
            if (w.top_k == 0) {
                append(graph, dense_ffn(ctx, w, input), 2 * layer + 2);
                out->finish_graphs.push_back(0);
                out->staged_layers.push_back(-1);
                continue;
            }
            const auto inputs = build_moe_inputs(ctx, input, w);
            out->staged->build_inputs(ctx, graph, inputs.latent, inputs.routing);
            out->staged_layers.push_back(static_cast<int64_t>(out->staged->add_layer(
                modules::MoeModule({modules::MoeActivation::SwiGlu}),
                {w.up_exps.tensor, w.gate_exps.tensor, w.down_exps.tensor}, *disk_store_, w.disk_group)));
            const auto finish = out->prepared->add_graph("finish_" + std::to_string(layer), nodes);
            auto * combined = modules::MoeModule::combine(ctx, out->staged->output(), out->staged->routing_weights());
            append(out->prepared->graph(finish), linear(ctx, combined, w.latent_up), 2 * layer + 2);
            out->finish_graphs.push_back(finish);
        }
        out->final_graph = out->prepared->add_graph("final", nodes);
        out->logits = head(ctx, route_stacked(Q, Q - 1), T, 1);
        ggml_set_output(out->logits);
        ggml_build_forward_expand(out->prepared->graph(out->final_graph), out->logits);
        out->workspace->add(*out->prepared);
        out->workspace->prepare();
        out->staged->prepare();
        return out;
    }

    void set_common_inputs(const GraphInputs & in, const GraphShape & shape, const std::vector<int32_t> & ids,
                           const std::vector<int32_t> & positions) {
        const int T = shape.rows_per_slot;
        const int N = shape.rows();
        const int nkv = hp_.n_head_kv;
        ggml_backend_tensor_set(in.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in.pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        kv_rows_host_.resize(static_cast<size_t>(nkv * N));
        mask_host_.resize(static_cast<size_t>(capacity_) * static_cast<size_t>(N));
        auto & kv_rows = kv_rows_host_;
        auto & mask = mask_host_;
        const auto zero = ggml_fp32_to_fp16(0.0f);
        const auto blocked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        for (int n = 0; n < N; ++n) {
            const int slot = shape.first_slot + n / T;
            const int position = positions[static_cast<size_t>(n)];
            for (int h = 0; h < nkv; ++h) kv_rows[static_cast<size_t>(n * nkv + h)] = (slot * nkv + h) * capacity_ + position;
            auto * row = mask.data() + static_cast<size_t>(n) * static_cast<size_t>(capacity_);
            std::fill(row, row + position + 1, zero);
            std::fill(row + position + 1, row + capacity_, blocked);
        }
        ggml_backend_tensor_set(in.kv_rows, kv_rows.data(), 0, kv_rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    void set_prefill_inputs(const GraphInputs & in, const GraphShape & shape, const Zonos3Prompt & prompt, int position) {
        const int T = shape.rows_per_slot;
        const int width = hp_.n_codebooks + 1;
        std::vector<int32_t> ids(static_cast<size_t>(width * T));
        std::vector<int32_t> positions(static_cast<size_t>(T));
        std::vector<float> keep(static_cast<size_t>(T), 1.0f);
        std::vector<int32_t> sel(static_cast<size_t>(T), 0);
        const auto & c = prompt.cond;
        for (int t = 0; t < T; ++t) {
            const int row = position + t;
            positions[static_cast<size_t>(t)] = row;
            for (int k = 0; k < width; ++k) {
                ids[static_cast<size_t>(k * T + t)] = prompt.ids[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(k)];
            }
            if (c.speaker_row >= 0 && row == c.speaker_row) {
                keep[static_cast<size_t>(t)] = 0.0f;
                sel[static_cast<size_t>(t)] = 1;
            } else if (c.condition_row >= 0 && row >= c.condition_row && row < c.condition_row + hp_.n_cond) {
                keep[static_cast<size_t>(t)] = 0.0f;
                sel[static_cast<size_t>(t)] = 2 + row - c.condition_row;
            }
        }
        set_common_inputs(in, shape, ids, positions);
        ggml_backend_tensor_set(in.keep, keep.data(), 0, keep.size() * sizeof(float));
        ggml_backend_tensor_set(in.sel, sel.data(), 0, sel.size() * sizeof(int32_t));
        std::vector<float> speaker(static_cast<size_t>(hp_.speaker_dim), 0.0f);
        if (c.speaker_row >= 0) speaker = c.speaker;
        ggml_backend_tensor_set(in.speaker, speaker.data(), 0, speaker.size() * sizeof(float));
        std::vector<int32_t> buckets(static_cast<size_t>(hp_.n_cond), 0);
        if (c.condition_row >= 0) buckets = c.buckets;
        ggml_backend_tensor_set(in.buckets, buckets.data(), 0, buckets.size() * sizeof(int32_t));
    }

    void read_logits(ggml_tensor * logits, int rows, std::vector<float> & out) const {
        out.resize(static_cast<size_t>(hp_.audio_vocab) * hp_.n_codebooks * static_cast<size_t>(rows));
        ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
    }

    std::shared_ptr<const Zonos3Assets> assets_;
    const Zonos3Hparams & hp_;
    core::ExecutionContext & execution_;
    Zonos3BackboneOptions options_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::vector<ggml_backend_buffer_type_t> extra_buffer_types_;
    std::vector<std::pair<ggml_backend_buffer_type_t, std::unique_ptr<core::BackendWeightStore>>> extra_stores_;
    std::unique_ptr<core::MoeResources> resources_;
    core::DiskExpertWeightStore * disk_store_ = nullptr;
    std::set<std::string> consumed_;
    std::vector<TensorValue> audio_embd_, cond_embd_;
    TensorValue text_embd_, output_norm_, output_, attnres_q_, speaker_proj_, speaker_bias_;
    std::vector<LayerWeights> layers_;
    int attention_layers_ = 0;
    int gdn_layers_ = 0;
    int top_k_ = 0;
    int latent_ = 0;

    int capacity_ = 0;
    int slots_ = 0;
    std::vector<int> n_past_ = std::vector<int>(2, 0);
    std::vector<int32_t> step_ids_, step_positions_, kv_rows_host_;
    std::vector<ggml_fp16_t> mask_host_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer_{nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> keys_, values_, gdn_states_, gdn_history_;
    std::vector<std::unique_ptr<ForwardGraph>> prefill_, decode_;
    std::vector<std::unique_ptr<OffloadGraph>> offload_;
};

Zonos3Backbone::Zonos3Backbone(std::shared_ptr<const Zonos3Assets> assets, core::ExecutionContext & execution,
                               Zonos3BackboneOptions options)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, options)) {}

Zonos3Backbone::~Zonos3Backbone() = default;

void Zonos3Backbone::begin(int slots, int capacity_rows) { impl_->begin(slots, capacity_rows); }

void Zonos3Backbone::prefill(int slot, const Zonos3Prompt & prompt, std::vector<float> & logits) {
    impl_->prefill(slot, prompt, logits);
}

void Zonos3Backbone::step(const std::array<const int32_t *, 2> & rows, int slots, std::vector<float> & logits) {
    impl_->step(rows, slots, logits);
}

}  // namespace engine::models::zonos3
