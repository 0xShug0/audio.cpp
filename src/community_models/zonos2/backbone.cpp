#include "engine/community_models/zonos2/backbone.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/moe_engine/moe_prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos2 {
namespace {

// Prefill attention scores are [rows, block, n_head] per query block instead of [rows, rows, n_head]: a long-form
// continuation prompt carries earlier chunks' audio rows and reaches thousands of rows.
constexpr int kPrefillQueryBlock = 1024;
constexpr int kGraphNodes = 8192;
constexpr size_t kPrefillGraphCache = 2;

using core::TensorValue;

struct LayerWeights {
    int top_k = 0;  // 0 = dense FFN
    TensorValue attn_norm, ffn_norm;
    TensorValue wq, wk, wv, wo, attn_temp, attn_gate;
    TensorValue ffn_up, ffn_gate, ffn_down;
    moe::LayerId expert;
    TensorValue router_down, router_down_b, router_mlp0, router_mlp0_b, router_mlp2, router_mlp2_b, router_mlp4;
    TensorValue router_norm, router_bias, router_eda_scale;  // router_eda_scale is absent on the first MoE layer
    bool has_eda = false;
};

// Persistent graph inputs; one set per prepared graph.
struct GraphInputs {
    ggml_tensor * ids = nullptr;      // I32 [N, n_codebooks + 1], codebook-major
    ggml_tensor * pos = nullptr;      // I32 [N]
    ggml_tensor * kv_rows = nullptr;  // I32 [N], KV cache rows written by this graph
    ggml_tensor * mask = nullptr;     // F16 [N, N] causal (prefill) or [capacity, 1, 1, 1] (decode)
    ggml_tensor * keep = nullptr;     // F32 [1, N] (prefill)
    ggml_tensor * sel = nullptr;      // I32 [N] (prefill)
    ggml_tensor * speaker = nullptr;  // F32 [spk_dim, 1] (prefill)
    ggml_tensor * zero = nullptr;     // F32 [n_embd, 1] (prefill)
};

}  // namespace

class Zonos2Backbone::Impl {
public:
    Impl(std::shared_ptr<const Zonos2Assets> assets, core::ExecutionContext & execution, Zonos2BackboneOptions options)
        : assets_(std::move(assets)), hp_(assets_->hparams), execution_(execution), options_(options) {
        load_weights();
    }

    ~Impl() {
        decode_.reset();
        prefill_.clear();
        offload_.clear();
    }

    void begin(int capacity_rows) {
        if (capacity_rows <= 0 || capacity_rows > hp_.n_ctx_train) {
            throw std::runtime_error("zonos2: backbone capacity must be 1.." + std::to_string(hp_.n_ctx_train));
        }
        const int capacity = std::min(hp_.n_ctx_train, (capacity_rows + 255) / 256 * 256);
        if (capacity > capacity_) {
            decode_.reset();
            prefill_.clear();
            offload_.clear();
            state_buffer_.reset();
            state_context_.reset();
            capacity_ = capacity;
            allocate_state();
        }
        ggml_backend_buffer_clear(state_buffer_.get(), 0);
        n_past_ = 0;
        experts_->begin_generation();
    }

    void prefill(const Zonos2Prompt & prompt, const float * speaker, std::vector<float> & logits) {
        const int width = hp_.n_codebooks + 1;
        if (prompt.n_rows <= 0 || prompt.ids.size() != static_cast<size_t>(prompt.n_rows) * static_cast<size_t>(width)) {
            throw std::runtime_error("zonos2: malformed prompt");
        }
        if (prompt.n_rows > capacity_) throw std::runtime_error("zonos2: prompt does not fit the backbone capacity");
        if ((prompt.speaker_row >= 0) != (speaker != nullptr) || prompt.speaker_row >= prompt.n_rows) {
            throw std::runtime_error("zonos2: a speaker embedding needs exactly one speaker row in the prompt");
        }
        if (experts_->staged()) {
            auto & graph = offload_graph(prompt.n_rows);
            set_prefill_inputs(graph.inputs, prompt, speaker);
            graph.run();
            read_logits(graph.logits, logits);
        } else {
            auto & graph = prefill_graph(prompt.n_rows);
            set_prefill_inputs(graph.inputs, prompt, speaker);
            graph.prepared->compute();
            read_logits(graph.logits, logits);
        }
        n_past_ = prompt.n_rows;
    }

    void step(const int32_t * row, std::vector<float> & logits) {
        if (n_past_ <= 0) throw std::runtime_error("zonos2: step before prefill");
        if (n_past_ >= capacity_) throw std::runtime_error("zonos2: backbone capacity exceeded");
        auto & graph = decode_graph();
        const std::vector<int32_t> ids(row, row + hp_.n_codebooks + 1);
        set_common_inputs(graph.inputs, ids, {n_past_});
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(capacity_));
        std::fill(mask.begin(), mask.begin() + n_past_ + 1, ggml_fp32_to_fp16(0.0f));
        std::fill(mask.begin() + n_past_ + 1, mask.end(), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        ggml_backend_tensor_set(graph.inputs.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        graph.prepared->compute();
        read_logits(graph.logits, logits);
        ++n_past_;
    }

private:
    struct ForwardGraph {
        std::unique_ptr<moe::MoePreparedGraph> prepared;
        int rows = 0;
        GraphInputs inputs;
        ggml_tensor * logits = nullptr;
    };

    // Expert-cache prefill: the prompt routes more experts than the per-layer cache holds, so every MoE layer runs
    // through the engine between per-layer graphs. The residual stream, the FFN output and the EDA router state
    // cross graph boundaries through persistent tensors.
    struct OffloadGraph {
        int rows = 0;
        GraphInputs inputs;
        std::shared_ptr<core::GraphWorkspace> workspace;
        std::unique_ptr<moe::MoePreparedGraph> prepared;
        moe::MoePreparedGraph::SequenceId sequence{0};
        size_t final_graph = 0;
        ggml_tensor * logits = nullptr;

        void run() { prepared->compute(sequence); }
    };

    static TensorValue matrix(ggml_tensor * x) { return core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[1], x->ne[0]})); }

    ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & weight,
                         const TensorValue * bias = nullptr) const {
        const auto out_features = weight.shape.dims[0];
        const auto in_features = weight.shape.dims[1];
        modules::LinearWeights weights{weight, std::nullopt};
        if (bias) weights.bias = *bias;
        return modules::LinearModule({in_features, out_features, bias != nullptr}).build(ctx, matrix(x), weights).tensor;
    }

    ggml_tensor * rms_weighted(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & weight) const {
        return ggml_mul(ctx.ggml, ggml_rms_norm(ctx.ggml, x, hp_.rms_eps), weight.tensor);
    }

    void load_weights() {
        const auto & source = *assets_->backbone;
        const int D = hp_.n_embd;
        const int64_t q_rows = static_cast<int64_t>(hp_.head_dim) * hp_.n_head;
        const int64_t kv_rows = static_cast<int64_t>(hp_.head_dim) * hp_.n_head_kv;
        if (options_.moe.pin_host_weights && options_.moe.residency != moe::Residency::Staged) {
            throw std::runtime_error("zonos2.pin_experts requires zonos2.disk_experts=true");
        }
        int max_top_k = 0;
        for (const auto used : hp_.expert_used) max_top_k = std::max(max_top_k, static_cast<int>(used));
        if (options_.moe.residency == moe::Residency::Staged) {
            if (max_top_k == 0) throw std::runtime_error("zonos2.disk_experts needs a backbone with MoE layers");
            if (options_.moe.cache_slots < 2 * max_top_k || options_.moe.cache_slots > hp_.n_expert) {
                throw std::runtime_error("zonos2.expert_cache_size must be between " + std::to_string(2 * max_top_k) +
                                         " and " + std::to_string(hp_.n_expert));
            }
        }
        // Every weight lives in the backend's plain buffer, as in Zyphra's runtime: no CPU repacking, so the expert
        // cache runs the same kernels as resident experts.
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos2.backbone.weights", 16 * 1024 * 1024);
        experts_ = std::make_unique<moe::MoeEngine>(execution_, *store_, source, options_.moe, "zonos2.backbone.experts");
        auto native = [&](const std::string & name, std::vector<int64_t> shape) {
            consumed_.insert(name);
            return store_->load_tensor(source, name, assets::TensorStorageType::Native, shape);
        };
        auto f32 = [&](const std::string & name, std::vector<int64_t> shape) {
            consumed_.insert(name);
            return store_->load_f32_tensor(source, name, shape);
        };
        for (int c = 0; c < hp_.n_codebooks; ++c) {
            audio_embd_.push_back(native("audio_embd." + std::to_string(c) + ".weight", {hp_.audio_vocab, D}));
        }
        text_embd_ = native("text_embd.weight", {hp_.text_vocab + 1, D});
        spk_lda_ = native("spk_lda.weight", {hp_.spk_lda_dim, hp_.spk_dim});
        spk_lda_b_ = f32("spk_lda.bias", {hp_.spk_lda_dim});
        spk_proj_ = native("spk_proj.weight", {D, hp_.spk_lda_dim});
        spk_proj_b_ = f32("spk_proj.bias", {D});
        output_norm_ = f32("output_norm.weight", {D});
        output_ = native("output.weight", {static_cast<int64_t>(hp_.audio_vocab) * hp_.n_codebooks, D});
        for (int layer = 0; layer < hp_.n_layer; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            LayerWeights w;
            w.attn_norm = f32(p + "attn_norm.weight", {D});
            w.ffn_norm = f32(p + "ffn_norm.weight", {D});
            w.wq = native(p + "attn_q.weight", {q_rows, D});
            w.wk = native(p + "attn_k.weight", {kv_rows, D});
            w.wv = native(p + "attn_v.weight", {kv_rows, D});
            w.wo = native(p + "attn_output.weight", {D, q_rows});
            w.attn_temp = f32(p + "attn_temp", {hp_.n_head});  // stored as |temp|
            w.attn_gate = native(p + "attn_gate.weight", {hp_.n_head, D});
            w.top_k = hp_.expert_used[static_cast<size_t>(layer)];
            if (w.top_k == 0) {
                w.ffn_up = native(p + "ffn_up.weight", {hp_.n_ff, D});
                w.ffn_gate = native(p + "ffn_gate.weight", {hp_.n_ff, D});
                w.ffn_down = native(p + "ffn_down.weight", {D, hp_.n_ff});
                layers_.push_back(std::move(w));
                continue;
            }
            const auto bank = source.require_metadata(p + "ffn_up_exps.weight").shape;  // {E, n_ff, D}
            if (bank.size() != 3 || bank[0] != hp_.n_expert || bank[2] != D) {
                throw std::runtime_error("zonos2: backbone GGUF: " + p + "ffn_up_exps.weight has an invalid shape");
            }
            const int64_t hidden = bank[1];
            const std::vector<std::string> names{p + "ffn_up_exps.weight", p + "ffn_gate_exps.weight",
                                                 p + "ffn_down_exps.weight"};
            moe::LayerBindings bindings;
            bindings.experts = {
                {names[0], names[0], {hp_.n_expert, hidden, D}, assets::TensorStorageType::Native, moe::ProjectionRole::Up},
                {names[1], names[1], {hp_.n_expert, hidden, D}, assets::TensorStorageType::Native, moe::ProjectionRole::Gate},
                {names[2], names[2], {hp_.n_expert, D, hidden}, assets::TensorStorageType::Native, moe::ProjectionRole::Down}};
            for (const auto & name : names) consumed_.insert(name);
            const auto router_shape = source.require_metadata(p + "router_down.weight").shape;  // {router_dim, D}
            if (router_shape.size() != 2 || router_shape[1] != D || router_shape[0] <= 0) {
                throw std::runtime_error("zonos2: backbone GGUF: " + p + "router_down.weight has an invalid shape");
            }
            const int64_t rd = router_shape[0];
            router_dim_ = static_cast<int>(rd);
            w.router_down = native(p + "router_down.weight", {rd, D});
            w.router_down_b = f32(p + "router_down.bias", {rd});
            w.router_mlp0 = native(p + "router_mlp0.weight", {rd, rd});
            w.router_mlp0_b = f32(p + "router_mlp0.bias", {rd});
            w.router_mlp2 = native(p + "router_mlp2.weight", {rd, rd});
            w.router_mlp2_b = f32(p + "router_mlp2.bias", {rd});
            w.router_mlp4 = native(p + "router_mlp4.weight", {hp_.n_expert, rd});
            w.router_norm = f32(p + "router_norm.weight", {rd});
            w.router_bias = f32(p + "router_bias", {hp_.n_expert});
            w.has_eda = source.has_tensor(p + "router_eda_scale");
            if (w.has_eda) w.router_eda_scale = f32(p + "router_eda_scale", {rd});
            bindings.router = w.router_mlp4; bindings.selection_bias = w.router_bias;
            moe::LayerMath math;
            math.routing.experts = hp_.n_expert; math.routing.selected = w.top_k;
            math.routing.scoring = moe::Scoring::Softmax; math.routing.normalization = moe::Normalization::None;
            math.routing.contiguous_ids = experts_->staged();
            w.expert = experts_->register_layer(std::move(bindings), math);
            layers_.push_back(std::move(w));
        }
        for (const auto & tensor : source.tensors()) {
            if (!consumed_.count(tensor.name)) {
                throw std::runtime_error("zonos2: backbone GGUF: unmapped tensor '" + tensor.name + "'");
            }
        }
        store_->upload();
        experts_->finalize_weights();
        assets_->backbone->release_storage();
        if (experts_->staged()) {
            const auto stats = experts_->statistics();
            debug::timing_log_scalar("zonos2.expert_cache_mb", stats.buffer_bytes / (1024.0 * 1024.0));
            debug::timing_log_scalar("zonos2.host_experts_mb", stats.host_buffer_bytes / (1024.0 * 1024.0));
        }
    }

    void allocate_state() {
        const size_t tensors = static_cast<size_t>(2 * hp_.n_layer + 4);
        state_context_.reset(ggml_init({tensors * ggml_tensor_overhead(), nullptr, true}));
        if (!state_context_) throw std::runtime_error("zonos2: cannot allocate backbone state metadata");
        auto * g = state_context_.get();
        keys_.clear();
        values_.clear();
        for (int i = 0; i < hp_.n_layer; ++i) {
            keys_.push_back(ggml_new_tensor_3d(g, GGML_TYPE_F16, hp_.head_dim, capacity_, hp_.n_head_kv));
            values_.push_back(ggml_new_tensor_3d(g, GGML_TYPE_F16, hp_.head_dim, capacity_, hp_.n_head_kv));
        }
        state_buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution_.backend()));
        if (!state_buffer_) throw std::runtime_error("zonos2: cannot allocate the KV cache");
        debug::timing_log_scalar("zonos2.kv_cache_mb", ggml_backend_buffer_get_size(state_buffer_.get()) / (1024.0 * 1024.0));
    }

    GraphInputs make_inputs(moe::MoePreparedGraph & prepared, int rows, bool prefill) const {
        auto * g = prepared.persistent_build_context().ggml;
        GraphInputs in;
        in.ids = ggml_new_tensor_2d(g, GGML_TYPE_I32, rows, hp_.n_codebooks + 1);
        in.pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, rows);
        in.kv_rows = ggml_new_tensor_1d(g, GGML_TYPE_I32, rows);
        if (prefill) {
            in.mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, rows, rows);
            in.keep = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, rows);
            in.sel = ggml_new_tensor_1d(g, GGML_TYPE_I32, rows);
            in.speaker = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp_.spk_dim, 1);
            in.zero = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp_.n_embd, 1);
        } else {
            in.mask = ggml_new_tensor_4d(g, GGML_TYPE_F16, capacity_, 1, 1, 1);
        }
        for (auto * t : {in.ids, in.pos, in.kv_rows, in.mask, in.keep, in.sel, in.speaker, in.zero}) {
            if (t) ggml_set_input(t);
        }
        return in;
    }

    // Embedding sum, speaker row replacement (prefill), weightless emb_norm.
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
            // keep is 0 at the speaker row and 1 elsewhere; sel picks the projected speaker there and zero elsewhere.
            auto * lda = linear(ctx, in.speaker, spk_lda_, &spk_lda_b_);
            auto * speaker = linear(ctx, lda, spk_proj_, &spk_proj_b_);
            auto * table = ggml_concat(g, in.zero, speaker, 1);
            emb = ggml_add(g, ggml_mul(g, emb, in.keep), ggml_get_rows(g, table, in.sel));
        }
        return ggml_rms_norm(g, emb, hp_.rms_eps);
    }

    // Sigmoid head gate, QK-RMSNorm (q scaled by |temp| per head), interleaved RoPE, causal GQA attention. Prefill
    // attends over the graph's own F32 K/V; decode reads the F16 cache through flash attention. Both write the
    // current rows' K/V into the cache.
    ggml_tensor * attention(core::ModuleBuildContext & ctx, ggml_cgraph * graph, const LayerWeights & w, int layer,
                            ggml_tensor * x, const GraphInputs & in, int rows, bool prefill) const {
        auto * g = ctx.ggml;
        const int hd = hp_.head_dim;
        const int nh = hp_.n_head;
        const int nkv = hp_.n_head_kv;
        const int N = rows;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
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
        auto * kp = ggml_cont(g, ggml_permute(g, k, 0, 2, 1, 3));  // [hd, N, nkv]
        auto * vp = ggml_cont(g, ggml_permute(g, v, 0, 2, 1, 3));
        auto * kc = keys_[static_cast<size_t>(layer)];
        auto * vc = values_[static_cast<size_t>(layer)];
        auto * kset = ggml_set_rows(g, kc, kp, in.kv_rows);  // F32 -> F16 cache rows
        auto * vset = ggml_set_rows(g, vc, vp, in.kv_rows);
        ggml_tensor * out = nullptr;
        if (prefill) {
            ggml_build_forward_expand(graph, kset);
            ggml_build_forward_expand(graph, vset);
            // One block for a prompt of at most kPrefillQueryBlock rows.
            auto * vt = ggml_cont(g, ggml_permute(g, v, 1, 2, 0, 3));  // [N, hd, nkv]
            for (int a = 0; a < N; a += kPrefillQueryBlock) {
                const int n = std::min(kPrefillQueryBlock, N - a);
                auto * qb = q;
                auto * mask = in.mask;
                if (n < N) {
                    qb = ggml_view_3d(g, q, hd, nh, n, q->nb[1], q->nb[2], static_cast<size_t>(a) * q->nb[2]);
                    mask = ggml_cont(g, ggml_view_2d(g, in.mask, N, n, in.mask->nb[1], static_cast<size_t>(a) * in.mask->nb[1]));
                }
                auto * qp = ggml_cont(g, ggml_permute(g, qb, 0, 2, 1, 3));  // [hd, n, nh]
                auto * kq = ggml_mul_mat(g, kp, qp);                         // [N, n, nh]
                kq = ggml_soft_max_ext(g, kq, mask, scale, 0.0f);
                auto * block = ggml_mul_mat(g, vt, kq);                      // [hd, n, nh]
                out = out ? ggml_concat(g, out, block, 1) : block;
            }
            out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3));  // [hd, nh, N]
        } else {
            // The attention views hang off the set_rows results, so ggml orders the reads after the writes.
            auto * k4 = ggml_view_4d(g, kset, hd, capacity_, nkv, 1, kc->nb[1], kc->nb[2], kc->nb[3], 0);
            auto * v4 = ggml_view_4d(g, vset, hd, capacity_, nkv, 1, vc->nb[1], vc->nb[2], vc->nb[3], 0);
            auto * q4 = ggml_reshape_4d(g, q, hd, 1, nh, 1);
            out = ggml_flash_attn_ext(g, q4, k4, v4, in.mask, scale, 0.0f, 0.0f);  // [hd, nh, 1, 1]
            out = ggml_reshape_3d(g, out, hd, nh, N);
        }
        out = ggml_mul(g, out, ggml_reshape_3d(g, gate, 1, nh, N));
        return linear(ctx, ggml_reshape_2d(g, out, hd * nh, N), w.wo);
    }

    ggml_tensor * dense_ffn(core::ModuleBuildContext & ctx, const LayerWeights & w, ggml_tensor * x) const {
        modules::GatedFeedForwardConfig config;
        config.hidden_size = hp_.n_embd;
        config.intermediate_size = hp_.n_ff;
        config.use_bias = false;
        config.activation = modules::GatedFeedForwardActivation::Silu;
        return modules::GatedFeedForwardModule(config)
            .build(ctx, matrix(x), {{w.ffn_gate, std::nullopt}, {w.ffn_up, std::nullopt}, {w.ffn_down, std::nullopt}})
            .tensor;
    }

    // The EDA router: down projection (+ the previous MoE layer's state * eda_scale), which becomes the next state,
    // then RMSNorm and two erf-GELU layers. The framework routing tail ranks softmax(router_mlp4 . h) + router_bias
    // and weighs the selected experts by their softmax probabilities, unnormalized.
    moe::Routing route(core::ModuleBuildContext & ctx, const LayerWeights & w, ggml_tensor * x,
                              ggml_tensor *& router_state) const {
        auto * g = ctx.ggml;
        auto * rh = linear(ctx, x, w.router_down, &w.router_down_b);
        if (w.has_eda && router_state) rh = ggml_add(g, rh, ggml_mul(g, router_state, w.router_eda_scale.tensor));
        router_state = rh;
        auto * h = rms_weighted(ctx, rh, w.router_norm);
        h = ggml_gelu_erf(g, linear(ctx, h, w.router_mlp0, &w.router_mlp0_b));
        h = ggml_gelu_erf(g, linear(ctx, h, w.router_mlp2, &w.router_mlp2_b));
        return experts_->route(ctx, w.expert, h);
    }

    ggml_tensor * moe(moe::MoePreparedGraph & graph, core::ModuleBuildContext & ctx,
        ggml_tensor * x, const LayerWeights & w, ggml_tensor *& router_state) const {
        const auto routing = route(ctx, w, x, router_state);
        return experts_->build(graph, w.expert, x, &routing);
    }

    // output_norm, the 9 x 1026 head and the logit softcap over the last row.
    ggml_tensor * head(core::ModuleBuildContext & ctx, ggml_tensor * residual, int rows) const {
        auto * g = ctx.ggml;
        auto * last = rows == 1 ? residual
            : ggml_view_2d(g, residual, hp_.n_embd, 1, residual->nb[1], static_cast<size_t>(rows - 1) * residual->nb[1]);
        auto * h = rms_weighted(ctx, last, output_norm_);
        auto * logits = ggml_reshape_3d(g, ggml_mul_mat(g, output_.tensor, h), hp_.audio_vocab, hp_.n_codebooks, 1);
        if (hp_.logit_softcap > 0) {
            logits = ggml_scale(g, ggml_tanh(g, ggml_scale(g, logits, 1.0f / hp_.logit_softcap)), hp_.logit_softcap);
        }
        return logits;
    }

    std::unique_ptr<ForwardGraph> build_forward(int rows, bool prefill) {
        auto out = std::make_unique<ForwardGraph>();
        out->rows = rows;
        const size_t metadata = ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead();
        out->prepared = std::make_unique<moe::MoePreparedGraph>(execution_, core::PreparedGraphConfig{
            prefill ? "zonos2.prefill" : "zonos2.decode", metadata, kGraphNodes, 64 * ggml_tensor_overhead()});
        out->inputs = make_inputs(*out->prepared, rows, prefill);
        out->prepared->allocate_persistent();
        out->prepared->clear_persistent();
        auto & ctx = out->prepared->build_context();
        auto * graph = out->prepared->graph();
        auto * g = ctx.ggml;
        auto * x = embed(ctx, out->inputs, rows);
        ggml_tensor * residual = nullptr;
        ggml_tensor * router_state = nullptr;
        for (int layer = 0; layer < hp_.n_layer; ++layer) {
            const auto & w = layers_[static_cast<size_t>(layer)];
            residual = residual ? ggml_add(g, x, residual) : x;
            auto * attn = attention(ctx, graph, w, layer, rms_weighted(ctx, residual, w.attn_norm), out->inputs, rows, prefill);
            residual = ggml_add(g, attn, residual);
            auto * input = rms_weighted(ctx, residual, w.ffn_norm);
            x = w.top_k == 0 ? dense_ffn(ctx, w, input) : moe(*out->prepared, ctx, input, w, router_state);
        }
        out->logits = head(ctx, ggml_add(g, x, residual), rows);
        ggml_set_output(out->logits);
        ggml_build_forward_expand(graph, out->logits);
        out->prepared->allocate();
        return out;
    }

    ForwardGraph & decode_graph() {
        if (!decode_) decode_ = build_forward(1, false);
        return *decode_;
    }

    ForwardGraph & prefill_graph(int rows) {
        for (auto & graph : prefill_) {
            if (graph->rows == rows) return *graph;
        }
        if (prefill_.size() >= kPrefillGraphCache) prefill_.erase(prefill_.begin());
        prefill_.push_back(build_forward(rows, true));
        return *prefill_.back();
    }

    OffloadGraph & offload_graph(int rows) {
        for (auto & graph : offload_) {
            if (graph->rows == rows) return *graph;
        }
        // One offload graph holds its own staged expert buffers; keep only the latest prompt length.
        offload_.clear();
        offload_.push_back(build_offload(rows));
        return *offload_.back();
    }

    std::unique_ptr<OffloadGraph> build_offload(int T) {
        auto out = std::make_unique<OffloadGraph>();
        out->rows = T;
        const int D = hp_.n_embd;
        out->workspace = std::make_shared<core::GraphWorkspace>(execution_.backend());
        const size_t nodes = 2048;
        const size_t graphs = static_cast<size_t>(2 * hp_.n_layer + 2);
        out->prepared = std::make_unique<moe::MoePreparedGraph>(execution_, core::PreparedGraphConfig{
            "zonos2.offload_prefill", graphs * (ggml_graph_overhead_custom(nodes, false) + 1024 * ggml_tensor_overhead()),
            nodes, 4 * 1024 * 1024}, out->workspace);
        out->inputs = make_inputs(*out->prepared, T, true);
        auto * persistent = out->prepared->persistent_build_context().ggml;
        auto * residual_state = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, D, T);  // residual stream
        auto * ffn_state = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, D, T);       // last FFN output
        auto * router_state = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, std::max(router_dim_, 1), T);
        out->prepared->allocate_persistent();
        out->prepared->clear_persistent();
        auto & ctx = out->prepared->build_context();
        auto * g = ctx.ggml;
        ggml_build_forward_expand(out->prepared->graph(0), ggml_cpy(g, embed(ctx, out->inputs, T), ffn_state));
        std::vector<moe::MoePreparedGraph::Step> sequence{size_t{0}};
        bool has_router_state = false;
        for (int layer = 0; layer < hp_.n_layer; ++layer) {
            const auto & w = layers_[static_cast<size_t>(layer)];
            const auto id = out->prepared->add_graph("layer_" + std::to_string(layer), nodes);
            auto * graph = out->prepared->graph(id);
            auto * residual = layer == 0 ? ffn_state : ggml_add(g, ffn_state, residual_state);
            auto * attn = attention(ctx, graph, w, layer, rms_weighted(ctx, residual, w.attn_norm), out->inputs, T, true);
            residual = ggml_add(g, attn, residual);
            auto * input = rms_weighted(ctx, residual, w.ffn_norm);
            if (w.top_k == 0) {
                ggml_build_forward_expand(graph, ggml_cpy(g, dense_ffn(ctx, w, input), ffn_state));
                ggml_build_forward_expand(graph, ggml_cpy(g, residual, residual_state));
                sequence.emplace_back(id);
                continue;
            }
            ggml_tensor * state = has_router_state ? router_state : nullptr;
            const auto routing = route(ctx, w, input, state);
            has_router_state = true;
            ggml_build_forward_expand(graph, ggml_cpy(g, state, router_state));
            const auto call = experts_->add_call(w.expert, {T, D, D, 32, GGML_TYPE_I32}, out->workspace);
            experts_->build_inputs(call, *out->prepared, id, input, &routing);
            ggml_build_forward_expand(graph, ggml_cpy(g, residual, residual_state));
            const auto finish = out->prepared->add_graph("finish_" + std::to_string(layer), nodes);
            auto * combined = experts_->build_result(call, *out->prepared);
            ggml_build_forward_expand(out->prepared->graph(finish), ggml_cpy(g, combined, ffn_state));
            sequence.emplace_back(id); sequence.emplace_back(call); sequence.emplace_back(finish);
        }
        out->final_graph = out->prepared->add_graph("final", nodes);
        out->logits = head(ctx, ggml_add(g, ffn_state, residual_state), T);
        ggml_set_output(out->logits);
        ggml_build_forward_expand(out->prepared->graph(out->final_graph), out->logits);
        sequence.emplace_back(out->final_graph);
        out->sequence = out->prepared->add_sequence(std::move(sequence));
        out->prepared->allocate();
        return out;
    }

    void set_common_inputs(const GraphInputs & in, const std::vector<int32_t> & ids, const std::vector<int32_t> & positions) const {
        ggml_backend_tensor_set(in.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in.pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in.kv_rows, positions.data(), 0, positions.size() * sizeof(int32_t));
    }

    void set_prefill_inputs(const GraphInputs & in, const Zonos2Prompt & prompt, const float * speaker) const {
        const int T = prompt.n_rows;
        const int width = hp_.n_codebooks + 1;
        std::vector<int32_t> ids(static_cast<size_t>(width) * static_cast<size_t>(T));
        std::vector<int32_t> positions(static_cast<size_t>(T));
        std::vector<float> keep(static_cast<size_t>(T), 1.0f);
        std::vector<int32_t> sel(static_cast<size_t>(T), 0);
        for (int t = 0; t < T; ++t) {
            positions[static_cast<size_t>(t)] = t;
            for (int c = 0; c < width; ++c) {
                ids[static_cast<size_t>(c) * static_cast<size_t>(T) + static_cast<size_t>(t)] =
                    prompt.ids[static_cast<size_t>(t) * static_cast<size_t>(width) + static_cast<size_t>(c)];
            }
        }
        if (prompt.speaker_row >= 0) {
            keep[static_cast<size_t>(prompt.speaker_row)] = 0.0f;
            sel[static_cast<size_t>(prompt.speaker_row)] = 1;
        }
        set_common_inputs(in, ids, positions);
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(T) * static_cast<size_t>(T));
        const auto zero = ggml_fp32_to_fp16(0.0f);
        const auto blocked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        for (int q = 0; q < T; ++q) {
            for (int kv = 0; kv < T; ++kv) mask[static_cast<size_t>(q) * static_cast<size_t>(T) + static_cast<size_t>(kv)] = kv <= q ? zero : blocked;
        }
        ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(in.keep, keep.data(), 0, keep.size() * sizeof(float));
        ggml_backend_tensor_set(in.sel, sel.data(), 0, sel.size() * sizeof(int32_t));
        std::vector<float> spk(static_cast<size_t>(hp_.spk_dim), 0.0f);
        if (speaker) std::copy(speaker, speaker + hp_.spk_dim, spk.begin());
        ggml_backend_tensor_set(in.speaker, spk.data(), 0, spk.size() * sizeof(float));
        const std::vector<float> zeros(static_cast<size_t>(hp_.n_embd), 0.0f);
        ggml_backend_tensor_set(in.zero, zeros.data(), 0, zeros.size() * sizeof(float));
    }

    void read_logits(ggml_tensor * logits, std::vector<float> & out) const {
        out.resize(static_cast<size_t>(hp_.audio_vocab) * static_cast<size_t>(hp_.n_codebooks));
        ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
    }

    std::shared_ptr<const Zonos2Assets> assets_;
    const Zonos2Hparams & hp_;
    core::ExecutionContext & execution_;
    Zonos2BackboneOptions options_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::unique_ptr<moe::MoeEngine> experts_;
    std::set<std::string> consumed_;
    std::vector<TensorValue> audio_embd_;
    TensorValue text_embd_, spk_lda_, spk_lda_b_, spk_proj_, spk_proj_b_, output_norm_, output_;
    std::vector<LayerWeights> layers_;
    int router_dim_ = 0;

    int capacity_ = 0;
    int n_past_ = 0;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer_{nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> keys_, values_;
    std::vector<std::unique_ptr<ForwardGraph>> prefill_;
    std::unique_ptr<ForwardGraph> decode_;
    std::vector<std::unique_ptr<OffloadGraph>> offload_;
};

Zonos2Backbone::Zonos2Backbone(std::shared_ptr<const Zonos2Assets> assets, core::ExecutionContext & execution,
                               Zonos2BackboneOptions options)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, options)) {}

Zonos2Backbone::~Zonos2Backbone() = default;

void Zonos2Backbone::begin(int capacity_rows) { impl_->begin(capacity_rows); }

void Zonos2Backbone::prefill(const Zonos2Prompt & prompt, const float * speaker, std::vector<float> & logits) {
    impl_->prefill(prompt, speaker, logits);
}

void Zonos2Backbone::step(const int32_t * row, std::vector<float> & logits) { impl_->step(row, logits); }

}  // namespace engine::models::zonos2
