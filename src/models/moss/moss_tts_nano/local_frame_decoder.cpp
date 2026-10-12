#include "engine/models/moss/moss_tts_nano/local_frame_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_nano {
namespace {

namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TransformerLayerWeights {
    core::TensorValue ln1_weight;
    core::TensorValue ln1_bias;
    modules::LinearWeights c_attn;
    modules::LinearWeights c_proj;
    core::TensorValue ln2_weight;
    core::TensorValue ln2_bias;
    modules::LinearWeights fc_in;
    modules::LinearWeights fc_out;
};

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

// Attention over a cached K/V prefix. The caller passes a view of the rows that are actually
// visible, so no attention mask (and no ggml_diag_mask_inf) is needed: a single query row sees
// exactly the slots it is allowed to see.
core::TensorValue attention_from_cache_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t head_dim) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

// Single row through the local transformer: this row's K/V is appended to the frame cache at slot
// `slot` and the row then attends to the cache prefix [0, visible_steps).
core::TensorValue cached_local_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const MossTTSNanoLocalTransformerConfig & config,
    const core::TensorValue & cache_keys,
    const core::TensorValue & cache_values,
    const core::TensorValue & cache_slot,
    int64_t visible_steps,
    const modules::FastKVSetRowsModule & set_rows) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    const modules::AddModule add;
    const modules::LayerNormModule norm({config.hidden_size, config.layer_norm_epsilon, true, true});

    auto x = norm.build(ctx, input, {weights.ln1_weight, weights.ln1_bias});
    auto qkv = modules::LinearModule({config.hidden_size, config.hidden_size * 3, true})
                   .build(ctx, x, {weights.c_attn.weight, weights.c_attn.bias});
    auto q = modules::SliceModule({2, 0, config.hidden_size}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.hidden_size, config.hidden_size}).build(ctx, qkv);
    auto v = modules::SliceModule({2, config.hidden_size * 2, config.hidden_size}).build(ctx, qkv);
    q = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_base})
            .build(ctx, reshape_heads(ctx, q, config.num_attention_heads, head_dim), positions);
    k = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_base})
            .build(ctx, reshape_heads(ctx, k, config.num_attention_heads, head_dim), positions);
    v = reshape_heads(ctx, v, config.num_attention_heads, head_dim);

    const auto k_row = core::ensure_backend_addressable_layout(ctx, k);
    const auto v_row = core::ensure_backend_addressable_layout(ctx, v);
    const auto k_cache = set_rows.build(ctx, cache_keys, k_row, cache_slot);
    const auto v_cache = set_rows.build(ctx, cache_values, v_row, cache_slot);
    const auto k_used = engine::runtime::view_transformer_kv_cache_steps(
        ctx, k_cache, 0, visible_steps, config.num_attention_heads, head_dim, "moss_nano.local.k");
    const auto v_used = engine::runtime::view_transformer_kv_cache_steps(
        ctx, v_cache, 0, visible_steps, config.num_attention_heads, head_dim, "moss_nano.local.v");

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k_used.shape.rank}).build(ctx, k_used);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v_used.shape.rank}).build(ctx, v_used);
    auto context = attention_from_cache_heads(ctx, q_heads, k_heads, v_heads, head_dim);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    x = add.build(
        ctx,
        input,
        modules::LinearModule({config.hidden_size, config.hidden_size, true})
            .build(ctx, context, {weights.c_proj.weight, weights.c_proj.bias}));
    auto ff = norm.build(ctx, x, {weights.ln2_weight, weights.ln2_bias});
    ff = modules::LinearModule({config.hidden_size, config.intermediate_size, true})
             .build(ctx, ff, {weights.fc_in.weight, weights.fc_in.bias});
    ff = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ff);
    ff = modules::LinearModule({config.intermediate_size, config.hidden_size, true})
             .build(ctx, ff, {weights.fc_out.weight, weights.fc_out.bias});
    return add.build(ctx, x, ff);
}

}  // namespace

struct MossTTSNanoLocalFrameDecoderRuntime::Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue text_lm_head;
    std::vector<core::TensorValue> audio_embeddings;
    std::vector<core::TensorValue> audio_lm_heads;
    std::vector<TransformerLayerWeights> layers;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
};

// One frame = a chain of single-row forwards over a shared 18-slot K/V cache:
//
//   row 0  : the global transformer's hidden vector (gate: "speak" vs end-of-audio)
//   row 1  : embedding of the assistant slot token           -> codebook 0 logits
//   row k  : embedding of the token sampled for codebook k-2 -> codebook k-1 logits
//
// Positions and cache slots are compile-time constants per row kind, so every row only has to
// upload its 4-byte token id (or the hidden vector for the gate).
struct MossTTSNanoLocalFrameDecoderRuntime::FrameGraphs {
    struct Row {
        ggml_cgraph * graph = nullptr;
        ggml_tensor * vector_in = nullptr;  // row 0 only
        ggml_tensor * id_in = nullptr;      // rows >= 1
        ggml_tensor * positions = nullptr;  // constant per row kind
        ggml_tensor * cache_slot = nullptr; // constant per row kind
        ggml_tensor * logits = nullptr;
        std::vector<float> logits_values;
    };

    FrameGraphs(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        const Weights & weights,
        core::ExecutionContext & execution,
        size_t arena_bytes,
        int64_t active_codebooks)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          active_codebooks_(active_codebooks) {
        if (assets_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("MOSS local frame graphs require assets and a backend");
        }
        if (active_codebooks_ <= 0 || active_codebooks_ > assets_->config.n_vq) {
            throw std::runtime_error("MOSS local frame graphs active codebooks are out of range");
        }
        const auto & config = assets_->config.local_transformer;
        const int64_t hidden_size = config.hidden_size;
        const int64_t head_dim = hidden_size / config.num_attention_heads;
        cache_steps_ = assets_->config.n_vq + 2;

        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MOSS local frame graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "moss_tts_nano.local.frame", backend_type_};

        // The shared per-frame cache is created first so that alloc_ctx_tensors assigns its buffer
        // before any view of it is initialised.
        for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
            (void) layer;
            auto keys = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.num_attention_heads, head_dim}));
            auto values = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.num_attention_heads, head_dim}));
            if (keys.tensor == nullptr || values.tensor == nullptr) {
                throw std::runtime_error("failed to allocate MOSS local frame K/V cache tensors");
            }
            cache_keys_.push_back(keys);
            cache_values_.push_back(values);
        }

        rows_.reserve(static_cast<size_t>(active_codebooks_ + 1));
        for (int64_t k = 0; k <= active_codebooks_; ++k) {
            rows_.push_back(build_row(ctx, weights, k));
        }

        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MOSS local frame graphs");
        }

        // Row kinds are fixed, so positions/slots are written once at build time.
        for (size_t k = 0; k < rows_.size(); ++k) {
            const int32_t slot = static_cast<int32_t>(k);
            ggml_backend_tensor_set(rows_[k].positions, &slot, 0, sizeof(int32_t));
            ggml_backend_tensor_set(rows_[k].cache_slot, &slot, 0, sizeof(int32_t));
        }
        clear_cache();
    }

    ~FrameGraphs() {
        for (auto & row : rows_) {
            engine::core::release_backend_graph_resources(backend_, row.graph);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        ctx_.reset();
    }

    FrameGraphs(const FrameGraphs &) = delete;
    FrameGraphs & operator=(const FrameGraphs &) = delete;

    // Slots never written hold uninitialised memory; a NaN pattern would survive the softmax
    // (NaN + -inf stays NaN), so the cache must be zeroed before the first frame.
    void clear_cache() {
        for (auto & keys : cache_keys_) {
            ggml_backend_tensor_memset(keys.tensor, 0, 0, ggml_nbytes(keys.tensor));
        }
        for (auto & values : cache_values_) {
            ggml_backend_tensor_memset(values.tensor, 0, 0, ggml_nbytes(values.tensor));
        }
    }

    int64_t active_codebooks() const noexcept {
        return active_codebooks_;
    }

    const std::vector<float> & run_gate(const std::vector<float> & global_hidden) {
        if (global_hidden.size() != rows_[0].logits_values.capacity() && global_hidden.size() == 0) {
            throw std::runtime_error("MOSS local frame gate requires a hidden vector");
        }
        Row & row = rows_[0];
        ggml_backend_tensor_set(row.vector_in, global_hidden.data(), 0, global_hidden.size() * sizeof(float));
        return run_row_graph(row);
    }

    const std::vector<float> & run_row(int64_t k, int32_t token) {
        if (k <= 0 || k >= static_cast<int64_t>(rows_.size())) {
            throw std::runtime_error("MOSS local frame row index is out of range");
        }
        Row & row = rows_[static_cast<size_t>(k)];
        const int64_t table_rows = embedding_rows_for_row(k);
        if (token < 0 || token >= table_rows) {
            throw std::runtime_error("MOSS local frame token is outside its embedding table");
        }
        ggml_backend_tensor_set(row.id_in, &token, 0, sizeof(int32_t));
        return run_row_graph(row);
    }

private:
    int64_t embedding_rows_for_row(int64_t k) const {
        if (k == 1) {
            return assets_->config.global_transformer.vocab_size;
        }
        return assets_->config.audio_codebook_sizes.at(static_cast<size_t>(k - 2));
    }

    const std::vector<float> & run_row_graph(Row & row) {
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = ggml_backend_graph_compute(backend_, row.graph);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS local frame graph compute failed");
        }
        core::read_tensor_f32_into(row.logits, row.logits_values);
        return row.logits_values;
    }

    Row build_row(
        core::ModuleBuildContext & ctx,
        const Weights & weights,
        int64_t k) {
        const auto & config = assets_->config.local_transformer;
        const int64_t hidden_size = config.hidden_size;
        const int64_t head_dim = hidden_size / config.num_attention_heads;
        Row row;
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.ggml, 4096, false);

        core::TensorValue x;
        if (k == 0) {
            auto vector = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({hidden_size}));
            ggml_set_input(vector.tensor);
            row.vector_in = vector.tensor;
            x = core::reshape_tensor(ctx, vector, core::TensorShape::from_dims({1, 1, hidden_size}));
        } else {
            auto id = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
            ggml_set_input(id.tensor);
            row.id_in = id.tensor;
            const bool is_text_row = k == 1;
            const int64_t table_rows = is_text_row
                                           ? assets_->config.global_transformer.vocab_size
                                           : assets_->config.audio_codebook_sizes.at(static_cast<size_t>(k - 2));
            const auto & table = is_text_row
                                     ? weights.text_embedding
                                     : weights.audio_embeddings.at(static_cast<size_t>(k - 2));
            auto embedding = modules::EmbeddingModule({table_rows, hidden_size}).build(ctx, id, table);
            x = core::reshape_tensor(ctx, embedding, core::TensorShape::from_dims({1, 1, hidden_size}));
        }

        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        row.positions = positions.tensor;
        auto cache_slot = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        row.cache_slot = cache_slot.tensor;

        const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::Exact});
        for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
            x = cached_local_layer(
                ctx,
                x,
                positions,
                weights.layers[layer],
                config,
                cache_keys_[layer],
                cache_values_[layer],
                cache_slot,
                k + 1,
                set_rows);
        }
        x = modules::LayerNormModule({hidden_size, config.layer_norm_epsilon, true, true})
                .build(ctx, x, {weights.final_norm_weight, weights.final_norm_bias});
        if (k == 0) {
            auto logits = modules::LinearModule({hidden_size, assets_->config.global_transformer.vocab_size, false})
                              .build(ctx, x, {weights.text_lm_head, std::nullopt});
            row.logits = core::reshape_tensor(
                             ctx,
                             logits,
                             core::TensorShape::from_dims({assets_->config.global_transformer.vocab_size}))
                             .tensor;
            row.logits_values.resize(static_cast<size_t>(assets_->config.global_transformer.vocab_size));
        } else {
            const size_t codebook = static_cast<size_t>(k - 1);
            const int64_t codebook_size = assets_->config.audio_codebook_sizes.at(codebook);
            auto logits = modules::LinearModule({hidden_size, codebook_size, false})
                              .build(ctx, x, {weights.audio_lm_heads.at(codebook), std::nullopt});
            row.logits = core::reshape_tensor(ctx, logits, core::TensorShape::from_dims({codebook_size})).tensor;
            row.logits_values.resize(static_cast<size_t>(codebook_size));
        }
        ggml_set_output(row.logits);
        ggml_build_forward_expand(graph, row.logits);
        row.graph = graph;
        return row;
    }

    std::shared_ptr<const MossTTSNanoAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t active_codebooks_ = 0;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<core::TensorValue> cache_keys_;
    std::vector<core::TensorValue> cache_values_;
    std::vector<Row> rows_;
};

MossTTSNanoLocalFrameDecoderRuntime::MossTTSNanoLocalFrameDecoderRuntime(
    std::shared_ptr<const MossTTSNanoAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      execution_context_(execution_context),
      sampling_policy_(
          execution_context.backend_type() == core::BackendType::Cuda
              ? engine::sampling::resolve_torch_cuda_sampling_policy(
                    execution_context.backend_type(),
                    execution_context.config().device,
                    "moss_tts_nano.local_frame.cuda_sampling_policy",
                    "MOSS-TTS-Nano",
                    engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda)
              : engine::sampling::TorchCudaSamplingPolicy{}),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS local frame decoder requires assets");
    }
    const auto & source = *assets_->model_weights;
    const auto & config = assets_->config.local_transformer;
    const auto weight_storage =
        resolve_transformer_weight_storage(execution_context_.backend_type(), source, "local_transformer.h.0.attn.c_attn.weight", weight_storage_type);
    weights_ = std::make_shared<Weights>();
    weights_->store = std::make_shared<core::BackendWeightStore>(
        execution_context_.backend(),
        execution_context_.backend_type(),
        "moss_tts_nano.local.weights",
        weight_context_bytes);
    weights_->text_embedding = weights_->store->load_tensor(
        source,
        "transformer.wte.weight",
        weight_storage,
        {assets_->config.global_transformer.vocab_size, config.hidden_size});
    weights_->text_lm_head = weights_->store->load_tensor(
        source,
        "text_lm_head.weight",
        weight_storage,
        {assets_->config.global_transformer.vocab_size, config.hidden_size});
    weights_->audio_embeddings.reserve(static_cast<size_t>(assets_->config.n_vq));
    weights_->audio_lm_heads.reserve(static_cast<size_t>(assets_->config.n_vq));
    for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
        weights_->audio_embeddings.push_back(weights_->store->load_tensor(
            source,
            "audio_embeddings." + std::to_string(q) + ".weight",
            weight_storage,
            {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size}));
        weights_->audio_lm_heads.push_back(weights_->store->load_tensor(
            source,
            "audio_lm_heads." + std::to_string(q) + ".weight",
            weight_storage,
            {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size}));
    }
    weights_->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "local_transformer.h." + std::to_string(layer);
        TransformerLayerWeights w;
        w.ln1_weight = weights_->store->load_f32_tensor(source, prefix + ".ln_1.weight", {config.hidden_size});
        w.ln1_bias = weights_->store->load_f32_tensor(source, prefix + ".ln_1.bias", {config.hidden_size});
        w.c_attn = modules::binding::linear_from_source(*weights_->store, source, prefix + ".attn.c_attn", weight_storage, config.hidden_size * 3, config.hidden_size, true);
        w.c_proj = modules::binding::linear_from_source(*weights_->store, source, prefix + ".attn.c_proj", weight_storage, config.hidden_size, config.hidden_size, true);
        w.ln2_weight = weights_->store->load_f32_tensor(source, prefix + ".ln_2.weight", {config.hidden_size});
        w.ln2_bias = weights_->store->load_f32_tensor(source, prefix + ".ln_2.bias", {config.hidden_size});
        w.fc_in = modules::binding::linear_from_source(*weights_->store, source, prefix + ".mlp.fc_in", weight_storage, config.intermediate_size, config.hidden_size, true);
        w.fc_out = modules::binding::linear_from_source(*weights_->store, source, prefix + ".mlp.fc_out", weight_storage, config.hidden_size, config.intermediate_size, true);
        weights_->layers.push_back(std::move(w));
    }
    weights_->final_norm_weight = weights_->store->load_f32_tensor(source, "local_transformer.ln_f.weight", {config.hidden_size});
    weights_->final_norm_bias = weights_->store->load_f32_tensor(source, "local_transformer.ln_f.bias", {config.hidden_size});
    weights_->store->upload();
}

MossTTSNanoLocalFrameDecoderRuntime::~MossTTSNanoLocalFrameDecoderRuntime() = default;

void MossTTSNanoLocalFrameDecoderRuntime::prepare(int64_t active_codebooks) {
    if (active_codebooks <= 0 || active_codebooks > assets_->config.n_vq) {
        throw std::runtime_error("MOSS local frame decoder active codebooks are out of range");
    }
    if (frame_graphs_ == nullptr || frame_graphs_->active_codebooks() != active_codebooks) {
        frame_graphs_ = std::make_unique<FrameGraphs>(
            assets_,
            *weights_,
            execution_context_,
            graph_arena_bytes_,
            active_codebooks);
    }
}

int32_t MossTTSNanoLocalFrameDecoderRuntime::assistant_slot_token_id() const noexcept {
    return static_cast<int32_t>(assets_->config.audio_assistant_slot_token_id);
}

int32_t MossTTSNanoLocalFrameDecoderRuntime::audio_pad_token_id() const noexcept {
    return static_cast<int32_t>(assets_->config.audio_pad_token_id);
}

std::vector<int32_t> MossTTSNanoLocalFrameDecoderRuntime::generate_frame(
    const std::vector<float> & global_hidden,
    int64_t active_codebooks,
    const std::vector<int32_t> & history,
    const MossTTSNanoSamplingOptions & sampling,
    std::mt19937 & rng,
    uint64_t seed,
    uint64_t & sample_call_index) {
    if (active_codebooks <= 0 || active_codebooks > assets_->config.n_vq) {
        throw std::runtime_error("MOSS local frame decoder active codebooks are out of range");
    }
    if (frame_graphs_ == nullptr || frame_graphs_->active_codebooks() != active_codebooks) {
        prepare(active_codebooks);
    }
    const auto & text_logits = frame_graphs_->run_gate(global_hidden);
    const std::array<int32_t, 2> text_candidates = {
        static_cast<int32_t>(assets_->config.audio_assistant_slot_token_id),
        static_cast<int32_t>(assets_->config.audio_end_token_id),
    };
    const std::vector<float> text_candidate_logits = {
        text_logits.at(static_cast<size_t>(text_candidates[0])),
        text_logits.at(static_cast<size_t>(text_candidates[1])),
    };
    engine::sampling::HfSampler sampler;
    engine::sampling::HfSamplingOptions text_sampling;
    text_sampling.do_sample = sampling.do_sample;
    text_sampling.temperature = sampling.text_temperature;
    text_sampling.top_k = sampling.text_top_k;
    text_sampling.top_p = sampling.text_top_p;
    const engine::sampling::HfTorchSamplingState text_torch_state{
        &sampling_policy_,
        seed,
        sample_call_index,
    };
    const int32_t best_text = text_candidates[static_cast<size_t>(
        sampler.sample(
            text_candidate_logits,
            {},
            text_sampling,
            sampler_scratch_,
            rng,
            sampling.do_sample && sampling_policy_.cuda_fast_path ? &text_torch_state : nullptr,
            "MOSS-TTS-Nano local decoder"))];
    if (sampling.do_sample) {
        ++sample_call_index;
    }
    if (best_text == assets_->config.audio_end_token_id) {
        return {};
    }

    std::vector<int32_t> frame(static_cast<size_t>(assets_->config.n_vq), audio_pad_token_id());
    int32_t current = best_text;
    for (int64_t q = 0; q < active_codebooks; ++q) {
        const auto & logits = frame_graphs_->run_row(q + 1, current);
        std::vector<int32_t> codebook_history;
        codebook_history.reserve(history.size() / static_cast<size_t>(assets_->config.n_vq));
        for (size_t frame_index = 0; frame_index * static_cast<size_t>(assets_->config.n_vq) + static_cast<size_t>(q) < history.size(); ++frame_index) {
            codebook_history.push_back(history[frame_index * static_cast<size_t>(assets_->config.n_vq) + static_cast<size_t>(q)]);
        }
        engine::sampling::HfSamplingOptions audio_sampling;
        audio_sampling.do_sample = sampling.do_sample;
        audio_sampling.temperature = sampling.audio_temperature;
        audio_sampling.top_k = sampling.audio_top_k;
        audio_sampling.top_p = sampling.audio_top_p;
        audio_sampling.repetition_penalty = sampling.audio_repetition_penalty;
        const engine::sampling::HfTorchSamplingState audio_torch_state{
            &sampling_policy_,
            seed,
            sample_call_index,
        };
        const int32_t token = sampler.sample(
            logits,
            codebook_history,
            audio_sampling,
            sampler_scratch_,
            rng,
            sampling.do_sample && sampling_policy_.cuda_fast_path ? &audio_torch_state : nullptr,
            "MOSS-TTS-Nano local decoder");
        if (sampling.do_sample) {
            ++sample_call_index;
        }
        frame[static_cast<size_t>(q)] = token;
        current = token;
    }
    return frame;
}

}  // namespace engine::models::moss_tts_nano
