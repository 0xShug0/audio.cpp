#include "engine/models/moss/moss_tts_nano/global_transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
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

core::TensorValue attention_from_heads(
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
    scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const MossTTSNanoGlobalTransformerConfig & config) {
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
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, head_dim);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
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

struct MossTTSNanoGlobalTransformerRuntime::Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    std::vector<core::TensorValue> audio_embeddings;
    std::vector<TransformerLayerWeights> layers;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
};

struct MossTTSNanoGlobalTransformerRuntime::Graph {
public:
    Graph(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        const MossTTSNanoGlobalTransformerRuntime::Weights & weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        int64_t rows,
        int64_t row_width)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          rows_(rows),
          row_width_(row_width) {
        if (assets_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("MOSS global graph requires assets and backend");
        }
        if (rows_ <= 0 || row_width_ != assets_->config.n_vq + 1) {
            throw std::runtime_error("MOSS global graph shape is invalid");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MOSS global graph context");
        }

        const auto & config = assets_->config.global_transformer;
        core::ModuleBuildContext ctx{ctx_.get(), "moss_tts_nano.global", backend_type_};
        auto text_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows_}));
        text_ids_ = text_ids.tensor;
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows_}));
        positions_ = positions.tensor;
        auto hidden = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                          .build(ctx, text_ids, weights.text_embedding);
        for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
            auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows_}));
            auto mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows_, config.hidden_size}));
            audio_ids_.push_back(ids.tensor);
            audio_masks_.push_back(mask.tensor);
            auto emb = modules::EmbeddingModule({assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size})
                           .build(ctx, ids, weights.audio_embeddings.at(static_cast<size_t>(q)));
            hidden = modules::AddModule().build(ctx, hidden, modules::MulModule().build(ctx, emb, mask));
        }
        hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, rows_, config.hidden_size}));
        for (const auto & layer : weights.layers) {
            hidden = transformer_layer(ctx, hidden, positions, layer, config);
        }
        hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_epsilon, true, true})
                     .build(ctx, hidden, {weights.final_norm_weight, weights.final_norm_bias});
        output_ = hidden.tensor;
        output_values_.resize(static_cast<size_t>(rows_ * config.hidden_size));
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MOSS global graph");
        }
    }

    ~Graph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool can_run(int64_t rows, int64_t row_width, ggml_backend_t backend, int threads) const {
        return rows > 0 && rows <= rows_ && row_width_ == row_width && backend_ == backend &&
               compute_threads_ == std::max(1, threads);
    }

    std::vector<float> run(const std::vector<int32_t> & rows, int64_t row_count) {
        if (row_count <= 0 || row_count > rows_) {
            throw std::runtime_error("MOSS global row count exceeds graph capacity");
        }
        if (static_cast<int64_t>(rows.size()) != row_count * row_width_) {
            throw std::runtime_error("MOSS global input row count mismatch");
        }
        std::vector<int32_t> text_ids(static_cast<size_t>(rows_), 0);
        std::vector<int32_t> positions(static_cast<size_t>(rows_), 0);
        std::vector<std::vector<int32_t>> audio_ids(static_cast<size_t>(assets_->config.n_vq), std::vector<int32_t>(static_cast<size_t>(rows_), 0));
        std::vector<std::vector<float>> audio_masks(
            static_cast<size_t>(assets_->config.n_vq),
            std::vector<float>(static_cast<size_t>(rows_ * assets_->config.global_transformer.hidden_size), 0.0F));
        for (int64_t row = 0; row < rows_; ++row) {
            positions[static_cast<size_t>(row)] = static_cast<int32_t>(row);
            if (row >= row_count) {
                text_ids[static_cast<size_t>(row)] = static_cast<int32_t>(assets_->config.pad_token_id);
                continue;
            }
            text_ids[static_cast<size_t>(row)] = rows[static_cast<size_t>(row * row_width_)];
            for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
                const int32_t token = rows[static_cast<size_t>(row * row_width_ + 1 + q)];
                audio_ids[static_cast<size_t>(q)][static_cast<size_t>(row)] =
                    token == assets_->config.audio_pad_token_id ? 0 : token;
                if (token != assets_->config.audio_pad_token_id) {
                    const int64_t offset = row * assets_->config.global_transformer.hidden_size;
                    std::fill_n(
                        audio_masks[static_cast<size_t>(q)].begin() + static_cast<std::ptrdiff_t>(offset),
                        static_cast<size_t>(assets_->config.global_transformer.hidden_size),
                        1.0F);
                }
            }
        }
        core::write_tensor_i32({text_ids_, core::TensorShape::from_dims({rows_}), GGML_TYPE_I32}, text_ids);
        core::write_tensor_i32({positions_, core::TensorShape::from_dims({rows_}), GGML_TYPE_I32}, positions);
        for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
            core::write_tensor_i32({audio_ids_[static_cast<size_t>(q)], core::TensorShape::from_dims({rows_}), GGML_TYPE_I32}, audio_ids[static_cast<size_t>(q)]);
            core::write_tensor_f32(
                {audio_masks_[static_cast<size_t>(q)],
                 core::TensorShape::from_dims({rows_, assets_->config.global_transformer.hidden_size}),
                 GGML_TYPE_F32},
                audio_masks[static_cast<size_t>(q)]);
        }
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS global graph compute failed");
        }
        core::read_tensor_f32_into(output_, output_values_);
        const int64_t hidden_size = assets_->config.global_transformer.hidden_size;
        std::vector<float> out(static_cast<size_t>(hidden_size));
        const auto offset = static_cast<size_t>((row_count - 1) * hidden_size);
        std::copy_n(output_values_.begin() + static_cast<std::ptrdiff_t>(offset), static_cast<size_t>(hidden_size), out.begin());
        return out;
    }

private:
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t rows_ = 0;
    int64_t row_width_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    std::vector<ggml_tensor *> audio_ids_;
    std::vector<ggml_tensor *> audio_masks_;
    ggml_tensor * output_ = nullptr;
    std::vector<float> output_values_;
};

MossTTSNanoGlobalTransformerRuntime::MossTTSNanoGlobalTransformerRuntime(
    std::shared_ptr<const MossTTSNanoAssets> assets,
    core::ExecutionContext & execution_context,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      execution_context_(execution_context) {
    (void) decode_graph_arena_bytes;
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS global transformer requires assets");
    }
    weights_ = std::make_shared<Weights>();
    const auto & source = *assets_->model_weights;
    const auto & config = assets_->config.global_transformer;
    weights_->store = std::make_shared<core::BackendWeightStore>(
        execution_context_.backend(),
        execution_context_.backend_type(),
        "moss_tts_nano.global.weights",
        weight_context_bytes);
    weights_->text_embedding = weights_->store->load_tensor(
        source,
        "transformer.wte.weight",
        weight_storage_type,
        {config.vocab_size, config.hidden_size});
    weights_->audio_embeddings.reserve(static_cast<size_t>(assets_->config.n_vq));
    for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
        weights_->audio_embeddings.push_back(weights_->store->load_tensor(
            source,
            "audio_embeddings." + std::to_string(q) + ".weight",
            weight_storage_type,
            {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size}));
    }
    weights_->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer);
        TransformerLayerWeights w;
        w.ln1_weight = weights_->store->load_f32_tensor(source, prefix + ".ln_1.weight", {config.hidden_size});
        w.ln1_bias = weights_->store->load_f32_tensor(source, prefix + ".ln_1.bias", {config.hidden_size});
        w.c_attn = modules::binding::linear_from_source(*weights_->store, source, prefix + ".attn.c_attn", weight_storage_type, config.hidden_size * 3, config.hidden_size, true);
        w.c_proj = modules::binding::linear_from_source(*weights_->store, source, prefix + ".attn.c_proj", weight_storage_type, config.hidden_size, config.hidden_size, true);
        w.ln2_weight = weights_->store->load_f32_tensor(source, prefix + ".ln_2.weight", {config.hidden_size});
        w.ln2_bias = weights_->store->load_f32_tensor(source, prefix + ".ln_2.bias", {config.hidden_size});
        w.fc_in = modules::binding::linear_from_source(*weights_->store, source, prefix + ".mlp.fc_in", weight_storage_type, config.intermediate_size, config.hidden_size, true);
        w.fc_out = modules::binding::linear_from_source(*weights_->store, source, prefix + ".mlp.fc_out", weight_storage_type, config.hidden_size, config.intermediate_size, true);
        weights_->layers.push_back(std::move(w));
    }
    weights_->final_norm_weight = weights_->store->load_f32_tensor(source, "transformer.ln_f.weight", {config.hidden_size});
    weights_->final_norm_bias = weights_->store->load_f32_tensor(source, "transformer.ln_f.bias", {config.hidden_size});
    weights_->store->upload();
    prefill_graph_arena_bytes_ = prefill_graph_arena_bytes;
}

MossTTSNanoGlobalTransformerRuntime::~MossTTSNanoGlobalTransformerRuntime() = default;

void MossTTSNanoGlobalTransformerRuntime::prepare_prefill(int64_t prompt_rows) {
    if (prompt_rows <= 0) {
        throw std::runtime_error("MOSS global prefill requires positive prompt rows");
    }
}

void MossTTSNanoGlobalTransformerRuntime::prepare_decode(int64_t cache_rows) {
    if (cache_rows <= 0) {
        throw std::runtime_error("MOSS global decode requires positive cache rows");
    }
    decode_capacity_ = cache_rows;
    graph_.reset();
}

std::vector<float> MossTTSNanoGlobalTransformerRuntime::last_hidden(
    const std::vector<int32_t> & rows,
    int64_t row_count,
    int64_t row_width) {
    const int64_t graph_rows = std::max(row_count, decode_capacity_);
    if (graph_ == nullptr) {
        graph_ = std::make_unique<Graph>(
            assets_,
            *weights_,
            execution_context_,
            prefill_graph_arena_bytes_,
            graph_rows,
            row_width);
        ++graph_builds_;
    } else if (!graph_->can_run(row_count, row_width, execution_context_.backend(), execution_context_.config().threads)) {
        throw std::runtime_error("MOSS global graph was prepared with an incompatible shape");
    }
    return graph_->run(rows, row_count);
}

int64_t MossTTSNanoGlobalTransformerRuntime::graph_builds() const noexcept {
    return graph_builds_;
}

}  // namespace engine::models::moss_tts_nano
