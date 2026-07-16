#include "engine/models/moss/moss_tts_nano/local_frame_decoder.h"

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
#include "engine/models/moss/shared/sampling.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
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
    return matmul.build(ctx, core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32), v_heads);
}

core::TensorValue transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const MossTTSNanoLocalTransformerConfig & config) {
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
    x = add.build(ctx, input, modules::LinearModule({config.hidden_size, config.hidden_size, true})
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

struct MossTTSNanoLocalFrameDecoderRuntime::TextGraph {
    TextGraph(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        const Weights & weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)) {
        if (assets_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("MOSS local text graph requires assets and backend");
        }
        const auto & config = assets_->config.local_transformer;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MOSS local text graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "moss_tts_nano.local.text", backend_type_};
        auto hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({config.hidden_size}));
        global_hidden_ = hidden.tensor;
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        positions_ = positions.tensor;
        position_values_ = {0};
        auto x = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        for (const auto & layer : weights.layers) {
            x = transformer_layer(ctx, x, positions, layer, config);
        }
        x = modules::LayerNormModule({config.hidden_size, config.layer_norm_epsilon, true, true})
                .build(ctx, x, {weights.final_norm_weight, weights.final_norm_bias});
        x = modules::LinearModule({config.hidden_size, assets_->config.global_transformer.vocab_size, false})
                .build(ctx, x, {weights.text_lm_head, std::nullopt});
        logits_ = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({assets_->config.global_transformer.vocab_size})).tensor;
        logits_values_.resize(static_cast<size_t>(assets_->config.global_transformer.vocab_size));
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, logits_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MOSS local text graph");
        }
    }

    ~TextGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    std::vector<float> run(const std::vector<float> & global_hidden) {
        if (static_cast<int64_t>(global_hidden.size()) != assets_->config.local_transformer.hidden_size) {
            throw std::runtime_error("MOSS local text graph hidden size mismatch");
        }
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(global_hidden_, global_hidden.data(), 0, global_hidden.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS local text graph compute failed");
        }
        core::read_tensor_f32_into(logits_, logits_values_);
        return logits_values_;
    }

    std::shared_ptr<const MossTTSNanoAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_tensor * global_hidden_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    std::vector<int32_t> position_values_;
    ggml_tensor * logits_ = nullptr;
    std::vector<float> logits_values_;
};

struct MossTTSNanoLocalFrameDecoderRuntime::AudioGraph {
    AudioGraph(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        const Weights & weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        int64_t codebook)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          codebook_(codebook) {
        if (assets_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("MOSS local audio graph requires assets and backend");
        }
        const auto & config = assets_->config.local_transformer;
        if (codebook_ < 0 || codebook_ >= assets_->config.n_vq) {
            throw std::runtime_error("MOSS local audio graph codebook is out of range");
        }
        rows_ = codebook_ + 2;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MOSS local audio graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "moss_tts_nano.local.audio", backend_type_};
        auto global_hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({config.hidden_size}));
        global_hidden_ = global_hidden.tensor;
        auto text_id = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        text_id_ = text_id.tensor;
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows_}));
        positions_ = positions.tensor;
        auto x = core::reshape_tensor(ctx, global_hidden, core::TensorShape::from_dims({1, config.hidden_size}));
        auto text = modules::EmbeddingModule({assets_->config.global_transformer.vocab_size, config.hidden_size})
                        .build(ctx, text_id, weights.text_embedding);
        x = modules::ConcatModule({0}).build(ctx, x, text);
        if (codebook_ > 0) {
            auto prev = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({codebook_}));
            prev_ids_ = prev.tensor;
            for (int64_t q = 0; q < codebook_; ++q) {
                auto id = modules::SliceModule({0, q, 1}).build(ctx, prev);
                if (backend_type_ == core::BackendType::Vulkan) {
                    id = core::ensure_backend_addressable_layout(ctx, id);
                }
                auto emb = modules::EmbeddingModule({assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size})
                               .build(ctx, id, weights.audio_embeddings.at(static_cast<size_t>(q)));
                x = modules::ConcatModule({0}).build(ctx, x, emb);
            }
        }
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, rows_, config.hidden_size}));
        for (const auto & layer : weights.layers) {
            x = transformer_layer(ctx, x, positions, layer, config);
        }
        x = modules::LayerNormModule({config.hidden_size, config.layer_norm_epsilon, true, true})
                .build(ctx, x, {weights.final_norm_weight, weights.final_norm_bias});
        auto last = modules::SliceModule({1, rows_ - 1, 1}).build(ctx, x);
        last = modules::LinearModule({config.hidden_size, assets_->config.audio_codebook_sizes.at(static_cast<size_t>(codebook_)), false})
                   .build(ctx, last, {weights.audio_lm_heads.at(static_cast<size_t>(codebook_)), std::nullopt});
        logits_ = core::reshape_tensor(ctx, last, core::TensorShape::from_dims({assets_->config.audio_codebook_sizes.at(static_cast<size_t>(codebook_))})).tensor;
        logits_values_.resize(static_cast<size_t>(assets_->config.audio_codebook_sizes.at(static_cast<size_t>(codebook_))));
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, logits_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MOSS local audio graph");
        }
        position_values_.resize(static_cast<size_t>(rows_));
        for (int64_t i = 0; i < rows_; ++i) {
            position_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
    }

    ~AudioGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    std::vector<float> run(const std::vector<float> & global_hidden, const std::vector<int32_t> & previous_tokens, int32_t text_token) {
        if (static_cast<int64_t>(global_hidden.size()) != assets_->config.local_transformer.hidden_size) {
            throw std::runtime_error("MOSS local audio graph hidden size mismatch");
        }
        if (static_cast<int64_t>(previous_tokens.size()) != codebook_) {
            throw std::runtime_error("MOSS local audio graph previous token count mismatch");
        }
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(global_hidden_, global_hidden.data(), 0, global_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(text_id_, &text_token, 0, sizeof(text_token));
        if (codebook_ > 0) {
            ggml_backend_tensor_set(prev_ids_, previous_tokens.data(), 0, previous_tokens.size() * sizeof(int32_t));
        }
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS local audio graph compute failed");
        }
        core::read_tensor_f32_into(logits_, logits_values_);
        return logits_values_;
    }

    std::shared_ptr<const MossTTSNanoAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t codebook_ = 0;
    int64_t rows_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_tensor * global_hidden_ = nullptr;
    ggml_tensor * text_id_ = nullptr;
    ggml_tensor * prev_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    std::vector<int32_t> position_values_;
    ggml_tensor * logits_ = nullptr;
    std::vector<float> logits_values_;
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
    weights_ = std::make_shared<Weights>();
    weights_->store = std::make_shared<core::BackendWeightStore>(
        execution_context_.backend(),
        execution_context_.backend_type(),
        "moss_tts_nano.local.weights",
        weight_context_bytes);
    weights_->text_embedding = weights_->store->load_tensor(
        source,
        "transformer.wte.weight",
        weight_storage_type,
        {assets_->config.global_transformer.vocab_size, config.hidden_size});
    weights_->text_lm_head = weights_->store->load_tensor(
        source,
        "text_lm_head.weight",
        weight_storage_type,
        {assets_->config.global_transformer.vocab_size, config.hidden_size});
    weights_->audio_embeddings.reserve(static_cast<size_t>(assets_->config.n_vq));
    weights_->audio_lm_heads.reserve(static_cast<size_t>(assets_->config.n_vq));
    for (int64_t q = 0; q < assets_->config.n_vq; ++q) {
        weights_->audio_embeddings.push_back(weights_->store->load_tensor(
            source,
            "audio_embeddings." + std::to_string(q) + ".weight",
            weight_storage_type,
            {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size}));
        weights_->audio_lm_heads.push_back(weights_->store->load_tensor(
            source,
            "audio_lm_heads." + std::to_string(q) + ".weight",
            weight_storage_type,
            {assets_->config.audio_codebook_sizes.at(static_cast<size_t>(q)), config.hidden_size}));
    }
    weights_->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "local_transformer.h." + std::to_string(layer);
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
    weights_->final_norm_weight = weights_->store->load_f32_tensor(source, "local_transformer.ln_f.weight", {config.hidden_size});
    weights_->final_norm_bias = weights_->store->load_f32_tensor(source, "local_transformer.ln_f.bias", {config.hidden_size});
    weights_->store->upload();
    audio_graphs_.resize(static_cast<size_t>(assets_->config.n_vq));
}

MossTTSNanoLocalFrameDecoderRuntime::~MossTTSNanoLocalFrameDecoderRuntime() = default;

void MossTTSNanoLocalFrameDecoderRuntime::prepare(int64_t active_codebooks) {
    if (active_codebooks <= 0 || active_codebooks > assets_->config.n_vq) {
        throw std::runtime_error("MOSS local frame decoder active codebooks are out of range");
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
    if (text_graph_ == nullptr) {
        text_graph_ = std::make_unique<TextGraph>(assets_, *weights_, execution_context_, graph_arena_bytes_);
    }
    const auto text_logits = text_graph_->run(global_hidden);
    const std::array<int32_t, 2> text_candidates = {
        static_cast<int32_t>(assets_->config.audio_assistant_slot_token_id),
        static_cast<int32_t>(assets_->config.audio_end_token_id),
    };
    const std::vector<float> text_candidate_logits = {
        text_logits.at(static_cast<size_t>(text_candidates[0])),
        text_logits.at(static_cast<size_t>(text_candidates[1])),
    };
    const int32_t best_text = text_candidates[static_cast<size_t>(
        sampling.do_sample
            ? moss::sample_index(
                  text_candidate_logits,
                  sampling.text_top_k,
                  sampling.text_top_p,
                  sampling.text_temperature,
                  rng,
                  "MOSS-TTS-Nano local decoder",
                  &sampling_policy_,
                  seed,
                  sample_call_index++)
            : moss::argmax_index(text_candidate_logits, "MOSS-TTS-Nano local decoder"))];
    if (best_text == assets_->config.audio_end_token_id) {
        return {};
    }

    std::vector<int32_t> frame(static_cast<size_t>(assets_->config.n_vq), audio_pad_token_id());
    std::vector<int32_t> previous;
    previous.reserve(static_cast<size_t>(active_codebooks));
    for (int64_t q = 0; q < active_codebooks; ++q) {
        if (audio_graphs_[static_cast<size_t>(q)] == nullptr) {
            audio_graphs_[static_cast<size_t>(q)] =
                std::make_unique<AudioGraph>(assets_, *weights_, execution_context_, graph_arena_bytes_, q);
        }
        auto logits = audio_graphs_[static_cast<size_t>(q)]->run(global_hidden, previous, best_text);
        std::vector<int32_t> codebook_history;
        codebook_history.reserve(history.size() / static_cast<size_t>(assets_->config.n_vq));
        for (size_t frame = 0; frame * static_cast<size_t>(assets_->config.n_vq) + static_cast<size_t>(q) < history.size(); ++frame) {
            codebook_history.push_back(history[frame * static_cast<size_t>(assets_->config.n_vq) + static_cast<size_t>(q)]);
        }
        moss::apply_repetition_penalty(
            logits,
            codebook_history,
            sampling.audio_repetition_penalty,
            "MOSS-TTS-Nano local decoder");
        const int32_t token = sampling.do_sample
            ? moss::sample_index(
                  logits,
                  sampling.audio_top_k,
                  sampling.audio_top_p,
                  sampling.audio_temperature,
                  rng,
                  "MOSS-TTS-Nano local decoder",
                  &sampling_policy_,
                  seed,
                  sample_call_index++)
            : moss::argmax_index(logits, "MOSS-TTS-Nano local decoder");
        frame[static_cast<size_t>(q)] = token;
        previous.push_back(token);
    }
    return frame;
}

}  // namespace engine::models::moss_tts_nano
