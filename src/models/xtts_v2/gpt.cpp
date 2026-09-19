#include "engine/models/xtts_v2/gpt.h"

#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace engine::models::xtts_v2 {
namespace {
namespace modules = engine::modules;

constexpr int64_t kDim = 1024;
constexpr int64_t kHeads = 16;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kAudioCodes = 1026;

struct ContextDeleter { void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); } };

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], kHeads, kHeadDim}));
}

core::TensorValue project(core::ModuleBuildContext & ctx, const core::TensorValue & x,
                          int64_t input, int64_t output, const modules::LinearWeights & weights) {
    return modules::LinearModule({input, output, true, GGML_PREC_F32}).build(ctx, x, weights);
}

core::TensorValue gpt_layer(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                            const XttsV2GptLayerWeights & weights) {
    auto normed = modules::LayerNormModule({kDim, 1.0e-5F, true, true}).build(ctx, input, weights.attn_norm);
    auto qkv = project(ctx, normed, kDim, 3 * kDim, weights.qkv);
    auto q = modules::SliceModule({2, 0, kDim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, kDim, kDim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * kDim, kDim}).build(ctx, qkv);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, k));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, v));
    auto scores = modules::MatMulModule{}.build(ctx, q, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(64.0F)), scores.shape, GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
    auto probability = core::wrap_tensor(
        ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto mixed = modules::MatMulModule{}.build(ctx, probability, v);
    mixed = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, mixed);
    mixed = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mixed), input.shape);
    auto x = modules::AddModule{}.build(ctx, input, project(ctx, mixed, kDim, kDim, weights.attn_out));
    auto hidden = modules::LayerNormModule({kDim, 1.0e-5F, true, true}).build(ctx, x, weights.mlp_norm);
    hidden = project(ctx, hidden, kDim, 4 * kDim, weights.mlp_in);
    hidden = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, hidden);
    hidden = project(ctx, hidden, 4 * kDim, kDim, weights.mlp_out);
    return modules::AddModule{}.build(ctx, x, hidden);
}

}  // namespace

std::shared_ptr<const XttsV2GptWeights> load_xtts_v2_gpt_weights(
    const XttsV2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    namespace binding = engine::modules::binding;
    constexpr int64_t dim = 1024;
    constexpr int64_t mlp = 4096;
    auto weights = std::make_shared<XttsV2GptWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "xtts_v2.gpt.weights", weight_context_bytes);
    const auto & source = *assets.gpt;
    weights->text_embedding = weights->store->load_tensor(source, "text_embedding.weight", storage_type, {6681, dim});
    weights->audio_embedding = weights->store->load_tensor(source, "mel_embedding.weight", storage_type, {1026, dim});
    weights->text_positions = weights->store->load_f32_tensor(source, "text_pos_embedding.emb.weight", {404, dim});
    weights->audio_positions = weights->store->load_f32_tensor(source, "mel_pos_embedding.emb.weight", {608, dim});
    weights->layers.reserve(30);
    for (int64_t index = 0; index < 30; ++index) {
        const std::string prefix = "gpt.h." + std::to_string(index);
        XttsV2GptLayerWeights layer;
        layer.attn_norm = binding::norm_from_source(*weights->store, source, prefix + ".ln_1", dim);
        layer.qkv = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".attn.c_attn", storage_type, dim, 3 * dim, true);
        layer.attn_out = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".attn.c_proj", storage_type, dim, dim, true);
        layer.mlp_norm = binding::norm_from_source(*weights->store, source, prefix + ".ln_2", dim);
        layer.mlp_in = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".mlp.c_fc", storage_type, dim, mlp, true);
        layer.mlp_out = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".mlp.c_proj", storage_type, mlp, dim, true);
        weights->layers.push_back(std::move(layer));
    }
    weights->transformer_norm = binding::norm_from_source(*weights->store, source, "gpt.ln_f", dim);
    weights->output_norm = binding::norm_from_source(*weights->store, source, "final_norm", dim);
    weights->audio_head = binding::linear_from_source(
        *weights->store, source, "mel_head", storage_type, 1026, dim, true);
    weights->store->upload();
    return weights;
}

class XttsV2GptRuntime::PrefillGraph {
public:
    PrefillGraph(core::ExecutionContext & execution, std::shared_ptr<const XttsV2GptWeights> weights,
                 int64_t text_count, int64_t audio_count, size_t arena)
        : execution_(execution), weights_(std::move(weights)), text_count_(text_count), audio_count_(audio_count),
          steps_(32 + text_count + 2 + audio_count) {
        ctx_.reset(ggml_init({arena, nullptr, true}));
        input_ctx_.reset(ggml_init({8U * 1024U * 1024U, nullptr, true}));
        if (!ctx_ || !input_ctx_) throw std::runtime_error("failed to initialize XTTS v2 GPT prefill graph");
        core::ModuleBuildContext ctx{ctx_.get(), "xtts_v2.gpt.prefill", execution_.backend_type()};
        core::ModuleBuildContext ictx{input_ctx_.get(), "xtts_v2.gpt.prefill.inputs", execution_.backend_type()};
        conditioning_ = core::make_tensor(ictx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 32, kDim})).tensor;
        text_ids_ = ggml_new_tensor_1d(input_ctx_.get(), GGML_TYPE_I32, text_count_ + 2);
        audio_id_ = ggml_new_tensor_1d(input_ctx_.get(), GGML_TYPE_I32, audio_count_);
        ggml_set_input(conditioning_); ggml_set_input(text_ids_); ggml_set_input(audio_id_);
        auto condition = core::wrap_tensor(conditioning_, core::TensorShape::from_dims({1, 32, kDim}), GGML_TYPE_F32);
        auto ids = core::wrap_tensor(text_ids_, core::TensorShape::from_dims({text_count_ + 2}), GGML_TYPE_I32);
        auto text = modules::EmbeddingModule({6681, kDim}).build(ctx, ids, weights_->text_embedding);
        text = modules::AddModule{}.build(ctx, text, modules::SliceModule({0, 0, text_count_ + 2}).build(ctx, weights_->text_positions));
        text = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, text), core::TensorShape::from_dims({1, text_count_ + 2, kDim}));
        auto aid = core::wrap_tensor(audio_id_, core::TensorShape::from_dims({audio_count_}), GGML_TYPE_I32);
        auto audio = modules::EmbeddingModule({kAudioCodes, kDim}).build(ctx, aid, weights_->audio_embedding);
        audio = modules::AddModule{}.build(ctx, audio, modules::SliceModule({0, 0, audio_count_}).build(ctx, weights_->audio_positions));
        audio = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, audio), core::TensorShape::from_dims({1, audio_count_, kDim}));
        auto x = modules::ConcatModule({1}).build(ctx, condition, text);
        x = modules::ConcatModule({1}).build(ctx, x, audio);
        for (const auto & layer : weights_->layers) x = gpt_layer(ctx, x, layer);
        x = modules::LayerNormModule({kDim, 1.0e-5F, true, true}).build(ctx, x, weights_->transformer_norm);
        x = modules::SliceModule({1, steps_ - 1, 1}).build(ctx, x);
        x = modules::LayerNormModule({kDim, 1.0e-5F, true, true}).build(ctx, x, weights_->output_norm);
        latent_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        logits_ = core::ensure_backend_addressable_layout(ctx, project(ctx, x, kDim, kAudioCodes, weights_->audio_head)).tensor;
        ggml_set_output(latent_); ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, logits_); ggml_build_forward_expand(graph_, latent_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!input_buffer_ || !allocator_ || !ggml_gallocr_reserve(allocator_, graph_) || !ggml_gallocr_alloc_graph(allocator_, graph_))
            throw std::runtime_error("failed to allocate XTTS v2 GPT prefill graph");
    }
    ~PrefillGraph() {
        if (graph_) core::release_backend_graph_resources(execution_.backend(), graph_);
        if (allocator_) ggml_gallocr_free(allocator_);
        if (input_buffer_) ggml_backend_buffer_free(input_buffer_);
    }
    bool matches(int64_t text_count, int64_t audio_count) const noexcept {
        return text_count == text_count_ && audio_count == audio_count_;
    }
    XttsV2GptPrefillResult run(const std::vector<float> & condition, const std::vector<int32_t> & tokens,
                               const std::vector<int32_t> & audio_tokens) {
        if (condition.size() != 32U * 1024U || static_cast<int64_t>(tokens.size()) != text_count_ ||
            static_cast<int64_t>(audio_tokens.size()) != audio_count_)
            throw std::runtime_error("XTTS v2 GPT prefill input shape mismatch");
        std::vector<int32_t> ids; ids.reserve(tokens.size() + 2); ids.push_back(261); ids.insert(ids.end(), tokens.begin(), tokens.end()); ids.push_back(0);
        ggml_backend_tensor_set(conditioning_, condition.data(), 0, condition.size() * sizeof(float));
        ggml_backend_tensor_set(text_ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(audio_id_, audio_tokens.data(), 0, audio_tokens.size() * sizeof(int32_t));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("XTTS v2 GPT prefill compute failed");
        ggml_backend_synchronize(execution_.backend());
        XttsV2GptPrefillResult result; result.logits.resize(kAudioCodes); result.latent.resize(kDim);
        ggml_backend_tensor_get(logits_, result.logits.data(), 0, result.logits.size() * sizeof(float));
        ggml_backend_tensor_get(latent_, result.latent.data(), 0, result.latent.size() * sizeof(float));
        return result;
    }
private:
    core::ExecutionContext & execution_; std::shared_ptr<const XttsV2GptWeights> weights_;
    int64_t text_count_, audio_count_, steps_; std::unique_ptr<ggml_context, ContextDeleter> ctx_, input_ctx_;
    ggml_tensor * conditioning_ = nullptr, * text_ids_ = nullptr, * audio_id_ = nullptr, * logits_ = nullptr, * latent_ = nullptr;
    ggml_cgraph * graph_ = nullptr; ggml_gallocr_t allocator_ = nullptr; ggml_backend_buffer_t input_buffer_ = nullptr;
};

XttsV2GptRuntime::XttsV2GptRuntime(const XttsV2Assets & assets, core::ExecutionContext & execution,
    size_t weight_context_bytes, size_t graph_context_bytes, assets::TensorStorageType storage_type)
    : execution_(execution), graph_context_bytes_(graph_context_bytes),
      weights_(load_xtts_v2_gpt_weights(assets, execution, weight_context_bytes, storage_type)) {}
XttsV2GptRuntime::~XttsV2GptRuntime() = default;
XttsV2GptPrefillResult XttsV2GptRuntime::prefill(const std::vector<float> & condition, const std::vector<int32_t> & tokens) {
    if (!prefill_ || !prefill_->matches(static_cast<int64_t>(tokens.size()), 1))
        prefill_ = std::make_unique<PrefillGraph>(execution_, weights_, static_cast<int64_t>(tokens.size()), 1, graph_context_bytes_);
    return prefill_->run(condition, tokens, {1024});
}

XttsV2GptGeneration XttsV2GptRuntime::generate(
    const std::vector<float> & condition, const std::vector<int32_t> & text,
    const XttsV2GenerationOptions & options) {
    if (text.empty() || text.size() >= 400) throw std::runtime_error("XTTS v2 GPT text token count is out of range");
    std::vector<int32_t> input{1024};
    XttsV2GptGeneration result;
    std::mt19937 random(options.seed);
    const int64_t limit = std::min<int64_t>(options.max_tokens, 603);
    for (int64_t step = 0; step < limit; ++step) {
        if (!prefill_ || !prefill_->matches(static_cast<int64_t>(text.size()), static_cast<int64_t>(input.size())))
            prefill_ = std::make_unique<PrefillGraph>(execution_, weights_, static_cast<int64_t>(text.size()), static_cast<int64_t>(input.size()), graph_context_bytes_);
        auto output = prefill_->run(condition, text, input);
        std::vector<float> logits = output.logits;
        std::vector<bool> penalized(1025, false);
        // Transformers applies the processor to the complete synthetic input-id
        // sequence. XTTS fills its 34+text prefix positions with token 1 and ends
        // that prefix with the audio BOS token, even though those ids are replaced
        // by cached embeddings inside the model.
        penalized[1] = true;
        penalized[1024] = true;
        for (int32_t prefix_id : {1, 1024}) {
            float & value = logits[static_cast<size_t>(prefix_id)];
            value = value < 0.0F ? value * options.repetition_penalty : value / options.repetition_penalty;
        }
        for (int32_t prior : result.codes) {
            // Hugging Face's RepetitionPenaltyLogitsProcessor applies the
            // penalty once per token id present in the history, not once per
            // occurrence. Reapplying it to repeated codes quickly destroys the
            // acoustic distribution and produces rough, overlong speech.
            if (prior >= 0 && prior < 1025 && !penalized[static_cast<size_t>(prior)]) {
                float & value = logits[static_cast<size_t>(prior)];
                value = value < 0.0F ? value * options.repetition_penalty : value / options.repetition_penalty;
                penalized[static_cast<size_t>(prior)] = true;
            }
        }
        logits[1025] = step < 2 ? -std::numeric_limits<float>::infinity() : logits[1025];
        std::vector<int32_t> order(1026); for (int32_t i = 0; i < 1026; ++i) order[static_cast<size_t>(i)] = i;
        std::partial_sort(order.begin(), order.begin() + std::min<int64_t>(options.top_k, 1026), order.end(),
            [&](int32_t a, int32_t b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });
        const size_t keep_k = static_cast<size_t>(std::min<int64_t>(options.top_k, 1026));
        order.resize(keep_k);
        const float max_logit = logits[static_cast<size_t>(order.front())];
        std::vector<double> probability(order.size()); double total = 0.0;
        for (size_t i = 0; i < order.size(); ++i) { probability[i] = std::exp((logits[static_cast<size_t>(order[i])] - max_logit) / options.temperature); total += probability[i]; }
        double cumulative = 0.0; size_t nucleus = probability.size();
        for (size_t i = 0; i < probability.size(); ++i) { cumulative += probability[i] / total; if (cumulative >= options.top_p) { nucleus = i + 1; break; } }
        probability.resize(nucleus); order.resize(nucleus);
        std::discrete_distribution<size_t> distribution(probability.begin(), probability.end());
        const int32_t next = order[distribution(random)];
        if (next == 1025) break;
        result.codes.push_back(next);
        result.latents.insert(result.latents.end(), output.latent.begin(), output.latent.end());
        input.push_back(next);
    }
    return result;
}

}  // namespace engine::models::xtts_v2
