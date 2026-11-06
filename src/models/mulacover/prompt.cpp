#include "engine/models/mulacover/prompt.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "unicode.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::mulacover {

class PromptEncoder::Impl {
    using Shape = core::TensorShape;
    using Value = core::TensorValue;

    struct Graph {
        ggml_backend_t backend;
        int64_t steps, batch;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
        ggml_cgraph * graph = nullptr;
        core::HostGraphPlan plan;
        ggml_tensor * ids = nullptr;
        ggml_tensor * zeros = nullptr;
        ggml_tensor * mask = nullptr;
        ggml_tensor * inverse_mask = nullptr;
        ggml_tensor * qwen = nullptr;
        ggml_tensor * muq = nullptr;
        ggml_tensor * rows = nullptr;
        ggml_tensor * output = nullptr;

        Graph(Impl & owner, int64_t steps, int64_t batch) : backend(owner.backend_), steps(steps), batch(batch) {
            const auto dims = owner.dims_;
            context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
            if (!context) throw std::runtime_error("MuLaCover prompt graph context allocation failed");
            core::ModuleBuildContext ctx{context.get(), "mulacover.prompt"};
            ids = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, steps * batch);
            zeros = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, steps * batch);
            mask = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, 1, steps * batch);
            inverse_mask = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, 1, steps * batch);
            qwen = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, owner.qwen_dim_, 1);
            muq = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, owner.muq_dim_, 1);
            rows = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, 2);
            for (auto * input : {ids, zeros, mask, inverse_mask, qwen, muq, rows}) ggml_set_input(input);
            auto text = modules::EmbeddingModule({owner.vocab_, dims}).build(ctx,
                core::wrap_tensor(ids, Shape::from_dims({steps * batch}), GGML_TYPE_I32), owner.text_);
            auto uncond = modules::EmbeddingModule({1, dims}).build(ctx,
                core::wrap_tensor(zeros, Shape::from_dims({steps * batch}), GGML_TYPE_I32), owner.uncond_);
            auto style_input = core::wrap_tensor(qwen, Shape::from_dims({1, owner.qwen_dim_}), GGML_TYPE_F32);
            if (owner.compute_type_ == GGML_TYPE_BF16) style_input = modules::Bf16RoundModule{}.build(ctx, style_input);
            else if (owner.compute_type_ == GGML_TYPE_F16) style_input = core::wrap_tensor(
                ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, style_input.tensor, GGML_TYPE_F16), GGML_TYPE_F32), style_input.shape, GGML_TYPE_F32);
            auto q = modules::LinearModule(modules::binding::linear_config(owner.qwen_dim_, dims, true)).build(ctx,
                style_input, owner.qwen_weights_);
            auto m = modules::LinearModule(modules::binding::linear_config(owner.muq_dim_, dims, true)).build(ctx,
                core::wrap_tensor(muq, Shape::from_dims({1, owner.muq_dim_}), GGML_TYPE_F32), owner.muq_weights_);
            auto projected = modules::ConcatModule({0}).build(ctx, q, m);
            if (owner.compute_type_ == GGML_TYPE_BF16) projected = modules::Bf16RoundModule{}.build(ctx, projected);
            else if (owner.compute_type_ == GGML_TYPE_F16) projected = core::wrap_tensor(
                ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, projected.tensor, GGML_TYPE_F16), GGML_TYPE_F32), projected.shape, GGML_TYPE_F32);
            // Only branch zero gets projected placeholders; branch one is entirely
            // unconditional, including both style positions.
            text = core::wrap_tensor(ggml_set_rows(ctx.ggml, text.tensor, projected.tensor, rows),
                                    text.shape, GGML_TYPE_F32);
            auto cond_mask = modules::RepeatModule({Shape::from_dims({steps * batch, dims})}).build(ctx,
                core::wrap_tensor(mask, Shape::from_dims({steps * batch, 1}), GGML_TYPE_F32));
            auto uncond_mask = modules::RepeatModule({Shape::from_dims({steps * batch, dims})}).build(ctx,
                core::wrap_tensor(inverse_mask, Shape::from_dims({steps * batch, 1}), GGML_TYPE_F32));
            output = modules::AddModule{}.build(ctx,
                modules::MulModule{}.build(ctx, text, cond_mask),
                modules::MulModule{}.build(ctx, uncond, uncond_mask)).tensor;
            ggml_set_output(output);
            graph = ggml_new_graph_custom(ctx.ggml, 1024, false);
            ggml_build_forward_expand(graph, output);
            core::validate_backend_graph_supported(backend, graph, "mulacover.prompt");
        }
        void allocate_workspace() {
            if (allocator) return;
            allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
            if (!allocator || !ggml_gallocr_reserve(allocator.get(), graph) ||
                !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                throw std::runtime_error("MuLaCover prompt graph workspace allocation failed");
            }
            plan.backend = backend;
            plan.plan = core::create_backend_graph_plan_if_host(backend, graph);
            debug::timing_log_scalar("mulacover.prompt.graph_buffer_mb",
                static_cast<double>(ggml_gallocr_get_buffer_size(allocator.get(), 0)) / (1024.0 * 1024.0));
        }
        void release_workspace() {
            plan.reset();
            if (!allocator) return;
            core::release_backend_graph_resources(backend, graph, true);
            allocator.reset();
            for (auto * tensor = ggml_get_first_tensor(context.get()); tensor;
                 tensor = ggml_get_next_tensor(context.get(), tensor)) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
                tensor->extra = nullptr;
            }
            debug::timing_log_scalar("mulacover.prompt.graph_buffer_mb", 0.0);
        }
        ~Graph() { release_workspace(); }
    };

public:
    Impl(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
         const std::filesystem::path & tokenizer_json, MuLaCoverGenerationConfig config,
         const PromptWeights & weights)
        : backend_(execution.backend()), config_(std::move(config)),
          tokenizer_(tokenizers::LlamaBpeTokenizerSpec{{}, {}, {}, tokenizer_json,
                     tokenizers::LlamaBpePreTokenizer::Llama3}) {
        if (!source) throw std::invalid_argument("MuLaCover prompt requires a tensor source");
        const auto shape = source->require_metadata("text_embeddings.weight").shape;
        if (shape.size() != 2) throw std::invalid_argument("MuLaCover text embedding must be a matrix");
        vocab_ = shape[0]; dims_ = shape[1];
        const auto qshape = source->require_metadata("qwen_linear.weight").shape;
        const auto mshape = source->require_metadata("muq_linear.weight").shape;
        if (qshape.size() != 2 || mshape.size() != 2 || qshape[0] != dims_ || mshape[0] != dims_)
            throw std::invalid_argument("MuLaCover prompt projection dimensions do not match embeddings");
        qwen_dim_ = qshape[1]; muq_dim_ = mshape[1];
        compute_type_ = weights.compute_type;
        text_ = weights.text;
        uncond_ = weights.unconditional;
        qwen_weights_ = weights.qwen;
        muq_weights_ = weights.muq;
        if (text_.shape.dims[0] != vocab_ || text_.shape.dims[1] != dims_) {
            throw std::invalid_argument("MuLaCover shared prompt embedding dimensions mismatch");
        }
    }

    PromptEmbeddings encode(
        const std::string & lyrics, const std::string & tags,
        const std::vector<float> & style, float cfg) {
        const auto started = std::chrono::steady_clock::now();
        if (!std::isfinite(cfg) || cfg <= 0) throw std::invalid_argument("cfg_scale must be finite and positive");
        if (style.size() != static_cast<size_t>(qwen_dim_)) throw std::invalid_argument("MuLaCover Qwen style dimension mismatch");
        // Use the tokenizer framework's Unicode tables, including Unicode space.
        auto normalize = [](const std::string & input, bool trim) {
            auto points = unicode_cpts_from_utf8(input);
            size_t begin = 0, end = points.size();
            if (trim) {
                while (begin < end && unicode_cpt_flags_from_cpt(points[begin]).is_whitespace) ++begin;
                while (end > begin && unicode_cpt_flags_from_cpt(points[end - 1]).is_whitespace) --end;
            }
            std::string out;
            for (size_t i = begin; i < end; ++i) out += unicode_cpt_to_utf8(unicode_tolower(points[i]));
            return out;
        };
        const auto lyric_text = normalize(lyrics, false);
        if (normalize(lyrics, true).empty()) throw std::invalid_argument("lyrics must not be empty");
        auto tag_text = PromptEncoder::normalize_tags(tags);
        std::vector<int32_t> ids{static_cast<int32_t>(config_.tag_start_id)};
        auto tag_ids = tokenizer_.encode(tag_text);
        ids.insert(ids.end(), tag_ids.begin(), tag_ids.end());
        ids.push_back(static_cast<int32_t>(config_.tag_end_id));
        const int32_t qrow = static_cast<int32_t>(ids.size() + 1), mrow = qrow + 3;
        ids.insert(ids.end(), {static_cast<int32_t>(config_.qwen_start_id), 0,
            static_cast<int32_t>(config_.qwen_end_id), static_cast<int32_t>(config_.muq_start_id), 0,
            static_cast<int32_t>(config_.muq_end_id), static_cast<int32_t>(config_.text_bos_id)});
        auto lyric_ids = tokenizer_.encode(lyric_text);
        ids.insert(ids.end(), lyric_ids.begin(), lyric_ids.end());
        ids.push_back(static_cast<int32_t>(config_.text_eos_id));
        const int64_t steps = static_cast<int64_t>(ids.size()), batch = cfg > 1.0f ? 2 : 1;
        if (steps > 8192) throw std::invalid_argument("MuLaCover prompt exceeds the 8192-token context");
        for (auto id : ids) if (id < 0 || id >= vocab_) throw std::invalid_argument("MuLaCover prompt token outside vocabulary");
        if (!graph_ || graph_->steps != steps || graph_->batch != batch) {
            graph_.reset();
            graph_ = std::make_unique<Graph>(*this, steps, batch);
        }
        auto & graph = *graph_;
        graph.allocate_workspace();
        if (batch == 2) { const auto first = ids; ids.insert(ids.end(), first.begin(), first.end()); }
        std::vector<int32_t> zero_ids(ids.size(), 0);
        std::vector<float> cond(ids.size(), 0), uncond(ids.size(), 0), zero_muq(static_cast<size_t>(muq_dim_), 0);
        for (int64_t i = 0; i < steps; ++i) cond[static_cast<size_t>(i)] = 1;
        for (size_t i = static_cast<size_t>(steps); i < uncond.size(); ++i) uncond[i] = 1;
        const int32_t rows[]{qrow, mrow};
        ggml_backend_tensor_set(graph.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(graph.zeros, zero_ids.data(), 0, zero_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(graph.mask, cond.data(), 0, cond.size() * sizeof(float));
        ggml_backend_tensor_set(graph.inverse_mask, uncond.data(), 0, uncond.size() * sizeof(float));
        ggml_backend_tensor_set(graph.qwen, style.data(), 0, style.size() * sizeof(float));
        ggml_backend_tensor_set(graph.muq, zero_muq.data(), 0, zero_muq.size() * sizeof(float));
        ggml_backend_tensor_set(graph.rows, rows, 0, sizeof(rows));
        if (core::compute_backend_graph(backend_, graph.graph, graph.plan.plan, "mulacover.prompt") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("MuLaCover prompt graph execution failed");
        PromptEmbeddings result{core::read_tensor_f32(graph.output), batch, steps, dims_};
        graph.release_workspace();
        debug::timing_log_scalar("mulacover.prompt.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        return result;
    }

private:
    ggml_backend_t backend_;
    ggml_type compute_type_ = GGML_TYPE_F32;
    MuLaCoverGenerationConfig config_;
    tokenizers::LlamaBpeTokenizer tokenizer_;
    int64_t vocab_ = 0, dims_ = 0, qwen_dim_ = 0, muq_dim_ = 0;
    Value text_, uncond_;
    modules::LinearWeights qwen_weights_, muq_weights_;
    std::unique_ptr<Graph> graph_;
};

PromptEncoder::PromptEncoder(core::ExecutionContext & execution, const PromptWeights & weights,
    std::shared_ptr<const assets::TensorSource> source, const std::filesystem::path & tokenizer,
    MuLaCoverGenerationConfig config)
    : impl_(std::make_unique<Impl>(execution, std::move(source), tokenizer, std::move(config), weights)) {}

std::string PromptEncoder::normalize_tags(const std::string & tags) {
    auto points = unicode_cpts_from_utf8(tags);
    size_t begin = 0, end = points.size();
    while (begin < end && unicode_cpt_flags_from_cpt(points[begin]).is_whitespace) ++begin;
    while (end > begin && unicode_cpt_flags_from_cpt(points[end - 1]).is_whitespace) --end;
    std::string result;
    for (size_t i = begin; i < end; ++i) result += unicode_cpt_to_utf8(unicode_tolower(points[i]));
    if (result.size() >= 11 && result.compare(0, 5, "<tag>") == 0 &&
        result.compare(result.size() - 6, 6, "</tag>") == 0) {
        result = result.substr(5, result.size() - 11);
        points = unicode_cpts_from_utf8(result);
        begin = 0; end = points.size();
        while (begin < end && unicode_cpt_flags_from_cpt(points[begin]).is_whitespace) ++begin;
        while (end > begin && unicode_cpt_flags_from_cpt(points[end - 1]).is_whitespace) --end;
        result.clear();
        for (size_t i = begin; i < end; ++i) result += unicode_cpt_to_utf8(points[i]);
    }
    return result;
}

PromptEncoder::~PromptEncoder() = default;

PromptEmbeddings PromptEncoder::encode(
    const std::string & lyrics, const std::string & tags, const std::vector<float> & style, float cfg_scale) {
    return impl_->encode(lyrics, tags, style, cfg_scale);
}

}  // namespace engine::models::mulacover
