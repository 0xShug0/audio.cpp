#include "engine/models/midashenglm_gen/prompt_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::midashenglm_gen {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::shared_ptr<const MiDashengLmGenPromptEncoderWeights> load_weights(
    const MiDashengLmGenAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen prompt encoder requires tensor source");
    }
    auto weights = std::make_shared<MiDashengLmGenPromptEncoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "midashenglm_gen.prompt_encoder.weights",
        weight_context_bytes);
    weights->token_embedding = weights->store->load_tensor(
        *assets.weights,
        "model.mdl_model.decoder.model.embed_tokens.weight",
        storage_type,
        {assets.config.vocab_size, assets.config.hidden_size});
    weights->store->upload();
    return weights;
}

}  // namespace

class MiDashengLmGenPromptEncoderRuntime::EncodeGraph {
public:
    EncodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const MiDashengLmGenPromptEncoderWeights> weights,
        MiDashengLmGenConfig config,
        int64_t batch,
        int64_t tokens,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          batch_(batch),
          tokens_(tokens) {
        if (weights_ == nullptr) {
            throw std::runtime_error("MiDashengLM-Gen prompt encoder graph requires weights");
        }
        if (batch_ <= 0 || tokens_ <= 0) {
            throw std::runtime_error("MiDashengLM-Gen prompt encoder graph requires positive shape");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen prompt encoder graph context");
        }
        ggml_init_params input_params{4ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen prompt encoder input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "midashenglm_gen.prompt_encoder", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "midashenglm_gen.prompt_encoder.inputs", execution_.backend_type()};
        token_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch_, tokens_})).tensor;
        ggml_set_input(token_ids_);
        auto input = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({batch_, tokens_}), GGML_TYPE_I32);
        auto out = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size})
                       .build(ctx, input, weights_->token_embedding);
        output_ = core::ensure_backend_addressable_layout(ctx, out).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(
            ctx_.get(),
            static_cast<size_t>(std::max<int64_t>(4096, batch_ * tokens_ * 16 + 1024)),
            false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiDashengLM-Gen prompt encoder input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate MiDashengLM-Gen prompt encoder graph");
        }
        engine::debug::timing_log_scalar(
            "midashenglm_gen.prompt_encoder.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~EncodeGraph() {
        clear_graph();
    }

    int64_t batch() const noexcept {
        return batch_;
    }

    int64_t tokens() const noexcept {
        return tokens_;
    }

    MiDashengLmGenPromptEncoderOutput run(const MiDashengLmGenPromptBatch & input) {
        if (input.batch != batch_ || input.tokens != tokens_) {
            throw std::runtime_error("MiDashengLM-Gen prompt encoder input shape mismatch");
        }
        if (static_cast<int64_t>(input.token_ids.size()) != batch_ * tokens_) {
            throw std::runtime_error("MiDashengLM-Gen prompt encoder token count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, input.token_ids.data(), 0, input.token_ids.size() * sizeof(int32_t));
        engine::debug::timing_log_scalar(
            "midashenglm_gen.prompt_encoder.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        engine::debug::timing_log_scalar(
            "midashenglm_gen.prompt_encoder.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiDashengLM-Gen prompt encoder graph compute failed");
        }
        MiDashengLmGenPromptEncoderOutput out;
        out.batch = batch_;
        out.tokens = tokens_;
        out.hidden = config_.hidden_size;
        out.attention_mask = input.attention_mask;
        timing_start = Clock::now();
        core::read_tensor_float_into(output_, out.embeddings);
        engine::debug::timing_log_scalar(
            "midashenglm_gen.prompt_encoder.output_read_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const MiDashengLmGenPromptEncoderWeights> weights_;
    MiDashengLmGenConfig config_;
    int64_t batch_ = 0;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

MiDashengLmGenPromptEncoderRuntime::MiDashengLmGenPromptEncoderRuntime(
    std::shared_ptr<const MiDashengLmGenAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen prompt encoder runtime requires assets");
    }
    weights_ = load_weights(*assets_, execution.backend(), execution.backend_type(), weight_context_bytes, storage_type);
}

MiDashengLmGenPromptEncoderRuntime::~MiDashengLmGenPromptEncoderRuntime() = default;

void MiDashengLmGenPromptEncoderRuntime::prepare(int64_t batch, int64_t tokens) {
    if (graph_ != nullptr && graph_->batch() == batch && graph_->tokens() == tokens) {
        return;
    }
    graph_.reset();
    graph_ = std::make_unique<EncodeGraph>(*execution_, weights_, assets_->config, batch, tokens, graph_arena_bytes_);
}

MiDashengLmGenPromptEncoderOutput MiDashengLmGenPromptEncoderRuntime::encode(
    const MiDashengLmGenPromptBatch & input) {
    if (graph_ == nullptr || graph_->batch() != input.batch || graph_->tokens() != input.tokens) {
        throw std::runtime_error("MiDashengLM-Gen prompt encoder graph was not prepared for this input shape");
    }
    return graph_->run(input);
}

void MiDashengLmGenPromptEncoderRuntime::release_graphs() {
    graph_.reset();
}

}  // namespace engine::models::midashenglm_gen
