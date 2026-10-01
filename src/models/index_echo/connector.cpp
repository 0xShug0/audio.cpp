#include "engine/models/index_echo/connector.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::index_echo {
namespace {

namespace binding = engine::modules::binding;
namespace modules = engine::modules;

struct ConnectorWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::LinearWeights projection;
    modules::LinearWeights w1;
    modules::LinearWeights w2;
    core::TensorValue beta;
    core::TensorValue alpha;
    bool projected = false;
};

ConnectorWeights load_weights(
    const IndexEchoAssets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type) {
    ConnectorWeights weights;
    weights.store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "index_echo.connector.weights", 4ull * 1024ull * 1024ull);
    const auto & source = *assets.connector_weights;
    weights.projected = source.has_tensor("proj.weight");
    if (weights.projected) {
        weights.projection = binding::linear_from_source(
            *weights.store, source, "proj", storage_type,
            assets.text_hidden_size, assets.audio_hidden_size, false);
    } else {
        if (assets.audio_hidden_size != assets.text_hidden_size) {
            throw std::runtime_error("Index-Echo residual connector requires matching audio and text widths");
        }
        weights.w1 = binding::linear_from_source(
            *weights.store, source, "w1", storage_type,
            assets.text_hidden_size, assets.audio_hidden_size, false);
        weights.w2 = binding::linear_from_source(
            *weights.store, source, "w2", storage_type,
            assets.text_hidden_size, assets.text_hidden_size, false);
        weights.beta = weights.store->make_f32(
            core::TensorShape::from_dims({1}), source.require_f32("beta", std::vector<int64_t>{}));
        const auto log_alpha = source.require_f32("log_alpha", std::vector<int64_t>{});
        weights.alpha = weights.store->make_f32(
            core::TensorShape::from_dims({1}), {std::exp(log_alpha.at(0))});
    }
    weights.store->upload();
    return weights;
}

struct ConnectorGraph {
    int64_t tokens = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan host_plan;
    core::TensorValue input;
    core::TensorValue output;

    ~ConnectorGraph() {
        host_plan.reset();
        if (backend != nullptr && graph != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

}  // namespace

class IndexEchoAudioConnectorRuntime::Impl {
public:
    Impl(std::shared_ptr<const IndexEchoAssets> model_assets,
         core::ExecutionContext & model_execution,
         assets::TensorStorageType storage_type)
        : assets_(std::move(model_assets)), execution_(&model_execution) {
        if (assets_ == nullptr || assets_->connector_weights == nullptr) {
            throw std::runtime_error("Index-Echo connector weights are missing");
        }
        weights_ = load_weights(*assets_, *execution_, storage_type);
    }

    IndexEchoConnectedAudio connect(const qwen3_asr::Qwen3ASRAudioEmbeddings & encoded) {
        if (encoded.tokens <= 0 || encoded.hidden_size != assets_->audio_hidden_size ||
            encoded.values.size() != static_cast<size_t>(encoded.tokens * encoded.hidden_size)) {
            throw std::runtime_error("Index-Echo connector input shape does not match audio encoder output");
        }
        ensure_graph(encoded.tokens);
        core::write_tensor_f32(graph_->input, encoded.values);
        if (core::compute_graph(*execution_, graph_->graph, graph_->host_plan, "Index-Echo connector") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Index-Echo connector execution failed");
        }
        return {core::read_tensor_f32(graph_->output.tensor), encoded.tokens, assets_->text_hidden_size};
    }

private:
    void ensure_graph(int64_t tokens) {
        if (graph_ != nullptr && graph_->tokens == tokens && graph_->backend == execution_->backend()) {
            return;
        }
        auto next = std::make_unique<ConnectorGraph>();
        next->tokens = tokens;
        next->backend = execution_->backend();
        ggml_init_params params{4ull * 1024ull * 1024ull, nullptr, true};
        next->ggml = ggml_init(params);
        if (next->ggml == nullptr) {
            throw std::runtime_error("Index-Echo connector graph context allocation failed");
        }
        core::ModuleBuildContext context{next->ggml, "index_echo.connector", execution_->backend_type()};
        next->input = core::make_tensor(
            context, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, tokens, assets_->audio_hidden_size}));
        ggml_set_input(next->input.tensor);
        if (weights_.projected) {
            next->output = modules::LinearModule({assets_->audio_hidden_size, assets_->text_hidden_size, false})
                               .build(context, next->input, weights_.projection);
        } else {
            auto hidden = modules::LinearModule({assets_->audio_hidden_size, assets_->text_hidden_size, false})
                              .build(context, next->input, weights_.w1);
            hidden = modules::GeluModule().build(context, hidden);
            hidden = modules::LinearModule({assets_->text_hidden_size, assets_->text_hidden_size, false})
                         .build(context, hidden, weights_.w2);
            const auto beta = modules::RepeatModule({hidden.shape}).build(
                context,
                core::reshape_tensor(context, weights_.beta, core::TensorShape::from_dims({1, 1, 1})));
            hidden = modules::MulModule().build(context, hidden, beta);
            next->output = modules::AddModule().build(context, next->input, hidden);
            const auto alpha = modules::RepeatModule({next->output.shape}).build(
                context,
                core::reshape_tensor(context, weights_.alpha, core::TensorShape::from_dims({1, 1, 1})));
            next->output = modules::MulModule().build(context, next->output, alpha);
        }
        ggml_set_output(next->output.tensor);
        next->graph = ggml_new_graph_custom(next->ggml, 512, false);
        ggml_build_forward_expand(next->graph, next->output.tensor);
        core::validate_backend_graph_supported(next->backend, next->graph, "Index-Echo connector");
        next->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(next->backend));
        if (next->allocator == nullptr ||
            !ggml_gallocr_reserve(next->allocator, next->graph) ||
            !ggml_gallocr_alloc_graph(next->allocator, next->graph)) {
            throw std::runtime_error("Index-Echo connector graph allocation failed");
        }
        core::prepare_host_graph_plan(*execution_, next->graph, next->host_plan);
        graph_ = std::move(next);
    }

    std::shared_ptr<const IndexEchoAssets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    ConnectorWeights weights_;
    std::unique_ptr<ConnectorGraph> graph_;
};

IndexEchoAudioConnectorRuntime::IndexEchoAudioConnectorRuntime(
    std::shared_ptr<const IndexEchoAssets> model_assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(model_assets), execution, storage_type)) {}

IndexEchoAudioConnectorRuntime::~IndexEchoAudioConnectorRuntime() = default;

IndexEchoConnectedAudio IndexEchoAudioConnectorRuntime::connect(
    const qwen3_asr::Qwen3ASRAudioEmbeddings & input) {
    return impl_->connect(input);
}

}  // namespace engine::models::index_echo
