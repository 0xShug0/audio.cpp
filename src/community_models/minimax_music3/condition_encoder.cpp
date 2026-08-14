#include "engine/community_models/minimax_music3/condition_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

}  // namespace

struct MiniMaxMusic3ConditionEncoderRuntime::Impl {
    core::ExecutionContext & execution;
    MiniMaxMusic3Config config;
    core::BackendWeightStore store;
    core::TensorValue proj_weight;
    core::TensorValue proj_bias;
    std::vector<float> layer_weights;  // softmax(layer_weight_logits) * layer_scale
    std::shared_ptr<const assets::TensorSource> source;

    Impl(
        core::ExecutionContext & execution_context,
        std::shared_ptr<const assets::TensorSource> tensor_source,
        const MiniMaxMusic3Config & cfg,
        size_t weight_context_bytes)
        : execution(execution_context),
          config(cfg),
          store(execution.backend(), execution.backend_type(), "minimax_music3.condition_encoder", weight_context_bytes),
          source(std::move(tensor_source)) {
        proj_weight = store.load_tensor(
            *source,
            "proj.weight",
            assets::TensorStorageType::Native,
            {config.cond_out_dim, config.cond_hidden, 3});
        proj_bias = store.load_tensor(
            *source,
            "proj.bias",
            assets::TensorStorageType::F32,
            {config.cond_out_dim});
        const auto logits = source->require_f32("layer_weight_logits");
        const auto scale = source->require_f32("layer_scale");
        if (logits.size() != static_cast<size_t>(config.cond_layers) || scale.size() != 1) {
            throw std::runtime_error("MiniMax-Music3 condition encoder mixing weights have unexpected shape");
        }
        float max_logit = logits[0];
        for (const float value : logits) {
            max_logit = std::max(max_logit, value);
        }
        double denom = 0.0;
        layer_weights.resize(logits.size());
        for (size_t index = 0; index < logits.size(); ++index) {
            layer_weights[index] = std::exp(logits[index] - max_logit);
            denom += layer_weights[index];
        }
        for (float & value : layer_weights) {
            value = static_cast<float>(value / denom * scale[0]);
        }
        store.upload();
        source->release_storage();
    }

    std::vector<float> run_conv(const std::vector<float> & mixed, int64_t frames) {
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init({64 * 1024 * 1024, nullptr, true}));
        std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx(ggml_init({1 * 1024 * 1024, nullptr, true}));
        if (ctx == nullptr || input_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-Music3 condition encoder graph context");
        }
        core::ModuleBuildContext inputs{input_ctx.get(), "minimax_music3.condition_encoder.inputs", execution.backend_type()};
        auto input = core::make_tensor(
            inputs,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.cond_hidden, frames}));
        ggml_set_input(input.tensor);

        core::ModuleBuildContext build{ctx.get(), "minimax_music3.condition_encoder", execution.backend_type()};
        auto projected = modules::Conv1dModule({config.cond_hidden, config.cond_out_dim, 3, 1, 1, 1, true})
                             .build(build, input, {proj_weight, proj_bias});
        auto * output = core::ensure_backend_addressable_layout(build, projected).tensor;

        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 4096, false);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), execution.backend());
        ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (input_buffer == nullptr || gallocr == nullptr ||
            !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
            }
            if (input_buffer != nullptr) {
                ggml_backend_buffer_free(input_buffer);
            }
            throw std::runtime_error("failed to allocate MiniMax-Music3 condition encoder graph");
        }
        core::HostGraphPlan plan;
        core::prepare_host_graph_plan(execution, graph, plan);
        core::write_tensor_f32(input, mixed);
        const ggml_status status = core::compute_graph(execution, graph, plan, "minimax_music3.condition_encoder");
        std::vector<float> projected_rows;
        if (status == GGML_STATUS_SUCCESS) {
            projected_rows = core::read_tensor_f32(output);
        }
        plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph);
        ggml_gallocr_free(gallocr);
        ggml_backend_buffer_free(input_buffer);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-Music3 condition encoder graph compute failed");
        }
        return projected_rows;
    }
};

MiniMaxMusic3ConditionEncoderRuntime::MiniMaxMusic3ConditionEncoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source,
    const MiniMaxMusic3Config & config,
    size_t weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("MiniMax-Music3 condition encoder tensor source is missing");
    }
    impl_ = std::make_unique<Impl>(execution, std::move(source), config, weight_context_bytes);
}

MiniMaxMusic3ConditionEncoderRuntime::~MiniMaxMusic3ConditionEncoderRuntime() = default;

int64_t MiniMaxMusic3ConditionEncoderRuntime::latent_length(int64_t frames) const {
    const auto & config = impl_->config;
    const double scale =
        static_cast<double>(config.cond_output_sampling_rate) / static_cast<double>(config.cond_input_sampling_rate) *
        static_cast<double>(config.cond_input_hop) / static_cast<double>(config.cond_output_hop);
    return std::max<int64_t>(1, static_cast<int64_t>(static_cast<double>(frames) * scale));
}

std::vector<float> MiniMaxMusic3ConditionEncoderRuntime::encode(
    const std::vector<float> & frame_hiddens,
    int64_t frames) {
    const auto & config = impl_->config;
    const int64_t layers = config.cond_layers;
    const int64_t hidden = config.cond_hidden;
    if (frames <= 0 || frame_hiddens.size() != static_cast<size_t>(frames * layers * hidden)) {
        throw std::runtime_error("MiniMax-Music3 condition encoder input size mismatch");
    }

    // Softmax-weighted mix of the per-frame hidden slots, scaled; layout [hidden, frames]
    // for the Conv1d graph.
    std::vector<float> mixed(static_cast<size_t>(hidden * frames), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = frame_hiddens.data() + frame * layers * hidden;
        for (int64_t layer = 0; layer < layers; ++layer) {
            const float weight = impl_->layer_weights[static_cast<size_t>(layer)];
            const float * slot = row + layer * hidden;
            for (int64_t channel = 0; channel < hidden; ++channel) {
                mixed[static_cast<size_t>(channel * frames + frame)] += weight * slot[channel];
            }
        }
    }

    const auto projected = impl_->run_conv(mixed, frames);  // [cond_out_dim, frames]
    if (projected.size() != static_cast<size_t>(config.cond_out_dim * frames)) {
        throw std::runtime_error("MiniMax-Music3 condition encoder projection size mismatch");
    }

    // Nearest-neighbor resample to the latent timeline; output row-major [latent, out_dim].
    const int64_t out_length = latent_length(frames);
    std::vector<float> condition(static_cast<size_t>(out_length * config.cond_out_dim));
    for (int64_t index = 0; index < out_length; ++index) {
        int64_t src = static_cast<int64_t>(static_cast<double>(index) * static_cast<double>(frames) /
                                           static_cast<double>(out_length));
        src = std::min(src, frames - 1);
        for (int64_t channel = 0; channel < config.cond_out_dim; ++channel) {
            condition[static_cast<size_t>(index * config.cond_out_dim + channel)] =
                projected[static_cast<size_t>(channel * frames + src)];
        }
    }
    return condition;
}

}  // namespace engine::models::minimax_music3
