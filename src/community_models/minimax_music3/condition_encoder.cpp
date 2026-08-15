#include "engine/community_models/minimax_music3/condition_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

std::vector<float> scaled_softmax_weights(
    const assets::TensorSource & source,
    const MiniMaxMusic3ConditionConfig & config) {
    auto logits = source.require_f32("layer_weight_logits", {config.condition_layers});
    const auto scale = source.require_f32("layer_scale", {1}).front();
    const float max_value = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float & value : logits) {
        value = std::exp(value - max_value);
        sum += value;
    }
    if (!(sum > 0.0)) {
        throw std::runtime_error("MiniMax Music 3 condition layer weights are invalid");
    }
    for (float & value : logits) {
        value = static_cast<float>(static_cast<double>(value) / sum) * scale;
    }
    return logits;
}

MiniMaxMusic3ConditionWeights load_condition_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.condition;
    const auto & source = *assets.condition_encoder_weights;
    MiniMaxMusic3ConditionWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.condition.weights",
        weight_context_bytes);
    out.layer_weights = scaled_softmax_weights(source, config);
    out.projection = binding::conv1d_from_source(
        *out.store,
        source,
        "proj",
        conv_safe_storage_type(source, "proj", storage_type),
        config.out_dim,
        config.condition_hidden_dim,
        3,
        true);
    out.store->upload();
    assets.condition_encoder_weights->release_storage();
    return out;
}

}  // namespace

struct MiniMaxMusic3ConditionEncoderRuntime::Impl {
    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::ExecutionContext & input_execution,
        size_t input_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          execution(input_execution),
          graph_arena_bytes(input_graph_arena_bytes),
          weights(load_condition_weights(*assets, execution, weight_context_bytes, storage_type)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 condition encoder requires assets");
        }
    }

    ~Impl() {
        release_runtime_graphs();
    }

    void release_runtime_graphs() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph);
        }
        graph = nullptr;
        input = {};
        output = nullptr;
        gallocr.reset();
        ggml.reset();
        plan.reset();
        frames = 0;
        condition_frames = 0;
    }

    void ensure_graph(int64_t input_frames, int64_t output_frames) {
        if (graph != nullptr && frames == input_frames && condition_frames == output_frames) {
            return;
        }
        release_runtime_graphs();
        const auto & config = assets->config.condition;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ggml.reset(ggml_init(params));
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 condition graph context");
        }
        core::ModuleBuildContext ctx{ggml.get(), "minimax_music3.condition", execution.backend_type()};
        input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.condition_hidden_dim, input_frames}));
        ggml_set_input(input.tensor);
        auto projected = modules::Conv1dModule({
            config.condition_hidden_dim,
            config.out_dim,
            3,
            1,
            1,
            1,
            true,
        }).build(ctx, input, weights.projection);
        auto interpolated = modules::Interpolate1dModule({
            output_frames,
            modules::Interpolate1dMode::Nearest,
        }).build(ctx, projected);
        output = interpolated.tensor;
        graph = ggml_new_graph_custom(ggml.get(), 65536, false);
        ggml_build_forward_expand(graph, output);
        gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr.get(), graph) ||
            !ggml_gallocr_alloc_graph(gallocr.get(), graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 condition graph");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
        frames = input_frames;
        condition_frames = output_frames;
    }

    std::vector<float> encode(const std::vector<float> & frame_hiddens, int64_t input_frames, int64_t & out_frames) {
        const auto & config = assets->config.condition;
        const int64_t expected = input_frames * config.condition_layers * config.condition_hidden_dim;
        if (static_cast<int64_t>(frame_hiddens.size()) != expected) {
            throw std::runtime_error("MiniMax Music 3 condition frame hidden shape mismatch");
        }
        out_frames = static_cast<int64_t>(
            static_cast<double>(input_frames) * static_cast<double>(config.output_sample_rate) /
            static_cast<double>(config.input_sample_rate) * static_cast<double>(config.input_hop_length) /
            static_cast<double>(config.output_hop_length));
        if (out_frames <= 0) {
            throw std::runtime_error("MiniMax Music 3 condition encoder produced non-positive frame count");
        }
        std::vector<float> projected_input(static_cast<size_t>(config.condition_hidden_dim * input_frames), 0.0F);
        for (int64_t frame = 0; frame < input_frames; ++frame) {
            for (int64_t layer = 0; layer < config.condition_layers; ++layer) {
                const float weight = weights.layer_weights[static_cast<size_t>(layer)];
                const size_t src_base = static_cast<size_t>((frame * config.condition_layers + layer) * config.condition_hidden_dim);
                for (int64_t channel = 0; channel < config.condition_hidden_dim; ++channel) {
                    projected_input[static_cast<size_t>(channel * input_frames + frame)] +=
                        frame_hiddens[src_base + static_cast<size_t>(channel)] * weight;
                }
            }
        }
        ensure_graph(input_frames, out_frames);
        core::write_tensor_f32(input, projected_input);
        if (core::compute_graph(execution, graph, plan, "minimax_music3.condition") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 condition graph compute failed");
        }
        auto bct = core::read_tensor_f32(output);
        std::vector<float> btc(static_cast<size_t>(out_frames * config.out_dim));
        for (int64_t channel = 0; channel < config.out_dim; ++channel) {
            for (int64_t frame = 0; frame < out_frames; ++frame) {
                btc[static_cast<size_t>(frame * config.out_dim + channel)] =
                    bct[static_cast<size_t>(channel * out_frames + frame)];
            }
        }
        return btc;
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    MiniMaxMusic3ConditionWeights weights;
    int64_t frames = 0;
    int64_t condition_frames = 0;
    std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    core::HostGraphPlan plan;
    core::TensorValue input;
    ggml_tensor * output = nullptr;
};

MiniMaxMusic3ConditionEncoderRuntime::MiniMaxMusic3ConditionEncoderRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3ConditionEncoderRuntime::~MiniMaxMusic3ConditionEncoderRuntime() = default;

std::vector<float> MiniMaxMusic3ConditionEncoderRuntime::encode(
    const std::vector<float> & frame_hiddens,
    int64_t frames,
    int64_t & condition_frames) {
    return impl_->encode(frame_hiddens, frames, condition_frames);
}

void MiniMaxMusic3ConditionEncoderRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
