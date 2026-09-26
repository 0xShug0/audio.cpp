#include "engine/models/gigaam_asr/model.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/framework/runtime/cache_slots.h"

#include <ggml-alloc.h>

#include <numeric>
#include <stdexcept>

namespace engine::models::gigaam_asr {
namespace {

using core::TensorShape;
using core::TensorValue;

struct Graph {
    core::ExecutionContext & execution;
    std::string name;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan host_plan;

    Graph(core::ExecutionContext & execution, size_t nodes, std::string name)
        : execution(execution), name(std::move(name)) {
        const size_t bytes = ggml_graph_overhead_custom(nodes, false) + nodes * ggml_tensor_overhead();
        context.reset(ggml_init({bytes, nullptr, true}));
        if (!context) {
            throw std::runtime_error("GigaAM graph descriptor allocation failed");
        }
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    }
    ~Graph() {
        host_plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph, true);
        ggml_gallocr_free(allocator);
    }
    core::ModuleBuildContext build_context() {
        return {context.get(), "gigaam_asr", execution.backend_type()};
    }
    void output(const TensorValue & value) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph, value.tensor);
    }
    void allocate() {
        if (execution.backend_type() == core::BackendType::Cuda) {
            runtime::optimize_graph(*graph, runtime::GraphOptimizationBackend::Gpu);
        } else if (execution.backend_type() == core::BackendType::Cpu) {
            auto options = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Cpu);
            // Keep views in the allocation graph so their buffer lifetimes remain visible.
            options.elide_metadata_only_ops = false;
            runtime::optimize_graph(*graph, options);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "GigaAM");
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("GigaAM graph buffer allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, host_plan);
        debug::timing_log_context_reservation(name, context.get());
        debug::timing_log_scalar(name + ".buffer_bytes", ggml_gallocr_get_buffer_size(allocator, 0));
    }
    void compute() {
        if (core::compute_graph(execution, graph, host_plan, "GigaAM") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GigaAM graph execution failed");
        }
    }
};

}  // namespace

struct GigaAMRuntime::Impl {
    const GigaAMAssets & assets;
    const GigaAMWeights & weights;
    core::ExecutionContext & execution;
    struct EncoderGraph {
        std::unique_ptr<Graph> graph;
        TensorValue features, positions, encoded;
    };
    runtime::CacheSlots<int64_t, std::unique_ptr<EncoderGraph>> encoders{2};
    std::unique_ptr<Graph> predictor, joint;
    TensorValue label, has_label, hidden, cell, next_hidden, next_cell, prediction;
    TensorValue joint_frame, joint_prediction, next_label;

    Impl(const GigaAMAssets & assets, const GigaAMWeights & weights, core::ExecutionContext & execution)
        : assets(assets), weights(weights), execution(execution) {
        if (assets.rnnt) {
            build_decoder();
        }
    }

    EncoderGraph & build_encoder(int64_t frames) {
        if (auto * cached = encoders.find(frames)) {
            return **cached;
        }
        // Evict before allocation so a shape change never holds three graph buffers.
        encoders.set_capacity(1);
        encoders.set_capacity(2);
        auto state = std::make_unique<EncoderGraph>();
        auto & encoder = state->graph;
        auto & features = state->features;
        auto & positions = state->positions;
        auto & encoded = state->encoded;
        encoder = std::make_unique<Graph>(execution, 512 + 256 * weights.encoder.layers.size(), "gigaam_asr.encoder");
        auto ctx = encoder->build_context();
        features = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, assets.encoder.features, frames}));
        ggml_set_input(features.tensor);
        const int64_t steps = (frames + 3) / 4;
        positions = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({steps}));
        ggml_set_input(positions.tensor);
        ggml_set_output(positions.tensor);
        auto config = assets.encoder;
        auto x = build_conformer_encoder(ctx, features, positions, config, weights.encoder);
        if (assets.rnnt) {
            encoded = modules::LinearModule({config.hidden_size, assets.joint_hidden, true})
                .build(ctx, x, weights.joint_encoder);
        } else {
            x = modules::LinearModule({config.hidden_size, assets.classes, true}).build(ctx, x, weights.ctc);
            x = core::reshape_tensor(ctx, x, TensorShape::from_dims({steps, assets.classes}));
            encoded = core::wrap_tensor(ggml_argmax(ctx.ggml, x.tensor), TensorShape::from_dims({steps}), GGML_TYPE_I32);
        }
        encoder->output(encoded);
        encoder->allocate();
        std::vector<int32_t> pos(static_cast<size_t>(steps));
        std::iota(pos.begin(), pos.end(), 0);
        core::write_tensor_i32(positions, pos);
        auto & result = *state;
        encoders.put(frames, std::move(state));
        return result;
    }

    void build_decoder() {
        const auto hidden_size = assets.predictor_hidden;
        predictor = std::make_unique<Graph>(execution, 512, "gigaam_asr.predictor");
        auto ctx = predictor->build_context();
        label = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        has_label = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1}));
        hidden = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, hidden_size}));
        cell = core::make_tensor(ctx, GGML_TYPE_F32, hidden.shape);
        for (auto t : {label, has_label, hidden, cell}) {
            ggml_set_input(t.tensor);
        }
        ggml_set_output(has_label.tensor);
        auto x = modules::EmbeddingModule({assets.classes, hidden_size}).build(ctx, label, weights.embedding);
        x = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, has_label));
        const auto state = modules::LSTMCellModule({hidden_size, hidden_size}).build(ctx, x, hidden, cell, weights.predictor);
        next_hidden = state.hidden;
        next_cell = state.cell;
        prediction = modules::LinearModule({hidden_size, assets.joint_hidden, true})
            .build(ctx, state.hidden, weights.joint_predictor);
        predictor->output(prediction);
        predictor->output(next_hidden);
        predictor->output(next_cell);
        predictor->allocate();

        joint = std::make_unique<Graph>(execution, 128, "gigaam_asr.joint");
        ctx = joint->build_context();
        joint_frame = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, assets.joint_hidden}));
        joint_prediction = core::make_tensor(ctx, GGML_TYPE_F32, joint_frame.shape);
        ggml_set_input(joint_frame.tensor);
        ggml_set_input(joint_prediction.tensor);
        // Both inputs survive multiple joint evaluations between updates.
        ggml_set_output(joint_frame.tensor);
        ggml_set_output(joint_prediction.tensor);
        x = modules::AddModule().build(ctx, joint_frame, joint_prediction);
        x = modules::ReluModule().build(ctx, x);
        x = modules::LinearModule({assets.joint_hidden, assets.classes, true}).build(ctx, x, weights.joint_output);
        next_label = core::wrap_tensor(ggml_argmax(ctx.ggml, x.tensor), TensorShape::from_dims({1}), GGML_TYPE_I32);
        joint->output(next_label);
        joint->allocate();
    }

    GigaAMDecodedTokens infer(const audio::AudioTensor & mel, bool timestamps) {
        const auto started = std::chrono::steady_clock::now();
        const int64_t frames = mel.shape.at(2);
        auto & encoder = build_encoder(frames);
        const auto & encoded = encoder.encoded;
        core::write_tensor_f32(encoder.features, mel.values);
        const auto encoder_start = std::chrono::steady_clock::now();
        encoder.graph->compute();
        const int32_t blank = static_cast<int32_t>(assets.classes - 1);
        GigaAMDecodedTokens result;
        result.encoder_ms = debug::elapsed_ms(encoder_start);
        result.encoder_frames = (frames + 3) / 4;
        auto & ids = result.ids;
        if (!assets.rnnt) {
            const auto labels = core::read_tensor_i32(encoded.tensor);
            int32_t previous = blank;
            for (size_t t = 0; t < labels.size(); ++t) {
                const auto token = labels[t];
                if (token != blank && token != previous) {
                    ids.push_back(token);
                    if (timestamps) result.frames.push_back(static_cast<int64_t>(t));
                }
                previous = token;
            }
        } else {
            const auto projected = core::read_tensor_f32(encoded.tensor);
            const auto steps = encoded.shape.dims[1];
            const std::vector<float> zeros(static_cast<size_t>(assets.predictor_hidden), 0.0f);
            core::write_tensor_f32(hidden, zeros);
            core::write_tensor_f32(cell, zeros);
            core::write_tensor_i32(label, &blank, 1);
            const float zero = 0.0f;
            core::write_tensor_f32(has_label, &zero, 1);
            predictor->compute();
            ggml_backend_tensor_copy(prediction.tensor, joint_prediction.tensor);
            const float one = 1.0f;
            core::write_tensor_f32(has_label, &one, 1);
            for (int64_t t = 0; t < steps; ++t) {
                core::write_tensor_f32(joint_frame, projected.data() + t * assets.joint_hidden,
                    static_cast<size_t>(assets.joint_hidden));
                for (int64_t symbol = 0; symbol < assets.max_symbols_per_step; ++symbol) {
                    joint->compute();
                    int32_t token;
                    ggml_backend_tensor_get(next_label.tensor, &token, 0, sizeof(token));
                    if (token == blank) {
                        break;
                    }
                    ids.push_back(token);
                    if (timestamps) result.frames.push_back(t);
                    // Blank transitions leave the predictor state untouched.
                    ggml_backend_tensor_copy(next_hidden.tensor, hidden.tensor);
                    ggml_backend_tensor_copy(next_cell.tensor, cell.tensor);
                    core::write_tensor_i32(label, &token, 1);
                    predictor->compute();
                    ggml_backend_tensor_copy(prediction.tensor, joint_prediction.tensor);
                }
            }
        }
        result.inference_ms = debug::elapsed_ms(started);
        return result;
    }

};

GigaAMRuntime::GigaAMRuntime(
    const GigaAMAssets & assets, const GigaAMWeights & weights, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(assets, weights, execution)) {}

GigaAMRuntime::~GigaAMRuntime() = default;

GigaAMDecodedTokens GigaAMRuntime::transcribe(const audio::AudioTensor & features, bool timestamps) {
    return impl_->infer(features, timestamps);
}

}  // namespace engine::models::gigaam_asr
