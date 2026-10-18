#include "engine/models/kugelaudio/diffusion.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/debug/trace.h"

#include <ggml-alloc.h>
#include <cmath>
#include <stdexcept>

namespace engine::models::kugelaudio {

ConnectorWeights load_connector_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    assets::TensorStorageType storage, int64_t latent_size, int64_t hidden_size) {
    namespace binding = modules::binding;
    const std::string prefix = "model.acoustic_connector.";
    return {
        binding::linear_from_source(store, source, prefix + "fc1", storage, hidden_size, latent_size, true),
        binding::norm_weight_from_source(store, source, prefix + "norm", hidden_size),
        binding::linear_from_source(store, source, prefix + "fc2", storage, hidden_size, hidden_size, true),
    };
}

DiffusionWeights load_diffusion_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    assets::TensorStorageType storage, const DiffusionConfig & config) {
    namespace binding = modules::binding;
    const std::string prefix = "model.prediction_head.";
    const auto hidden = config.hidden_size;
    DiffusionWeights weights;
    weights.input = binding::linear_from_source(store, source, prefix + "noisy_images_proj",
                                                storage, hidden, config.latent_size, false);
    weights.condition = binding::linear_from_source(store, source, prefix + "cond_proj",
                                                    storage, hidden, hidden, false);
    weights.time1 = binding::linear_from_source(store, source, prefix + "t_embedder.mlp.0",
                                                storage, hidden, 256, false);
    weights.time2 = binding::linear_from_source(store, source, prefix + "t_embedder.mlp.2",
                                                storage, hidden, hidden, false);
    for (int64_t i = 0; i < config.layers; ++i) {
        const auto layer = prefix + "layers." + std::to_string(i) + ".";
        weights.layers.push_back({
            binding::norm_weight_from_source(store, source, layer + "norm", hidden),
            binding::linear_from_source(store, source, layer + "adaLN_modulation.1", storage, 3 * hidden, hidden, false),
            binding::linear_from_source(store, source, layer + "ffn.gate_proj", storage, config.intermediate_size, hidden, false),
            binding::linear_from_source(store, source, layer + "ffn.up_proj", storage, config.intermediate_size, hidden, false),
            binding::linear_from_source(store, source, layer + "ffn.down_proj", storage, hidden, config.intermediate_size, false),
        });
    }
    weights.final_modulation = binding::linear_from_source(store, source,
        prefix + "final_layer.adaLN_modulation.1", storage, 2 * hidden, hidden, false);
    weights.output = binding::linear_from_source(store, source, prefix + "final_layer.linear",
                                                 storage, config.latent_size, hidden, false);
    weights.ones = store.make_f32(core::TensorShape::from_dims({1, hidden}),
                                  std::vector<float>(static_cast<size_t>(hidden), 1.0F));
    return weights;
}

core::TensorValue build_connector(
    core::ModuleBuildContext & ctx, const core::TensorValue & input,
    const ConnectorWeights & weights, int64_t hidden_size) {
    auto x = modules::LinearModule({input.shape.dims[input.shape.rank - 1], hidden_size, true})
                 .build(ctx, input, weights.fc1);
    x = modules::RMSNormModule({hidden_size, 1e-6F, true, false}).build(ctx, x, weights.norm);
    return modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, x, weights.fc2);
}

namespace {

std::vector<core::TensorValue> build_modulations(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & condition, const core::TensorValue & time_features,
    const DiffusionWeights & weights, const DiffusionConfig & config) {
    const auto hidden = config.hidden_size;
    auto time = modules::LinearModule({256, hidden, false}).build(ctx, time_features, weights.time1);
    time = modules::SiluModule{}.build(ctx, time);
    time = modules::LinearModule({hidden, hidden, false}).build(ctx, time, weights.time2);
    auto conditioning = modules::LinearModule({hidden, hidden, false})
                            .build(ctx, condition, weights.condition);
    conditioning = modules::AddModule{}.build(ctx, conditioning, time);
    conditioning = modules::SiluModule{}.build(ctx, conditioning);
    std::vector<core::TensorValue> result;
    for (const auto & layer : weights.layers) {
        result.push_back(modules::LinearModule({hidden, 3 * hidden, false})
            .build(ctx, conditioning, layer.modulation));
    }
    result.push_back(modules::LinearModule({hidden, 2 * hidden, false})
        .build(ctx, conditioning, weights.final_modulation));
    return result;
}

core::TensorValue build_modulated_head(
    core::ModuleBuildContext & ctx, const core::TensorValue & latent,
    const std::vector<core::TensorValue> & modulations,
    const DiffusionWeights & weights, const DiffusionConfig & config) {
    const auto hidden = config.hidden_size;
    auto x = modules::LinearModule({config.latent_size, hidden, false})
                 .build(ctx, latent, weights.input);
    const auto ones = modules::RepeatModule({x.shape}).build(ctx, weights.ones);
    const int axis = static_cast<int>(x.shape.rank) - 1;

    for (size_t i = 0; i < weights.layers.size(); ++i) {
        const auto & layer = weights.layers[i];
        const auto & modulation = modulations[i];
        auto shift = modules::SliceModule({axis, 0, hidden}).build(ctx, modulation);
        auto scale = modules::SliceModule({axis, hidden, hidden}).build(ctx, modulation);
        auto gate = modules::SliceModule({axis, 2 * hidden, hidden}).build(ctx, modulation);
        auto normalized = modules::RMSNormModule({hidden, config.norm_eps, true, false})
                              .build(ctx, x, layer.norm);
        scale = modules::AddModule{}.build(ctx, scale, ones);
        auto modulated = modules::MulModule{}.build(ctx, normalized, scale);
        modulated = modules::AddModule{}.build(ctx, modulated, shift);
        auto ff_gate = modules::LinearModule({hidden, config.intermediate_size, false})
                           .build(ctx, modulated, layer.gate);
        ff_gate = modules::SiluModule{}.build(ctx, ff_gate);
        auto ff_up = modules::LinearModule({hidden, config.intermediate_size, false})
                         .build(ctx, modulated, layer.up);
        auto ff = modules::MulModule{}.build(ctx, ff_gate, ff_up);
        ff = modules::LinearModule({config.intermediate_size, hidden, false}).build(ctx, ff, layer.down);
        ff = modules::MulModule{}.build(ctx, ff, gate);
        x = modules::AddModule{}.build(ctx, x, ff);
    }

    const auto & modulation = modulations.back();
    auto shift = modules::SliceModule({axis, 0, hidden}).build(ctx, modulation);
    auto scale = modules::SliceModule({axis, hidden, hidden}).build(ctx, modulation);
    x = modules::RMSNormModule({hidden, config.norm_eps, false, false}).build(ctx, x, {});
    scale = modules::AddModule{}.build(ctx, scale, ones);
    x = modules::MulModule{}.build(ctx, x, scale);
    x = modules::AddModule{}.build(ctx, x, shift);
    return modules::LinearModule({hidden, config.latent_size, false}).build(ctx, x, weights.output);
}

}  // namespace

core::TensorValue build_diffusion_head(
    core::ModuleBuildContext & ctx, const core::TensorValue & latent,
    const core::TensorValue & condition, const core::TensorValue & time_features,
    const DiffusionWeights & weights, const DiffusionConfig & config) {
    return build_modulated_head(ctx, latent,
        build_modulations(ctx, condition, time_features, weights, config), weights, config);
}

struct DiffusionHead::Impl {
    core::ExecutionContext & execution;
    DiffusionConfig config;
    core::BackendWeightStore store;
    DiffusionWeights weights;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    core::HostGraphPlan plan;
    core::HostGraphPlan conditioning_plan;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> conditioning_allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * conditioning_graph = nullptr;
    core::TensorValue input, condition, time, output, indices;
    std::vector<core::TensorValue> modulation_outputs, modulation_inputs;
    int64_t batch = 0;
    int64_t steps = 0;

    Impl(const assets::TensorSource & source, core::ExecutionContext & execution_,
         assets::TensorStorageType storage, DiffusionConfig config_)
        : execution(execution_), config(config_),
          store(execution.backend(), execution.backend_type(), "kugelaudio.diffusion.weights", 4 * 1024 * 1024),
          weights(load_diffusion_weights(store, source, storage, config)) {
        store.upload();
    }

    ~Impl() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        if (conditioning_graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), conditioning_graph, true);
        }
    }

    void prepare(int64_t requested_batch, int64_t requested_steps) {
        if (batch == requested_batch && steps == requested_steps) {
            return;
        }
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        graph = nullptr;
        if (conditioning_graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), conditioning_graph, true);
        }
        conditioning_graph = nullptr;
        plan.reset();
        conditioning_plan.reset();
        allocator.reset();
        conditioning_allocator.reset();
        context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        core::ModuleBuildContext ctx{context.get(), "kugelaudio.diffusion", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({requested_batch, config.latent_size}));
        const auto rows = requested_batch * requested_steps;
        condition = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows, config.hidden_size}));
        time = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows, 256}));
        indices = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({requested_batch}));
        ggml_set_input(input.tensor);
        ggml_set_input(condition.tensor);
        ggml_set_input(time.tensor);
        ggml_set_input(indices.tensor);
        modulation_outputs = build_modulations(ctx, condition, time, weights, config);
        conditioning_graph = ggml_new_graph_custom(context.get(), 512, false);
        for (const auto & modulation : modulation_outputs) {
            ggml_set_output(modulation.tensor);
            ggml_build_forward_expand(conditioning_graph, modulation.tensor);
        }
        core::validate_backend_graph_supported(execution.backend(), conditioning_graph, "KugelAudio diffusion conditioning");
        conditioning_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(conditioning_allocator.get(), conditioning_graph)) {
            throw std::runtime_error("KugelAudio diffusion conditioning allocation failed");
        }
        core::prepare_host_graph_plan(execution, conditioning_graph, conditioning_plan);
        modulation_inputs.clear();
        std::vector<core::TensorValue> selected;
        for (const auto & modulation : modulation_outputs) {
            auto cached = core::make_tensor(ctx, GGML_TYPE_F32, modulation.shape);
            ggml_set_input(cached.tensor);
            ggml_set_output(cached.tensor);
            modulation_inputs.push_back(cached);
            selected.push_back(modules::EmbeddingModule({rows, modulation.shape.last_dim()})
                .build(ctx, indices, cached));
        }
        output = build_modulated_head(ctx, input, selected, weights, config);
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), 2048, false);
        ggml_build_forward_expand(graph, output.tensor);
        core::validate_backend_graph_supported(execution.backend(), graph, "KugelAudio diffusion");
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("KugelAudio diffusion graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
        batch = requested_batch;
        steps = requested_steps;
        debug::timing_log_context_reservation("kugelaudio.diffusion.graph", context.get());
        debug::timing_log_scalar("kugelaudio.diffusion.graph.buffer_mb",
            static_cast<double>(ggml_gallocr_get_buffer_size(allocator.get(), 0)) / (1024 * 1024));
        debug::timing_log_scalar("kugelaudio.diffusion.conditioning.buffer_mb",
            static_cast<double>(ggml_gallocr_get_buffer_size(conditioning_allocator.get(), 0)) / (1024 * 1024));
    }

    void begin_frame(const std::vector<float> & conditioning, const std::vector<int64_t> & timesteps) {
        if (conditioning.empty() || conditioning.size() % config.hidden_size != 0 || timesteps.empty()) {
            throw std::runtime_error("KugelAudio diffusion conditioning shape mismatch");
        }
        prepare(conditioning.size() / config.hidden_size, timesteps.size());
        std::vector<float> features(steps * batch * 256);
        std::vector<float> repeated;
        repeated.reserve(steps * conditioning.size());
        for (int64_t step = 0; step < steps; ++step) {
            repeated.insert(repeated.end(), conditioning.begin(), conditioning.end());
            for (int64_t b = 0; b < batch; ++b) {
                for (int i = 0; i < 128; ++i) {
                    const float phase = static_cast<float>(timesteps[step]) * std::exp(-std::log(10000.0F) * static_cast<float>(i) / 128.0F);
                    features[(step * batch + b) * 256 + i] = std::cos(phase);
                    features[(step * batch + b) * 256 + 128 + i] = std::sin(phase);
                }
            }
        }
        core::write_tensor_f32(condition, repeated);
        core::write_tensor_f32(time, features);
        if (core::compute_graph(execution, conditioning_graph, conditioning_plan, "KugelAudio diffusion conditioning") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("KugelAudio diffusion conditioning compute failed");
        }
        for (size_t i = 0; i < modulation_outputs.size(); ++i) {
            ggml_backend_tensor_copy(modulation_outputs[i].tensor, modulation_inputs[i].tensor);
        }
    }

    std::vector<float> predict(const std::vector<float> & latent, size_t step) {
        if (latent.size() != static_cast<size_t>(batch * config.latent_size) ||
            steps == 0 || step >= static_cast<size_t>(steps)) {
            throw std::runtime_error("KugelAudio diffusion step or latent shape mismatch");
        }
        std::vector<int32_t> rows(batch);
        for (int64_t b = 0; b < batch; ++b) {
            rows[b] = static_cast<int32_t>(step * batch + b);
        }
        ggml_backend_tensor_set(indices.tensor, rows.data(), 0, rows.size() * sizeof(int32_t));
        core::write_tensor_f32(input, latent);
        if (core::compute_graph(execution, graph, plan, "KugelAudio diffusion") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("KugelAudio diffusion compute failed");
        }
        return core::read_tensor_f32(output.tensor);
    }
};

DiffusionHead::DiffusionHead(const assets::TensorSource & source, core::ExecutionContext & execution,
    assets::TensorStorageType storage, DiffusionConfig config)
    : impl_(std::make_unique<Impl>(source, execution, storage, config)) {}
DiffusionHead::~DiffusionHead() = default;
void DiffusionHead::begin_frame(const std::vector<float> & condition, const std::vector<int64_t> & timesteps) {
    impl_->begin_frame(condition, timesteps);
}
std::vector<float> DiffusionHead::predict(const std::vector<float> & latent, size_t step) {
    return impl_->predict(latent, step);
}

}  // namespace engine::models::kugelaudio
