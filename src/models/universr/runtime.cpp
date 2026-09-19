#include "engine/models/universr/runtime.h"

#include "engine/models/universr/network.h"
#include "engine/framework/modules/flow_sampler_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::universr {
namespace {

class UniverSRGraphs {
public:
    UniverSRGraphs(core::ExecutionContext & execution, const UniverSRConfig & config,
        const ConditioningWeights & conditioning, const UNetBackboneWeights & backbone,
        int64_t frames, int sample_rate_khz)
        : backend_(execution.backend()) {
        const auto started = std::chrono::steady_clock::now();
        constexpr size_t nodes = 16384;
        const size_t arena = nodes * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(nodes, false);
        condition_context_.reset(ggml_init({arena, nullptr, true}));
        denoise_context_.reset(ggml_init({arena, nullptr, true}));
        if (!condition_context_ || !denoise_context_) {
            throw std::runtime_error("UniverSR graph context allocation failed");
        }
        const auto rate_it = config.sr_to_lr_bins.find(sample_rate_khz);
        if (rate_it == config.sr_to_lr_bins.end() || frames < 65) {
            throw std::runtime_error("UniverSR requires a supported input rate and at least 65 STFT frames");
        }
        const int64_t rate_index = std::distance(config.sr_to_lr_bins.begin(), rate_it);
        const int64_t pad = (16 - frames % 16) % 16;
        core::ModuleBuildContext cond_ctx{condition_context_.get(), "universr.condition", execution.backend_type()};
        auto low = core::make_tensor(cond_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 2, rate_it->second, frames}));
        low_ = low.tensor;
        ggml_set_input(low_);
        if (pad != 0) {
            low = modules::ReflectPad1dModule({0, pad}).build(cond_ctx, low);
        }
        const auto rate = modules::SliceModule({0, rate_index, 1}).build(cond_ctx, conditioning.sample_rate_embedding);
        auto condition = build_conditioning_encoder(cond_ctx, low, rate, conditioning);
        const auto condition_projection = modules::SliceModule({1, 2, config.cond_dim}).build(cond_ctx, backbone.input.weight);
        condition = build_projected_conditioning(cond_ctx, condition, config, conditioning, condition_projection);
        auto uncond = core::reshape_tensor(cond_ctx, conditioning.unconditional_embedding,
            core::TensorShape::from_dims({1, config.cond_dim, 1, 1}));
        uncond = build_projected_conditioning(cond_ctx, uncond, config, conditioning, condition_projection);
        uncond = modules::RepeatModule({condition.shape}).build(cond_ctx, uncond);
        conditional_ = condition.tensor;
        unconditional_ = uncond.tensor;
        ggml_set_output(conditional_);
        ggml_set_output(unconditional_);
        condition_graph_ = ggml_new_graph_custom(cond_ctx.ggml, nodes, false);
        ggml_build_forward_expand(condition_graph_, conditional_);
        ggml_build_forward_expand(condition_graph_, unconditional_);
        // Fold explicit module broadcasts without changing other graph lowerings.
        runtime::GraphOptimizationOptions optimization;
        optimization.fold_commutative_lhs_repeats = false;
        optimization.fold_two_sided_broadcast_repeats = false;
        optimization.fold_unary_broadcast_repeats = false;
        optimization.fold_identity_materializations = false;
        optimization.elide_noop_nodes = false;
        optimization.elide_metadata_only_ops = false;
        runtime::optimize_graph(*condition_graph_, optimization);
        core::validate_backend_graph_supported(backend_, condition_graph_, "UniverSR conditioning");

        core::ModuleBuildContext ctx{denoise_context_.get(), "universr.denoise", execution.backend_type()};
        auto noise = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 2, config.hr_freq_bins, frames}));
        noise_ = noise.tensor;
        ggml_set_input(noise_);
        if (pad != 0) {
            noise = modules::ReflectPad1dModule({0, pad}).build(ctx, noise);
        }
        auto spatial = core::make_tensor(ctx, GGML_TYPE_F32, condition.shape);
        spatial_ = spatial.tensor;
        ggml_set_input(spatial_);
        auto time = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1}));
        time_ = time.tensor;
        ggml_set_input(time_);
        const auto denoise_rate = modules::SliceModule({0, rate_index, 1}).build(ctx, conditioning.sample_rate_embedding);
        auto embedding = build_time_embedding(ctx, time, denoise_rate, conditioning);
        auto input = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, noise);
        input = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, input);
        const auto noise_projection = modules::SliceModule({1, 0, 2}).build(ctx, backbone.input.weight);
        input = modules::LinearModule({2, config.dims.front(), true})
            .build(ctx, input, {noise_projection, backbone.input.bias});
        input = modules::AddModule().build(ctx, input, spatial);
        input = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, input);
        input = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, input);
        auto output = build_unet_backbone(ctx, input, embedding, backbone, true);
        if (pad != 0) {
            // Slice views require a unit innermost stride; the backbone returns a transposed NCHW view.
            output = core::ensure_backend_addressable_layout(ctx, output);
            output = modules::SliceModule({3, 0, frames}).build(ctx, output);
        }
        output_ = ggml_cont(ctx.ggml, output.tensor);
        ggml_set_output(output_);
        denoise_graph_ = ggml_new_graph_custom(ctx.ggml, nodes, false);
        ggml_build_forward_expand(denoise_graph_, output_);
        runtime::optimize_graph(*denoise_graph_, optimization);
        core::validate_backend_graph_supported(backend_, denoise_graph_, "UniverSR denoising");
        // Execution is sequential; share scratch space while output flags retain both conditioning tensors.
        allocation_graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
        ggml_build_forward_expand(allocation_graph, conditional_);
        ggml_build_forward_expand(allocation_graph, unconditional_);
        ggml_build_forward_expand(allocation_graph, output_);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> planner(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)), ggml_gallocr_free);
        if (!planner) {
            throw std::runtime_error("UniverSR graph planner allocation failed");
        }
        ggml_gallocr_reserve_n_size(planner.get(), allocation_graph, nullptr, nullptr, &workspace_bytes);
        debug::timing_log_scalar("universr.graph.build_ms", debug::elapsed_ms(started));
    }

    ~UniverSRGraphs() {
        core::release_backend_graph_resources(backend_, denoise_graph_, true);
        core::release_backend_graph_resources(backend_, condition_graph_, true);
    }

    void prepare(const std::vector<float> & low) {
        if (low.size() * sizeof(float) != ggml_nbytes(low_)) {
            throw std::runtime_error("UniverSR low spectrum size mismatch");
        }
        ready_ = false;
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(low_, low.data(), 0, low.size() * sizeof(float));
        if (core::compute_backend_graph(backend_, condition_graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("UniverSR conditioning graph execution failed");
        }
        ready_ = true;
        debug::timing_log_scalar("universr.condition.ms", debug::elapsed_ms(started));
    }

    std::vector<float> predict(const std::vector<float> & noise, float time, bool conditional) {
        if (!ready_ || noise.size() * sizeof(float) != ggml_nbytes(noise_)) {
            throw std::runtime_error("UniverSR vector field requires prepared conditioning and matching noise");
        }
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(noise_, noise.data(), 0, noise.size() * sizeof(float));
        ggml_backend_tensor_set(time_, &time, 0, sizeof(time));
        ggml_backend_tensor_copy(conditional ? conditional_ : unconditional_, spatial_);
        if (core::compute_backend_graph(backend_, denoise_graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("UniverSR denoising graph execution failed");
        }
        std::vector<float> output(static_cast<size_t>(ggml_nelements(output_)));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        debug::timing_log_scalar("universr.denoise.ms", debug::elapsed_ms(started));
        return output;
    }

    ggml_cgraph * allocation_graph = nullptr;
    size_t workspace_bytes = 0;

private:
    ggml_backend_t backend_;
    bool ready_ = false;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> condition_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> denoise_context_{nullptr, ggml_free};
    ggml_cgraph * condition_graph_ = nullptr;
    ggml_cgraph * denoise_graph_ = nullptr;
    ggml_tensor * low_ = nullptr;
    ggml_tensor * conditional_ = nullptr;
    ggml_tensor * unconditional_ = nullptr;
    ggml_tensor * noise_ = nullptr;
    ggml_tensor * spatial_ = nullptr;
    ggml_tensor * time_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

class UniverSRRuntime::Impl {
public:
    Impl(std::shared_ptr<const UniverSRAssets> assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "universr.weights", 4 * 1024 * 1024) {
        conditioning_ = load_conditioning_weights(store_, *assets_->tensors, assets_->config, storage);
        backbone_ = load_unet_backbone_weights(store_, *assets_->tensors, assets_->config, storage);
        store_.upload();
        assets_->tensors->release_storage();
        core::set_backend_threads(execution.backend(), execution.config().threads);
    }

    void prepare(const std::vector<float> & low, int64_t frames, int sample_rate_khz) {
        const std::vector<int64_t> key{frames, sample_rate_khz};
        if (auto * found = graphs_.find(key)) {
            active_graph_ = found->get();
        } else {
            auto graph = std::make_unique<UniverSRGraphs>(execution_, assets_->config,
                conditioning_, backbone_, frames, sample_rate_khz);
            const auto buffer_type = ggml_backend_get_default_buffer_type(execution_.backend());
            const bool single_buffer = graph->workspace_bytes <= ggml_backend_buft_get_max_size(buffer_type);
            // Cached tensor addresses remain valid only while the shared buffer is unchanged.
            if (!single_buffer || !single_buffer_ || !allocator_ ||
                graph->workspace_bytes > ggml_gallocr_get_buffer_size(allocator_.get(), 0)) {
                active_graph_ = nullptr;
                graphs_.clear();
            }
            if (!allocator_ || !single_buffer || !single_buffer_) {
                allocator_.reset(ggml_gallocr_new(buffer_type));
            }
            if (!allocator_ || !ggml_gallocr_reserve(allocator_.get(), graph->allocation_graph) ||
                !ggml_gallocr_alloc_graph(allocator_.get(), graph->allocation_graph)) {
                throw std::runtime_error("UniverSR shared graph allocation failed");
            }
            single_buffer_ = single_buffer;
            active_graph_ = graph.get();
            graphs_.put(key, std::move(graph));
            debug::timing_log_scalar("universr.graph.workspace_bytes",
                ggml_gallocr_get_buffer_size(allocator_.get(), 0));
        }
        active_graph_->prepare(low);
    }

    std::vector<float> predict(const std::vector<float> & noise, float time, bool conditional) {
        if (!active_graph_) {
            throw std::runtime_error("UniverSR conditioning must be prepared before prediction");
        }
        return active_graph_->predict(noise, time, conditional);
    }

private:
    std::shared_ptr<const UniverSRAssets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    ConditioningWeights conditioning_;
    UNetBackboneWeights backbone_;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    runtime::CacheSlots<std::vector<int64_t>, std::unique_ptr<UniverSRGraphs>> graphs_{2};
    UniverSRGraphs * active_graph_ = nullptr;
    bool single_buffer_ = true;
};

UniverSRRuntime::UniverSRRuntime(std::shared_ptr<const UniverSRAssets> assets,
    core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}

UniverSRRuntime::~UniverSRRuntime() = default;

void UniverSRRuntime::prepare_condition(const std::vector<float> & low_spectrum, int64_t frames, int sample_rate_khz) {
    impl_->prepare(low_spectrum, frames, sample_rate_khz);
}

std::vector<float> UniverSRRuntime::vector_field(const std::vector<float> & noise, float time, bool conditional) {
    return impl_->predict(noise, time, conditional);
}

std::vector<float> UniverSRRuntime::integrate_flow(const std::vector<float> & initial_noise, int steps,
    const std::string & method, float guidance_scale) {
    if (steps < 1 || initial_noise.empty() || !std::isfinite(guidance_scale) || guidance_scale < 0 ||
        (method != "euler" && method != "midpoint" && method != "rk4")) {
        throw std::runtime_error("UniverSR flow requires positive steps, noise, nonnegative guidance, and euler/midpoint/rk4");
    }
    const auto started = std::chrono::steady_clock::now();
    auto velocity = [&](const std::vector<float> & latent, float time) {
        auto conditional = vector_field(latent, time, true);
        if (guidance_scale > 0 && guidance_scale != 1) {
            const auto unconditional = vector_field(latent, time, false);
            // Preserve upstream's weighted-sum order; generic CFG subtracts the branches first.
            for (size_t i = 0; i < conditional.size(); ++i) {
                conditional[i] = (1.0f - guidance_scale) * unconditional[i] + guidance_scale * conditional[i];
            }
        }
        return conditional;
    };
    auto latent = initial_noise;
    auto updater = modules::make_flow_sampler_euler_update();
    modules::FlowSamplerStepState state;
    const float spacing = 1.0f / static_cast<float>(steps);
    for (int step = 0; step < steps; ++step) {
        // Match torch.linspace's two-sided construction, including exact endpoints.
        const float t = step < (steps + 1) / 2 ? spacing * step : 1.0f - spacing * (steps - step);
        const int next = step + 1;
        const float t_next = next < (steps + 1) / 2 ? spacing * next : 1.0f - spacing * (steps - next);
        const float dt = t_next - t;
        state.schedule.index = step;
        state.schedule.t = t;
        state.schedule.t_next = t_next;
        auto k1 = velocity(latent, t);
        if (method == "euler") {
            updater->update_latent({state, k1, latent});
        } else if (method == "midpoint") {
            auto midpoint = latent;
            for (size_t i = 0; i < latent.size(); ++i) {
                midpoint[i] += k1[i] * (0.5f * dt);
            }
            const auto k2 = velocity(midpoint, t + 0.5f * dt);
            updater->update_latent({state, k2, latent});
        } else {
            auto stage = latent;
            for (size_t i = 0; i < latent.size(); ++i) {
                stage[i] += (dt * k1[i]) * (1.0f / 3.0f);
            }
            const auto k2 = velocity(stage, t + dt * (1.0f / 3.0f));
            for (size_t i = 0; i < latent.size(); ++i) {
                stage[i] = latent[i] + dt * (k2[i] - k1[i] * (1.0f / 3.0f));
            }
            const auto k3 = velocity(stage, t + dt * (2.0f / 3.0f));
            for (size_t i = 0; i < latent.size(); ++i) {
                stage[i] = latent[i] + dt * (k1[i] - k2[i] + k3[i]);
            }
            const auto k4 = velocity(stage, t_next);
            for (size_t i = 0; i < latent.size(); ++i) {
                latent[i] += ((k1[i] + 3.0f * (k2[i] + k3[i]) + k4[i]) * dt) * 0.125f;
            }
        }
    }
    debug::timing_log_scalar("universr.flow.ms", debug::elapsed_ms(started));
    return latent;
}

}  // namespace engine::models::universr
