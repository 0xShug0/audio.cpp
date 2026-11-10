#include "engine/framework/sampling/device_denoiser_runtime.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::modules {

namespace {

void validate_f32(const core::TensorValue & value) {
    if (!value.tensor || value.type != GGML_TYPE_F32 || value.tensor->type != GGML_TYPE_F32)
        throw std::runtime_error("Flow sampler graph math requires F32 tensors");
}

void validate_scalar(const core::TensorValue & value) {
    validate_f32(value);
    if (ggml_nelements(value.tensor) != 1)
        throw std::runtime_error("Flow sampler graph math requires a scalar input");
}

void validate_upload(const core::TensorValue & target, size_t offset, size_t bytes) {
    if (!target.tensor || !target.tensor->data || !ggml_is_contiguous(target.tensor) ||
        !(target.tensor->view_src ? target.tensor->view_src->buffer : target.tensor->buffer))
        throw std::runtime_error("Flow sampler upload requires an allocated contiguous tensor");
    const size_t capacity = ggml_nbytes(target.tensor);
    if (bytes == 0 || offset > capacity || bytes > capacity - offset)
        throw std::runtime_error("Flow sampler upload exceeds tensor bounds");
}

}  // namespace

DeviceFlowSamplerRuntime::DeviceFlowSamplerRuntime(core::PreparedGraph & graph, core::TensorValue latent)
    : graph_(graph), latent_(latent) {
    validate_f32(latent_);
}

std::vector<float> DeviceFlowSamplerRuntime::run(
    const std::vector<float> & initial_latent, const FlowSamplerGraphSequence & sequence) {
    if (!graph_.allocated()) throw std::runtime_error("Flow sampler graph is not allocated");
    if (sequence.steps == 0)
        throw std::runtime_error("Flow sampler requires a positive step count");
    if (!sequence.graph_indices.empty()) {
        if (sequence.graph_indices.size() != sequence.steps)
            throw std::runtime_error("Flow sampler graph schedule size mismatch");
        for (size_t index : sequence.graph_indices)
            if (index >= graph_.graph_count()) throw std::runtime_error("Flow sampler graph index out of range");
    }
    validate_upload(latent_, 0, ggml_nbytes(latent_.tensor));
    if (initial_latent.size() != static_cast<size_t>(ggml_nelements(latent_.tensor)))
        throw std::runtime_error("Flow sampler initial latent shape mismatch");
    for (const auto & binding : sequence.step_bindings) {
        validate_upload(binding.target, binding.offset_bytes, binding.bytes_per_step);
        if (!binding.data)
            throw std::runtime_error("Flow sampler step upload source is null");
        if (sequence.steps > std::numeric_limits<size_t>::max() / binding.bytes_per_step ||
            binding.data_bytes != sequence.steps * binding.bytes_per_step)
            throw std::runtime_error("Flow sampler step upload sequence size mismatch");
    }
    // Addresses may be synthetic offsets (e.g. WebGPU). Compare byte ranges
    // only within the same backend buffer, resolving views to their storage.
    struct UploadRange { ggml_backend_buffer_t buffer; size_t offset; size_t bytes; };
    std::vector<UploadRange> ranges;
    ranges.reserve(sequence.step_bindings.size() + 1);
    const auto add_range = [&](const core::TensorValue & value, size_t offset, size_t bytes) {
        const auto * storage = value.tensor->view_src ? value.tensor->view_src : value.tensor;
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(storage->buffer));
        const size_t tensor_offset = reinterpret_cast<uintptr_t>(storage->data) - base;
        ranges.push_back({storage->buffer, tensor_offset + value.tensor->view_offs + offset, bytes});
    };
    add_range(latent_, 0, ggml_nbytes(latent_.tensor));
    for (const auto & binding : sequence.step_bindings)
        add_range(binding.target, binding.offset_bytes, binding.bytes_per_step);
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            const auto & a = ranges[i];
            const auto & b = ranges[j];
            if (a.buffer == b.buffer &&
                (a.offset <= b.offset ? b.offset - a.offset < a.bytes : a.offset - b.offset < b.bytes))
                throw std::runtime_error("Flow sampler upload bindings overlap");
        }
    }

    const auto backend = graph_.execution().backend();
    core::write_tensor_f32(latent_, initial_latent);
    try {
        const auto started = std::chrono::steady_clock::now();
        for (size_t step = 0; step < sequence.steps; ++step) {
            for (const auto & binding : sequence.step_bindings)
                ggml_backend_tensor_set_async(backend, binding.target.tensor,
                    static_cast<const std::byte *>(binding.data) + step * binding.bytes_per_step,
                    binding.offset_bytes, binding.bytes_per_step);
            const auto id = sequence.graph_indices.empty() ? 0 : sequence.graph_indices[step];
            if (graph_.has_host_plan(id)) {
                graph_.compute(id);
            } else if (ggml_backend_graph_compute_async(backend, graph_.graph(id)) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(graph_.label() + " sampling graph execution failed");
            }
        }
        ggml_backend_synchronize(backend);
        debug::timing_log_scalar(graph_.label() + ".sequence.compute_ms", debug::elapsed_ms(started));
    } catch (...) {
        ggml_backend_synchronize(backend);
        throw;
    }
    std::vector<float> result;
    core::read_tensor_float_into(latent_.tensor, result);
    return result;
}

core::TensorValue FlowSamplerCfgGraphModule::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & positive, const core::TensorValue & negative,
    const core::TensorValue & guidance) const {
    validate_f32(positive);
    validate_f32(negative);
    validate_scalar(guidance);
    core::validate_shape(negative, positive.shape, "negative prediction");
    auto difference = core::wrap_tensor(ggml_sub(ctx.ggml, positive.tensor, negative.tensor),
        positive.shape, GGML_TYPE_F32);
    return AddModule().build(ctx, negative, MulModule(true).build(ctx, difference, guidance));
}

FlowSamplerEulerGraphModule::FlowSamplerEulerGraphModule(FlowSamplerEulerGraphConfig config)
    : config_(config) {}

core::TensorValue FlowSamplerEulerGraphModule::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & latent, const core::TensorValue & velocity,
    const core::TensorValue & dt) const {
    validate_f32(latent);
    validate_f32(velocity);
    validate_scalar(dt);
    core::validate_shape(velocity, latent.shape, "velocity");
    auto delta = MulModule(true).build(ctx, velocity, dt);
    if (config_.delta_rounding == FlowSamplerGraphRounding::Bf16)
        delta = Bf16RoundModule().build(ctx, delta);
    auto updated = AddModule().build(ctx, latent, delta);
    if (config_.output_rounding == FlowSamplerGraphRounding::Bf16)
        updated = Bf16RoundModule().build(ctx, updated);
    return updated;
}

DeviceDenoiserRuntime::DeviceDenoiserRuntime(core::ExecutionContext & execution,
    core::PreparedGraphConfig config, const Builder & builder, DeviceDenoiserOptimizations optimizations)
    : DeviceDenoiserRuntime(execution, std::move(config), std::vector<Builder>{builder}, optimizations) {}

DeviceDenoiserRuntime::DeviceDenoiserRuntime(core::ExecutionContext & execution,
    core::PreparedGraphConfig config, const std::vector<Builder> & phases,
    DeviceDenoiserOptimizations optimizations)
    : graph_(execution, std::move(config)) {
    if (phases.empty()) throw std::runtime_error("Device denoiser requires at least one phase");
    const auto device = ggml_backend_get_device(execution.backend());
    if (device) {
        set_fusions_ = reinterpret_cast<ggml_backend_set_graph_fusions_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),
                "ggml_backend_set_graph_fusions"));
    }
    uint32_t mask = 0;
    if (optimizations.norm_affine) mask |= GGML_BACKEND_GRAPH_FUSION_NORM_AFFINE;
    if (optimizations.bias_activation) mask |= GGML_BACKEND_GRAPH_FUSION_BIAS_ACTIVATION;
    if (optimizations.gated_residual) mask |= GGML_BACKEND_GRAPH_FUSION_GATED_RESIDUAL;
    try {
        tensors_.reserve(phases.size());
        for (size_t phase = 0; phase < phases.size(); ++phase) {
            const auto id = phase == 0 ? 0 : graph_.add_graph(std::to_string(phase));
            tensors_.push_back(phases[phase](graph_.build_context()));
            const auto & tensors = tensors_.back();
            if (!tensors.latent.tensor || !tensors.prediction.tensor || !tensors.updated_latent.tensor)
                throw std::runtime_error("Device denoiser requires latent, prediction and updated latent tensors");
            core::validate_shape(tensors.updated_latent, tensors.latent.shape, "updated denoiser latent");
            if (tensors.latent.tensor->type != GGML_TYPE_F32 ||
                tensors.updated_latent.tensor->type != GGML_TYPE_F32)
                throw std::runtime_error("Device denoiser sampling requires F32 latent state");
            if (phases.size() > 1 && !tensors.latent.tensor->buffer)
                throw std::runtime_error("Multiphase denoising requires separately allocated latent storage");
            if (tensors.latent.tensor != tensors_.front().latent.tensor)
                throw std::runtime_error("Device denoiser phases must share one persistent latent tensor");
            ggml_set_output(tensors.prediction.tensor);
            ggml_build_forward_expand(graph_.graph(id), tensors.prediction.tensor);
            ggml_build_forward_expand(graph_.graph(id), ggml_cpy(graph_.build_context().ggml,
                tensors.updated_latent.tensor, tensors.latent.tensor));
            if (set_fusions_) set_fusions_(execution.backend(), graph_.graph(id), mask);
        }
        graph_.allocate();
        sampler_ = std::make_unique<DeviceFlowSamplerRuntime>(graph_, tensors_.front().latent);
    } catch (...) {
        if (set_fusions_)
            for (size_t id = 0; id < graph_.graph_count(); ++id)
                set_fusions_(execution.backend(), graph_.graph(id), 0);
        throw;
    }
}

DeviceDenoiserRuntime::~DeviceDenoiserRuntime() {
    if (set_fusions_)
        for (size_t id = 0; id < graph_.graph_count(); ++id)
            set_fusions_(graph_.execution().backend(), graph_.graph(id), 0);
}

std::vector<float> DeviceDenoiserRuntime::predict(size_t phase) {
    graph_.compute(phase);
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> result;
    core::read_tensor_float_into(tensors_.at(phase).prediction.tensor, result);
    debug::timing_log_scalar(graph_.label() + ".output_read_ms", debug::elapsed_ms(started));
    return result;
}

std::vector<float> DeviceDenoiserRuntime::sample(const std::vector<float> & initial_latent,
    const FlowSamplerGraphSequence & sequence) {
    return sampler_->run(initial_latent, sequence);
}

}  // namespace engine::modules
