#include "engine/framework/audio/activity_graph.h"

#include <stdexcept>

namespace engine::sampling {

namespace {

const engine::core::ModulePortSpec kSingleInput[] = {
    {"input", engine::core::PortKind::Activation, false},
};

const engine::core::ModulePortSpec kSingleOutput[] = {
    {"output", engine::core::PortKind::Activation, false},
};

const engine::core::ModuleSchema kVADGateSchema = {
    "VADGate",
    "sampling.gating",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Thresholds frame energy into a binary speech mask.",
};

}

VADGateModule::VADGateModule(VADGateConfig config) : config_(config) {}

const VADGateConfig & VADGateModule::config() const noexcept {
    return config_;
}

const engine::core::ModuleSchema & VADGateModule::schema() const noexcept {
    return static_schema();
}

engine::core::TensorValue VADGateModule::build(engine::core::ModuleBuildContext & ctx, const engine::core::TensorValue & energy) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    engine::core::validate_rank_between(energy, 2, 2, "energy");
    auto shifted = engine::core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, energy.tensor, 1.0f, -config_.threshold),
        energy.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_step(ctx.ggml, shifted.tensor), energy.shape, GGML_TYPE_F32);
}

const engine::core::ModuleSchema & VADGateModule::static_schema() noexcept {
    return kVADGateSchema;
}

}  // namespace engine::sampling
