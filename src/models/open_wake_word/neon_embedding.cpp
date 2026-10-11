#include "engine/models/open_wake_word/neon_embedding.h"
#include "engine/models/open_wake_word/weights.h"

namespace engine::models::open_wake_word {
namespace cpu = engine::neon;

std::unique_ptr<cpu::ExecutionPlan> build_native_embedding(const Weights &weights) {
    auto plan = std::make_unique<cpu::ExecutionPlan>();
    auto value = plan->input();
    for (const auto &layer : weights.embedding) {
        if (const auto *conv = std::get_if<ConvLayer>(&layer)) {
            value = plan->add(cpu::Conv2d(conv->config, conv->weight, conv->bias), {value});
        } else if (const auto *activation = std::get_if<LeakyReluLayer>(&layer)) {
            value = plan->add(cpu::Activation(cpu::ActivationKind::LeakyReLU, activation->slope), {value});
        } else if (const auto *clamp = std::get_if<ClampLayer>(&layer)) {
            value = plan->add(cpu::ClampMin(clamp->minimum), {value});
        } else {
            const auto &pool = std::get<PoolLayer>(layer);
            value = plan->add(cpu::MaxPool2d(pool.height, pool.width, pool.stride_height, pool.stride_width), {value});
        }
    }
    plan->set_output(value);
    return plan;
}

} // namespace engine::models::open_wake_word
