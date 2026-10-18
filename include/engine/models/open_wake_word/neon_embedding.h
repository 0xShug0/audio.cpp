#pragma once

#include "engine/framework/neon/execution.h"

namespace engine::models::open_wake_word {
struct Weights;

std::unique_ptr<neon::ExecutionPlan> build_native_embedding(const Weights &weights);

} // namespace engine::models::open_wake_word
