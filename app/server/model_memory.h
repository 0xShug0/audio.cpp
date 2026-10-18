#pragma once

#include "config.h"

#include <cstddef>
#include <optional>

namespace minitts::server {

// Estimated resident bytes a model will occupy once loaded (weights plus a
// runtime overhead factor for GPU buffers / compute graphs). A family that
// declares compiled engines through an `exec_plan` reports them, and the
// residency it will keep of them, through its loader's resource plan: the target
// runtime prices each engine, and where it cannot answer (every build without an
// RKNN runtime) the engine is priced at its file size. Engines are priced at 1.0
// and never at the runtime factor, and the family's ceiling caps that set. The
// server reads both plans and applies no plan grammar of its own.
// Returns nullopt
// when the footprint is indeterminate: a model directory holding several GGUFs
// and no model.gguf. The guard must not refuse such a load with a 503 -- the
// loader either rejects the ambiguous directory itself (the spec-driven path
// fails with its "contains N GGUF files" error) or accepts it under a
// family-specific layout the estimator cannot resolve, so the guard skips
// rather than guess.
std::optional<size_t> estimate_model_memory_bytes(const ServerModelConfig & model);

}  // namespace minitts::server
