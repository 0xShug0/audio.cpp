#pragma once

#include "config.h"

#include <cstddef>
#include <optional>

namespace minitts::server {

// Estimated resident bytes a model will occupy once loaded (weights plus a
// runtime overhead factor for GPU buffers / compute graphs). A compiled engine
// declared through an `exec_plan` is counted at the residency the NPU driver
// reports for it -- the image size of a Rockchip engine is a fraction of what
// loading it occupies -- and at its file size where no driver can answer.
// Returns nullopt
// when the footprint is indeterminate: a model directory holding several GGUFs
// and no model.gguf. The guard must not refuse such a load with a 503 -- the
// loader either rejects the ambiguous directory itself (the spec-driven path
// fails with its "contains N GGUF files" error) or accepts it under a
// family-specific layout the estimator cannot resolve, so the guard skips
// rather than guess.
std::optional<size_t> estimate_model_memory_bytes(const ServerModelConfig & model);

}  // namespace minitts::server
