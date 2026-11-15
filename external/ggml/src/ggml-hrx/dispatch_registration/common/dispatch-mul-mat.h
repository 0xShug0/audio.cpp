#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

// Precision-preserving strided SIMT GEMM, also used for attention QK/PV.
bool build_float_matmul_dispatch(const Value & weights, const Value & inputs, const Value & outputs, Dispatch & dispatch);

void register_mul_mat_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
