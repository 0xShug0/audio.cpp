#pragma once

#include "engine/framework/core/backend.h"

namespace engine::models::f5_tts {

// Use the loaded CPU backend's public API. The private ggml-cpu symbol is
// unavailable to MSVC and may reside in an RTLD_LOCAL module on Linux.
inline ggml_status f5_cpu_graph_compute(ggml_backend_t backend, ggml_cgraph * graph, int threads) {
    engine::core::set_backend_threads(backend, threads);
    return engine::core::compute_backend_graph(backend, graph, nullptr, "f5_cpu");
}

} // namespace engine::models::f5_tts
