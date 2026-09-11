#pragma once

// ggml_graph_compute_with_ctx lives in the CPU backend (ggml-cpu). With
// GGML_BACKEND_DL builds (ENGINE_ENABLE_CPU_ALL_VARIANTS, e.g. the docker
// image) that backend is a dlopen'd MODULE library, so the symbol is not
// linkable from engine_runtime. Resolve it at runtime instead:
//   - static-link builds bind the weak reference directly;
//   - dlopen'd builds find the already-loaded libggml-cpu via the dynamic
//     linker's loaded-object list (it is RTLD_LOCAL, so plain
//     dlsym(RTLD_DEFAULT) would miss it).

#include "ggml.h"

#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <dlfcn.h>
#include <link.h>
#endif

namespace engine::models::f5_tts {

#if defined(__GNUC__)
extern "C" ggml_status ggml_graph_compute_with_ctx(
    ggml_context * ctx, ggml_cgraph * cgraph, int n_threads) __attribute__((weak));
#endif

using F5CpuGraphComputeFn = ggml_status (*)(ggml_context *, ggml_cgraph *, int);

inline F5CpuGraphComputeFn f5_cpu_graph_compute_fn() {
#if defined(__GNUC__)
    if (ggml_graph_compute_with_ctx != nullptr) {
        return ggml_graph_compute_with_ctx;  // statically linked build
    }
#endif
#if defined(__linux__)
    static F5CpuGraphComputeFn fn = [] {
        void * sym = nullptr;
        dl_iterate_phdr(
            [](struct dl_phdr_info * info, size_t, void * data) -> int {
                if (info->dlpi_name == nullptr ||
                    std::strstr(info->dlpi_name, "libggml-cpu") == nullptr) {
                    return 0;
                }
                void * h = dlopen(info->dlpi_name, RTLD_NOW | RTLD_NOLOAD);
                if (h == nullptr) {
                    return 0;
                }
                void * s = dlsym(h, "ggml_graph_compute_with_ctx");
                if (s != nullptr) {
                    *reinterpret_cast<void **>(data) = s;
                    return 1;
                }
                return 0;
            },
            &sym);
        return reinterpret_cast<F5CpuGraphComputeFn>(sym);
    }();
    if (fn != nullptr) {
        return fn;
    }
#endif
    throw std::runtime_error(
        "F5-TTS: ggml_graph_compute_with_ctx unavailable (CPU backend not loaded)");
}

inline ggml_status f5_cpu_graph_compute(ggml_context * ctx, ggml_cgraph * graph, int threads) {
    return f5_cpu_graph_compute_fn()(ctx, graph, threads);
}

}  // namespace engine::models::f5_tts
