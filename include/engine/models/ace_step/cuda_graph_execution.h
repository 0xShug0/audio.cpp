#pragma once

#include "engine/framework/core/backend.h"
#include "engine/models/ace_step/assets.h"
#include <mutex>
#include <map>
#include <memory>

namespace engine::models::ace_step {
namespace detail {
inline thread_local std::mutex * cuda_graph_execution_mutex = nullptr;
inline std::shared_ptr<std::mutex> cuda_device_graph_mutex(int device) {
    static std::mutex registry_mutex;
    static std::map<int, std::weak_ptr<std::mutex>> devices;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto & entry = devices[device];
    auto mutex = entry.lock();
    if (!mutex) { mutex = std::make_shared<std::mutex>(); entry = mutex; }
    return mutex;
}
}

// A request keeps its backend and graph storage private. Serialize complete
// CUDA graph execution, while permitting CPU graph construction, input uploads,
// output downloads and request postprocessing to overlap across slots.
class AceStepCudaGraphScope {
public:
    AceStepCudaGraphScope(const AceStepAssets &, core::BackendType backend, int device = 0)
        : owner_(backend == core::BackendType::Cuda ? detail::cuda_device_graph_mutex(device) : nullptr),
          previous_(detail::cuda_graph_execution_mutex) {
        detail::cuda_graph_execution_mutex = backend == core::BackendType::Cuda
            ? owner_.get() : nullptr;
    }
    ~AceStepCudaGraphScope() { detail::cuda_graph_execution_mutex = previous_; }
    AceStepCudaGraphScope(const AceStepCudaGraphScope &) = delete;
    AceStepCudaGraphScope & operator=(const AceStepCudaGraphScope &) = delete;
private:
    std::shared_ptr<std::mutex> owner_;
    std::mutex * previous_;
};

inline ggml_status ace_step_compute_backend_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    std::unique_lock<std::mutex> lock;
    if (detail::cuda_graph_execution_mutex) {
        lock = std::unique_lock<std::mutex>(*detail::cuda_graph_execution_mutex);
    }
    // compute_backend_graph synchronizes the backend before returning: scratch
    // buffers and captured graph replay are finished before the next lease.
    return core::compute_backend_graph(backend, graph);
}
} // namespace engine::models::ace_step
