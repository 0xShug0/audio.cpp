#include "engine/models/ace_step/cuda_graph_execution.h"
#include <future>
#include <iostream>
#include <stdexcept>

using namespace engine::models::ace_step;
using engine::core::BackendType;
using namespace std::chrono_literals;
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    AceStepAssets base, turbo;
    require(detail::cuda_graph_execution_mutex == nullptr, "unexpected initial scope");
    {
        AceStepCudaGraphScope scope(base, BackendType::Cuda, 0);
        auto * original = detail::cuda_graph_execution_mutex;
        require(original != nullptr, "CUDA graph scope has no mutex");
        std::unique_lock<std::mutex> locked(*original);
        // An independent package on the same device shares the graph guard.
        auto another = std::async(std::launch::async, [&] {
            AceStepCudaGraphScope scope(turbo, BackendType::Cuda, 0);
            require(detail::cuda_graph_execution_mutex == original, "packages do not share device guard");
            require(!detail::cuda_graph_execution_mutex->try_lock(), "GPU graph exclusion failed");
            AceStepCudaGraphScope other_device(turbo, BackendType::Cuda, 1);
            require(detail::cuda_graph_execution_mutex != original, "different devices share guard");
            std::lock_guard<std::mutex> independent(*detail::cuda_graph_execution_mutex);
        });
        require(another.wait_for(2s) == std::future_status::ready, "different-device work was blocked");
        another.get();
        try {
            AceStepCudaGraphScope inner(turbo, BackendType::Cuda, 1);
            throw std::runtime_error("injected inference failure");
        } catch (const std::runtime_error &) {}
        require(detail::cuda_graph_execution_mutex == original, "exception did not restore outer scope");
        {
            AceStepCudaGraphScope cpu(turbo, BackendType::Cpu);
            require(detail::cuda_graph_execution_mutex == nullptr, "CPU acquired GPU graph guard");
        }
        {
            AceStepCudaGraphScope vulkan(turbo, BackendType::Vulkan);
            require(detail::cuda_graph_execution_mutex == nullptr, "Vulkan lost GPU overlap");
        }
        require(detail::cuda_graph_execution_mutex == original, "nested scope lost CUDA guard");
    }
    require(detail::cuda_graph_execution_mutex == nullptr, "scope leaked onto later requests");
    std::cout << "PASS: cross-package exclusion, device isolation, CPU/Vulkan bypass, nested/exception restoration\n";
}
