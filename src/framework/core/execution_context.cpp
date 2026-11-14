#include "engine/framework/core/execution_context.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::core {

ExecutionContext::ExecutionContext(const BackendConfig & config, float moe_cpu_share)
    : config_(config), moe_cpu_share_(moe_cpu_share) {
    if (!std::isfinite(moe_cpu_share_) || moe_cpu_share_ < 0 || moe_cpu_share_ > 1)
        throw std::invalid_argument("moe_cpu_share must be between zero and one");
    backend_ = init_backend(config);
    if (backend_ != nullptr && config_.threads > 0) {
        set_backend_threads(backend_, config_.threads);
    }
}

ExecutionContext::~ExecutionContext() {
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

ExecutionContext::ExecutionContext(ExecutionContext && other) noexcept
    : config_(std::move(other.config_)),
      moe_cpu_share_(other.moe_cpu_share_),
      moe_resources_registered_(other.moe_resources_registered_),
      backend_(other.backend_) {
    other.backend_ = nullptr;
}

ExecutionContext & ExecutionContext::operator=(ExecutionContext && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
    config_ = std::move(other.config_);
    moe_cpu_share_ = other.moe_cpu_share_;
    moe_resources_registered_ = other.moe_resources_registered_;
    backend_ = other.backend_;
    other.backend_ = nullptr;
    return *this;
}

ggml_backend_t ExecutionContext::backend() const noexcept {
    return backend_;
}

const BackendConfig & ExecutionContext::config() const noexcept {
    return config_;
}

BackendType ExecutionContext::backend_type() const noexcept {
    if (config_.type != BackendType::BestAvailable) {
        return config_.type;
    }
    return core::backend_type(backend_);
}

bool ExecutionContext::uses_host_graph_plan() const noexcept {
    return core::uses_host_graph_plan(backend_);
}

BackendMemorySnapshot ExecutionContext::memory_snapshot() const {
    if (config_.type != BackendType::BestAvailable) {
        return query_backend_memory(config_);
    }
    return query_backend_memory(backend_, config_.device);
}

}  // namespace engine::core
