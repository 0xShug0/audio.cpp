#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/options.h"

namespace engine::runtime {

RuntimeSessionBase::RuntimeSessionBase(const SessionOptions & options)
    : options_(options),
      execution_context_(options.backend, parse_finite_float_option(options.options, {"moe_cpu_share"}).value_or(0)),
      graph_executor_(execution_context_) {}

engine::core::ExecutionContext & RuntimeSessionBase::execution_context() noexcept {
    return execution_context_;
}

const engine::core::ExecutionContext & RuntimeSessionBase::execution_context() const noexcept {
    return execution_context_;
}

ArtifactStore & RuntimeSessionBase::artifacts() noexcept {
    return artifacts_;
}

const ArtifactStore & RuntimeSessionBase::artifacts() const noexcept {
    return artifacts_;
}

RuntimeCache & RuntimeSessionBase::cache() noexcept {
    return cache_;
}

const RuntimeCache & RuntimeSessionBase::cache() const noexcept {
    return cache_;
}

RuntimeWorkspace & RuntimeSessionBase::workspace() noexcept {
    return workspace_;
}

const RuntimeWorkspace & RuntimeSessionBase::workspace() const noexcept {
    return workspace_;
}

GraphExecutor & RuntimeSessionBase::graph_executor() noexcept {
    return graph_executor_;
}

const GraphExecutor & RuntimeSessionBase::graph_executor() const noexcept {
    return graph_executor_;
}

void RuntimeSessionBase::mark_prepared() {
    for (const auto & [key, value] : options_.options) {
        (void) value;
        if (key.rfind("moe_", 0) != 0) continue;
        if (key != "moe_cpu_share" && key != "moe_disk_experts" && key != "moe_pin_experts" &&
            key != "moe_expert_cache_size" && key != "moe_disk_expert_read_threads" && key != "moe_cpu_expert_layers")
            throw std::invalid_argument("unknown shared MoE session option: " + key);
        // Explicit zero preserves the existing disabled behavior on non-MoE sessions.
        if (key == "moe_cpu_share" && execution_context_.moe_cpu_share() == 0) continue;
        if (!execution_context_.has_moe_resources())
            throw std::invalid_argument("session does not support shared MoE resources: " + key);
    }
    prepared_ = true;
}

void RuntimeSessionBase::require_prepared(std::string_view operation) const {
    if (!prepared_) {
        throw std::runtime_error(
            "session prepare() must be called before " + std::string(operation));
    }
}

std::unique_lock<std::mutex> RuntimeSessionBase::acquire_moe_execution() {
    std::unique_lock<std::mutex> lock(moe_execution_mutex_, std::defer_lock);
    if (!lock.try_lock())
        throw std::runtime_error("overlapping requests on one MoE session are not supported");
    return lock;
}

void RuntimeSessionBase::trace(engine::debug::LogLevel level, std::string_view category, std::string_view message) const {
    engine::debug::log_message(level, category, message);
}

engine::debug::ScopeTimer RuntimeSessionBase::profile(
    engine::debug::LogLevel level,
    std::string_view category,
    std::string_view name) const {
    return engine::debug::ScopeTimer(level, category, name);
}

const SessionOptions & RuntimeSessionBase::options() const noexcept {
    return options_;
}

}  // namespace engine::runtime
