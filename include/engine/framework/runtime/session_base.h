#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/artifacts.h"
#include "engine/framework/runtime/cache.h"
#include "engine/framework/runtime/graph_executor.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/workspace.h"

#include <stdexcept>
#include <mutex>
#include <string>
#include <string_view>

namespace engine::runtime {

class RuntimeSessionBase {
public:
    explicit RuntimeSessionBase(const SessionOptions & options);
    virtual ~RuntimeSessionBase() = default;

protected:
    engine::core::ExecutionContext & execution_context() noexcept;
    const engine::core::ExecutionContext & execution_context() const noexcept;
    ArtifactStore & artifacts() noexcept;
    const ArtifactStore & artifacts() const noexcept;
    RuntimeCache & cache() noexcept;
    const RuntimeCache & cache() const noexcept;
    RuntimeWorkspace & workspace() noexcept;
    const RuntimeWorkspace & workspace() const noexcept;
    GraphExecutor & graph_executor() noexcept;
    const GraphExecutor & graph_executor() const noexcept;
    void mark_prepared();
    void require_prepared(std::string_view operation) const;
    // MoE graphs and expert caches are session-owned mutable state. Reject a
    // second request before it can change routes, caches or backbone state.
    std::unique_lock<std::mutex> acquire_moe_execution();
    void trace(engine::debug::LogLevel level, std::string_view category, std::string_view message) const;
    engine::debug::ScopeTimer profile(engine::debug::LogLevel level, std::string_view category, std::string_view name) const;
    const SessionOptions & options() const noexcept;

private:
    SessionOptions options_;
    engine::core::ExecutionContext execution_context_;
    ArtifactStore artifacts_;
    RuntimeCache cache_;
    RuntimeWorkspace workspace_;
    GraphExecutor graph_executor_;
    bool prepared_ = false;
    std::mutex moe_execution_mutex_;
};

}  // namespace engine::runtime
