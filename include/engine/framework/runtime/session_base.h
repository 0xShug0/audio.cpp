#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/artifacts.h"
#include "engine/framework/runtime/cache.h"
#include "engine/framework/runtime/graph_executor.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/workspace.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::runtime {

class RuntimeSessionBase {
public:
    explicit RuntimeSessionBase(const SessionOptions & options);
    // 共享 backend 模式（llama.cpp 单 context 多 slot）：`external_context` 非空时
    // 本 session 不自建 ExecutionContext/backend，而是借用外部持有的那个（所有权在
    // 调用方，通常是 model/scheduler 级，生命周期须长于本 session 及其 runtime）。
    // nullptr = 原行为（每个 session 自建自己的 context）。
    RuntimeSessionBase(
        const SessionOptions & options,
        std::shared_ptr<engine::core::ExecutionContext> external_context);
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
    void mark_prepared() noexcept;
    void require_prepared(std::string_view operation) const;
    void trace(engine::debug::LogLevel level, std::string_view category, std::string_view message) const;
    engine::debug::ScopeTimer profile(engine::debug::LogLevel level, std::string_view category, std::string_view name) const;
    const SessionOptions & options() const noexcept;

private:
    SessionOptions options_;
    // 本 session 的 backend context。默认自建（shared_ptr 持有）；共享模式外部传入。
    // 借用的外部 context 同样以 shared_ptr 持有，保证它在本 session 存活期间不析构。
    std::shared_ptr<engine::core::ExecutionContext> context_;
    ArtifactStore artifacts_;
    RuntimeCache cache_;
    RuntimeWorkspace workspace_;
    GraphExecutor graph_executor_;
    bool prepared_ = false;
};

}  // namespace engine::runtime
