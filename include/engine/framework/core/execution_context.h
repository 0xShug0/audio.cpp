#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/moe_policy.h"

namespace engine::core {

class ExecutionContext {
public:
    explicit ExecutionContext(const BackendConfig & config, MoePolicyConfig policy = {});
    ~ExecutionContext();

    ExecutionContext(const ExecutionContext &) = delete;
    ExecutionContext & operator=(const ExecutionContext &) = delete;
    ExecutionContext(ExecutionContext && other) noexcept;
    ExecutionContext & operator=(ExecutionContext && other) noexcept;

    ggml_backend_t backend() const noexcept;
    const BackendConfig & config() const noexcept;
    BackendType backend_type() const noexcept;
    // Model session policy, separate from backend selection and configuration.
    const MoePolicyConfig & moe_policy() const noexcept { return moe_policy_; }
    // Set by the shared expert-resource integration, not individual options.
    void register_moe_resources() noexcept { moe_resources_registered_ = true; }
    bool has_moe_resources() const noexcept { return moe_resources_registered_; }
    bool uses_host_graph_plan() const noexcept;
    BackendMemorySnapshot memory_snapshot() const;

private:
    BackendConfig config_;
    MoePolicyConfig moe_policy_;
    bool moe_resources_registered_ = false;
    ggml_backend_t backend_ = nullptr;
};

}  // namespace engine::core
