#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::moe {

using PolicyParameters = std::unordered_map<std::string, std::string>;
struct PolicySelection {
    std::string name = "default";
    PolicyParameters parameters;
};
struct PolicyConfig {
    PolicySelection placement;
    PolicySelection eviction;
};
struct PolicyRequirements {
    bool cpu_execution = false;
    bool phase_times = false;
    bool cache_accesses = false;
};
struct PlacementInput {
    const std::vector<int32_t> & experts;
    const std::vector<size_t> & assignment_counts;
    const std::vector<int32_t> & resident;
    uint64_t cache_epoch = 0;
    size_t expert_bytes = 0;
    uint64_t program = 0;
};
struct PlacementDecision {
    std::vector<int32_t> gpu;
    std::vector<int32_t> cpu;
};
struct ExecutionObservation {
    size_t cpu_rows = 0;
    size_t gpu_rows = 0;
    size_t uploaded_bytes = 0;
    double cpu_ms = 0;
    double gpu_ms = 0;
    double upload_ms = 0;
    uint64_t program = 0;
};
enum class PolicyEvent { GenerationBegin, CacheInvalidated, ProgramRetired, ExecutionFailed, ExecutionReleased };
struct PolicyNotification {
    PolicyEvent event;
    uint64_t program = 0;
    uint64_t cache_epoch = 0;
    size_t cpu_rows = 0, gpu_rows = 0;
};

// Lists partition unique logical experts exactly once. Decision storage remains
// valid until execution and requested observations complete. Existing GPU hits
// stay on GPU. Observations depend on declared requirements, not every call.
class PlacementPolicy {
public:
    virtual ~PlacementPolicy() = default;
    virtual void select(const PlacementInput &, PlacementDecision &) = 0;
    virtual void prepare_program(uint64_t) {}
    virtual void observe(const ExecutionObservation &) {}
    virtual void event(const PolicyNotification &) {}
};
struct EvictionCandidate {
    size_t slot = 0;
    int32_t expert = -1;
    uint64_t last_used = 0;
};
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    // Candidates are unpinned slots only; engine mechanisms enforce capacity.
    virtual size_t select(const std::vector<EvictionCandidate> &) const = 0;
    virtual void accessed(int32_t expert, size_t assignments) = 0;
};
struct PreparedPolicy {
    PolicyRequirements requirements;
    std::function<std::unique_ptr<PlacementPolicy>(size_t)> placement;
    std::function<std::unique_ptr<EvictionPolicy>(size_t)> eviction;
    bool active_placement = false;
};

// Pure resolution; no model/session/backend state and no named combinations.
PreparedPolicy prepare_policy(const PolicyConfig & config);

} // namespace engine::moe
