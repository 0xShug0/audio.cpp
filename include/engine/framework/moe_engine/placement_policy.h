#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::modules {

struct MoePlacementDecision {
    std::vector<int32_t> gpu;
    std::vector<int32_t> cpu;
    bool time_native = false;
    bool time_output_copy = false;
};

struct MoeExecutionSample {
    size_t cpu_rows = 0;
    double input_ms = 0, cpu_ms = 0, gpu_branch_ms = 0, merge_ms = 0, handoff_ms = 0;
};

struct MoePlacementObservation {
    enum class Kind { GpuCompletion, CpuCompletion, InputSubmission, OutputCompletion };
    Kind kind;
    bool region = false, mixed = false, native = false;
    size_t rows = 0;
    uint64_t loads = 0;
    double upload_ms = 0;
    // Native warm-up submissions are not timed; zero is a valid measurement.
    std::optional<double> elapsed_ms{};
    MoeExecutionSample cpu{};
    uint64_t cpu_assignments = 0;
    double cpu_compute_ms = 0;
};

class MoePlacementPolicy {
public:
    virtual ~MoePlacementPolicy() = default;
    // gpu and cpu must partition ids: each routed logical expert appears
    // exactly once, with no omissions, duplicates or additional experts.
    // Return policy-owned storage that stays valid and unchanged through
    // execution and its observation callbacks, until the next select().
    virtual const MoePlacementDecision & select(const std::vector<int32_t> & ids,
        const std::vector<int32_t> & resident, uint64_t epoch,
        const std::vector<std::vector<int32_t>> & assignments, bool region) = 0;
    // Delivery depends on the descriptor's timing requirements and execution
    // path. This is not an unconditional completion callback: do not rely on
    // receiving an observation after every decision.
    virtual void observe(const MoePlacementObservation & observation) = 0;
};

} // namespace engine::modules
