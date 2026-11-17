#pragma once

// Built-in implementation details, also used by independent numeric fixtures.
#include "engine/framework/moe_engine/placement_policy.h"
#include "engine/framework/moe_engine/moe_policy.h"

namespace engine::modules {

struct MoeCpuShareState {
    uint64_t cache_epoch = 0;
    uint32_t credit = 0;
    int64_t tokens_seen = 0;
    std::vector<float> frequency;
    std::vector<int64_t> last_use;
};
// Opt-in execution policy, separate from weight residency. GPU hits precede
// misses, so bounded staging cannot evict a hit that has not been consumed.
// Storage is reserved once; updates reuse it. Cache invalidation resets credit.
struct StagedMoeConfig;

// Measured costs for this prepared layer/shape. CPU entries match its actual
// graphs; zero means unsampled. Native GPU nullopt means no original region.
struct MoePlacementCosts {
    struct CpuBucket { int64_t rows; double ms; };
    std::vector<CpuBucket> cpu;
    double cpu_scale = 1;
    double gpu_ms_per_row = 0;
    double mixed_gpu_ms_per_row = 0;
    double output_copy_ms = 0;
    double input_submit_ms = 0;
    double upload_ms_per_expert = 0;
    double mixed_upload_ms_per_expert = 0;
    double serial_ms = 0;
    std::optional<double> native_gpu_ms_per_row;
};

class MoePlacementPlan : public MoePlacementDecision {
public:
    explicit MoePlacementPlan(size_t experts, MoeCpuShareState * history = nullptr);
    void update(const std::vector<int32_t> & ids, const std::vector<int32_t> & resident,
        uint64_t cache_epoch, float cpu_share, int64_t tokens = 1);
    // Distinct logical experts; assignments retain original output-lane indices.
    void update_cost_aware(const std::vector<int32_t> & ids, const std::vector<int32_t> & resident,
        const std::vector<std::vector<int32_t>> & assignments,
        const MoePlacementCosts & costs, const StagedMoeConfig & config);
    // True only when unknown candidate costs force CPU execution to sample them.
    bool calibration_probe = false;
    struct ChosenPrediction {
        double cpu_ms; // Unscaled submission-to-completion latency.
        double gpu_branch_ms;
        double serial_ms;
        double gpu_baseline_ms;
    };
    std::optional<ChosenPrediction> chosen_prediction;
private:
    std::vector<int32_t> misses_;
    std::vector<uint8_t> seen_;
    MoeCpuShareState owned_history_;
    MoeCpuShareState * history_;
};

struct MoeCostDiagnostics {
    double planning_ms = 0;
    uint64_t cpu_probe_calls = 0, cpu_probe_assignments = 0;
    uint64_t cpu_chosen_calls = 0, cpu_chosen_assignments = 0;
    double cpu_probe_ms = 0, cpu_chosen_ms = 0;
    // Chosen cost-policy phases only; CPU latency is unscaled and includes
    // worker scheduling. GPU branch wall is not isolated kernel time.
    double chosen_cpu_predicted_ms = 0, chosen_cpu_observed_ms = 0;
    double chosen_gpu_branch_predicted_ms = 0, chosen_gpu_branch_observed_ms = 0;
    double chosen_serial_predicted_ms = 0, chosen_serial_observed_ms = 0;
    double chosen_score_ms = 0, chosen_observed_ms = 0;
    double chosen_gpu_baseline_ms = 0;
    ~MoeCostDiagnostics();
};

core::MoePlacementDescriptor prepare_cpu_share(float share);
core::MoePlacementDescriptor prepare_cost_aware(float scale);
MoeLayerPlacementPolicy make_cost_aware_layer(float scale, size_t experts,
    std::shared_ptr<MoeCostDiagnostics> diagnostics);

} // namespace engine::modules
