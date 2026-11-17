#include "placement_policy_internal.h"
#include "engine/framework/moe_engine/staged_moe_runtime.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <utility>
#include "engine/framework/debug/profiler.h"

namespace engine::modules {

MoeCostDiagnostics::~MoeCostDiagnostics() {
    if (planning_ms) debug::timing_log_scalar("framework.moe.planning_ms", planning_ms);
    if (cpu_probe_calls || cpu_chosen_calls) {
        debug::timing_log_scalar("framework.moe.cpu_probe_calls", cpu_probe_calls);
        debug::timing_log_scalar("framework.moe.cpu_probe_assignments", cpu_probe_assignments);
        debug::timing_log_scalar("framework.moe.cpu_probe_ms", cpu_probe_ms);
        debug::timing_log_scalar("framework.moe.cpu_chosen_calls", cpu_chosen_calls);
        debug::timing_log_scalar("framework.moe.cpu_chosen_assignments", cpu_chosen_assignments);
        debug::timing_log_scalar("framework.moe.cpu_chosen_ms", cpu_chosen_ms);
    }
    if (cpu_chosen_calls) {
        debug::timing_log_scalar("framework.moe.chosen_cpu_predicted_ms", chosen_cpu_predicted_ms);
        debug::timing_log_scalar("framework.moe.chosen_cpu_observed_ms", chosen_cpu_observed_ms);
        debug::timing_log_scalar("framework.moe.chosen_gpu_branch_predicted_ms", chosen_gpu_branch_predicted_ms);
        debug::timing_log_scalar("framework.moe.chosen_gpu_branch_observed_ms", chosen_gpu_branch_observed_ms);
        debug::timing_log_scalar("framework.moe.chosen_serial_predicted_ms", chosen_serial_predicted_ms);
        debug::timing_log_scalar("framework.moe.chosen_serial_observed_ms", chosen_serial_observed_ms);
        debug::timing_log_scalar("framework.moe.chosen_score_ms", chosen_score_ms);
        debug::timing_log_scalar("framework.moe.chosen_observed_ms", chosen_observed_ms);
        debug::timing_log_scalar("framework.moe.chosen_gpu_baseline_ms", chosen_gpu_baseline_ms);
    }
}

MoePlacementPlan::MoePlacementPlan(size_t experts, MoeCpuShareState * history)
    : seen_(experts), history_(history) {
    if (!experts) throw std::invalid_argument("MoE placement requires an expert count");
    if (!history) {
        owned_history_.frequency.assign(experts, 0);
        owned_history_.last_use.assign(experts, -1);
    } else if (history_->frequency.size() != experts || history_->last_use.size() != experts) {
        throw std::invalid_argument("MoE placement history does not match the logical expert bank");
    }
    gpu.reserve(experts);
    cpu.reserve(experts);
    misses_.reserve(experts);
}

void MoePlacementPlan::update(const std::vector<int32_t> & ids, const std::vector<int32_t> & resident,
    uint64_t cache_epoch, float cpu_share, int64_t tokens) {
    calibration_probe = false;
    chosen_prediction.reset();
    if (!std::isfinite(cpu_share) || cpu_share < 0 || cpu_share > 1 || resident.empty() || tokens <= 0)
        throw std::invalid_argument("invalid MoE placement policy");
    for (auto id : ids)
        if (id < 0 || static_cast<size_t>(id) >= seen_.size()) throw std::invalid_argument("MoE expert ID out of range");
    auto & history = history_ ? *history_ : owned_history_;
    if (history.cache_epoch != cache_epoch) {
        history.cache_epoch = cache_epoch;
        history.credit = 0;
        history.tokens_seen = 0;
        std::fill(history.frequency.begin(), history.frequency.end(), 0);
        std::fill(history.last_use.begin(), history.last_use.end(), -1);
    }
    gpu.clear(); cpu.clear(); misses_.clear();
    std::fill(seen_.begin(), seen_.end(), uint8_t{0});
    for (auto id : ids) {
        if (seen_[static_cast<size_t>(id)]) continue;
        seen_[static_cast<size_t>(id)] = 1;
        // Same CRF update as the reference controller: one use per distinct
        // expert at the current token clock, including cache hits.
        history.frequency[id] = history.frequency[id] *
            std::exp2(-(history.tokens_seen - history.last_use[id]) / 128.0F) + 1.0F;
        history.last_use[id] = history.tokens_seen;
        if (std::find(resident.begin(), resident.end(), id) != resident.end()) gpu.push_back(id);
        else misses_.push_back(id);
    }
    // Match fixed-point carry and full-cache-at-entry eligibility. Bounded
    // multi-group execution remains generic; there is no token-count bypass.
    const uint32_t share = std::find(resident.begin(), resident.end(), -1) == resident.end()
        ? static_cast<uint32_t>(std::lround(cpu_share * 65536.0F)) : 0;
    const uint64_t credit = static_cast<uint64_t>(misses_.size()) * share + history.credit;
    history.credit = static_cast<uint32_t>(credit & 0xffff);
    const auto count = static_cast<size_t>(credit >> 16);
    if (count) {
        std::sort(misses_.begin(), misses_.end(), [&](int32_t a, int32_t b) {
            const float fa = history.frequency[a], fb = history.frequency[b];
            return fa != fb ? fa < fb : a < b;
        });
        cpu.insert(cpu.end(), misses_.begin(), misses_.begin() + static_cast<std::ptrdiff_t>(count));
        for (auto id : cpu) seen_[id] = 2;
    }
    // Hits precede GPU misses, preserving bounded-cache protection. Keep
    // original route order for GPU work; CPU results scatter to logical lanes.
    for (auto id : ids) {
        if (seen_[id] == 1 && std::find(resident.begin(), resident.end(), id) == resident.end()) {
            gpu.push_back(id);
            seen_[id] = 3;
        }
    }
    history.tokens_seen += tokens;
}

void MoePlacementPlan::update_cost_aware(const std::vector<int32_t> & ids,
    const std::vector<int32_t> & resident, const std::vector<std::vector<int32_t>> & assignments,
    const MoePlacementCosts & costs, const StagedMoeConfig & config) {
    calibration_probe = false;
    chosen_prediction.reset();
    gpu.clear(); cpu.clear(); misses_.clear();
    for (auto id : ids) {
        if (std::find(resident.begin(), resident.end(), id) != resident.end()) gpu.push_back(id);
        else misses_.push_back(id);
    }
    gpu.insert(gpu.end(), misses_.begin(), misses_.end());
    if (misses_.empty()) return;
    // Establish the warmed GPU baseline before probing CPU alternatives.
    // One-use graphs then perform no speculative CPU work; repeated graphs
    // supply their normal direct/capture/replay executions for calibration.
    const bool native_baseline = costs.native_gpu_ms_per_row &&
        gpu.size() <= static_cast<size_t>(config.cache_slots);
    if (native_baseline ? *costs.native_gpu_ms_per_row == 0 : costs.gpu_ms_per_row == 0) return;
    const auto gpu_work = [&] {
        size_t work = 0;
        for (size_t first = 0; first < gpu.size(); first += config.cache_slots) {
            size_t maximum = 0;
            for (size_t i = first; i < std::min(gpu.size(), first + static_cast<size_t>(config.cache_slots)); ++i)
                maximum = std::max(maximum, assignments[gpu[i]].size());
            for (size_t begin = 0; begin < maximum; begin += config.rows_per_expert.back()) {
                const auto needed = std::min(maximum - begin, static_cast<size_t>(config.rows_per_expert.back()));
                work += config.cache_slots * *std::lower_bound(config.rows_per_expert.begin(), config.rows_per_expert.end(), needed);
            }
        }
        return work;
    };
    double best = gpu_work() * costs.gpu_ms_per_row + misses_.size() * costs.upload_ms_per_expert + costs.output_copy_ms + costs.input_submit_ms;
    if (native_baseline) {
        size_t rows = 0;
        for (auto id : gpu) rows += assignments[id].size();
        best = rows * *costs.native_gpu_ms_per_row + misses_.size() * costs.upload_ms_per_expert;
    }
    const auto gpu_baseline_ms = best;
    size_t best_count = 0;
    int32_t best_single = -1;
    const auto evaluate = [&](size_t rows, int32_t single) {
        const auto bucket = std::lower_bound(costs.cpu.begin(), costs.cpu.end(), rows,
            [](const auto & entry, size_t count) { return static_cast<size_t>(entry.rows) < count; });
        if (bucket == costs.cpu.end()) throw std::logic_error("MoE cost model has no prepared CPU bucket for route");
        const auto gpu_misses = misses_.size() - cpu.size();
        // Unknown candidates are measured by executing the requested work once;
        // no extrapolation from a different CPU bucket or GPU context.
        if (bucket->ms == 0 || (!gpu.empty() && costs.mixed_gpu_ms_per_row == 0) ||
            (gpu_misses && costs.mixed_upload_ms_per_expert == 0)) {
            calibration_probe = true;
            chosen_prediction.reset();
            return false;
        }
        const double cpu_ms = bucket->ms * costs.cpu_scale;
        const double gpu_ms = gpu_work() * costs.mixed_gpu_ms_per_row +
            gpu_misses * costs.mixed_upload_ms_per_expert;
        const double completion = costs.serial_ms + std::max(cpu_ms, gpu_ms);
        if (completion < best) {
            best = completion;
            best_count = cpu.size();
            best_single = single;
            chosen_prediction = ChosenPrediction{bucket->ms, gpu_ms, costs.serial_ms, gpu_baseline_ms};
        }
        return true;
    };
    size_t rows = 0;
    // Complete small-miss prefixes preserve profitable whole-batch choices.
    for (auto it = misses_.rbegin(); it != misses_.rend(); ++it) {
        const auto id = *it;
        gpu.erase(std::find(gpu.begin(), gpu.end(), id));
        cpu.push_back(id);
        rows += assignments[id].size();
        if (!evaluate(rows, -1)) return;
    }
    // An individual large miss can dominate padded GPU work. Test every miss
    // independently as well, without model/shape gates or new route storage.
    for (auto id : misses_) {
        cpu.assign(1, id);
        gpu.clear();
        for (auto routed : ids)
            if (std::find(resident.begin(), resident.end(), routed) != resident.end()) gpu.push_back(routed);
        for (auto missed : misses_) if (missed != id) gpu.push_back(missed);
        if (!evaluate(assignments[id].size(), id)) return;
    }
    cpu.clear();
    if (best_single >= 0) cpu.push_back(best_single);
    else cpu.insert(cpu.end(), misses_.rbegin(), misses_.rbegin() + static_cast<std::ptrdiff_t>(best_count));
    gpu.clear();
    for (auto id : ids)
        if (std::find(resident.begin(), resident.end(), id) != resident.end()) gpu.push_back(id);
    for (auto id : misses_)
        if (std::find(cpu.begin(), cpu.end(), id) == cpu.end()) gpu.push_back(id);
}

namespace {
constexpr core::MoePlacementRequirements share_requirements{true, false};
constexpr core::MoePlacementRequirements cost_requirements{true, true};
struct MoeCostModel {
    MoePlacementCosts & values;
    explicit MoeCostModel(MoePlacementCosts & values) : values(values) {}
    unsigned native_executions = 0;
    std::array<unsigned, 2> staged_executions{};
    static void observe(double & estimate, double sample) {
        estimate = estimate == 0 ? sample : 0.8 * estimate + 0.2 * sample;
    }
    void observe_gpu(size_t rows, uint64_t loads, double upload_ms, double elapsed_ms, bool mixed) {
        // Initial direct execution and capture are not steady graph throughput.
        if (rows) {
            auto & executions = staged_executions[mixed ? 1 : 0];
            if (executions < 2) ++executions;
            else observe(mixed ? values.mixed_gpu_ms_per_row : values.gpu_ms_per_row,
                (elapsed_ms - upload_ms) / rows);
        }
        if (loads) observe(mixed ? values.mixed_upload_ms_per_expert : values.upload_ms_per_expert, upload_ms / loads);
    }
    void observe_cpu(const MoeExecutionSample & sample) {
        const auto bucket = std::lower_bound(values.cpu.begin(), values.cpu.end(), sample.cpu_rows,
            [](const auto & entry, size_t rows) { return static_cast<size_t>(entry.rows) < rows; });
        if (bucket == values.cpu.end() || static_cast<size_t>(bucket->rows) != sample.cpu_rows)
            throw std::logic_error("MoE CPU execution did not match a prepared cost bucket");
        observe(bucket->ms, sample.cpu_ms);
        observe(values.serial_ms, sample.input_ms + sample.merge_ms + sample.handoff_ms);
    }
};

class SharePlacementPolicy : public MoePlacementPolicy {
public:
    StagedMoeConfig shape;
    std::shared_ptr<MoeCpuShareState> history;
    MoePlacementPlan plan;
    float share;
    SharePlacementPolicy(const StagedMoeConfig & shape, size_t experts, std::shared_ptr<MoeCpuShareState> history, float share)
        : shape(shape), history(std::move(history)), plan(experts, this->history.get()), share(share) {}
    const MoePlacementDecision & select(const std::vector<int32_t> & ids, const std::vector<int32_t> & resident,
        uint64_t epoch, const std::vector<std::vector<int32_t>> &, bool) override {
        plan.update(ids, resident, epoch, share, shape.tokens);
        return plan;
    }
    void observe(const MoePlacementObservation &) override {}
};

struct MoeShapeCosts {
    StagedMoeConfig shape;
    MoeExpertConfig math;
    MoePlacementCosts direct, region;
    MoeShapeCosts(const StagedMoeConfig & shape, const MoeExpertConfig & math, float scale)
        : shape(shape), math(math) {
        for (auto * values : {&direct, &region}) {
            values->cpu_scale = scale;
            const auto rows = shape.tokens * shape.selected;
            for (int64_t count = 1; count < rows; count *= 2) values->cpu.push_back({count, 0});
            values->cpu.push_back({rows, 0});
        }
    }
};

class CostPlacementPolicy final : public MoePlacementPolicy {
public:
    StagedMoeConfig shape;
    MoePlacementPlan plan;
    std::shared_ptr<MoeShapeCosts> estimates;
    MoeCostModel direct, region;
    std::shared_ptr<MoeCostDiagnostics> diagnostics;
    CostPlacementPolicy(const StagedMoeConfig & shape, size_t experts, std::shared_ptr<MoeShapeCosts> estimates,
        std::shared_ptr<MoeCostDiagnostics> diagnostics)
        : shape(shape), plan(experts), estimates(std::move(estimates)),
          direct(this->estimates->direct), region(this->estimates->region), diagnostics(std::move(diagnostics)) {}
    const MoePlacementDecision & select(const std::vector<int32_t> & ids, const std::vector<int32_t> & resident,
        uint64_t, const std::vector<std::vector<int32_t>> & assignments, bool native_region) override {
        if (native_region && !region.values.native_gpu_ms_per_row) region.values.native_gpu_ms_per_row = 0;
        const auto started = std::chrono::steady_clock::now();
        plan.update_cost_aware(ids, resident, assignments, (native_region ? region : direct).values, shape);
        diagnostics->planning_ms += debug::elapsed_ms(started);
        plan.time_native = region.values.native_gpu_ms_per_row && *region.values.native_gpu_ms_per_row == 0 && region.native_executions >= 2;
        plan.time_output_copy = region.values.output_copy_ms == 0;
        return plan;
    }
    void observe(const MoePlacementObservation & observation) override {
        if (observation.kind == MoePlacementObservation::Kind::InputSubmission) {
            MoeCostModel::observe(region.values.input_submit_ms, observation.elapsed_ms.value());
            return;
        }
        if (observation.kind == MoePlacementObservation::Kind::OutputCompletion) {
            region.values.output_copy_ms = observation.elapsed_ms.value();
            return;
        }
        if (observation.kind == MoePlacementObservation::Kind::CpuCompletion) {
            observe_cpu(observation);
            return;
        }
        auto & cost = observation.region ? region : direct;
        if (!observation.native) {
            cost.observe_gpu(observation.rows, observation.loads, observation.upload_ms,
                observation.elapsed_ms.value(), observation.mixed);
            return;
        }
        const bool warming = cost.values.native_gpu_ms_per_row && *cost.values.native_gpu_ms_per_row == 0;
        if (warming && observation.elapsed_ms)
            cost.values.native_gpu_ms_per_row = (*observation.elapsed_ms - observation.upload_ms) / observation.rows;
        if (warming) ++cost.native_executions;
        if (observation.loads)
            MoeCostModel::observe(cost.values.upload_ms_per_expert, observation.upload_ms / observation.loads);
    }
    void observe_cpu(const MoePlacementObservation & observation) {
        auto & stats = *diagnostics;
        auto & cost = observation.region ? region : direct;
        const auto & sample = observation.cpu;
        if (!sample.cpu_rows) return;
        if (plan.calibration_probe) {
            ++stats.cpu_probe_calls;
            stats.cpu_probe_assignments += observation.cpu_assignments;
            stats.cpu_probe_ms += observation.cpu_compute_ms;
        } else {
            ++stats.cpu_chosen_calls;
            stats.cpu_chosen_assignments += observation.cpu_assignments;
            stats.cpu_chosen_ms += observation.cpu_compute_ms;
            const auto & predicted = plan.chosen_prediction.value();
            const auto serial_ms = sample.input_ms + sample.merge_ms + sample.handoff_ms;
            stats.chosen_cpu_predicted_ms += predicted.cpu_ms;
            stats.chosen_cpu_observed_ms += sample.cpu_ms;
            stats.chosen_gpu_branch_predicted_ms += predicted.gpu_branch_ms;
            stats.chosen_gpu_branch_observed_ms += sample.gpu_branch_ms;
            stats.chosen_serial_predicted_ms += predicted.serial_ms;
            stats.chosen_serial_observed_ms += serial_ms;
            stats.chosen_score_ms += predicted.serial_ms +
                std::max(predicted.cpu_ms * cost.values.cpu_scale, predicted.gpu_branch_ms);
            stats.chosen_observed_ms += serial_ms + std::max(sample.cpu_ms, sample.gpu_branch_ms);
            stats.chosen_gpu_baseline_ms += predicted.gpu_baseline_ms;
        }
        cost.observe_cpu(sample);

    }
};

bool same_shape(const StagedMoeConfig & a, const StagedMoeConfig & b) {
    return a.hidden_size == b.hidden_size && a.input_size == b.input_size && a.tokens == b.tokens &&
        a.selected == b.selected && a.experts == b.experts && a.cache_slots == b.cache_slots &&
        a.rows_per_expert == b.rows_per_expert && a.scatter_index_type == b.scatter_index_type;
}
bool same_math(const MoeExpertConfig & a, const MoeExpertConfig & b) {
    return a.activation == b.activation && a.fused_swiglu == b.fused_swiglu &&
        a.round_projections == b.round_projections && a.round_activation == b.round_activation &&
        a.round_product == b.round_product && a.round_output == b.round_output;
}
} // namespace

core::MoePlacementDescriptor prepare_cpu_share(float share) {
    if (!std::isfinite(share) || share < 0 || share > 1)
        throw std::invalid_argument("MoE CPU share must be between zero and one");
    if (share == 0) return {};
    return {share_requirements, [share]() -> core::MoePlacementFactory {
        return [share](size_t experts) -> MoeLayerPlacementPolicy {
            auto history = std::make_shared<MoeCpuShareState>();
            history->frequency.assign(experts, 0);
            history->last_use.assign(experts, -1);
            return [history, experts, share](const StagedMoeConfig & shape, const MoeExpertConfig &) {
                return std::make_shared<SharePlacementPolicy>(shape, experts, history, share);
            };
        };
    }};
}

MoeLayerPlacementPolicy make_cost_aware_layer(float scale, size_t experts,
    std::shared_ptr<MoeCostDiagnostics> diagnostics) {
    auto estimates = std::make_shared<std::vector<std::shared_ptr<MoeShapeCosts>>>();
    return [experts, scale, estimates, diagnostics](const StagedMoeConfig & shape, const MoeExpertConfig & math)
        -> std::shared_ptr<MoePlacementPolicy> {
        for (const auto & entry : *estimates)
            if (same_shape(entry->shape, shape) && same_math(entry->math, math))
                return std::make_shared<CostPlacementPolicy>(shape, experts, entry, diagnostics);
        auto entry = std::make_shared<MoeShapeCosts>(shape, math, scale);
        estimates->push_back(entry);
        return std::make_shared<CostPlacementPolicy>(shape, experts, std::move(entry), diagnostics);
    };
}

core::MoePlacementDescriptor prepare_cost_aware(float scale) {
    if (!std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("cpu_cost_scale must be finite and positive");
    return {cost_requirements, [scale]() -> core::MoePlacementFactory {
        auto diagnostics = std::make_shared<MoeCostDiagnostics>();
        return [scale, diagnostics](size_t experts) {
            return make_cost_aware_layer(scale, experts, diagnostics);
        };
    }};
}
} // namespace engine::modules
