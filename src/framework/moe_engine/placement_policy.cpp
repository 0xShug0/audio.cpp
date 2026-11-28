#include "engine/framework/moe_engine/moe_policy.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::moe {
namespace {
void validate_parameters(const PolicyParameters & parameters, std::initializer_list<const char *> keys) {
    for (const auto & parameter : parameters) {
        if (std::none_of(keys.begin(), keys.end(), [&](const char * key) { return parameter.first == key; }))
            throw std::invalid_argument("unknown MoE placement parameter: " + parameter.first);
    }
}
class CpuShare final : public PlacementPolicy {
public:
    explicit CpuShare(double share) : share_(share) {}
    void prepare_program(uint64_t program) override { carries_.emplace(program, Carry{}); }
    void select(const PlacementInput & input, PlacementDecision & out) override {
        auto & state = carries_.at(input.program);
        if (state.epoch != input.cache_epoch) state = {input.cache_epoch, 0};
        ordered_.assign(input.experts.begin(), input.experts.end());
        // Legacy placement ranks work by assignment count, with logical IDs
        // breaking ties, independently of the routing tensor's occurrence order.
        std::sort(ordered_.begin(), ordered_.end(), [&](int32_t a, int32_t b) {
            const auto x = input.assignment_counts.at(static_cast<size_t>(a));
            const auto y = input.assignment_counts.at(static_cast<size_t>(b));
            return x != y ? x > y : a < b;
        });
        out.cpu.clear(); out.gpu.clear(); misses_.clear();
        size_t vacant = static_cast<size_t>(std::count(input.resident.begin(), input.resident.end(), -1));
        for (auto id : ordered_) {
            if (std::find(input.resident.begin(), input.resident.end(), id) != input.resident.end()) out.gpu.push_back(id);
            else if (vacant) { --vacant; misses_.push_back(id); }
            else {
                state.fraction += share_;
                if (state.fraction >= 1) { state.fraction -= 1; out.cpu.push_back(id); }
                else misses_.push_back(id);
            }
        }
        out.gpu.insert(out.gpu.end(), misses_.begin(), misses_.end());
    }
    void event(const PolicyNotification & notification) override {
        if (notification.event == PolicyEvent::ProgramRetired) carries_.erase(notification.program);
        else if (notification.event == PolicyEvent::CacheInvalidated)
            for (auto & item : carries_) item.second = {notification.cache_epoch, 0};
    }
    void reserve(size_t count) { misses_.reserve(count); ordered_.reserve(count); }
private:
    struct Carry { uint64_t epoch = 0; double fraction = 0; };
    double share_;
    std::unordered_map<uint64_t, Carry> carries_;
    std::vector<int32_t> misses_, ordered_;
};
class CostAware final : public PlacementPolicy {
public:
    CostAware(double cpu, double gpu, double upload) : initial_{cpu, gpu, upload} {}
    void prepare_program(uint64_t program) override { costs_.emplace(program, initial_); }
    void event(const PolicyNotification & notification) override {
        if (notification.event == PolicyEvent::ProgramRetired) costs_.erase(notification.program);
    }
    void select(const PlacementInput & input, PlacementDecision & out) override {
        auto & costs = costs_.at(input.program);
        out.cpu.clear(); out.gpu.clear();
        double cpu_work = 0, gpu_work = 0;
        for (auto id : input.experts) {
            if (std::find(input.resident.begin(), input.resident.end(), id) != input.resident.end()) {
                out.gpu.push_back(id);
                gpu_work += input.assignment_counts.at(static_cast<size_t>(id)) * costs.gpu_per_row;
            }
        }
        for (auto id : input.experts) {
            if (std::find(input.resident.begin(), input.resident.end(), id) != input.resident.end()) continue;
            const auto rows = input.assignment_counts.at(static_cast<size_t>(id));
            const double gpu_cost = rows * costs.gpu_per_row + input.expert_bytes / costs.bytes_per_ms;
            if (cpu_work + rows * costs.cpu_per_row < gpu_work + gpu_cost) {
                out.cpu.push_back(id); cpu_work += rows * costs.cpu_per_row;
            } else { out.gpu.push_back(id); gpu_work += gpu_cost; }
        }
    }

    void observe(const ExecutionObservation & observation) override {
        auto & costs = costs_.at(observation.program);
        if (observation.cpu_rows && observation.cpu_ms > 0)
            costs.cpu_per_row = 0.8 * costs.cpu_per_row + 0.2 * observation.cpu_ms / observation.cpu_rows;
        if (observation.gpu_rows && observation.gpu_ms > 0)
            costs.gpu_per_row = 0.8 * costs.gpu_per_row + 0.2 * observation.gpu_ms / observation.gpu_rows;
        if (observation.uploaded_bytes && observation.upload_ms > 0)
            costs.bytes_per_ms = 0.8 * costs.bytes_per_ms + 0.2 * observation.uploaded_bytes / observation.upload_ms;
    }
private:
    struct Costs { double cpu_per_row, gpu_per_row, bytes_per_ms; };
    Costs initial_;
    std::unordered_map<uint64_t, Costs> costs_;
};
}
PreparedPolicy prepare_cpu_share(const PolicyParameters & parameters) {
    validate_parameters(parameters, {"share"});
    const double share = runtime::parse_finite_float_option(parameters, {"share"}).value_or(0);
    if (share < 0 || share > 1) throw std::invalid_argument("MoE placement share must be between zero and one");
    PreparedPolicy prepared;
    prepared.requirements.cpu_execution = share > 0;
    prepared.active_placement = share > 0;
    prepared.placement = [share](size_t experts) {
        auto policy = std::make_unique<CpuShare>(share); policy->reserve(experts); return policy;
    };
    return prepared;
}
PreparedPolicy prepare_cost_aware(const PolicyParameters & parameters) {
    validate_parameters(parameters, {"cpu_ms_per_row", "gpu_ms_per_row", "upload_bytes_per_ms"});
    const double cpu = runtime::parse_finite_float_option(parameters, {"cpu_ms_per_row"}).value_or(0.02F);
    const double gpu = runtime::parse_finite_float_option(parameters, {"gpu_ms_per_row"}).value_or(0.002F);
    const double upload = runtime::parse_finite_float_option(parameters, {"upload_bytes_per_ms"}).value_or(10000000.F);
    if (cpu <= 0 || gpu <= 0 || upload <= 0) throw std::invalid_argument("MoE cost estimates must be positive");
    PreparedPolicy prepared;
    prepared.requirements.cpu_execution = true;
    prepared.requirements.phase_times = true;
    prepared.requirements.completed_work = true;
    prepared.active_placement = true;
    prepared.placement = [cpu, gpu, upload](size_t) { return std::make_unique<CostAware>(cpu, gpu, upload); };
    return prepared;
}
} // namespace engine::moe
