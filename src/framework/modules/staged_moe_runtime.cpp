#include "engine/framework/modules/staged_moe_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include <ggml-alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <variant>
#include <tuple>
#include <unordered_set>

namespace engine::modules {

MoePlacementPlan::MoePlacementPlan(size_t experts, core::MoeCpuShareState * history)
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
    if (costs.gpu_ms_per_row == 0 &&
        (!costs.native_gpu_ms_per_row || *costs.native_gpu_ms_per_row == 0)) return;
    if (costs.native_gpu_ms_per_row && *costs.native_gpu_ms_per_row == 0 &&
        costs.gpu_ms_per_row > 0 && gpu.size() <= static_cast<size_t>(config.cache_slots)) return;
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
    if (costs.native_gpu_ms_per_row && *costs.native_gpu_ms_per_row > 0 &&
        gpu.size() <= static_cast<size_t>(config.cache_slots)) {
        size_t rows = 0;
        for (auto id : gpu) rows += assignments[id].size();
        best = rows * *costs.native_gpu_ms_per_row + misses_.size() * costs.upload_ms_per_expert;
    }
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
            (gpu_misses && costs.mixed_upload_ms_per_expert == 0)) return false;
        const double cpu_ms = bucket->ms * costs.cpu_scale;
        const double gpu_ms = gpu_work() * costs.mixed_gpu_ms_per_row +
            gpu_misses * costs.mixed_upload_ms_per_expert;
        const double completion = costs.serial_ms + std::max(cpu_ms, gpu_ms);
        if (completion < best) {
            best = completion;
            best_count = cpu.size();
            best_single = single;
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
struct MoeExecutionBoundary {
    ggml_cgraph * output_copy;
    double input_submit_ms;
    bool output_copied = false;
};

struct MoeExecutionSample {
    size_t cpu_rows = 0;
    double input_ms = 0, cpu_ms = 0, merge_ms = 0, handoff_ms = 0;
};

struct MoeCostModel {
    MoePlacementCosts values;
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
} // namespace

class StagedMoeRuntime::Impl {
public:
    struct Layer {
        core::DiskExpertWeightStore * store;
        size_t group;
        std::vector<ggml_cgraph *> graphs;
        std::vector<core::PreparedGraph::GraphId> cpu_graphs;
        std::vector<ggml_tensor *> cpu_outputs;
        std::unique_ptr<MoePlacementPlan> placement;
        bool cpu_staged = false;
        MoeCostModel cost{}, region_cost{};
    };
    struct Hybrid {
        core::ExecutionContext & execution;
        core::PreparedGraph graphs;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * source = nullptr;
        ggml_tensor * ids = nullptr;
        std::vector<int64_t> buckets;
        std::vector<ggml_cgraph *> scatter;
        std::vector<int32_t> source_rows, expert_ids;
        std::variant<std::vector<int32_t>, std::vector<int64_t>> target_rows;
        std::mutex mutex;
        std::condition_variable ready, finished;
        std::thread worker;
        bool stopping = false, pending = false;
        Layer * layer = nullptr;
        size_t bucket = 0;
        std::exception_ptr error;
        double compute_ms = 0;
        std::chrono::steady_clock::time_point submitted;
        MoeExecutionSample * sample = nullptr;

        Hybrid(core::ExecutionContext & cpu_execution, const StagedMoeConfig & config)
            : execution(cpu_execution),
              graphs(execution, {"framework.moe.cpu", 16 * 1024 * 1024, 128, 64 * ggml_tensor_overhead()}) {
            const auto rows = config.tokens * config.selected;
            for (int64_t count = 1; count < rows; count *= 2) buckets.push_back(count);
            buckets.push_back(rows);
            auto & ctx = graphs.persistent_build_context();
            hidden = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, config.input_size, config.tokens);
            source = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, rows);
            ids = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_I32, 1, rows);
            for (auto * tensor : {hidden, source, ids}) ggml_set_input(tensor);
            graphs.allocate_persistent();
            source_rows.resize(static_cast<size_t>(rows));
            expert_ids.resize(static_cast<size_t>(rows));
            if (config.scatter_index_type == GGML_TYPE_I64) target_rows.emplace<std::vector<int64_t>>();
            std::visit([rows](auto & targets) { targets.resize(static_cast<size_t>(rows)); }, target_rows);
        }
        ~Hybrid() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            ready.notify_one();
            if (worker.joinable()) worker.join();
        }
        void start() {
            worker = std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                    ready.wait(lock, [this] { return pending || stopping; });
                    if (stopping && !pending) return;
                    lock.unlock();
                    const auto started = std::chrono::steady_clock::now();
                    try { graphs.compute(layer->cpu_graphs[bucket]); }
                    catch (...) { error = std::current_exception(); }
                    compute_ms = debug::elapsed_ms(started);
                    if (sample) sample->cpu_ms = debug::elapsed_ms(submitted);
                    lock.lock();
                    pending = false;
                    finished.notify_one();
                }
            });
        }
        void launch(Layer & next_layer, size_t next_bucket, MoeExecutionSample * observation) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                layer = &next_layer;
                bucket = next_bucket;
                sample = observation;
                if (sample) submitted = std::chrono::steady_clock::now();
                error = nullptr;
                pending = true;
            }
            ready.notify_one();
        }
        void wait() {
            std::unique_lock<std::mutex> lock(mutex);
            finished.wait(lock, [this] { return !pending; });
            if (error) std::rethrow_exception(error);
        }
    };
    struct CpuStage {
        core::PreparedGraph graphs;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * source = nullptr;
        ggml_tensor * slots = nullptr;
        bool has_graphs = false;
        std::vector<float> batch_output, ordered_output;

        CpuStage(core::ExecutionContext & execution, const StagedMoeConfig & config)
            : graphs(execution, {"framework.moe.cpu_staged", 16 * 1024 * 1024, 128, 64 * ggml_tensor_overhead()}) {
            const auto rows = config.cache_slots * config.rows_per_expert.back();
            int64_t slot_rows = 0;
            for (auto count : config.rows_per_expert) slot_rows += config.cache_slots * count;
            auto & ctx = graphs.persistent_build_context();
            hidden = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, config.input_size, config.tokens);
            source = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, rows);
            slots = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_I32, 1, slot_rows);
            for (auto * tensor : {hidden, source, slots}) ggml_set_input(tensor);
            graphs.allocate_persistent();
            std::vector<int32_t> slot_ids(static_cast<size_t>(slot_rows));
            size_t offset = 0;
            for (auto count : config.rows_per_expert) {
                for (int64_t slot = 0; slot < config.cache_slots; ++slot)
                    std::fill_n(slot_ids.begin() + offset + slot * count, count, static_cast<int32_t>(slot));
                offset += static_cast<size_t>(config.cache_slots * count);
            }
            ggml_backend_tensor_set(slots, slot_ids.data(), 0, slot_ids.size() * sizeof(int32_t));
            batch_output.resize(static_cast<size_t>(config.hidden_size * rows));
            ordered_output.resize(static_cast<size_t>(config.hidden_size * config.selected * config.tokens));
        }
    };
    const core::ExecutionContext & execution;
    StagedMoeConfig config;
    float cpu_share = 0;
    bool hybrid_enabled = false;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free}, inputs{nullptr, ggml_free};
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::shared_ptr<core::GraphWorkspace> workspace;
    std::vector<Layer> layers;
    bool prepared = false;
    std::map<ggml_cgraph *, core::HostGraphPlan> plans;
    ggml_tensor * hidden = nullptr, * ids = nullptr, * weights = nullptr, * outputs = nullptr, * result = nullptr;
    ggml_tensor * source = nullptr, * target = nullptr, * slots = nullptr;
    std::vector<int32_t> logical_ids, source_rows;
    std::variant<std::vector<int32_t>, std::vector<int64_t>> target_rows;
    std::vector<std::vector<int32_t>> assignments;
    core::ExecutionContext * cpu_execution;
    std::unique_ptr<CpuStage> cpu_stage;
    std::unique_ptr<Hybrid> hybrid;
    ggml_tensor * cpu_output = nullptr, * cpu_target = nullptr;
    Statistics stats;

    uint64_t run_cpu_staged(Layer & binding, const std::vector<int32_t> & experts);
    uint64_t run_gpu(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary * boundary = nullptr);
    void select_route(Layer & binding);
    void select_region_route(Layer & binding);
    uint64_t run(Layer & binding, MoeExecutionBoundary * boundary = nullptr);
    std::vector<int32_t> routed_experts;
    uint64_t run_hybrid(Layer & binding, const MoePlacementPlan & placement, MoeExecutionSample * sample = nullptr, MoeExecutionBoundary * boundary = nullptr);
    uint64_t run_cpu_share(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary * boundary);
    uint64_t run_cost_aware(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary * boundary);
    uint64_t run_gpu_observed(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary * boundary);
    using Executor = uint64_t (Impl::*)(Layer &, const std::vector<int32_t> &, MoeExecutionBoundary *);
    Executor execute = &Impl::run_gpu;
    Executor gpu_execute = &Impl::run_gpu;
    double observed_input_ms = 0, observed_merge_ms = 0, observed_handoff_ms = 0, observed_planning_ms = 0;
    double observed_native_upload_ms = 0, observed_mixed_upload_ms = 0;
    uint64_t observed_native_loads = 0, observed_mixed_loads = 0;
    using Selector = void (Impl::*)(Layer &, bool);
    void select_gpu(Layer &, bool) {}
    void select_cpu_share(Layer & binding, bool) {
        binding.placement->update(routed_experts, binding.store->cached_experts(binding.group),
            binding.store->cache_epoch(), cpu_share, config.tokens);
    }
    void select_cost_aware(Layer & binding, bool region) {
        auto & cost = region ? binding.region_cost : binding.cost;
        const auto started = std::chrono::steady_clock::now();
        binding.placement->update_cost_aware(routed_experts, binding.store->cached_experts(binding.group), assignments,
            cost.values, config);
        observed_planning_ms += debug::elapsed_ms(started);
    }
    Selector select = &Impl::select_gpu;
    void configure_policy() {
        switch (execution.moe_policy().placement) {
        case core::MoePlacement::Default: break;
        case core::MoePlacement::CpuShare:
            if (hybrid_enabled) { execute = &Impl::run_cpu_share; select = &Impl::select_cpu_share; }
            break;
        case core::MoePlacement::CostAware:
            select = &Impl::select_cost_aware;
            execute = &Impl::run_cost_aware;
            gpu_execute = &Impl::run_gpu_observed;
            break;
        }
    }

    Impl(const core::ExecutionContext & e, StagedMoeConfig c, std::shared_ptr<core::GraphWorkspace> shared_workspace,
        core::ExecutionContext * cpu_execution)
        : execution(e), config(std::move(c)), workspace(std::move(shared_workspace)), cpu_execution(cpu_execution) {
        if (!workspace) throw std::invalid_argument("staged MoE requires a caller-owned graph workspace");
        if (core::is_host_backend(execution.backend()) && execution.moe_policy().hybrid())
            throw std::invalid_argument("hybrid MoE placement requires a GPU primary backend");
        cpu_share = core::is_host_backend(execution.backend()) ? 0 : execution.moe_policy().cpu_share;
        hybrid_enabled = !core::is_host_backend(execution.backend()) && execution.moe_policy().hybrid();
        if (hybrid_enabled && (!cpu_execution || !core::is_host_backend(cpu_execution->backend())))
            throw std::invalid_argument("hybrid MoE requires a borrowed host execution context");
        configure_policy();
        if (config.input_size == 0) config.input_size = config.hidden_size;
        if (config.hidden_size <= 0 || config.input_size <= 0 || config.tokens <= 0 || config.selected <= 0 || config.experts < config.selected ||
            config.cache_slots <= 0 || config.cache_slots > config.experts || config.rows_per_expert.empty() ||
            (config.scatter_index_type != GGML_TYPE_I32 && config.scatter_index_type != GGML_TYPE_I64))
            throw std::invalid_argument("invalid staged MoE configuration");
        int64_t previous = 0;
        for (auto count : config.rows_per_expert) {
            if (count <= previous) throw std::invalid_argument("staged MoE row buckets must increase");
            previous = count;
        }
        if (config.cache_slots > std::numeric_limits<int32_t>::max() / previous)
            throw std::invalid_argument("staged MoE row buckets exceed indexed graph limits");
        const auto rows = config.cache_slots * previous;
        if (config.tokens > std::numeric_limits<int32_t>::max() / config.selected ||
            rows > std::numeric_limits<int32_t>::max() - config.tokens * config.selected)
            throw std::invalid_argument("staged MoE assignments exceed indexed graph limits");
        if (hybrid_enabled && config.tokens * config.selected > std::numeric_limits<int32_t>::max() / 2)
            throw std::invalid_argument("hybrid MoE assignments exceed indexed graph limits");
        int64_t slot_rows = 0;
        for (auto count : config.rows_per_expert) slot_rows += config.cache_slots * count;
        inputs.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        context.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!inputs || !context) throw std::runtime_error("cannot allocate staged MoE contexts");
        auto * g = inputs.get();
        hidden = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.input_size, config.tokens);
        ids = ggml_new_tensor_2d(g, GGML_TYPE_I32, config.selected, config.tokens);
        weights = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.selected, config.tokens);
        const auto padding_rows = hybrid_enabled ? std::max(rows, config.selected * config.tokens) : rows;
        outputs = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, config.selected * config.tokens + padding_rows);
        source = ggml_new_tensor_1d(g, GGML_TYPE_I32, rows);
        target = ggml_new_tensor_1d(g, config.scatter_index_type, rows);
        slots = ggml_new_tensor_2d(g, GGML_TYPE_I32, 1, slot_rows);
        result = ggml_view_3d(g, outputs, config.hidden_size, config.selected, config.tokens,
            config.hidden_size * sizeof(float), config.hidden_size * config.selected * sizeof(float), 0);
        for (auto * t : {hidden, ids, weights, outputs, source, target, slots}) ggml_set_input(t);
        if (hybrid_enabled) {
            cpu_output = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, config.selected * config.tokens);
            cpu_target = ggml_new_tensor_1d(g, config.scatter_index_type, config.selected * config.tokens);
            ggml_set_input(cpu_output);
            ggml_set_input(cpu_target);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (!buffer) throw std::runtime_error("cannot allocate staged MoE buffers");
        ggml_backend_buffer_clear(buffer.get(), 0);
        logical_ids.resize(static_cast<size_t>(config.selected * config.tokens));
        source_rows.resize(static_cast<size_t>(rows));
        if (config.scatter_index_type == GGML_TYPE_I64) target_rows.emplace<std::vector<int64_t>>();
        std::visit([rows](auto & targets) { targets.resize(static_cast<size_t>(rows)); }, target_rows);
        // Slot IDs depend only on the row bucket, not routing or cache contents.
        std::vector<int32_t> slot_ids(static_cast<size_t>(slot_rows));
        size_t offset = 0;
        for (auto count : config.rows_per_expert) {
            for (int64_t slot = 0; slot < config.cache_slots; ++slot)
                std::fill_n(slot_ids.begin() + offset + slot * count, count, static_cast<int32_t>(slot));
            offset += static_cast<size_t>(config.cache_slots * count);
        }
        ggml_backend_tensor_set(slots, slot_ids.data(), 0, slot_ids.size() * sizeof(int32_t));
        assignments.resize(static_cast<size_t>(config.experts));
        routed_experts.reserve(static_cast<size_t>(config.experts));
    }

    ~Impl() {
        // Join CPU work before freeing graphs or borrowed expert views.
        const bool has_cpu_execution = hybrid || cpu_stage;
        if (hybrid) {
            auto scatter = std::move(hybrid->scatter);
            hybrid.reset();
            for (auto * graph : scatter) core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        cpu_stage.reset();
        if (observed_planning_ms) {
            debug::timing_log_scalar("framework.moe.planning_ms", observed_planning_ms);
        }
        if (observed_input_ms || observed_merge_ms) {
            debug::timing_log_scalar("framework.moe.input_ms", observed_input_ms);
            debug::timing_log_scalar("framework.moe.merge_ms", observed_merge_ms);
            debug::timing_log_scalar("framework.moe.handoff_ms", observed_handoff_ms);
        }
        if (observed_native_loads)
            debug::timing_log_scalar("framework.moe.native_upload_mean_ms", observed_native_upload_ms / observed_native_loads);
        if (observed_mixed_loads)
            debug::timing_log_scalar("framework.moe.mixed_upload_mean_ms", observed_mixed_upload_ms / observed_mixed_loads);
        if (has_cpu_execution) {
            debug::timing_log_scalar("framework.moe.cpu_assignments", stats.cpu_assignments);
            debug::timing_log_scalar("framework.moe.gpu_assignments", stats.gpu_assignments);
            debug::timing_log_scalar("framework.moe.cpu_ms", stats.cpu_ms);
        }
        for (auto & layer : layers)
            for (auto * graph : layer.graphs) core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

StagedMoeRuntime::StagedMoeRuntime(const core::ExecutionContext & execution, StagedMoeConfig config,
    std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu_execution)
    : impl_(std::make_unique<Impl>(execution, std::move(config), std::move(workspace), cpu_execution)) {}
StagedMoeRuntime::~StagedMoeRuntime() = default;

size_t StagedMoeRuntime::add_layer(const MoeModule & module, const MoeExpertWeights & weights,
    core::DiskExpertWeightStore & store, size_t group) {
    auto & s = *impl_;
    if (s.prepared) throw std::logic_error("cannot add staged MoE layers after preparation");
    if (weights.up->ne[2] != s.config.cache_slots || weights.down->ne[1] != s.config.hidden_size)
        throw std::invalid_argument("staged MoE expert bank does not match cache configuration");
    Impl::Layer layer{&store, group, {}, {}, {}, {}};
    const auto backend = store.group_backend(group);
    layer.cpu_staged = backend != s.execution.backend() && core::is_host_backend(backend);
    if (layer.cpu_staged) {
        if (!s.cpu_execution || s.cpu_execution->backend() != backend)
            throw std::invalid_argument("staged CPU MoE requires the cache's borrowed host execution context");
        if (!s.cpu_stage) s.cpu_stage = std::make_unique<Impl::CpuStage>(*s.cpu_execution, s.config);
        auto & host = *s.cpu_stage;
        auto & ctx = host.graphs.build_context();
        size_t slot_offset = 0;
        for (auto count : s.config.rows_per_expert) {
            const auto rows = count * s.config.cache_slots;
            const auto id = !host.has_graphs ? size_t{0}
                : host.graphs.add_graph("layer_" + std::to_string(s.layers.size()) + "_rows_" + std::to_string(count));
            host.has_graphs = true;
            auto * source = ggml_view_1d(ctx.ggml, host.source, rows, 0);
            auto * slots = ggml_view_2d(ctx.ggml, host.slots, 1, rows, host.slots->nb[1], slot_offset);
            slot_offset += static_cast<size_t>(rows) * sizeof(int32_t);
            auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, host.hidden, source),
                s.config.input_size, 1, rows);
            auto * output = module.build_experts(ctx, input, slots, weights);
            output = ggml_reshape_2d(ctx.ggml, output, s.config.hidden_size, rows);
            ggml_set_output(output);
            ggml_build_forward_expand(host.graphs.graph(id), output);
            core::validate_backend_graph_supported(backend, host.graphs.graph(id), "staged CPU MoE experts");
            layer.cpu_graphs.push_back(id);
            layer.cpu_outputs.push_back(output);
        }
        s.layers.push_back(std::move(layer));
        return s.layers.size() - 1;
    }
    if (backend != s.execution.backend())
        throw std::invalid_argument("staged MoE cache backend does not match execution");
    if (s.hybrid_enabled && !s.hybrid) {
        s.hybrid = std::make_unique<Impl::Hybrid>(*s.cpu_execution, s.config);
        for (auto count : s.hybrid->buckets) {
            auto * graph = ggml_new_graph_custom(s.context.get(), 128, false);
            auto * values = ggml_view_2d(s.context.get(), s.cpu_output, s.config.hidden_size, count, s.cpu_output->nb[1], 0);
            auto * targets = ggml_view_1d(s.context.get(), s.cpu_target, count, 0);
            ggml_build_forward_expand(graph, ggml_set_rows(s.context.get(), s.outputs, values, targets));
            s.hybrid->scatter.push_back(graph);
            s.workspace->add(graph);
        }
    }
    if (s.hybrid && !layer.cpu_staged) {
        auto * history = s.execution.moe_policy().placement == core::MoePlacement::CpuShare
            ? &store.cpu_share_state(group) : nullptr;
        layer.placement = std::make_unique<MoePlacementPlan>(static_cast<size_t>(s.config.experts), history);
    }
    if (!layer.cpu_staged && s.gpu_execute == &Impl::run_gpu_observed) {
        for (auto * cost : {&layer.cost, &layer.region_cost}) {
            cost->values.cpu_scale = s.execution.moe_policy().cpu_cost_scale;
            cost->values.cpu.reserve(s.hybrid->buckets.size());
            for (auto rows : s.hybrid->buckets) cost->values.cpu.push_back({rows, 0});
        }
    }
    core::ModuleBuildContext ctx{s.context.get(), "framework.moe.staged", s.execution.backend_type()};
    size_t slot_offset = 0;
    for (auto count : s.config.rows_per_expert) {
        const auto rows = count * s.config.cache_slots;
        auto * graph = ggml_new_graph_custom(ctx.ggml, 128, false);
        auto * source = ggml_view_1d(ctx.ggml, s.source, rows, 0);
        auto * target = ggml_view_1d(ctx.ggml, s.target, rows, 0);
        auto * slots = ggml_view_2d(ctx.ggml, s.slots, 1, rows, s.slots->nb[1], slot_offset);
        slot_offset += static_cast<size_t>(rows) * sizeof(int32_t);
        auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, s.hidden, source), s.config.input_size, 1, rows);
        auto * output = module.build_experts(ctx, input, slots, weights);
        ggml_build_forward_expand(graph, ggml_set_rows(ctx.ggml, s.outputs,
            ggml_reshape_2d(ctx.ggml, output, s.config.hidden_size, rows), target));
        core::validate_backend_graph_supported(s.execution.backend(), graph, "staged MoE experts");
        layer.graphs.push_back(graph);
        s.workspace->add(graph);
    }
    if (s.hybrid) {
        const MoeExpertWeights host_weights{store.host_tensor(group, weights.up).tensor,
            weights.gate ? store.host_tensor(group, weights.gate).tensor : nullptr,
            store.host_tensor(group, weights.down).tensor};
        auto & host = *s.hybrid;
        auto & ctx = host.graphs.build_context();
        for (auto count : host.buckets) {
            const auto id = host.graphs.graph_count() == 1 && ggml_graph_n_nodes(host.graphs.graph()) == 0 ? size_t{0}
                : host.graphs.add_graph("layer_" + std::to_string(s.layers.size()) + "_rows_" + std::to_string(count));
            auto * source = ggml_view_1d(ctx.ggml, host.source, count, 0);
            auto * ids = ggml_view_2d(ctx.ggml, host.ids, 1, count, host.ids->nb[1], 0);
            auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, host.hidden, source),
                s.config.input_size, 1, count);
            auto * output = module.build_experts(ctx, input, ids, host_weights);
            ggml_set_output(output);
            ggml_build_forward_expand(host.graphs.graph(id), output);
            core::validate_backend_graph_supported(host.execution.backend(), host.graphs.graph(id), "hybrid MoE CPU experts");
            layer.cpu_graphs.push_back(id);
            layer.cpu_outputs.push_back(output);
        }
    }
    s.layers.push_back(std::move(layer));
    return s.layers.size() - 1;
}

void StagedMoeRuntime::prepare() {
    auto & s = *impl_;
    if (s.prepared || s.layers.empty()) throw std::logic_error("staged MoE preparation requires unprepared layers");
    for (auto & entry : s.layers) for (auto * graph : entry.graphs)
        core::prepare_host_graph_plan(s.execution, graph, s.plans[graph]);
    if (s.cpu_stage) s.cpu_stage->graphs.allocate();
    if (s.hybrid) {
        s.hybrid->graphs.allocate();
        for (auto * graph : s.hybrid->scatter)
            core::prepare_host_graph_plan(s.execution, graph, s.plans[graph]);
        s.hybrid->start();
    }
    s.prepared = true;
}

void StagedMoeRuntime::build_inputs(core::ModuleBuildContext & ctx, ggml_cgraph * graph,
    ggml_tensor * hidden, const MoeRouting & routing) const {
    auto & s = *impl_;
    if (!ctx.elide_identity_reshapes || !ggml_are_same_shape(hidden, s.hidden))
        hidden = ggml_reshape_2d(ctx.ggml, hidden, s.config.input_size, s.config.tokens);
    // Materialize mixing weights before exposing IDs, preserving the existing
    // routing graph order and backend top-k fusion opportunities.
    for (const auto & pair : {std::make_pair(hidden, s.hidden), std::make_pair(routing.weights, s.weights), std::make_pair(routing.ids, s.ids)})
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, pair.first, pair.second));
}

ggml_tensor * StagedMoeRuntime::hidden_input() const { return impl_->hidden; }
ggml_tensor * StagedMoeRuntime::output() const { return impl_->result; }
ggml_tensor * StagedMoeRuntime::routing_weights() const { return impl_->weights; }
size_t StagedMoeRuntime::workspace_bytes() const {
    return impl_->workspace->bytes() + (impl_->cpu_stage ? impl_->cpu_stage->graphs.workspace_bytes() : 0);
}
StagedMoeRuntime::Statistics StagedMoeRuntime::statistics() const { return impl_->stats; }

uint64_t StagedMoeRuntime::run(size_t layer) {
    auto & s = *impl_;
    if (!s.prepared) throw std::logic_error("staged MoE graphs must be prepared before execution");
    auto & binding = s.layers.at(layer);
    ggml_backend_tensor_get(s.ids, s.logical_ids.data(), 0, s.logical_ids.size() * sizeof(int32_t));
    s.select_route(binding);
    return s.run(binding);
}

uint64_t StagedMoeRuntime::run(size_t layer, const std::vector<int32_t> & ids) {
    auto & s = *impl_;
    if (!s.prepared || ids.size() != s.logical_ids.size())
        throw std::logic_error("hybrid MoE requires prepared graphs and matching routing IDs");
    std::copy(ids.begin(), ids.end(), s.logical_ids.begin());
    auto & binding = s.layers.at(layer);
    s.select_route(binding);
    return s.run(binding);
}

void StagedMoeRuntime::Impl::select_route(Layer & binding) {
    // Ordinary staged callers keep their existing assignment preparation.
    for (auto & pairs : assignments) pairs.clear();
    for (size_t pair = 0; pair < logical_ids.size(); ++pair)
        assignments.at(static_cast<size_t>(logical_ids[pair])).push_back(static_cast<int32_t>(pair));
    routed_experts.clear();
    for (size_t expert = 0; expert < assignments.size(); ++expert)
        if (!assignments[expert].empty()) routed_experts.push_back(static_cast<int32_t>(expert));
    std::stable_sort(routed_experts.begin(), routed_experts.end(), [&](int32_t a, int32_t b) {
        return assignments[a].size() > assignments[b].size();
    });
    if (!binding.cpu_staged) (this->*select)(binding, false);
}

void StagedMoeRuntime::Impl::select_region_route(Layer & binding) {
    auto & s = *this;
    auto & experts = s.routed_experts;
    for (auto expert : experts) s.assignments[expert].clear();
    experts.clear();
    for (size_t pair = 0; pair < s.logical_ids.size(); ++pair) {
        const auto expert = s.logical_ids[pair];
        auto & assignments = s.assignments.at(static_cast<size_t>(expert));
        if (assignments.empty()) experts.push_back(expert);
        assignments.push_back(static_cast<int32_t>(pair));
    }
    // Preserve the original stable-sort order: descending assignment count,
    // then ascending logical ID. Only selected experts need to be visited.
    std::sort(experts.begin(), experts.end(), [&](int32_t a, int32_t b) {
        const auto na = s.assignments[a].size(), nb = s.assignments[b].size();
        return na != nb ? na > nb : a < b;
    });
    if (!binding.cpu_staged) (this->*select)(binding, true);
}

uint64_t StagedMoeRuntime::Impl::run(Layer & binding, MoeExecutionBoundary * boundary) {
    if (binding.cpu_staged) return run_cpu_staged(binding, routed_experts);
    return (this->*execute)(binding, routed_experts, boundary);
}

bool StagedMoeRuntime::prepare_route(size_t layer, const std::vector<int32_t> & ids) {
    auto & s = *impl_;
    if (!s.prepared || ids.size() != s.logical_ids.size())
        throw std::logic_error("MoE route requires prepared graphs and matching IDs");
    std::copy(ids.begin(), ids.end(), s.logical_ids.begin());
    auto & binding = s.layers.at(layer);
    if (s.gpu_execute == &Impl::run_gpu_observed && !binding.region_cost.values.native_gpu_ms_per_row)
        binding.region_cost.values.native_gpu_ms_per_row = 0;
    s.select_region_route(binding);
    return !binding.cpu_staged && binding.placement && binding.placement->cpu.empty() &&
        s.routed_experts.size() <= static_cast<size_t>(s.config.cache_slots);
}

ggml_status StagedMoeRuntime::run_prepared(size_t layer, ggml_cgraph * output_copy, double input_submit_ms) {
    auto & s = *impl_;
    auto & binding = s.layers.at(layer);
    MoeExecutionBoundary boundary{output_copy, input_submit_ms};
    if (s.gpu_execute == &Impl::run_gpu_observed)
        MoeCostModel::observe(binding.region_cost.values.input_submit_ms, input_submit_ms);
    s.run(binding, &boundary);
    if (boundary.output_copied) return GGML_STATUS_SUCCESS;
    if (s.gpu_execute == &Impl::run_gpu_observed && binding.region_cost.values.output_copy_ms == 0) {
        // Bound this calibration to one actual tail execution. Steady GPU-only
        // submission remains asynchronous, as does fixed-share execution.
        const auto started = std::chrono::steady_clock::now();
        const auto status = ggml_backend_graph_compute(s.execution.backend(), output_copy);
        if (status == GGML_STATUS_SUCCESS) binding.region_cost.values.output_copy_ms = debug::elapsed_ms(started);
        return status;
    }
    return ggml_backend_graph_compute_async(s.execution.backend(), output_copy);
}

ggml_status StagedMoeRuntime::run_native(size_t layer, ggml_cgraph * graph, ggml_tensor * bank_ids) {
    auto & s = *impl_;
    auto & binding = s.layers.at(layer);
    const bool observe = s.gpu_execute == &Impl::run_gpu_observed;
    const bool warming = observe && binding.region_cost.values.native_gpu_ms_per_row == 0;
    const bool calibrate = warming && binding.region_cost.native_executions >= 2;
    const auto before = binding.store->statistics();
    const auto started = calibrate ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto mapped = binding.store->stage(binding.group, s.logical_ids);
    ggml_backend_tensor_set(bank_ids, mapped.data(), 0, mapped.size() * sizeof(int32_t));
    // Calibrate this prepared shape once. Steady execution must retain the
    // scheduler's asynchronous submission, not synchronize after every region.
    const auto status = calibrate ? ggml_backend_graph_compute(s.execution.backend(), graph)
        : ggml_backend_graph_compute_async(s.execution.backend(), graph);
    const auto after = binding.store->statistics();
    if (observe) {
        const auto upload_ms = after.expert_loads != before.expert_loads ? after.upload_ms - before.upload_ms : 0;
        if (calibrate)
            binding.region_cost.values.native_gpu_ms_per_row = (debug::elapsed_ms(started) - upload_ms) / s.logical_ids.size();
        if (warming) ++binding.region_cost.native_executions;
        const auto loads = after.expert_loads - before.expert_loads;
        if (loads) {
            MoeCostModel::observe(binding.region_cost.values.upload_ms_per_expert, upload_ms / loads);
            s.observed_native_upload_ms += upload_ms;
            s.observed_native_loads += loads;
        }
    }
    s.stats.gpu_assignments += s.logical_ids.size();
    return status;
}

uint64_t StagedMoeRuntime::Impl::run_cpu_share(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary *) {
    (void) experts;
    return run_hybrid(binding, *binding.placement);
}

uint64_t StagedMoeRuntime::Impl::run_cost_aware(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary * boundary) {
    auto & cost = boundary ? binding.region_cost : binding.cost;
    (void) experts;
    MoeExecutionSample sample;
    if (boundary) sample.input_ms = boundary->input_submit_ms;
    const auto work = run_hybrid(binding, *binding.placement, &sample, boundary);
    if (sample.cpu_rows) {
        cost.observe_cpu(sample);
        observed_input_ms += sample.input_ms;
        observed_merge_ms += sample.merge_ms;
        observed_handoff_ms += sample.handoff_ms;
    }
    return work;
}

uint64_t StagedMoeRuntime::Impl::run_gpu_observed(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary * boundary) {
    const auto before = binding.store->statistics();
    const auto started = std::chrono::steady_clock::now();
    const auto work = run_gpu(binding, experts);
    const auto after = binding.store->statistics();
    const auto loads = after.expert_loads - before.expert_loads;
    const auto upload_ms = after.expert_loads != before.expert_loads ? after.upload_ms - before.upload_ms : 0;
    const bool mixed = !binding.placement->cpu.empty();
    (boundary ? binding.region_cost : binding.cost).observe_gpu(static_cast<size_t>(work), loads, upload_ms, debug::elapsed_ms(started), mixed);
    if (loads) {
        (mixed ? observed_mixed_upload_ms : observed_native_upload_ms) += upload_ms;
        (mixed ? observed_mixed_loads : observed_native_loads) += loads;
    }
    return work;
}

uint64_t StagedMoeRuntime::Impl::run_cpu_staged(Layer & binding, const std::vector<int32_t> & experts) {
    auto & s = *this;
    auto & host = *s.cpu_stage;
    ggml_backend_tensor_get(s.hidden, host.hidden->data, 0, ggml_nbytes(s.hidden));
    uint64_t work_rows = 0;
    for (size_t first = 0; first < experts.size(); first += s.config.cache_slots) {
        const std::vector<int32_t> group(experts.begin() + first,
            experts.begin() + std::min(experts.size(), first + static_cast<size_t>(s.config.cache_slots)));
        const auto mapped = binding.store->stage(binding.group, group);
        size_t maximum = 0;
        for (auto expert : group) maximum = std::max(maximum, s.assignments[expert].size());
        for (size_t begin = 0; begin < maximum; begin += s.config.rows_per_expert.back()) {
            const auto needed = std::min(maximum - begin, static_cast<size_t>(s.config.rows_per_expert.back()));
            const auto bucket = std::lower_bound(s.config.rows_per_expert.begin(), s.config.rows_per_expert.end(), needed);
            const auto count = *bucket;
            const auto rows = static_cast<size_t>(count * s.config.cache_slots);
            std::fill_n(s.source_rows.begin(), rows, 0);
            std::visit([&](auto & targets) {
                using Index = typename std::decay_t<decltype(targets)>::value_type;
                for (size_t row = 0; row < rows; ++row)
                    targets[row] = static_cast<Index>(s.config.selected * s.config.tokens + row);
                for (size_t e = 0; e < group.size(); ++e) {
                    const auto & pairs = s.assignments[group[e]];
                    for (size_t row = begin; row < std::min(pairs.size(), begin + static_cast<size_t>(count)); ++row) {
                        const auto execution_row = mapped[e] * count + row - begin;
                        s.source_rows[execution_row] = static_cast<int32_t>(pairs[row] / s.config.selected);
                        targets[execution_row] = static_cast<Index>(pairs[row]);
                    }
                }
                ggml_backend_tensor_set(host.source, s.source_rows.data(), 0, rows * sizeof(int32_t));
            }, s.target_rows);
            const auto index = static_cast<size_t>(bucket - s.config.rows_per_expert.begin());
            const auto started = std::chrono::steady_clock::now();
            host.graphs.compute(binding.cpu_graphs[index]);
            s.stats.cpu_ms += debug::elapsed_ms(started);
            ggml_backend_tensor_get(binding.cpu_outputs[index], host.batch_output.data(), 0,
                rows * static_cast<size_t>(s.config.hidden_size) * sizeof(float));
            std::visit([&](const auto & targets) {
                const auto width = static_cast<size_t>(s.config.hidden_size);
                for (size_t row = 0; row < rows; ++row) {
                    const auto assignment = static_cast<size_t>(targets[row]);
                    if (assignment < s.logical_ids.size())
                        std::copy_n(host.batch_output.data() + row * width, width,
                            host.ordered_output.data() + assignment * width);
                }
            }, s.target_rows);
            work_rows += rows;
        }
    }
    ggml_backend_tensor_set(s.outputs, host.ordered_output.data(), 0, host.ordered_output.size() * sizeof(float));
    s.stats.cpu_assignments += s.logical_ids.size();
    return work_rows;
}

uint64_t StagedMoeRuntime::Impl::run_gpu(Layer & binding, const std::vector<int32_t> & experts, MoeExecutionBoundary *) {
    auto & s = *this;
    uint64_t work_rows = 0;
    for (size_t first = 0; first < experts.size(); first += s.config.cache_slots) {
        const std::vector<int32_t> group(experts.begin() + first,
            experts.begin() + std::min(experts.size(), first + static_cast<size_t>(s.config.cache_slots)));
        const auto mapped = binding.store->stage(binding.group, group);
        size_t maximum = 0;
        for (auto expert : group) maximum = std::max(maximum, s.assignments[expert].size());
        for (size_t begin = 0; begin < maximum; begin += s.config.rows_per_expert.back()) {
            const auto needed = std::min(maximum - begin, static_cast<size_t>(s.config.rows_per_expert.back()));
            const auto bucket = std::lower_bound(s.config.rows_per_expert.begin(), s.config.rows_per_expert.end(), needed);
            const auto count = *bucket;
            const auto rows = static_cast<size_t>(count * s.config.cache_slots);
            std::fill_n(s.source_rows.begin(), rows, 0);
            std::visit([&](auto & targets) {
                using Index = typename std::decay_t<decltype(targets)>::value_type;
                for (size_t row = 0; row < rows; ++row)
                    targets[row] = static_cast<Index>(s.config.selected * s.config.tokens + row);
                for (size_t e = 0; e < group.size(); ++e) {
                    const auto & pairs = s.assignments[group[e]];
                    for (size_t row = begin; row < std::min(pairs.size(), begin + static_cast<size_t>(count)); ++row) {
                        const auto execution_row = mapped[e] * count + row - begin;
                        s.source_rows[execution_row] = static_cast<int32_t>(pairs[row] / s.config.selected);
                        targets[execution_row] = static_cast<Index>(pairs[row]);
                    }
                }
                ggml_backend_tensor_set(s.source, s.source_rows.data(), 0, rows * sizeof(int32_t));
                ggml_backend_tensor_set(s.target, targets.data(), 0, rows * sizeof(Index));
            }, s.target_rows);
            auto * graph = binding.graphs[static_cast<size_t>(bucket - s.config.rows_per_expert.begin())];
            if (core::compute_graph(s.execution, graph, s.plans.at(graph), "framework.moe.staged") != GGML_STATUS_SUCCESS)
                throw std::runtime_error("staged MoE expert execution failed");
            work_rows += rows;
        }
    }
    return work_rows;
}

uint64_t StagedMoeRuntime::Impl::run_hybrid(Layer & binding, const MoePlacementPlan & placement, MoeExecutionSample * sample, MoeExecutionBoundary * boundary) {
    auto & host = *hybrid;
    for (auto expert : placement.gpu) stats.gpu_assignments += assignments[expert].size();
    size_t cpu_rows = 0;
    for (auto expert : placement.cpu) cpu_rows += assignments[expert].size();
    if (cpu_rows == 0) return (this->*gpu_execute)(binding, placement.gpu, boundary);
    const auto input_started = sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto found = std::lower_bound(host.buckets.begin(), host.buckets.end(), static_cast<int64_t>(cpu_rows));
    const auto bucket = static_cast<size_t>(found - host.buckets.begin());
    const auto rows = static_cast<size_t>(*found);
    std::fill_n(host.source_rows.begin(), rows, 0);
    std::fill_n(host.expert_ids.begin(), rows, 0);
    std::visit([&](auto & targets) {
        using Index = typename std::decay_t<decltype(targets)>::value_type;
        for (size_t row = 0; row < rows; ++row) targets[row] = static_cast<Index>(config.selected * config.tokens + row);
        size_t row = 0;
        for (auto expert : placement.cpu) for (auto assignment : assignments[expert]) {
            host.source_rows[row] = static_cast<int32_t>(assignment / config.selected);
            host.expert_ids[row] = expert;
            targets[row++] = static_cast<Index>(assignment);
        }
        ggml_backend_tensor_set(cpu_target, targets.data(), 0, rows * sizeof(Index));
    }, host.target_rows);
    // Only activations cross the device boundary; expert payloads stay native
    // in the mapped/pinned host bank. CPU work overlaps uploads/GPU execution.
    ggml_backend_tensor_get_async(execution.backend(), hidden, host.hidden->data, 0, ggml_nbytes(hidden));
    ggml_backend_synchronize(execution.backend());
    ggml_backend_tensor_set(host.source, host.source_rows.data(), 0, rows * sizeof(int32_t));
    ggml_backend_tensor_set(host.ids, host.expert_ids.data(), 0, rows * sizeof(int32_t));
    if (sample) {
        sample->cpu_rows = rows;
        sample->input_ms += debug::elapsed_ms(input_started);
    }
    const auto parallel_started = sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    host.launch(binding, bucket, sample);
    const auto gpu_started = sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    uint64_t work_rows;
    try { work_rows = (this->*gpu_execute)(binding, placement.gpu, boundary); }
    catch (...) { host.wait(); throw; }
    const auto gpu_ms = sample ? debug::elapsed_ms(gpu_started) : 0;
    host.wait();
    if (sample) {
        // Reconcile the complete parallel interval with the two measured
        // branches. The residual retains launch/notification and resume work;
        // the GPU branch timer must not absorb launch before its own work.
        sample->handoff_ms = debug::elapsed_ms(parallel_started) - std::max(sample->cpu_ms, gpu_ms);
    }
    stats.cpu_assignments += cpu_rows;
    stats.cpu_ms += host.compute_ms;
    const auto merge_started = sample ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto * output = binding.cpu_outputs[bucket];
    ggml_backend_tensor_set_async(execution.backend(), cpu_output, output->data, 0, ggml_nbytes(output));
    auto * scatter = host.scatter[bucket];
    if (sample && boundary) {
        // Move the terminal wait after the existing output copy, rather than
        // adding another wait. CPU scratch is protected through completion.
        if (ggml_backend_graph_compute_async(execution.backend(), scatter) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(execution.backend(), boundary->output_copy) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("hybrid MoE output completion failed");
        boundary->output_copied = true;
    } else if (core::compute_graph(execution, scatter, plans.at(scatter), "framework.moe.cpu_merge") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("hybrid MoE output scatter failed");
    }
    // Completion protects CPU scratch before the next execution reuses it.
    // Mixing remains in original logical order.
    if (sample) sample->merge_ms = debug::elapsed_ms(merge_started);
    return work_rows + rows;
}

namespace {

struct MoeGraphState {
    struct Binding {
        MoeModule module;
        MoeExpertWeights weights;
        core::DiskExpertWeightStore::Route route;
        ggml_tensor * bank_ids;
        ggml_tensor * first;
        ggml_tensor * output;
        size_t batch = 0, layer = 0;
        core::PreparedGraph::GraphId input_copy = 0, output_copy = 0;
        std::vector<int32_t> ids;
    };
    const core::ExecutionContext & execution;
    std::vector<Binding> bindings;
    std::shared_ptr<core::GraphWorkspace> workspace;
    std::vector<std::unique_ptr<StagedMoeRuntime>> batches;
    std::unique_ptr<core::PreparedGraph> copies;
    size_t active = 0;
    uint64_t hybrid_regions = 0;
    bool prepared = false;

    explicit MoeGraphState(const core::ExecutionContext & execution) : execution(execution) {}
    ~MoeGraphState() {
        release();
        debug::timing_log_scalar("framework.moe.hybrid_regions", hybrid_regions);
    }
    void prepare() {
        if (prepared) return;
        workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
        using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
        std::map<Key, size_t> shapes;
        for (auto & binding : bindings) {
            auto * input = binding.first->src[1];
            auto * ids = binding.first->src[2];
            if (input->type != GGML_TYPE_F32 || binding.output->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32)
                throw std::runtime_error("hybrid MoE regions require F32 activations and I32 routing");
            StagedMoeConfig config;
            config.input_size = input->ne[0];
            config.hidden_size = binding.output->ne[0];
            config.tokens = binding.output->ne[2];
            config.selected = binding.output->ne[1];
            config.experts = binding.route.store->host_tensor(binding.route.group, binding.weights.up).tensor->ne[2];
            config.cache_slots = binding.weights.up->ne[2];
            const Key key{config.input_size, config.hidden_size, config.tokens, config.selected, config.experts, config.cache_slots};
            const auto found = shapes.find(key);
            if (found == shapes.end()) {
                binding.batch = batches.size();
                shapes.emplace(key, binding.batch);
                for (int64_t rows = 1; rows < config.tokens; rows *= 2) config.rows_per_expert.push_back(rows);
                config.rows_per_expert.push_back(config.tokens);
                batches.push_back(std::make_unique<StagedMoeRuntime>(execution, config, workspace,
                    binding.route.store->cpu_execution()));
            } else binding.batch = found->second;
            binding.layer = batches[binding.batch]->add_layer(binding.module, binding.weights,
                *binding.route.store, binding.route.group);
            binding.ids.resize(static_cast<size_t>(ggml_nelements(ids)));
        }
        workspace->prepare();
        for (auto & batch : batches) batch->prepare();
        const auto metadata = (bindings.size() * 16 + 64) * ggml_tensor_overhead() +
            (bindings.size() * 2 + 1) * ggml_graph_overhead_custom(32, false);
        copies = std::make_unique<core::PreparedGraph>(execution,
            core::PreparedGraphConfig{"framework.moe.boundary_copy", metadata, 32});
        auto & ctx = copies->build_context();
        for (size_t i = 0; i < bindings.size(); ++i) {
            auto & binding = bindings[i];
            auto & batch = *batches[binding.batch];
            const std::array<ggml_tensor *, 2> original{binding.first->src[1], binding.output};
            std::array<ggml_tensor *, 2> boundary{};
            for (size_t j = 0; j < boundary.size(); ++j) {
                // Borrow data as leaves: following the original DAG here would
                // recompute the backbone prefix and its state updates.
                boundary[j] = ggml_dup_tensor(ctx.ggml, original[j]);
                if (ggml_backend_tensor_alloc(original[j]->buffer, boundary[j], original[j]->data) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("cannot bind hybrid MoE boundary storage");
            }
            binding.input_copy = i == 0 ? 0 : copies->add_graph("input_" + std::to_string(i));
            auto * flat = ggml_reshape_2d(ctx.ggml, boundary[0], original[0]->ne[0], binding.output->ne[2]);
            ggml_build_forward_expand(copies->graph(binding.input_copy), ggml_cpy(ctx.ggml, flat, batch.hidden_input()));
            binding.output_copy = copies->add_graph("output_" + std::to_string(i));
            ggml_build_forward_expand(copies->graph(binding.output_copy), ggml_cpy(ctx.ggml, batch.output(), boundary[1]));
        }
        copies->allocate();
        prepared = true;
    }
    void release() noexcept {
        copies.reset();
        batches.clear();
        workspace.reset();
        prepared = false;
    }
};

class MoeRegionExecutor final : public core::GraphRegionExecutor {
public:
    MoeRegionExecutor(std::shared_ptr<MoeGraphState> state, size_t binding) : state_(std::move(state)), binding_(binding) {}
    void prepare(ggml_backend_t backend, ggml_cgraph * graph) override {
        if (backend != state_->execution.backend()) throw std::runtime_error("hybrid MoE region must execute on its primary backend");
        state_->prepare();
        native_graph_ = graph;
        active_ = true;
        ++state_->active;
    }
    ggml_status compute() override {
        auto & binding = state_->bindings[binding_];
        ggml_backend_tensor_get(binding.bank_ids, binding.ids.data(), 0, binding.ids.size() * sizeof(int32_t));
        auto & batch = *state_->batches[binding.batch];
        if (batch.prepare_route(binding.layer, binding.ids))
            // The scheduler may give the expert op a backend-local routing
            // copy. Write physical slots to the consumer, not its source.
            return batch.run_native(binding.layer, native_graph_, binding.first->src[2]);
        auto * backend = state_->execution.backend();
        // The following CPU readback or staged compute already waits for
        // this stream. An additional boundary-copy synchronization is redundant.
        const bool observe = state_->execution.moe_policy().placement == core::MoePlacement::CostAware;
        const auto started = observe ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto status = ggml_backend_graph_compute_async(backend, state_->copies->graph(binding.input_copy));
        if (status != GGML_STATUS_SUCCESS) return status;
        const auto input_submit_ms = observe ? debug::elapsed_ms(started) : 0;
        const auto completed = batch.run_prepared(binding.layer, state_->copies->graph(binding.output_copy), input_submit_ms);
        ++state_->hybrid_regions;
        return completed;
    }
    void reset() noexcept override {
        if (active_) {
            --state_->active;
            active_ = false;
        }
        if (state_->active == 0) state_->release();
    }
private:
    std::shared_ptr<MoeGraphState> state_;
    size_t binding_;
    ggml_cgraph * native_graph_ = nullptr;
    bool active_ = false;
};

} // namespace

void MoeModule::record_region(core::ModuleBuildContext & ctx, ggml_tensor * first, ggml_tensor * output,
    const MoeExpertWeights & weights, ggml_tensor * bank_ids) const {
    const auto route = core::DiskExpertWeightStore::find_route(ctx, bank_ids);
    if (!route) return; // Resident banks have no disk cache misses to place.
    // CPU-selected banks retain scheduler execution. Automatic replacement is
    // only for GPU banks; explicit staged runtimes handle bounded CPU batches.
    if (core::is_host_backend(route->store->group_backend(route->group))) return;
    auto * owner = ctx.region_owner;
    auto state = owner->region_state<MoeGraphState>(owner->execution());
    const auto binding = state->bindings.size();
    state->bindings.push_back({*this, weights, *route, bank_ids, first, output, 0, 0, 0, 0, {}});
    core::GraphRegion region;
    region.outputs = {output};
    region.deferred_eval = {bank_ids};
    std::unordered_set<ggml_tensor *> visited;
    const auto visit = [&](const auto & self, ggml_tensor * node) -> void {
        if (!node || node == first->src[1] || node == bank_ids || node->op == GGML_OP_NONE ||
            !visited.insert(node).second) return;
        for (auto * source : node->src) self(self, source);
        region.nodes.push_back(node);
    };
    visit(visit, output);
    owner->record_region(std::move(region), [state, binding] { return std::make_unique<MoeRegionExecutor>(state, binding); });
}

} // namespace engine::modules
