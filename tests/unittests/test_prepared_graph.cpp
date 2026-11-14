#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/session_base.h"
#include "test_assert.h"
#include "test_allocation_failure.h"
#include "ggml-backend-impl.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

namespace core = engine::core;
namespace modules = engine::modules;
namespace debug = engine::debug;
using engine::test::require;

namespace {

void check_session_moe_option() {
    struct Session : engine::runtime::RuntimeSessionBase {
        using RuntimeSessionBase::RuntimeSessionBase;
        using RuntimeSessionBase::execution_context;
        using RuntimeSessionBase::acquire_moe_execution;
        using RuntimeSessionBase::mark_prepared;
        using RuntimeSessionBase::require_prepared;
    };
    engine::runtime::SessionOptions options;
    options.backend.type = core::BackendType::Cpu;
    options.backend.threads = 8;
    Session ordinary(options);
    require(ordinary.execution_context().moe_cpu_share() == 0, "omitted session option must disable hybrid execution");
    ordinary.mark_prepared();
    for (const char * value : {"0", "0.0", "-0", "0e0"}) {
        auto disabled_options = options;
        disabled_options.options["moe_cpu_share"] = value;
        Session disabled(disabled_options);
        disabled.mark_prepared();
        disabled.require_prepared("disabled MoE option regression test");
        require(disabled.execution_context().moe_cpu_share() == 0 &&
                !disabled.execution_context().has_moe_resources(),
                "explicit zero must not require or create MoE resources");
    }
    for (const char * key : {"moe_disk_experts", "moe_pin_experts", "moe_expert_cache_size",
             "moe_disk_expert_read_threads", "moe_cpu_expert_layers"}) {
        auto unsupported_options = options;
        unsupported_options.options[key] = "0";
        Session unsupported(unsupported_options);
        bool rejected = false;
        try { unsupported.mark_prepared(); }
        catch (const std::invalid_argument & error) {
            rejected = std::string(error.what()).find("does not support shared MoE resources") != std::string::npos;
        }
        require(rejected, "session silently accepted MoE controls without resource integration");
    }
    options.options["moe_cpu_share"] = "0.5";
    Session enabled(options);
    bool active_rejected = false;
    try { enabled.mark_prepared(); }
    catch (const std::invalid_argument & error) {
        active_rejected = std::string(error.what()).find("does not support shared MoE resources") != std::string::npos;
    }
    require(active_rejected, "unsupported active MoE CPU share was accepted");
    require(enabled.execution_context().moe_cpu_share() == 0.5F, "session option did not reach execution context");
    require(ordinary.execution_context().moe_cpu_share() == 0, "session option leaked across sessions");
    Session independent(engine::runtime::SessionOptions{options});
    for (bool registered : {false, true}) {
        if (registered) {
            ordinary.execution_context().register_moe_resources();
            independent.execution_context().register_moe_resources();
        }
        std::promise<void> release;
        std::promise<void> acquired;
        auto release_signal = release.get_future();
        auto active = std::async(std::launch::async, [&] {
            auto lease = ordinary.acquire_moe_execution();
            acquired.set_value();
            release_signal.wait();
        });
        acquired.get_future().wait();
        bool overlap_rejected = false;
        try { auto lease = ordinary.acquire_moe_execution(); }
        catch (const std::runtime_error & error) {
            overlap_rejected = std::string(error.what()).find("overlapping requests") != std::string::npos;
        }
        bool independent_acquired = false;
        try {
            auto lease = independent.acquire_moe_execution();
            independent_acquired = lease.owns_lock();
        } catch (const std::runtime_error &) {
            // Drain the first request before reporting a shared-lock defect.
        }
        release.set_value();
        active.get();
        require(overlap_rejected, "overlapping MoE session execution was accepted");
        require(independent_acquired, "independent MoE sessions share a request lock");
        require(ordinary.acquire_moe_execution().owns_lock(), "completed execution did not release session ownership");
    }
    core::ExecutionContext moved(std::move(enabled.execution_context()));
    require(moved.moe_cpu_share() == 0.5F, "move lost session execution policy");
    for (const char * value : {"-0.1", "1.1", "nan", "typo"}) {
        options.options["moe_cpu_share"] = value;
        bool rejected = false;
        try { Session invalid(options); } catch (const std::exception & error) {
            rejected = std::string(error.what()).find("moe_cpu_share") != std::string::npos;
        }
        require(rejected, "invalid session CPU share was accepted");
    }
}

// Exercise explicit recorder publication in CPU-only CI. Only device classification is
// substituted; no graph computation is performed on this borrowed device.
void check_owner_allocation_failures() {
    core::BackendConfig backend;
    backend.type = core::BackendType::Cpu;
    backend.threads = 8;
    core::ExecutionContext execution(backend, 0.5F);
    auto * handle = execution.backend();
    auto * original = handle->device;
    ggml_backend_device device = *original;
    device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    struct RestoreDevice {
        ggml_backend_t backend;
        ggml_backend_dev_t original;
        ~RestoreDevice() { backend->device = original; }
    } restore{handle, original};
    handle->device = &device;
    const core::PreparedGraphConfig config{std::string(96, 'r'), 256 * 1024, 128};
    // A GPU-classified backend with the option disabled must never publish a
    // recorder, including after reset and release/reprepare.
    {
        auto ordinary_config = backend;
        core::ExecutionContext ordinary(ordinary_config);
        auto * ordinary_handle = ordinary.backend();
        RestoreDevice restore_ordinary{ordinary_handle, ordinary_handle->device};
        ordinary_handle->device = &device;
        core::PreparedGraph owner(ordinary, config);
        require(owner.build_context().region_owner == nullptr, "default graph enabled region recording");
        owner.reset();
        require(owner.build_context().region_owner == nullptr, "default reset enabled region recording");
        owner.release();
        owner.reset();
        require(owner.build_context().region_owner == nullptr, "default reprepare enabled region recording");
    }
    core::PreparedGraph warm(execution, config);
    require(warm.build_context().region_owner == &warm,
        "owner failure fixture did not enable registration");
    size_t construction_failures = 0, reset_failures = 0;
    bool constructed = false, reset = false;
    for (size_t allocation = 0; allocation < 32; ++allocation) {
        bool rejected = false;
        try {
            engine::test::AllocationFailure fault(allocation);
            core::PreparedGraph owner(execution, config);
            fault.disable();
            require(owner.build_context().region_owner == &owner,
                "constructed owner resolved to an incorrect recorder");
            constructed = true;
        } catch (const std::bad_alloc &) {
            rejected = true;
            ++construction_failures;
        }
        require(!rejected || engine::test::AllocationFailure::triggered(), "unexpected construction allocation failure");
        require(engine::test::AllocationFailure::outstanding() == 0,
            "failed construction leaked a recorder allocation");
        {
            core::PreparedGraph retry(execution, config);
            require(retry.build_context().region_owner == &retry,
                "construction retry found an incorrect recorder owner");
        }
        if (constructed) break;
    }
    for (size_t allocation = 0; allocation < 32; ++allocation) {
        {
            core::PreparedGraph owner(execution, config);
            bool rejected = false;
            try {
                engine::test::AllocationFailure fault(allocation);
                owner.reset();
                fault.disable();
                reset = true;
            } catch (const std::bad_alloc &) {
                rejected = true;
                ++reset_failures;
            }
            if (rejected) {
                require(engine::test::AllocationFailure::triggered(), "unexpected reset allocation failure");
                require(owner.build_context().region_owner == nullptr,
                    "failed reset published an incomplete graph owner");
                owner.reset();
            }
            require(owner.build_context().region_owner == &owner,
                "reset retry did not publish the recovered owner");
        }
        require(engine::test::AllocationFailure::outstanding() == 0, "reset recovery leaked graph allocations");
        if (reset) break;
    }
    require(constructed && reset && construction_failures >= 3 && reset_failures >= 2,
        "owner failure fixture did not cover construction and reset allocation boundaries");
    std::cout << "owner_allocation_failures construction=" << construction_failures << " reset=" << reset_failures << '\n';
}

// Observe the real backend boundary, rather than assuming work stays in flight.
// The original backend still performs the synchronization.
struct ResetSyncCheck {
    ggml_backend_t backend;
    core::PreparedGraph & owner;
    void (*original)(ggml_backend_t);
    int calls = 0;
    bool storage_alive = true;
    static ResetSyncCheck * active;

    ResetSyncCheck(core::PreparedGraph & graph)
        : backend(graph.execution().backend()), owner(graph), original(backend->iface.synchronize) {
        require(!active, "nested synchronization observer");
        active = this;
        backend->iface.synchronize = [](ggml_backend_t value) {
            auto & check = *active;
            ++check.calls;
            check.storage_alive = check.storage_alive && check.owner.allocated();
            for (size_t id = 0; id < check.owner.graph_count(); ++id) {
                auto * graph = check.owner.graph(id);
                check.storage_alive = check.storage_alive && graph && ggml_graph_n_nodes(graph) > 0 &&
                    ggml_graph_node(graph, 0)->data != nullptr;
            }
            if (check.original) check.original(value);
        };
    }
    ~ResetSyncCheck() {
        backend->iface.synchronize = original;
        active = nullptr;
    }
};
ResetSyncCheck * ResetSyncCheck::active = nullptr;

// Observe the real backend's plan lifecycle without replacing graph execution.
struct PlanCheck {
    ggml_backend_t backend;
    ggml_backend_graph_plan_t (*create)(ggml_backend_t, const ggml_cgraph *);
    void (*free)(ggml_backend_t, ggml_backend_graph_plan_t);
    size_t created = 0;
    size_t freed = 0;
    static PlanCheck * active;
    explicit PlanCheck(core::ExecutionContext & execution)
        : backend(execution.backend()), create(backend->iface.graph_plan_create), free(backend->iface.graph_plan_free) {
        require(!active, "nested plan observer");
        active = this;
        if (create) backend->iface.graph_plan_create = [](ggml_backend_t value, const ggml_cgraph * graph) {
            auto * plan = active->create(value, graph);
            if (plan) ++active->created;
            return plan;
        };
        if (free) backend->iface.graph_plan_free = [](ggml_backend_t value, ggml_backend_graph_plan_t plan) {
            ++active->freed;
            active->free(value, plan);
        };
    }
    ~PlanCheck() {
        backend->iface.graph_plan_create = create;
        backend->iface.graph_plan_free = free;
        active = nullptr;
    }
};
PlanCheck * PlanCheck::active = nullptr;

struct BufferFreeCheck {
    ggml_backend_buffer_t buffer;
    core::PreparedGraph & owner;
    void (*original)(ggml_backend_buffer_t);
    bool freed = false;
    bool graphs_released = false;
    static BufferFreeCheck * active;
    BufferFreeCheck(ggml_backend_buffer_t value, core::PreparedGraph & graph)
        : buffer(value), owner(graph), original(buffer->iface.free_buffer) {
        require(!active, "nested buffer observer");
        active = this;
        buffer->iface.free_buffer = [](ggml_backend_buffer_t value) {
            active->freed = true;
            active->graphs_released = !active->owner.allocated() && active->owner.graph_count() == 0;
            if (active->original) active->original(value);
        };
    }
    ~BufferFreeCheck() {
        if (!freed) buffer->iface.free_buffer = original;
        active = nullptr;
    }
};
BufferFreeCheck * BufferFreeCheck::active = nullptr;

struct RegionStats {
    int prepared = 0, computed = 0, reset = 0;
    bool fail_prepare = false, fail_compute = false, storage_alive = true;
    ggml_cgraph * view = nullptr;
};

class TestRegion final : public core::GraphRegionExecutor {
public:
    explicit TestRegion(RegionStats & stats) : stats_(stats) {}
    void prepare(ggml_backend_t backend, ggml_cgraph * region) override {
        require(backend && ggml_graph_n_nodes(region) == 2, "replacement received the wrong graph region");
        require(!stats_.view || stats_.view == region, "buffer reuse rebuilt region metadata");
        stats_.view = region;
        first_ = ggml_graph_node(region, 0);
        last_ = ggml_graph_node(region, 1);
        input_ = first_->src[0];
        require(input_->data && first_->data && last_->data, "region prepared before allocation");
        ++stats_.prepared;
        if (stats_.fail_prepare) throw std::runtime_error("injected region preparation failure");
    }
    ggml_status compute() override {
        ++stats_.computed;
        if (stats_.fail_compute) return GGML_STATUS_FAILED;
        auto values = core::read_tensor_f32(input_);
        for (auto & value : values) value *= 7;
        ggml_backend_tensor_set(first_, values.data(), 0, values.size() * sizeof(float));
        for (auto & value : values) value += 2;
        ggml_backend_tensor_set(last_, values.data(), 0, values.size() * sizeof(float));
        return GGML_STATUS_SUCCESS;
    }
    void reset() noexcept override {
        stats_.storage_alive = stats_.storage_alive && input_ && input_->data && last_->data;
        ++stats_.reset;
        input_ = first_ = last_ = nullptr;
    }
private:
    RegionStats & stats_;
    ggml_tensor * input_ = nullptr, * first_ = nullptr, * last_ = nullptr;
};

// Inspect both native compute and host-plan creation: replacement nodes must
// never execute, including execution before a later overwrite by the callback.
struct RegionNativeCheck {
    ggml_backend_t backend;
    ggml_tensor * first, * last;
    ggml_status (*compute)(ggml_backend_t, ggml_cgraph *);
    ggml_backend_graph_plan_t (*plan)(ggml_backend_t, const ggml_cgraph *);
    static RegionNativeCheck * active;
    static void check(const ggml_cgraph * graph) {
        auto * view = const_cast<ggml_cgraph *>(graph);
        for (int i = 0; i < ggml_graph_n_nodes(view); ++i)
            require(ggml_graph_node(view, i) != active->first && ggml_graph_node(view, i) != active->last,
                "original replacement nodes reached native execution");
    }
    RegionNativeCheck(ggml_backend_t value, ggml_tensor * a, ggml_tensor * b)
        : backend(value), first(a), last(b), compute(value->iface.graph_compute), plan(value->iface.graph_plan_create) {
        require(!active, "nested region execution observer");
        active = this;
        backend->iface.graph_compute = [](ggml_backend_t value, ggml_cgraph * graph) {
            check(graph);
            return active->compute(value, graph);
        };
        if (plan) backend->iface.graph_plan_create = [](ggml_backend_t value, const ggml_cgraph * graph) {
            check(graph);
            return active->plan(value, graph);
        };
    }
    ~RegionNativeCheck() {
        backend->iface.graph_compute = compute;
        backend->iface.graph_plan_create = plan;
        active = nullptr;
    }
};
RegionNativeCheck * RegionNativeCheck::active = nullptr;

void check_regions(core::ExecutionContext & execution) {
    for (int mode = 0; mode < 4; ++mode) {
        RegionStats stats, body_stats;
        std::unique_ptr<core::GraphWorkspace> workspace;
        std::unique_ptr<core::ExecutionContext> host;
        core::PreparedGraph owner(execution, {"prepared.region", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
        auto & persistent = owner.persistent_build_context();
        auto input = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1024}));
        ggml_set_input(input.tensor);
        owner.allocate_persistent();
        auto & ctx = owner.build_context();
        auto * prefix = ggml_scale_bias(ctx.ggml, input.tensor, 1, 1);
        auto * first = ggml_scale(ctx.ggml, prefix, 3);
        auto * last = ggml_scale_bias(ctx.ggml, first, 1, 1);
        auto * output = ggml_scale(ctx.ggml, last, 5);
        ggml_set_output(output);
        ggml_build_forward_expand(owner.graph(), output);
        owner.replace_region(0, {{first, last}, {last}, {}}, std::make_unique<TestRegion>(stats));
        core::PreparedGraph::GraphId body = 0;
        if (mode == 3) {
            ggml_set_output(prefix); // Body execution reuses the saved prefix.
            body = owner.add_entry_point("body", 0);
            for (auto * node : {first, last, output}) ggml_graph_add_node(owner.graph(body), node);
            owner.replace_region(body, {{first, last}, {last}, {}}, std::make_unique<TestRegion>(body_stats));
        }
        struct Eval { ggml_tensor * prefix, * output; int observed = 0; bool cancel = false; } eval{prefix, output};
        if (mode == 1) {
            std::vector<ggml_backend_t> backends{execution.backend()};
            if (!core::is_host_backend(execution.backend())) {
                host = std::make_unique<core::ExecutionContext>(core::BackendConfig{core::BackendType::Cpu, 0, 8});
                backends.push_back(host->backend());
            }
            auto * scheduler = owner.create_scheduler(std::move(backends));
            ggml_backend_sched_set_eval_callback(scheduler, [](ggml_tensor * tensor, bool ask, void * data) {
                auto & state = *static_cast<Eval *>(data);
                if (ask) return tensor == state.prefix || tensor == state.output;
                ++state.observed;
                return !state.cancel;
            }, &eval);
        }
        RegionNativeCheck observer(execution.backend(), first, last);
        if (mode == 0) owner.allocate_releasable();
        else if (mode == 1) owner.allocate();
        else if (mode == 2) {
            workspace = std::make_unique<core::GraphWorkspace>(execution.backend());
            workspace->add(owner);
            workspace->prepare();
        } else owner.allocate();
        const auto bytes = owner.workspace_bytes();
        for (int request = 0; request < 8; ++request) {
            std::vector<float> values(1024), expected(1024);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = float((i + request) % 61);
                expected[i] = ((values[i] + 1) * 7 + 2) * 5;
            }
            core::write_tensor_f32(input, values);
            owner.compute();
            require(core::read_tensor_f32(output) == expected, "region boundary math or input reuse failed");
            require(owner.workspace_bytes() == bytes && stats.prepared == 1, "region execution rebuilt allocations or plans");
            if (body) {
                owner.compute(body);
                require(core::read_tensor_f32(output) == expected, "entry-point replacement used another allocation layout");
            }
        }
        if (mode == 1) {
            require(eval.observed == 16, "ordinary-node evaluation callbacks changed with replacement enabled");
            eval.cancel = true;
            owner.compute();
            require(stats.computed == 8, "evaluation cancellation did not stop before the replacement");
            eval.cancel = false;
        }
        stats.fail_compute = true;
        bool failed = false;
        try { owner.compute(); } catch (const std::runtime_error &) { failed = true; }
        require(failed, "replacement compute failure was hidden");
        stats.fail_compute = false;
        owner.compute();
        if (mode == 0) {
            owner.release_buffers();
            require(stats.reset == 1 && stats.storage_alive, "replacement released after tensor storage");
            owner.allocate_persistent();
            owner.allocate_releasable();
            core::write_tensor_f32(input, std::vector<float>(1024, 4));
            owner.compute();
            require(core::read_tensor_f32(output) == std::vector<float>(1024, 185),
                "replacement retained stale addresses after buffer reallocation");
            require(stats.prepared == 2, "replacement was not reprepared after relocation");
        }
        if (workspace) workspace->reset();
        owner.release();
        require(stats.storage_alive && stats.reset == stats.prepared, "replacement lifecycle cleanup mismatch");
        require(body_stats.storage_alive && body_stats.reset == body_stats.prepared, "entry-point replacement cleanup mismatch");
    }
}

void check_recorded_regions(core::ExecutionContext & execution) {
    for (bool partial : {false, true}) {
        RegionStats stats[2];
        size_t factories = 0;
        core::PreparedGraph owner(execution, {"prepared.region.recorded", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
        auto input = core::make_tensor(owner.persistent_build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({33}));
        ggml_set_input(input.tensor);
        owner.allocate_persistent();
        auto & ctx = owner.build_context();
        auto * prefix = ggml_scale_bias(ctx.ggml, input.tensor, 1, 1);
        auto * first = ggml_scale(ctx.ggml, prefix, 3);
        auto * last = ggml_scale_bias(ctx.ggml, first, 1, 1);
        auto * output = ggml_scale(ctx.ggml, last, 5);
        ggml_set_output(prefix);
        ggml_set_output(output);
        ggml_build_forward_expand(owner.graph(), output);
        const auto body = owner.add_entry_point("body", 0);
        if (!partial) ggml_graph_add_node(owner.graph(body), first);
        for (auto * node : {last, output}) ggml_graph_add_node(owner.graph(body), node);
        owner.record_region({{first, last}, {last}, {}}, [&] {
            require(factories < 2, "recorded region created an extra executor");
            return std::make_unique<TestRegion>(stats[factories++]);
        });
        bool rejected = false;
        try { owner.allocate(); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected == partial, "recorded region did not reject a partial entry point");
        if (partial) {
            require(stats[0].prepared == 0, "partial recorded region failed after preparation");
            continue;
        }
        require(factories == 2, "complete entry points did not each bind their recorded region");
        for (int request = 0; request < 4; ++request) {
            core::write_tensor_f32(input, std::vector<float>(33, float(request)));
            owner.compute();
            core::write_tensor_f32(input, std::vector<float>(33, float(request + 100)));
            owner.compute(body);
            require(core::read_tensor_f32(output) == std::vector<float>(33, float(((request + 1) * 7 + 2) * 5)),
                "recorded body region replayed the prefix or used stale output");
        }
        owner.release();
        for (const auto & state : stats)
            require(state.prepared == 1 && state.computed == 4 && state.reset == 1 && state.storage_alive,
                "recorded entry-point executor lifecycle changed");
    }
}

void check_region_contracts(core::ExecutionContext & execution) {
    for (int variant = 0; variant < 6; ++variant) {
        RegionStats stats, second_stats;
        std::unique_ptr<core::ExecutionContext> host;
        core::PreparedGraph owner(execution, {"prepared.region.invalid", 1024 * 1024, 128});
        auto & ctx = owner.build_context();
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({16}));
        auto * first = ggml_scale(ctx.ggml, input.tensor, 3);
        auto * last = ggml_scale(ctx.ggml, first, 2);
        auto * result = last;
        if (variant == 5) {
            auto * second = ggml_scale(ctx.ggml, last, 3);
            result = ggml_scale(ctx.ggml, second, 2);
            owner.replace_region(0, {{second, result}, {result}, {}}, std::make_unique<TestRegion>(second_stats));
            std::vector<ggml_backend_t> backends{execution.backend()};
            if (!core::is_host_backend(execution.backend())) {
                host = std::make_unique<core::ExecutionContext>(core::BackendConfig{core::BackendType::Cpu, 0, 8});
                backends.push_back(host->backend());
            }
            owner.create_scheduler(std::move(backends));
        }
        ggml_set_output(result);
        ggml_build_forward_expand(owner.graph(), result);
        owner.replace_region(0, {{variant == 0 ? last : variant == 1 ? input.tensor : first,
            variant == 0 ? first : last}, {last}, {}}, std::make_unique<TestRegion>(stats));
        if (variant == 2) owner.replace_region(0, {{last}, {last}, {}}, std::make_unique<TestRegion>(stats));
        if (variant == 3) stats.fail_prepare = true;
        bool failed = false;
        try { owner.allocate(); } catch (const std::exception &) { failed = true; }
        if (variant >= 4) {
            require(!failed, "whole-graph replacement failed preparation");
            core::write_tensor_f32(input, std::vector<float>(16, 1));
            owner.compute();
            require(core::read_tensor_f32(result) == std::vector<float>(16, variant == 4 ? 9 : 65), "whole-graph or adjacent replacements failed");
        } else require(failed, "invalid region or failed preparation was accepted");
        require(stats.prepared == (variant >= 3 ? 1 : 0), "region failed at the wrong lifecycle stage");
        owner.reset();
        require(stats.prepared == stats.reset && stats.storage_alive, "failed region preparation leaked resources");
        require(second_stats.prepared == second_stats.reset && second_stats.storage_alive, "adjacent replacement cleanup mismatch");
    }
}

// Observe the real optimizer, preserving its implementation. Replacement
// nodes must never be offered to it; adjacent work still is optimized.
struct RegionOptimizerCheck {
    ggml_backend_t backend;
    ggml_tensor * first;
    ggml_tensor * last;
    void (*original)(ggml_backend_t, ggml_cgraph *);
    size_t calls = 0;
    static RegionOptimizerCheck * active;
    RegionOptimizerCheck(ggml_backend_t value, ggml_tensor * a, ggml_tensor * b)
        : backend(value), first(a), last(b), original(value->iface.graph_optimize) {
        require(!active, "nested optimizer observer");
        active = this;
        backend->iface.graph_optimize = [](ggml_backend_t value, ggml_cgraph * graph) {
            ++active->calls;
            for (int n = 0; n < ggml_graph_n_nodes(graph); ++n)
                require(ggml_graph_node(graph, n) != active->first && ggml_graph_node(graph, n) != active->last,
                    "backend optimizer received replacement nodes");
            if (active->original) active->original(value, graph);
        };
    }
    ~RegionOptimizerCheck() { backend->iface.graph_optimize = original; active = nullptr; }
};
RegionOptimizerCheck * RegionOptimizerCheck::active = nullptr;

void check_region_optimizer_barriers(core::ExecutionContext & execution) {
    for (bool replacement : {false, true}) {
        RegionStats stats;
        std::unique_ptr<core::ExecutionContext> host;
        core::PreparedGraph owner(execution, {"prepared.region.optimizer", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
        auto & persistent = owner.persistent_build_context();
        auto x = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({16}));
        auto y = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({16}));
        auto counter = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        auto one = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        for (auto tensor : {x, y, counter, one}) ggml_set_input(tensor.tensor);
        owner.allocate_persistent();
        auto & ctx = owner.build_context();
        auto * first = ggml_scale(ctx.ggml, x.tensor, 3);
        auto * last = ggml_scale_bias(ctx.ggml, first, 1, 1);
        auto * independent = ggml_scale(ctx.ggml, y.tensor, 11);
        auto * joined = ggml_add(ctx.ggml, last, independent);
        auto * advance = ggml_cpy(ctx.ggml, ggml_add(ctx.ggml, counter.tensor, one.tensor), counter.tensor);
        ggml_set_output(joined);
        ggml_set_output(advance);
        ggml_build_forward_expand(owner.graph(), joined);
        ggml_build_forward_expand(owner.graph(), advance);
        std::vector<ggml_backend_t> backends{execution.backend()};
        if (!core::is_host_backend(execution.backend())) {
            host = std::make_unique<core::ExecutionContext>(core::BackendConfig{core::BackendType::Cpu, 0, 8});
            backends.push_back(host->backend());
        }
        owner.create_scheduler(std::move(backends));
        std::unique_ptr<RegionOptimizerCheck> optimizer;
        std::unique_ptr<RegionNativeCheck> native;
        if (replacement) {
            owner.replace_region(0, {{first, last}, {last}, {}}, std::make_unique<TestRegion>(stats));
            optimizer = std::make_unique<RegionOptimizerCheck>(execution.backend(), first, last);
            native = std::make_unique<RegionNativeCheck>(execution.backend(), first, last);
        }
        owner.allocate();
        if (optimizer) require(optimizer->calls > 0, "ordinary segments were not optimized");
        if (!replacement && execution.backend_type() == core::BackendType::Vulkan) {
            int first_pos = -1, last_pos = -1, independent_pos = -1;
            for (int n = 0; n < ggml_graph_n_nodes(owner.graph()); ++n) {
                auto * node = ggml_graph_node(owner.graph(), n);
                if (node == first) first_pos = n;
                if (node == last) last_pos = n;
                if (node == independent) independent_pos = n;
            }
            require(first_pos < independent_pos && independent_pos < last_pos,
                "Vulkan control did not reproduce independent work moving between the endpoints");
        }
        core::write_tensor_f32(counter, {0});
        core::write_tensor_f32(one, {1});
        for (int request = 0; request < 4; ++request) {
            const float a = static_cast<float>(request + 1), b = static_cast<float>(request + 2);
            core::write_tensor_f32(x, std::vector<float>(16, a));
            core::write_tensor_f32(y, std::vector<float>(16, b));
            owner.compute();
            require(core::read_tensor_f32(joined) == std::vector<float>(16, (replacement ? 7 * a + 2 : 3 * a + 1) + 11 * b),
                "independent work adjacent to a region was skipped or reordered incorrectly");
            require(core::read_tensor_f32(counter.tensor) == std::vector<float>{static_cast<float>(request + 1)},
                "adjacent state update was skipped or replayed");
        }
    }
    for (bool scheduler : {false, true}) {
        RegionStats stats;
        core::PreparedGraph owner(execution, {"prepared.region.escaping", 1024 * 1024, 128});
        auto & ctx = owner.build_context();
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({16}));
        auto * first = ggml_scale(ctx.ggml, x.tensor, 3);
        auto * last = ggml_scale(ctx.ggml, first, 2);
        auto * output = ggml_add(ctx.ggml, first, last);
        ggml_set_output(output);
        ggml_build_forward_expand(owner.graph(), output);
        owner.replace_region(0, {{first, last}, {last}, {}}, std::make_unique<TestRegion>(stats));
        std::unique_ptr<core::ExecutionContext> host;
        if (scheduler) {
            std::vector<ggml_backend_t> backends{execution.backend()};
            if (!core::is_host_backend(execution.backend())) {
                host = std::make_unique<core::ExecutionContext>(core::BackendConfig{core::BackendType::Cpu, 0, 8});
                backends.push_back(host->backend());
            }
            owner.create_scheduler(std::move(backends));
        }
        bool rejected = false;
        try { owner.allocate(); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected && stats.prepared == 0, "undeclared intermediate escaped a replacement");
    }
}

void check_scheduler_region_contracts(core::ExecutionContext & execution) {
    std::unique_ptr<core::ExecutionContext> host;
    auto * cpu = execution.backend();
    if (!core::is_host_backend(cpu)) {
        host = std::make_unique<core::ExecutionContext>(core::BackendConfig{core::BackendType::Cpu, 0, 8});
        cpu = host->backend();
    }
    const auto callback = [](ggml_backend_t, ggml_cgraph *, void *) { return GGML_STATUS_FAILED; };
    for (int variant = 0; variant < (host ? 4 : 3); ++variant) {
        core::PreparedGraph owner(execution, {"prepared.region.scheduler.invalid", 1024 * 1024, 128});
        auto & ctx = owner.build_context();
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({16}));
        ggml_set_input(input.tensor);
        auto * first = ggml_scale(ctx.ggml, input.tensor, 3);
        auto * last = ggml_scale(ctx.ggml, first, 2);
        ggml_set_output(last);
        ggml_build_forward_expand(owner.graph(), last);
        std::vector<ggml_backend_t> backends{execution.backend()};
        if (host) backends.push_back(cpu);
        auto * scheduler = owner.create_scheduler(std::move(backends));
        ggml_tensor * nodes[]{first, last};
        ggml_tensor * outputs[]{last};
        ggml_backend_graph_region regions[]{{nodes, 2, outputs, 1, nullptr, 0, callback, nullptr},
            {outputs, 1, outputs, 1, nullptr, 0, callback, nullptr}};
        if (variant == 0) nodes[0] = input.tensor;
        if (variant == 1) std::swap(nodes[0], nodes[1]);
        if (variant == 3) {
            ggml_backend_sched_set_tensor_backend(scheduler, first, cpu);
            ggml_backend_sched_set_tensor_backend(scheduler, last, execution.backend());
        }
        require(ggml_backend_sched_set_regions(scheduler, regions, variant == 2 ? 2 : 1), "valid descriptor registration failed");
        require(!ggml_backend_sched_reserve(scheduler, owner.graph()), "scheduler reserved an invalid region");
        bool failed = false;
        try { owner.allocate(); } catch (const std::runtime_error & error) {
            failed = std::string(error.what()).find("Cannot allocate") != std::string::npos;
        }
        require(failed, "scheduler accepted an absent, reversed, overlapping or cross-split region");
        if (variant == 3) require(ggml_backend_sched_get_n_splits(scheduler) >= 2, "cross-split test did not place two backends");
    }
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> parallel(
        ggml_backend_sched_new(&cpu, nullptr, 1, 128, true, false), ggml_backend_sched_free);
    ggml_tensor boundary{};
    ggml_tensor * nodes[]{&boundary};
    const ggml_backend_graph_region region{nodes, 1, nodes, 1, nullptr, 0, callback, nullptr};
    require(!ggml_backend_sched_set_regions(parallel.get(), &region, 1), "parallel scheduler accepted unsupported region replacement");
    require(ggml_backend_sched_set_regions(parallel.get(), nullptr, 0), "disabling replacement rejected normal parallel scheduling");

    core::PreparedGraph restored(execution, {"prepared.region.callback", 1024 * 1024, 128});
    auto & ctx = restored.build_context();
    auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({16}));
    ggml_set_input(input.tensor);
    auto * first = ggml_scale(ctx.ggml, input.tensor, 3);
    auto * last = ggml_scale(ctx.ggml, first, 2);
    ggml_set_output(last);
    ggml_build_forward_expand(restored.graph(), last);
    std::vector<ggml_backend_t> backends{execution.backend()};
    if (cpu != execution.backend()) backends.push_back(cpu);
    auto * scheduler = restored.create_scheduler(std::move(backends));
    int original_calls = 0, updated_calls = 0;
    const auto observe = [](ggml_tensor *, bool ask, void * data) {
        if (!ask) ++*static_cast<int *>(data);
        return true;
    };
    ggml_backend_sched_set_eval_callback(scheduler, observe, &original_calls);
    ggml_tensor * exact[]{first, last};
    ggml_tensor * outputs[]{last};
    ggml_tensor * deferred[]{first};
    const ggml_backend_graph_region enabled{exact, 2, outputs, 1, deferred, 1, callback, nullptr};
    require(ggml_backend_sched_set_regions(scheduler, &enabled, 1), "callback test could not enable regions");
    ggml_backend_sched_set_eval_callback(scheduler, observe, &updated_calls);
    require(ggml_backend_sched_set_regions(scheduler, nullptr, 0), "callback test could not disable regions");
    restored.allocate();
    core::write_tensor_f32(input, std::vector<float>(16, 2));
    restored.compute();
    require(original_calls == 0 && updated_calls == 2, "disabling regions did not restore the latest unfiltered callback");
    require(core::read_tensor_f32(last) == std::vector<float>(16, 12), "disabled regions replaced normal computation");
}

void check_owned_graph(core::ExecutionContext & execution) {
    core::PreparedGraph completed(execution, {"test.completed_output", 64 * 1024, 32});
    const auto sink = core::make_tensor(completed.build_context(), GGML_TYPE_F32,
        core::TensorShape::from_dims({4}));
    ggml_set_output(sink.tensor);
    ggml_build_forward_expand(completed.graph(), sink.tensor);
    completed.allocate();
    core::PreparedGraph graph(execution, {"test.prepared_graph", 256 * 1024, 128});
    auto & ctx = graph.build_context();
    const auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({4}));
    ggml_set_input(input.tensor);
    const auto first = modules::ScaleBiasModule({2.0F, 1.0F}).build(ctx, input);
    const auto second = modules::ScaleBiasModule({-1.0F, 3.0F}).build(ctx, input);
    for (const auto & output : {first, second}) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph.graph(), output.tensor);
    }
    ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx.ggml, first.tensor, sink.tensor));
    require(!graph.allocated(), "graph must start unallocated");
    bool rejected = false;
    try { graph.compute(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "execution before allocation must fail");
    graph.allocate();
    require(graph.allocated(), "graph allocation state");
    require(graph.has_host_plan() == execution.uses_host_graph_plan(), "backend plan selection");
    const auto * input_address = input.tensor->data;
    const auto * first_address = first.tensor->data;
    const auto * second_address = second.tensor->data;
    for (int request = 0; request < 3; ++request) {
        const float offset = static_cast<float>(request);
        core::write_tensor_f32(input, {offset, offset + 1, offset + 2, offset + 3});
        graph.compute();
        require(core::read_tensor_f32(first.tensor) ==
            std::vector<float>{2 * offset + 1, 2 * offset + 3, 2 * offset + 5, 2 * offset + 7},
            "first output differs from host arithmetic");
        require(core::read_tensor_f32(second.tensor) ==
            std::vector<float>{3 - offset, 2 - offset, 1 - offset, -offset},
            "second output differs from host arithmetic");
        require(input.tensor->data == input_address && first.tensor->data == first_address &&
            second.tensor->data == second_address, "cached graph addresses changed");
    }
    rejected = false;
    try { graph.allocate(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "second owned allocation must fail");

    // Reset must drain the old submission before releasing/reusing its storage.
    core::write_tensor_f32(input, {10, 20, 30, 40});
    require(ggml_backend_graph_compute_async(execution.backend(), graph.graph()) == GGML_STATUS_SUCCESS,
        "queued execution before reset failed");
    const auto * metadata = ctx.ggml;
    {
        ResetSyncCheck sync(graph);
        graph.reset();
        require(sync.calls == 1 && sync.storage_alive,
            "reset must synchronize exactly once before invalidating graph/allocation");
    }
    require(core::read_tensor_f32(sink.tensor) == std::vector<float>{21, 41, 61, 81},
        "reset did not finish queued work");
    for (int64_t width : {33, 7}) {
        require(!graph.allocated() && !graph.has_host_plan(), "reset retained allocation/plan state");
        require(graph.build_context().ggml == metadata && ggml_graph_n_nodes(graph.graph()) == 0,
            "reset did not produce an empty graph in the retained metadata allocation");
        auto & build = graph.build_context();
        const auto values = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({width}));
        const auto result = modules::ScaleBiasModule({3.0F, -2.0F}).build(build, values);
        ggml_set_input(values.tensor);
        ggml_set_output(result.tensor);
        ggml_build_forward_expand(graph.graph(), result.tensor);
        graph.allocate();
        std::vector<float> data(static_cast<size_t>(width)), expected(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i);
            expected[i] = 3 * data[i] - 2;
        }
        core::write_tensor_f32(values, data);
        graph.compute();
        require(core::read_tensor_f32(result.tensor) == expected, "rebuilt graph used stale data or captures");
        graph.reset();
    }
}

void check_buffer_release(core::ExecutionContext & execution) {
    constexpr int64_t width = 32768;
    core::PreparedGraph parameters(execution, {"test.release.parameters", 65536, 32, 65536});
    const auto weights = core::make_tensor(parameters.persistent_build_context(), GGML_TYPE_F32,
        core::TensorShape::from_dims({width}));
    parameters.allocate_persistent();
    core::write_tensor_f32(weights, std::vector<float>(width, 7));
    const auto * weight_address = weights.tensor->data;
    core::PreparedGraph owner(execution, {"test.release.buffers", 1024 * 1024, 1024, 65536});
    const auto input = core::make_tensor(owner.persistent_build_context(), GGML_TYPE_F32,
        core::TensorShape::from_dims({width}));
    const auto saved = core::make_tensor(owner.persistent_build_context(), GGML_TYPE_F32, input.shape);
    auto & ctx = owner.build_context();
    const auto foreign_view = modules::SliceModule({0, 0, width}).build(ctx, weights);
    const auto output = modules::AddModule().build(ctx,
        modules::ScaleBiasModule({3.f, -2.f}).build(ctx, input), foreign_view);
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(owner.graph(), ggml_cpy(ctx.ggml, output.tensor, saved.tensor));
    const auto small_id = owner.add_graph("small");
    const auto small = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    ggml_set_input(small.tensor);
    const auto small_output = modules::ScaleBiasModule({2.f, 1.f}).build(ctx, small);
    ggml_set_output(small_output.tensor);
    ggml_build_forward_expand(owner.graph(small_id), small_output.tensor);
    auto * metadata = ctx.ggml;
    auto * allocation_graph = owner.graph();
    for (int cycle = 0; cycle < 3; ++cycle) {
        owner.allocate_persistent();
        owner.allocate_releasable();
        for (int request = 0; request < 6; ++request) {
            const float value = static_cast<float>(cycle * 9 + request);
            core::write_tensor_f32(input, std::vector<float>(width, value));
            owner.compute();
            require(core::read_tensor_f32(saved.tensor) == std::vector<float>(width, 3 * value + 5),
                "buffer reallocation used stale persistent data or external parameters");
            core::write_tensor_f32(small, std::vector<float>(17, -value));
            owner.compute(small_id);
            require(core::read_tensor_f32(small_output.tensor) == std::vector<float>(17, 1 - 2 * value),
                "buffer reallocation used stale scratch or graph captures");
        }
        {
            ResetSyncCheck sync(owner);
            owner.release_buffers();
            require(sync.calls == 1 && sync.storage_alive, "buffer release did not drain execution before freeing storage");
        }
        require(!owner.allocated() && !owner.has_host_plan() && owner.workspace_bytes() == 0 && owner.persistent_bytes() == 0,
            "buffer release retained allocations or plans");
        require(owner.graph_count() == 2 && owner.graph() == allocation_graph && owner.build_context().ggml == metadata,
            "buffer release rebuilt the graph metadata");
        require(!input.tensor->data && !saved.tensor->data && !small.tensor->data && !output.tensor->data,
            "released tensor handles retained freed pointers");
        require(weights.tensor->data == weight_address && foreign_view.tensor->data == weight_address &&
            core::read_tensor_f32(weights.tensor) == std::vector<float>(width, 7),
            "buffer release modified externally owned parameter storage");
        bool rejected = false;
        try { owner.compute(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "execution with released buffers was accepted");
    }
    owner.reset();
    owner.allocate_persistent();
    const auto ordinary = core::make_tensor(owner.build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    ggml_build_forward_expand(owner.graph(), ordinary.tensor);
    owner.allocate();
    bool rejected = false;
    try { owner.release_buffers(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "ordinary allocations were silently made releasable");
}

void check_workspace(core::ExecutionContext & execution) {
    // Persistent state and scratch have separate owners. Graphs/captures must
    // be released before the shared scratch allocation is freed.
    core::PreparedGraph state(execution, {"test.prepared_state", 64 * 1024, 32});
    auto & state_ctx = state.build_context();
    const auto latent = core::make_tensor(state_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({4}));
    ggml_set_input(latent.tensor);
    ggml_set_output(latent.tensor);
    ggml_build_forward_expand(state.graph(), latent.tensor);
    state.allocate();

    core::GraphWorkspace workspace(execution.backend());
    {
        core::PreparedGraph first(execution, {"test.prepared_first", 64 * 1024, 32});
        core::PreparedGraph second(execution, {"test.prepared_second", 64 * 1024, 32});
        const auto first_output = modules::ScaleBiasModule({2.0F, 1.0F})
            .build(first.build_context(), latent);
        const auto second_output = modules::ScaleBiasModule({-1.0F, 0.5F})
            .build(second.build_context(), latent);
        ggml_set_output(first_output.tensor);
        ggml_build_forward_expand(first.graph(), first_output.tensor);
        ggml_set_output(second_output.tensor);
        ggml_build_forward_expand(second.graph(), second_output.tensor);
        workspace.add(first);
        workspace.add(second);
        bool rejected = false;
        try { first.reset(); } catch (const std::logic_error &) { rejected = true; }
        require(rejected, "reset of a registered, unprepared graph must fail");
        {
            core::GraphWorkspace other(execution.backend());
            rejected = false;
            try { other.add(first); } catch (const std::logic_error &) { rejected = true; }
            require(rejected, "one graph must not be registered with two workspaces");
        }
        workspace.prepare();
        require(workspace.bytes() > 0, "shared workspace is empty");
        require(first.allocated() && second.allocated(), "shared graphs were not prepared");
        rejected = false;
        try { first.allocate(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "owned allocation after shared preparation must fail");
        rejected = false;
        try { workspace.add(state); } catch (const std::logic_error &) { rejected = true; }
        require(rejected, "an allocated graph must not enter another workspace");
        rejected = false;
        try { first.reset(); } catch (const std::logic_error &) { rejected = true; }
        require(rejected, "reset of a prepared shared graph must fail");
        for (int request = 0; request < 3; ++request) {
            const float x = static_cast<float>(request * 7 - 3);
            core::write_tensor_f32(latent, {x, x + 1, x + 2, x + 3});
            first.compute();
            require(core::read_tensor_f32(first_output.tensor) == std::vector<float>{2*x+1, 2*x+3, 2*x+5, 2*x+7},
                "first shared graph arithmetic");
            second.compute();
            require(core::read_tensor_f32(second_output.tensor) == std::vector<float>{0.5F-x, -0.5F-x, -1.5F-x, -2.5F-x},
                "second shared graph arithmetic");
            first.compute();
            require(core::read_tensor_f32(first_output.tensor) == std::vector<float>{2*x+1, 2*x+3, 2*x+5, 2*x+7},
                "returning to the first shared graph changed its result");
        }
        const auto capacity = workspace.bytes();
        {
            ResetSyncCheck sync(first);
            workspace.reset();
            require(sync.calls == 1 && sync.storage_alive,
                "workspace must drain work before invalidating registered graphs");
        }
        core::write_tensor_f32(latent, {1, 2, 3, 4});
        require(workspace.bytes() == capacity, "workspace reset dropped retained allocation capacity");
        require(!first.allocated() && !second.allocated() && !first.has_host_plan() && !second.has_host_plan(),
            "workspace reset retained execution state");
        rejected = false;
        try { first.compute(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalidated shared graph executed");
        rejected = false;
        try { first.allocate(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalidated graph allocated without reset");
        first.reset();
        second.reset();
        const auto rebuilt_first = modules::ScaleBiasModule({4.0F, -1.0F})
            .build(first.build_context(), latent);
        const auto rebuilt_second = modules::ScaleBiasModule({-2.0F, 0.0F})
            .build(second.build_context(), latent);
        ggml_set_output(rebuilt_first.tensor);
        ggml_build_forward_expand(first.graph(), rebuilt_first.tensor);
        ggml_set_output(rebuilt_second.tensor);
        ggml_build_forward_expand(second.graph(), rebuilt_second.tensor);
        workspace.add(first);
        workspace.prepare();
        second.allocate();
        first.compute();
        second.compute();
        require(core::read_tensor_f32(rebuilt_first.tensor) == std::vector<float>{3, 7, 11, 15},
            "rebuilt shared graph arithmetic");
        require(core::read_tensor_f32(rebuilt_second.tensor) == std::vector<float>{-2, -4, -6, -8},
            "shared-to-owned allocation transition failed");
    }
    workspace.reset();
    // Existing raw-graph callers remain valid after using prepared owners.
    core::PreparedGraph legacy(execution, {"test.raw_workspace", 64 * 1024, 32});
    const auto output = modules::ScaleBiasModule({3.0F, 0.0F}).build(legacy.build_context(), latent);
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(legacy.graph(), output.tensor);
    workspace.add(legacy.graph());
    workspace.prepare();
    require(core::compute_backend_graph(execution.backend(), legacy.graph()) == GGML_STATUS_SUCCESS,
        "legacy raw-graph execution failed");
    require(core::read_tensor_f32(output.tensor) == std::vector<float>{3, 6, 9, 12},
        "legacy workspace arithmetic changed");

    core::PreparedGraph survivor(execution, {"test.workspace_survivor", 64 * 1024, 32});
    const auto shared_output = modules::ScaleBiasModule({2.0F, 0.0F})
        .build(survivor.build_context(), latent);
    ggml_set_output(shared_output.tensor);
    ggml_build_forward_expand(survivor.graph(), shared_output.tensor);
    {
        core::GraphWorkspace temporary(execution.backend());
        temporary.add(survivor);
        temporary.prepare();
        survivor.compute();
    }
    require(!survivor.allocated() && !survivor.has_host_plan() && !survivor.graph(),
        "workspace destruction left a graph using freed scratch");
    survivor.reset();
    const auto owned_output = modules::ScaleBiasModule({5.0F, 0.0F})
        .build(survivor.build_context(), latent);
    ggml_set_output(owned_output.tensor);
    ggml_build_forward_expand(survivor.graph(), owned_output.tensor);
    survivor.allocate();
    survivor.compute();
    require(core::read_tensor_f32(owned_output.tensor) == std::vector<float>{5, 10, 15, 20},
        "workspace destruction prevented independent graph reuse");
}

void check_workspace_sizes(core::ExecutionContext & execution) {
    core::GraphWorkspace workspace(execution.backend());
    size_t previous_capacity = 0;
    // Grow, then return to smaller graphs without dropping retained storage.
    for (int64_t width : {256, 65536, 1024}) {
        core::PreparedGraph small(execution, {"test.small_workspace", 256 * 1024, 128});
        core::PreparedGraph large(execution, {"test.large_workspace", 256 * 1024, 128});
        auto & small_ctx = small.build_context();
        auto & large_ctx = large.build_context();
        const auto small_input = core::make_tensor(small_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({17}));
        const auto large_input = core::make_tensor(large_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({width}));
        const auto small_output = modules::ScaleBiasModule({2.0F, 1.0F}).build(small_ctx, small_input);
        const auto intermediate = modules::ScaleBiasModule({3.0F, -2.0F}).build(large_ctx, large_input);
        const auto large_output = modules::ScaleBiasModule({2.0F, 1.0F}).build(large_ctx, intermediate);
        // Keep both roots live: the large graph requires materially more scratch.
        ggml_set_input(small_input.tensor);
        ggml_set_input(large_input.tensor);
        ggml_set_output(small_output.tensor);
        ggml_set_output(intermediate.tensor);
        ggml_set_output(large_output.tensor);
        ggml_build_forward_expand(small.graph(), small_output.tensor);
        ggml_build_forward_expand(large.graph(), intermediate.tensor);
        ggml_build_forward_expand(large.graph(), large_output.tensor);
        // Exercise both registration orders; largest must be found, not assumed.
        if (width == 65536) { workspace.add(large); workspace.add(small); }
        else { workspace.add(small); workspace.add(large); }
        workspace.prepare();
        const auto capacity = workspace.bytes();
        require(capacity >= previous_capacity, "workspace lost retained capacity");
        if (width == 65536) require(capacity > previous_capacity, "large graph did not grow workspace");
        const auto * small_address = small_output.tensor->data;
        const auto * large_address = large_output.tensor->data;
        const auto * input_address = large_input.tensor->data;
        for (int request = 0; request < 5; ++request) {
            std::vector<float> small_values(17), small_expected(17);
            std::vector<float> large_values(static_cast<size_t>(width)), large_expected(large_values.size());
            for (size_t i = 0; i < large_values.size(); ++i) {
                large_values[i] = static_cast<float>(static_cast<int>(i % 31) - 15 + request * 3);
                large_expected[i] = 6 * large_values[i] - 3;
            }
            for (size_t i = 0; i < small_values.size(); ++i) {
                small_values[i] = static_cast<float>(static_cast<int>(i) - request * 5);
                small_expected[i] = 2 * small_values[i] + 1;
            }
            core::write_tensor_f32(small_input, small_values);
            small.compute();
            require(core::read_tensor_f32(small_output.tensor) == small_expected, "small workspace output stale");
            core::write_tensor_f32(large_input, large_values);
            large.compute();
            require(core::read_tensor_f32(large_output.tensor) == large_expected, "large workspace output stale");
            // Large graph may overwrite shared scratch, including the small input.
            core::write_tensor_f32(small_input, small_values);
            small.compute();
            require(core::read_tensor_f32(small_output.tensor) == small_expected, "small graph corrupted after large graph");
            require(small_output.tensor->data == small_address && large_output.tensor->data == large_address &&
                large_input.tensor->data == input_address && workspace.bytes() == capacity,
                "prepared execution changed addresses or grew workspace");
        }
        previous_capacity = capacity;
        workspace.reset();
        require(workspace.bytes() == capacity, "reset released workspace capacity");
    }
}

void check_graph_set(core::ExecutionContext & execution) {
    PlanCheck plans(execution);
    core::PreparedGraph owner(execution, {"test.graph_set", 512 * 1024, 128, 64 * 1024});
    auto & persistent = owner.persistent_build_context();
    const auto input = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    const auto hidden = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    const auto result = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    const auto indices = core::make_tensor(persistent, GGML_TYPE_I32, core::TensorShape::from_dims({3}));
    owner.allocate_persistent();
    const auto * persistent_context = persistent.ggml;
    const auto * input_address = input.tensor->data;
    const auto * hidden_address = hidden.tensor->data;
    const auto persistent_bytes = owner.persistent_bytes();
    require(persistent_bytes > 0, "persistent buffer is empty");
    core::write_tensor_i32(indices, {7, -3, 19});
    bool rejected = false;
    try { owner.allocate_persistent(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "persistent tensors were allocated twice");
    rejected = false;
    try { owner.persistent_build_context(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "allocated persistent metadata remained open for declaration");

    size_t previous_capacity = 0;
    int cycle = 0;
    for (int64_t width : {256, 65536, 1024}) {
        auto & ctx = owner.build_context();
        const auto * metadata = ctx.ggml;
        const auto attention = modules::ScaleBiasModule({2.0F, static_cast<float>(cycle)}).build(ctx, input);
        ggml_build_forward_expand(owner.graph(), ggml_cpy(ctx.ggml, attention.tensor, hidden.tensor));
        const auto clobber_id = owner.add_graph("clobber");
        const auto clobber_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({width}));
        const auto clobber_output = modules::ScaleBiasModule({7.0F, -5.0F}).build(ctx, clobber_input);
        ggml_set_input(clobber_input.tensor);
        ggml_set_output(clobber_output.tensor);
        ggml_build_forward_expand(owner.graph(clobber_id), clobber_output.tensor);
        const auto expert_id = owner.add_graph("expert");
        const auto expert = modules::ScaleBiasModule({-1.0F, 3.0F}).build(ctx, hidden);
        ggml_build_forward_expand(owner.graph(expert_id), ggml_cpy(ctx.ggml, expert.tensor, result.tensor));
        const auto finish_id = owner.add_graph("finish");
        const auto output = modules::ScaleBiasModule({3.0F, 1.0F}).build(ctx, result);
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(owner.graph(finish_id), output.tensor);
        owner.allocate();
        const auto prepared_plans = plans.created;
        require(plans.created - plans.freed == (execution.uses_host_graph_plan() ? owner.graph_count() : size_t{0}),
            "graph set did not own exactly one plan per graph");
        require(owner.graph_count() == 4 && owner.allocated(), "graph set was not prepared together");
        require(owner.workspace_bytes() >= previous_capacity, "graph-set rebuild shrank scratch capacity");
        if (cycle == 1) require(owner.workspace_bytes() > previous_capacity, "larger graph did not grow scratch");
        const auto capacity = owner.workspace_bytes();
        const auto * output_address = output.tensor->data;
        const auto * scratch_address = clobber_input.tensor->data;
        require(input.tensor->buffer != output.tensor->buffer && hidden.tensor->buffer != output.tensor->buffer,
            "persistent state was placed in shared scratch");
        for (size_t id = 0; id < owner.graph_count(); ++id)
            require(owner.has_host_plan(id) == execution.uses_host_graph_plan(), "missing per-graph host plan");
        rejected = false;
        try { owner.add_graph("too_late"); } catch (const std::logic_error &) { rejected = true; }
        require(rejected, "graph set accepted registration after preparation");
        rejected = false;
        try { owner.compute(owner.graph_count()); } catch (const std::out_of_range &) { rejected = true; }
        require(rejected, "unknown graph ID was accepted");

        std::vector<float> expected_hidden(17), expected_result(17);
        for (int request = 0; request < 5; ++request) {
            std::vector<float> values(17), expected(17);
            std::vector<float> noise(static_cast<size_t>(width));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<float>(static_cast<int>(i) - request * 3 + cycle * 5);
                expected_hidden[i] = 2 * values[i] + static_cast<float>(cycle);
                expected_result[i] = -expected_hidden[i] + 3;
                expected[i] = 3 * expected_result[i] + 1;
            }
            for (size_t i = 0; i < noise.size(); ++i)
                noise[i] = static_cast<float>(static_cast<int>(i % 23) + request + cycle * 9);
            core::write_tensor_f32(input, values);
            owner.compute();
            core::write_tensor_f32(clobber_input, noise);
            owner.compute(clobber_id);
            owner.compute(expert_id);
            core::write_tensor_f32(clobber_input, noise);
            owner.compute(clobber_id);
            owner.compute(finish_id);
            require(core::read_tensor_f32(output.tensor) == expected, "multi-phase graph output differs from host arithmetic");
            require(core::read_tensor_f32(hidden.tensor) == expected_hidden &&
                core::read_tensor_f32(result.tensor) == expected_result && core::read_tensor_f32(input.tensor) == values,
                "scratch execution corrupted persistent inputs or handoffs");
            require(core::read_tensor_i32(indices.tensor) == std::vector<int32_t>{7, -3, 19}, "unrelated persistent state changed");
            require(input.tensor->data == input_address && hidden.tensor->data == hidden_address &&
                output.tensor->data == output_address && clobber_input.tensor->data == scratch_address &&
                owner.workspace_bytes() == capacity && owner.persistent_bytes() == persistent_bytes,
                "repeated execution reallocated persistent state or graph scratch");
        }
        require(plans.created == prepared_plans, "execution rebuilt host plans");
        previous_capacity = capacity;
        {
            ResetSyncCheck sync(owner);
            owner.reset();
            require(sync.calls == 1 && sync.storage_alive, "graph-set reset did not drain all graphs before release");
        }
        require(plans.created == plans.freed, "reset leaked per-graph host plans");
        require(owner.graph_count() == 1 && !owner.allocated() && !owner.has_host_plan() &&
            ggml_graph_n_nodes(owner.graph()) == 0 && owner.build_context().ggml == metadata,
            "graph-set reset did not rebuild empty metadata");
        require(owner.workspace_bytes() == capacity && owner.persistent_bytes() == persistent_bytes &&
            persistent.ggml == persistent_context && input.tensor->data == input_address && hidden.tensor->data == hidden_address,
            "graph reset changed persistent ownership or dropped scratch capacity");
        require(core::read_tensor_f32(hidden.tensor) == expected_hidden && core::read_tensor_f32(result.tensor) == expected_result,
            "graph reset cleared persistent state");
        ++cycle;
    }
    owner.clear_persistent();
    require(core::read_tensor_f32(input.tensor) == std::vector<float>(17, 0) &&
        core::read_tensor_f32(hidden.tensor) == std::vector<float>(17, 0) &&
        core::read_tensor_f32(result.tensor) == std::vector<float>(17, 0) &&
        core::read_tensor_i32(indices.tensor) == std::vector<int32_t>(3, 0), "explicit persistent clear failed");
    core::write_tensor_f32(input, std::vector<float>(17, 4));
    const auto final = modules::ScaleBiasModule({2.0F, 1.0F}).build(owner.build_context(), input);
    ggml_set_output(final.tensor);
    ggml_build_forward_expand(owner.graph(), final.tensor);
    owner.allocate();
    owner.compute();
    require(core::read_tensor_f32(final.tensor) == std::vector<float>(17, 9), "graph rebuild after state clear failed");
    {
        BufferFreeCheck persistent_free(input.tensor->buffer, owner);
        ResetSyncCheck sync(owner);
        owner.release();
        require(sync.calls == 1 && sync.storage_alive, "release freed storage before draining execution");
        require(persistent_free.freed && persistent_free.graphs_released,
            "release did not free the real persistent buffer after graph teardown");
        require(plans.created == plans.freed, "release leaked host plans");
    }
    require(!owner.allocated() && owner.graph_count() == 0 && owner.persistent_bytes() == 0 &&
        owner.workspace_bytes() == 0 && owner.build_context().ggml == nullptr, "release retained owned resources");
    owner.release();
    owner.reset();
    auto & new_state = owner.persistent_build_context();
    const auto fresh = core::make_tensor(new_state, GGML_TYPE_F32, core::TensorShape::from_dims({7}));
    owner.allocate_persistent();
    core::write_tensor_f32(fresh, std::vector<float>(7, -2));
    const auto fresh_output = modules::ScaleBiasModule({4.0F, 3.0F}).build(owner.build_context(), fresh);
    ggml_set_output(fresh_output.tensor);
    ggml_build_forward_expand(owner.graph(), fresh_output.tensor);
    owner.allocate();
    owner.compute();
    require(core::read_tensor_f32(fresh_output.tensor) == std::vector<float>(7, -5), "released graph set could not be recreated");
}

void check_external_graph_set(core::ExecutionContext & execution) {
    core::PreparedGraph owner(execution, {"test.external_set", 256 * 1024, 128, 64 * 1024});
    auto & state = owner.persistent_build_context();
    const auto input = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
    const auto hidden = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
    auto & ctx = owner.build_context();
    const auto first = modules::ScaleBiasModule({2.0F, 1.0F}).build(ctx, input);
    ggml_build_forward_expand(owner.graph(), ggml_cpy(ctx.ggml, first.tensor, hidden.tensor));
    const auto second_id = owner.add_graph("second");
    const auto window = core::wrap_tensor(ggml_view_1d(ctx.ggml, hidden.tensor, 17, 4 * sizeof(float)),
        core::TensorShape::from_dims({17}));
    const auto second = modules::ScaleBiasModule({3.0F, -2.0F}).build(ctx, window);
    ggml_set_output(second.tensor);
    ggml_build_forward_expand(owner.graph(second_id), second.tensor);
    core::GraphWorkspace workspace(execution.backend());
    bool rejected = false;
    try { workspace.add(owner); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "workspace registered graph set with unallocated persistent tensors");
    rejected = false;
    try { owner.allocate(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "graph set prepared with unallocated persistent tensors");
    owner.allocate_persistent();
    workspace.add(owner);
    workspace.prepare();
    require(owner.allocated() && owner.workspace_bytes() == workspace.bytes(), "external graph-set preparation failed");
    for (int request = 0; request < 5; ++request) {
        std::vector<float> values(33), expected(17);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i) + request);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = 6 * values[i + 4] + 1;
        core::write_tensor_f32(input, values);
        owner.compute();
        owner.compute(second_id);
        require(core::read_tensor_f32(second.tensor) == expected, "external graph set output stale");
    }
    const auto * state_address = hidden.tensor->data;
    {
        ResetSyncCheck sync(owner);
        workspace.reset();
        require(sync.calls == 1 && sync.storage_alive, "external workspace invalidated graphs before synchronization");
    }
    require(!owner.allocated() && !owner.graph() && !owner.graph(second_id) &&
        !owner.has_host_plan() && !owner.has_host_plan(second_id), "external reset did not invalidate every graph/plan");
    std::vector<float> expected_hidden(33), expected_rebuilt(33);
    for (size_t i = 0; i < expected_hidden.size(); ++i) {
        expected_hidden[i] = 2 * static_cast<float>(i) + 9;
        expected_rebuilt[i] = -2 * expected_hidden[i] + 1;
    }
    require(hidden.tensor->data == state_address && core::read_tensor_f32(hidden.tensor) == expected_hidden,
        "external reset lost persistent state");
    owner.reset();
    const auto rebuilt = modules::ScaleBiasModule({-2.0F, 1.0F}).build(owner.build_context(), hidden);
    ggml_set_output(rebuilt.tensor);
    ggml_build_forward_expand(owner.graph(), rebuilt.tensor);
    owner.allocate();
    owner.compute();
    require(core::read_tensor_f32(rebuilt.tensor) == expected_rebuilt, "external-to-owned graph-set reuse failed");
}

void check_graph_set_contracts(core::ExecutionContext & execution) {
    core::PreparedGraph owner(execution, {"test.graph_set_contracts", 256 * 1024, 128});
    bool rejected = false;
    try { owner.persistent_build_context(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "persistent declaration without metadata capacity was accepted");
    rejected = false;
    try { owner.allocate_persistent(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "persistent allocation without tensors was accepted");
    rejected = false;
    try { owner.clear_persistent(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "clearing absent persistent storage was accepted");
    rejected = false;
    try { owner.add_graph(""); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "empty graph name was accepted");
    const auto second_id = owner.add_graph("second");
    rejected = false;
    try { owner.add_graph("second"); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && owner.graph_count() == 2, "duplicate graph name changed graph set");
    const auto scratch = core::make_tensor(owner.build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({33}));
    ggml_set_input(scratch.tensor);
    const auto first = modules::ScaleBiasModule({2.0F, 1.0F}).build(owner.build_context(), scratch);
    const auto second = modules::ScaleBiasModule({3.0F, -2.0F}).build(owner.build_context(), scratch);
    ggml_set_output(first.tensor);
    ggml_set_output(second.tensor);
    ggml_build_forward_expand(owner.graph(), first.tensor);
    ggml_build_forward_expand(owner.graph(second_id), second.tensor);
    rejected = false;
    try { owner.allocate(); } catch (const std::logic_error & error) {
        rejected = std::string(error.what()).find("persistent storage") != std::string::npos;
    }
    require(rejected && !owner.allocated(), "shared transient input was silently given conflicting scratch layouts");
    core::GraphWorkspace workspace(execution.backend());
    rejected = false;
    try { workspace.add(owner); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "external workspace accepted shared transient inputs");
    owner.reset();
    // A validation failure must leave the owner recoverable, not partly registered.
    const auto input = core::make_tensor(owner.build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({7}));
    const auto output = modules::ScaleBiasModule({2.0F, 1.0F}).build(owner.build_context(), input);
    ggml_set_input(input.tensor);
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(owner.graph(), output.tensor);
    workspace.add(owner);
    workspace.prepare();
    core::write_tensor_f32(input, std::vector<float>(7, 3));
    owner.compute();
    require(core::read_tensor_f32(output.tensor) == std::vector<float>(7, 7), "validation failure poisoned owner/workspace reuse");
}

void check_graph_set_destruction(core::ExecutionContext & execution) {
    core::PreparedGraph completed(execution, {"test.destruction_sink", 64 * 1024, 32});
    const auto sink = core::make_tensor(completed.build_context(), GGML_TYPE_F32,
        core::TensorShape::from_dims({33}));
    ggml_set_output(sink.tensor);
    ggml_build_forward_expand(completed.graph(), sink.tensor);
    completed.allocate();
    for (bool external : {false, true}) {
        PlanCheck plans(execution);
        core::GraphWorkspace workspace(execution.backend());
        auto owner = std::make_unique<core::PreparedGraph>(execution,
            core::PreparedGraphConfig{"test.destruction_set", 256 * 1024, 128, 64 * 1024});
        auto & state = owner->persistent_build_context();
        const auto input = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
        const auto hidden = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
        owner->allocate_persistent();
        auto & ctx = owner->build_context();
        const auto first = modules::ScaleBiasModule({2.0F, 1.0F}).build(ctx, input);
        ggml_build_forward_expand(owner->graph(), ggml_cpy(ctx.ggml, first.tensor, hidden.tensor));
        const auto second_id = owner->add_graph("second");
        const auto second = modules::ScaleBiasModule({3.0F, -2.0F}).build(ctx, hidden);
        ggml_build_forward_expand(owner->graph(second_id), ggml_cpy(ctx.ggml, second.tensor, sink.tensor));
        if (external) { workspace.add(*owner); workspace.prepare(); }
        else owner->allocate();
        for (int request = 0; request < 4; ++request) {
            core::write_tensor_f32(input, std::vector<float>(33, static_cast<float>(request)));
            owner->compute();
            owner->compute(second_id);
        }
        core::write_tensor_f32(input, std::vector<float>(33, 7));
        require(ggml_backend_graph_compute_async(execution.backend(), owner->graph()) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute_async(execution.backend(), owner->graph(second_id)) == GGML_STATUS_SUCCESS,
            "queued graph set failed before destruction");
        {
            ResetSyncCheck sync(*owner);
            BufferFreeCheck persistent_free(input.tensor->buffer, *owner);
            owner.reset();
            require(sync.calls == 1 && sync.storage_alive, "destruction did not synchronize before graph teardown");
            require(persistent_free.freed && persistent_free.graphs_released,
                "destruction did not release persistent buffer after all graphs");
        }
        require(plans.created == plans.freed, "destruction leaked graph plans");
        require(core::read_tensor_f32(sink.tensor) == std::vector<float>(33, 43), "destruction lost pending graph output");
        // In the external case, destruction must remove every registered graph.
        workspace.reset();
    }
}

void check_entry_points(core::ExecutionContext & execution) {
    for (bool external : {false, true}) {
        size_t baseline_bytes = 0;
        std::vector<size_t> baseline_layout;
        for (bool shared_entry : {false, true}) {
            PlanCheck plans(execution);
            core::PreparedGraph owner(execution, {"test.entry_points", 512 * 1024, 128, 64 * 1024});
            auto & ctx = owner.build_context();
            auto & state = owner.persistent_build_context();
            const auto body_state = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({11, 3}));
            const auto full_state = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({11, 3}));
            owner.allocate_persistent();
            const auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({11, 3}));
            const auto head_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({64, 256}));
            ggml_set_input(input.tensor);
            ggml_set_input(head_input.tensor);
            const auto body_id = shared_entry ? owner.add_entry_point("body", 0) : size_t{0};
            // Same construction as the full/body pattern: build state updates,
            // copy the body into the full graph, then append the full-only head.
            auto * body = shared_entry ? owner.graph(body_id) : ggml_new_graph_custom(ctx.ggml, 128, false);
            const auto hidden = modules::ScaleBiasModule({2.0F, 1.0F}).build(ctx, input);
            ggml_build_forward_expand(body, ggml_cpy(ctx.ggml, hidden.tensor, body_state.tensor));
            ggml_graph_cpy(body, owner.graph());
            const auto full_output = modules::ScaleBiasModule({3.0F, 2.0F}).build(ctx, hidden);
            ggml_build_forward_expand(owner.graph(), ggml_cpy(ctx.ggml, full_output.tensor, full_state.tensor));
            const auto head_output = modules::ScaleBiasModule({-2.0F, 7.0F}).build(ctx, head_input);
            ggml_set_output(head_output.tensor);
            ggml_build_forward_expand(owner.graph(), head_output.tensor);
            core::GraphWorkspace workspace(execution.backend());
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> direct_allocator{nullptr, ggml_gallocr_free};
            core::HostGraphPlan direct_plan, direct_body_plan;
            if (!shared_entry) {
                // Independent reference: Nemotron's existing direct GGML allocation.
                direct_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
                require(direct_allocator && ggml_gallocr_alloc_graph(direct_allocator.get(), owner.graph()),
                    "direct allocation baseline failed");
                core::prepare_host_graph_plan(execution, owner.graph(), direct_plan);
                core::prepare_host_graph_plan(execution, body, direct_body_plan);
            } else if (external) { workspace.add(owner); workspace.prepare(); }
            else owner.allocate();
            const auto bytes = [&]() {
                return direct_allocator ? ggml_gallocr_get_buffer_size(direct_allocator.get(), 0) : owner.workspace_bytes();
            };
            const auto prepared_plans = plans.created;
            require(prepared_plans == (execution.uses_host_graph_plan() ? size_t{2} : size_t{0}),
                "entry points did not receive distinct host plans");
            if (shared_entry) require(owner.allocation_graph(body_id) == 0 && owner.allocation_graph() == 0,
                "body entry point did not select full graph allocation");
            const std::vector<ggml_tensor *> scratch{
                input.tensor, hidden.tensor, full_output.tensor, head_input.tensor, head_output.tensor};
            std::vector<size_t> layout;
            std::vector<void *> addresses;
            for (auto * tensor : scratch) {
                require(tensor->buffer && tensor->data, "allocation graph did not allocate shared transient tensor");
                const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
                const auto data = reinterpret_cast<uintptr_t>(tensor->data);
                require(data >= base, "transient tensor is outside its allocation buffer");
                layout.push_back(static_cast<size_t>(data - base));
                addresses.push_back(tensor->data);
            }
            if (!shared_entry) {
                baseline_bytes = bytes();
                baseline_layout = layout;
            } else {
                require(bytes() == baseline_bytes && layout == baseline_layout,
                    "entry-point preparation changed the full graph allocation size or tensor layout");
            }
            std::vector<float> values(33), expected_body(33), expected_full(33);
            std::vector<float> head_values(64 * 256), expected_head(head_values.size());
            for (int request = 0; request < 3; ++request) {
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<float>(static_cast<int>(i) - request * 3);
                    expected_body[i] = 2 * values[i] + 1;
                    expected_full[i] = 6 * values[i] + 5;
                }
                for (size_t i = 0; i < head_values.size(); ++i) {
                    head_values[i] = static_cast<float>(static_cast<int>(i % 31) + request * 5);
                    expected_head[i] = -2 * head_values[i] + 7;
                }
                // Repeated full/body blocks exercise capture reuse and switching.
                for (int repeat = 0; repeat < 3; ++repeat) {
                    core::write_tensor_f32(input, values);
                    core::write_tensor_f32(head_input, head_values);
                    if (shared_entry) owner.compute();
                    else require(core::compute_graph(execution, owner.graph(), direct_plan, "test.direct_entry") == GGML_STATUS_SUCCESS,
                        "direct full-graph execution failed");
                }
                require(core::read_tensor_f32(body_state.tensor) == expected_body &&
                    core::read_tensor_f32(full_state.tensor) == expected_full &&
                    core::read_tensor_f32(head_output.tensor) == expected_head,
                    "full entry point differs from direct full-graph arithmetic");
                for (int repeat = 0; repeat < 4; ++repeat) {
                    for (size_t i = 0; i < values.size(); ++i) {
                        values[i] = static_cast<float>(static_cast<int>(i) + request * 7 + repeat);
                        expected_body[i] = 2 * values[i] + 1;
                    }
                    core::write_tensor_f32(input, values);
                    if (shared_entry) owner.compute(body_id);
                    else require(core::compute_graph(execution, body, direct_body_plan, "test.direct_body") == GGML_STATUS_SUCCESS,
                        "direct body-graph execution failed");
                    require(core::read_tensor_f32(body_state.tensor) == expected_body &&
                        core::read_tensor_f32(hidden.tensor) == expected_body,
                        "body entry point lost its shared transient/state updates");
                    require(core::read_tensor_f32(full_state.tensor) == expected_full,
                        "body entry point incorrectly executed the full-only head");
                }
                for (size_t i = 0; i < scratch.size(); ++i)
                    require(scratch[i]->data == addresses[i], "entry-point execution changed a tensor address");
                require(plans.created == prepared_plans && bytes() == baseline_bytes,
                    "entry-point execution rebuilt plans or grew workspace");
            }
            // Queue the selected entry and check teardown while both plans and
            // every shared tensor are still live at the synchronization boundary.
            core::write_tensor_f32(input, values);
            core::write_tensor_f32(head_input, head_values);
            require(ggml_backend_graph_compute_async(execution.backend(), body) == GGML_STATUS_SUCCESS,
                "entry-point asynchronous submission failed");
            if (shared_entry) {
                ResetSyncCheck sync(owner);
                if (external) workspace.reset();
                else owner.reset();
                require(sync.calls == 1 && sync.storage_alive, "entry-point reset released shared storage before synchronization");
            } else {
                ggml_backend_synchronize(execution.backend());
                direct_plan.reset();
                direct_body_plan.reset();
                core::release_backend_graph_resources(execution.backend(), body, true);
                core::release_backend_graph_resources(execution.backend(), owner.graph(), true);
                direct_allocator.reset();
                owner.reset();
            }
            require(plans.created == plans.freed && !owner.allocated(), "entry-point reset leaked execution plans");
            require(core::read_tensor_f32(body_state.tensor) == expected_body,
                "entry-point reset discarded the queued state update");
        }
    }
}

void check_entry_point_contracts(core::ExecutionContext & execution) {
    core::PreparedGraph owner(execution, {"test.entry_contracts", 256 * 1024, 128});
    bool rejected = false;
    try { owner.add_entry_point("bad", 99); } catch (const std::out_of_range &) { rejected = true; }
    require(rejected && owner.graph_count() == 1, "unknown allocation graph was accepted");
    const auto body_id = owner.add_entry_point("body", 0);
    rejected = false;
    try { owner.add_entry_point("chained", body_id); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && owner.graph_count() == 2, "entry point accepted another entry point as allocation owner");
    auto & ctx = owner.build_context();
    const auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
    ggml_set_input(input.tensor);
    const auto first = modules::ScaleBiasModule({2.0F, 1.0F}).build(ctx, input);
    const auto missing = modules::ScaleBiasModule({3.0F, -2.0F}).build(ctx, first);
    ggml_set_output(missing.tensor);
    ggml_build_forward_expand(owner.graph(), first.tensor);
    ggml_build_forward_expand(owner.graph(body_id), missing.tensor);
    rejected = false;
    try { owner.allocate(); } catch (const std::logic_error & error) {
        rejected = std::string(error.what()).find("ordered subset") != std::string::npos;
    }
    require(rejected && !owner.allocated(), "entry point with unallocated extra nodes was accepted");
    // Extend the allocation graph to cover the entry point, then prepare again.
    ggml_build_forward_expand(owner.graph(), missing.tensor);
    owner.allocate();
    core::write_tensor_f32(input, std::vector<float>(33, 4));
    owner.compute(body_id);
    require(core::read_tensor_f32(missing.tensor) == std::vector<float>(33, 25),
        "entry-point validation failure poisoned subsequent preparation");
    owner.reset();
    const auto reversed_id = owner.add_entry_point("reversed", 0);
    const auto fresh = core::make_tensor(owner.build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    ggml_set_input(fresh.tensor);
    const auto stage1 = modules::ScaleBiasModule({2.0F, 1.0F}).build(owner.build_context(), fresh);
    const auto stage2 = modules::ScaleBiasModule({3.0F, 2.0F}).build(owner.build_context(), stage1);
    ggml_build_forward_expand(owner.graph(), stage2.tensor);
    require(ggml_graph_n_nodes(owner.graph()) == 2, "reordering test does not cover two dependent operations");
    ggml_graph_add_node(owner.graph(reversed_id), stage2.tensor);
    ggml_graph_add_node(owner.graph(reversed_id), stage1.tensor);
    rejected = false;
    try { owner.allocate(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "entry point with reversed execution order was accepted");
    owner.reset();
    const auto root_id = owner.add_graph("independent");
    const auto entry_id = owner.add_entry_point("independent_entry", root_id);
    const auto small = core::make_tensor(owner.build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({7}));
    const auto large = core::make_tensor(owner.build_context(), GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    ggml_set_input(small.tensor);
    ggml_set_input(large.tensor);
    const auto small_output = modules::ScaleBiasModule({2.0F, 1.0F}).build(owner.build_context(), small);
    const auto large_output = modules::ScaleBiasModule({3.0F, 2.0F}).build(owner.build_context(), large);
    ggml_set_output(small_output.tensor);
    ggml_set_output(large_output.tensor);
    ggml_build_forward_expand(owner.graph(), small_output.tensor);
    ggml_build_forward_expand(owner.graph(root_id), large_output.tensor);
    ggml_graph_cpy(owner.graph(root_id), owner.graph(entry_id));
    require(owner.allocation_graph(entry_id) == root_id && owner.allocation_graph(root_id) == root_id,
        "entry points can only reference graph zero");
    owner.allocate();
    for (int request = 0; request < 4; ++request) {
        core::write_tensor_f32(small, std::vector<float>(7, static_cast<float>(request)));
        owner.compute();
        require(core::read_tensor_f32(small_output.tensor) == std::vector<float>(7, static_cast<float>(2 * request + 1)),
            "independent graph failed beside an entry-point allocation group");
        core::write_tensor_f32(large, std::vector<float>(17, static_cast<float>(request + 2)));
        owner.compute(entry_id);
        require(core::read_tensor_f32(large_output.tensor) == std::vector<float>(17, static_cast<float>(3 * request + 8)),
            "nonzero allocation graph entry point failed");
    }
}

void check_scheduler(core::ExecutionContext & execution) {
    core::ExecutionContext cpu({core::BackendType::Cpu, 0, 8});
    struct Evaluation { ggml_tensor * watched = nullptr; int completed = 0; } evaluation;
    core::PreparedGraph owner(execution, {"test.scheduled", 256 * 1024, 128, 64 * 1024});
    auto & state = owner.persistent_build_context();
    const auto input = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
    const auto saved = core::make_tensor(state, GGML_TYPE_F32, core::TensorShape::from_dims({33}));
    owner.allocate_persistent();
    auto & ctx = owner.build_context();
    const auto first = modules::ScaleBiasModule({2.0F, 1.0F}).build(ctx, input);
    const auto second = modules::ScaleBiasModule({3.0F, -2.0F}).build(ctx, first);
    const auto output = modules::ScaleBiasModule({-1.0F, 3.0F}).build(ctx, second);
    auto * copy = ggml_cpy(ctx.ggml, output.tensor, saved.tensor);
    ggml_build_forward_expand(owner.graph(), copy);
    // CPU-only CI exercises the real single-backend scheduler. GPU tests force
    // a real CPU split; two CPU handles share a buffer type and GGML coalesces them.
    auto * secondary = execution.backend_type() == core::BackendType::Cpu ? execution.backend() : cpu.backend();
    std::vector<ggml_backend_t> backends{execution.backend()};
    if (secondary != execution.backend()) backends.push_back(secondary);
    auto * scheduler = owner.create_scheduler(std::move(backends));
    ggml_backend_sched_set_tensor_backend(scheduler, first.tensor, execution.backend());
    ggml_backend_sched_set_tensor_backend(scheduler, second.tensor, secondary);
    ggml_backend_sched_set_tensor_backend(scheduler, output.tensor, execution.backend());
    ggml_backend_sched_set_tensor_backend(scheduler, copy, execution.backend());
    evaluation.watched = second.tensor;
    ggml_backend_sched_set_eval_callback(scheduler, [](ggml_tensor * node, bool ask, void * data) {
        auto & state = *static_cast<Evaluation *>(data);
        if (ask) return node == state.watched;
        if (node == state.watched) ++state.completed;
        return true;
    }, &evaluation);
    bool rejected = false;
    try { owner.add_graph("late"); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "scheduler owner accepted an independent graph");
    core::GraphWorkspace workspace(execution.backend());
    rejected = false;
    try { workspace.add(owner); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "scheduler allocations were also registered as shared scratch");
    owner.allocate();
    require(owner.allocated() && !owner.has_host_plan(), "scheduler did not own its execution path");
    require(ggml_backend_sched_get_tensor_backend(scheduler, second.tensor) == secondary,
        "scheduler ignored explicit CPU placement");
    const auto capacity = owner.workspace_bytes();
    const auto * input_address = input.tensor->data;
    const auto * output_address = output.tensor->data;
    std::vector<float> values(33), expected(33);
    for (int request = 0; request < 5; ++request) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i) - request * 7);
            expected[i] = -6 * values[i] + 2;
        }
        core::write_tensor_f32(input, values);
        owner.compute();
        require(core::read_tensor_f32(saved.tensor) == expected, "scheduled backend splits changed arithmetic");
        require(evaluation.completed == request + 1, "scheduler did not preserve its evaluation callback");
        require(owner.workspace_bytes() == capacity && input.tensor->data == input_address && output.tensor->data == output_address,
            "scheduler reallocated a prepared graph during execution");
    }
    {
        ResetSyncCheck sync(owner);
        owner.reset();
        require(sync.calls == 1 && sync.storage_alive, "scheduler reset did not drain primary execution before cleanup");
    }
    require(owner.workspace_bytes() == 0 && owner.persistent_bytes() > 0 && input.tensor->data == input_address &&
        core::read_tensor_f32(saved.tensor) == expected, "scheduler reset lost persistent state or retained split allocations");
    const auto ordinary = modules::ScaleBiasModule({4.0F, -2.0F}).build(owner.build_context(), input);
    ggml_set_output(ordinary.tensor);
    ggml_build_forward_expand(owner.graph(), ordinary.tensor);
    owner.allocate();
    owner.compute();
    std::vector<float> ordinary_expected(33);
    for (size_t i = 0; i < values.size(); ++i) ordinary_expected[i] = 4 * values[i] - 2;
    require(core::read_tensor_f32(ordinary.tensor) == ordinary_expected, "scheduler-to-workspace transition failed");
    owner.reset();
    const auto again = modules::ScaleBiasModule({5.0F, 1.0F}).build(owner.build_context(), input);
    ggml_set_output(again.tensor);
    ggml_build_forward_expand(owner.graph(), again.tensor);
    if (execution.backend_type() != core::BackendType::Cpu) {
        rejected = false;
        try { owner.create_scheduler({execution.backend()}); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "scheduler without the required CPU backend was accepted");
    }
    std::vector<ggml_backend_t> next_backends{execution.backend()};
    if (secondary != execution.backend()) next_backends.push_back(secondary);
    owner.create_scheduler(std::move(next_backends));
    require(owner.workspace_bytes() == 0, "allocator mode switch retained an unused second scratch arena");
    owner.allocate();
    owner.compute();
    for (size_t i = 0; i < values.size(); ++i) expected[i] = 5 * values[i] + 1;
    require(core::read_tensor_f32(again.tensor) == expected, "workspace-to-scheduler transition failed");
    {
        ResetSyncCheck sync(owner);
        BufferFreeCheck freed(input.tensor->buffer, owner);
        owner.release();
        require(sync.calls == 1 && sync.storage_alive && freed.freed && freed.graphs_released,
            "scheduler release did not drain and free resources in dependency order");
    }
    owner.release();
    owner.reset();
    owner.add_entry_point("body", 0);
    rejected = false;
    try { owner.create_scheduler({execution.backend()}); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "scheduler accepted entry points with incompatible split allocation lifetimes");
}

} // namespace

int main(int argc, char ** argv) {
    if ((argc != 3 && argc != 5) || std::string(argv[1]) != "--log" ||
        (argc == 5 && std::string(argv[3]) != "--backend")) {
        std::cerr << "Usage: prepared_graph_test --log PATH [--backend cpu|cuda|vulkan]\n";
        return 1;
    }
    debug::LoggingConfig logging;
    logging.enabled = true;
    logging.file_path = argv[2];
    debug::configure_logging(logging);
    const auto started = std::chrono::steady_clock::now();
    try {
        auto type = core::BackendType::Cpu;
        if (argc == 5) {
            const std::string backend = argv[4];
            if (backend == "cuda") type = core::BackendType::Cuda;
            else if (backend == "vulkan") type = core::BackendType::Vulkan;
            else if (backend != "cpu") throw std::runtime_error("Unsupported test backend: " + backend);
        }
        core::ExecutionContext execution({type, 0, 8});
        check_session_moe_option();
        check_owner_allocation_failures();
        for (int lifetime = 0; lifetime < 2; ++lifetime) {
            check_owned_graph(execution);
            check_buffer_release(execution);
            check_workspace(execution);
            check_workspace_sizes(execution);
            check_graph_set(execution);
            check_external_graph_set(execution);
            check_graph_set_contracts(execution);
            check_graph_set_destruction(execution);
            check_entry_points(execution);
            check_entry_point_contracts(execution);
            check_scheduler(execution);
            check_regions(execution);
            check_recorded_regions(execution);
            check_region_optimizer_barriers(execution);
            check_region_contracts(execution);
            check_scheduler_region_contracts(execution);
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "prepared_graph: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
