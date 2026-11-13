#pragma once

#include "engine/framework/core/module.h"

#include <ggml-backend.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::core {

class ExecutionContext;

// Session-owned, bounded native-format expert cache. GGUF expert payloads are
// read by range only after routing, or explicitly pinned in host RAM at load.
// Dense weights remain in BackendWeightStore.
class DiskExpertWeightStore {
public:
    struct Route {
        DiskExpertWeightStore * store;
        size_t group;
    };
    struct Statistics {
        uint64_t bytes_read = 0;
        uint64_t expert_loads = 0;
        uint64_t cache_hits = 0;
        size_t buffer_bytes = 0;
        size_t host_buffer_bytes = 0;
        double read_ms = 0;
        double upload_ms = 0;
        double stage_ms = 0;
    };

    // Borrow the pinned host-bank allocation provider; null uses each group's backend.
    // The provider must remain valid while groups are added.
    explicit DiskExpertWeightStore(const std::filesystem::path & gguf_path, bool pin_experts = false,
                                   size_t read_threads = 1, ExecutionContext * cpu_execution = nullptr,
                                   ggml_backend_t host_buffer_backend = nullptr);
    ~DiskExpertWeightStore();
    DiskExpertWeightStore(const DiskExpertWeightStore &) = delete;
    DiskExpertWeightStore & operator=(const DiskExpertWeightStore &) = delete;

    // Borrowed CPU execution context; its owner must outlive staged runtimes and this store.
    ExecutionContext * cpu_execution() const noexcept;

    // Projections in a group share an expert axis and are evicted together.
    size_t add_group(const std::vector<std::string> & tensor_names,
                     ggml_backend_t backend, int64_t cache_slots);
    TensorValue tensor(size_t group, size_t projection) const;
    // Borrow the backend owning this bounded cache.
    ggml_backend_t group_backend(size_t group) const;
    // Borrow native host weights without copying or expanding the expert bank.
    // The returned tensor and its mapped/pinned storage live with this store.
    TensorValue host_tensor(size_t group, const ggml_tensor * device_tensor);
    // Residency inspection does not change eviction order or choose execution.
    const std::vector<int32_t> & cached_experts(size_t group) const;
    uint64_t cache_epoch() const;
    // Stage one bounded group and return cache-slot IDs in the caller's order.
    std::vector<int32_t> stage(size_t group, const std::vector<int32_t> & expert_ids);
    // Invalidate placement at an independent request boundary, retaining all allocations and statistics.
    void invalidate_cache();
    ggml_tensor * route(ModuleBuildContext & ctx, ggml_tensor * expert_ids, size_t group) const;
    // Graph-local binding; no tensor-name or model-family inference.
    static std::optional<Route> find_route(ModuleBuildContext & ctx, ggml_tensor * bank_ids);

    // Install on a scheduler before computing graphs containing route() nodes.
    // The scheduler synchronizes the routing result before this callback reads
    // it and uploads weights; failures propagate instead of executing stale data.
    static bool evaluate(ggml_tensor * tensor, bool ask, void * store);
    Statistics statistics() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::core
