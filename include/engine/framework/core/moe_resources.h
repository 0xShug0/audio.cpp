#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::core {

class PreparedGraph;

enum class MoeExpertResidency { Backend, Staged };

struct MoeExecutionPolicy {
    MoeExpertResidency residency = MoeExpertResidency::Backend;
    bool pin_host_weights = false;
    int64_t cache_slots = 8;
    int64_t read_threads = 1;
    // MoE ordinals, not backbone indices. Registration supplies their mapping.
    std::vector<size_t> cpu_layers;
    float cpu_share = 0;
};

// Resolve shared session controls and existing family-prefixed aliases using
// model-provided defaults. Supplying both spellings is an explicit error.
MoeExecutionPolicy resolve_moe_execution_policy(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view family, MoeExecutionPolicy defaults, size_t moe_layers);

struct MoeExpertTensorBinding {
    std::string name;       // Logical name in TensorSource.
    std::string disk_name;  // Logical name in the containing GGUF.
    std::vector<int64_t> shape;
    assets::TensorStorageType storage = assets::TensorStorageType::Native;
};

struct MoeExpertLayer {
    size_t ordinal = 0;
    size_t backbone_index = 0;
    bool cpu = false;
    std::vector<TensorValue> weights;
    DiskExpertWeightStore * disk_store = nullptr;
    size_t disk_group = 0;
};

// Session-owned expert resources. Existing primary weights can share the dense
// BackendWeightStore without changing its allocation/upload order. That store
// and the primary execution context must outlive this coordinator; graphs and
// workers borrowing layer handles must be destroyed before it.
class MoeResources {
public:
    MoeResources(ExecutionContext & primary, const assets::TensorSource & source,
        BackendWeightStore & primary_weights, MoeExecutionPolicy policy,
        std::string cpu_weight_label, size_t cpu_weight_context_bytes);
    ~MoeResources();
    MoeResources(const MoeResources &) = delete;
    MoeResources & operator=(const MoeResources &) = delete;

    const MoeExpertLayer & bind_layer(size_t ordinal, size_t backbone_index,
        const std::vector<MoeExpertTensorBinding> & tensors, int64_t selected);
    // Finalize registration and upload auxiliary CPU banks. The caller uploads
    // the shared primary store in its original order before this call.
    void finalize_weights();
    const MoeExpertLayer & layer(size_t ordinal) const;
    const MoeExecutionPolicy & policy() const noexcept;
    DiskExpertWeightStore * disk_store() const noexcept;
    ExecutionContext * cpu_execution() const noexcept;
    bool has_cpu_layers() const noexcept;
    // Install the existing scheduler and disk evaluator only when needed.
    ggml_backend_sched_t configure_scheduler(PreparedGraph & graph) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::core
