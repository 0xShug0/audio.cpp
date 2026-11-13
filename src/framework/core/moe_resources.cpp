#include "engine/framework/core/moe_resources.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::core {

MoeExecutionPolicy resolve_moe_execution_policy(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view family, MoeExecutionPolicy policy, size_t moe_layers) {
    const std::string prefix = std::string(family) + ".";
    const auto alias = [&](std::string_view shared, std::string_view local) {
        const std::string legacy = prefix + std::string(local);
        if (options.count(std::string(shared)) && options.count(legacy))
            throw std::invalid_argument("MoE options contain both " + std::string(shared) + " and " + legacy);
        return legacy;
    };
    const auto disk = alias("moe_disk_experts", "disk_experts");
    if (const auto value = runtime::find_option(options, {"moe_disk_experts", disk}))
        policy.residency = runtime::parse_bool_option(*value, "moe_disk_experts")
            ? MoeExpertResidency::Staged : MoeExpertResidency::Backend;
    const auto pin = alias("moe_pin_experts", "pin_experts");
    if (const auto value = runtime::find_option(options, {"moe_pin_experts", pin}))
        policy.pin_host_weights = runtime::parse_bool_option(*value, "moe_pin_experts");
    const auto cache = alias("moe_expert_cache_size", "expert_cache_size");
    policy.cache_slots = runtime::parse_positive_i64_option(options, {"moe_expert_cache_size", cache}, policy.cache_slots);
    const auto readers = alias("moe_disk_expert_read_threads", "disk_expert_read_threads");
    policy.read_threads = runtime::parse_positive_i64_option(options,
        {"moe_disk_expert_read_threads", readers}, policy.read_threads);
    if (policy.read_threads > 64) throw std::invalid_argument("MoE disk expert read threads must be between 1 and 64");
    const auto cpu = alias("moe_cpu_expert_layers", "cpu_expert_layers");
    if (const auto value = runtime::parse_i64_option(options, {"moe_cpu_expert_layers", cpu})) {
        if (*value < 0 || static_cast<uint64_t>(*value) > moe_layers)
            throw std::invalid_argument("MoE CPU expert layer count exceeds the number of MoE layers");
        policy.cpu_layers.resize(static_cast<size_t>(*value));
        for (size_t i = 0; i < policy.cpu_layers.size(); ++i) policy.cpu_layers[i] = i;
    }
    if (const auto share = runtime::parse_finite_float_option(options, {"moe_cpu_share"})) policy.cpu_share = *share;
    if (!std::isfinite(policy.cpu_share) || policy.cpu_share < 0 || policy.cpu_share > 1)
        throw std::invalid_argument("moe_cpu_share must be between zero and one");
    if (policy.cpu_share > 0 && policy.residency != MoeExpertResidency::Staged)
        throw std::invalid_argument("moe_cpu_share requires staged MoE experts");
    auto layers = policy.cpu_layers;
    std::sort(layers.begin(), layers.end());
    if (std::adjacent_find(layers.begin(), layers.end()) != layers.end() ||
        (!layers.empty() && layers.back() >= moe_layers))
        throw std::invalid_argument("invalid MoE CPU layer ordinals");
    return policy;
}

class MoeResources::Impl {
public:
    ExecutionContext & primary;
    const assets::TensorSource & source;
    BackendWeightStore & primary_weights;
    MoeExecutionPolicy policy;
    std::string cpu_label;
    size_t cpu_context_bytes;
    // Reverse destruction releases CPU weights/caches before their backend.
    std::unique_ptr<ExecutionContext> cpu;
    std::unique_ptr<BackendWeightStore> cpu_weights;
    std::unique_ptr<DiskExpertWeightStore> disk;
    std::vector<std::unique_ptr<MoeExpertLayer>> layers;
    bool finalized = false;

    Impl(ExecutionContext & execution, const assets::TensorSource & tensors,
        BackendWeightStore & weights, MoeExecutionPolicy p, std::string label, size_t context_bytes)
        : primary(execution), source(tensors), primary_weights(weights), policy(std::move(p)),
          cpu_label(std::move(label)), cpu_context_bytes(context_bytes) {
        if (policy.cache_slots <= 0 || policy.read_threads < 1 || policy.read_threads > 64 ||
            !std::isfinite(policy.cpu_share) || policy.cpu_share < 0 || policy.cpu_share > 1)
            throw std::invalid_argument("invalid MoE resource policy");
        if (policy.cpu_share > 0 && policy.residency != MoeExpertResidency::Staged)
            throw std::invalid_argument("moe_cpu_share requires staged MoE experts");
        if (policy.cpu_share != primary.moe_cpu_share())
            throw std::invalid_argument("MoE policy CPU share must match the execution context");
        auto ordinals = policy.cpu_layers;
        std::sort(ordinals.begin(), ordinals.end());
        if (std::adjacent_find(ordinals.begin(), ordinals.end()) != ordinals.end())
            throw std::invalid_argument("duplicate MoE CPU layer ordinal");
        if (!policy.cpu_layers.empty() || policy.cpu_share > 0) {
            if (!is_host_backend(primary.backend())) {
                BackendConfig host;
                host.threads = primary.config().threads;
                cpu = std::make_unique<ExecutionContext>(host);
            }
        }
        if (policy.residency == MoeExpertResidency::Staged)
            disk = std::make_unique<DiskExpertWeightStore>(source.source_path(), policy.pin_host_weights,
                static_cast<size_t>(policy.read_threads), cpu ? cpu.get() :
                    (is_host_backend(primary.backend()) ? &primary : nullptr), primary.backend());
        primary.register_moe_resources();
    }
};

MoeResources::MoeResources(ExecutionContext & primary, const assets::TensorSource & source,
    BackendWeightStore & primary_weights, MoeExecutionPolicy policy, std::string label, size_t context_bytes)
    : impl_(std::make_unique<Impl>(primary, source, primary_weights, std::move(policy), std::move(label), context_bytes)) {}
MoeResources::~MoeResources() = default;

const MoeExpertLayer & MoeResources::bind_layer(size_t ordinal, size_t backbone_index,
    const std::vector<MoeExpertTensorBinding> & tensors, int64_t selected) {
    auto & s = *impl_;
    if (s.finalized) throw std::logic_error("cannot register MoE weights after finalization");
    if (ordinal != s.layers.size() || tensors.empty() || selected <= 0)
        throw std::invalid_argument("MoE layers must be registered in ordinal order with expert tensors");
    const int64_t experts = tensors.front().shape.empty() ? 0 : tensors.front().shape.front();
    if (experts < selected || (s.disk && (s.policy.cache_slots < selected || s.policy.cache_slots > experts)))
        throw std::invalid_argument("MoE expert cache must cover selected experts and not exceed the expert count");
    for (const auto & tensor : tensors) {
        if (tensor.shape.size() != 3 || tensor.shape.front() != experts ||
            s.source.require_metadata(tensor.name).shape != tensor.shape)
            throw std::invalid_argument("MoE expert bank shape mismatch: " + tensor.name);
        if (s.disk && tensor.storage != assets::TensorStorageType::Native)
            throw std::invalid_argument("staged MoE experts require native weight storage");
    }
    auto layer = std::make_unique<MoeExpertLayer>();
    // Reserve publication storage before adding a cache group or loading a
    // bank. No allocating container operation follows successful registration.
    s.layers.reserve(s.layers.size() + 1);
    layer->weights.reserve(tensors.size());
    layer->ordinal = ordinal;
    layer->backbone_index = backbone_index;
    layer->cpu = std::find(s.policy.cpu_layers.begin(), s.policy.cpu_layers.end(), ordinal) != s.policy.cpu_layers.end();
    if (s.disk) {
        std::vector<std::string> names;
        names.reserve(tensors.size());
        for (const auto & tensor : tensors) names.push_back(tensor.disk_name);
        const auto backend = layer->cpu ? cpu_execution()->backend() : s.primary.backend();
        layer->disk_store = s.disk.get();
        layer->disk_group = s.disk->add_group(names, backend, s.policy.cache_slots);
        for (size_t i = 0; i < tensors.size(); ++i) layer->weights.push_back(s.disk->tensor(layer->disk_group, i));
    } else {
        BackendWeightStore * weights = &s.primary_weights;
        if (layer->cpu && s.cpu) {
            if (!s.cpu_weights) s.cpu_weights = std::make_unique<BackendWeightStore>(
                s.cpu->backend(), BackendType::Cpu, s.cpu_label, s.cpu_context_bytes);
            weights = s.cpu_weights.get();
        }
        for (const auto & tensor : tensors)
            layer->weights.push_back(weights->load_tensor(s.source, tensor.name, tensor.storage, tensor.shape));
    }
    s.layers.push_back(std::move(layer));
    return *s.layers.back();
}

void MoeResources::finalize_weights() {
    auto & s = *impl_;
    if (s.finalized) throw std::logic_error("MoE weights already finalized");
    for (auto ordinal : s.policy.cpu_layers)
        if (ordinal >= s.layers.size()) throw std::invalid_argument("CPU expert layer ordinal was not registered");
    if (s.cpu_weights) s.cpu_weights->upload();
    s.finalized = true;
}
const MoeExpertLayer & MoeResources::layer(size_t ordinal) const { return *impl_->layers.at(ordinal); }
const MoeExecutionPolicy & MoeResources::policy() const noexcept { return impl_->policy; }
DiskExpertWeightStore * MoeResources::disk_store() const noexcept { return impl_->disk.get(); }
ExecutionContext * MoeResources::cpu_execution() const noexcept {
    if (impl_->cpu) return impl_->cpu.get();
    return is_host_backend(impl_->primary.backend()) ? &impl_->primary : nullptr;
}
bool MoeResources::has_cpu_layers() const noexcept { return !impl_->policy.cpu_layers.empty(); }
ggml_backend_sched_t MoeResources::configure_scheduler(PreparedGraph & graph) const {
    if (!impl_->disk && !impl_->cpu) return nullptr;
    // GGML schedulers require a CPU backend last. Direct staged GPU execution
    // does not use a scheduler and never needs this routing context.
    if (!impl_->cpu && !is_host_backend(impl_->primary.backend())) {
        BackendConfig host;
        host.threads = impl_->primary.config().threads;
        impl_->cpu = std::make_unique<ExecutionContext>(host);
    }
    std::vector<ggml_backend_t> backends{graph.execution().backend()};
    if (impl_->cpu) backends.push_back(impl_->cpu->backend());
    auto * scheduler = graph.create_scheduler(std::move(backends));
    if (impl_->disk) ggml_backend_sched_set_eval_callback(scheduler, DiskExpertWeightStore::evaluate, impl_->disk.get());
    return scheduler;
}

} // namespace engine::core
