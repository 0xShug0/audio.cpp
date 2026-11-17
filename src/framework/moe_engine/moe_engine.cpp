#include "engine/framework/moe_engine/moe_engine.h"
#include "engine/framework/moe_engine/staged_moe_runtime.h"
#include "engine/framework/core/backend.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>

namespace engine::modules {

using namespace core;

class MoeEngine::Impl {
public:
    ExecutionContext & primary;
    const assets::TensorSource & source;
    BackendWeightStore & primary_weights;
    MoeExecutionPolicy policy;
    core::MoePlacementFactory placement;
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
        if (policy.cache_slots <= 0 || policy.read_threads < 1 || policy.read_threads > 64)
            throw std::invalid_argument("invalid MoE resource policy");
        if (primary.moe_policy().active() && policy.residency != MoeExpertResidency::Staged)
            throw std::invalid_argument("selected moe_policy requires staged MoE experts");
        if (primary.moe_policy().placement_requirements().cpu && is_host_backend(primary.backend()))
            throw std::invalid_argument("MoE placement policy requires a GPU primary backend");
        auto ordinals = policy.cpu_layers;
        std::sort(ordinals.begin(), ordinals.end());
        if (std::adjacent_find(ordinals.begin(), ordinals.end()) != ordinals.end())
            throw std::invalid_argument("duplicate MoE CPU layer ordinal");
        if (!policy.cpu_layers.empty() || primary.moe_policy().placement_requirements().cpu) {
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
        if (primary.moe_policy().has_placement()) placement = primary.moe_policy().placement.create();
        if (disk && primary.moe_policy().cache) disk->configure_cache(primary.moe_policy().cache);
        primary.register_moe_resources();
    }
};

MoeEngine::MoeEngine(ExecutionContext & primary, const assets::TensorSource & source,
    BackendWeightStore & primary_weights, MoeExecutionPolicy policy, std::string label, size_t context_bytes)
    : impl_(std::make_unique<Impl>(primary, source, primary_weights, std::move(policy), std::move(label), context_bytes)) {
    if (impl_->disk) impl_->disk->engine_ = this;
}
MoeEngine::~MoeEngine() = default;

const MoeExpertLayer & MoeEngine::bind_layer(size_t ordinal, size_t backbone_index,
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
    std::optional<size_t> up, gate, down;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto & projection = tensors[i].role == MoeExpertProjection::Up ? up :
            tensors[i].role == MoeExpertProjection::Gate ? gate : down;
        if (projection) throw std::invalid_argument("duplicate MoE expert projection");
        projection = i;
    }
    if (!up || !down)
        throw std::invalid_argument("missing MoE expert projection");
    auto layer = std::make_unique<MoeExpertLayer>();
    // Reserve publication storage before adding a cache group or loading a
    // bank. No allocating container operation follows successful registration.
    s.layers.reserve(s.layers.size() + 1);
    layer->weights.reserve(tensors.size());
    layer->ordinal = ordinal;
    layer->backbone_index = backbone_index;
    layer->experts = experts;
    layer->selected = selected;
    layer->cpu = std::find(s.policy.cpu_layers.begin(), s.policy.cpu_layers.end(), ordinal) != s.policy.cpu_layers.end();
    if (!layer->cpu && s.placement) layer->placement = s.placement(static_cast<size_t>(experts));
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
    layer->projections = {layer->weights[*up].tensor,
        gate ? layer->weights[*gate].tensor : nullptr, layer->weights[*down].tensor};
    s.layers.push_back(std::move(layer));
    return *s.layers.back();
}

void MoeEngine::finalize_weights() {
    auto & s = *impl_;
    if (s.finalized) throw std::logic_error("MoE weights already finalized");
    for (auto ordinal : s.policy.cpu_layers)
        if (ordinal >= s.layers.size()) throw std::invalid_argument("CPU expert layer ordinal was not registered");
    if (s.cpu_weights) s.cpu_weights->upload();
    s.finalized = true;
}
const MoeExpertLayer & MoeEngine::layer(size_t ordinal) const { return *impl_->layers.at(ordinal); }
const MoeExpertLayer & MoeEngine::layer_for_group(size_t group) const {
    const auto found = std::find_if(impl_->layers.begin(), impl_->layers.end(), [group](const auto & layer) {
        return layer->disk_store && layer->disk_group == group;
    });
    if (found == impl_->layers.end()) throw std::out_of_range("unregistered MoE disk group");
    return **found;
}
const MoeExecutionPolicy & MoeEngine::policy() const noexcept { return impl_->policy; }
DiskExpertWeightStore * MoeEngine::disk_store() const noexcept { return impl_->disk.get(); }
ExecutionContext * MoeEngine::cpu_execution() const noexcept {
    if (impl_->cpu) return impl_->cpu.get();
    return is_host_backend(impl_->primary.backend()) ? &impl_->primary : nullptr;
}
bool MoeEngine::has_cpu_layers() const noexcept { return !impl_->policy.cpu_layers.empty(); }
ggml_backend_sched_t MoeEngine::configure_scheduler(PreparedGraph & graph) const {
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

std::unique_ptr<MoeExecutionPlan> MoeEngine::create_plan(std::shared_ptr<core::GraphWorkspace> workspace) const {
    if (!impl_->finalized) throw std::logic_error("cannot plan MoE execution before weight finalization");
    return std::unique_ptr<MoeExecutionPlan>(new MoeExecutionPlan(*this, impl_->primary, std::move(workspace), cpu_execution()));
}

class MoeExecutionPlan::Impl {
public:
    struct Shape {
        StagedMoeConfig config;
        std::unique_ptr<StagedMoeRuntime> runtime;
    };
    struct Layer { StagedMoeRuntime * runtime; size_t index; };
    const MoeEngine & engine;
    const core::ExecutionContext & execution;
    std::shared_ptr<core::GraphWorkspace> workspace;
    core::ExecutionContext * cpu;
    std::vector<Shape> shapes;
    std::vector<Layer> layers;
    enum class State { Building, Preparing, Prepared, Failed };
    State state = State::Building;

    Impl(const MoeEngine & engine, const core::ExecutionContext & execution,
        std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu)
        : engine(engine), execution(execution), workspace(std::move(workspace)), cpu(cpu) {}

    static bool compatible(const StagedMoeConfig & a, const StagedMoeConfig & b) {
        return a.hidden_size == b.hidden_size && a.input_size == b.input_size && a.tokens == b.tokens &&
            a.selected == b.selected && a.experts == b.experts && a.cache_slots == b.cache_slots &&
            a.rows_per_expert == b.rows_per_expert && a.scatter_index_type == b.scatter_index_type;
    }
};

MoeExecutionPlan::MoeExecutionPlan(const MoeEngine & engine, const core::ExecutionContext & execution,
    std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu)
    : impl_(std::make_unique<Impl>(engine, execution, std::move(workspace), cpu)) {}
MoeExecutionPlan::~MoeExecutionPlan() = default;

MoeExecutionPlan::LayerId MoeExecutionPlan::add_layer(size_t ordinal, const MoeBatchConfig & batch,
    const MoeExpertConfig & math) {
    auto & s = *impl_;
    if (s.state != Impl::State::Building) throw std::logic_error("cannot bind experts after preparing MoE execution");
    const auto & layer = s.engine.layer(ordinal);
    if (!layer.disk_store) throw std::invalid_argument("bounded MoE execution requires a staged layer");
    StagedMoeConfig config;
    config.hidden_size = batch.hidden_size;
    config.tokens = batch.tokens;
    config.selected = layer.selected;
    config.experts = layer.experts;
    config.cache_slots = layer.weights.front().tensor->ne[2];
    config.rows_per_expert = batch.rows_per_expert;
    config.scatter_index_type = batch.scatter_index_type;
    config.input_size = batch.input_size;
    s.layers.reserve(s.layers.size() + 1);
    auto shape = std::find_if(s.shapes.begin(), s.shapes.end(), [&](const Impl::Shape & item) {
        return Impl::compatible(item.config, config);
    });
    if (shape == s.shapes.end()) {
        s.shapes.push_back({config, std::make_unique<StagedMoeRuntime>(s.execution, config, s.workspace, s.cpu)});
        shape = std::prev(s.shapes.end());
    }
    const auto index = shape->runtime->add_layer(MoeModule(math), layer.projections, *layer.disk_store, layer.disk_group,
        layer.placement ? layer.placement(config, math) : nullptr);
    s.layers.push_back({shape->runtime.get(), index});
    return s.layers.size() - 1;
}
StagedMoeRuntime & MoeExecutionPlan::runtime(LayerId layer) const { return *impl_->layers.at(layer).runtime; }
size_t MoeExecutionPlan::runtime_layer(LayerId layer) const { return impl_->layers.at(layer).index; }

void MoeExecutionPlan::build_inputs(LayerId layer, core::ModuleBuildContext & ctx, ggml_cgraph * graph,
    ggml_tensor * hidden, const MoeRouting & routing) const {
    impl_->layers.at(layer).runtime->build_inputs(ctx, graph, hidden, routing);
}
ggml_tensor * MoeExecutionPlan::hidden_input(LayerId layer) const { return impl_->layers.at(layer).runtime->hidden_input(); }
ggml_tensor * MoeExecutionPlan::output(LayerId layer) const { return impl_->layers.at(layer).runtime->output(); }
ggml_tensor * MoeExecutionPlan::routing_weights(LayerId layer) const { return impl_->layers.at(layer).runtime->routing_weights(); }

void MoeExecutionPlan::prepare() {
    auto & s = *impl_;
    if (s.state != Impl::State::Building) throw std::logic_error("MoE execution cannot be prepared again; rebuild graph owner and workspace");
    s.state = Impl::State::Preparing;
    try {
        for (auto & shape : s.shapes) shape.runtime->prepare();
        s.state = Impl::State::Prepared;
    } catch (...) {
        // Invalidate this entire graph generation while graph registrations
        // still have live owners. The caller must rebuild the owning bundle.
        s.state = Impl::State::Failed;
        s.workspace->reset();
        throw;
    }
}
uint64_t MoeExecutionPlan::run(LayerId layer) {
    auto & s = *impl_;
    if (s.state != Impl::State::Prepared) throw std::logic_error("MoE execution was not prepared; rebuild graph owner and workspace after failure");
    const auto & binding = s.layers.at(layer);
    return binding.runtime->run(binding.index);
}

} // namespace engine::modules
