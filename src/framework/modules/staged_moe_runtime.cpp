#include "engine/framework/modules/staged_moe_runtime.h"

#include "engine/framework/core/backend.h"
#include <ggml-alloc.h>

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace engine::modules {

class StagedMoeRuntime::Impl {
public:
    struct Layer {
        core::DiskExpertWeightStore * store;
        size_t group;
        std::vector<ggml_cgraph *> graphs;
    };
    core::ExecutionContext & execution;
    StagedMoeConfig config;
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

    Impl(core::ExecutionContext & e, StagedMoeConfig c, std::shared_ptr<core::GraphWorkspace> shared_workspace)
        : execution(e), config(std::move(c)), workspace(std::move(shared_workspace)) {
        if (!workspace) throw std::invalid_argument("staged MoE requires a caller-owned graph workspace");
        if (config.hidden_size <= 0 || config.tokens <= 0 || config.selected <= 0 || config.experts < config.selected ||
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
        int64_t slot_rows = 0;
        for (auto count : config.rows_per_expert) slot_rows += config.cache_slots * count;
        inputs.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        context.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!inputs || !context) throw std::runtime_error("cannot allocate staged MoE contexts");
        auto * g = inputs.get();
        hidden = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, config.tokens);
        ids = ggml_new_tensor_2d(g, GGML_TYPE_I32, config.selected, config.tokens);
        weights = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.selected, config.tokens);
        outputs = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, config.selected * config.tokens + rows);
        source = ggml_new_tensor_1d(g, GGML_TYPE_I32, rows);
        target = ggml_new_tensor_1d(g, config.scatter_index_type, rows);
        slots = ggml_new_tensor_2d(g, GGML_TYPE_I32, 1, slot_rows);
        result = ggml_view_3d(g, outputs, config.hidden_size, config.selected, config.tokens,
            config.hidden_size * sizeof(float), config.hidden_size * config.selected * sizeof(float), 0);
        for (auto * t : {hidden, ids, weights, outputs, source, target, slots}) ggml_set_input(t);
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
    }

    ~Impl() {
        for (auto & layer : layers)
            for (auto * graph : layer.graphs) core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

StagedMoeRuntime::StagedMoeRuntime(core::ExecutionContext & execution, StagedMoeConfig config,
    std::shared_ptr<core::GraphWorkspace> workspace)
    : impl_(std::make_unique<Impl>(execution, std::move(config), std::move(workspace))) {}
StagedMoeRuntime::~StagedMoeRuntime() = default;

size_t StagedMoeRuntime::add_layer(const MoeModule & module, const MoeExpertWeights & weights,
    core::DiskExpertWeightStore & store, size_t group) {
    auto & s = *impl_;
    if (s.prepared) throw std::logic_error("cannot add staged MoE layers after preparation");
    if (weights.up->ne[2] != s.config.cache_slots || weights.down->ne[1] != s.config.hidden_size)
        throw std::invalid_argument("staged MoE expert bank does not match cache configuration");
    Impl::Layer layer{&store, group, {}};
    core::ModuleBuildContext ctx{s.context.get(), "framework.moe.staged", s.execution.backend_type()};
    size_t slot_offset = 0;
    for (auto count : s.config.rows_per_expert) {
        const auto rows = count * s.config.cache_slots;
        auto * graph = ggml_new_graph_custom(ctx.ggml, 128, false);
        auto * source = ggml_view_1d(ctx.ggml, s.source, rows, 0);
        auto * target = ggml_view_1d(ctx.ggml, s.target, rows, 0);
        auto * slots = ggml_view_2d(ctx.ggml, s.slots, 1, rows, s.slots->nb[1], slot_offset);
        slot_offset += static_cast<size_t>(rows) * sizeof(int32_t);
        auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, s.hidden, source), s.config.hidden_size, 1, rows);
        auto * output = module.build_experts(ctx, input, slots, weights);
        ggml_build_forward_expand(graph, ggml_set_rows(ctx.ggml, s.outputs,
            ggml_reshape_2d(ctx.ggml, output, s.config.hidden_size, rows), target));
        core::validate_backend_graph_supported(s.execution.backend(), graph, "staged MoE experts");
        layer.graphs.push_back(graph);
        s.workspace->add(graph);
    }
    s.layers.push_back(std::move(layer));
    return s.layers.size() - 1;
}

void StagedMoeRuntime::prepare() {
    auto & s = *impl_;
    if (s.prepared || s.layers.empty()) throw std::logic_error("staged MoE preparation requires unprepared layers");
    for (auto & entry : s.layers) for (auto * graph : entry.graphs)
        core::prepare_host_graph_plan(s.execution, graph, s.plans[graph]);
    s.prepared = true;
}

void StagedMoeRuntime::build_inputs(core::ModuleBuildContext & ctx, ggml_cgraph * graph,
    ggml_tensor * hidden, const MoeRouting & routing) const {
    auto & s = *impl_;
    if (!ctx.elide_identity_reshapes || !ggml_are_same_shape(hidden, s.hidden))
        hidden = ggml_reshape_2d(ctx.ggml, hidden, s.config.hidden_size, s.config.tokens);
    // Materialize mixing weights before exposing IDs, preserving the existing
    // routing graph order and backend top-k fusion opportunities.
    for (const auto & pair : {std::make_pair(hidden, s.hidden), std::make_pair(routing.weights, s.weights), std::make_pair(routing.ids, s.ids)})
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, pair.first, pair.second));
}

ggml_tensor * StagedMoeRuntime::hidden_input() const { return impl_->hidden; }
ggml_tensor * StagedMoeRuntime::output() const { return impl_->result; }
ggml_tensor * StagedMoeRuntime::routing_weights() const { return impl_->weights; }
size_t StagedMoeRuntime::workspace_bytes() const { return impl_->workspace->bytes(); }

uint64_t StagedMoeRuntime::run(size_t layer) {
    auto & s = *impl_;
    if (!s.prepared) throw std::logic_error("staged MoE graphs must be prepared before execution");
    auto & binding = s.layers.at(layer);
    ggml_backend_tensor_get(s.ids, s.logical_ids.data(), 0, s.logical_ids.size() * sizeof(int32_t));
    for (auto & pairs : s.assignments) pairs.clear();
    for (size_t pair = 0; pair < s.logical_ids.size(); ++pair)
        s.assignments.at(static_cast<size_t>(s.logical_ids[pair])).push_back(static_cast<int32_t>(pair));
    std::vector<int32_t> experts;
    for (size_t expert = 0; expert < s.assignments.size(); ++expert)
        if (!s.assignments[expert].empty()) experts.push_back(static_cast<int32_t>(expert));
    std::stable_sort(experts.begin(), experts.end(), [&](int32_t a, int32_t b) { return s.assignments[a].size() > s.assignments[b].size(); });
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

} // namespace engine::modules
