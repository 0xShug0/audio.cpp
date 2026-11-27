#include "engine/framework/modules/text_encoders/t5_encoder_runtime.h"

#include <ggml-alloc.h>
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::modules {

struct T5EncoderRuntime::Impl {
    struct Graph {
        Graph(Impl & owner, int64_t length, int64_t batch)
            : owner(owner), length(length), batch(batch),
              context(ggml_init({owner.config.graph_metadata_bytes, nullptr, true}), ggml_free),
              allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.execution.backend())),
                        ggml_gallocr_free) {
            if (!context || !allocator)
                throw std::runtime_error(owner.config.label + " graph initialization failed");
            core::ModuleBuildContext ctx{context.get(), owner.config.label.c_str(), owner.execution.backend_type()};
            const auto heads = std::visit([](const auto & c) { return c.attention_heads; }, owner.architecture);
            ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch, length}));
            mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, heads, length, length}));
            ggml_set_input(ids.tensor);
            ggml_set_input(mask.tensor);
            core::TensorValue encoded;
            if (const auto * base = std::get_if<T5BaseEncoderConfig>(&owner.architecture)) {
                positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({length, length}));
                ggml_set_input(positions.tensor);
                ggml_set_output(positions.tensor); // Uploaded once, survives allocator reuse.
                encoded = T5BaseEncoderModule(*base).build(ctx, ids, positions, mask,
                    std::get<T5BaseEncoderWeights>(owner.weights->encoder));
            } else {
                positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({length}));
                ggml_set_input(positions.tensor);
                encoded = T5GemmaEncoderModule(std::get<T5GemmaEncoderConfig>(owner.architecture)).build(
                    ctx, ids, positions, mask, std::get<T5GemmaEncoderWeights>(owner.weights->encoder));
            }
            if (owner.config.projection) {
                encoded = LinearModule(*owner.config.projection).build(ctx, encoded, *owner.weights->projection);
                encoded = core::ensure_backend_addressable_layout(ctx, encoded);
            }
            output = encoded;
            ggml_set_output(output.tensor);
            graph = ggml_new_graph_custom(context.get(), owner.config.graph_nodes, false);
            ggml_build_forward_expand(graph, output.tensor);
            if (owner.config.optimize_graph)
                runtime::optimize_graph(*graph, runtime::GraphOptimizationBackend::Gpu);
            if (!ggml_gallocr_reserve(allocator.get(), graph) || !ggml_gallocr_alloc_graph(allocator.get(), graph))
                throw std::runtime_error(owner.config.label + " graph allocation failed");
            debug::timing_log_context_reservation(owner.config.label + ".graph", context.get());
            debug::timing_log_scalar(owner.config.label + ".buffer_mb",
                ggml_gallocr_get_buffer_size(allocator.get(), 0) / (1024.0 * 1024.0));
            if (owner.config.prepare_host_graph_plan)
                core::prepare_host_graph_plan(owner.execution, graph, plan);
            if (const auto * base = std::get_if<T5BaseEncoderConfig>(&owner.architecture)) {
                core::write_tensor_i32(positions, t5_base_relative_position_buckets(length, length,
                    base->relative_attention_num_buckets, base->relative_attention_max_distance));
            }
        }

        ~Graph() {
            plan.reset();
            core::release_backend_graph_resources(owner.execution.backend(), graph, true);
        }

        std::vector<float> run(const std::vector<int32_t> & input, const int64_t * valid_lengths) {
            const auto started = std::chrono::steady_clock::now();
            const bool needs_padding = input.size() != static_cast<size_t>(batch * length);
            if (!needs_padding) {
                core::write_tensor_i32(ids, input);
            } else {
                auto padded = input;
                padded.resize(static_cast<size_t>(batch * length), owner.config.pad_token_id);
                core::write_tensor_i32(ids, padded);
            }
            const auto heads = std::visit([](const auto & c) { return c.attention_heads; }, owner.architecture);
            std::vector<float> attention(static_cast<size_t>(batch * heads * length * length), 0.0F);
            for (int64_t row = 0; row < batch; ++row) {
                if (valid_lengths[row] == length) continue;
                const int64_t offset = row * heads * length * length;
                for (int64_t query = 0; query < heads * length; ++query)
                    std::fill(attention.begin() + offset + query * length + valid_lengths[row],
                              attention.begin() + offset + (query + 1) * length,
                              -std::numeric_limits<float>::max());
            }
            core::write_tensor_f32(mask, attention);
            if (std::holds_alternative<T5GemmaEncoderConfig>(owner.architecture)) {
                std::vector<int32_t> values(static_cast<size_t>(length));
                std::iota(values.begin(), values.end(), 0);
                core::write_tensor_i32(positions, values);
            }
            core::set_backend_threads(owner.execution.backend(), owner.execution.config().threads);
            const auto status = owner.config.prepare_host_graph_plan
                ? core::compute_graph(owner.execution, graph, plan, owner.config.label.c_str())
                : core::compute_backend_graph(owner.execution.backend(), graph, nullptr, owner.config.label.c_str());
            if (status != GGML_STATUS_SUCCESS)
                throw std::runtime_error(owner.config.label + " graph execution failed");
            auto values = core::read_tensor_f32(output.tensor);
            debug::timing_log_scalar(owner.config.label + ".wall_ms", debug::elapsed_ms(started));
            return values;
        }

        Impl & owner;
        int64_t length;
        int64_t batch;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context;
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator;
        ggml_cgraph * graph = nullptr;
        core::HostGraphPlan plan;
        core::TensorValue ids, positions, mask, output;
    };

    Impl(core::ExecutionContext & execution, T5EncoderArchitecture architecture,
         WeightLoader loader, T5EncoderRuntimeConfig config)
        : execution(execution), architecture(std::move(architecture)), loader(std::move(loader)), config(std::move(config)) {
        if (this->config.sequence_padding_multiple <= 0 || this->config.graph_nodes == 0)
            throw std::invalid_argument("T5 encoder requires positive padding multiple and graph capacity");
        if (this->config.retain_graphs && !this->config.retain_weights)
            throw std::invalid_argument("T5 encoder cannot retain graphs without their weights");
        if (this->config.graph_metadata_bytes == 0)
            this->config.graph_metadata_bytes = this->config.graph_nodes * ggml_tensor_overhead()
                + ggml_graph_overhead_custom(this->config.graph_nodes, false);
        load_weights();
    }

    void load_weights() {
        if (weights) return;
        auto loaded = loader();
        if (!loaded.storage || loaded.encoder.index() != architecture.index() ||
            loaded.projection.has_value() != config.projection.has_value())
            throw std::invalid_argument("T5 encoder weight bindings do not match its configuration");
        weights = std::make_unique<T5EncoderRuntimeWeights>(std::move(loaded));
    }

    int64_t sequence_capacity(int64_t tokens) const {
        // Singleton transposed attention views are not valid GGML matmul
        // operands. Mask one extra position and trim it from returned rows.
        tokens = (std::max)(tokens, int64_t{2});
        const auto multiple = config.sequence_padding_multiple;
        return (tokens + multiple - 1) / multiple * multiple;
    }

    void prepare(int64_t length, int64_t batch) {
        load_weights();
        if (!graph || graph->length != length || graph->batch != batch) {
            graph.reset();
            graph = std::make_unique<Graph>(*this, length, batch);
        }
    }

    int64_t output_width() const {
        return config.projection ? config.projection->out_features
            : std::visit([](const auto & c) { return c.hidden_size; }, architecture);
    }

    core::ExecutionContext & execution;
    T5EncoderArchitecture architecture;
    WeightLoader loader;
    T5EncoderRuntimeConfig config;
    std::unique_ptr<T5EncoderRuntimeWeights> weights;
    std::unique_ptr<Graph> graph; // Destroy before weight storage.
};

T5EncoderRuntime::T5EncoderRuntime(core::ExecutionContext & execution, T5EncoderArchitecture architecture,
                                 WeightLoader loader, T5EncoderRuntimeConfig config)
    : impl_(std::make_unique<Impl>(execution, std::move(architecture), std::move(loader), std::move(config))) {}
T5EncoderRuntime::~T5EncoderRuntime() = default;

std::vector<float> T5EncoderRuntime::encode(const std::vector<int32_t> & token_ids) {
    if (token_ids.empty()) throw std::invalid_argument("T5 encoder requires non-empty input");
    auto & state = *impl_;
    const auto valid = static_cast<int64_t>(token_ids.size());
    state.prepare(state.sequence_capacity(valid), 1);
    auto output = state.graph->run(token_ids, &valid);
    output.resize(token_ids.size() * static_cast<size_t>(state.output_width()));
    if (!state.config.retain_graphs) release_graphs();
    if (!state.config.retain_weights) release_weights();
    return output;
}

std::vector<std::vector<float>> T5EncoderRuntime::encode_batch(
    const std::vector<std::vector<int32_t>> & token_ids,
    const std::vector<int64_t> & valid_lengths) {
    auto & state = *impl_;
    const auto actual = static_cast<int64_t>(token_ids.size());
    if (actual == 0 || (!valid_lengths.empty() && valid_lengths.size() != token_ids.size()))
        throw std::invalid_argument("T5 encoder batch/valid-length count mismatch");
    const auto batch = actual;
    int64_t longest = 0;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        const auto valid = valid_lengths.empty() ? static_cast<int64_t>(token_ids[i].size()) : valid_lengths[i];
        if (valid < 0 || valid > static_cast<int64_t>(token_ids[i].size()))
            throw std::invalid_argument("T5 encoder invalid token count");
        longest = std::max(longest, static_cast<int64_t>(token_ids[i].size()));
    }
    if (longest == 0) {
        if (!state.config.retain_graphs) release_graphs();
        if (!state.config.retain_weights) release_weights();
        return std::vector<std::vector<float>>(token_ids.size());
    }
    const auto length = state.sequence_capacity(longest);
    std::vector<int32_t> padded(static_cast<size_t>(batch * length), state.config.pad_token_id);
    std::vector<int64_t> valid(static_cast<size_t>(batch));
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::copy(token_ids[i].begin(), token_ids[i].end(), padded.begin() + i * length);
        valid[i] = valid_lengths.empty() ? static_cast<int64_t>(token_ids[i].size()) : valid_lengths[i];
    }
    state.prepare(length, batch);
    auto values = state.graph->run(padded, valid.data());
    const auto width = state.output_width();
    std::vector<std::vector<float>> output;
    output.reserve(token_ids.size());
    for (int64_t i = 0; i < actual; ++i) {
        const auto begin = values.begin() + i * length * width;
        output.emplace_back(begin, begin + valid[static_cast<size_t>(i)] * width);
    }
    if (!state.config.retain_graphs) release_graphs();
    if (!state.config.retain_weights) release_weights();
    return output;
}

void T5EncoderRuntime::release_graphs() { impl_->graph.reset(); }
void T5EncoderRuntime::release_weights() {
    release_graphs();
    impl_->weights.reset();
}

} // namespace engine::modules
