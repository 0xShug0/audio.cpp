#include "engine/framework/modules/transformers/dit_runtime.h"

#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {
using core::TensorShape;
using core::TensorValue;
constexpr float kPi = 3.14159265358979323846f;

} // namespace

class DitRuntime::Impl {
public:
    core::ExecutionContext & execution;
    DitRuntimeConfig config;
    core::BackendWeightStore store;
    DitModule module;
    DitWeights weights;
    DitConditioning conditions;
    std::vector<float> frequencies;
    std::unique_ptr<core::ConstantTensorCache> context_store;
    std::unique_ptr<core::PreparedGraph> context_graph;
    TensorValue context_input;
    std::vector<DitContextKV> context_kv;
    bool prepared = false;
    struct State {
        int64_t branches = 0, condition_branch = 0;
        std::unique_ptr<core::ConstantTensorCache> constants;
        std::unique_ptr<DeviceDenoiserRuntime> device;
        bool guided = false;
        DitSampler sampler = DitSampler::Euler;
        TensorValue latent, time, step, guidance, apg, sigma, branch_indices, latent_mask, noise, next_time;
        DitForwardInputs inputs;
        TensorValue output;
    };
    std::unique_ptr<State> state;
    struct ModulationState {
        std::unique_ptr<core::PreparedGraph> graph;
        TensorValue time, duration, global, output;
        int64_t rows = 0, global_rows = 0;
        bool has_duration = false;
        std::vector<float> times, durations, global_values;
    };
    struct SequenceState {
        std::unique_ptr<core::PreparedGraph> graph;
        TensorValue tokens, time, duration, global, positions, mask, indices, output, guidance;
        bool guided = false;
        int64_t batch = 0, frames = 0, output_start = 0, output_tokens = 0;
        bool precomputed = false, has_mask = false, has_duration = false;
    };
    struct PrefixInputs {
        TensorValue tokens, positions, mask, slots, indices, guidance;
    };
    struct PrefixState {
        DitPrefixCacheConfig config;
        std::unique_ptr<core::PreparedGraph> graph;
        std::vector<DitContextKV> cache;
        std::vector<int64_t> valid;
        int64_t prefill_frames = 0;
        PrefixInputs prefill, step;
        std::vector<core::PreparedGraph::GraphId> steps;
        std::vector<TensorValue> outputs;
        std::vector<int32_t> positions, slots, indices;
        std::vector<ggml_fp16_t> mask;
        int64_t last_past = -1;
        float last_guidance = std::numeric_limits<float>::quiet_NaN();
    };
    std::unique_ptr<core::ConstantTensorCache> frequency_cache;
    TensorValue graph_frequencies;
    std::unique_ptr<ModulationState> modulation;
    std::unique_ptr<SequenceState> sequence;
    std::unique_ptr<PrefixState> prefix;

    Impl(core::ExecutionContext & execution_, const assets::TensorSource & source, DitRuntimeConfig config_, const DitWeightNames & names)
        : execution(execution_), config(std::move(config_)),
          store(execution.backend(), execution.backend_type(), config.label + ".weights", config.weight_metadata_bytes),
          module(config.model), weights(load_dit_weights(store, source, config.model, names, config.weight_type, execution.backend_type())) {
        const auto half = config.model.time_features / 2;
        frequencies.resize(static_cast<size_t>(half));
        const float denominator = static_cast<float>(config.time_frequency_denominator > 0
            ? config.time_frequency_denominator : half);
        for (int64_t i = 0; i < half; ++i) {
            if (config.time_features == DitTimeFeatures::Fourier) {
                const float ramp = static_cast<float>(i) / static_cast<float>(half - 1);
                frequencies[static_cast<size_t>(i)] = std::exp(ramp * (std::log(config.time_frequency_max) -
                    std::log(config.time_frequency_min)) + std::log(config.time_frequency_min)) * 2.f * kPi;
            } else {
                frequencies[static_cast<size_t>(i)] = std::exp(-std::log(static_cast<float>(config.time_period)) *
                    static_cast<float>(i) / denominator);
            }
        }
        if (config.modulation_storage_type != GGML_TYPE_F32 && config.modulation_storage_type != GGML_TYPE_F16 &&
            config.modulation_storage_type != GGML_TYPE_BF16)
            throw std::invalid_argument("DiT modulation storage must be F32, F16 or BF16");
        store.upload();
    }

    std::vector<float> time_features(const std::vector<float> & timesteps, int64_t rows) const {
        const auto width = config.model.time_features, half = width / 2;
        std::vector<float> out(static_cast<size_t>(rows * width));
        for (int64_t b = 0; b < rows; ++b) {
            const float t = timesteps.at(static_cast<size_t>(b % conditions.batch));
            for (int64_t i = 0; i < half; ++i) {
                const float angle = config.time_scale * t * frequencies[static_cast<size_t>(i)];
                const float cosine = std::cos(angle), sine = std::sin(angle);
                const auto at = static_cast<size_t>(b * width + i);
                out[at] = config.time_features == DitTimeFeatures::SinCos ? sine : cosine;
                out[at + static_cast<size_t>(half)] = config.time_features == DitTimeFeatures::SinCos ? cosine : sine;
            }
        }
        return out;
    }

    void prepare_context() {
        if (!config.model.context_dim || config.conditioning_cache != DitConditioningCache::ProjectedKV) return;
        const auto rows = conditions.batch * conditions.branches;
        const auto & c = config.model;
        if (!context_graph) {
            context_store = std::make_unique<core::ConstantTensorCache>(execution.backend(), execution.config().threads, config.label + ".context.cache", 1024 * 1024);
            context_graph = std::make_unique<core::PreparedGraph>(execution,
                core::PreparedGraphConfig{config.label + ".condition", config.graph_metadata_bytes, config.graph_nodes});
            auto & ctx = context_graph->build_context();
            ctx.elide_identity_reshapes = config.model.projection_type == DitProjectionType::Bf16Autocast;
            context_input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({rows, conditions.context_tokens, c.context_dim}));
            ggml_set_input(context_input.tensor);
            const auto outputs = module.build_context(ctx, context_input, weights);
            for (const auto & output : outputs) {
                DitContextKV cache;
                for (auto pair : {std::pair<TensorValue, TensorValue *>{output.key, &cache.key}, {output.value, &cache.value}, {output.differential_key, &cache.differential_key}}) {
                    auto value = pair.first;
                    if (!value.valid()) continue;
                    const auto type = c.projection_type == DitProjectionType::Bf16Autocast &&
                        c.cross_attention.lowering != ScaledDotProductAttentionLowering::Explicit ? GGML_TYPE_F16 : GGML_TYPE_F32;
                    const std::vector<uint8_t> zero(static_cast<size_t>(value.shape.num_elements()) * ggml_type_size(type), 0);
                    *pair.second = context_store->make_tensor(value.shape, type, zero.data(), zero.size());
                    ggml_build_forward_expand(context_graph->graph(), ggml_cpy(ctx.ggml, value.tensor, pair.second->tensor));
                }
                context_kv.push_back(cache);
            }
            context_store->ensure_uploaded();
        }
        context_graph->allocate_releasable();
        core::write_tensor_f32(context_input, conditions.context);
        context_graph->compute();
        context_graph->release_buffers();
    }

    void prepare(DitConditioning in) {
        const auto & c = config.model;
        if (in.frames <= 0 || in.batch <= 0 || in.branches <= 0 ||
            in.frame_features.size() != in.frame_channels.size() || in.vector_features.size() != in.vector_channels.size())
            throw std::invalid_argument(config.label + " invalid conditioning dimensions");
        const auto rows = in.batch * in.branches;
        int64_t channels = c.output_channels;
        for (size_t i = 0; i < in.frame_channels.size(); ++i) {
            if (in.frame_channels[i] <= 0 || in.frame_features[i].size() != static_cast<size_t>(rows * in.frames * in.frame_channels[i]))
                throw std::invalid_argument(config.label + " frame conditioning shape mismatch");
            channels += in.frame_channels[i];
        }
        for (size_t i = 0; i < in.vector_channels.size(); ++i) {
            if (in.vector_channels[i] <= 0 || in.vector_features[i].size() != static_cast<size_t>(rows * in.vector_channels[i]))
                throw std::invalid_argument(config.label + " vector conditioning shape mismatch");
            channels += in.vector_channels[i];
        }
        if (c.local_condition_dim && in.local_condition.size() != static_cast<size_t>(rows * in.frames * c.local_condition_dim))
            throw std::invalid_argument(config.label + " local conditioning shape mismatch");
        if (channels != c.input_channels || (c.context_dim &&
            (in.context_tokens <= 0 || in.context.size() != static_cast<size_t>(rows * in.context_tokens * c.context_dim))) ||
            (c.global_dim && in.global.size() != static_cast<size_t>(rows * c.global_dim)))
            throw std::invalid_argument(config.label + " conditioning feature shape mismatch");
        if ((!in.value_mask.empty() && in.value_mask.size() != static_cast<size_t>(rows * (in.frames + c.memory_tokens))) ||
            (!in.latent_mask.empty() && in.latent_mask.size() != static_cast<size_t>(in.batch * in.frames)) ||
            (config.guidance == DitGuidance::AdaptiveProjected && in.branches > 1 && in.latent_mask.empty()))
            throw std::invalid_argument(config.label + " conditioning mask shape mismatch");
        const bool changed = in.value_mask.empty() != conditions.value_mask.empty() || in.frames != conditions.frames || in.batch != conditions.batch || in.branches != conditions.branches ||
            in.context_tokens != conditions.context_tokens || in.frame_channels != conditions.frame_channels || in.vector_channels != conditions.vector_channels;
        prepared = false;
        if (changed) { state.reset(); context_graph.reset(); context_kv.clear(); context_store.reset(); }
        conditions = std::move(in);
        prepare_context();
        if (state) upload_conditions();
        prepared = true;
    }

    void upload_conditions(int64_t branch = 0) {
        if (!state) return;
        const auto count = state->branches * conditions.batch;
        for (size_t i = 0; i < state->inputs.frame_conditions.size(); ++i) {
            const auto n = static_cast<size_t>(count * conditions.frames * conditions.frame_channels[i]);
            core::write_tensor_f32_slice(state->inputs.frame_conditions[i], 0, conditions.frame_features[i].data() + branch * conditions.batch * conditions.frames * conditions.frame_channels[i], n);
        }
        for (size_t i = 0; i < state->inputs.vector_conditions.size(); ++i) {
            const auto n = static_cast<size_t>(count * conditions.vector_channels[i]);
            core::write_tensor_f32_slice(state->inputs.vector_conditions[i], 0, conditions.vector_features[i].data() + branch * conditions.batch * conditions.vector_channels[i], n);
        }
        if (state->latent_mask.valid()) core::write_tensor_f32(state->latent_mask, conditions.latent_mask);
        if (state->inputs.context.valid()) core::write_tensor_f32_slice(state->inputs.context, 0,
            conditions.context.data() + branch * conditions.batch * conditions.context_tokens * config.model.context_dim, static_cast<size_t>(count * conditions.context_tokens * config.model.context_dim));
        if (state->inputs.local_condition.valid()) core::write_tensor_f32_slice(state->inputs.local_condition, 0,
            conditions.local_condition.data() + branch * conditions.batch * conditions.frames * config.model.local_condition_dim, static_cast<size_t>(count * conditions.frames * config.model.local_condition_dim));
        if (state->inputs.global.valid()) core::write_tensor_f32_slice(state->inputs.global, 0, conditions.global.data() + branch * conditions.batch * config.model.global_dim, static_cast<size_t>(count * config.model.global_dim));
        if (state->inputs.value_mask.valid()) core::write_tensor_f32_slice(state->inputs.value_mask, 0, conditions.value_mask.data() + branch * conditions.batch * (conditions.frames + config.model.memory_tokens),
            static_cast<size_t>(count * (conditions.frames + config.model.memory_tokens)));
        state->condition_branch = branch;
    }

    TensorValue build_guidance(core::ModuleBuildContext & ctx, State & s, const TensorValue & prediction) {
        s.guidance = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1}));
        ggml_set_input(s.guidance.tensor); ggml_set_output(s.guidance.tensor);
        const auto positive = SliceModule({0, 0, conditions.batch}).build(ctx, prediction);
        const auto negative = SliceModule({0, conditions.batch, conditions.batch}).build(ctx, prediction);
        if (config.guidance == DitGuidance::Linear)
            return FlowSamplerCfgGraphModule().build(ctx, positive, negative, s.guidance);
        s.sigma = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({conditions.batch, 1, 1}));
        s.apg = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1}));
        ggml_set_input(s.sigma.tensor); ggml_set_input(s.apg.tensor); ggml_set_output(s.apg.tensor);
        const auto scaled = MulModule(true).build(ctx, positive, s.sigma);
        const auto x = core::ensure_backend_addressable_layout(ctx,
            TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, s.latent));
        const auto reference = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, scaled.tensor), positive.shape);
        return FlowSamplerApgGraphModule().build(ctx, positive, negative, reference, s.guidance, s.apg, s.latent_mask);
    }

    TensorValue build_sampling_step(core::ModuleBuildContext & ctx, State & s) {
        s.step = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({conditions.batch, 1, 1}));
        ggml_set_input(s.step.tensor);
        if (s.sampler == DitSampler::PingPong) {
            s.noise = core::make_tensor(ctx, GGML_TYPE_F32, s.latent.shape);
            s.next_time = core::make_tensor(ctx, GGML_TYPE_F32, s.step.shape);
            ggml_set_input(s.noise.tensor); ggml_set_input(s.next_time.tensor);
            return FlowSamplerPingPongGraphModule().build(ctx, s.latent, s.output, s.noise, s.step, s.next_time);
        }
        const auto rounding = s.branches == 1 && config.model.projection_type == DitProjectionType::Bf16Autocast
            ? FlowSamplerGraphRounding::Bf16 : FlowSamplerGraphRounding::FullPrecision;
        return FlowSamplerEulerGraphModule({rounding, rounding}).build(ctx, s.latent, s.output, s.step);
    }

    void ensure(int64_t branches, bool guided = false, DitSampler sampler = DitSampler::Euler) {
        if (!prepared) throw std::logic_error(config.label + " conditioning has not been prepared");
        if (branches <= 0 || branches > conditions.branches) throw std::invalid_argument(config.label + " unavailable guidance branches");
        if (state && state->branches == branches && state->guided == guided && state->sampler == sampler) return;
        state.reset();
        state = std::make_unique<State>();
        auto & s = *state;
        s.branches = branches; s.guided = guided; s.sampler = sampler;
        s.constants = std::make_unique<core::ConstantTensorCache>(execution.backend(), execution.config().threads, config.label + ".inputs", 1024 * 1024);
        const auto rows = branches * conditions.batch;
        const auto & c = config.model;
        const auto constant = [&](const TensorShape & shape) { return s.constants->make_f32(shape, std::vector<float>(static_cast<size_t>(shape.num_elements()), 0)); };
        const auto tokens = conditions.frames + c.memory_tokens;
        {
            for (const auto width : conditions.frame_channels) s.inputs.frame_conditions.push_back(constant(TensorShape::from_dims({rows, conditions.frames, width})));
            for (const auto width : conditions.vector_channels) s.inputs.vector_conditions.push_back(constant(TensorShape::from_dims({rows, width})));
            if (c.context_dim && config.conditioning_cache == DitConditioningCache::Unprojected)
                s.inputs.context = constant(TensorShape::from_dims({rows, conditions.context_tokens, c.context_dim}));
            if (c.local_condition_dim) s.inputs.local_condition = constant(TensorShape::from_dims({rows, conditions.frames, c.local_condition_dim}));
            if (c.global_dim) s.inputs.global = constant(TensorShape::from_dims({rows, c.global_dim}));
            if (!conditions.value_mask.empty()) s.inputs.value_mask = constant(TensorShape::from_dims({rows, 1, tokens, 1}));
        }
        std::vector<int32_t> positions(static_cast<size_t>(tokens));
        for (int64_t i = 0; i < tokens; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        s.inputs.positions = s.constants->make_tensor(TensorShape::from_dims({tokens}), GGML_TYPE_I32, positions.data(), positions.size() * sizeof(int32_t));
        if (guided && branches == 2 && config.guidance == DitGuidance::AdaptiveProjected && !conditions.latent_mask.empty())
            s.latent_mask = s.constants->make_f32(TensorShape::from_dims({conditions.batch, conditions.frames, 1}), conditions.latent_mask);
        s.constants->ensure_uploaded();
        const auto builder = [&](core::ModuleBuildContext & ctx) {
            ctx.elide_identity_reshapes = c.projection_type == DitProjectionType::Bf16Autocast;
            const auto latent_rows = conditions.batch;
            s.latent = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({latent_rows, c.output_channels, conditions.frames}));
            ggml_set_input(s.latent.tensor);
            auto latent = s.latent;
            if (latent_rows != rows) latent = RepeatModule({TensorShape::from_dims({rows, c.output_channels, conditions.frames})}).build(ctx, latent);
            s.inputs.latent = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, latent);
            // Without branch-specific global conditioning, both CFG branches
            // share timestep modulation rather than computing duplicate gates.
            const auto time_rows = c.global_dim ? rows : conditions.batch;
            s.time = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({time_rows, c.time_features}));
            ggml_set_input(s.time.tensor); s.inputs.time = s.time;
            if (!context_kv.empty()) {
                s.inputs.context_kv = context_kv;
                if (branches != conditions.branches) {
                    s.branch_indices = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({conditions.batch}));
                    ggml_set_input(s.branch_indices.tensor); ggml_set_output(s.branch_indices.tensor);
                    for (auto & cache : s.inputs.context_kv) {
                        for (auto * value : {&cache.key, &cache.value, &cache.differential_key}) {
                            if (!value->valid()) continue;
                            const auto shape = TensorShape::from_dims({rows, value->shape.dims[1], value->shape.dims[2], value->shape.dims[3]});
                            const auto width = shape.dims[1] * shape.dims[2] * shape.dims[3];
                            auto flat = core::reshape_tensor(ctx, *value, TensorShape::from_dims({conditions.batch * conditions.branches, width}));
                            *value = core::reshape_tensor(ctx, EmbeddingModule({conditions.batch * conditions.branches, width}).build(ctx, s.branch_indices, flat), shape);
                        }
                    }
                }
            }
            auto output = module.build(ctx, s.inputs, weights);
            if (guided && branches == 2) output = build_guidance(ctx, s, output);
            s.output = core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, output));
            return DeviceDenoiserGraph{s.latent, s.output, guided ? build_sampling_step(ctx, s) : s.latent};

        };
        s.device = std::make_unique<DeviceDenoiserRuntime>(execution,
            core::PreparedGraphConfig{config.label, config.graph_metadata_bytes, config.graph_nodes}, builder, config.optimizations);
        upload_conditions();
    }

    std::vector<float> predict(const std::vector<float> & latent, const std::vector<float> & times,
        int64_t branches, int64_t branch) {
        ensure(branches);
        auto & s = *state;
        const auto size = static_cast<size_t>(conditions.batch * config.model.output_channels * conditions.frames);
        if (latent.size() != size || times.size() != static_cast<size_t>(conditions.batch) || branch < 0 || branch + branches > conditions.branches)
            throw std::invalid_argument(config.label + " prediction input shape/branch mismatch");
        const auto started = std::chrono::steady_clock::now();
        if (s.condition_branch != branch) upload_conditions(branch);
        core::write_tensor_f32_slice(s.latent, 0, latent.data(), latent.size());
        core::write_tensor_f32(s.time, time_features(times, s.time.shape.dims[0]));

        if (s.branch_indices.valid()) {
            std::vector<int32_t> ids(static_cast<size_t>(conditions.batch));
            for (int64_t i = 0; i < conditions.batch; ++i) ids[static_cast<size_t>(i)] = static_cast<int32_t>(branch * conditions.batch + i);
            core::write_tensor_i32(s.branch_indices, ids);
        }
        debug::timing_log_scalar(config.label + ".input_upload_ms", debug::elapsed_ms(started));
        return s.device->predict();
    }

    std::vector<float> sample(const DitSamplingRequest & request);
    TensorValue embed_time(core::ModuleBuildContext & ctx, const TensorValue & input,
        const std::vector<LinearWeights> & embedding);
    void ensure_graph_frequencies();
    DitModulationBank prepare_modulations(const std::vector<float> & times,
        const std::vector<float> & durations, const std::vector<float> & global, DitOperationStats * stats);
    std::vector<float> predict_sequence(const DitSequenceRequest & request, DitOperationStats * stats);
    void reserve_prefix_cache(DitPrefixCacheConfig requested);
    void prepare_prefix_graphs(int64_t prefill_frames);
    void initialize_prefix_state();
    void prefill_prefix(const std::vector<float> & tokens, int64_t frames, int64_t bank, DitOperationStats * stats);
    std::vector<float> predict_prefix(const std::vector<float> & tokens, int64_t past,
        int64_t bank, float guidance, DitOperationStats * stats);
};


void DitRuntime::Impl::ensure_graph_frequencies() {
    if (frequency_cache) return;
    frequency_cache = std::make_unique<core::ConstantTensorCache>(execution.backend(), execution.config().threads,
        config.label + ".frequencies", 1024 * 1024);
    graph_frequencies = frequency_cache->make_f32(TensorShape::from_dims({static_cast<int64_t>(frequencies.size())}), frequencies);
    frequency_cache->ensure_uploaded();
}

TensorValue DitRuntime::Impl::embed_time(core::ModuleBuildContext & ctx, const TensorValue & input,
    const std::vector<LinearWeights> & embedding) {
    if (embedding.size() != 2) throw std::invalid_argument("DiT timestep embedding requires two projections");
    auto scaled = input;
    if (config.time_scale != 1) scaled = ScaleBiasModule({config.time_scale, 0}).build(ctx, input);
    TimestepEmbeddingConfig time_config{config.model.time_features, config.model.hidden_size,
        config.model.norm_eps, TimestepEmbeddingNorm::None};
    return TimestepEmbeddingModule(time_config).build(ctx, scaled, {graph_frequencies, embedding[0], embedding[1], std::nullopt});
}

DitModulationBank DitRuntime::Impl::prepare_modulations(const std::vector<float> & times,
    const std::vector<float> & durations, const std::vector<float> & global, DitOperationStats * stats) {
    const auto rows = static_cast<int64_t>(times.size());
    const auto & c = config.model;
    const auto global_rows = c.global_dim ? static_cast<int64_t>(global.size() / c.global_dim) : 0;
    if (rows <= 0 || (!durations.empty() && durations.size() != times.size()) ||
        (c.global_dim ? (global_rows <= 0 || global.size() % c.global_dim || rows % global_rows) : !global.empty()))
        throw std::invalid_argument(config.label + " modulation conditioning shape mismatch");
    const auto width = (6 * c.layers + 2) * c.hidden_size;
    if (modulation && modulation->times == times && modulation->durations == durations &&
        modulation->global_values == global) return {rows, width};
    const auto started = std::chrono::steady_clock::now();
    ensure_graph_frequencies();
    if (!modulation || modulation->rows != rows || modulation->global_rows != global_rows ||
        modulation->has_duration != !durations.empty()) {
        // Captures borrow the bank's address. Release them before replacing it.
        prefix.reset(); sequence.reset(); modulation.reset();
        modulation = std::make_unique<ModulationState>();
        auto & m = *modulation;
        m.rows = rows; m.global_rows = global_rows; m.has_duration = !durations.empty();
        m.graph = std::make_unique<core::PreparedGraph>(execution,
            core::PreparedGraphConfig{config.label + ".modulation", config.graph_metadata_bytes, config.graph_nodes, 1024 * 1024});
        m.output = core::make_tensor(m.graph->persistent_build_context(), config.modulation_storage_type, TensorShape::from_dims({rows, width}));
        m.graph->allocate_persistent();
        auto & ctx = m.graph->build_context();
        const auto input = [&](int64_t columns) {
            auto value = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({rows, columns}));
            ggml_set_input(value.tensor);
            return value;
        };
        m.time = input(1);
        DitForwardInputs in;
        in.time_condition = embed_time(ctx, m.time, weights.time_embedding);
        if (c.global_dim) {
            m.global = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({global_rows, c.global_dim}));
            ggml_set_input(m.global.tensor); in.global = m.global;
        }
        if (m.has_duration) { m.duration = input(1); in.duration_condition = embed_time(ctx, m.duration, weights.duration_embedding); }
        const auto output = module.build_modulations(ctx, in, weights);
        ggml_build_forward_expand(m.graph->graph(), ggml_cpy(ctx.ggml, output.tensor, m.output.tensor));
        m.graph->allocate();
    }
    const auto prepared_at = std::chrono::steady_clock::now();
    auto & m = *modulation;
    m.times.clear();
    if (prefix) {
        std::fill(prefix->valid.begin(), prefix->valid.end(), 0);
        prefix->last_past = -1;
        prefix->last_guidance = std::numeric_limits<float>::quiet_NaN();
    }
    core::write_tensor_f32(m.time, times);
    if (m.duration.valid()) core::write_tensor_f32(m.duration, durations);
    if (m.global.valid()) core::write_tensor_f32(m.global, global);
    const auto uploaded_at = std::chrono::steady_clock::now();
    m.graph->compute();
    m.times = times; m.durations = durations; m.global_values = global;
    if (stats) {
        stats->prepare_ms += debug::elapsed_ms(started, prepared_at);
        stats->upload_ms += debug::elapsed_ms(prepared_at, uploaded_at);
        stats->compute_ms += debug::elapsed_ms(uploaded_at);
    }
    return {rows, width};
}

std::vector<float> DitRuntime::Impl::predict_sequence(const DitSequenceRequest & request, DitOperationStats * stats) {
    const auto & c = config.model;
    const bool precomputed = request.modulation_start >= 0;
    const bool has_duration = request.durations && !request.durations->empty();
    const bool has_mask = request.attention_mask && !request.attention_mask->empty();
    const auto rows = request.batch, frames = request.frames;
    if (rows <= 0 || frames <= 0 || request.tokens.size() != static_cast<size_t>(rows * frames * c.input_channels) ||
        request.output_start < 0 || request.output_tokens < 0 || request.output_start + request.output_tokens > frames)
        throw std::invalid_argument(config.label + " sequence shape/range mismatch");
    if (precomputed && (!modulation || request.modulation_start + rows > modulation->rows))
        throw std::invalid_argument(config.label + " modulation bank row range mismatch");
    if (!precomputed && (!request.timesteps || request.timesteps->size() != static_cast<size_t>(rows) ||
        (c.global_dim && (!request.global || request.global->size() != static_cast<size_t>(rows * c.global_dim))) ||
        (has_duration && request.durations->size() != static_cast<size_t>(rows))))
        throw std::invalid_argument(config.label + " sequence conditioning shape mismatch");
    if ((request.positions && !request.positions->empty() && request.positions->size() != static_cast<size_t>(frames)) ||
        (has_mask && request.attention_mask->size() != static_cast<size_t>(rows * frames * frames)))
        throw std::invalid_argument(config.label + " sequence positions/mask shape mismatch");
    const auto started = std::chrono::steady_clock::now();
    if (!precomputed) ensure_graph_frequencies();
    if (!sequence || sequence->batch != rows || sequence->frames != frames || sequence->precomputed != precomputed ||
        sequence->has_duration != has_duration || sequence->has_mask != has_mask ||
        sequence->output_start != request.output_start || sequence->output_tokens != request.output_tokens ||
        sequence->guided != request.guidance.has_value()) {
        sequence.reset();
        sequence = std::make_unique<SequenceState>();
        auto & s = *sequence;
        s.batch = rows; s.frames = frames; s.precomputed = precomputed; s.has_duration = has_duration; s.has_mask = has_mask;
        s.output_start = request.output_start; s.output_tokens = request.output_tokens; s.guided = request.guidance.has_value();
        s.graph = std::make_unique<core::PreparedGraph>(execution,
            core::PreparedGraphConfig{config.label + ".sequence", config.graph_metadata_bytes, config.graph_nodes});
        auto & ctx = s.graph->build_context();
        const auto input = [&](ggml_type type, TensorShape shape) {
            auto value = core::make_tensor(ctx, type, shape); ggml_set_input(value.tensor); return value;
        };
        s.tokens = input(GGML_TYPE_F32, TensorShape::from_dims({rows, frames, c.input_channels}));
        s.positions = input(GGML_TYPE_I32, TensorShape::from_dims({frames}));
        DitForwardInputs in;
        in.latent = s.tokens; in.positions = s.positions; in.output_start = request.output_start; in.output_tokens = request.output_tokens;
        if (precomputed) {
            s.indices = input(GGML_TYPE_I32, TensorShape::from_dims({rows}));
            in.modulations = EmbeddingModule({modulation->rows, modulation->output.shape.last_dim()}).build(ctx, s.indices, modulation->output);
        } else {
            s.time = input(GGML_TYPE_F32, TensorShape::from_dims({rows, 1}));
            in.time_condition = embed_time(ctx, s.time, weights.time_embedding);
            if (c.global_dim) { s.global = input(GGML_TYPE_F32, TensorShape::from_dims({rows, c.global_dim})); in.global = s.global; }
            if (has_duration) { s.duration = input(GGML_TYPE_F32, TensorShape::from_dims({rows, 1})); in.duration_condition = embed_time(ctx, s.duration, weights.duration_embedding); }
        }
        if (has_mask) { s.mask = input(GGML_TYPE_F16, TensorShape::from_dims({rows, 1, frames, frames})); in.attention_mask = s.mask; }
        s.output = module.build(ctx, in, weights);
        if (s.guided) {
            // Preserve the predictor output dtype boundary when moving the
            // formerly host-side guidance into this graph.
            if (c.projection_type == DitProjectionType::Bf16Autocast) s.output = Bf16RoundModule().build(ctx, s.output);
            if (rows % 2) throw std::invalid_argument(config.label + " CFG requires paired sequence branches");
            s.guidance = input(GGML_TYPE_F32, TensorShape::from_dims({1}));
            s.output = FlowSamplerCfgGraphModule().build(ctx,
                SliceModule({0, 0, rows / 2}).build(ctx, s.output),
                SliceModule({0, rows / 2, rows / 2}).build(ctx, s.output), s.guidance);
        }
        ggml_set_output(s.output.tensor);
        ggml_build_forward_expand(s.graph->graph(), s.output.tensor);
        s.graph->allocate();
    }
    const auto prepared_at = std::chrono::steady_clock::now();
    auto & s = *sequence;
    core::write_tensor_f32(s.tokens, request.tokens);
    if (s.guidance.valid()) core::write_tensor_f32(s.guidance, &*request.guidance, 1);
    if (precomputed) {
        std::vector<int32_t> ids(static_cast<size_t>(rows));
        for (int64_t b = 0; b < rows; ++b) ids[static_cast<size_t>(b)] = static_cast<int32_t>(request.modulation_start + b);
        core::write_tensor_i32(s.indices, ids);
    } else {
        core::write_tensor_f32(s.time, *request.timesteps);
        if (s.duration.valid()) core::write_tensor_f32(s.duration, *request.durations);
        if (s.global.valid()) core::write_tensor_f32(s.global, *request.global);
    }
    std::vector<int32_t> positions(static_cast<size_t>(frames));
    if (request.positions && !request.positions->empty()) positions = *request.positions;
    else for (int64_t i = 0; i < frames; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    core::write_tensor_i32(s.positions, positions);
    if (has_mask) {
        std::vector<ggml_fp16_t> mask(request.attention_mask->size());
        for (size_t i = 0; i < mask.size(); ++i) mask[i] = ggml_fp32_to_fp16((*request.attention_mask)[i] ? 0.f : -std::numeric_limits<float>::infinity());
        ggml_backend_tensor_set(s.mask.tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    const auto uploaded_at = std::chrono::steady_clock::now();
    s.graph->compute();
    const auto computed_at = std::chrono::steady_clock::now();
    auto output = core::read_tensor_f32(s.output.tensor);
    if (stats) {
        stats->prepare_ms += debug::elapsed_ms(started, prepared_at);
        stats->upload_ms += debug::elapsed_ms(prepared_at, uploaded_at);
        stats->compute_ms += debug::elapsed_ms(uploaded_at, computed_at);
        stats->read_ms += debug::elapsed_ms(computed_at);
    }
    return output;
}

void DitRuntime::Impl::reserve_prefix_cache(DitPrefixCacheConfig requested) {
    const int64_t maximum = std::numeric_limits<int32_t>::max();
    if (requested.capacity <= 0 || requested.banks <= 0 || requested.batch <= 0 ||
        requested.branches <= 0 || requested.branches > 2 || requested.unit_tokens <= 0 ||
        requested.unit_tokens > requested.capacity || requested.output_tokens <= 0 ||
        requested.output_tokens > 2 * requested.unit_tokens ||
        requested.batch > maximum / requested.branches ||
        requested.batch * requested.branches > maximum / requested.banks ||
        requested.batch * requested.branches * requested.banks > maximum / requested.capacity)
        throw std::invalid_argument(config.label + " invalid prefix-cache dimensions or int32 row range");
    if (requested.branches == 2 && config.guidance == DitGuidance::AdaptiveProjected)
        throw std::invalid_argument("Hidden-token prefix decoding does not define latent-space projected guidance");
    if (prefix) {
        const auto & old = prefix->config;
        if (old.capacity == requested.capacity && old.banks == requested.banks && old.batch == requested.batch &&
            old.branches == requested.branches && old.unit_tokens == requested.unit_tokens &&
            old.output_tokens == requested.output_tokens && old.storage_type == requested.storage_type) return;
    }
    prefix.reset();
    prefix = std::make_unique<PrefixState>();
    prefix->config = requested;
    prefix->valid.assign(static_cast<size_t>(requested.banks), 0);
}

void DitRuntime::Impl::prepare_prefix_graphs(int64_t prefill_frames) {
    auto & p = *prefix;
    if (p.graph && p.prefill_frames == prefill_frames) {
        if (p.graph->allocated()) return;
        p.graph->allocate_persistent();
        p.graph->clear_persistent();
        p.graph->allocate_releasable();
        initialize_prefix_state();
        return;
    }
    const auto & cache = p.config;
    const auto & c = config.model;
    const auto rows = cache.batch * cache.branches, tail = 2 * cache.unit_tokens;
    if (!modulation || modulation->rows < cache.banks * rows || prefill_frames < 0 || prefill_frames > cache.capacity)
        throw std::invalid_argument(config.label + " prefix cache requires matching prepared modulation rows");
    p.graph.reset(); p.cache.clear(); p.steps.clear(); p.outputs.clear();
    p.prefill = {}; p.step = {}; p.last_past = -1;
    p.last_guidance = std::numeric_limits<float>::quiet_NaN();
    std::fill(p.valid.begin(), p.valid.end(), 0);
    p.prefill_frames = prefill_frames;
    p.graph = std::make_unique<core::PreparedGraph>(execution,
        core::PreparedGraphConfig{config.label + ".prefix", config.graph_metadata_bytes, config.graph_nodes, 4 * 1024 * 1024});
    auto & persistent = p.graph->persistent_build_context();
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        DitContextKV values;
        for (auto * value : {&values.key, &values.value})
            *value = core::make_tensor(persistent, cache.storage_type,
                TensorShape::from_dims({1, cache.banks * rows * cache.capacity, c.self_attention.kv_heads, c.self_attention.head_dim}));
        if (c.self_attention.differential) values.differential_key = core::make_tensor(persistent, cache.storage_type, values.key.shape);
        p.cache.push_back(values);
    }
    const auto inputs = [&](PrefixInputs & in, int64_t frames, int64_t key_frames, int64_t appended) {
        in.tokens = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({rows, frames, c.input_channels}));
        in.positions = core::make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({frames}));
        in.mask = core::make_tensor(persistent, GGML_TYPE_F16, TensorShape::from_dims({1, 1, frames, key_frames}));
        in.slots = core::make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({rows * appended}));
        in.indices = core::make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({rows}));
    };
    if (prefill_frames) inputs(p.prefill, prefill_frames, prefill_frames, prefill_frames);
    inputs(p.step, tail, cache.capacity + tail, cache.unit_tokens);
    if (cache.branches == 2) p.step.guidance = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({1}));
    p.graph->allocate_persistent();
    p.graph->clear_persistent();
    auto & ctx = p.graph->build_context();
    const auto build = [&](PrefixInputs & input, int64_t frames, int64_t appended, int64_t bank, size_t graph_id, bool prefill) {
        DitForwardInputs in;
        in.latent = input.tokens; in.positions = input.positions; in.attention_mask = input.mask;
        in.modulations = EmbeddingModule({modulation->rows, modulation->output.shape.last_dim()}).build(ctx, input.indices, modulation->output);
        DitSelfCacheInputs state;
        state.destination = p.cache; state.slots = input.slots; state.append_tokens = appended; state.storage_type = cache.storage_type;
        if (!prefill) {
            for (const auto & layer : p.cache) {
                DitContextKV view;
                for (const auto & pair : {std::pair<TensorValue, TensorValue *>{layer.key, &view.key},
                    {layer.value, &view.value}, {layer.differential_key, &view.differential_key}}) {
                    if (!pair.first.valid()) continue;
                    auto * tensor = pair.first.tensor;
                    *pair.second = core::wrap_tensor(ggml_view_4d(ctx.ggml, tensor, c.self_attention.head_dim,
                        cache.capacity, c.self_attention.kv_heads, rows, tensor->nb[2], tensor->nb[1],
                        static_cast<size_t>(cache.capacity) * tensor->nb[2],
                        static_cast<size_t>(bank * rows * cache.capacity) * tensor->nb[2]),
                        TensorShape::from_dims({rows, c.self_attention.kv_heads, cache.capacity, c.self_attention.head_dim}), cache.storage_type);
                }
                state.prefix.push_back(view);
            }
            in.output_start = frames - cache.output_tokens; in.output_tokens = cache.output_tokens;
        }
        in.self_cache = std::move(state);
        auto output = module.build(ctx, in, weights, p.graph->graph(graph_id));
        if (prefill) return;
        if (cache.branches == 2) {
            const auto positive = SliceModule({0, 0, cache.batch}).build(ctx, output);
            const auto negative = SliceModule({0, cache.batch, cache.batch}).build(ctx, output);
            output = FlowSamplerCfgGraphModule().build(ctx, positive, negative, input.guidance);
        }
        p.outputs.push_back(output);
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(p.graph->graph(graph_id), output.tensor);
    };
    if (prefill_frames) build(p.prefill, prefill_frames, prefill_frames, 0, 0, true);
    for (int64_t bank = 0; bank < cache.banks; ++bank) {
        const auto id = !prefill_frames && bank == 0 ? size_t{0} : p.graph->add_graph("step" + std::to_string(bank));
        p.steps.push_back(id);
        build(p.step, tail, cache.unit_tokens, bank, id, false);
    }
    p.graph->allocate_releasable();
    initialize_prefix_state();
}

void DitRuntime::Impl::initialize_prefix_state() {
    auto & p = *prefix;
    const auto & cache = p.config;
    const auto prefill_frames = p.prefill_frames;
    const auto rows = cache.batch * cache.branches, tail = 2 * cache.unit_tokens;
    std::fill(p.valid.begin(), p.valid.end(), 0);
    p.last_past = -1;
    p.last_guidance = std::numeric_limits<float>::quiet_NaN();
    if (prefill_frames) {
        std::vector<int32_t> positions(static_cast<size_t>(prefill_frames));
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(prefill_frames * prefill_frames), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t row = 0; row < prefill_frames; ++row) {
            positions[static_cast<size_t>(row)] = static_cast<int32_t>(row);
            for (int64_t column = 0; column <= row; ++column) mask[static_cast<size_t>(row * prefill_frames + column)] = ggml_fp32_to_fp16(0);
        }
        core::write_tensor_i32(p.prefill.positions, positions);
        ggml_backend_tensor_set(p.prefill.mask.tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    p.positions.resize(static_cast<size_t>(tail));
    p.slots.resize(static_cast<size_t>(rows * (std::max)(prefill_frames, cache.unit_tokens)));
    p.indices.resize(static_cast<size_t>(rows));
    p.mask.resize(static_cast<size_t>(tail * (cache.capacity + tail)));
}

void DitRuntime::Impl::prefill_prefix(const std::vector<float> & tokens, int64_t frames, int64_t bank, DitOperationStats * stats) {
    if (!prefix || bank < 0 || bank >= prefix->config.banks || frames < 0)
        throw std::invalid_argument(config.label + " invalid prefix-cache bank or prefill length");
    auto & p = *prefix;
    if (!frames) { p.valid[static_cast<size_t>(bank)] = 0; return; }
    const auto rows = p.config.batch * p.config.branches;
    if (tokens.size() != static_cast<size_t>(rows * frames * config.model.input_channels))
        throw std::invalid_argument(config.label + " prefix prefill token shape mismatch");
    const auto started = std::chrono::steady_clock::now();
    prepare_prefix_graphs(frames);
    const auto prepared_at = std::chrono::steady_clock::now();
    core::write_tensor_f32(p.prefill.tokens, tokens);
    for (int64_t row = 0; row < rows; ++row) {
        p.indices[static_cast<size_t>(row)] = static_cast<int32_t>(bank * rows + row);
        for (int64_t t = 0; t < frames; ++t) p.slots[static_cast<size_t>(row * frames + t)] = static_cast<int32_t>((bank * rows + row) * p.config.capacity + t);
    }
    core::write_tensor_i32(p.prefill.indices, p.indices);
    ggml_backend_tensor_set(p.prefill.slots.tensor, p.slots.data(), 0, static_cast<size_t>(rows * frames) * sizeof(int32_t));
    const auto uploaded_at = std::chrono::steady_clock::now();
    p.graph->compute(0);
    p.valid[static_cast<size_t>(bank)] = frames;
    if (stats) {
        stats->prepare_ms += debug::elapsed_ms(started, prepared_at);
        stats->upload_ms += debug::elapsed_ms(prepared_at, uploaded_at);
        stats->compute_ms += debug::elapsed_ms(uploaded_at);
    }
}

std::vector<float> DitRuntime::Impl::predict_prefix(const std::vector<float> & tokens, int64_t past,
    int64_t bank, float guidance, DitOperationStats * stats) {
    if (!prefix || bank < 0 || bank >= prefix->config.banks) throw std::invalid_argument(config.label + " prefix-cache bank is not reserved");
    auto & p = *prefix;
    const auto & cache = p.config;
    const auto rows = cache.batch * cache.branches, tail = 2 * cache.unit_tokens, columns = cache.capacity + tail;
    if (past < 0 || past + cache.unit_tokens > cache.capacity || p.valid[static_cast<size_t>(bank)] != past ||
        tokens.size() != static_cast<size_t>(rows * tail * config.model.input_channels) || !std::isfinite(guidance))
        throw std::invalid_argument(config.label + " prefix decode shape or history mismatch");
    const auto started = std::chrono::steady_clock::now();
    if (!p.graph || !p.graph->allocated()) prepare_prefix_graphs(p.graph ? p.prefill_frames : 0);
    const auto prepared_at = std::chrono::steady_clock::now();
    core::write_tensor_f32(p.step.tokens, tokens);
    if (p.last_past != past) {
        const auto visible = ggml_fp32_to_fp16(0), hidden = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        std::fill(p.mask.begin(), p.mask.end(), hidden);
        for (int64_t row = 0; row < tail; ++row) {
            p.positions[static_cast<size_t>(row)] = static_cast<int32_t>(past + row);
            for (int64_t col = 0; col < past; ++col) p.mask[static_cast<size_t>(row * columns + col)] = visible;
            const auto tail_columns = row < cache.unit_tokens ? row + 1 : tail;
            for (int64_t col = 0; col < tail_columns; ++col) p.mask[static_cast<size_t>(row * columns + cache.capacity + col)] = visible;
        }
        core::write_tensor_i32(p.step.positions, p.positions);
        ggml_backend_tensor_set(p.step.mask.tensor, p.mask.data(), 0, p.mask.size() * sizeof(ggml_fp16_t));
        p.last_past = past;
    }
    for (int64_t row = 0; row < rows; ++row) {
        p.indices[static_cast<size_t>(row)] = static_cast<int32_t>(bank * rows + row);
        for (int64_t t = 0; t < cache.unit_tokens; ++t) p.slots[static_cast<size_t>(row * cache.unit_tokens + t)] = static_cast<int32_t>((bank * rows + row) * cache.capacity + past + t);
    }
    core::write_tensor_i32(p.step.indices, p.indices);
    ggml_backend_tensor_set(p.step.slots.tensor, p.slots.data(), 0, static_cast<size_t>(rows * cache.unit_tokens) * sizeof(int32_t));
    if (p.step.guidance.valid() && p.last_guidance != guidance) {
        core::write_tensor_f32(p.step.guidance, &guidance, 1);
        p.last_guidance = guidance;
    }
    const auto uploaded_at = std::chrono::steady_clock::now();
    p.graph->compute(p.steps.at(static_cast<size_t>(bank)));
    const auto computed_at = std::chrono::steady_clock::now();
    p.valid[static_cast<size_t>(bank)] += cache.unit_tokens;
    auto result = core::read_tensor_f32(p.outputs.at(static_cast<size_t>(bank)).tensor);
    if (stats) {
        stats->prepare_ms += debug::elapsed_ms(started, prepared_at);
        stats->upload_ms += debug::elapsed_ms(prepared_at, uploaded_at);
        stats->compute_ms += debug::elapsed_ms(uploaded_at, computed_at);
        stats->read_ms += debug::elapsed_ms(computed_at);
    }
    return result;
}

std::vector<float> DitRuntime::Impl::sample(const DitSamplingRequest & input_request) {
    auto request = input_request;
    if (!prepared) throw std::logic_error(config.label + " conditioning has not been prepared");
    const auto count = static_cast<size_t>(conditions.batch * config.model.output_channels * conditions.frames);
    if (request.noise.size() != count || request.steps <= 0 || !std::isfinite(request.guidance))
        throw std::invalid_argument(config.label + " invalid sampling inputs");
    const auto started = std::chrono::steady_clock::now();
    const int64_t branches = request.guidance == 1 ? 1 : 2;
    ensure(branches, true, request.sampler);
    const auto points = request.steps + 1;
    const auto schedule_size = static_cast<size_t>(conditions.batch * points);
    if (request.schedule.empty()) {
        if (!(request.sigma_shift > 0) || !std::isfinite(request.sigma_shift))
            throw std::invalid_argument(config.label + " invalid schedule shift");
        request.schedule.resize(schedule_size);
        for (int64_t b = 0; b < conditions.batch; ++b) {
            for (int64_t i = 0; i < points; ++i) {
                const float u = static_cast<float>(i) / static_cast<float>(request.steps);
                float value = 0;
                if (request.schedule_type == DitSchedule::Cosine) value = 1.f - std::cos(u * .5f * kPi);
                else if (request.schedule_type == DitSchedule::Shifted) {
                    const float r = 1.f - u;
                    value = request.sigma_shift * r / (1.f + (request.sigma_shift - 1.f) * r);
                } else value = u;
                request.schedule[static_cast<size_t>(b * points + i)] = value;
            }
        }
    }
    if (request.schedule.size() != schedule_size)
        throw std::invalid_argument(config.label + " schedule shape mismatch");
    auto & s = *state;
    if (s.condition_branch != 0) upload_conditions();
    if (s.guidance.valid()) core::write_tensor_f32(s.guidance, &request.guidance, 1);
    if (s.apg.valid()) core::write_tensor_f32(s.apg, &request.apg_scale, 1);
    if (s.branch_indices.valid()) {
        std::vector<int32_t> indices(static_cast<size_t>(conditions.batch));
        for (int64_t i = 0; i < conditions.batch; ++i) indices[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        core::write_tensor_i32(s.branch_indices, indices);
    }
    const auto width = static_cast<size_t>(s.time.shape.num_elements());
    const auto step_rows = static_cast<size_t>(request.steps * conditions.batch);
    std::vector<float> features(static_cast<size_t>(request.steps) * width), times(step_rows), next_times(step_rows), deltas(step_rows);
    std::vector<float> batch_times(static_cast<size_t>(conditions.batch));
    std::vector<float> noise;
    if (request.sampler == DitSampler::PingPong) {
        if (!request.rng_offset_blocks) throw std::invalid_argument(config.label + " ping-pong sampling requires an RNG counter");
        noise.reserve(static_cast<size_t>(request.steps) * count);
    }
    for (int64_t step = 0; step < request.steps; ++step) {
        for (int64_t b = 0; b < conditions.batch; ++b) {
            const auto source = static_cast<size_t>(b * points + step), target = static_cast<size_t>(step * conditions.batch + b);
            const auto time = request.schedule[source], next = request.schedule[source + 1];
            if (!std::isfinite(time) || !std::isfinite(next)) throw std::invalid_argument(config.label + " non-finite sampling schedule");
            times[target] = time; next_times[target] = next; deltas[target] = next - time;
            batch_times[static_cast<size_t>(b)] = time * request.timestep_scale;
        }
        const auto embedded = time_features(batch_times, s.time.shape.dims[0]);
        std::copy(embedded.begin(), embedded.end(), features.begin() + static_cast<std::ptrdiff_t>(step * width));
        if (request.sampler == DitSampler::PingPong) {
            const auto random = sampling::generate_torch_cuda_tensor_iterator_randn(count, request.seed, *request.rng_offset_blocks,
                request.rng_policy, sampling::TorchRandnPrecision::Float32);
            *request.rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(static_cast<uint64_t>(count), request.rng_policy);
            noise.insert(noise.end(), random.begin(), random.end());
        }
    }
    FlowSamplerGraphSequence sequence;
    sequence.steps = static_cast<size_t>(request.steps);
    const size_t time_bytes = static_cast<size_t>(conditions.batch) * sizeof(float);
    sequence.step_bindings.push_back({s.time, 0, width * sizeof(float), features.data(), features.size() * sizeof(float)});
    const auto & step_values = request.sampler == DitSampler::Euler ? deltas : times;
    sequence.step_bindings.push_back({s.step, 0, time_bytes, step_values.data(), step_values.size() * sizeof(float)});
    if (s.sigma.valid()) sequence.step_bindings.push_back({s.sigma, 0, time_bytes, times.data(), times.size() * sizeof(float)});
    if (s.noise.valid()) {
        sequence.step_bindings.push_back({s.noise, 0, count * sizeof(float), noise.data(), noise.size() * sizeof(float)});
        sequence.step_bindings.push_back({s.next_time, 0, time_bytes, next_times.data(), next_times.size() * sizeof(float)});
    }
    auto result = s.device->sample(request.noise, sequence);
    debug::timing_log_scalar(config.label + ".total_ms", debug::elapsed_ms(started));
    return result;
}

DitRuntime::DitRuntime(core::ExecutionContext & execution, const assets::TensorSource & source, DitRuntimeConfig config, const DitWeightNames & names)
    : impl_(std::make_unique<Impl>(execution, source, std::move(config), names)) {}
DitRuntime::~DitRuntime() = default;
void DitRuntime::prepare_conditioning(DitConditioning conditioning) { impl_->prepare(std::move(conditioning)); }
std::vector<float> DitRuntime::predict(const std::vector<float> & latent, float timestep, int64_t branch) {
    return impl_->predict(latent, std::vector<float>(static_cast<size_t>(impl_->conditions.batch), timestep), 1, branch);
}
std::vector<float> DitRuntime::predict_cfg(const std::vector<float> & latent, float timestep) {
    return impl_->predict(latent, std::vector<float>(static_cast<size_t>(impl_->conditions.batch), timestep), 2, 0);
}
std::vector<float> DitRuntime::sample(const DitSamplingRequest & request) { return impl_->sample(request); }
DitModulationBank DitRuntime::prepare_modulations(const std::vector<float> & times,
    const std::vector<float> & durations, const std::vector<float> & global, DitOperationStats * stats) {
    return impl_->prepare_modulations(times, durations, global, stats);
}
std::vector<float> DitRuntime::predict_sequence(const DitSequenceRequest & request, DitOperationStats * stats) {
    return impl_->predict_sequence(request, stats);
}
void DitRuntime::reserve_prefix_cache(DitPrefixCacheConfig config) { impl_->reserve_prefix_cache(config); }
void DitRuntime::reset_prefix_cache() {
    if (!impl_->prefix) return;
    std::fill(impl_->prefix->valid.begin(), impl_->prefix->valid.end(), 0);
    impl_->prefix->last_past = -1;
    impl_->prefix->last_guidance = std::numeric_limits<float>::quiet_NaN();
}
void DitRuntime::release_prefix_buffers() {
    if (!impl_->prefix) return;
    auto & p = *impl_->prefix;
    if (p.graph && p.graph->allocated()) p.graph->release_buffers();
    reset_prefix_cache();
}
int64_t DitRuntime::prefix_tokens(int64_t bank) const {
    if (!impl_->prefix) throw std::logic_error("DiT prefix cache has not been reserved");
    return impl_->prefix->valid.at(static_cast<size_t>(bank));
}
void DitRuntime::prefill_prefix(const std::vector<float> & tokens, int64_t frames, int64_t bank, DitOperationStats * stats) {
    impl_->prefill_prefix(tokens, frames, bank, stats);
}
std::vector<float> DitRuntime::predict_prefix(const std::vector<float> & tokens, int64_t past,
    int64_t bank, float guidance, DitOperationStats * stats) {
    return impl_->predict_prefix(tokens, past, bank, guidance, stats);
}
void DitRuntime::release_graphs() {
    impl_->prefix.reset(); impl_->sequence.reset(); impl_->modulation.reset(); impl_->frequency_cache.reset(); impl_->graph_frequencies = {};
    impl_->state.reset(); impl_->context_graph.reset(); impl_->context_kv.clear(); impl_->context_store.reset(); impl_->prepared = false;
}

} // namespace engine::modules
