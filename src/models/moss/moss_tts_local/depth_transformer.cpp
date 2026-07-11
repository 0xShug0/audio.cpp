#include "engine/models/moss/moss_tts_local/depth_transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct DepthWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::NormWeights ln_1;
    modules::LinearWeights c_attn;
    modules::LinearWeights c_proj;
    modules::NormWeights ln_2;
    modules::LinearWeights fc_in;
    modules::LinearWeights fc_out;
    modules::NormWeights ln_f;
};

DepthWeights load_depth_weights(
    const MossTTSLocalAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes) {
    const auto & config = assets.config.local;
    const auto & source = *assets.model_weights;
    const int64_t hidden = config.hidden_size;
    const int64_t inner = config.intermediate_size;
    DepthWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "moss_tts_local.depth.weights",
        weight_context_bytes);
    auto & store = *weights.store;
    const std::string prefix = "local_transformer.h.0";
    weights.ln_1 = binding::norm_from_source(store, source, prefix + ".ln_1", hidden);
    weights.c_attn = binding::linear_from_source(
        store, source, prefix + ".attn.c_attn", assets::TensorStorageType::F32, 3 * hidden, hidden, true);
    weights.c_proj = binding::linear_from_source(
        store, source, prefix + ".attn.c_proj", assets::TensorStorageType::F32, hidden, hidden, true);
    weights.ln_2 = binding::norm_from_source(store, source, prefix + ".ln_2", hidden);
    weights.fc_in = binding::linear_from_source(
        store, source, prefix + ".mlp.fc_in", assets::TensorStorageType::F32, inner, hidden, true);
    weights.fc_out = binding::linear_from_source(
        store, source, prefix + ".mlp.fc_out", assets::TensorStorageType::F32, hidden, inner, true);
    weights.ln_f = binding::norm_from_source(store, source, "local_transformer.ln_f", hidden);
    store.upload();
    return weights;
}

core::TensorValue local_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DepthWeights & weights,
    const MossLocalTransformerConfig & config,
    const core::TensorValue & attention_mask) {
    const int64_t hidden = config.hidden_size;
    const int64_t heads = config.num_heads;
    const int64_t dim = hidden / heads;
    const modules::LayerNormModule attn_norm({hidden, config.layer_norm_eps, true, true});
    const modules::LinearModule c_attn(binding::linear_config(hidden, 3 * hidden, true));
    const modules::LinearModule c_proj(binding::linear_config(hidden, hidden, true));

    auto x_norm = attn_norm.build(ctx, input, weights.ln_1);
    auto qkv = c_attn.build(ctx, x_norm, weights.c_attn);
    auto q = modules::SliceModule({2, 0, hidden}).build(ctx, qkv);
    auto k = modules::SliceModule({2, hidden, hidden}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * hidden, hidden}).build(ctx, qkv);
    q = modules::ReshapeModule({
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], heads, dim}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, q));
    k = modules::ReshapeModule({
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], heads, dim}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, k));
    v = modules::ReshapeModule({
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], heads, dim}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, v));
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, config.rope_base}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, config.rope_base}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context =
        modules::ScaledDotProductAttentionModule({
            dim,
            modules::ScaledDotProductAttentionLowering::Explicit,
            GGML_PREC_F32,
        })
            .build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = modules::ReshapeModule({
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, context));
    auto x = modules::AddModule{}.build(ctx, input, c_proj.build(ctx, context, weights.c_proj));

    auto ff_in = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, x, weights.ln_2);
    auto ff = modules::LinearModule(binding::linear_config(hidden, config.intermediate_size, true))
                  .build(ctx, ff_in, weights.fc_in);
    ff = modules::SiluModule{}.build(ctx, ff);
    ff = modules::LinearModule(binding::linear_config(config.intermediate_size, hidden, true))
             .build(ctx, ff, weights.fc_out);
    return modules::AddModule{}.build(ctx, x, ff);
}

}  // namespace

struct MossDepthTransformer::Impl {
    std::shared_ptr<const MossTTSLocalAssets> assets;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    DepthWeights weights;

    // Persistent forward graphs, one per step count the generation loop uses
    // (steps = 1..num_codebooks). The generator calls forward() 12 times per frame;
    // rebuilding and reallocating the graph on every call was almost entirely
    // host-side overhead, so the graphs are built once here and reused. Positions
    // and the causal mask only depend on the step count, so they are uploaded once
    // at build time; only the input embeddings change per call.
    struct StepPlan {
        ggml_tensor * embeds = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * mask = nullptr;
        ggml_tensor * hidden = nullptr;
        ggml_cgraph * graph = nullptr;
    };
    std::unique_ptr<ggml_context, GgmlContextDeleter> plan_ctx;
    ggml_backend_buffer_t plan_buffer = nullptr;
    std::vector<StepPlan> plans;  // plans[steps - 1]
    double step_input_upload_ms = 0.0;
    double step_graph_compute_ms = 0.0;
    double step_output_read_ms = 0.0;
    int64_t step_calls = 0;
    double slow_graph_build_ms = 0.0;
    double slow_input_upload_ms = 0.0;
    double slow_graph_compute_ms = 0.0;
    double slow_output_read_ms = 0.0;
    int64_t slow_calls = 0;

    ~Impl() {
        for (const auto & plan : plans) {
            if (plan.graph != nullptr && backend != nullptr) {
                core::release_backend_graph_resources(backend, plan.graph);
            }
        }
        if (plan_buffer != nullptr) {
            ggml_backend_buffer_free(plan_buffer);
        }
    }

    void build_step_plans(int64_t max_steps);
};

void MossDepthTransformer::Impl::build_step_plans(int64_t max_steps) {
    const auto & config = assets->config.local;
    const int64_t hidden = config.hidden_size;

    ggml_init_params params{graph_arena_bytes, nullptr, true};
    plan_ctx.reset(ggml_init(params));
    if (plan_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS-TTS-Local depth plan context");
    }
    core::ModuleBuildContext ctx{plan_ctx.get(), "moss_tts_local.depth.step", backend_type};

    plans.reserve(static_cast<size_t>(max_steps));
    for (int64_t steps = 1; steps <= max_steps; ++steps) {
        StepPlan plan;
        auto embeds_input =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, hidden}));
        ggml_set_input(embeds_input.tensor);
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(positions.tensor);
        auto attention_mask =
            core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, steps}));
        ggml_set_input(attention_mask.tensor);

        auto x = local_layer(ctx, embeds_input, positions, weights, config, attention_mask);
        x = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, x, weights.ln_f);
        x = core::ensure_backend_addressable_layout(ctx, x);
        auto hidden_states = modules::ReshapeModule({
            core::TensorShape::from_dims({steps, hidden}),
        }).build(ctx, x);
        hidden_states = core::ensure_backend_addressable_layout(ctx, hidden_states);
        ggml_set_output(hidden_states.tensor);

        plan.graph = ggml_new_graph_custom(plan_ctx.get(), 2048, false);
        ggml_build_forward_expand(plan.graph, hidden_states.tensor);
        plan.embeds = embeds_input.tensor;
        plan.positions = positions.tensor;
        plan.mask = attention_mask.tensor;
        plan.hidden = hidden_states.tensor;
        plans.push_back(plan);
    }

    plan_buffer = ggml_backend_alloc_ctx_tensors(plan_ctx.get(), backend);
    if (plan_buffer == nullptr) {
        throw std::runtime_error("failed to allocate MOSS-TTS-Local depth step graphs");
    }

    for (int64_t steps = 1; steps <= max_steps; ++steps) {
        const auto & plan = plans[static_cast<size_t>(steps - 1)];
        std::vector<int32_t> position_host(static_cast<size_t>(steps));
#ifdef _OPENMP
#pragma omp parallel for if(steps >= 4096)
#endif
        for (int64_t i = 0; i < steps; ++i) {
            position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(plan.positions, position_host.data(), 0, position_host.size() * sizeof(int32_t));

        std::vector<ggml_fp16_t> mask_host(
            static_cast<size_t>(steps * steps),
            ggml_fp32_to_fp16(kMaskedAttentionBias));
#ifdef _OPENMP
#pragma omp parallel for if(steps * steps >= 4096)
#endif
        for (int64_t q = 0; q < steps; ++q) {
            for (int64_t k = 0; k <= q; ++k) {
                mask_host[static_cast<size_t>(q * steps + k)] = ggml_fp32_to_fp16(0.0F);
            }
        }
        ggml_backend_tensor_set(plan.mask, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));
    }
}

MossDepthTransformer::MossDepthTransformer(
    std::shared_ptr<const MossTTSLocalAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t weight_context_bytes)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local depth transformer requires assets");
    }
    if (assets->model_weights == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local depth transformer requires model weights");
    }
    if (assets->config.local.num_layers != 1) {
        throw std::runtime_error("MOSS-TTS-Local depth transformer expects a single local_transformer layer");
    }
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local depth transformer backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->weights = load_depth_weights(*assets, impl_->backend, impl_->backend_type, weight_context_bytes);
    impl_->assets = std::move(assets);
    // The generation loop only ever runs the local transformer over 1..num_codebooks
    // positions, so every graph it needs can be prebuilt up front.
    if (impl_->assets->config.num_codebooks > 0) {
        impl_->build_step_plans(impl_->assets->config.num_codebooks);
    }
}

MossDepthTransformer::~MossDepthTransformer() = default;

int64_t MossDepthTransformer::hidden_size() const noexcept {
    return impl_->assets->config.local.hidden_size;
}

std::vector<float> MossDepthTransformer::forward(const std::vector<float> & inputs_embeds, int64_t steps) const {
    const auto & config = impl_->assets->config.local;
    const int64_t hidden = config.hidden_size;
    if (steps <= 0) {
        throw std::runtime_error("MOSS-TTS-Local depth transformer forward requires a non-empty prefix");
    }
    if (static_cast<int64_t>(inputs_embeds.size()) != steps * hidden) {
        throw std::runtime_error("MOSS-TTS-Local depth transformer input size does not match [steps, hidden]");
    }

    // Fast path: reuse the prebuilt graph for this step count (positions and mask are
    // already resident); only the embeddings are uploaded per call.
    if (steps <= static_cast<int64_t>(impl_->plans.size())) {
        const auto & plan = impl_->plans[static_cast<size_t>(steps - 1)];
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(plan.embeds, inputs_embeds.data(), 0, inputs_embeds.size() * sizeof(float));
        impl_->step_input_upload_ms += engine::debug::elapsed_ms(timing_start);
        timing_start = Clock::now();
        core::set_backend_threads(impl_->backend, impl_->threads);
        const ggml_status status =
            core::compute_backend_graph(impl_->backend, plan.graph, nullptr, "MOSS-TTS-Local depth step");
        ggml_backend_synchronize(impl_->backend);
        impl_->step_graph_compute_ms += engine::debug::elapsed_ms(timing_start);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS-TTS-Local depth step graph compute failed");
        }
        std::vector<float> last_hidden(static_cast<size_t>(hidden));
        timing_start = Clock::now();
        ggml_backend_tensor_get(
            plan.hidden,
            last_hidden.data(),
            static_cast<size_t>((steps - 1) * hidden) * sizeof(float),
            static_cast<size_t>(hidden) * sizeof(float));
        impl_->step_output_read_ms += engine::debug::elapsed_ms(timing_start);
        ++impl_->step_calls;
        return last_hidden;
    }

    // Slow path for step counts outside the prebuilt range: build a one-off graph.
    const auto & weights = impl_->weights;

    auto timing_start = Clock::now();
    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS-TTS-Local depth graph context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss_tts_local.depth.forward", impl_->backend_type};

    auto embeds_input =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, hidden}));
    ggml_set_input(embeds_input.tensor);
    auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
    ggml_set_input(positions.tensor);
    auto attention_mask =
        core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, steps}));
    ggml_set_input(attention_mask.tensor);

    auto x = local_layer(ctx, embeds_input, positions, weights, config, attention_mask);
    x = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, x, weights.ln_f);
    x = core::ensure_backend_addressable_layout(ctx, x);
    auto hidden_states = modules::ReshapeModule({
        core::TensorShape::from_dims({steps, hidden}),
    }).build(ctx, x);
    hidden_states = core::ensure_backend_addressable_layout(ctx, hidden_states);
    ggml_set_output(hidden_states.tensor);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 8192, false);
    ggml_build_forward_expand(graph, hidden_states.tensor);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS-TTS-Local depth forward graph");
    }
    impl_->slow_graph_build_ms += engine::debug::elapsed_ms(timing_start);

    timing_start = Clock::now();
    ggml_backend_tensor_set(
        embeds_input.tensor, inputs_embeds.data(), 0, inputs_embeds.size() * sizeof(float));

    std::vector<int32_t> position_host(static_cast<size_t>(steps));
#ifdef _OPENMP
#pragma omp parallel for if(steps >= 4096)
#endif
    for (int64_t i = 0; i < steps; ++i) {
        position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions.tensor, position_host.data(), 0, position_host.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask_host(
        static_cast<size_t>(steps * steps),
        ggml_fp32_to_fp16(kMaskedAttentionBias));
#ifdef _OPENMP
#pragma omp parallel for if(steps * steps >= 4096)
#endif
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = 0; k <= q; ++k) {
            mask_host[static_cast<size_t>(q * steps + k)] = ggml_fp32_to_fp16(0.0F);
        }
    }
    ggml_backend_tensor_set(attention_mask.tensor, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));
    impl_->slow_input_upload_ms += engine::debug::elapsed_ms(timing_start);

    timing_start = Clock::now();
    core::set_backend_threads(impl_->backend, impl_->threads);
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    impl_->slow_graph_compute_ms += engine::debug::elapsed_ms(timing_start);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS-TTS-Local depth forward graph compute failed");
    }

    std::vector<float> last_hidden(static_cast<size_t>(hidden));
    timing_start = Clock::now();
    ggml_backend_tensor_get(
        hidden_states.tensor,
        last_hidden.data(),
        static_cast<size_t>((steps - 1) * hidden) * sizeof(float),
        static_cast<size_t>(hidden) * sizeof(float));
    impl_->slow_output_read_ms += engine::debug::elapsed_ms(timing_start);
    ggml_gallocr_free(gallocr);
    ++impl_->slow_calls;
    return last_hidden;
}

void MossDepthTransformer::reset_timing() const {
    auto & impl = *impl_;
    impl.step_input_upload_ms = 0.0;
    impl.step_graph_compute_ms = 0.0;
    impl.step_output_read_ms = 0.0;
    impl.step_calls = 0;
    impl.slow_graph_build_ms = 0.0;
    impl.slow_input_upload_ms = 0.0;
    impl.slow_graph_compute_ms = 0.0;
    impl.slow_output_read_ms = 0.0;
    impl.slow_calls = 0;
}

void MossDepthTransformer::log_timing() const {
    const auto & impl = *impl_;
    engine::debug::timing_log_scalar("moss_tts_local.depth.step.input_upload_ms", impl.step_input_upload_ms);
    engine::debug::timing_log_scalar("moss_tts_local.depth.step.graph.compute_ms", impl.step_graph_compute_ms);
    engine::debug::timing_log_scalar("moss_tts_local.depth.step.output_read_ms", impl.step_output_read_ms);
    engine::debug::trace_log_scalar("moss_tts_local.depth.step.calls", impl.step_calls);
    engine::debug::timing_log_scalar("moss_tts_local.depth.forward.graph.build_ms", impl.slow_graph_build_ms);
    engine::debug::timing_log_scalar("moss_tts_local.depth.forward.input_upload_ms", impl.slow_input_upload_ms);
    engine::debug::timing_log_scalar("moss_tts_local.depth.forward.graph.compute_ms", impl.slow_graph_compute_ms);
    engine::debug::timing_log_scalar("moss_tts_local.depth.forward.output_read_ms", impl.slow_output_read_ms);
    engine::debug::trace_log_scalar("moss_tts_local.depth.forward.calls", impl.slow_calls);
}

}  // namespace engine::models::moss_tts_local
