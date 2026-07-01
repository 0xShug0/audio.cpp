#include "engine/models/moss_tts_local/depth_transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
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

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue split_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & fused,
    int64_t index,
    int64_t width) {
    return modules::SliceModule({2, index * width, width}).build(ctx, fused);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            attention_mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
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
    auto q = reshape_heads(ctx, split_projection(ctx, qkv, 0, hidden), heads, dim);
    auto k = reshape_heads(ctx, split_projection(ctx, qkv, 1, hidden), heads, dim);
    auto v = reshape_heads(ctx, split_projection(ctx, qkv, 2, hidden), heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, config.rope_base}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, config.rope_base}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden}));
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
    size_t graph_arena_bytes = 0;
    DepthWeights weights;
};

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
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->weights = load_depth_weights(*assets, impl_->backend, impl_->backend_type, weight_context_bytes);
    impl_->assets = std::move(assets);
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
    const auto & weights = impl_->weights;

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
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
    ggml_set_input(attention_mask.tensor);

    auto x = local_layer(ctx, embeds_input, positions, weights, config, attention_mask);
    x = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, x, weights.ln_f);
    x = ensure_contiguous(ctx, x);
    auto hidden_states = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({steps, hidden}));
    hidden_states = ensure_contiguous(ctx, hidden_states);
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

    ggml_backend_tensor_set(
        embeds_input.tensor, inputs_embeds.data(), 0, inputs_embeds.size() * sizeof(float));

    std::vector<int32_t> position_host(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions.tensor, position_host.data(), 0, position_host.size() * sizeof(int32_t));

    std::vector<float> mask_host(static_cast<size_t>(steps * steps), kMaskedAttentionBias);
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = 0; k <= q; ++k) {
            mask_host[static_cast<size_t>(q * steps + k)] = 0.0F;
        }
    }
    ggml_backend_tensor_set(attention_mask.tensor, mask_host.data(), 0, mask_host.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS-TTS-Local depth forward graph compute failed");
    }

    std::vector<float> last_hidden(static_cast<size_t>(hidden));
    ggml_backend_tensor_get(
        hidden_states.tensor,
        last_hidden.data(),
        static_cast<size_t>((steps - 1) * hidden) * sizeof(float),
        static_cast<size_t>(hidden) * sizeof(float));
    ggml_gallocr_free(gallocr);
    return last_hidden;
}

}  // namespace engine::models::moss_tts_local
