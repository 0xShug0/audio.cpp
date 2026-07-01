#include "engine/models/moss_tts_local/backbone.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
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

struct BackboneLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct BackboneWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue embed_tokens;
    std::vector<BackboneLayerWeights> layers;
    core::TensorValue norm;
};

void validate_weight_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "MOSS-TTS-Local backbone weight_type supports only native, f32, f16, bf16, and q8_0");
    }
}

BackboneWeights load_backbone_weights(
    const MossTTSLocalAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    validate_weight_storage_type(storage_type);
    const auto & config = assets.config.backbone;
    const auto & source = *assets.model_weights;
    BackboneWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "moss_tts_local.backbone.weights",
        weight_context_bytes);
    weights.embed_tokens = weights.store->load_tensor(
        source,
        "transformer.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    const int64_t dim = config.head_dim;
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer);
        BackboneLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.q_proj.weight",
            storage_type,
            {config.num_attention_heads * dim, config.hidden_size});
        w.k_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.k_proj.weight",
            storage_type,
            {config.num_key_value_heads * dim, config.hidden_size});
        w.v_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.v_proj.weight",
            storage_type,
            {config.num_key_value_heads * dim, config.hidden_size});
        w.o_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            storage_type,
            {config.hidden_size, config.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(
            source,
            prefix + ".post_attention_layernorm.weight",
            {config.hidden_size});
        w.gate_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.gate_proj.weight",
            storage_type,
            {config.intermediate_size, config.hidden_size});
        w.up_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.up_proj.weight",
            storage_type,
            {config.intermediate_size, config.hidden_size});
        w.down_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.down_proj.weight",
            storage_type,
            {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "transformer.norm.weight", {config.hidden_size});
    weights.store->upload();
    return weights;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
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

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    auto contiguous = ensure_contiguous(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = ensure_contiguous(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
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

core::TensorValue decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const BackboneLayerWeights & weights,
    const MossBackboneConfig & config,
    const core::TensorValue & attention_mask) {
    const int64_t dim = config.head_dim;
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const modules::LinearModule q_proj(
        binding::linear_config(config.hidden_size, config.num_attention_heads * dim, false));
    const modules::LinearModule k_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, false));
    const modules::LinearModule v_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, false));
    const modules::LinearModule o_proj(
        binding::linear_config(config.num_attention_heads * dim, config.hidden_size, false));
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.rms_norm_eps, true, false});
    auto x_norm = hidden_norm.build(ctx, input, binding::norm_data(ctx, weights.input_norm));
    auto q = q_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.q_proj));
    auto k = k_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.k_proj));
    auto v = v_proj.build(ctx, x_norm, binding::linear_data(ctx, weights.v_proj));
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), binding::norm_data(ctx, weights.q_norm));
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), binding::norm_data(ctx, weights.k_norm));
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, binding::linear_data(ctx, weights.o_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(ctx, weights.post_norm));
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, config.intermediate_size, false))
                    .build(ctx, ff_in, binding::linear_data(ctx, weights.gate_proj));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, config.intermediate_size, false))
                  .build(ctx, ff_in, binding::linear_data(ctx, weights.up_proj));
    auto ff = modules::LinearModule(
                  binding::linear_config(config.intermediate_size, config.hidden_size, false))
                  .build(ctx, modules::MulModule{}.build(ctx, gate, up), binding::linear_data(ctx, weights.down_proj));
    return modules::AddModule{}.build(ctx, x, ff);
}

}  // namespace

struct MossBackboneRuntime::Impl {
    std::shared_ptr<const MossTTSLocalAssets> assets;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    size_t graph_arena_bytes = 0;
    BackboneWeights weights;
};

MossBackboneRuntime::MossBackboneRuntime(
    std::shared_ptr<const MossTTSLocalAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local backbone requires assets");
    }
    if (assets->model_weights == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local backbone requires model weights");
    }
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local backbone backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->weights = load_backbone_weights(
        *assets,
        impl_->backend,
        impl_->backend_type,
        weight_context_bytes,
        weight_storage_type);
    impl_->assets = std::move(assets);
}

MossBackboneRuntime::~MossBackboneRuntime() = default;

int64_t MossBackboneRuntime::hidden_size() const noexcept {
    return impl_->assets->config.backbone.hidden_size;
}

std::vector<float> MossBackboneRuntime::forward_prefill(const std::vector<int32_t> & token_ids) const {
    return run_prefill(token_ids, nullptr);
}

std::vector<float> MossBackboneRuntime::forward_prefill_fused(
    const std::vector<int32_t> & token_ids,
    const std::vector<float> & audio_bias) const {
    const int64_t steps = static_cast<int64_t>(token_ids.size());
    const int64_t hidden = impl_->assets->config.backbone.hidden_size;
    if (static_cast<int64_t>(audio_bias.size()) != steps * hidden) {
        throw std::runtime_error("MOSS-TTS-Local backbone audio bias size does not match [steps, hidden]");
    }
    return run_prefill(token_ids, audio_bias.data());
}

std::vector<float> MossBackboneRuntime::run_prefill(
    const std::vector<int32_t> & token_ids,
    const float * audio_bias) const {
    const int64_t steps = static_cast<int64_t>(token_ids.size());
    if (steps <= 0) {
        throw std::runtime_error("MOSS-TTS-Local backbone prefill requires a non-empty token sequence");
    }
    const auto & config = impl_->assets->config.backbone;
    const auto & weights = impl_->weights;

    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS-TTS-Local backbone graph context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss_tts_local.backbone.forward", impl_->backend_type};

    auto token_input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, steps}));
    ggml_set_input(token_input.tensor);
    auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
    ggml_set_input(positions.tensor);
    auto attention_mask =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
    ggml_set_input(attention_mask.tensor);
    core::TensorValue audio_bias_input;
    if (audio_bias != nullptr) {
        audio_bias_input =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.hidden_size}));
        ggml_set_input(audio_bias_input.tensor);
    }

    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                 .build(ctx, token_input, weights.embed_tokens);
    if (audio_bias != nullptr) {
        x = modules::AddModule{}.build(ctx, x, audio_bias_input);
    }
    for (const auto & layer : weights.layers) {
        x = decoder_layer(ctx, x, positions, layer, config, attention_mask);
    }
    x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
            .build(ctx, x, binding::norm_data(ctx, weights.norm));
    x = ensure_contiguous(ctx, x);
    auto hidden = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({steps, config.hidden_size}));
    hidden = ensure_contiguous(ctx, hidden);
    ggml_set_output(hidden.tensor);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 65536, false);
    ggml_build_forward_expand(graph, hidden.tensor);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS-TTS-Local backbone forward graph");
    }

    ggml_backend_tensor_set(token_input.tensor, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));

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

    if (audio_bias != nullptr) {
        ggml_backend_tensor_set(
            audio_bias_input.tensor,
            audio_bias,
            0,
            static_cast<size_t>(steps * config.hidden_size) * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS-TTS-Local backbone forward graph compute failed");
    }

    std::vector<float> hidden_states(static_cast<size_t>(steps * config.hidden_size));
    ggml_backend_tensor_get(hidden.tensor, hidden_states.data(), 0, hidden_states.size() * sizeof(float));
    ggml_gallocr_free(gallocr);
    return hidden_states;
}

}  // namespace engine::models::moss_tts_local
