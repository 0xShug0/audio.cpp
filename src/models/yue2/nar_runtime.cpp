#include "engine/models/yue2/nar_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace engine::models::yue2 {
namespace {

namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct Yue2NarWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    engine::modules::LinearWeights vae2llm;
    engine::modules::LinearWeights time0;
    engine::modules::LinearWeights time2;
    core::TensorValue latent_pos_embed;
    engine::modules::QwenDecoderStackWeights nar_stack;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights llm2vae;
};

engine::modules::QwenDecoderLayerWeights load_nar_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Yue2ModelConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    engine::modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.q_proj.weight",
        storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".nar_self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".nar_pre_mlp_layernorm", config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".nar_mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".nar_mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".nar_mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

std::shared_ptr<const Yue2NarWeights> load_nar_weights(
    const Yue2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto total_start = Clock::now();
    auto weights = std::make_shared<Yue2NarWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "yue2.nar.weights",
        weight_context_bytes);
    const auto & source = *assets.model_weights;
    const auto & config = assets.config.model;
    const auto bind_start = Clock::now();
    weights->vae2llm = binding::linear_from_source(
        *weights->store,
        source,
        "vae2llm",
        storage_type,
        config.hidden_size,
        config.latent_dim,
        true);
    weights->time0 = binding::linear_from_source(
        *weights->store,
        source,
        "time_embedder.mlp.0",
        storage_type,
        config.hidden_size,
        256,
        true);
    weights->time2 = binding::linear_from_source(
        *weights->store,
        source,
        "time_embedder.mlp.2",
        storage_type,
        config.hidden_size,
        config.hidden_size,
        true);
    weights->latent_pos_embed = weights->store->load_tensor(
        source,
        "latent_pos_embed.pe",
        storage_type,
        {config.max_latent_frames, config.hidden_size});
    weights->nar_stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights->nar_stack.layers.push_back(load_nar_layer(*weights->store, source, config, storage_type, layer));
    }
    weights->final_norm = binding::norm_weight_from_source(*weights->store, source, "model.norm", config.hidden_size);
    weights->llm2vae = binding::linear_from_source(
        *weights->store,
        source,
        "llm2vae",
        storage_type,
        config.latent_dim,
        config.hidden_size,
        true);
    engine::debug::timing_log_scalar("yue2.nar.weights_bind_ms", engine::debug::elapsed_ms(bind_start));
    const auto upload_start = Clock::now();
    weights->store->upload();
    engine::debug::timing_log_scalar("yue2.nar.weights_upload_ms", engine::debug::elapsed_ms(upload_start));
    engine::debug::timing_log_scalar("yue2.nar.weights_total_ms", engine::debug::elapsed_ms(total_start));
    return weights;
}

std::vector<std::pair<int64_t, int64_t>> chunk_ranges(int64_t frames, int64_t prefix_tokens, int64_t context) {
    const int64_t chunk = std::min((context - prefix_tokens - 3) / 2, context);
    if (frames < 1 || chunk < 1) {
        throw std::runtime_error("Yue2 NAR chunk leaves no acoustic context");
    }
    std::vector<std::pair<int64_t, int64_t>> out;
    for (int64_t begin = 0; begin < frames; begin += chunk) {
        out.push_back({begin, std::min(begin + chunk, frames)});
    }
    return out;
}

std::vector<float> timestep_features(float shifted_t, int64_t rows) {
    constexpr int64_t kDim = 256;
    std::vector<float> out(static_cast<size_t>(rows * kDim), 0.0F);
    const int64_t half = kDim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
        const float value = shifted_t * freq;
        for (int64_t row = 0; row < rows; ++row) {
            out[static_cast<size_t>(row * kDim + i)] = std::cos(value);
            out[static_cast<size_t>(row * kDim + half + i)] = std::sin(value);
        }
    }
    return out;
}

float shifted_t_value(float raw_t, float shift) {
    const float sigmoid = 1.0F / (1.0F + std::exp(-raw_t));
    return shift * sigmoid / (1.0F + (shift - 1.0F) * sigmoid);
}

float logit_clamped(float t) {
    const float safe = std::min(std::max(t, 1.0e-7F), 1.0F - 1.0e-7F);
    return std::min(std::max(std::log(safe / (1.0F - safe)), -20.0F), 20.0F);
}

struct QKVParts {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
};

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

QKVParts build_qkv_part(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const engine::modules::QwenDecoderLayerWeights & weights,
    const Yue2ModelConfig & config) {
    auto q = engine::modules::LinearModule({config.hidden_size, config.attention_heads * config.head_dim, false})
                 .build(ctx, input, {weights.self_attention.q_weight, std::nullopt});
    auto k = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                 .build(ctx, input, {weights.self_attention.k_weight, std::nullopt});
    auto v = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                 .build(ctx, input, {weights.self_attention.v_weight, std::nullopt});
    q = reshape_heads(ctx, q, config.attention_heads, config.head_dim);
    k = reshape_heads(ctx, k, config.kv_heads, config.head_dim);
    q = engine::modules::RMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
    k = engine::modules::RMSNormModule({config.head_dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
    return {q, k, reshape_heads(ctx, v, config.kv_heads, config.head_dim)};
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = engine::modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = core::ensure_backend_addressable_layout(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue mixed_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    const core::TensorValue * attention_mask,
    const Yue2ModelConfig & config,
    core::BackendType backend_type) {
    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    if (backend_type != core::BackendType::Cpu) {
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_heads.tensor,
            k_heads.tensor,
            v_heads.tensor,
            attention_mask != nullptr ? attention_mask->tensor : nullptr,
            1.0F / std::sqrt(static_cast<float>(config.head_dim)),
            0.0F,
            0.0F);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        return core::wrap_tensor(
            flash,
            core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], config.head_dim}),
            GGML_TYPE_F32);
    }

    const int64_t repeats = config.attention_heads / config.kv_heads;
    k_heads = repeat_kv_heads(ctx, k_heads, repeats);
    v_heads = repeat_kv_heads(ctx, v_heads, repeats);
    auto scores = engine::modules::MatMulModule{}.build(
        ctx,
        q_heads,
        engine::modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            attention_mask != nullptr ? attention_mask->tensor : nullptr,
            1.0F / std::sqrt(static_cast<float>(config.head_dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = engine::modules::MatMulModule{}.build(ctx, attn, v_heads);
    return engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

core::TensorValue build_mlp_part(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const engine::modules::QwenMLPWeights & weights,
    const Yue2ModelConfig & config) {
    auto gate = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, input, {weights.gate_proj.weight, std::nullopt});
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    auto up = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, input, {weights.up_proj.weight, std::nullopt});
    auto gated = engine::modules::MulModule{}.build(ctx, gate, up);
    return engine::modules::LinearModule({config.intermediate_size, config.hidden_size, false})
        .build(ctx, gated, {weights.down_proj.weight, std::nullopt});
}

core::TensorValue build_cached_nar_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const engine::modules::QwenDecoderLayerWeights & nar_weights,
    const core::TensorValue & ar_key,
    const core::TensorValue & ar_value,
    const Yue2ModelConfig & config,
    core::BackendType backend_type) {
    auto norm = engine::modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, nar_weights.input_norm);
    auto qkv = build_qkv_part(ctx, norm, nar_weights, config);
    qkv.q = engine::modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta})
                .build(ctx, qkv.q, positions);
    qkv.k = engine::modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta})
                .build(ctx, qkv.k, positions);
    qkv.k = core::ensure_backend_addressable_layout(ctx, qkv.k);
    qkv.v = core::ensure_backend_addressable_layout(ctx, qkv.v);
    qkv.k = core::wrap_tensor(ggml_cast(ctx.ggml, qkv.k.tensor, GGML_TYPE_F16), qkv.k.shape, GGML_TYPE_F16);
    qkv.v = core::wrap_tensor(ggml_cast(ctx.ggml, qkv.v.tensor, GGML_TYPE_F16), qkv.v.shape, GGML_TYPE_F16);
    auto k = engine::modules::ConcatModule({1}).build(ctx, ar_key, qkv.k);
    auto v = engine::modules::ConcatModule({1}).build(ctx, ar_value, qkv.v);
    auto context = mixed_attention(ctx, qkv.q, k, v, nullptr, config, backend_type);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.attention_heads * config.head_dim}));
    auto attn = engine::modules::LinearModule({config.attention_heads * config.head_dim, config.hidden_size, false})
                    .build(ctx, context, {nar_weights.self_attention.out_weight, std::nullopt});
    auto x = engine::modules::AddModule{}.build(ctx, input, attn);
    norm = engine::modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
               .build(ctx, x, nar_weights.post_norm);
    return engine::modules::AddModule{}.build(ctx, x, build_mlp_part(ctx, norm, nar_weights.mlp, config));
}

}  // namespace

struct Yue2NarRuntime::Impl {
    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t graph_arena_bytes)
        : execution(execution),
          assets(std::move(assets)),
          graph_arena_bytes(graph_arena_bytes) {
        if (!this->assets) {
            throw std::runtime_error("Yue2 NAR runtime requires assets");
        }
        const auto start = Clock::now();
        weights = load_nar_weights(*this->assets, execution, weight_context_bytes, weight_type);
        engine::debug::timing_log_scalar("yue2.nar.init_total_ms", engine::debug::elapsed_ms(start));
    }

    struct HostVelocityTiming {
        double pad_ms = 0.0;
        double timestep_ms = 0.0;
    };

    struct Graph {
        Graph(
            Impl & owner,
            int64_t frames,
            const Yue2ArDevicePrefixState & ar_state)
            : owner(&owner),
              frames(frames),
              ar_length(ar_state.current_end) {
            const auto & config = owner.assets->config.model;
            if (frames <= 0 || ar_length <= 0) {
                throw std::runtime_error("Yue2 NAR graph requires positive shapes");
            }
            if (static_cast<int64_t>(ar_state.keys.size()) != config.layers ||
                static_cast<int64_t>(ar_state.values.size()) != config.layers) {
                throw std::runtime_error("Yue2 NAR prefix cache layer count mismatch");
            }
            const auto total_start = Clock::now();
            nar_length = frames + 2;
            total_length = ar_length + nar_length;
            ggml_init_params input_params{
                ggml_tensor_overhead() * static_cast<size_t>(3 + config.layers * 2),
                nullptr,
                true};
            input_ctx.reset(ggml_init(input_params));
            ggml_init_params graph_params{owner.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(graph_params));
            if (!input_ctx || !ctx) {
                throw std::runtime_error("failed to initialize Yue2 NAR graph context");
            }
            core::ModuleBuildContext input_build{input_ctx.get(), "yue2.nar.input", owner.execution.backend_type()};
            core::ModuleBuildContext build{ctx.get(), "yue2.nar", owner.execution.backend_type()};
            const auto build_start = Clock::now();
            state = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({1, nar_length, config.latent_dim}));
            time = core::make_tensor(input_build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 256}));
            positions = core::make_tensor(input_build, GGML_TYPE_I32, core::TensorShape::from_dims({nar_length}));
            ggml_set_input(state.tensor);
            ggml_set_input(time.tensor);
            ggml_set_input(positions.tensor);
            ar_keys.reserve(static_cast<size_t>(config.layers));
            ar_values.reserve(static_cast<size_t>(config.layers));
            for (int64_t layer = 0; layer < config.layers; ++layer) {
                const auto & key = ar_state.keys[static_cast<size_t>(layer)];
                const auto & value = ar_state.values[static_cast<size_t>(layer)];
                if (key.shape.rank != 4 || value.shape.rank != 4 ||
                    key.shape.dims[0] != 1 || value.shape.dims[0] != 1 ||
                    key.shape.dims[1] != ar_length || value.shape.dims[1] != ar_length ||
                    key.shape.dims[2] != config.kv_heads || value.shape.dims[2] != config.kv_heads ||
                    key.shape.dims[3] != config.head_dim || value.shape.dims[3] != config.head_dim ||
                    key.type != GGML_TYPE_F16 || value.type != GGML_TYPE_F16) {
                    throw std::runtime_error("Yue2 NAR prefix cache tensor shape mismatch");
                }
                ar_keys.push_back(key);
                ar_values.push_back(value);
            }
            auto nar_hidden = engine::modules::LinearModule({config.latent_dim, config.hidden_size, true})
                                  .build(build, state, owner.weights->vae2llm);
            auto time_hidden = engine::modules::LinearModule({256, config.hidden_size, true})
                                   .build(build, time, owner.weights->time0);
            time_hidden = engine::modules::SiluModule{}.build(build, time_hidden);
            time_hidden = engine::modules::LinearModule({config.hidden_size, config.hidden_size, true})
                              .build(build, time_hidden, owner.weights->time2);
            time_hidden = core::reshape_tensor(build, time_hidden, core::TensorShape::from_dims({1, 1, config.hidden_size}));
            time_hidden = engine::modules::RepeatModule(
                {core::TensorShape::from_dims({1, nar_length, config.hidden_size})})
                              .build(build, time_hidden);
            auto pos = engine::modules::SliceModule({0, 0, nar_length}).build(build, owner.weights->latent_pos_embed);
            pos = core::reshape_tensor(build, pos, core::TensorShape::from_dims({1, nar_length, config.hidden_size}));
            if (pos.type != GGML_TYPE_F32) {
                pos = core::wrap_tensor(ggml_cast(build.ggml, pos.tensor, GGML_TYPE_F32), pos.shape, GGML_TYPE_F32);
            }
            nar_hidden = engine::modules::AddModule{}.build(build, nar_hidden, time_hidden);
            nar_hidden = engine::modules::AddModule{}.build(build, nar_hidden, pos);
            auto hidden = nar_hidden;

            for (int64_t layer = 0; layer < config.layers; ++layer) {
                hidden = build_cached_nar_layer(
                    build,
                    hidden,
                    positions,
                    owner.weights->nar_stack.layers[static_cast<size_t>(layer)],
                    ar_keys[static_cast<size_t>(layer)],
                    ar_values[static_cast<size_t>(layer)],
                    config,
                    owner.execution.backend_type());
            }
            hidden = engine::modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                         .build(build, hidden, owner.weights->final_norm);
            auto latent = engine::modules::LinearModule({config.hidden_size, config.latent_dim, true})
                              .build(build, hidden, owner.weights->llm2vae);
            auto content = engine::modules::SliceModule({1, 1, frames}).build(build, latent);
            output = core::ensure_backend_addressable_layout(build, content);
            ggml_set_output(output.tensor);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_build_forward_expand(graph, output.tensor);
            engine::debug::timing_log_scalar("yue2.nar.graph.build_ms", engine::debug::elapsed_ms(build_start));
            const auto alloc_start = Clock::now();
            input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), owner.execution.backend());
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.execution.backend()));
            if (input_buffer == nullptr || gallocr == nullptr ||
                !ggml_gallocr_reserve(gallocr, graph) ||
                !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("failed to allocate Yue2 NAR graph");
            }
            engine::debug::timing_log_scalar("yue2.nar.graph.alloc_ms", engine::debug::elapsed_ms(alloc_start));
            std::vector<int32_t> pos_values(static_cast<size_t>(nar_length));
            for (int64_t i = 0; i < nar_length; ++i) {
                pos_values[static_cast<size_t>(i)] = static_cast<int32_t>(ar_length + i);
            }
            ggml_backend_tensor_set(positions.tensor, pos_values.data(), 0, pos_values.size() * sizeof(int32_t));
            engine::debug::timing_log_scalar("yue2.nar.graph.static_upload_ms", 0.0);
            engine::debug::timing_log_scalar("yue2.nar.graph.frames", frames);
            engine::debug::timing_log_scalar("yue2.nar.graph.ar_tokens", ar_length);
            engine::debug::timing_log_scalar("yue2.nar.graph.nar_tokens", nar_length);
            engine::debug::timing_log_scalar("yue2.nar.graph.total_ms", engine::debug::elapsed_ms(total_start));
        }

        ~Graph() {
            core::release_backend_graph_resources(owner->execution.backend(), graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (input_buffer) {
                ggml_backend_buffer_free(input_buffer);
            }
        }

        bool matches(int64_t f, int64_t ar) const noexcept {
            return frames == f && ar_length == ar;
        }

        std::vector<float> run(
            const std::vector<float> & padded_state,
            const std::vector<float> & time_features) {
            const auto & config = owner->assets->config.model;
            if (static_cast<int64_t>(padded_state.size()) != nar_length * config.latent_dim ||
                static_cast<int64_t>(time_features.size()) != 256) {
                throw std::runtime_error("Yue2 NAR step input shape mismatch");
            }
            const auto upload_start = Clock::now();
            core::write_tensor_f32(state, padded_state);
            core::write_tensor_f32(time, time_features);
            input_upload_ms += engine::debug::elapsed_ms(upload_start);
            const auto compute_start = Clock::now();
            const auto status = core::compute_backend_graph(owner->execution.backend(), graph, nullptr, "yue2.nar.velocity");
            ggml_backend_synchronize(owner->execution.backend());
            graph_compute_ms += engine::debug::elapsed_ms(compute_start);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Yue2 NAR graph compute failed");
            }
            std::vector<float> out;
            const auto read_start = Clock::now();
            core::read_tensor_f32_into(output.tensor, out);
            output_read_ms += engine::debug::elapsed_ms(read_start);
            ++runs;
            return out;
        }

        Impl * owner = nullptr;
        int64_t frames = 0;
        int64_t ar_length = 0;
        int64_t nar_length = 0;
        int64_t total_length = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t input_buffer = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue state;
        core::TensorValue time;
        core::TensorValue positions;
        std::vector<core::TensorValue> ar_keys;
        std::vector<core::TensorValue> ar_values;
        core::TensorValue output;
        double input_upload_ms = 0.0;
        double graph_compute_ms = 0.0;
        double output_read_ms = 0.0;
        int64_t runs = 0;
    };

    std::vector<float> velocity(
        Graph & graph,
        const Yue2ArDevicePrefixState & ar_state,
        const std::vector<float> & state,
        float raw_t,
        HostVelocityTiming & host_timing) {
        const auto & config = assets->config.model;
        const int64_t frames = static_cast<int64_t>(state.size()) / config.latent_dim;
        if (frames <= 0 || frames * config.latent_dim != static_cast<int64_t>(state.size())) {
            throw std::runtime_error("Yue2 NAR state shape mismatch");
        }
        if (!graph.matches(frames, ar_state.current_end)) {
            throw std::runtime_error("Yue2 NAR graph shape changed during chunk solve");
        }
        const auto pad_start = Clock::now();
        std::vector<float> padded(static_cast<size_t>((frames + 2) * config.latent_dim), 0.0F);
        std::copy(state.begin(), state.end(), padded.begin() + static_cast<std::ptrdiff_t>(config.latent_dim));
        host_timing.pad_ms += engine::debug::elapsed_ms(pad_start);
        const auto shifted = shifted_t_value(raw_t, config.timestep_shift);
        const auto timestep_start = Clock::now();
        auto features = timestep_features(shifted, 1);
        host_timing.timestep_ms += engine::debug::elapsed_ms(timestep_start);
        return graph.run(padded, features);
    }

    std::vector<float> solve_chunk(
        const Yue2ArDevicePrefixState & ar_state,
        const std::vector<float> & noise,
        int64_t ode_steps) {
        auto state = noise;
        const auto total_start = Clock::now();
        const auto & config = assets->config.model;
        const int64_t frames = static_cast<int64_t>(state.size()) / config.latent_dim;
        graph = std::make_unique<Graph>(*this, frames, ar_state);
        auto & chunk_graph = *graph;
        const float dt = 1.0F / static_cast<float>(ode_steps);
        HostVelocityTiming host_timing;
        double host_update_ms = 0.0;
        for (int64_t step = 0; step < ode_steps; ++step) {
            const float t = 1.0F - static_cast<float>(step) * dt;
            const auto first = velocity(chunk_graph, ar_state, state, logit_clamped(t), host_timing);
            const auto mid_start = Clock::now();
            std::vector<float> mid(state.size(), 0.0F);
            for (size_t i = 0; i < state.size(); ++i) {
                mid[i] = state[i] - first[i] * (dt / 2.0F);
            }
            host_update_ms += engine::debug::elapsed_ms(mid_start);
            const auto second = velocity(chunk_graph, ar_state, mid, logit_clamped(t - dt / 2.0F), host_timing);
            const auto update_start = Clock::now();
            for (size_t i = 0; i < state.size(); ++i) {
                state[i] -= second[i] * dt;
            }
            host_update_ms += engine::debug::elapsed_ms(update_start);
        }
        engine::debug::timing_log_scalar("yue2.nar.chunk.frames", frames);
        engine::debug::timing_log_scalar("yue2.nar.chunk.ode_steps", ode_steps);
        engine::debug::timing_log_scalar("yue2.nar.chunk.velocity_runs", chunk_graph.runs);
        engine::debug::timing_log_scalar("yue2.nar.chunk.host_pad_ms", host_timing.pad_ms);
        engine::debug::timing_log_scalar("yue2.nar.chunk.host_timestep_ms", host_timing.timestep_ms);
        engine::debug::timing_log_scalar("yue2.nar.chunk.host_update_ms", host_update_ms);
        engine::debug::timing_log_scalar("yue2.nar.chunk.input_upload_ms", chunk_graph.input_upload_ms);
        engine::debug::timing_log_scalar("yue2.nar.chunk.graph_compute_ms", chunk_graph.graph_compute_ms);
        engine::debug::timing_log_scalar("yue2.nar.chunk.output_read_ms", chunk_graph.output_read_ms);
        engine::debug::timing_log_scalar("yue2.nar.chunk.total_ms", engine::debug::elapsed_ms(total_start));
        return state;
    }

    std::vector<float> synthesize(
        const std::vector<int32_t> & prefix,
        const std::vector<int32_t> & codec,
        const std::function<Yue2ArDevicePrefixState(const std::vector<int32_t> &)> & prefill_state,
        const std::vector<float> & noise,
        uint64_t seed,
        int64_t ode_steps,
        int64_t context) {
        const auto total_start = Clock::now();
        const auto & config = assets->config.model;
        const auto ranges = chunk_ranges(static_cast<int64_t>(codec.size()), static_cast<int64_t>(prefix.size()), context);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.prefix_tokens", prefix.size());
        engine::debug::timing_log_scalar("yue2.nar.synthesize.codec_tokens", codec.size());
        engine::debug::timing_log_scalar("yue2.nar.synthesize.chunks", ranges.size());
        engine::debug::timing_log_scalar("yue2.nar.synthesize.ode_steps", ode_steps);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.context", context);
        std::vector<float> full_noise(static_cast<size_t>(codec.size() * config.latent_dim), 0.0F);
        const auto noise_start = Clock::now();
        if (noise.empty()) {
            std::mt19937 rng(static_cast<uint32_t>(seed));
            std::normal_distribution<float> normal(0.0F, 1.0F);
            for (float & value : full_noise) {
                value = normal(rng);
            }
        } else {
            if (noise.size() != full_noise.size()) {
                throw std::runtime_error("Yue2 nar_noise_file frame count does not match semantic codec count");
            }
            full_noise = noise;
        }
        engine::debug::timing_log_scalar("yue2.nar.synthesize.noise_ms", engine::debug::elapsed_ms(noise_start));
        std::vector<float> out;
        out.reserve(full_noise.size());
        double token_build_ms = 0.0;
        double noise_slice_ms = 0.0;
        double prefill_ms = 0.0;
        double solve_ms = 0.0;
        double append_ms = 0.0;
        for (const auto & [begin, end] : ranges) {
            const auto token_start = Clock::now();
            std::vector<int32_t> ar_tokens = prefix;
            ar_tokens.reserve(prefix.size() + static_cast<size_t>(end - begin) + 1);
            for (int64_t i = begin; i < end; ++i) {
                ar_tokens.push_back(codec[static_cast<size_t>(i)] + kCodecOffset);
            }
            ar_tokens.push_back(kMusicEndToken);
            token_build_ms += engine::debug::elapsed_ms(token_start);
            const auto slice_start = Clock::now();
            const size_t begin_elem = static_cast<size_t>(begin * config.latent_dim);
            const size_t end_elem = static_cast<size_t>(end * config.latent_dim);
            std::vector<float> noise(end_elem - begin_elem);
            std::copy(full_noise.begin() + static_cast<std::ptrdiff_t>(begin_elem),
                      full_noise.begin() + static_cast<std::ptrdiff_t>(end_elem),
                      noise.begin());
            noise_slice_ms += engine::debug::elapsed_ms(slice_start);
            const auto prefill_start = Clock::now();
            auto ar_state = prefill_state(ar_tokens);
            prefill_ms += engine::debug::elapsed_ms(prefill_start);
            const auto solve_start = Clock::now();
            auto chunk = solve_chunk(ar_state, noise, ode_steps);
            solve_ms += engine::debug::elapsed_ms(solve_start);
            const auto append_start = Clock::now();
            out.insert(out.end(), chunk.begin(), chunk.end());
            append_ms += engine::debug::elapsed_ms(append_start);
        }
        engine::debug::timing_log_scalar("yue2.nar.synthesize.token_build_ms", token_build_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.noise_slice_ms", noise_slice_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.prefill_state_ms", prefill_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.solve_chunks_ms", solve_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.output_append_ms", append_ms);
        engine::debug::timing_log_scalar("yue2.nar.synthesize.output_latents", out.size());
        engine::debug::timing_log_scalar("yue2.nar.synthesize.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const Yue2Assets> assets;
    size_t graph_arena_bytes = 0;
    std::shared_ptr<const Yue2NarWeights> weights;
    std::unique_ptr<Graph> graph;
};

Yue2NarRuntime::Yue2NarRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const Yue2Assets> assets,
    assets::TensorStorageType weight_type,
    size_t weight_context_bytes,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          weight_type,
          weight_context_bytes,
          graph_arena_bytes)) {}

Yue2NarRuntime::~Yue2NarRuntime() = default;

std::vector<float> Yue2NarRuntime::synthesize(
    const std::vector<int32_t> & prefix,
    const std::vector<int32_t> & codec,
    const std::function<Yue2ArDevicePrefixState(const std::vector<int32_t> &)> & prefill_state,
    const std::vector<float> & noise,
    uint64_t seed,
    int64_t ode_steps,
    int64_t context) {
    return impl_->synthesize(prefix, codec, prefill_state, noise, seed, ode_steps, context);
}

void Yue2NarRuntime::release_runtime_graphs() {
    impl_->graph.reset();
}

}  // namespace engine::models::yue2
