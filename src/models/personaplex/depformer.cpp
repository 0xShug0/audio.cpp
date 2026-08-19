#include "engine/models/personaplex/depformer.h"

#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::personaplex {
namespace sampling = engine::sampling;
namespace {

modules::NormWeights load_rms_alpha(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t hidden_size) {
    return {
        store.make_f32(core::TensorShape::from_dims({hidden_size}), source.require_f32(name, {1, 1, hidden_size})),
        std::nullopt,
    };
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::QwenDecoderLayerConfig depformer_layer_config(
    const PersonaPlexConfig & config,
    core::BackendType backend_type) {
    modules::QwenDecoderLayerConfig out;
    out.hidden_size = config.depformer.hidden_size;
    out.num_attention_heads = config.depformer.num_attention_heads;
    out.num_key_value_heads = config.depformer.num_attention_heads;
    out.head_dim = config.depformer.head_dim;
    out.intermediate_size = config.depformer.intermediate_size;
    out.rms_norm_eps = config.lm.rms_norm_eps;
    out.position_encoding = modules::QwenDecoderPositionEncoding::None;
    out.attention_precision = GGML_PREC_F32;
    out.projection_precision = GGML_PREC_DEFAULT;
    out.activation_cast.enabled = backend_type != core::BackendType::Vulkan;
    out.activation_cast.type = GGML_TYPE_BF16;
    out.activation_cast.after_input_norm = true;
    out.activation_cast.after_qkv_projection = true;
    out.activation_cast.after_attention = true;
    out.activation_cast.after_attention_output = true;
    out.activation_cast.after_residual = true;
    out.activation_cast.after_ffn_norm = true;
    out.activation_cast.after_mlp_projection = true;
    out.activation_cast.after_mlp_silu = true;
    out.activation_cast.after_mlp_mul = true;
    out.activation_cast.after_output = true;
    out.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.use_qk_norm = false;
    out.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    return out;
}

core::TensorValue view_linear_rows(
    ggml_context * ctx,
    const core::TensorValue & weight,
    int64_t row_offset,
    int64_t rows,
    int64_t cols,
    const char * label) {
    if (weight.shape.rank != 2 || weight.shape.dims[0] < row_offset + rows ||
        weight.shape.dims[1] != cols) {
        throw std::runtime_error(std::string("PersonaPlex depformer ") + label + " weight view is invalid");
    }
    const size_t row_stride = weight.tensor->nb[1];
    const size_t byte_offset = static_cast<size_t>(row_offset) * row_stride;
    return core::wrap_tensor(
        ggml_view_2d(ctx, weight.tensor, cols, rows, row_stride, byte_offset),
        core::TensorShape::from_dims({rows, cols}),
        weight.type);
}

modules::QwenDecoderLayerWeights depformer_step_layer_weights(
    ggml_context * ctx,
    const PersonaPlexDepformerLayerWeights & layer,
    int64_t step,
    const PersonaPlexConfig & config) {
    const int64_t hidden = config.depformer.hidden_size;
    if (!layer.attention.qkv_weight.has_value()) {
        throw std::runtime_error("PersonaPlex depformer requires packed QKV weight");
    }
    modules::QwenDecoderLayerWeights out;
    out.input_norm = layer.norm1;
    out.self_attention.qkv_weight =
        view_linear_rows(ctx, *layer.attention.qkv_weight, step * 3 * hidden, 3 * hidden, hidden, "qkv");
    out.self_attention.out_weight =
        view_linear_rows(ctx, layer.attention.out_weight, step * hidden, hidden, hidden, "out");
    out.post_norm = layer.norm2;
    out.mlp.gate_up_proj = layer.gate_up.at(static_cast<size_t>(step));
    out.mlp.down_proj = layer.down.at(static_cast<size_t>(step));
    return out;
}

PersonaPlexDepformerLayerWeights load_depformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PersonaPlexConfig & config,
    int64_t layer,
    assets::TensorStorageType storage_type) {
    const auto & dep = config.depformer;
    const auto steps = config.lm.depformer_steps;
    const std::string prefix = "depformer.layers." + std::to_string(layer) + ".";
    PersonaPlexDepformerLayerWeights weights;
    weights.norm1 = load_rms_alpha(store, source, prefix + "norm1.alpha", dep.hidden_size);
    weights.attention.qkv_weight = store.load_tensor(
        source,
        prefix + "self_attn.in_proj_weight",
        storage_type,
        {steps * 3 * dep.hidden_size, dep.hidden_size});
    weights.attention.out_weight = store.load_tensor(
        source,
        prefix + "self_attn.out_proj.weight",
        storage_type,
        {steps * dep.hidden_size, dep.hidden_size});
    weights.norm2 = load_rms_alpha(store, source, prefix + "norm2.alpha", dep.hidden_size);
    weights.gate_up.reserve(static_cast<size_t>(steps));
    weights.down.reserve(static_cast<size_t>(steps));
    for (int64_t step = 0; step < steps; ++step) {
        const std::string step_prefix = prefix + "gating." + std::to_string(step) + ".";
        weights.gate_up.push_back(modules::LinearWeights{
            store.load_tensor(
                source,
                step_prefix + "linear_in.weight",
                storage_type,
                {2 * dep.intermediate_size, dep.hidden_size}),
            std::nullopt,
        });
        weights.down.push_back(modules::LinearWeights{
            store.load_tensor(
                source,
                step_prefix + "linear_out.weight",
                storage_type,
                {dep.hidden_size, dep.intermediate_size}),
            std::nullopt,
        });
    }
    return weights;
}

}  // namespace

std::shared_ptr<const PersonaPlexDepformerWeights> load_personaplex_depformer_weights(
    const PersonaPlexAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    if (assets.lm_weights == nullptr) {
        throw std::runtime_error("PersonaPlex LM weights source is not loaded");
    }
    auto weights = std::make_shared<PersonaPlexDepformerWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PersonaPlex.depformer.weights",
        context_bytes);
    auto & store = *weights->store;
    const auto & source = *assets.lm_weights;
    const auto & lm = assets.config.lm;
    const auto & dep = assets.config.depformer;

    weights->input_from_lm.reserve(static_cast<size_t>(lm.depformer_steps));
    for (int64_t step = 0; step < lm.depformer_steps; ++step) {
        weights->input_from_lm.push_back(modules::LinearWeights{
            store.load_tensor(
                source,
                "depformer_in." + std::to_string(step) + ".weight",
                storage_type,
                {dep.hidden_size, lm.hidden_size}),
            std::nullopt,
        });
    }
    weights->text_embedding = store.load_tensor(
        source,
        "depformer_text_emb.weight",
        storage_type,
        {lm.text_vocab_size + 1, dep.hidden_size});
    weights->audio_embeddings.reserve(static_cast<size_t>(lm.depformer_steps - 1));
    for (int64_t step = 0; step < lm.depformer_steps - 1; ++step) {
        weights->audio_embeddings.push_back(store.load_tensor(
            source,
            "depformer_emb." + std::to_string(step) + ".weight",
            storage_type,
            {lm.audio_codebook_size + 1, dep.hidden_size}));
    }
    weights->layers.reserve(static_cast<size_t>(dep.num_layers));
    for (int64_t layer = 0; layer < dep.num_layers; ++layer) {
        weights->layers.push_back(load_depformer_layer(store, source, assets.config, layer, storage_type));
    }
    weights->heads.reserve(static_cast<size_t>(lm.depformer_steps));
    for (int64_t step = 0; step < lm.depformer_steps; ++step) {
        weights->heads.push_back(modules::LinearWeights{
            store.load_tensor(
                source,
                "linears." + std::to_string(step) + ".weight",
                storage_type,
                {lm.audio_codebook_size, dep.hidden_size}),
            std::nullopt,
        });
    }
    weights->store->upload();
    return weights;
}

struct PersonaPlexDepformerRuntime::Impl {
    struct StepGraph {
        ggml_cgraph * graph = nullptr;
        ggml_tensor * logits = nullptr;
    };

    Impl(
        std::shared_ptr<const PersonaPlexDepformerWeights> graph_weights,
        PersonaPlexConfig graph_config,
        ggml_backend_t graph_backend,
        core::BackendType graph_backend_type,
        int graph_threads,
        size_t graph_arena_bytes)
        : weights(std::move(graph_weights)),
          config(std::move(graph_config)),
          backend(graph_backend),
          backend_type(graph_backend_type),
          threads(graph_threads) {
        if (weights == nullptr) {
            throw std::runtime_error("PersonaPlex depformer runtime requires weights");
        }
        if (backend == nullptr) {
            throw std::runtime_error("PersonaPlex depformer runtime requires backend");
        }
        if (threads <= 0) {
            throw std::runtime_error("PersonaPlex depformer runtime requires positive thread count");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize PersonaPlex depformer graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "personaplex.depformer", backend_type};
        transformer_hidden = core::make_tensor(
                                 build_ctx,
                                 GGML_TYPE_F32,
                                 core::TensorShape::from_dims({1, 1, config.lm.hidden_size}))
                                 .tensor;
        prev_token = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_set_input(prev_token);
        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_set_input(positions);
        cache_slot = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_set_input(cache_slot);
        attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, config.lm.depformer_steps, 1, 1, 1);
        ggml_set_input(attention_mask);

        cache_keys.reserve(static_cast<size_t>(config.depformer.num_layers));
        cache_values.reserve(static_cast<size_t>(config.depformer.num_layers));
        const ggml_type cache_type = backend_type == core::BackendType::Vulkan ? GGML_TYPE_F16 : GGML_TYPE_BF16;
        for (int64_t layer = 0; layer < config.depformer.num_layers; ++layer) {
            cache_keys.push_back(core::make_tensor(
                build_ctx,
                cache_type,
                core::TensorShape::from_dims({1, config.lm.depformer_steps, config.depformer.num_attention_heads, config.depformer.head_dim})));
            cache_values.push_back(core::make_tensor(
                build_ctx,
                cache_type,
                core::TensorShape::from_dims({1, config.lm.depformer_steps, config.depformer.num_attention_heads, config.depformer.head_dim})));
        }

        auto position_value = core::wrap_tensor(positions, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto cache_slot_value = core::wrap_tensor(cache_slot, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto mask_value = core::wrap_tensor(
            attention_mask,
            core::TensorShape::from_dims({1, 1, 1, config.lm.depformer_steps}),
            GGML_TYPE_F16);
        graphs.resize(static_cast<size_t>(config.lm.depformer_steps));
        for (int64_t step = 0; step < config.lm.depformer_steps; ++step) {
            auto & step_graph = graphs[static_cast<size_t>(step)];
            step_graph.graph = ggml_new_graph_custom(ctx.get(), 32768, false);
            auto hidden = modules::LinearModule({config.lm.hidden_size, config.depformer.hidden_size, false})
                              .build(build_ctx, core::wrap_tensor(
                                                    transformer_hidden,
                                                    core::TensorShape::from_dims({1, 1, config.lm.hidden_size}),
                                                    GGML_TYPE_F32),
                                     weights->input_from_lm[static_cast<size_t>(step)]);
            core::TensorValue token_embedding;
            if (step == 0) {
                token_embedding = modules::EmbeddingModule({config.lm.text_vocab_size + 1, config.depformer.hidden_size})
                                      .build(build_ctx,
                                             core::wrap_tensor(prev_token, core::TensorShape::from_dims({1}), GGML_TYPE_I32),
                                             weights->text_embedding);
            } else {
                token_embedding = modules::EmbeddingModule({config.lm.audio_codebook_size + 1, config.depformer.hidden_size})
                                      .build(build_ctx,
                                             core::wrap_tensor(prev_token, core::TensorShape::from_dims({1}), GGML_TYPE_I32),
                                             weights->audio_embeddings[static_cast<size_t>(step - 1)]);
            }
            token_embedding = core::reshape_tensor(
                build_ctx,
                token_embedding,
                core::TensorShape::from_dims({1, 1, config.depformer.hidden_size}));
            hidden = core::wrap_tensor(ggml_add(ctx.get(), hidden.tensor, token_embedding.tensor), hidden.shape, GGML_TYPE_F32);

            const auto layer_config = depformer_layer_config(config, backend_type);
            const modules::QwenDecoderLayerModule layer_module(layer_config);
            for (int64_t layer_index = 0; layer_index < config.depformer.num_layers; ++layer_index) {
                const auto layer_weights = depformer_step_layer_weights(
                    ctx.get(),
                    weights->layers[static_cast<size_t>(layer_index)],
                    step,
                    config);
                auto layer_out = layer_module.build_with_static_cache_tail(
                    build_ctx,
                    step_graph.graph,
                    hidden,
                    position_value,
                    layer_weights,
                    cache_keys[static_cast<size_t>(layer_index)],
                    cache_values[static_cast<size_t>(layer_index)],
                    cache_slot_value,
                    mask_value);
                hidden = layer_out.output;
            }
            auto logits = modules::LinearModule({config.depformer.hidden_size, config.lm.audio_codebook_size, false})
                              .build(build_ctx, hidden, weights->heads[static_cast<size_t>(step)]);
            step_graph.logits = logits.tensor;
            ggml_set_output(step_graph.logits);
            ggml_build_forward_expand(step_graph.graph, step_graph.logits);
        }

        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate PersonaPlex depformer graph");
        }
        mask_values.assign(
            static_cast<size_t>(config.lm.depformer_steps),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        discard_logits.assign(static_cast<size_t>(config.lm.audio_codebook_size), 0.0F);
        logits.assign(static_cast<size_t>(config.lm.audio_codebook_size), 0.0F);
    }

    ~Impl() {
        for (auto & graph : graphs) {
            engine::core::release_backend_graph_resources(backend, graph.graph);
        }
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    PersonaPlexDepformerOutput run(
        int32_t text_token,
        const std::vector<float> & transformer_hidden_values,
        const std::array<int32_t, kPersonaPlexDepformerAudioStreams> & audio_target,
        const std::array<uint8_t, kPersonaPlexDepformerAudioStreams> & audio_provided,
        const PersonaPlexDepformerSamplingOptions & sampling,
        sampling::HfSamplerScratch & scratch,
        std::mt19937 & fallback_rng,
        const sampling::HfTorchSamplingState * torch_state) {
        if (static_cast<int64_t>(transformer_hidden_values.size()) != config.lm.hidden_size) {
            throw std::runtime_error("PersonaPlex depformer hidden size mismatch");
        }
        if (static_cast<int64_t>(audio_target.size()) != config.lm.depformer_steps ||
            static_cast<int64_t>(audio_provided.size()) != config.lm.depformer_steps) {
            throw std::runtime_error("PersonaPlex depformer target/provided size mismatch");
        }
        ggml_backend_tensor_set(
            transformer_hidden,
            transformer_hidden_values.data(),
            0,
            transformer_hidden_values.size() * sizeof(float));

        sampling::HfSamplingOptions hf_options;
        hf_options.do_sample = sampling.do_sample;
        hf_options.temperature = sampling.temperature;
        hf_options.top_k = sampling.top_k;
        hf_options.top_p = 1.0F;

        PersonaPlexDepformerOutput out;
        int64_t last_unprovided_step = -1;
        for (int64_t step = config.lm.depformer_steps - 1; step >= 0; --step) {
            if (audio_provided[static_cast<size_t>(step)] == 0) {
                last_unprovided_step = step;
                break;
            }
        }
        if (last_unprovided_step < 0) {
            if (sampling.do_sample) {
                for (int64_t step = 0; step < config.lm.depformer_steps; ++step) {
                    (void)sampler.sample(
                        discard_logits,
                        {},
                        hf_options,
                        scratch,
                        fallback_rng,
                        torch_state,
                        "PersonaPlex depformer");
                }
            }
            std::copy(audio_target.begin(), audio_target.end(), out.sampled_audio_tokens.begin());
            return out;
        }
        int32_t previous_token = text_token;
        for (int64_t step = 0; step <= last_unprovided_step; ++step) {
            ggml_backend_tensor_set(prev_token, &previous_token, 0, sizeof(previous_token));
            const int32_t position = static_cast<int32_t>(step);
            ggml_backend_tensor_set(positions, &position, 0, sizeof(position));
            ggml_backend_tensor_set(cache_slot, &position, 0, sizeof(position));
            std::fill(
                mask_values.begin(),
                mask_values.end(),
                ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int64_t i = 0; i <= step; ++i) {
                mask_values[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
            }
            ggml_backend_tensor_set(attention_mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
            core::set_backend_threads(backend, threads);
            const ggml_status status = core::compute_backend_graph(
                backend,
                graphs[static_cast<size_t>(step)].graph,
                nullptr,
                "personaplex.depformer");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("PersonaPlex depformer graph compute failed");
            }
            ggml_backend_tensor_get(
                graphs[static_cast<size_t>(step)].logits,
                logits.data(),
                0,
                logits.size() * sizeof(float));
            const int32_t sampled = sampler.sample(
                logits,
                {},
                hf_options,
                scratch,
                fallback_rng,
                torch_state,
                "PersonaPlex depformer");
            out.sampled_audio_tokens[static_cast<size_t>(step)] = sampled;
            previous_token = audio_provided[static_cast<size_t>(step)] != 0
                ? audio_target[static_cast<size_t>(step)]
                : sampled;
        }
        for (int64_t step = last_unprovided_step + 1; step < config.lm.depformer_steps; ++step) {
            if (sampling.do_sample) {
                (void)sampler.sample(
                    discard_logits,
                    {},
                    hf_options,
                    scratch,
                    fallback_rng,
                    torch_state,
                    "PersonaPlex depformer");
            }
            out.sampled_audio_tokens[static_cast<size_t>(step)] = audio_target[static_cast<size_t>(step)];
        }
        return out;
    }

    std::shared_ptr<const PersonaPlexDepformerWeights> weights;
    PersonaPlexConfig config;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * transformer_hidden = nullptr;
    ggml_tensor * prev_token = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * cache_slot = nullptr;
    ggml_tensor * attention_mask = nullptr;
    std::vector<core::TensorValue> cache_keys;
    std::vector<core::TensorValue> cache_values;
    std::vector<StepGraph> graphs;
    std::vector<ggml_fp16_t> mask_values;
    std::vector<float> discard_logits;
    std::vector<float> logits;
    sampling::HfSampler sampler;
    ggml_backend_buffer_t buffer = nullptr;
};

PersonaPlexDepformerRuntime::PersonaPlexDepformerRuntime(
    std::shared_ptr<const PersonaPlexDepformerWeights> weights,
    PersonaPlexConfig config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(weights),
          std::move(config),
          backend,
          backend_type,
          threads,
          graph_arena_bytes)) {}

PersonaPlexDepformerRuntime::~PersonaPlexDepformerRuntime() = default;

PersonaPlexDepformerOutput PersonaPlexDepformerRuntime::run(
    int32_t text_token,
    const std::vector<float> & transformer_hidden,
    const std::array<int32_t, kPersonaPlexDepformerAudioStreams> & audio_target,
    const std::array<uint8_t, kPersonaPlexDepformerAudioStreams> & audio_provided,
    const PersonaPlexDepformerSamplingOptions & sampling,
    sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const sampling::HfTorchSamplingState * torch_state) {
    return impl_->run(
        text_token,
        transformer_hidden,
        audio_target,
        audio_provided,
        sampling,
        scratch,
        fallback_rng,
        torch_state);
}

}  // namespace engine::models::personaplex
