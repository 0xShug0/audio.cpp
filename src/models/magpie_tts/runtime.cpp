#include "engine/models/magpie_tts/runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/codecs/nemo_nano_codec.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::magpie_tts {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int32_t kAudioSpecialCount = 8;
constexpr int32_t kTextPadId = 0;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GraphArena {
    explicit GraphArena(ggml_backend_t backend) {
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr) {
            throw std::runtime_error("MagpieTTS failed to create graph allocator");
        }
    }

    ~GraphArena() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    GraphArena(const GraphArena &) = delete;
    GraphArena & operator=(const GraphArena &) = delete;

    void allocate(ggml_cgraph * graph) {
        if (!ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("MagpieTTS failed to allocate graph");
        }
    }

    ggml_gallocr_t gallocr = nullptr;
};

core::TensorValue mask_sequence(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask) {
    auto m = core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape, GGML_TYPE_F32);
    m = core::reshape_tensor(ctx, m, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], 1}));
    auto repeated = modules::RepeatModule({input.shape}).build(ctx, m);
    return modules::MulModule().build(ctx, input, repeated);
}

core::TensorValue add_position_embeddings(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & position_weight) {
    auto positions = modules::SliceModule({0, 0, input.shape.dims[1]}).build(ctx, position_weight);
    positions = core::reshape_tensor(ctx, positions, core::TensorShape::from_dims({1, input.shape.dims[1], input.shape.dims[2]}));
    auto repeated = modules::RepeatModule({input.shape}).build(ctx, positions);
    return modules::AddModule().build(ctx, input, repeated);
}

core::TensorValue add_position_embeddings_from_ids(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & position_ids,
    const core::TensorValue & position_weight) {
    auto positions = modules::EmbeddingModule({position_weight.shape.dims[0], input.shape.dims[2]})
                         .build(ctx, position_ids, position_weight);
    positions = core::reshape_tensor(ctx, positions, core::TensorShape::from_dims({1, input.shape.dims[1], input.shape.dims[2]}));
    auto repeated = modules::RepeatModule({input.shape}).build(ctx, positions);
    return modules::AddModule().build(ctx, input, repeated);
}

struct TransformerLayerWeights {
    modules::NormWeights norm_self;
    modules::LinearWeights self_qkv;
    modules::LinearWeights self_out;
    modules::NormWeights norm_xattn_query;
    modules::NormWeights norm_xattn_memory;
    modules::LinearWeights cross_q;
    modules::LinearWeights cross_kv;
    modules::LinearWeights cross_out;
    modules::NormWeights norm_pos_ff;
    modules::Conv1dWeights ff_proj;
    modules::Conv1dWeights ff_out;
};

struct TransformerWeights {
    std::vector<TransformerLayerWeights> layers;
    core::TensorValue position_embedding;
    std::optional<modules::NormWeights> norm_out;
};

struct DecoderCrossCacheView {
    const std::vector<core::TensorValue> * keys = nullptr;
    const std::vector<core::TensorValue> * values = nullptr;
    int64_t text_steps = 0;
};

struct ModelWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue baked_context_embedding;
    std::vector<core::TensorValue> audio_embeddings;
    TransformerWeights encoder;
    TransformerWeights decoder;
    TransformerWeights local;
    modules::LinearWeights final_proj;
    std::vector<modules::LinearWeights> local_out;
};

TransformerLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t ffn,
    int64_t kernel,
    int64_t cross_heads,
    int64_t cross_head_dim,
    bool cross,
    assets::TensorStorageType storage_type) {
    TransformerLayerWeights weights;
    weights.norm_self = {store.load_f32_tensor(source, prefix + ".norm_self.weight", {hidden}), std::nullopt};
    weights.self_qkv = modules::binding::linear_from_source(store, source, prefix + ".self_attention.qkv_net", storage_type, 3 * hidden, hidden, false);
    weights.self_out = modules::binding::linear_from_source(store, source, prefix + ".self_attention.o_net", storage_type, hidden, hidden, false);
    if (cross) {
        weights.norm_xattn_query = {store.load_f32_tensor(source, prefix + ".norm_xattn_query.weight", {hidden}), std::nullopt};
        weights.norm_xattn_memory = {store.load_f32_tensor(source, prefix + ".norm_xattn_memory.weight", {hidden}), std::nullopt};
        weights.cross_q = modules::binding::linear_from_source(store, source, prefix + ".cross_attention.q_net", storage_type, cross_heads * cross_head_dim, hidden, false);
        weights.cross_kv = modules::binding::linear_from_source(store, source, prefix + ".cross_attention.kv_net", storage_type, 2 * cross_heads * cross_head_dim, hidden, false);
        weights.cross_out = modules::binding::linear_from_source(store, source, prefix + ".cross_attention.o_net", storage_type, hidden, cross_heads * cross_head_dim, false);
    }
    weights.norm_pos_ff = {store.load_f32_tensor(source, prefix + ".norm_pos_ff.weight", {hidden}), std::nullopt};
    weights.ff_proj = modules::binding::conv1d_from_source(store, source, prefix + ".pos_ff.proj.conv", storage_type, ffn, hidden, kernel, false);
    weights.ff_out = modules::binding::conv1d_from_source(store, source, prefix + ".pos_ff.o_net.conv", storage_type, hidden, ffn, kernel, false);
    return weights;
}

TransformerWeights load_transformer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t layers,
    int64_t hidden,
    int64_t ffn,
    int64_t kernel,
    int64_t max_pos,
    int64_t cross_heads,
    int64_t cross_head_dim,
    bool cross,
    assets::TensorStorageType storage_type) {
    TransformerWeights weights;
    weights.layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        weights.layers.push_back(load_layer(
            store,
            source,
            prefix + ".layers." + std::to_string(layer),
            hidden,
            ffn,
            kernel,
            cross_heads,
            cross_head_dim,
            cross,
            storage_type));
    }
    weights.position_embedding =
        store.load_tensor(source, prefix + ".position_embeddings.weight", storage_type, {max_pos, hidden});
    if (source.has_tensor(prefix + ".norm_out.weight")) {
        weights.norm_out = modules::NormWeights{store.load_f32_tensor(source, prefix + ".norm_out.weight", {hidden}), std::nullopt};
    }
    return weights;
}

ModelWeights load_model_weights(
    const MagpieTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const MagpieTTSRuntimeOptions & options) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("MagpieTTS runtime requires model tensor source");
    }
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    ModelWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "magpie_tts.model.weights",
        options.weight_context_bytes);
    weights.text_embedding = weights.store->load_tensor(
        source,
        "text_embedding.weight",
        options.matmul_weight_storage_type,
        {config.text_vocab_size, config.embedding_dim});
    weights.baked_context_embedding = weights.store->load_tensor(
        source,
        "baked_context_embedding.weight",
        assets::TensorStorageType::F32,
        {config.speakers, config.context_length * config.context_dim});
    const int64_t stacked = config.audio_codebooks * config.frame_stacking_factor;
    weights.audio_embeddings.reserve(static_cast<size_t>(stacked));
    weights.local_out.reserve(static_cast<size_t>(stacked));
    for (int64_t i = 0; i < stacked; ++i) {
        weights.audio_embeddings.push_back(weights.store->load_tensor(
            source,
            "audio_embeddings." + std::to_string(i) + ".weight",
            options.matmul_weight_storage_type,
            {config.all_tokens_per_codebook, config.embedding_dim}));
        weights.local_out.push_back(modules::binding::linear_from_source(
            *weights.store,
            source,
            "local_transformer_out_projections." + std::to_string(i),
            options.matmul_weight_storage_type,
            config.all_tokens_per_codebook,
            config.local_hidden_dim,
            true));
    }
    weights.encoder = load_transformer(
        *weights.store,
        source,
        "encoder",
        6,
        config.embedding_dim,
        3072,
        3,
        2048,
        0,
        0,
        false,
        options.matmul_weight_storage_type);
    weights.decoder = load_transformer(
        *weights.store,
        source,
        "decoder",
        config.decoder_layers,
        config.embedding_dim,
        config.decoder_ffn_dim,
        1,
        config.decoder_max_length,
        config.decoder_cross_heads,
        config.decoder_cross_head_dim,
        true,
        options.matmul_weight_storage_type);
    weights.local = load_transformer(
        *weights.store,
        source,
        "local_transformer",
        config.local_layers,
        config.local_hidden_dim,
        config.local_ffn_dim,
        1,
        config.local_context,
        0,
        0,
        false,
        options.matmul_weight_storage_type);
    weights.final_proj = modules::binding::linear_from_source(
        *weights.store,
        source,
        "final_proj",
        options.matmul_weight_storage_type,
        stacked * config.all_tokens_per_codebook,
        config.embedding_dim,
        true);
    weights.store->upload();
    return weights;
}

modules::NemoNanoCodecConfig make_nemo_nano_codec_config(const MagpieTTSConfig & config) {
    modules::NemoNanoCodecConfig out;
    out.sample_rate = config.sample_rate;
    out.input_dim = config.codec_input_dim;
    out.base_channels = config.codec_base_channels;
    out.audio_codebooks = config.audio_codebooks;
    out.upsample_rates = config.codec_upsample_rates;
    out.resblock_kernel_sizes = config.codec_resblock_kernel_sizes;
    out.resblock_dilation_sizes = config.codec_resblock_dilation_sizes;
    out.fsq_num_levels = config.codec_fsq_num_levels;
    out.fsq_dim_base_index = config.codec_fsq_dim_base_index;
    return out;
}

core::TensorValue build_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & input_mask,
    const TransformerLayerWeights & weights,
    int64_t hidden,
    int64_t heads,
    bool causal,
    int64_t kernel,
    const std::optional<core::TensorValue> & self_attention_mask,
    const std::optional<core::TensorValue> & cond,
    const std::optional<core::TensorValue> & cond_mask,
    const DecoderCrossCacheView * cross_cache,
    const core::TensorValue * attention_prior,
    std::vector<core::TensorValue> * alignment_outputs,
    int64_t layer_index,
    int64_t cross_heads,
    int64_t cross_head_dim) {
    auto x = mask_sequence(ctx, input, input_mask);
    auto self_norm = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, weights.norm_self);
    modules::AttentionConfig self_attention_config{hidden, heads, false};
    self_attention_config.projection_precision = GGML_PREC_DEFAULT;
    self_attention_config.attention_precision = GGML_PREC_F32;
    self_attention_config.use_packed_qkv = true;
    self_attention_config.causal = causal;
    modules::AttentionWeights self_attention_weights;
    self_attention_weights.qkv_weight = weights.self_qkv.weight;
    self_attention_weights.qkv_bias = weights.self_qkv.bias;
    self_attention_weights.out_weight = weights.self_out.weight;
    self_attention_weights.out_bias = weights.self_out.bias;
    auto sa = modules::SelfAttentionModule(self_attention_config).build(ctx, self_norm, self_attention_weights, self_attention_mask);
    x = modules::ResidualAddModule().build(ctx, x, sa);
    if (cond_mask.has_value() && cross_cache != nullptr) {
        if (cross_cache->keys == nullptr || cross_cache->values == nullptr ||
            layer_index < 0 ||
            layer_index >= static_cast<int64_t>(cross_cache->keys->size()) ||
            layer_index >= static_cast<int64_t>(cross_cache->values->size())) {
            throw std::runtime_error("MagpieTTS decoder cross-attention cache is incomplete");
        }
        auto qa = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, weights.norm_xattn_query);
        core::TensorValue last_attention;
        modules::AttentionConfig cross_attention_config{hidden, cross_heads, false};
        cross_attention_config.key_value_size = hidden;
        cross_attention_config.attention_size = cross_heads * cross_head_dim;
        cross_attention_config.head_dim = cross_head_dim;
        cross_attention_config.use_packed_kv = true;
        modules::AttentionWeights cross_attention_weights;
        cross_attention_weights.q_weight = weights.cross_q.weight;
        cross_attention_weights.q_bias = weights.cross_q.bias;
        cross_attention_weights.qkv_weight = weights.cross_kv.weight;
        cross_attention_weights.qkv_bias = weights.cross_kv.bias;
        cross_attention_weights.out_weight = weights.cross_out.weight;
        cross_attention_weights.out_bias = weights.cross_out.bias;
        auto ca = modules::CrossAttentionModule(cross_attention_config).build_cached(
            ctx,
            qa,
            {
                cross_cache->keys->at(static_cast<size_t>(layer_index)),
                cross_cache->values->at(static_cast<size_t>(layer_index)),
            },
            cross_attention_weights,
            *cond_mask,
            attention_prior,
            alignment_outputs != nullptr ? &last_attention : nullptr);
        if (alignment_outputs != nullptr && last_attention.valid()) {
            alignment_outputs->push_back(last_attention);
        }
        x = modules::ResidualAddModule().build(ctx, x, ca);
    } else if (cond.has_value() && cond_mask.has_value()) {
        auto memory = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, *cond, weights.norm_xattn_memory);
        auto qa = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, weights.norm_xattn_query);
        core::TensorValue last_attention;
        modules::AttentionConfig cross_attention_config{hidden, cross_heads, false};
        cross_attention_config.key_value_size = hidden;
        cross_attention_config.attention_size = cross_heads * cross_head_dim;
        cross_attention_config.head_dim = cross_head_dim;
        cross_attention_config.use_packed_kv = true;
        modules::AttentionWeights cross_attention_weights;
        cross_attention_weights.q_weight = weights.cross_q.weight;
        cross_attention_weights.q_bias = weights.cross_q.bias;
        cross_attention_weights.qkv_weight = weights.cross_kv.weight;
        cross_attention_weights.qkv_bias = weights.cross_kv.bias;
        cross_attention_weights.out_weight = weights.cross_out.weight;
        cross_attention_weights.out_bias = weights.cross_out.bias;
        auto ca = modules::CrossAttentionModule(cross_attention_config).build(
            ctx,
            qa,
            memory,
            cross_attention_weights,
            *cond_mask,
            attention_prior,
            alignment_outputs != nullptr ? &last_attention : nullptr);
        if (alignment_outputs != nullptr && last_attention.valid()) {
            alignment_outputs->push_back(last_attention);
        }
        x = modules::ResidualAddModule().build(ctx, x, ca);
    }
    auto ff = modules::ConvFeedForwardModule({
        hidden,
        weights.ff_proj.weight.shape.dims[0],
        kernel,
        causal,
        false,
        modules::GeluApproximation::Tanh,
    }).build(
        ctx,
        modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, weights.norm_pos_ff),
        {weights.ff_proj, weights.ff_out});
    x = modules::ResidualAddModule().build(ctx, x, ff);
    return mask_sequence(ctx, x, input_mask);
}

core::TensorValue build_transformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & input_mask,
    const TransformerWeights & weights,
    int64_t hidden,
    int64_t heads,
    bool causal,
    int64_t kernel,
    const std::optional<core::TensorValue> & self_attention_mask = std::nullopt,
    const std::optional<core::TensorValue> & cond = std::nullopt,
    const std::optional<core::TensorValue> & cond_mask = std::nullopt,
    const DecoderCrossCacheView * cross_cache = nullptr,
    const core::TensorValue * attention_prior = nullptr,
    const std::vector<int64_t> * apply_prior_to_layers = nullptr,
    const std::vector<int64_t> * estimate_alignment_from_layers = nullptr,
    std::vector<core::TensorValue> * alignment_outputs = nullptr,
    int64_t cross_heads = 0,
    int64_t cross_head_dim = 0) {
    auto x = add_position_embeddings(ctx, input, weights.position_embedding);
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const bool apply_prior =
            attention_prior != nullptr &&
            (apply_prior_to_layers == nullptr || apply_prior_to_layers->empty() ||
             std::find(apply_prior_to_layers->begin(), apply_prior_to_layers->end(), static_cast<int64_t>(layer_index)) != apply_prior_to_layers->end());
        const bool collect_alignment =
            alignment_outputs != nullptr &&
            (estimate_alignment_from_layers == nullptr || estimate_alignment_from_layers->empty() ||
             std::find(estimate_alignment_from_layers->begin(), estimate_alignment_from_layers->end(), static_cast<int64_t>(layer_index)) != estimate_alignment_from_layers->end());
        x = build_transformer_layer(
            ctx,
            x,
            input_mask,
            weights.layers[layer_index],
            hidden,
            heads,
            causal,
            kernel,
            self_attention_mask,
            cond,
            cond_mask,
            cross_cache,
            apply_prior ? attention_prior : nullptr,
            collect_alignment ? alignment_outputs : nullptr,
            static_cast<int64_t>(layer_index),
            cross_heads,
            cross_head_dim);
    }
    if (weights.norm_out.has_value()) {
        x = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, *weights.norm_out);
    }
    return x;
}

core::TensorValue build_transformer_cached_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & position_ids,
    const TransformerWeights & weights,
    int64_t hidden,
    int64_t heads,
    int64_t kernel,
    const std::vector<core::TensorValue> & cache_keys,
    const std::vector<core::TensorValue> & cache_values,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    auto x = add_position_embeddings_from_ids(ctx, input, position_ids, weights.position_embedding);
    if (cache_keys.size() != weights.layers.size() || cache_values.size() != weights.layers.size()) {
        throw std::runtime_error("MagpieTTS local transformer cache layer count mismatch");
    }
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto & layer = weights.layers[layer_index];
        auto self_norm = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, layer.norm_self);
        modules::AttentionConfig self_attention_config{hidden, heads, false};
        self_attention_config.projection_precision = GGML_PREC_DEFAULT;
        self_attention_config.attention_precision = GGML_PREC_F32;
        self_attention_config.use_packed_qkv = true;
        self_attention_config.causal = true;
        modules::AttentionWeights self_attention_weights;
        self_attention_weights.qkv_weight = layer.self_qkv.weight;
        self_attention_weights.qkv_bias = layer.self_qkv.bias;
        self_attention_weights.out_weight = layer.self_out.weight;
        self_attention_weights.out_bias = layer.self_out.bias;
        auto sa = modules::SelfAttentionModule(self_attention_config).build_cached_tail(
            ctx,
            self_norm,
            self_attention_weights,
            cache_keys[layer_index],
            cache_values[layer_index],
            cache_slot,
            attention_mask,
            modules::FastKVSetRowsMode::BackendViewOptimized).output;
        x = modules::ResidualAddModule().build(ctx, x, sa);
        auto ff = modules::ConvFeedForwardModule({
            hidden,
            layer.ff_proj.weight.shape.dims[0],
            kernel,
            true,
            false,
            modules::GeluApproximation::Tanh,
        }).build(
            ctx,
            modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, layer.norm_pos_ff),
            {layer.ff_proj, layer.ff_out});
        x = modules::ResidualAddModule().build(ctx, x, ff);
    }
    if (weights.norm_out.has_value()) {
        x = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(ctx, x, *weights.norm_out);
    }
    return x;
}

std::vector<float> read_exact_f32_tensor(ggml_tensor * tensor, size_t count, const char * name) {
    auto out = core::read_tensor_f32(tensor);
    if (out.size() != count) {
        throw std::runtime_error(
            std::string(name) + " readback element count mismatch: expected " +
            std::to_string(count) + ", got " + std::to_string(out.size()));
    }
    return out;
}

std::vector<int32_t> make_mask(int64_t batch, int64_t steps, int64_t valid_steps) {
    std::vector<int32_t> mask(static_cast<size_t>(batch * steps), 0);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < valid_steps; ++t) {
            mask[static_cast<size_t>(b * steps + t)] = 1;
        }
    }
    return mask;
}

std::vector<float> baked_context_host(
    const std::vector<float> & values,
    const MagpieTTSConfig & config,
    int32_t speaker) {
    if (static_cast<int64_t>(values.size()) != config.speakers * config.context_length * config.context_dim) {
        throw std::runtime_error("MagpieTTS baked context host cache shape mismatch");
    }
    const int64_t offset = static_cast<int64_t>(speaker) * config.context_length * config.context_dim;
    return std::vector<float>(
        values.begin() + offset,
        values.begin() + offset + config.context_length * config.context_dim);
}

struct DecoderStepOutput {
    std::vector<float> hidden;
    std::vector<float> logits;
    std::vector<float> alignment_scores;
    double compute_ms = 0.0;
};

struct DecoderConditioning {
    std::vector<float> values;
    std::vector<int32_t> mask;
    int64_t text_steps = 0;
};

DecoderConditioning make_decoder_conditioning(
    const std::vector<float> & text_encoded,
    int64_t text_steps,
    int64_t hidden) {
    if (text_steps <= 0 || hidden <= 0 ||
        static_cast<int64_t>(text_encoded.size()) != text_steps * hidden) {
        throw std::runtime_error("MagpieTTS decoder conditioning shape mismatch");
    }
    DecoderConditioning out;
    out.text_steps = text_steps;
    out.values.assign(static_cast<size_t>(2 * text_steps * hidden), 0.0F);
    std::copy(text_encoded.begin(), text_encoded.end(), out.values.begin());
    out.mask.assign(static_cast<size_t>(2 * text_steps), 0);
    for (int64_t t = 0; t < text_steps; ++t) {
        out.mask[static_cast<size_t>(t)] = 1;
    }
    out.mask[static_cast<size_t>(text_steps)] = 1;
    return out;
}

struct AttentionPriorState {
    void reset(int64_t text_steps) {
        text_len = std::max<int64_t>(0, text_steps);
        last_attended = std::min<int64_t>(1, std::max<int64_t>(0, text_len - 1));
        attended_counts.assign(static_cast<size_t>(text_len), 0);
        prior.clear();
        unfinished = false;
        finished_counter.reset();
        ended = false;
    }

    const std::vector<float> * prior_for_step(const MagpieTTSConfig & config) const {
        if (!config.apply_attention_prior || prior.empty()) {
            return nullptr;
        }
        return &prior;
    }

    void update(const MagpieTTSConfig & config, int64_t step, const std::vector<float> & alignment_scores) {
        if (!config.apply_attention_prior || step < config.start_prior_after_n_audio_steps ||
            text_len <= 0 || static_cast<int64_t>(alignment_scores.size()) != text_len) {
            return;
        }

        int64_t search_start = std::clamp<int64_t>(last_attended, 0, text_len - 1);
        if (search_start < static_cast<int64_t>(attended_counts.size()) &&
            attended_counts[static_cast<size_t>(search_start)] >= config.attention_sink_threshold) {
            search_start = std::min<int64_t>(search_start + 1, text_len - 1);
        }

        int64_t attended = text_len - 1;
        const int64_t window_end =
            std::min<int64_t>(search_start + std::max<int64_t>(0, config.attention_prior_lookahead_window), text_len - 3);
        if (window_end > search_start) {
            attended = search_start;
            float best = alignment_scores[static_cast<size_t>(search_start)];
            for (int64_t index = search_start + 1; index < window_end; ++index) {
                const float score = alignment_scores[static_cast<size_t>(index)];
                if (score > best) {
                    best = score;
                    attended = index;
                }
            }
        }

        last_attended = std::clamp<int64_t>(attended, 0, text_len - 1);
        if (last_attended < static_cast<int64_t>(attended_counts.size())) {
            ++attended_counts[static_cast<size_t>(last_attended)];
        }

        prior.assign(static_cast<size_t>(text_len), config.attention_prior_epsilon);
        if (text_len <= 5) {
            std::fill(prior.begin(), prior.end(), 1.0F);
        } else {
            prior[static_cast<size_t>(std::max<int64_t>(1, last_attended - 1))] = 1.0F;
            prior[static_cast<size_t>(last_attended)] = 1.0F;
            for (int64_t offset = 1; offset <= config.attention_prior_lookahead_window; ++offset) {
                prior[static_cast<size_t>(std::min<int64_t>(last_attended + offset, text_len - 1))] = 1.0F;
            }
        }

        for (int64_t index = 0; index < static_cast<int64_t>(attended_counts.size()); ++index) {
            if (attended_counts[static_cast<size_t>(index)] >= config.attention_sink_threshold) {
                const int64_t fill_end = std::min<int64_t>(index + 1, text_len);
                std::fill(prior.begin(), prior.begin() + fill_end, config.attention_prior_epsilon);
            }
        }

        unfinished = last_attended < text_len - 3 && !ended;
        if (last_attended >= text_len - 2 || ended) {
            if (!finished_counter.has_value()) {
                finished_counter = 0;
            }
        }
        if (finished_counter.has_value()) {
            ++*finished_counter;
            if (*finished_counter > 5) {
                unfinished = false;
            }
        }
    }

    void mark_ended() {
        ended = true;
    }

    bool should_forbid_eos(bool min_frame_guard, bool ignore_finished_tracking) const {
        return min_frame_guard || (!ignore_finished_tracking && unfinished);
    }

    bool should_force_eos(bool ignore_finished_tracking) const {
        return !ignore_finished_tracking && finished_counter.has_value() && *finished_counter >= 20;
    }

    int64_t text_len = 0;
    int64_t last_attended = 1;
    std::vector<int64_t> attended_counts;
    std::vector<float> prior;
    bool unfinished = false;
    bool ended = false;
    std::optional<int64_t> finished_counter;
};

}  // namespace

struct MagpieTTSRuntime::Impl {
    class DecoderCrossCacheGraph {
    public:
        DecoderCrossCacheGraph(
            Impl & owner,
            const DecoderConditioning & conditioning)
            : owner_(owner),
              text_steps_(conditioning.text_steps) {
            const auto & config = owner_.assets->config;
            constexpr int64_t batch = 2;
            const int64_t hidden = config.embedding_dim;
            const int64_t cross = config.decoder_cross_heads * config.decoder_cross_head_dim;
            if (text_steps_ <= 0 || cross <= 0 ||
                static_cast<int64_t>(conditioning.values.size()) != batch * text_steps_ * hidden) {
                throw std::runtime_error("MagpieTTS decoder cross-cache conditioning shape mismatch");
            }

            ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("MagpieTTS failed to create decoder cross-cache graph context");
            }
            core::ModuleBuildContext build{ctx_.get(), "magpie_tts.decoder.cross_kv", owner_.backend_type};
            auto cond = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({batch, text_steps_, hidden}));
            cond_ = cond.tensor;
            keys_.reserve(owner_.weights->decoder.layers.size());
            values_.reserve(owner_.weights->decoder.layers.size());
            graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
            for (const auto & layer : owner_.weights->decoder.layers) {
                auto memory = modules::LayerNormModule({hidden, 1.0e-5F, true, false}).build(build, cond, layer.norm_xattn_memory);
                modules::AttentionConfig cross_attention_config{hidden, config.decoder_cross_heads, false};
                cross_attention_config.key_value_size = hidden;
                cross_attention_config.attention_size = cross;
                cross_attention_config.head_dim = config.decoder_cross_head_dim;
                cross_attention_config.use_packed_kv = true;
                modules::AttentionWeights cross_attention_weights;
                cross_attention_weights.qkv_weight = layer.cross_kv.weight;
                cross_attention_weights.qkv_bias = layer.cross_kv.bias;
                const auto key_value = modules::CrossAttentionModule(cross_attention_config).build_key_value(
                    build,
                    memory,
                    cross_attention_weights);
                auto key_cache = core::make_tensor(build, GGML_TYPE_F16, core::TensorShape::from_dims({batch, config.decoder_cross_heads, text_steps_, config.decoder_cross_head_dim}));
                auto value_cache = core::make_tensor(build, GGML_TYPE_F16, core::TensorShape::from_dims({batch, config.decoder_cross_heads, text_steps_, config.decoder_cross_head_dim}));
                auto key_copy = core::wrap_tensor(ggml_cpy(build.ggml, key_value.key.tensor, key_cache.tensor), key_cache.shape, GGML_TYPE_F16);
                auto value_copy = core::wrap_tensor(ggml_cpy(build.ggml, key_value.value.tensor, value_cache.tensor), value_cache.shape, GGML_TYPE_F16);
                keys_.push_back(key_cache);
                values_.push_back(value_cache);
                ggml_set_output(key_copy.tensor);
                ggml_set_output(value_copy.tensor);
                ggml_build_forward_expand(graph_, key_copy.tensor);
                ggml_build_forward_expand(graph_, value_copy.tensor);
            }
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner_.backend));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("MagpieTTS failed to allocate decoder cross-cache graph");
            }
            ggml_backend_tensor_set(cond_, conditioning.values.data(), 0, conditioning.values.size() * sizeof(float));
            const auto start = Clock::now();
            const ggml_status status = core::compute_backend_graph(owner_.backend, graph_);
            debug::timing_log_scalar("magpie_tts.decoder.cross_kv.graph.compute_ms", debug::elapsed_ms(start));
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MagpieTTS decoder cross-cache graph compute failed");
            }
        }

        ~DecoderCrossCacheGraph() {
            core::release_backend_graph_resources(owner_.backend, graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
        }

        DecoderCrossCacheGraph(const DecoderCrossCacheGraph &) = delete;
        DecoderCrossCacheGraph & operator=(const DecoderCrossCacheGraph &) = delete;

        DecoderCrossCacheView view() const noexcept {
            return {&keys_, &values_, text_steps_};
        }

    private:
        Impl & owner_;
        int64_t text_steps_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * cond_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        std::vector<core::TensorValue> keys_;
        std::vector<core::TensorValue> values_;
    };

    class LocalTransformerGraph {
    public:
        struct Output {
            std::vector<float> logits;
            double compute_ms = 0.0;
        };

        explicit LocalTransformerGraph(Impl & owner)
            : owner_(owner) {
            const auto & config = owner_.assets->config;
            constexpr int64_t batch = 2;
            const int64_t hidden = config.embedding_dim;
            const int64_t stacked = config.audio_codebooks * config.frame_stacking_factor;
            const int64_t head_dim = hidden / config.local_heads;
            ggml_init_params params{256ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("MagpieTTS failed to create local transformer graph context");
            }
            core::ModuleBuildContext build{ctx_.get(), "magpie_tts.local_transformer", owner_.backend_type};
            auto input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, hidden}));
            input_ = input.tensor;
            position_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
            auto position = core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, batch);
            auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({batch}), GGML_TYPE_I32);
            attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, stacked, 1, 1, batch);
            auto attention_mask = core::wrap_tensor(
                attention_mask_,
                core::TensorShape::from_dims({batch, 1, 1, stacked}),
                GGML_TYPE_F16);
            cache_keys_.reserve(owner_.weights->local.layers.size());
            cache_values_.reserve(owner_.weights->local.layers.size());
            for (size_t layer = 0; layer < owner_.weights->local.layers.size(); ++layer) {
                cache_keys_.push_back(core::make_tensor(build, GGML_TYPE_F16, core::TensorShape::from_dims({batch, stacked, config.local_heads, head_dim})));
                cache_values_.push_back(core::make_tensor(build, GGML_TYPE_F16, core::TensorShape::from_dims({batch, stacked, config.local_heads, head_dim})));
            }
            graphs_.resize(static_cast<size_t>(stacked), nullptr);
            logits_.resize(static_cast<size_t>(stacked), nullptr);
            for (int64_t codebook = 0; codebook < stacked; ++codebook) {
                auto out = build_transformer_cached_tail(
                    build,
                    input,
                    position,
                    owner_.weights->local,
                    hidden,
                    config.local_heads,
                    1,
                    cache_keys_,
                    cache_values_,
                    cache_slot,
                    attention_mask);
                auto logits = modules::LinearModule({hidden, config.all_tokens_per_codebook, true}).build(
                    build,
                    out,
                    owner_.weights->local_out[static_cast<size_t>(codebook)]);
                logits_[static_cast<size_t>(codebook)] = logits.tensor;
                auto * graph = ggml_new_graph_custom(ctx_.get(), 65536, false);
                ggml_set_output(logits.tensor);
                ggml_build_forward_expand(graph, logits.tensor);
                graphs_[static_cast<size_t>(codebook)] = graph;
            }
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), owner_.backend);
            if (buffer_ == nullptr) {
                throw std::runtime_error("MagpieTTS failed to allocate local transformer graph");
            }
            attention_mask_values_.assign(static_cast<size_t>(batch * stacked), ggml_fp32_to_fp16(-INFINITY));
        }

        ~LocalTransformerGraph() {
            for (auto * graph : graphs_) {
                core::release_backend_graph_resources(owner_.backend, graph);
            }
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
                buffer_ = nullptr;
            }
        }

        LocalTransformerGraph(const LocalTransformerGraph &) = delete;
        LocalTransformerGraph & operator=(const LocalTransformerGraph &) = delete;

        void reset() noexcept {
            valid_steps_ = 0;
        }

        Output run(int64_t codebook, const std::vector<float> & embedding) {
            const auto & config = owner_.assets->config;
            constexpr int64_t batch = 2;
            const int64_t stacked = config.audio_codebooks * config.frame_stacking_factor;
            if (codebook < 0 || codebook >= stacked || codebook != valid_steps_) {
                throw std::runtime_error("MagpieTTS local transformer cached step order mismatch");
            }
            if (valid_steps_ >= stacked) {
                throw std::runtime_error("MagpieTTS local transformer cache exceeds capacity");
            }
            const size_t expected = static_cast<size_t>(batch * config.embedding_dim);
            if (embedding.size() != expected) {
                throw std::runtime_error("MagpieTTS local transformer embedding shape mismatch");
            }
            ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
            const int32_t position = static_cast<int32_t>(valid_steps_);
            ggml_backend_tensor_set(position_, &position, 0, sizeof(position));
            const int32_t cache_slots[2] = {
                static_cast<int32_t>(valid_steps_),
                static_cast<int32_t>(stacked + valid_steps_),
            };
            ggml_backend_tensor_set(cache_slot_, cache_slots, 0, sizeof(cache_slots));
            std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), ggml_fp32_to_fp16(-INFINITY));
            for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
                const size_t offset = static_cast<size_t>(batch_index * stacked);
                for (int64_t step = 0; step <= valid_steps_; ++step) {
                    attention_mask_values_[offset + static_cast<size_t>(step)] = ggml_fp32_to_fp16(0.0F);
                }
            }
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_values_.data(),
                0,
                attention_mask_values_.size() * sizeof(ggml_fp16_t));
            const auto start = Clock::now();
            const ggml_status status = core::compute_backend_graph(owner_.backend, graphs_[static_cast<size_t>(codebook)]);
            const double compute_ms = debug::elapsed_ms(start);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MagpieTTS local transformer graph compute failed");
            }
            Output output{
                read_exact_f32_tensor(
                    logits_[static_cast<size_t>(codebook)],
                    static_cast<size_t>(batch * config.all_tokens_per_codebook),
                    "MagpieTTS local transformer logits"),
                compute_ms,
            };
            ++valid_steps_;
            return output;
        }

    private:
        Impl & owner_;
        int64_t valid_steps_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * position_ = nullptr;
        ggml_tensor * cache_slot_ = nullptr;
        ggml_tensor * attention_mask_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
        std::vector<core::TensorValue> cache_keys_;
        std::vector<core::TensorValue> cache_values_;
        std::vector<ggml_cgraph *> graphs_;
        std::vector<ggml_tensor *> logits_;
        std::vector<ggml_fp16_t> attention_mask_values_;
    };

    Impl(
        std::shared_ptr<const MagpieTTSAssets> input_assets,
        core::ExecutionContext & execution,
        MagpieTTSRuntimeOptions input_options)
        : assets(std::move(input_assets)),
          backend(execution.backend()),
          backend_type(execution.backend_type()),
          options(input_options),
          arena(backend) {
        if (assets == nullptr) {
            throw std::runtime_error("MagpieTTS runtime requires assets");
        }
        if (assets->model_weights == nullptr || assets->codec_weights == nullptr) {
            throw std::runtime_error("MagpieTTS runtime requires model and codec tensor sources");
        }
        if (backend_type == core::BackendType::Cuda) {
            sampling_policy = sampling::resolve_torch_cuda_sampling_policy(
                backend_type,
                execution.config().device,
                "magpie_tts.sampling",
                "MagpieTTS",
                sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
        }
        weights = std::make_shared<ModelWeights>(
            load_model_weights(*assets, backend, backend_type, options));
        modules::NemoNanoCodecRuntimeOptions codec_options;
        codec_options.graph_arena_bytes = options.graph_arena_bytes;
        codec_options.weight_context_bytes = options.weight_context_bytes;
        codec_options.weight_storage_type = options.conv_weight_storage_type;
        codec = std::make_unique<modules::NemoNanoCodecRuntime>(
            assets->codec_weights,
            execution,
            make_nemo_nano_codec_config(assets->config),
            codec_options);
        const int64_t stacked = assets->config.audio_codebooks * assets->config.frame_stacking_factor;
        local_graph = std::make_unique<LocalTransformerGraph>(*this);
        audio_embedding_host.reserve(static_cast<size_t>(stacked));
        for (int64_t index = 0; index < stacked; ++index) {
            audio_embedding_host.push_back(assets->model_weights->require_f32(
                "audio_embeddings." + std::to_string(index) + ".weight",
                std::vector<int64_t>{assets->config.all_tokens_per_codebook, assets->config.embedding_dim}));
        }
        baked_context_host_cache = assets->model_weights->require_f32(
            "baked_context_embedding.weight",
            std::vector<int64_t>{
                assets->config.speakers,
                assets->config.context_length * assets->config.context_dim});
    }

    std::vector<float> encode_text(const std::vector<int32_t> & tokens) {
        const auto & config = assets->config;
        const int64_t steps = static_cast<int64_t>(tokens.size());
        ggml_init_params params{options.graph_arena_bytes, nullptr, true};
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("MagpieTTS failed to create text encoder graph context");
        }
        core::ModuleBuildContext build{ctx.get(), "magpie_tts.encoder", backend_type};
        auto token_ids = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({1, steps}));
        auto mask = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({1, steps}));
        auto attention_mask_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, steps, steps, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_tensor,
            core::TensorShape::from_dims({1, 1, steps, steps}),
            GGML_TYPE_F16);
        auto embedded = modules::EmbeddingModule({config.text_vocab_size, config.embedding_dim}).build(build, token_ids, weights->text_embedding);
        auto encoded = build_transformer(build, embedded, mask, weights->encoder, config.embedding_dim, 12, true, 3, attention_mask);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 131072, false);
        ggml_build_forward_expand(graph, encoded.tensor);
        arena.allocate(graph);
        const auto mask_values = make_mask(1, steps, steps);
        const auto attention_mask_values = modules::qwen_causal_prefill_mask_values(1, steps);
        ggml_backend_tensor_set(token_ids.tensor, tokens.data(), 0, tokens.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask.tensor, mask_values.data(), 0, mask_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            attention_mask_tensor,
            attention_mask_values.data(),
            0,
            attention_mask_values.size() * sizeof(ggml_fp16_t));
        const auto start = Clock::now();
        core::compute_backend_graph(backend, graph);
        debug::timing_log_scalar("magpie_tts.encoder.graph.compute_ms", debug::elapsed_ms(start));
        auto out = read_exact_f32_tensor(
            encoded.tensor,
            static_cast<size_t>(steps * config.embedding_dim),
            "MagpieTTS encoder output");
        core::release_backend_graph_resources(backend, graph);
        return out;
    }

    DecoderStepOutput decoder_step(
        const DecoderConditioning & conditioning,
        const DecoderCrossCacheView & cross_cache,
        const std::vector<float> & audio_embedded,
        int64_t audio_steps,
        const std::vector<float> & context,
        const std::vector<float> * attention_prior) {
        const auto & config = assets->config;
        const int64_t batch = 2;
        const int64_t prefix = config.context_length;
        const int64_t steps = prefix + audio_steps;
        const int64_t hidden = config.embedding_dim;
        std::vector<float> decoder_input(static_cast<size_t>(batch * steps * hidden), 0.0F);
        std::copy(context.begin(), context.end(), decoder_input.begin());
        std::copy(audio_embedded.begin(), audio_embedded.end(), decoder_input.begin() + static_cast<size_t>(prefix * hidden));
        std::copy(audio_embedded.begin(), audio_embedded.end(), decoder_input.begin() + static_cast<size_t>(steps * hidden + prefix * hidden));
        const int64_t text_steps = conditioning.text_steps;

        ggml_init_params params{options.graph_arena_bytes, nullptr, true};
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("MagpieTTS failed to create decoder graph context");
        }
        core::ModuleBuildContext build{ctx.get(), "magpie_tts.decoder", backend_type};
        auto input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({batch, steps, hidden}));
        auto input_mask = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({batch, steps}));
        auto self_attention_mask_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, steps, steps, 1, batch);
        auto self_attention_mask = core::wrap_tensor(
            self_attention_mask_tensor,
            core::TensorShape::from_dims({batch, 1, steps, steps}),
            GGML_TYPE_F16);
        auto cond_mask = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({batch, text_steps}));
        std::vector<float> prior_values;
        std::optional<core::TensorValue> prior_tensor;
        if (attention_prior != nullptr) {
            prior_values.assign(static_cast<size_t>(batch * text_steps), config.attention_prior_epsilon);
            std::copy(attention_prior->begin(), attention_prior->end(), prior_values.begin());
            prior_tensor = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, 1, text_steps}));
        }
        std::vector<core::TensorValue> alignment_outputs;
        auto decoded = build_transformer(
            build,
            input,
            input_mask,
            weights->decoder,
            hidden,
            config.decoder_heads,
            true,
            1,
            self_attention_mask,
            std::nullopt,
            cond_mask,
            &cross_cache,
            prior_tensor.has_value() ? &*prior_tensor : nullptr,
            &config.apply_prior_to_layers,
            &config.estimate_alignment_from_layers,
            &alignment_outputs,
            config.decoder_cross_heads,
            config.decoder_cross_head_dim);
        auto last = modules::SliceModule({1, steps - 1, 1}).build(build, decoded);
        auto logits = modules::LinearModule({
            hidden,
            config.audio_codebooks * config.frame_stacking_factor * config.all_tokens_per_codebook,
            true,
        }).build(build, last, weights->final_proj);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 262144, false);
        for (const auto & alignment : alignment_outputs) {
            ggml_set_output(alignment.tensor);
            ggml_build_forward_expand(graph, alignment.tensor);
        }
        ggml_build_forward_expand(graph, logits.tensor);
        ggml_build_forward_expand(graph, last.tensor);
        arena.allocate(graph);
        auto input_mask_values = make_mask(batch, steps, steps);
        auto self_attention_mask_values = modules::qwen_causal_prefill_mask_values(batch, steps);
        ggml_backend_tensor_set(input.tensor, decoder_input.data(), 0, decoder_input.size() * sizeof(float));
        ggml_backend_tensor_set(input_mask.tensor, input_mask_values.data(), 0, input_mask_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            self_attention_mask_tensor,
            self_attention_mask_values.data(),
            0,
            self_attention_mask_values.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(cond_mask.tensor, conditioning.mask.data(), 0, conditioning.mask.size() * sizeof(int32_t));
        if (attention_prior != nullptr) {
            ggml_backend_tensor_set(prior_tensor->tensor, prior_values.data(), 0, prior_values.size() * sizeof(float));
        }
        const auto start = Clock::now();
        core::compute_backend_graph(backend, graph);
        const double compute_ms = debug::elapsed_ms(start);
        DecoderStepOutput output;
        output.compute_ms = compute_ms;
        output.hidden = read_exact_f32_tensor(
            last.tensor,
            static_cast<size_t>(batch * hidden),
            "MagpieTTS decoder hidden");
        output.logits = read_exact_f32_tensor(
            logits.tensor,
            static_cast<size_t>(batch * config.audio_codebooks * config.frame_stacking_factor * config.all_tokens_per_codebook),
            "MagpieTTS decoder logits");
        if (!alignment_outputs.empty()) {
            output.alignment_scores.assign(static_cast<size_t>(text_steps), 0.0F);
            int64_t layer_count = 0;
            for (const auto & alignment : alignment_outputs) {
                const auto values = core::read_tensor_f32(alignment.tensor);
                const int64_t heads = alignment.shape.dims[1];
                if (alignment.shape.dims[0] != batch || alignment.shape.dims[2] != 1 ||
                    alignment.shape.dims[3] != text_steps ||
                    static_cast<int64_t>(values.size()) != batch * heads * text_steps) {
                    throw std::runtime_error("MagpieTTS decoder alignment shape is invalid");
                }
                for (int64_t head = 0; head < heads; ++head) {
                    const size_t base = static_cast<size_t>(head * text_steps);
                    for (int64_t token = 0; token < text_steps; ++token) {
                        output.alignment_scores[static_cast<size_t>(token)] +=
                            values[base + static_cast<size_t>(token)];
                    }
                }
                ++layer_count;
            }
            const float scale = 1.0F / static_cast<float>(layer_count * config.decoder_cross_heads);
            for (float & score : output.alignment_scores) {
                score *= scale;
            }
        }
        core::release_backend_graph_resources(backend, graph);
        return output;
    }

    std::vector<int32_t> sample_local(
        const std::vector<float> & dec_hidden,
        const MagpieTTSGenerationOptions & generation,
        uint64_t & sample_call,
        double & local_compute_ms,
        bool forbid_audio_eos,
        bool force_audio_eos) {
        const auto & config = assets->config;
        const int64_t hidden = config.embedding_dim;
        const int64_t stacked = config.audio_codebooks * config.frame_stacking_factor;
        local_graph->reset();
        std::vector<float> step_input = dec_hidden;
        std::vector<int32_t> preds(static_cast<size_t>(stacked), 0);
        sampling::HfSamplerScratch scratch;
        std::mt19937 fallback(static_cast<uint32_t>(generation.seed));
        for (int64_t codebook = 0; codebook < stacked; ++codebook) {
            auto local_output = local_graph->run(codebook, step_input);
            local_compute_ms += local_output.compute_ms;
            const auto & logits_host = local_output.logits;
            const int64_t vocab = config.all_tokens_per_codebook;
            std::vector<float> scores(static_cast<size_t>(vocab));
            for (int64_t token = 0; token < vocab; ++token) {
                const float cond = logits_host[static_cast<size_t>(token)];
                const float uncond = logits_host[static_cast<size_t>(vocab + token)];
                scores[static_cast<size_t>(token)] = generation.guidance_scale * cond + (1.0F - generation.guidance_scale) * uncond;
            }
            for (int64_t token = config.codebook_size; token < vocab; ++token) {
                if (token != config.codebook_size + 1 || forbid_audio_eos) {
                    scores[static_cast<size_t>(token)] = -std::numeric_limits<float>::infinity();
                }
            }
            if (force_audio_eos) {
                std::fill(scores.begin(), scores.end(), -std::numeric_limits<float>::infinity());
                scores[static_cast<size_t>(config.codebook_size + 1)] = 0.0F;
            }
            sampling::HfLogitsProcessor::apply_top_k(scores, generation.top_k, 1, scratch);
            int32_t token = 0;
            if (generation.temperature <= 0.0F) {
                token = static_cast<int32_t>(
                    std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
            } else {
                sampling::HfLogitsProcessor::apply_temperature(scores, generation.temperature);
                sampling::HfTorchSamplingState torch_state;
                torch_state.policy = &sampling_policy;
                torch_state.seed = generation.seed;
                torch_state.call_index = sample_call;
                token = sampling::HfTokenSampler::sample_from_processed_scores(
                    scores,
                    scratch,
                    fallback,
                    &torch_state,
                    "MagpieTTS local transformer");
            }
            ++sample_call;
            preds[static_cast<size_t>(codebook)] = token;
            const auto emb = audio_embedding_row(codebook, token);
            std::copy(emb.begin(), emb.end(), step_input.begin());
            std::copy(emb.begin(), emb.end(), step_input.begin() + hidden);
        }
        return preds;
    }

    std::vector<int32_t> decoder_argmax_tokens(
        const std::vector<float> & logits,
        const MagpieTTSGenerationOptions & generation,
        bool forbid_audio_eos,
        bool force_audio_eos) const {
        const auto & config = assets->config;
        const int64_t stacked = config.audio_codebooks * config.frame_stacking_factor;
        const int64_t vocab = config.all_tokens_per_codebook;
        if (static_cast<int64_t>(logits.size()) != 2 * stacked * vocab) {
            throw std::runtime_error("MagpieTTS decoder logits shape is invalid");
        }
        std::vector<int32_t> out(static_cast<size_t>(stacked), 0);
        for (int64_t codebook = 0; codebook < stacked; ++codebook) {
            if (force_audio_eos) {
                out[static_cast<size_t>(codebook)] = static_cast<int32_t>(config.codebook_size + 1);
                continue;
            }
            int32_t best_token = 0;
            float best_score = -std::numeric_limits<float>::infinity();
            for (int64_t token = 0; token < vocab; ++token) {
                if (token >= config.codebook_size &&
                    (token != config.codebook_size + 1 || forbid_audio_eos)) {
                    continue;
                }
                const size_t cond_index = static_cast<size_t>(codebook * vocab + token);
                const size_t uncond_index = static_cast<size_t>(stacked * vocab + codebook * vocab + token);
                const float score =
                    generation.guidance_scale * logits[cond_index] +
                    (1.0F - generation.guidance_scale) * logits[uncond_index];
                if (score > best_score) {
                    best_score = score;
                    best_token = static_cast<int32_t>(token);
                }
            }
            out[static_cast<size_t>(codebook)] = best_token;
        }
        return out;
    }

    std::vector<float> embed_audio_stack(const std::vector<int32_t> & stack_codes) const {
        const auto & config = assets->config;
        const int64_t stacked = config.audio_codebooks * config.frame_stacking_factor;
        if (static_cast<int64_t>(stack_codes.size()) != stacked) {
            throw std::runtime_error("MagpieTTS audio stack code shape is invalid");
        }
        std::vector<float> out(static_cast<size_t>(config.embedding_dim), 0.0F);
        const float scale = 1.0F / static_cast<float>(stacked);
        for (int64_t index = 0; index < stacked; ++index) {
            const auto emb = audio_embedding_row(index, stack_codes[static_cast<size_t>(index)]);
            for (int64_t h = 0; h < config.embedding_dim; ++h) {
                out[static_cast<size_t>(h)] += emb[static_cast<size_t>(h)] * scale;
            }
        }
        return out;
    }

    runtime::AudioBuffer synthesize_short(
        const MagpieTokenizationResult & tokenized,
        const MagpieTTSGenerationOptions & generation) {
        const auto & config = assets->config;
        const auto text_encoded = encode_text(tokenized.tokens);
        const auto conditioning = make_decoder_conditioning(
            text_encoded,
            static_cast<int64_t>(tokenized.tokens.size()),
            config.embedding_dim);
        DecoderCrossCacheGraph cross_cache_graph(*this, conditioning);
        const DecoderCrossCacheView cross_cache = cross_cache_graph.view();
        const auto context = baked_context_host(baked_context_host_cache, config, generation.speaker);
        std::vector<int32_t> codes;
        codes.reserve(static_cast<size_t>(generation.max_tokens * config.audio_codebooks));
        std::vector<int32_t> initial_stack;
        initial_stack.reserve(static_cast<size_t>(config.frame_stacking_factor * config.audio_codebooks));
        for (int64_t fs = 0; fs < config.frame_stacking_factor; ++fs) {
            for (int64_t cb = 0; cb < config.audio_codebooks; ++cb) {
                initial_stack.push_back(static_cast<int32_t>(config.codebook_size));
            }
        }
        codes = initial_stack;
        std::vector<float> audio_embedded = embed_audio_stack(initial_stack);
        uint64_t sample_call = 0;
        double decoder_compute_ms = 0.0;
        double local_compute_ms = 0.0;
        AttentionPriorState attention_prior;
        attention_prior.reset(static_cast<int64_t>(tokenized.tokens.size()));
        const int64_t max_steps = std::min<int64_t>(generation.max_tokens, config.max_decoder_steps) / config.frame_stacking_factor;
        for (int64_t step = 0; step < max_steps; ++step) {
            const std::vector<float> * prior = attention_prior.prior_for_step(config);
            auto decoder = decoder_step(
                conditioning,
                cross_cache,
                audio_embedded,
                static_cast<int64_t>(audio_embedded.size() / config.embedding_dim),
                context,
                prior);
            decoder_compute_ms += decoder.compute_ms;
            attention_prior.update(config, step, decoder.alignment_scores);
            const bool forbid_eos = attention_prior.should_forbid_eos(
                step * config.frame_stacking_factor < config.min_generated_frames,
                config.ignore_finished_sentence_tracking);
            const bool force_eos = attention_prior.should_force_eos(config.ignore_finished_sentence_tracking);
            auto next = sample_local(decoder.hidden, generation, sample_call, local_compute_ms, forbid_eos, force_eos);
            auto decoder_argmax = decoder_argmax_tokens(decoder.logits, generation, forbid_eos, force_eos);
            bool hit_eos = false;
            int64_t eos_frame = config.frame_stacking_factor;
            for (int64_t fs = 0; fs < config.frame_stacking_factor; ++fs) {
                for (int64_t cb = 0; cb < config.audio_codebooks; ++cb) {
                    const size_t code_index = static_cast<size_t>(fs * config.audio_codebooks + cb);
                    if (next[code_index] == config.codebook_size + 1 ||
                        decoder_argmax[code_index] == config.codebook_size + 1) {
                        hit_eos = true;
                        eos_frame = std::min(eos_frame, fs);
                    }
                }
            }
            std::vector<int32_t> appended_stack;
            appended_stack.reserve(static_cast<size_t>(config.frame_stacking_factor * config.audio_codebooks));
            for (int64_t fs = 0; fs < eos_frame; ++fs) {
                for (int64_t cb = 0; cb < config.audio_codebooks; ++cb) {
                    const size_t code_index = static_cast<size_t>(fs * config.audio_codebooks + cb);
                    const int32_t token = std::min<int32_t>(next[code_index], static_cast<int32_t>(config.codebook_size - 1));
                    codes.push_back(token);
                    appended_stack.push_back(token);
                }
            }
            if (hit_eos && step > 3) {
                attention_prior.mark_ended();
                break;
            }
            if (static_cast<int64_t>(appended_stack.size()) == config.frame_stacking_factor * config.audio_codebooks) {
                auto stack_embedding = embed_audio_stack(appended_stack);
                audio_embedded.insert(audio_embedded.end(), stack_embedding.begin(), stack_embedding.end());
            }
        }
        debug::timing_log_scalar("magpie_tts.decoder.graph.compute_ms", decoder_compute_ms);
        debug::timing_log_scalar("magpie_tts.local_transformer.graph.compute_ms", local_compute_ms);
        const size_t bos_codes = static_cast<size_t>(config.frame_stacking_factor * config.audio_codebooks);
        std::vector<int32_t> predicted_codes(codes.begin() + static_cast<std::ptrdiff_t>(bos_codes), codes.end());
        debug::trace_log_scalar("magpie_tts.generated.frames", static_cast<int64_t>(predicted_codes.size() / config.audio_codebooks));
        return codec->decode_codes(predicted_codes);
    }

    std::vector<float> audio_embedding_row(int64_t table, int32_t token) const {
        const auto & config = assets->config;
        if (table < 0 || table >= static_cast<int64_t>(audio_embedding_host.size()) ||
            token < 0 || token >= config.all_tokens_per_codebook) {
            throw std::runtime_error("MagpieTTS audio embedding lookup is out of range");
        }
        const auto & weight = audio_embedding_host[static_cast<size_t>(table)];
        return std::vector<float>(
            weight.begin() + static_cast<int64_t>(token) * config.embedding_dim,
            weight.begin() + (static_cast<int64_t>(token) + 1) * config.embedding_dim);
    }

    std::shared_ptr<const MagpieTTSAssets> assets;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    MagpieTTSRuntimeOptions options;
    sampling::TorchCudaSamplingPolicy sampling_policy;
    GraphArena arena;
    std::shared_ptr<ModelWeights> weights;
    std::unique_ptr<modules::NemoNanoCodecRuntime> codec;
    std::unique_ptr<LocalTransformerGraph> local_graph;
    std::vector<std::vector<float>> audio_embedding_host;
    std::vector<float> baked_context_host_cache;
};

MagpieTTSRuntime::MagpieTTSRuntime(
    std::shared_ptr<const MagpieTTSAssets> assets,
    core::ExecutionContext & execution,
    MagpieTTSRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, options)) {}

MagpieTTSRuntime::~MagpieTTSRuntime() = default;

runtime::AudioBuffer MagpieTTSRuntime::synthesize(
    const MagpieTokenizationResult & tokenized,
    const MagpieTTSGenerationOptions & options) {
    if (tokenized.tokens.empty()) {
        throw std::runtime_error("MagpieTTS runtime requires non-empty text tokens");
    }
    const auto start = Clock::now();
    runtime::AudioBuffer audio;
    if (tokenized.chunks.size() <= 1) {
        audio = impl_->synthesize_short(tokenized, options);
    } else {
        audio.sample_rate = static_cast<int>(impl_->assets->config.sample_rate);
        audio.channels = 1;
        for (const auto & chunk : tokenized.chunks) {
            MagpieTokenizationResult chunked;
            chunked.language = tokenized.language;
            chunked.tokenizer_name = tokenized.tokenizer_name;
            chunked.tokens = chunk.tokens;
            chunked.chunks.push_back(chunk);
            auto chunk_audio = impl_->synthesize_short(chunked, options);
            if (chunk_audio.sample_rate != audio.sample_rate || chunk_audio.channels != audio.channels) {
                throw std::runtime_error("MagpieTTS chunk output audio format changed");
            }
            audio.samples.insert(audio.samples.end(), chunk_audio.samples.begin(), chunk_audio.samples.end());
        }
    }
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
    return audio;
}

}  // namespace engine::models::magpie_tts
