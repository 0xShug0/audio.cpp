#include "engine/models/breeze_tts/generator.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/breeze_tts/speech_decoder.h"
#include "engine/models/breeze_tts/speech_encoder.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::models::breeze_tts {
namespace {

namespace assets = engine::assets;
namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
namespace runtime = engine::runtime;
namespace sampling = engine::sampling;
using Clock = std::chrono::steady_clock;

constexpr int64_t kCodecCodebookSize = 2048;
constexpr float kRepetitionPenalty = 1.1F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::QwenCausalDecodeRuntimeConfig backbone_config(
    const BreezeTTSConfig & config,
    core::BackendType backend_type,
    size_t graph_arena_bytes) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = "breeze_tts.backbone";
    out.prefill_graph_arena_bytes = graph_arena_bytes;
    out.decode_graph_arena_bytes = graph_arena_bytes;
    out.decoder.stack.hidden_size = config.hidden_size;
    out.decoder.stack.num_attention_heads = config.heads;
    out.decoder.stack.num_key_value_heads = config.kv_heads;
    out.decoder.stack.head_dim = config.head_dim;
    out.decoder.stack.intermediate_size = config.intermediate_size;
    out.decoder.stack.layers = config.layers;
    out.decoder.stack.rms_norm_eps = config.rms_norm_eps;
    out.decoder.stack.rope_theta = config.rope_theta;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.use_qk_norm = true;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    if (backend_type == core::BackendType::Cuda || backend_type == core::BackendType::Hip ||
        backend_type == core::BackendType::Vulkan) {
        out.decoder.static_cache_type = GGML_TYPE_F16;
    }
    out.decoder.logits_size = config.lm_head_size;
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.output_mode = modules::QwenCausalDecodeOutputMode::Logits;
    out.return_hidden = true;
    out.logits_readback_token_ids.reserve(static_cast<size_t>(config.lm_head_size));
    for (int32_t token = 0; token < static_cast<int32_t>(config.lm_head_size); ++token) {
        out.logits_readback_token_ids.push_back(token);
    }
    out.decoder.lm_head_input_type = GGML_TYPE_F32;
    return out;
}

modules::QwenCausalDecodeRuntimeConfig depth_config(
    const BreezeTTSConfig & config,
    core::BackendType backend_type,
    size_t graph_arena_bytes) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = "breeze_tts.depth_decoder";
    out.prefill_graph_arena_bytes = graph_arena_bytes;
    out.decode_graph_arena_bytes = graph_arena_bytes;
    out.decoder.stack.hidden_size = config.depth_hidden_size;
    out.decoder.stack.num_attention_heads = config.depth_heads;
    out.decoder.stack.num_key_value_heads = config.depth_kv_heads;
    out.decoder.stack.head_dim = config.depth_head_dim;
    out.decoder.stack.intermediate_size = config.depth_intermediate_size;
    out.decoder.stack.layers = config.depth_layers;
    out.decoder.stack.rms_norm_eps = config.depth_rms_norm_eps;
    out.decoder.stack.rope_theta = config.depth_rope_theta;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.use_qk_norm = false;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    if (backend_type == core::BackendType::Cuda || backend_type == core::BackendType::Hip ||
        backend_type == core::BackendType::Vulkan) {
        out.decoder.static_cache_type = GGML_TYPE_F16;
    }
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.output_mode = modules::QwenCausalDecodeOutputMode::Hidden;
    out.return_hidden = true;
    return out;
}

std::vector<float> llama3_rope_factors(
    int64_t head_dim,
    float rope_theta,
    float scaling_factor,
    float low_freq_factor,
    float high_freq_factor,
    int64_t original_max_position_embeddings) {
    constexpr double pi = 3.14159265358979323846;
    const double low_wavelength =
        static_cast<double>(original_max_position_embeddings) /
        static_cast<double>(low_freq_factor);
    const double high_wavelength =
        static_cast<double>(original_max_position_embeddings) /
        static_cast<double>(high_freq_factor);
    std::vector<float> out(static_cast<size_t>(head_dim / 2), 1.0F);
    for (int64_t index = 0; index < head_dim / 2; ++index) {
        const double inv_freq = 1.0 / std::pow(
            static_cast<double>(rope_theta),
            static_cast<double>(2 * index) / static_cast<double>(head_dim));
        const double wavelength = 2.0 * pi / inv_freq;
        double scaled = inv_freq;
        if (wavelength > low_wavelength) {
            scaled = inv_freq / static_cast<double>(scaling_factor);
        } else if (wavelength >= high_wavelength) {
            const double smooth =
                (static_cast<double>(original_max_position_embeddings) / wavelength -
                 static_cast<double>(low_freq_factor)) /
                (static_cast<double>(high_freq_factor) -
                 static_cast<double>(low_freq_factor));
            scaled =
                (1.0 - smooth) * inv_freq / static_cast<double>(scaling_factor) +
                smooth * inv_freq;
        }
        out[static_cast<size_t>(index)] = static_cast<float>(inv_freq / scaled);
    }
    return out;
}

modules::QwenDecoderLayerWeights load_backbone_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const BreezeTTSConfig & config,
    assets::TensorStorageType storage_type,
    const std::optional<core::TensorValue> & rope_factors,
    int64_t layer) {
    const std::string prefix = "backbone_model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(store, source, prefix + ".mlp.gate_proj", storage_type, config.intermediate_size, config.hidden_size, false);
    out.mlp.up_proj = binding::linear_from_source(store, source, prefix + ".mlp.up_proj", storage_type, config.intermediate_size, config.hidden_size, false);
    out.mlp.down_proj = binding::linear_from_source(store, source, prefix + ".mlp.down_proj", storage_type, config.hidden_size, config.intermediate_size, false);
    out.rope_frequency_factors = rope_factors;
    return out;
}

modules::QwenDecoderLayerWeights load_depth_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const BreezeTTSConfig & config,
    assets::TensorStorageType storage_type,
    const std::optional<core::TensorValue> & rope_factors,
    int64_t layer) {
    const std::string prefix = "depth_decoder.model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.depth_hidden_size);
    out.self_attention.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.depth_heads * config.depth_head_dim, config.depth_hidden_size});
    out.self_attention.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.depth_kv_heads * config.depth_head_dim, config.depth_hidden_size});
    out.self_attention.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.depth_kv_heads * config.depth_head_dim, config.depth_hidden_size});
    out.self_attention.out_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.depth_hidden_size, config.depth_heads * config.depth_head_dim});
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.depth_hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(store, source, prefix + ".mlp.gate_proj", storage_type, config.depth_intermediate_size, config.depth_hidden_size, false);
    out.mlp.up_proj = binding::linear_from_source(store, source, prefix + ".mlp.up_proj", storage_type, config.depth_intermediate_size, config.depth_hidden_size, false);
    out.mlp.down_proj = binding::linear_from_source(store, source, prefix + ".mlp.down_proj", storage_type, config.depth_hidden_size, config.depth_intermediate_size, false);
    out.rope_frequency_factors = rope_factors;
    return out;
}

std::vector<float> frame_embedding(
    const std::vector<float> & table,
    int64_t rows,
    int64_t dim,
    int64_t vocab,
    const std::vector<int32_t> & codes) {
    std::vector<float> out(static_cast<size_t>(dim), 0.0F);
    for (int64_t codebook = 0; codebook < static_cast<int64_t>(codes.size()); ++codebook) {
        const int64_t row = codebook * vocab + codes[static_cast<size_t>(codebook)];
        if (row < 0 || row >= rows) {
            throw std::runtime_error("BreezeTTS audio code is outside embedding table");
        }
        const size_t begin = static_cast<size_t>(row * dim);
        for (int64_t i = 0; i < dim; ++i) {
            out[static_cast<size_t>(i)] += table[begin + static_cast<size_t>(i)];
        }
    }
    return out;
}

void suppress_reserved(std::vector<float> & logits, int64_t codebook_size, int64_t vocab_size) {
    for (int64_t token = codebook_size; token < vocab_size; ++token) {
        if (token >= 0 && token < static_cast<int64_t>(logits.size())) {
            logits[static_cast<size_t>(token)] = -std::numeric_limits<float>::infinity();
        }
    }
}

int32_t sample_logits(
    std::vector<float> logits,
    const std::vector<int32_t> & history,
    const sampling::HfSamplingOptions & options,
    sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const sampling::TorchCudaSamplingPolicy * policy,
    uint64_t seed,
    uint64_t & call_index,
    uint64_t & offset_blocks,
    std::string_view context) {
    const sampling::HfTorchSamplingState torch_state{policy, seed, call_index, offset_blocks, true};
    const int32_t token = sampling::HfSampler{}.sample(
        logits,
        history,
        options,
        scratch,
        fallback_rng,
        policy != nullptr && policy->cuda_fast_path ? &torch_state : nullptr,
        context);
    ++call_index;
    if (policy != nullptr && policy->cuda_fast_path) {
        offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(static_cast<uint64_t>(logits.size()), *policy);
    }
    return token;
}

struct BreezeWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::QwenCausalDecodeRuntimeWeights backbone;
    modules::QwenCausalDecodeRuntimeWeights depth;
    std::vector<float> audio_embedding;
    modules::LinearWeights depth_projector;
    core::TensorValue depth_heads;
};

std::shared_ptr<const BreezeWeights> load_weights(
    const BreezeTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type,
    const modules::QwenCausalDecodeRuntimeConfig & backbone_runtime_config) {
    auto out = std::make_shared<BreezeWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "breeze_tts.generator.weights",
        weight_context_bytes);
    const auto & source = *assets.weights;
    const auto & config = assets.config;
    std::optional<core::TensorValue> backbone_rope_factors;
    if (config.rope_scaling_enabled) {
        backbone_rope_factors = out->store->make_f32(
            core::TensorShape::from_dims({config.head_dim / 2}),
            llama3_rope_factors(
                config.head_dim,
                config.rope_theta,
                config.rope_scaling_factor,
                config.rope_low_freq_factor,
                config.rope_high_freq_factor,
                config.rope_original_max_position_embeddings));
    }
    std::optional<core::TensorValue> depth_rope_factors;
    if (config.depth_rope_scaling_enabled) {
        depth_rope_factors = out->store->make_f32(
            core::TensorShape::from_dims({config.depth_head_dim / 2}),
            llama3_rope_factors(
                config.depth_head_dim,
                config.depth_rope_theta,
                config.depth_rope_scaling_factor,
                config.depth_rope_low_freq_factor,
                config.depth_rope_high_freq_factor,
                config.depth_rope_original_max_position_embeddings));
    }
    out->backbone.token_embedding = out->store->load_tensor(
        source,
        "depth_decoder.model.embed_tokens.weight",
        storage_type,
        {config.num_codebooks * config.vocab_size, config.hidden_size});
    out->backbone.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        out->backbone.stack.layers.push_back(load_backbone_layer(*out->store, source, config, storage_type, backbone_rope_factors, layer));
    }
    out->backbone.final_norm = binding::norm_weight_from_source(*out->store, source, "backbone_model.norm", config.hidden_size);
    out->backbone.lm_head = binding::linear_from_source(
        *out->store,
        source,
        "lm_head",
        storage_type,
        backbone_runtime_config.decoder.logits_size,
        config.hidden_size,
        false);

    out->depth.token_embedding = out->backbone.token_embedding;
    out->depth.stack.layers.reserve(static_cast<size_t>(config.depth_layers));
    for (int64_t layer = 0; layer < config.depth_layers; ++layer) {
        out->depth.stack.layers.push_back(load_depth_layer(*out->store, source, config, storage_type, depth_rope_factors, layer));
    }
    out->depth.final_norm = binding::norm_weight_from_source(*out->store, source, "depth_decoder.model.norm", config.depth_hidden_size);

    out->audio_embedding = source.require_f32("depth_decoder.model.embed_tokens.weight", {config.num_codebooks * config.vocab_size, config.hidden_size});
    out->depth_projector = {
        out->store->load_f32_tensor(
            source,
            "depth_decoder.model.inputs_embeds_projector.weight",
            {config.depth_hidden_size, config.hidden_size}),
        std::nullopt};
    const auto depth_heads = source.require_f32(
        "depth_decoder.codebooks_head.weight",
        {config.num_codebooks - 1, config.depth_hidden_size, config.vocab_size});
    std::vector<float> transposed_depth_heads(
        static_cast<size_t>((config.num_codebooks - 1) * config.vocab_size * config.depth_hidden_size));
    for (int64_t codebook = 0; codebook < config.num_codebooks - 1; ++codebook) {
        const int64_t in_codebook_offset = codebook * config.depth_hidden_size * config.vocab_size;
        const int64_t out_codebook_offset = codebook * config.vocab_size * config.depth_hidden_size;
        for (int64_t row = 0; row < config.depth_hidden_size; ++row) {
            const int64_t in_row_offset = in_codebook_offset + row * config.vocab_size;
            for (int64_t token = 0; token < config.vocab_size; ++token) {
                transposed_depth_heads[static_cast<size_t>(out_codebook_offset + token * config.depth_hidden_size + row)] =
                    depth_heads[static_cast<size_t>(in_row_offset + token)];
            }
        }
    }
    out->depth_heads = out->store->make_f32(
        core::TensorShape::from_dims({config.num_codebooks - 1, config.vocab_size, config.depth_hidden_size}),
        transposed_depth_heads);
    out->store->upload();
    return out;
}

class BreezeDepthProjectionRuntime {
public:
    BreezeDepthProjectionRuntime(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes,
        const BreezeTTSConfig & config,
        const modules::LinearWeights & projector_weights,
        const core::TensorValue & packed_heads)
        : backend_(backend),
          threads_(std::max(1, threads)),
          hidden_(config.hidden_size),
          depth_hidden_(config.depth_hidden_size),
          vocab_(config.vocab_size) {
        if (backend_ == nullptr || hidden_ <= 0 || depth_hidden_ <= 0 || vocab_ <= 0 || config.num_codebooks <= 1) {
            throw std::runtime_error("BreezeTTS depth projection shape is invalid");
        }
        ctx_.reset(ggml_init({graph_arena_bytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize BreezeTTS depth projection graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "breeze_tts.depth_projection", backend_type};
        projector_single_ = build_linear_graph(
            ctx,
            1,
            hidden_,
            depth_hidden_,
            projector_weights,
            "breeze_tts.depth_projector.single");
        projector_pair_ = build_linear_graph(
            ctx,
            2,
            hidden_,
            depth_hidden_,
            projector_weights,
            "breeze_tts.depth_projector.pair");
        head_graphs_.reserve(static_cast<size_t>(config.num_codebooks - 1));
        for (int64_t codebook = 1; codebook < config.num_codebooks; ++codebook) {
            head_graphs_.push_back(build_head_graph(ctx, packed_heads, codebook));
        }
        head_paired_staging_.assign(static_cast<size_t>(2 * vocab_), 0.0F);
        graph_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (graph_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate BreezeTTS depth projection graphs");
        }
    }

    ~BreezeDepthProjectionRuntime() {
        release_graph(projector_single_);
        release_graph(projector_pair_);
        for (const auto & graph : head_graphs_) {
            release_graph(graph);
        }
        if (graph_buffer_ != nullptr) {
            ggml_backend_buffer_free(graph_buffer_);
        }
    }

    void project_single(const float * hidden, float * out) const {
        run_graph_direct(projector_single_, hidden, hidden_, out, depth_hidden_);
    }

    std::vector<float> project_single(const std::vector<float> & hidden) const {
        if (static_cast<int64_t>(hidden.size()) != hidden_) {
            throw std::runtime_error("BreezeTTS depth projection input size mismatch");
        }
        std::vector<float> out(static_cast<size_t>(depth_hidden_));
        project_single(hidden.data(), out.data());
        return out;
    }

    void project_pair(
        const float * cond_hidden,
        const float * uncond_hidden,
        float * out) const {
        ggml_backend_tensor_set(projector_pair_.input, cond_hidden, 0, static_cast<size_t>(hidden_) * sizeof(float));
        ggml_backend_tensor_set(
            projector_pair_.input,
            uncond_hidden,
            static_cast<size_t>(hidden_) * sizeof(float),
            static_cast<size_t>(hidden_) * sizeof(float));
        core::set_backend_threads(backend_, threads_);
        if (core::compute_backend_graph(backend_, projector_pair_.graph, nullptr, projector_pair_.label) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("BreezeTTS depth projection graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        ggml_backend_tensor_get(projector_pair_.output, out, 0, static_cast<size_t>(2 * depth_hidden_) * sizeof(float));
    }

    std::vector<float> project_pair(
        const std::vector<float> & cond_hidden,
        const std::vector<float> & uncond_hidden) const {
        if (static_cast<int64_t>(cond_hidden.size()) != hidden_ ||
            static_cast<int64_t>(uncond_hidden.size()) != hidden_) {
            throw std::runtime_error("BreezeTTS depth projector pair input size mismatch");
        }
        std::vector<float> out(static_cast<size_t>(2 * depth_hidden_));
        project_pair(cond_hidden.data(), uncond_hidden.data(), out.data());
        return out;
    }

    void logits_cfg(
        const float * cond_hidden,
        const float * uncond_hidden,
        int64_t codebook,
        float guidance_scale,
        float * out) const {
        if (codebook <= 0 || static_cast<size_t>(codebook) > head_graphs_.size()) {
            throw std::runtime_error("BreezeTTS depth codebook index is invalid");
        }
        const auto & graph = head_graphs_[static_cast<size_t>(codebook - 1)];
        ggml_backend_tensor_set(graph.input, cond_hidden, 0, static_cast<size_t>(depth_hidden_) * sizeof(float));
        ggml_backend_tensor_set(
            graph.input,
            uncond_hidden,
            static_cast<size_t>(depth_hidden_) * sizeof(float),
            static_cast<size_t>(depth_hidden_) * sizeof(float));
        core::set_backend_threads(backend_, threads_);
        if (core::compute_backend_graph(backend_, graph.graph, nullptr, graph.label) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("BreezeTTS depth projection graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        ggml_backend_tensor_get(graph.output, head_paired_staging_.data(), 0, head_paired_staging_.size() * sizeof(float));
        const size_t vocab = static_cast<size_t>(vocab_);
        for (int64_t token = 0; token < vocab_; ++token) {
            const size_t index = static_cast<size_t>(token);
            out[index] = head_paired_staging_[vocab + index] + guidance_scale * (head_paired_staging_[index] - head_paired_staging_[vocab + index]);
        }
    }

    std::vector<float> logits_cfg(
        const std::vector<float> & cond_hidden,
        const std::vector<float> & uncond_hidden,
        int64_t codebook,
        float guidance_scale) const {
        if (static_cast<int64_t>(cond_hidden.size()) != depth_hidden_ ||
            static_cast<int64_t>(uncond_hidden.size()) != depth_hidden_) {
            throw std::runtime_error("BreezeTTS depth head input size mismatch");
        }
        std::vector<float> out(static_cast<size_t>(vocab_));
        logits_cfg(cond_hidden.data(), uncond_hidden.data(), codebook, guidance_scale, out.data());
        return out;
    }

private:
    struct Graph {
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        int64_t input_size = 0;
        int64_t output_size = 0;
        const char * label = nullptr;
    };

    Graph build_linear_graph(
        core::ModuleBuildContext & ctx,
        int64_t batch,
        int64_t in_features,
        int64_t out_features,
        const modules::LinearWeights & weights,
        const char * label) {
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, in_features}));
        auto output = modules::LinearModule({in_features, out_features, false})
                          .build(ctx, input, weights)
                          .tensor;
        auto * graph = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        return {input.tensor, output, graph, batch * in_features, batch * out_features, label};
    }

    Graph build_head_graph(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & packed_heads,
        int64_t codebook) {
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, depth_hidden_}));
        const size_t row_stride = packed_heads.tensor->nb[1];
        const size_t codebook_stride = packed_heads.tensor->nb[2];
        const size_t offset = static_cast<size_t>(codebook - 1) * codebook_stride;
        auto weight = core::wrap_tensor(
            ggml_view_2d(ctx_.get(), packed_heads.tensor, depth_hidden_, vocab_, row_stride, offset),
            core::TensorShape::from_dims({vocab_, depth_hidden_}),
            packed_heads.type);
        auto output = modules::LinearModule({depth_hidden_, vocab_, false})
                          .build(ctx, input, modules::LinearWeights{weight, std::nullopt})
                          .tensor;
        auto * graph = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        return {input.tensor, output, graph, 2 * depth_hidden_, 2 * vocab_, "breeze_tts.depth_head"};
    }

    void release_graph(const Graph & graph) const {
        core::release_backend_graph_resources(backend_, graph.graph);
    }

    void run_graph_direct(
        const Graph & graph,
        const float * input,
        int64_t in_size,
        float * output,
        int64_t out_size) const {
        if (in_size != graph.input_size || out_size != graph.output_size) {
            throw std::runtime_error("BreezeTTS depth projection input size mismatch");
        }
        ggml_backend_tensor_set(graph.input, input, 0, static_cast<size_t>(in_size) * sizeof(float));
        core::set_backend_threads(backend_, threads_);
        if (core::compute_backend_graph(backend_, graph.graph, nullptr, graph.label) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("BreezeTTS depth projection graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        ggml_backend_tensor_get(graph.output, output, 0, static_cast<size_t>(out_size) * sizeof(float));
    }

    std::vector<float> run_graph(const Graph & graph, const std::vector<float> & input) const {
        if (static_cast<int64_t>(input.size()) != graph.input_size) {
            throw std::runtime_error("BreezeTTS depth projection input size mismatch");
        }
        std::vector<float> output(static_cast<size_t>(graph.output_size));
        run_graph_direct(graph, input.data(), static_cast<int64_t>(input.size()), output.data(), graph.output_size);
        return output;
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    int64_t hidden_ = 0;
    int64_t depth_hidden_ = 0;
    int64_t vocab_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t graph_buffer_ = nullptr;
    Graph projector_single_;
    Graph projector_pair_;
    std::vector<Graph> head_graphs_;
    mutable std::vector<float> head_paired_staging_;
};

}  // namespace

struct BreezeGeneratorRuntime::Impl {
    Impl(
        std::shared_ptr<const BreezeTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(assets)),
          execution(execution),
          tokenizer(this->assets),
          text_encoder(this->assets, execution, graph_arena_bytes, weight_context_bytes, storage_type),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "breeze_tts.sampling",
              "BreezeTTS",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (this->assets == nullptr) {
            throw std::runtime_error("BreezeTTS generator requires assets");
        }
        const auto & config = this->assets->config;
        backbone_runtime_config = backbone_config(config, execution.backend_type(), graph_arena_bytes);
        depth_runtime_config = depth_config(config, execution.backend_type(), graph_arena_bytes);
        weights = load_weights(*this->assets, execution, weight_context_bytes, storage_type, backbone_runtime_config);
        backbone_cond = std::make_unique<modules::QwenCausalDecodeRuntime>(execution, backbone_runtime_config, weights->backbone);
        backbone_uncond = std::make_unique<modules::QwenCausalDecodeRuntime>(execution, backbone_runtime_config, weights->backbone);
        depth_pair = std::make_unique<modules::QwenCausalDecodeRuntime>(execution, depth_runtime_config, weights->depth);
        depth_projection = std::make_unique<BreezeDepthProjectionRuntime>(
            execution.backend(),
            execution.backend_type(),
            execution.config().threads,
            graph_arena_bytes,
            config,
            weights->depth_projector,
            weights->depth_heads);
        speech_encoder = std::make_unique<BreezeSpeechEncoderRuntime>(
            this->assets,
            execution,
            graph_arena_bytes,
            storage_type,
            storage_type);
        speech_decoder = std::make_unique<BreezeSpeechDecoderRuntime>(
            this->assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            storage_type);
        depth_first_embed_staging_.assign(static_cast<size_t>(config.depth_hidden_size), 0.0F);
        depth_projected_pair_staging_.assign(static_cast<size_t>(2 * config.depth_hidden_size), 0.0F);
        depth_prefill_staging_.assign(static_cast<size_t>(4 * config.depth_hidden_size), 0.0F);
        depth_next_embed_staging_.assign(static_cast<size_t>(config.depth_hidden_size), 0.0F);
        depth_next_pair_staging_.assign(static_cast<size_t>(2 * config.depth_hidden_size), 0.0F);
        depth_logits_staging_.assign(static_cast<size_t>(config.vocab_size), 0.0F);
        depth_cond_hidden_now_.assign(static_cast<size_t>(config.depth_hidden_size), 0.0F);
        depth_uncond_hidden_now_.assign(static_cast<size_t>(config.depth_hidden_size), 0.0F);
    }

    std::vector<float> merge_prompt(const BreezePromptBranch & branch, const std::vector<int32_t> & reference_codes) {
        const auto & config = assets->config;
        const int64_t embedding_rows = config.num_codebooks * config.vocab_size;
        std::vector<float> out;
        out.reserve(branch.input_ids.size() * static_cast<size_t>(config.hidden_size));
        size_t text_segment_index = 0;
        int64_t text_segment_offset = 0;
        BreezeProjectedText projected_text;
        int64_t audio_frame = 0;
        for (size_t pos = 0; pos < branch.input_ids.size(); ++pos) {
            if (branch.text_mask[pos] != 0) {
                if (text_segment_offset == 0) {
                    if (text_segment_index >= branch.text_segments.size()) {
                        throw std::runtime_error("BreezeTTS text segment state mismatch");
                    }
                    projected_text = text_encoder.encode(branch.text_segments[text_segment_index]);
                }
                const size_t begin = static_cast<size_t>(text_segment_offset * config.hidden_size);
                out.insert(
                    out.end(),
                    projected_text.values.begin() + static_cast<std::ptrdiff_t>(begin),
                    projected_text.values.begin() + static_cast<std::ptrdiff_t>(begin + static_cast<size_t>(config.hidden_size)));
                ++text_segment_offset;
                if (text_segment_offset == projected_text.tokens) {
                    ++text_segment_index;
                    text_segment_offset = 0;
                }
                continue;
            }
            const int32_t id = branch.input_ids[pos];
            if (id == tokenizer.audio_token_id()) {
                if ((audio_frame + 1) * config.num_codebooks > static_cast<int64_t>(reference_codes.size())) {
                    throw std::runtime_error("BreezeTTS reference audio code count is shorter than prompt placeholders");
                }
                std::vector<int32_t> frame(static_cast<size_t>(config.num_codebooks));
                for (int64_t codebook = 0; codebook < config.num_codebooks; ++codebook) {
                    frame[static_cast<size_t>(codebook)] =
                        reference_codes[static_cast<size_t>(audio_frame * config.num_codebooks + codebook)];
                }
                const auto embedded = frame_embedding(
                    weights->audio_embedding,
                    embedding_rows,
                    config.hidden_size,
                    config.vocab_size,
                    frame);
                out.insert(out.end(), embedded.begin(), embedded.end());
                ++audio_frame;
            } else if (id == tokenizer.audio_eos_token_id()) {
                std::vector<int32_t> eos(static_cast<size_t>(config.num_codebooks), static_cast<int32_t>(config.codebook_eos_token_id));
                const auto embedded = frame_embedding(
                    weights->audio_embedding,
                    embedding_rows,
                    config.hidden_size,
                    config.vocab_size,
                    eos);
                out.insert(out.end(), embedded.begin(), embedded.end());
            } else {
                throw std::runtime_error("BreezeTTS prompt has non-text token that is not audio");
            }
        }
        return out;
    }

    std::vector<int32_t> generate_frame(
        const std::vector<float> & cond_hidden,
        const std::vector<float> & uncond_hidden,
        int32_t first_token,
        const BreezeGenerationRequest & request,
        sampling::HfSamplerScratch & scratch,
        std::mt19937 & fallback_rng,
        uint64_t & call_index,
        uint64_t & offset_blocks) {
        const auto & config = assets->config;
        std::vector<int32_t> frame;
        frame.reserve(static_cast<size_t>(config.num_codebooks));
        frame.push_back(first_token);

        const size_t depth_hidden_size = static_cast<size_t>(config.depth_hidden_size);
        const size_t depth_hidden_bytes = depth_hidden_size * sizeof(float);

        const auto project_audio_embedding_row = [&](int64_t row, float * out) {
            const int64_t rows = config.num_codebooks * config.vocab_size;
            if (row < 0 || row >= rows) {
                throw std::runtime_error("BreezeTTS embedding row is outside table");
            }
            const size_t begin = static_cast<size_t>(row * config.hidden_size);
            depth_projection->project_single(
                weights->audio_embedding.data() + begin,
                out);
        };

        project_audio_embedding_row(first_token, depth_first_embed_staging_.data());
        depth_projection->project_pair(
            cond_hidden.data(),
            uncond_hidden.data(),
            depth_projected_pair_staging_.data());

        std::memcpy(depth_prefill_staging_.data(),
                    depth_projected_pair_staging_.data(),
                    depth_hidden_bytes);
        std::memcpy(depth_prefill_staging_.data() + depth_hidden_size,
                    depth_first_embed_staging_.data(),
                    depth_hidden_bytes);
        std::memcpy(depth_prefill_staging_.data() + 2 * depth_hidden_size,
                    depth_projected_pair_staging_.data() + depth_hidden_size,
                    depth_hidden_bytes);
        std::memcpy(depth_prefill_staging_.data() + 3 * depth_hidden_size,
                    depth_first_embed_staging_.data(),
                    depth_hidden_bytes);

        auto depth = depth_pair->prefill_embeddings_batched(depth_prefill_staging_, 2, 2);
        if (static_cast<int64_t>(depth.hidden.size()) != 2 * config.depth_hidden_size) {
            throw std::runtime_error("BreezeTTS batched depth prefill hidden size mismatch");
        }
        std::memcpy(depth_cond_hidden_now_.data(),
                    depth.hidden.data(),
                    depth_hidden_bytes);
        std::memcpy(depth_uncond_hidden_now_.data(),
                    depth.hidden.data() + depth_hidden_size,
                    depth_hidden_bytes);
        depth_pair->start_decode_embeddings_batched(depth.state, config.num_codebooks + 1);

        sampling::HfSamplingOptions options;
        options.do_sample = true;
        options.temperature = request.depth_temperature;
        options.top_k = request.top_k;
        options.top_p = request.top_p;
        options.min_tokens_to_keep = 1;
        for (int64_t codebook = 1; codebook < config.num_codebooks; ++codebook) {
            depth_projection->logits_cfg(
                depth_cond_hidden_now_.data(),
                depth_uncond_hidden_now_.data(),
                codebook,
                request.guidance_scale,
                depth_logits_staging_.data());
            suppress_reserved(depth_logits_staging_, kCodecCodebookSize, config.vocab_size);
            const int32_t token = sample_logits(
                depth_logits_staging_,
                {},
                options,
                scratch,
                fallback_rng,
                sampling_policy.cuda_fast_path ? &sampling_policy : nullptr,
                request.seed,
                call_index,
                offset_blocks,
                "BreezeTTS depth sampler");
            frame.push_back(token);
            if (codebook + 1 < config.num_codebooks) {
                project_audio_embedding_row(codebook * config.vocab_size + token, depth_next_embed_staging_.data());
                std::memcpy(depth_next_pair_staging_.data(),
                            depth_next_embed_staging_.data(),
                            depth_hidden_bytes);
                std::memcpy(depth_next_pair_staging_.data() + depth_hidden_size,
                            depth_next_embed_staging_.data(),
                            depth_hidden_bytes);
                const auto step = depth_pair->decode_embeddings_batched(depth_next_pair_staging_, 2);
                if (static_cast<int64_t>(step.hidden.size()) != 2 * config.depth_hidden_size) {
                    throw std::runtime_error("BreezeTTS batched depth decode hidden size mismatch");
                }
                std::memcpy(depth_cond_hidden_now_.data(),
                            step.hidden.data(),
                            depth_hidden_bytes);
                std::memcpy(depth_uncond_hidden_now_.data(),
                            step.hidden.data() + depth_hidden_size,
                            depth_hidden_bytes);
            }
        }
        return frame;
    }

    runtime::AudioBuffer generate(const BreezeGenerationRequest & request) {
        if (request.text.empty()) {
            throw std::runtime_error("BreezeTTS requires text");
        }
        const auto & config = assets->config;
        BreezeSpeechCodes reference;
        if (request.reference_codes.has_value()) {
            reference = *request.reference_codes;
        } else if (request.reference_audio.has_value()) {
            reference = speech_encoder->encode(*request.reference_audio);
            speech_encoder->release_runtime_graphs();
        }
        std::vector<int32_t> reference_codes;
        int64_t reference_frames = 0;
        if (!reference.codes.empty()) {
            if (reference.frames < 0 || reference.code_groups <= 0) {
                throw std::runtime_error("BreezeTTS speech codes have invalid shape");
            }
            if (static_cast<int64_t>(reference.codes.size()) != reference.frames * reference.code_groups) {
                throw std::runtime_error("BreezeTTS speech code count does not match shape");
            }
            reference_codes = reference.codes;
            reference_frames = static_cast<int64_t>(reference_codes.size()) / config.num_codebooks;
        }
        BreezePromptBranch cond_branch;
        BreezePromptBranch uncond_branch;
        std::vector<float> cond_embeddings;
        std::vector<float> uncond_embeddings;
        int64_t cond_steps = 0;
        int64_t uncond_steps = 0;
        const double prompt_ms = engine::debug::measure_ms([&] {
            if (!reference_codes.empty()) {
                if (request.reference_text.empty()) {
                    throw std::runtime_error("BreezeTTS clone requires reference_text");
                }
                cond_branch = tokenizer.build_clone(request.text, request.instruction, request.reference_text, reference_frames);
                uncond_branch = tokenizer.build_clone_negative(request.text, request.reference_text, reference_frames);
            } else {
                cond_branch = tokenizer.build_tts_instruction(request.text, request.instruction);
                uncond_branch = tokenizer.build_tts_plain(request.text);
            }
            cond_embeddings = merge_prompt(cond_branch, reference_codes);
            uncond_embeddings = merge_prompt(uncond_branch, reference_codes);
            cond_steps = static_cast<int64_t>(cond_branch.input_ids.size());
            uncond_steps = static_cast<int64_t>(uncond_branch.input_ids.size());
        });
        engine::debug::timing_log_scalar("breeze_tts.generate.prompt_ms", prompt_ms);
        text_encoder.release_runtime_graphs();
        double backbone_cond_decode_ms = 0.0;
        double backbone_uncond_decode_ms = 0.0;
        std::vector<int32_t> first_codebook_history;
        std::vector<int32_t> codes;
        double backbone_cond_prefill_ms = 0.0;
        double backbone_uncond_prefill_ms = 0.0;
        const double ar_ms = engine::debug::measure_ms([&] {
            modules::QwenCausalPrefillResult cond;
            backbone_cond_prefill_ms = engine::debug::measure_ms([&] {
                cond = backbone_cond->prefill_embeddings(cond_embeddings, cond_steps);
            });
            modules::QwenCausalPrefillResult uncond;
            backbone_uncond_prefill_ms = engine::debug::measure_ms([&] {
                uncond = backbone_uncond->prefill_embeddings(uncond_embeddings, uncond_steps);
            });
            backbone_cond->start_decode_embeddings(cond.state, cond_steps + request.max_tokens);
            backbone_uncond->start_decode_embeddings(uncond.state, uncond_steps + request.max_tokens);

            sampling::HfSamplerScratch scratch;
            scratch.reserve_vocab(static_cast<size_t>(config.lm_head_size));
            std::mt19937 fallback_rng(static_cast<uint32_t>(request.seed));
            uint64_t sample_call_index = 0;
            uint64_t offset_blocks = 0;
            sampling::HfSamplingOptions first_options;
            first_options.do_sample = true;
            first_options.temperature = request.temperature;
            first_options.top_k = request.top_k;
            first_options.top_p = request.top_p;
            first_options.repetition_penalty = kRepetitionPenalty;
            first_options.min_tokens_to_keep = 1;

            codes.reserve(static_cast<size_t>(request.max_tokens * config.num_codebooks));
            for (int64_t step = 0; step < request.max_tokens; ++step) {
                if (cond.logits.size() != uncond.logits.size()) {
                    throw std::runtime_error("BreezeTTS CFG logits shape mismatch");
                }
                std::vector<float> logits(cond.logits.size(), 0.0F);
                for (size_t i = 0; i < logits.size(); ++i) {
                    logits[i] = uncond.logits[i] + request.guidance_scale * (cond.logits[i] - uncond.logits[i]);
                }
                suppress_reserved(logits, kCodecCodebookSize, config.vocab_size);
                const int32_t first_token = sample_logits(
                    std::move(logits),
                    first_codebook_history,
                    first_options,
                    scratch,
                    fallback_rng,
                    sampling_policy.cuda_fast_path ? &sampling_policy : nullptr,
                    request.seed,
                    sample_call_index,
                    offset_blocks,
                    "BreezeTTS semantic sampler");
                if (first_token == config.vocab_size) {
                    break;
                }
                if (first_token == config.codebook_pad_token_id) {
                    continue;
                }
                const auto frame = generate_frame(
                    cond.hidden,
                    uncond.hidden,
                    first_token,
                    request,
                    scratch,
                    fallback_rng,
                    sample_call_index,
                    offset_blocks);
                first_codebook_history.push_back(first_token);
                codes.insert(codes.end(), frame.begin(), frame.end());
                const auto embedded = frame_embedding(
                    weights->audio_embedding,
                    config.num_codebooks * config.vocab_size,
                    config.hidden_size,
                    config.vocab_size,
                    frame);
                modules::QwenCausalDecodeStepResult cond_step;
                backbone_cond_decode_ms += engine::debug::measure_ms([&] {
                    cond_step = backbone_cond->decode_embedding(embedded);
                });
                modules::QwenCausalDecodeStepResult uncond_step;
                backbone_uncond_decode_ms += engine::debug::measure_ms([&] {
                    uncond_step = backbone_uncond->decode_embedding(embedded);
                });
                cond.logits = cond_step.logits;
                cond.hidden = cond_step.hidden;
                uncond.logits = uncond_step.logits;
                uncond.hidden = uncond_step.hidden;
            }
        });
        engine::debug::timing_log_scalar("breeze_tts.ar.total_ms", ar_ms);
        engine::debug::timing_log_scalar("breeze_tts.ar.backbone_cond_prefill_ms", backbone_cond_prefill_ms);
        engine::debug::timing_log_scalar("breeze_tts.ar.backbone_uncond_prefill_ms", backbone_uncond_prefill_ms);
        engine::debug::timing_log_scalar("breeze_tts.ar.backbone_cond_decode_ms", backbone_cond_decode_ms);
        engine::debug::timing_log_scalar("breeze_tts.ar.backbone_uncond_decode_ms", backbone_uncond_decode_ms);
        backbone_cond->release_runtime_graphs();
        backbone_uncond->release_runtime_graphs();
        depth_pair->release_runtime_graphs();
        if (codes.empty()) {
            throw std::runtime_error("BreezeTTS generated no audio codes");
        }
        if (config.num_codebooks <= 0 || static_cast<int64_t>(codes.size()) % config.num_codebooks != 0) {
            throw std::runtime_error("BreezeTTS audio code count must be divisible by num_codebooks");
        }
        BreezeSpeechCodes speech_codes;
        speech_codes.codes = codes;
        speech_codes.code_groups = config.num_codebooks;
        speech_codes.frames = static_cast<int64_t>(codes.size()) / config.num_codebooks;
        runtime::AudioBuffer audio = speech_decoder->decode(speech_codes);
        speech_decoder->release_runtime_graphs();
        for (float & sample : audio.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return audio;
    }

    std::shared_ptr<const BreezeTTSAssets> assets;
    core::ExecutionContext & execution;
    BreezeTextTokenizer tokenizer;
    BreezeTextEncoderRuntime text_encoder;
    sampling::TorchCudaSamplingPolicy sampling_policy;
    modules::QwenCausalDecodeRuntimeConfig backbone_runtime_config;
    modules::QwenCausalDecodeRuntimeConfig depth_runtime_config;
    std::shared_ptr<const BreezeWeights> weights;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> backbone_cond;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> backbone_uncond;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> depth_pair;
    std::unique_ptr<BreezeDepthProjectionRuntime> depth_projection;
    std::unique_ptr<BreezeSpeechEncoderRuntime> speech_encoder;
    std::unique_ptr<BreezeSpeechDecoderRuntime> speech_decoder;
    std::vector<float> depth_first_embed_staging_;
    std::vector<float> depth_projected_pair_staging_;
    std::vector<float> depth_prefill_staging_;
    std::vector<float> depth_next_embed_staging_;
    std::vector<float> depth_next_pair_staging_;
    std::vector<float> depth_logits_staging_;
    std::vector<float> depth_cond_hidden_now_;
    std::vector<float> depth_uncond_hidden_now_;
};

BreezeGeneratorRuntime::BreezeGeneratorRuntime(
    std::shared_ptr<const BreezeTTSAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

BreezeGeneratorRuntime::~BreezeGeneratorRuntime() = default;

engine::runtime::AudioBuffer BreezeGeneratorRuntime::generate(const BreezeGenerationRequest & request) {
    const auto start = Clock::now();
    auto audio = impl_->generate(request);
    engine::debug::timing_log_scalar("breeze_tts.generate.total_ms", engine::debug::elapsed_ms(start));
    return audio;
}

BreezeSpeechCodes BreezeGeneratorRuntime::encode_reference(const engine::runtime::AudioBuffer & audio) const {
    return impl_->speech_encoder->encode(audio);
}

}  // namespace engine::models::breeze_tts
