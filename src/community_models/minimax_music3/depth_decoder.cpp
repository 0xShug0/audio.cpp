#include "engine/community_models/minimax_music3/depth_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

modules::QwenDecoderStackConfig make_depth_stack_config(
    const MiniMaxMusic3DepthConfig & config,
    core::BackendType backend_type) {
    modules::QwenDecoderStackConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.attention_heads;
    out.num_key_value_heads = config.attention_heads;
    out.head_dim = config.hidden_size / config.attention_heads;
    out.intermediate_size = config.intermediate_size;
    out.layers = config.layers;
    out.rms_norm_eps = config.rms_norm_eps;
    out.position_encoding = modules::QwenDecoderPositionEncoding::None;
    out.attention_precision = GGML_PREC_F32;
    out.projection_precision = GGML_PREC_DEFAULT;
    out.use_qk_norm = false;
    out.runtime.attention.prefill_mode = core::uses_ggml_cuda_or_hip_backend(backend_type)
        ? modules::QwenDecoderAttentionMode::FlashGroupedViewKV
        : modules::QwenDecoderAttentionMode::ManualRepeat;
    return out;
}

modules::QwenDecoderLayerWeights load_depth_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MiniMaxMusic3DepthConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".attn.to_q.weight",
        storage_type,
        {config.hidden_size, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".attn.to_k.weight",
        storage_type,
        {config.hidden_size, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".attn.to_v.weight",
        storage_type,
        {config.hidden_size, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".attn.to_out.weight",
        storage_type,
        {config.hidden_size, config.hidden_size});
    out.post_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".post_attention_layernorm",
        config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

MiniMaxMusic3DepthWeights load_depth_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.depth;
    const auto & source = *assets.depth_decoder_weights;
    MiniMaxMusic3DepthWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.depth.weights",
        weight_context_bytes);
    out.audio_embeddings = out.store->load_tensor(
        source,
        "audio_embeddings.weight",
        storage_type,
        {config.audio_vocab_size * (config.codebooks - 1), config.hidden_size});
    out.projection = binding::linear_from_source(
        *out.store,
        source,
        "projection",
        storage_type,
        config.hidden_size,
        config.hidden_size,
        false);
    out.position_embedding = out.store->load_tensor(
        source,
        "pos_embedding.weight",
        storage_type,
        {config.max_position_embeddings, config.hidden_size});
    out.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        out.stack.layers.push_back(load_depth_layer(*out.store, source, config, storage_type, layer));
    }
    out.norm = binding::norm_weight_from_source(*out.store, source, "norm", config.hidden_size);
    out.audio_heads.reserve(static_cast<size_t>(config.codebooks - 1));
    for (int64_t codebook = 0; codebook < config.codebooks - 1; ++codebook) {
        out.audio_heads.push_back(binding::linear_from_source(
            *out.store,
            source,
            "audio_heads." + std::to_string(codebook),
            storage_type,
            config.audio_vocab_size,
            config.hidden_size,
            false));
    }
    out.store->upload();
    assets.depth_decoder_weights->release_storage();
    return out;
}

int32_t sample_top_k(
    std::vector<float> logits,
    int64_t top_k,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & policy,
    engine::sampling::HfSamplerScratch & scratch,
    std::mt19937 & fallback_rng,
    const char * label) {
    sampling::HfLogitsProcessor::apply_top_k(logits, top_k, 1, scratch);
    const sampling::HfTorchSamplingState torch_state{
        &policy,
        seed,
        sample_call_index,
        rng_offset_blocks,
        true,
    };
    const int32_t token = sampling::HfTokenSampler::sample_from_processed_scores(
        logits,
        scratch,
        fallback_rng,
        policy.cuda_fast_path ? &torch_state : nullptr,
        label,
        false);
    ++sample_call_index;
    rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(logits.size()),
        policy);
    return token;
}

}  // namespace

struct MiniMaxMusic3DepthDecoderRuntime::Impl {
    struct DecodeGraph {
        int64_t codebook = 0;
        int64_t sequence_steps = 0;
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue last_hidden;
        core::TensorValue semantic_ids;
        core::TensorValue residual_ids;
        core::TensorValue positions;
        ggml_tensor * logits = nullptr;
        ggml_tensor * hidden = nullptr;
    };

    struct FeedbackGraph {
        std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
        ggml_cgraph * graph = nullptr;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        core::HostGraphPlan plan;
        core::TensorValue semantic_id;
        core::TensorValue residual_ids;
        ggml_tensor * output = nullptr;
    };

    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::TensorValue input_global_token_embedding,
        core::ExecutionContext & input_execution,
        size_t input_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          global_token_embedding(input_global_token_embedding),
          execution(input_execution),
          graph_arena_bytes(input_graph_arena_bytes),
          weights(load_depth_weights(*assets, execution, weight_context_bytes, storage_type)),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "minimax_music3.depth.sampling",
              "MiniMax Music 3 depth",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 depth runtime requires assets");
        }
        if (!global_token_embedding.valid()) {
            throw std::runtime_error("MiniMax Music 3 depth runtime requires global token embedding");
        }
        scratch.reserve_vocab(static_cast<size_t>(assets->config.depth.audio_vocab_size));
    }

    ~Impl() {
        release_runtime_graphs();
    }

    DecodeGraph & decode_graph(int64_t codebook) {
        const int64_t sequence_steps = codebook + 1;
        auto & slot = decode_graphs[static_cast<size_t>(codebook - 1)];
        if (slot.graph != nullptr) {
            return slot;
        }
        build_decode_graph(slot, codebook, sequence_steps);
        return slot;
    }

    void build_decode_graph(DecodeGraph & out, int64_t codebook, int64_t sequence_steps) {
        const auto & config = assets->config.depth;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        out.ggml.reset(ggml_init(params));
        if (out.ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 depth graph context");
        }
        core::ModuleBuildContext ctx{out.ggml.get(), "minimax_music3.depth", execution.backend_type()};
        out.codebook = codebook;
        out.sequence_steps = sequence_steps;
        out.last_hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, config.hidden_size}));
        out.semantic_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2}));
        out.positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({sequence_steps}));
        ggml_set_input(out.last_hidden.tensor);
        ggml_set_input(out.semantic_ids.tensor);
        ggml_set_input(out.positions.tensor);

        auto hidden0 = modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
            ctx,
            out.last_hidden,
            weights.projection);
        hidden0 = core::reshape_tensor(ctx, hidden0, core::TensorShape::from_dims({2, 1, config.hidden_size}));

        auto semantic = modules::EmbeddingModule({assets->config.qwen.vocab_size, config.hidden_size})
                            .build(ctx, out.semantic_ids, global_token_embedding);
        semantic = modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
            ctx,
            semantic,
            weights.projection);
        semantic = core::reshape_tensor(ctx, semantic, core::TensorShape::from_dims({2, 1, config.hidden_size}));
        auto sequence = modules::ConcatModule({1}).build(ctx, hidden0, semantic);
        if (codebook > 1) {
            out.residual_ids = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({2, codebook - 1}));
            ggml_set_input(out.residual_ids.tensor);
            auto residual = modules::EmbeddingModule({
                                config.audio_vocab_size * (config.codebooks - 1),
                                config.hidden_size})
                                .build(ctx, out.residual_ids, weights.audio_embeddings);
            residual = modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
                ctx,
                residual,
                weights.projection);
            sequence = modules::ConcatModule({1}).build(ctx, sequence, residual);
        }
        auto pos = modules::EmbeddingModule({config.max_position_embeddings, config.hidden_size})
                       .build(ctx, out.positions, weights.position_embedding);
        pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, sequence_steps, config.hidden_size}));
        pos = modules::RepeatModule({core::TensorShape::from_dims({2, sequence_steps, config.hidden_size})}).build(ctx, pos);
        sequence = core::wrap_tensor(
            ggml_add(ctx.ggml, sequence.tensor, pos.tensor),
            sequence.shape,
            GGML_TYPE_F32);

        const auto stack = modules::QwenDecoderStackModule(make_depth_stack_config(config, execution.backend_type()))
                               .build(ctx, sequence, out.positions, weights.stack, std::nullopt, std::nullopt);
        auto normalized = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(
            ctx,
            stack.output,
            weights.norm);
        auto last = modules::SliceModule({1, sequence_steps - 1, 1}).build(ctx, normalized);
        last = core::ensure_backend_addressable_layout(ctx, last);
        last = core::reshape_tensor(ctx, last, core::TensorShape::from_dims({2, config.hidden_size}));
        auto logits = modules::LinearModule({config.hidden_size, config.audio_vocab_size, false}).build(
            ctx,
            last,
            weights.audio_heads[static_cast<size_t>(codebook - 1)]);
        out.hidden = last.tensor;
        out.logits = logits.tensor;
        out.graph = ggml_new_graph_custom(out.ggml.get(), 262144, false);
        ggml_build_forward_expand(out.graph, out.hidden);
        ggml_build_forward_expand(out.graph, out.logits);
        out.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (out.gallocr == nullptr || !ggml_gallocr_reserve(out.gallocr.get(), out.graph) ||
            !ggml_gallocr_alloc_graph(out.gallocr.get(), out.graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 depth graph");
        }
        core::prepare_host_graph_plan(execution, out.graph, out.plan);
    }

    FeedbackGraph & ensure_feedback_graph() {
        if (feedback.graph != nullptr) {
            return feedback;
        }
        const auto & config = assets->config.depth;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        feedback.ggml.reset(ggml_init(params));
        if (feedback.ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 feedback graph context");
        }
        core::ModuleBuildContext ctx{feedback.ggml.get(), "minimax_music3.feedback", execution.backend_type()};
        feedback.semantic_id = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        feedback.residual_ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config.codebooks - 1}));
        ggml_set_input(feedback.semantic_id.tensor);
        ggml_set_input(feedback.residual_ids.tensor);
        auto semantic = modules::EmbeddingModule({assets->config.qwen.vocab_size, config.hidden_size})
                            .build(ctx, feedback.semantic_id, global_token_embedding);
        auto residual = modules::EmbeddingModule({
                            config.audio_vocab_size * (config.codebooks - 1),
                            config.hidden_size})
                            .build(ctx, feedback.residual_ids, weights.audio_embeddings);
        auto sum = modules::ReduceSumModule({0}).build(ctx, residual);
        auto added = modules::AddModule().build(ctx, semantic, sum);
        auto scaled = core::wrap_tensor(
            ggml_scale(ctx.ggml, added.tensor, 1.0F / std::sqrt(static_cast<float>(config.codebooks))),
            added.shape,
            GGML_TYPE_F32);
        feedback.output = scaled.tensor;
        feedback.graph = ggml_new_graph_custom(feedback.ggml.get(), 65536, false);
        ggml_build_forward_expand(feedback.graph, feedback.output);
        feedback.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (feedback.gallocr == nullptr || !ggml_gallocr_reserve(feedback.gallocr.get(), feedback.graph) ||
            !ggml_gallocr_alloc_graph(feedback.gallocr.get(), feedback.graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 feedback graph");
        }
        core::prepare_host_graph_plan(execution, feedback.graph, feedback.plan);
        return feedback;
    }

    MiniMaxMusic3DepthCodes generate(
        const std::vector<float> & last_hidden_cond,
        const std::vector<float> & last_hidden_uncond,
        int32_t semantic_code,
        float guidance_scale,
        int64_t top_k,
        uint64_t seed,
        uint64_t & sample_call_index,
        uint64_t & rng_offset_blocks) {
        const auto & config = assets->config.depth;
        if (static_cast<int64_t>(last_hidden_cond.size()) != config.hidden_size ||
            static_cast<int64_t>(last_hidden_uncond.size()) != config.hidden_size) {
            throw std::runtime_error("MiniMax Music 3 depth hidden input shape mismatch");
        }
        last_hidden_scratch.resize(static_cast<size_t>(2 * config.hidden_size));
        std::copy(last_hidden_cond.begin(), last_hidden_cond.end(), last_hidden_scratch.begin());
        std::copy(last_hidden_uncond.begin(), last_hidden_uncond.end(), last_hidden_scratch.begin() + config.hidden_size);
        std::vector<int32_t> out_codes{semantic_code};
        std::vector<float> out_hidden;
        out_hidden.reserve(static_cast<size_t>((config.codebooks - 1) * config.hidden_size));
        std::mt19937 fallback_rng(static_cast<uint32_t>(seed));

        for (int64_t codebook = 1; codebook < config.codebooks; ++codebook) {
            auto & graph = decode_graph(codebook);
            const int32_t semantic_ids[2] = {
                semantic_code + 151675,
                semantic_code + 151675,
            };
            active_residual_ids_scratch.assign(static_cast<size_t>(2 * std::max<int64_t>(1, codebook - 1)), 0);
            for (int64_t previous = 1; previous < codebook; ++previous) {
                const int32_t id = out_codes[static_cast<size_t>(previous)] +
                    static_cast<int32_t>((previous - 1) * config.audio_vocab_size);
                active_residual_ids_scratch[static_cast<size_t>(previous - 1)] = id;
                active_residual_ids_scratch[static_cast<size_t>((codebook - 1) + previous - 1)] = id;
            }
            positions_scratch.resize(static_cast<size_t>(codebook + 1));
            for (int64_t i = 0; i <= codebook; ++i) {
                positions_scratch[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            core::write_tensor_f32(graph.last_hidden, last_hidden_scratch);
            core::write_tensor_i32(graph.semantic_ids, semantic_ids, 2);
            if (codebook > 1) {
                core::write_tensor_i32(graph.residual_ids, active_residual_ids_scratch);
            }
            core::write_tensor_i32(graph.positions, positions_scratch);
            if (core::compute_graph(execution, graph.graph, graph.plan, "minimax_music3.depth") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MiniMax Music 3 depth graph compute failed");
            }
            auto hidden = core::read_tensor_f32(graph.hidden);
            core::round_f32_to_bf16_in_place(hidden);
            out_hidden.insert(out_hidden.end(), hidden.begin(), hidden.begin() + config.hidden_size);
            auto logits = core::read_tensor_f32(graph.logits);
            core::round_f32_to_bf16_in_place(logits);
            for (int64_t i = 0; i < config.audio_vocab_size; ++i) {
                const float cond = logits[static_cast<size_t>(i)];
                const float uncond = logits[static_cast<size_t>(config.audio_vocab_size + i)];
                logits[static_cast<size_t>(i)] = uncond + (cond - uncond) * guidance_scale;
            }
            logits.resize(static_cast<size_t>(config.audio_vocab_size));
            const int32_t code = sample_top_k(
                std::move(logits),
                top_k,
                seed,
                sample_call_index,
                rng_offset_blocks,
                sampling_policy,
                scratch,
                fallback_rng,
                "MiniMax Music 3 depth");
            out_codes.push_back(code);
        }
        return {std::move(out_codes), std::move(out_hidden)};
    }

    std::vector<float> feedback_embedding(const std::vector<int32_t> & codes) {
        const auto & config = assets->config.depth;
        if (static_cast<int64_t>(codes.size()) != config.codebooks) {
            throw std::runtime_error("MiniMax Music 3 feedback code count mismatch");
        }
        auto & graph = ensure_feedback_graph();
        const int32_t semantic_id = codes.front() + 151675;
        std::vector<int32_t> residual_ids(static_cast<size_t>(config.codebooks - 1));
        for (int64_t i = 1; i < config.codebooks; ++i) {
            residual_ids[static_cast<size_t>(i - 1)] =
                codes[static_cast<size_t>(i)] + static_cast<int32_t>((i - 1) * config.audio_vocab_size);
        }
        core::write_tensor_i32(graph.semantic_id, &semantic_id, 1);
        core::write_tensor_i32(graph.residual_ids, residual_ids);
        if (core::compute_graph(execution, graph.graph, graph.plan, "minimax_music3.feedback") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 feedback graph compute failed");
        }
        auto output = core::read_tensor_f32(graph.output);
        core::round_f32_to_bf16_in_place(output);
        return output;
    }

    void release_runtime_graphs() {
        for (auto & graph : decode_graphs) {
            if (graph.graph != nullptr) {
                core::release_backend_graph_resources(execution.backend(), graph.graph);
            }
            graph = {};
        }
        if (feedback.graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), feedback.graph);
        }
        feedback = {};
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::TensorValue global_token_embedding;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    MiniMaxMusic3DepthWeights weights;
    std::array<DecodeGraph, 7> decode_graphs;
    FeedbackGraph feedback;
    sampling::TorchCudaSamplingPolicy sampling_policy;
    sampling::HfSamplerScratch scratch;
    std::vector<float> last_hidden_scratch;
    std::vector<int32_t> active_residual_ids_scratch;
    std::vector<int32_t> positions_scratch;
};

MiniMaxMusic3DepthDecoderRuntime::MiniMaxMusic3DepthDecoderRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::TensorValue global_token_embedding,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          global_token_embedding,
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3DepthDecoderRuntime::~MiniMaxMusic3DepthDecoderRuntime() = default;

MiniMaxMusic3DepthCodes MiniMaxMusic3DepthDecoderRuntime::generate(
    const std::vector<float> & last_hidden_cond,
    const std::vector<float> & last_hidden_uncond,
    int32_t semantic_code,
    float guidance_scale,
    int64_t top_k,
    uint64_t seed,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks) {
    return impl_->generate(
        last_hidden_cond,
        last_hidden_uncond,
        semantic_code,
        guidance_scale,
        top_k,
        seed,
        sample_call_index,
        rng_offset_blocks);
}

std::vector<float> MiniMaxMusic3DepthDecoderRuntime::feedback_embedding(const std::vector<int32_t> & codes) const {
    return impl_->feedback_embedding(codes);
}

void MiniMaxMusic3DepthDecoderRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
