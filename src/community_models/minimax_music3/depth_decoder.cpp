#include "engine/community_models/minimax_music3/depth_decoder.h"

#include "engine/community_models/minimax_music3/types.h"
#include "engine/framework/core/backend.h"

#include <ggml.h>
#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::minimax_music3 {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;

// Sequence layout per frame: [global hidden, semantic code, c1..c6] = 8 rows per batch row.
constexpr int64_t kSeqLen = 8;
constexpr int64_t kBatch = 2;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct DepthLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue to_q;
    core::TensorValue to_k;
    core::TensorValue to_v;
    core::TensorValue to_out;
    core::TensorValue post_norm;
    core::TensorValue gate;
    core::TensorValue up;
    core::TensorValue down;
};

ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

}  // namespace

struct MiniMaxMusic3DepthDecoderRuntime::Impl {
    core::ExecutionContext & execution;
    MiniMaxMusic3Config config;
    core::BackendWeightStore store;
    core::TensorValue lm_embedding;
    core::TensorValue audio_embeddings;
    core::TensorValue projection;
    core::TensorValue pos_embedding;
    std::vector<DepthLayerWeights> layers;
    core::TensorValue final_norm;
    std::vector<core::TensorValue> audio_heads;

    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_gallocr_t gallocr = nullptr;

    ggml_tensor * in_last_hidden = nullptr;  // f32 [hidden, 2]
    ggml_tensor * in_sem_id = nullptr;       // i32 [1], global vocabulary id
    ggml_tensor * in_gumbel = nullptr;       // f32 [audio_vocab, codebooks - 1]
    ggml_tensor * in_mask = nullptr;         // f32 [seq, seq, 1]
    std::vector<ggml_tensor *> out_codes;    // i32 [1] per residual codebook
    ggml_tensor * out_hidden = nullptr;      // f32 [hidden, codebooks - 1]
    ggml_tensor * out_feedback = nullptr;    // f32 [hidden, 1]
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;

    Impl(
        core::ExecutionContext & execution_context,
        const assets::TensorSource & source,
        core::TensorValue lm_token_embedding,
        const MiniMaxMusic3Config & cfg,
        size_t weight_context_bytes)
        : execution(execution_context),
          config(cfg),
          store(execution.backend(), execution.backend_type(), "minimax_music3.depth_decoder", weight_context_bytes),
          lm_embedding(std::move(lm_token_embedding)) {
        const int64_t hidden = config.depth_hidden;
        audio_embeddings = store.load_tensor(
            source,
            "audio_embeddings.weight",
            assets::TensorStorageType::Native,
            {config.depth_audio_vocab * (config.depth_codebooks - 1), hidden});
        projection = store.load_tensor(source, "projection.weight", assets::TensorStorageType::Native, {hidden, hidden});
        pos_embedding = store.load_tensor(
            source,
            "pos_embedding.weight",
            assets::TensorStorageType::F32,
            {config.depth_max_positions, hidden});
        layers.reserve(static_cast<size_t>(config.depth_layers));
        for (int64_t layer = 0; layer < config.depth_layers; ++layer) {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            DepthLayerWeights out;
            out.input_norm = store.load_tensor(source, prefix + "input_layernorm.weight", assets::TensorStorageType::F32, {hidden});
            out.to_q = store.load_tensor(source, prefix + "attn.to_q.weight", assets::TensorStorageType::Native, {hidden, hidden});
            out.to_k = store.load_tensor(source, prefix + "attn.to_k.weight", assets::TensorStorageType::Native, {hidden, hidden});
            out.to_v = store.load_tensor(source, prefix + "attn.to_v.weight", assets::TensorStorageType::Native, {hidden, hidden});
            out.to_out = store.load_tensor(source, prefix + "attn.to_out.weight", assets::TensorStorageType::Native, {hidden, hidden});
            out.post_norm = store.load_tensor(source, prefix + "post_attention_layernorm.weight", assets::TensorStorageType::F32, {hidden});
            out.gate = store.load_tensor(source, prefix + "gate_proj.weight", assets::TensorStorageType::Native, {config.depth_intermediate, hidden});
            out.up = store.load_tensor(source, prefix + "up_proj.weight", assets::TensorStorageType::Native, {config.depth_intermediate, hidden});
            out.down = store.load_tensor(source, prefix + "down_proj.weight", assets::TensorStorageType::Native, {hidden, config.depth_intermediate});
            layers.push_back(std::move(out));
        }
        final_norm = store.load_tensor(source, "norm.weight", assets::TensorStorageType::F32, {hidden});
        audio_heads.reserve(static_cast<size_t>(config.depth_codebooks - 1));
        for (int64_t head = 0; head < config.depth_codebooks - 1; ++head) {
            audio_heads.push_back(store.load_tensor(
                source,
                "audio_heads." + std::to_string(head) + ".weight",
                assets::TensorStorageType::Native,
                {config.depth_audio_vocab, hidden}));
        }
        store.upload();
        source.release_storage();
        build_graph();
    }

    ~Impl() {
        plan.reset();
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
        }
    }

    // The 4-layer stack plus final norm over the assembled [hidden, batch * seq] matrix.
    ggml_tensor * depth_stack(ggml_context * ctx, ggml_tensor * x) {
        const int64_t hidden = config.depth_hidden;
        const int64_t heads = config.depth_heads;
        const int64_t head_dim = hidden / heads;
        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        for (const auto & layer : layers) {
            ggml_tensor * h = rms_norm_mul(ctx, x, layer.input_norm.tensor, config.depth_rms_eps);
            ggml_tensor * q = ggml_mul_mat(ctx, layer.to_q.tensor, h);
            ggml_tensor * k = ggml_mul_mat(ctx, layer.to_k.tensor, h);
            ggml_tensor * v = ggml_mul_mat(ctx, layer.to_v.tensor, h);
            auto split_heads = [&](ggml_tensor * t) {
                t = ggml_reshape_4d(ctx, t, head_dim, heads, kSeqLen, kBatch);
                t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
                return ggml_reshape_3d(ctx, t, head_dim, kSeqLen, heads * kBatch);
            };
            q = split_heads(q);
            k = split_heads(k);
            v = split_heads(v);
            ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [seq_k, seq_q, H]
            scores = ggml_add(ctx, ggml_scale(ctx, scores, scale), in_mask);
            ggml_tensor * probs = ggml_soft_max(ctx, scores);
            ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [seq_k, head_dim, H]
            ggml_tensor * attn = ggml_mul_mat(ctx, v_t, probs);  // [head_dim, seq_q, H]
            attn = ggml_reshape_4d(ctx, attn, head_dim, kSeqLen, heads, kBatch);
            attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(ctx, attn, hidden, kBatch * kSeqLen);
            x = ggml_add(ctx, x, ggml_mul_mat(ctx, layer.to_out.tensor, attn));

            ggml_tensor * h2 = rms_norm_mul(ctx, x, layer.post_norm.tensor, config.depth_rms_eps);
            ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, layer.gate.tensor, h2));
            ggml_tensor * up = ggml_mul_mat(ctx, layer.up.tensor, h2);
            x = ggml_add(ctx, x, ggml_mul_mat(ctx, layer.down.tensor, ggml_mul(ctx, gate, up)));
        }
        return rms_norm_mul(ctx, x, final_norm.tensor, config.depth_rms_eps);
    }

    void build_graph() {
        const int64_t hidden = config.depth_hidden;
        const int64_t vocab = config.depth_audio_vocab;
        const int64_t residual = config.depth_codebooks - 1;

        graph_ctx.reset(ggml_init({512 * 1024 * 1024, nullptr, true}));
        input_ctx.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (graph_ctx == nullptr || input_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-Music3 depth decoder graph context");
        }
        ggml_context * ictx = input_ctx.get();
        ggml_context * ctx = graph_ctx.get();

        in_last_hidden = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, hidden, kBatch);
        in_sem_id = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
        in_gumbel = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, vocab, residual);
        in_mask = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, kSeqLen, kSeqLen, 1);
        for (ggml_tensor * tensor : {in_last_hidden, in_sem_id, in_gumbel, in_mask}) {
            ggml_set_input(tensor);
        }

        graph = ggml_new_graph_custom(ctx, 16384, false);

        ggml_tensor * lh0 = ggml_view_2d(ctx, in_last_hidden, hidden, 1, in_last_hidden->nb[1], 0);
        ggml_tensor * lh1 = ggml_view_2d(
            ctx, in_last_hidden, hidden, 1, in_last_hidden->nb[1], in_last_hidden->nb[1]);
        ggml_tensor * emb_sem = ggml_get_rows(ctx, lm_embedding.tensor, in_sem_id);  // [hidden, 1]
        ggml_tensor * pad_row = ggml_scale(ctx, emb_sem, 0.0F);
        ggml_tensor * pos8 = ggml_view_2d(
            ctx, pos_embedding.tensor, hidden, kSeqLen, pos_embedding.tensor->nb[1], 0);
        ggml_tensor * pos_cat = ggml_concat(ctx, pos8, pos8, 1);

        std::vector<ggml_tensor *> code_rows = {emb_sem};  // rows after the hidden row
        ggml_tensor * feedback_sum = emb_sem;
        ggml_tensor * hidden_cat = nullptr;
        out_codes.clear();
        out_codes.reserve(static_cast<size_t>(residual));

        for (int64_t index = 1; index <= residual; ++index) {
            // Assemble the fixed-shape sequence; rows past the known prefix hold zeros and
            // are shielded by the causal mask.
            auto assemble = [&](ggml_tensor * hidden_row) {
                ggml_tensor * rows = hidden_row;
                for (ggml_tensor * row : code_rows) {
                    rows = ggml_concat(ctx, rows, row, 1);
                }
                for (int64_t pad = static_cast<int64_t>(code_rows.size()) + 1; pad < kSeqLen; ++pad) {
                    rows = ggml_concat(ctx, rows, pad_row, 1);
                }
                return rows;
            };
            ggml_tensor * x = ggml_concat(ctx, assemble(lh0), assemble(lh1), 1);  // [hidden, 16]
            x = ggml_mul_mat(ctx, projection.tensor, x);
            x = ggml_add(ctx, x, pos_cat);
            ggml_tensor * normed = depth_stack(ctx, x);

            ggml_tensor * h_cond = ggml_view_2d(
                ctx, normed, hidden, 1, normed->nb[1], static_cast<size_t>(index) * normed->nb[1]);
            ggml_tensor * h_uncond = ggml_view_2d(
                ctx, normed, hidden, 1, normed->nb[1], static_cast<size_t>(kSeqLen + index) * normed->nb[1]);
            hidden_cat = hidden_cat == nullptr
                ? ggml_cont(ctx, h_cond)
                : ggml_concat(ctx, hidden_cat, ggml_cont(ctx, h_cond), 1);

            ggml_tensor * head = audio_heads[static_cast<size_t>(index - 1)].tensor;
            ggml_tensor * logits_cond = ggml_mul_mat(ctx, head, ggml_cont(ctx, h_cond));
            ggml_tensor * logits_uncond = ggml_mul_mat(ctx, head, ggml_cont(ctx, h_uncond));
            // guided = uncond + scale * (cond - uncond)
            ggml_tensor * guided = ggml_add(
                ctx,
                ggml_scale(ctx, logits_cond, MiniMaxMusic3Contract::kArCfgScale),
                ggml_scale(ctx, logits_uncond, 1.0F - MiniMaxMusic3Contract::kArCfgScale));

            // Top-k mask: ggml_top_k returns descending-sorted indices, so entry k-1 holds
            // the threshold element.
            ggml_tensor * top = ggml_top_k(ctx, guided, MiniMaxMusic3Contract::kArSamplingTopK);
            ggml_tensor * threshold_id = ggml_view_1d(
                ctx, top, 1, static_cast<size_t>(MiniMaxMusic3Contract::kArSamplingTopK - 1) * top->nb[0]);
            ggml_tensor * guided_rows = ggml_reshape_2d(ctx, guided, 1, vocab);
            ggml_tensor * threshold = ggml_get_rows(ctx, guided_rows, threshold_id);  // [1, 1]
            ggml_tensor * below = ggml_step(
                ctx, ggml_neg(ctx, ggml_sub(ctx, guided, ggml_reshape_1d(ctx, threshold, 1))));
            ggml_tensor * masked = ggml_sub(ctx, guided, ggml_scale(ctx, below, 1.0e30F));

            ggml_tensor * noise = ggml_view_2d(
                ctx, in_gumbel, vocab, 1, in_gumbel->nb[1], static_cast<size_t>(index - 1) * in_gumbel->nb[1]);
            ggml_tensor * code = ggml_argmax(ctx, ggml_add(ctx, masked, noise));  // i32 [1]
            ggml_set_output(code);
            ggml_build_forward_expand(graph, code);
            out_codes.push_back(code);

            ggml_tensor * embed_slice = ggml_view_2d(
                ctx,
                audio_embeddings.tensor,
                hidden,
                vocab,
                audio_embeddings.tensor->nb[1],
                static_cast<size_t>(index - 1) * static_cast<size_t>(vocab) * audio_embeddings.tensor->nb[1]);
            ggml_tensor * emb = ggml_get_rows(ctx, embed_slice, code);  // [hidden, 1] f32
            feedback_sum = ggml_add(ctx, feedback_sum, emb);
            if (index < residual) {
                code_rows.push_back(emb);
            }
        }

        out_hidden = hidden_cat;
        out_feedback = ggml_scale(
            ctx, feedback_sum, 1.0F / std::sqrt(static_cast<float>(config.depth_codebooks)));
        ggml_set_output(out_hidden);
        ggml_set_output(out_feedback);
        ggml_build_forward_expand(graph, out_hidden);
        ggml_build_forward_expand(graph, out_feedback);

        input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), execution.backend());
        if (input_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 depth decoder inputs");
        }
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 depth decoder graph");
        }
        core::prepare_host_graph_plan(execution, graph, plan);

        // Static causal mask over the fixed sequence.
        std::vector<float> mask(static_cast<size_t>(kSeqLen * kSeqLen));
        for (int64_t query = 0; query < kSeqLen; ++query) {
            for (int64_t key = 0; key < kSeqLen; ++key) {
                mask[static_cast<size_t>(query * kSeqLen + key)] = key <= query ? 0.0F : -INFINITY;
            }
        }
        ggml_backend_tensor_set(in_mask, mask.data(), 0, mask.size() * sizeof(float));
    }
};

MiniMaxMusic3DepthDecoderRuntime::MiniMaxMusic3DepthDecoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source,
    core::TensorValue lm_token_embedding,
    const MiniMaxMusic3Config & config,
    size_t weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("MiniMax-Music3 depth decoder tensor source is missing");
    }
    impl_ = std::make_unique<Impl>(execution, *source, std::move(lm_token_embedding), config, weight_context_bytes);
}

MiniMaxMusic3DepthDecoderRuntime::~MiniMaxMusic3DepthDecoderRuntime() = default;

MiniMaxMusic3DepthDecoderRuntime::FrameOutput MiniMaxMusic3DepthDecoderRuntime::decode_frame(
    const std::vector<float> & last_hidden,
    int32_t semantic_code,
    const std::vector<float> & gumbel_noise) {
    auto & impl = *impl_;
    const auto & config = impl.config;
    const int64_t hidden = config.depth_hidden;
    const int64_t vocab = config.depth_audio_vocab;
    const int64_t residual = config.depth_codebooks - 1;
    if (last_hidden.size() != static_cast<size_t>(kBatch * hidden)) {
        throw std::runtime_error("MiniMax-Music3 depth decoder last_hidden size mismatch");
    }
    if (semantic_code < 0 || semantic_code >= MiniMaxMusic3Contract::kSemanticVocabSize) {
        throw std::runtime_error("MiniMax-Music3 depth decoder semantic code out of range");
    }
    if (!gumbel_noise.empty() && gumbel_noise.size() != static_cast<size_t>(residual * vocab)) {
        throw std::runtime_error("MiniMax-Music3 depth decoder Gumbel noise size mismatch");
    }

    ggml_backend_tensor_set(impl.in_last_hidden, last_hidden.data(), 0, last_hidden.size() * sizeof(float));
    const int32_t sem_global = semantic_code + MiniMaxMusic3Contract::kAudioCodeOffset;
    ggml_backend_tensor_set(impl.in_sem_id, &sem_global, 0, sizeof(int32_t));
    if (gumbel_noise.empty()) {
        const std::vector<float> zeros(static_cast<size_t>(residual * vocab), 0.0F);
        ggml_backend_tensor_set(impl.in_gumbel, zeros.data(), 0, zeros.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(impl.in_gumbel, gumbel_noise.data(), 0, gumbel_noise.size() * sizeof(float));
    }

    const ggml_status status = core::compute_graph(impl.execution, impl.graph, impl.plan, "minimax_music3.depth_decoder");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MiniMax-Music3 depth decoder graph compute failed");
    }

    FrameOutput out;
    out.codes[0] = semantic_code;
    for (int64_t index = 0; index < residual; ++index) {
        int32_t code = 0;
        ggml_backend_tensor_get(impl.out_codes[static_cast<size_t>(index)], &code, 0, sizeof(int32_t));
        if (code < 0 || code >= vocab) {
            throw std::runtime_error("MiniMax-Music3 depth decoder sampled code out of range");
        }
        out.codes[static_cast<size_t>(index + 1)] = code;
    }
    out.depth_hidden.resize(static_cast<size_t>(residual * hidden));
    ggml_backend_tensor_get(impl.out_hidden, out.depth_hidden.data(), 0, out.depth_hidden.size() * sizeof(float));
    out.feedback_embedding.resize(static_cast<size_t>(hidden));
    ggml_backend_tensor_get(impl.out_feedback, out.feedback_embedding.data(), 0, out.feedback_embedding.size() * sizeof(float));
    return out;
}

}  // namespace engine::models::minimax_music3
