#include "engine/models/kokoro_tts/plbert.h"

#include "engine/models/kokoro_tts/assets.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace kokoro_ggml {

constexpr size_t kPlbertCtxBytes = 128ull * 1024ull * 1024ull;

namespace core = engine::core;

namespace {

using engine::debug::measure_ms;

ggml_tensor * add_bias_3d(
    ggml_context * ctx,
    ggml_tensor * x,
    const core::TensorValue & bias,
    int64_t channels) {
    ggml_tensor * bias_tensor = ggml_reshape_3d(ctx, bias.tensor, channels, 1, 1);
    return ggml_add(ctx, x, bias_tensor);
}

ggml_tensor * linear_3d(
    ggml_context * ctx,
    ggml_tensor * x,
    const KokoroWeights::LinearWeights & linear) {
    ggml_tensor * y = ggml_mul_mat(ctx, linear.weight.tensor, x);
    if (linear.use_bias) {
        y = add_bias_3d(ctx, y, *linear.bias, linear.out_features);
    }
    return y;
}

ggml_tensor * layer_norm_3d(
    ggml_context * ctx,
    ggml_tensor * x,
    const KokoroWeights::LayerNormWeights & norm) {
    ggml_tensor * y = ggml_norm(ctx, x, norm.eps);
    ggml_tensor * gamma = ggml_reshape_3d(ctx, norm.weight.tensor, norm.channels, 1, 1);
    ggml_tensor * beta = ggml_reshape_3d(ctx, norm.bias.tensor, norm.channels, 1, 1);
    y = ggml_mul(ctx, y, gamma);
    y = ggml_add(ctx, y, beta);
    return y;
}

ggml_tensor * gelu_new_3d(ggml_context * ctx, ggml_tensor * x) {
    constexpr float kCubeCoeff = 0.044715f;
    constexpr float kScale = 0.7978845608028654f;  // sqrt(2 / pi)
    ggml_tensor * x_sq = ggml_mul(ctx, x, x);
    ggml_tensor * x_cube = ggml_mul(ctx, x_sq, x);
    ggml_tensor * inner = ggml_add(ctx, x, ggml_scale(ctx, x_cube, kCubeCoeff));
    ggml_tensor * tanh_term = ggml_tanh(ctx, ggml_scale(ctx, inner, kScale));
    ggml_tensor * shifted = ggml_scale_bias(ctx, tanh_term, 1.0f, 1.0f);
    ggml_tensor * scaled = ggml_scale(ctx, ggml_mul(ctx, x, shifted), 0.5f);
    return scaled;
}

ggml_tensor * embedding_lookup(
    ggml_context * ctx,
    const KokoroWeights::EmbeddingWeights & embedding,
    ggml_tensor * ids) {
    return ggml_get_rows(ctx, embedding.weight.tensor, ids);
}

core::TensorValue permute_tensor(
    ggml_context * ctx,
    const core::TensorValue & input,
    const std::array<int, core::kMaxTensorRank> & axes) {
    core::TensorShape output_shape = {};
    output_shape.rank = input.shape.rank;
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_axis = 0; out_axis < input.shape.rank; ++out_axis) {
        const int in_axis = axes[out_axis];
        if (in_axis < 0 || in_axis >= static_cast<int>(input.shape.rank)) {
            throw std::runtime_error("Kokoro PL-BERT permute axis out of range");
        }
        output_shape.dims[out_axis] = input.shape.dims[in_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_axis);
        ggml_axes[out_ggml_axis] = core::logical_axis_to_ggml_axis(input.shape.rank, in_axis);
    }
    return core::wrap_tensor(
        ggml_permute(ctx, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

core::TensorValue ensure_contiguous(ggml_context * ctx, const core::TensorValue & input) {
    return core::has_backend_addressable_layout(input.tensor)
               ? input
               : core::wrap_tensor(ggml_cont(ctx, input.tensor), input.shape, input.type);
}

core::TensorValue heads_from_linear(
    ggml_context * ctx,
    ggml_tensor * value,
    int64_t batch,
    int64_t token_count,
    int64_t num_heads,
    int64_t head_dim) {
    auto logical = core::wrap_tensor(
        ggml_reshape_4d(ctx, value, head_dim, num_heads, token_count, batch),
        core::TensorShape::from_dims({batch, token_count, num_heads, head_dim}),
        GGML_TYPE_F32);
    auto heads = permute_tensor(ctx, logical, {0, 2, 1, 3});
    return ensure_contiguous(ctx, heads);
}

std::vector<ggml_fp16_t> build_attention_mask(const std::vector<int32_t> & token_validity, int64_t token_count, int64_t batch) {
    constexpr float kMaskedAttentionBias = -65504.0f;
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(token_count * token_count * batch), ggml_fp32_to_fp16(0.0f));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t q = 0; q < token_count; ++q) {
            for (int64_t k = 0; k < token_count; ++k) {
                const bool keep =
                    token_validity[static_cast<size_t>(b * token_count + q)] != 0 &&
                    token_validity[static_cast<size_t>(b * token_count + k)] != 0;
                const float value = keep ? 0.0f : kMaskedAttentionBias;
                const size_t offset = static_cast<size_t>(b * token_count * token_count + q * token_count + k);
                mask[offset] = ggml_fp32_to_fp16(value);
            }
        }
    }
    return mask;
}

ggml_tensor * albert_attention(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * mask,
    const KokoroWeights::AlbertWeights & bert,
    const KokoroWeights::AlbertAttentionWeights & attention) {
    const int64_t token_count = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t head_dim = bert.hidden_size / bert.num_attention_heads;

    ggml_tensor * q = linear_3d(ctx, x, attention.query);
    ggml_tensor * k = linear_3d(ctx, x, attention.key);
    ggml_tensor * v = linear_3d(ctx, x, attention.value);

    auto q_heads = heads_from_linear(ctx, q, batch, token_count, bert.num_attention_heads, head_dim);
    auto k_heads = heads_from_linear(ctx, k, batch, token_count, bert.num_attention_heads, head_dim);
    auto v_heads = heads_from_linear(ctx, v, batch, token_count, bert.num_attention_heads, head_dim);

    auto scores = core::wrap_tensor(
        ggml_mul_mat(ctx, k_heads.tensor, q_heads.tensor),
        core::TensorShape::from_dims({batch, bert.num_attention_heads, token_count, token_count}),
        GGML_TYPE_F32);
    scores = core::wrap_tensor(
        ggml_scale(ctx, scores.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    if (mask != nullptr) {
        scores = core::wrap_tensor(
            ggml_add(ctx, scores.tensor, ggml_cast(ctx, mask, GGML_TYPE_F32)),
            scores.shape,
            GGML_TYPE_F32);
    }
    scores = ensure_contiguous(ctx, scores);
    auto probs = core::wrap_tensor(ggml_soft_max(ctx, scores.tensor), scores.shape, GGML_TYPE_F32);

    const auto v_transposed = ensure_contiguous(ctx, permute_tensor(ctx, v_heads, {0, 1, 3, 2}));
    auto context = core::wrap_tensor(
        ggml_mul_mat(ctx, v_transposed.tensor, probs.tensor),
        core::TensorShape::from_dims({batch, bert.num_attention_heads, token_count, head_dim}),
        GGML_TYPE_F32);
    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = ensure_contiguous(ctx, context);
    auto attn_input = core::wrap_tensor(
        ggml_reshape_3d(ctx, context.tensor, bert.hidden_size, token_count, batch),
        core::TensorShape::from_dims({batch, token_count, bert.hidden_size}),
        GGML_TYPE_F32);

    ggml_tensor * attn = linear_3d(ctx, attn_input.tensor, attention.dense);
    attn = ggml_add(ctx, attn, x);
    return layer_norm_3d(ctx, attn, attention.layer_norm);
}

ggml_tensor * albert_layer(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * mask,
    const KokoroWeights::AlbertWeights & bert,
    const KokoroWeights::AlbertLayerWeights & layer) {
    ggml_tensor * y = albert_attention(ctx, x, mask, bert, layer.attention);
    ggml_tensor * ffn = linear_3d(ctx, y, layer.ffn);
    ffn = gelu_new_3d(ctx, ffn);
    ffn = linear_3d(ctx, ffn, layer.ffn_output);
    ffn = ggml_add(ctx, ffn, y);
    return layer_norm_3d(ctx, ffn, layer.full_layer_layer_norm);
}

ggml_tensor * plbert_last_hidden_state(
    ggml_context * ctx,
    ggml_tensor * input_ids,
    ggml_tensor * attn_mask,
    ggml_tensor * position_ids,
    ggml_tensor * token_type_ids,
    const KokoroWeights & weights) {
    ggml_tensor * word = embedding_lookup(ctx, weights.bert.embeddings.word_embeddings, input_ids);
    ggml_tensor * pos = embedding_lookup(ctx, weights.bert.embeddings.position_embeddings, position_ids);
    ggml_tensor * tok = embedding_lookup(ctx, weights.bert.embeddings.token_type_embeddings, token_type_ids);
    ggml_tensor * x = ggml_add(ctx, ggml_add(ctx, word, pos), tok);
    x = layer_norm_3d(ctx, x, weights.bert.embeddings.layer_norm);
    x = linear_3d(ctx, x, weights.bert.embedding_hidden_mapping_in);

    for (int64_t i = 0; i < weights.bert.num_hidden_layers; ++i) {
        x = albert_layer(ctx, x, attn_mask, weights.bert, weights.bert.shared_layer);
    }
    return x;
}

}  // namespace

int64_t kokoro_plbert_output_dim(std::shared_ptr<const KokoroWeights> weights, bool project_hidden) {
    if (!weights) {
        throw std::runtime_error("Kokoro weights are null");
    }
    return project_hidden ? weights->hidden_dim : weights->bert.hidden_size;
}

struct PlbertSession {
    std::shared_ptr<const KokoroWeights> weights;
    ggml_backend_t backend = nullptr;
    int64_t token_count = 0;
    bool project_hidden = true;
    int n_threads = 1;
    bool use_device_backend = false;
    ggml_context * ctx = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * attn_mask = nullptr;
    ggml_tensor * position_ids = nullptr;
    ggml_tensor * token_type_ids = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    PlbertSession(
        std::shared_ptr<const KokoroWeights> weights_in,
        ggml_backend_t backend_in,
        int64_t token_count_in,
        bool project_hidden_in,
        int n_threads_in,
        bool use_device_backend_in)
        : weights(std::move(weights_in)),
          backend(backend_in),
          token_count(token_count_in),
          project_hidden(project_hidden_in),
          n_threads(n_threads_in),
          use_device_backend(use_device_backend_in) {
        ggml_init_params params{
            /*.mem_size   =*/ kPlbertCtxBytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("failed to initialize ggml context for kokoro_plbert_encode");
        }
        try {
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, token_count, 1);
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, token_count, token_count, 1, 1);
            position_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, token_count, 1);
            token_type_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, token_count, 1);
            ggml_set_input(ids);
            ggml_set_input(attn_mask);
            ggml_set_input(position_ids);
            ggml_set_input(token_type_ids);

            ggml_tensor * hidden = plbert_last_hidden_state(ctx, ids, attn_mask, position_ids, token_type_ids, *weights);
            output = project_hidden ? linear_3d(ctx, hidden, weights->bert_encoder) : hidden;
            graph = ggml_new_graph_custom(ctx, 4096, false);
            ggml_build_forward_expand(graph, output);

            core::set_backend_threads(backend, n_threads);
            const double alloc_ms = measure_ms([&]() {
                buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            });
            if (buffer == nullptr) {
                throw std::runtime_error("failed to allocate Kokoro PL-BERT tensors");
            }
            const double fixed_input_upload_ms = measure_ms([&]() {
                std::vector<int32_t> positions(static_cast<size_t>(token_count), 0);
                std::vector<int32_t> token_types(static_cast<size_t>(token_count), 0);
                for (int64_t t = 0; t < token_count; ++t) {
                    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
                }
                ggml_backend_tensor_set(position_ids, positions.data(), 0, ggml_nbytes(position_ids));
                ggml_backend_tensor_set(token_type_ids, token_types.data(), 0, ggml_nbytes(token_type_ids));
            });
            engine::debug::timing_log_scalar("kokoro.graph.build.plbert_alloc_ms", alloc_ms);
            engine::debug::timing_log_scalar("kokoro.graph.build.plbert_fixed_input_upload_ms", fixed_input_upload_ms);
        } catch (...) {
            if (buffer) {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
            ggml_free(ctx);
            ctx = nullptr;
            throw;
        }
    }

    ~PlbertSession() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    bool matches(
        const std::shared_ptr<const KokoroWeights> & weights_in,
        int64_t token_count_in,
        bool project_hidden_in,
        int n_threads_in,
        bool use_device_backend_in) const {
        return weights.get() == weights_in.get() &&
               token_count == token_count_in &&
               project_hidden == project_hidden_in &&
               n_threads == n_threads_in &&
               use_device_backend == use_device_backend_in;
    }

    std::vector<float> run(const std::vector<int32_t> & input_ids) {
        if (input_ids.empty() || static_cast<int64_t>(input_ids.size()) > token_count) {
            throw std::runtime_error("Kokoro PL-BERT input length exceeds prepared capacity");
        }
        std::vector<int32_t> padded_ids(static_cast<size_t>(token_count), 0);
        std::memcpy(
            padded_ids.data(),
            input_ids.data(),
            static_cast<size_t>(input_ids.size()) * sizeof(int32_t));
        std::vector<int32_t> valid_mask(static_cast<size_t>(token_count), 0);
        std::fill_n(valid_mask.begin(), input_ids.size(), 1);
        const std::vector<ggml_fp16_t> attn_mask_host = build_attention_mask(valid_mask, token_count, 1);

        ggml_backend_tensor_set(ids, padded_ids.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(attn_mask, attn_mask_host.data(), 0, ggml_nbytes(attn_mask));
        core::set_backend_threads(backend, n_threads);
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("kokoro_plbert_encode graph compute failed: ") + ggml_status_to_string(status));
        }
        std::vector<float> full(static_cast<size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output, full.data(), 0, ggml_nbytes(output));
        const int64_t hidden_dim = output->ne[0];
        std::vector<float> result(static_cast<size_t>(hidden_dim * static_cast<int64_t>(input_ids.size())));
        for (int64_t row = 0; row < static_cast<int64_t>(input_ids.size()); ++row) {
            std::memcpy(
                result.data() + static_cast<ptrdiff_t>(row * hidden_dim),
                full.data() + static_cast<ptrdiff_t>(row * hidden_dim),
                static_cast<size_t>(hidden_dim) * sizeof(float));
        }
        return result;
    }
};

struct KokoroPlbertRuntime::Impl {
    std::shared_ptr<const KokoroWeights> weights;
    ggml_backend_t backend = nullptr;
    int n_threads = 1;
    bool use_device_backend = false;
    int64_t fixed_token_capacity = 0;
    std::unique_ptr<PlbertSession> session;

    Impl(
        std::shared_ptr<const KokoroWeights> weights_in,
        ggml_backend_t backend_in,
        int n_threads_in,
        bool use_device_backend_in,
        int64_t fixed_token_capacity_in)
        : weights(std::move(weights_in)),
          backend(backend_in),
          n_threads(std::max(1, n_threads_in)),
          use_device_backend(use_device_backend_in),
          fixed_token_capacity(fixed_token_capacity_in) {}

    PlbertSession & session_for(int64_t token_count, bool project_hidden) {
        const int64_t effective_token_count = fixed_token_capacity > 0 ? fixed_token_capacity : token_count;
        if (token_count > effective_token_count) {
            throw std::runtime_error("Kokoro PL-BERT request length exceeds fixed token capacity");
        }
        if (session && session->matches(weights, effective_token_count, project_hidden, n_threads, use_device_backend)) {
            return *session;
        }
        session.reset();
        const double build_ms = measure_ms([&]() {
            session = std::make_unique<PlbertSession>(
                weights,
                backend,
                effective_token_count,
                project_hidden,
                n_threads,
                use_device_backend);
        });
        engine::debug::timing_log_scalar("kokoro.graph.build.plbert_ms", build_ms);
        return *session;
    }
};

KokoroPlbertRuntime::KokoroPlbertRuntime(
    std::shared_ptr<const KokoroWeights> weights,
    ggml_backend_t backend,
    int n_threads,
    bool use_device_backend,
    int64_t fixed_token_capacity)
    : impl_(std::make_unique<Impl>(std::move(weights), backend, n_threads, use_device_backend, fixed_token_capacity)) {
    if (!impl_->weights) {
        throw std::runtime_error("Kokoro weights are null");
    }
}

KokoroPlbertRuntime::~KokoroPlbertRuntime() = default;

std::vector<float> KokoroPlbertRuntime::encode(const std::vector<int32_t> & input_ids, bool project_hidden) {
    if (input_ids.empty()) {
        throw std::runtime_error("kokoro_plbert_encode requires non-empty input_ids");
    }
    if (static_cast<int64_t>(input_ids.size()) > impl_->weights->context_length) {
        throw std::runtime_error("kokoro_plbert_encode input length exceeds PL-BERT context length");
    }
    return impl_->session_for(static_cast<int64_t>(input_ids.size()), project_hidden).run(input_ids);
}

}  // namespace kokoro_ggml
