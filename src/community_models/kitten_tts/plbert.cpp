#include "engine/community_models/kitten_tts/plbert.h"

#include "engine/community_models/kitten_tts/assets.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace engine::models::kitten_tts {

constexpr size_t kPlbertCtxBytes = 128ull * 1024ull * 1024ull;

namespace core = engine::core;

namespace {

using engine::debug::measure_ms;

namespace modules = engine::modules;

core::TensorValue heads_from_linear(core::ModuleBuildContext &ctx, const core::TensorValue &value, int64_t batch,
                                    int64_t token_count, int64_t num_heads, int64_t head_dim) {
    auto logical =
        core::reshape_tensor(ctx, value, core::TensorShape::from_dims({batch, token_count, num_heads, head_dim}));
    auto heads = modules::TransposeModule({{0, 2, 1, 3}, logical.shape.rank}).build(ctx, logical);
    return core::ensure_backend_addressable_layout(ctx, heads);
}

std::vector<ggml_fp16_t> build_attention_mask(const std::vector<int32_t> &token_validity, int64_t token_count,
                                              int64_t batch) {
    constexpr float kMaskedAttentionBias = -65504.0f;
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(token_count * token_count * batch), ggml_fp32_to_fp16(0.0f));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t q = 0; q < token_count; ++q) {
            for (int64_t k = 0; k < token_count; ++k) {
                const bool keep = token_validity[static_cast<size_t>(b * token_count + q)] != 0 &&
                                  token_validity[static_cast<size_t>(b * token_count + k)] != 0;
                const float value = keep ? 0.0f : kMaskedAttentionBias;
                const size_t offset = static_cast<size_t>(b * token_count * token_count + q * token_count + k);
                mask[offset] = ggml_fp32_to_fp16(value);
            }
        }
    }
    return mask;
}

core::TensorValue albert_attention(core::ModuleBuildContext &ctx, const core::TensorValue &x, ggml_tensor *mask,
                                   const KittenWeights::AlbertWeights &bert,
                                   const KittenWeights::AlbertAttentionWeights &attention) {
    const int64_t batch = x.shape.dims[0];
    const int64_t token_count = x.shape.dims[1];
    const int64_t head_dim = bert.hidden_size / bert.num_attention_heads;

    auto q =
        modules::LinearModule({attention.query.in_features, attention.query.out_features, attention.query.use_bias})
            .build(ctx, x, {attention.query.weight, attention.query.bias});
    auto k = modules::LinearModule({attention.key.in_features, attention.key.out_features, attention.key.use_bias})
                 .build(ctx, x, {attention.key.weight, attention.key.bias});
    auto v =
        modules::LinearModule({attention.value.in_features, attention.value.out_features, attention.value.use_bias})
            .build(ctx, x, {attention.value.weight, attention.value.bias});

    auto q_heads = heads_from_linear(ctx, q, batch, token_count, bert.num_attention_heads, head_dim);
    auto k_heads = heads_from_linear(ctx, k, batch, token_count, bert.num_attention_heads, head_dim);
    auto v_heads = heads_from_linear(ctx, v, batch, token_count, bert.num_attention_heads, head_dim);

    auto k_transposed = modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads);
    auto scores = modules::MatMulModule{}.build(ctx, q_heads, k_transposed);
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))),
                               scores.shape, GGML_TYPE_F32);
    if (mask != nullptr) {
        auto mask_value =
            core::wrap_tensor(ggml_cast(ctx.ggml, mask, GGML_TYPE_F32),
                              core::TensorShape::from_dims({batch, 1, token_count, token_count}), GGML_TYPE_F32);
        mask_value = modules::RepeatModule({scores.shape}).build(ctx, mask_value);
        scores = modules::AddModule{}.build(ctx, scores, mask_value);
    }
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto probs = modules::SoftmaxModule{}.build(ctx, scores);

    auto context = modules::MatMulModule{}.build(ctx, probs, v_heads);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    auto attn_input =
        core::reshape_tensor(ctx, context, core::TensorShape::from_dims({batch, token_count, bert.hidden_size}));

    auto attn =
        modules::LinearModule({attention.dense.in_features, attention.dense.out_features, attention.dense.use_bias})
            .build(ctx, attn_input, {attention.dense.weight, attention.dense.bias});
    attn = modules::AddModule{}.build(ctx, attn, x);
    return modules::LayerNormModule({attention.layer_norm.channels, attention.layer_norm.eps, true, true})
        .build(ctx, attn, {attention.layer_norm.weight, attention.layer_norm.bias});
}

core::TensorValue albert_layer(core::ModuleBuildContext &ctx, const core::TensorValue &x, ggml_tensor *mask,
                               const KittenWeights::AlbertWeights &bert,
                               const KittenWeights::AlbertLayerWeights &layer) {
    auto y = albert_attention(ctx, x, mask, bert, layer.attention);
    auto ffn = modules::LinearModule({layer.ffn.in_features, layer.ffn.out_features, layer.ffn.use_bias})
                   .build(ctx, y, {layer.ffn.weight, layer.ffn.bias});
    ffn = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ffn);
    ffn =
        modules::LinearModule({layer.ffn_output.in_features, layer.ffn_output.out_features, layer.ffn_output.use_bias})
            .build(ctx, ffn, {layer.ffn_output.weight, layer.ffn_output.bias});
    ffn = modules::AddModule{}.build(ctx, ffn, y);
    return modules::LayerNormModule({layer.full_layer_layer_norm.channels, layer.full_layer_layer_norm.eps, true, true})
        .build(ctx, ffn, {layer.full_layer_layer_norm.weight, layer.full_layer_layer_norm.bias});
}

core::TensorValue plbert_last_hidden_state(core::ModuleBuildContext &ctx, ggml_tensor *input_ids,
                                           ggml_tensor *attn_mask, ggml_tensor *position_ids,
                                           ggml_tensor *token_type_ids, const KittenWeights &weights) {
    auto input = core::wrap_tensor(input_ids, core::TensorShape::from_dims({1, input_ids->ne[0]}), GGML_TYPE_I32);
    auto positions =
        core::wrap_tensor(position_ids, core::TensorShape::from_dims({1, position_ids->ne[0]}), GGML_TYPE_I32);
    auto token_types =
        core::wrap_tensor(token_type_ids, core::TensorShape::from_dims({1, token_type_ids->ne[0]}), GGML_TYPE_I32);
    auto word = modules::EmbeddingModule({weights.bert.embeddings.word_embeddings.num_embeddings,
                                          weights.bert.embeddings.word_embeddings.embedding_dim})
                    .build(ctx, input, weights.bert.embeddings.word_embeddings.weight);
    auto pos = modules::EmbeddingModule({weights.bert.embeddings.position_embeddings.num_embeddings,
                                         weights.bert.embeddings.position_embeddings.embedding_dim})
                   .build(ctx, positions, weights.bert.embeddings.position_embeddings.weight);
    auto tok = modules::EmbeddingModule({weights.bert.embeddings.token_type_embeddings.num_embeddings,
                                         weights.bert.embeddings.token_type_embeddings.embedding_dim})
                   .build(ctx, token_types, weights.bert.embeddings.token_type_embeddings.weight);
    auto x = modules::AddModule{}.build(ctx, modules::AddModule{}.build(ctx, word, pos), tok);
    x = modules::LayerNormModule(
            {weights.bert.embeddings.layer_norm.channels, weights.bert.embeddings.layer_norm.eps, true, true})
            .build(ctx, x, {weights.bert.embeddings.layer_norm.weight, weights.bert.embeddings.layer_norm.bias});
    x = modules::LinearModule({weights.bert.embedding_hidden_mapping_in.in_features,
                               weights.bert.embedding_hidden_mapping_in.out_features,
                               weights.bert.embedding_hidden_mapping_in.use_bias})
            .build(ctx, x,
                   {weights.bert.embedding_hidden_mapping_in.weight, weights.bert.embedding_hidden_mapping_in.bias});

    for (int64_t i = 0; i < weights.bert.num_hidden_layers; ++i) {
        x = albert_layer(ctx, x, attn_mask, weights.bert, weights.bert.shared_layer);
    }
    return x;
}

} // namespace

int64_t kitten_plbert_output_dim(std::shared_ptr<const KittenWeights> weights, bool project_hidden) {
    if (!weights) {
        throw std::runtime_error("Kitten weights are null");
    }
    return project_hidden ? weights->hidden_dim : weights->bert.hidden_size;
}

struct PlbertSession {
    std::shared_ptr<const KittenWeights> weights;
    ggml_backend_t backend = nullptr;
    int64_t token_count = 0;
    bool project_hidden = true;
    int n_threads = 1;
    bool use_device_backend = false;
    ggml_context *ctx = nullptr;
    ggml_tensor *ids = nullptr;
    ggml_tensor *attn_mask = nullptr;
    ggml_tensor *position_ids = nullptr;
    ggml_tensor *token_type_ids = nullptr;
    ggml_tensor *output = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    std::vector<int32_t> position_ids_host;
    std::vector<int32_t> token_type_ids_host;

    PlbertSession(std::shared_ptr<const KittenWeights> weights_in, ggml_backend_t backend_in, int64_t token_count_in,
                  bool project_hidden_in, int n_threads_in, bool use_device_backend_in)
        : weights(std::move(weights_in)), backend(backend_in), token_count(token_count_in),
          project_hidden(project_hidden_in), n_threads(n_threads_in), use_device_backend(use_device_backend_in) {
        ggml_init_params params{
            /*.mem_size   =*/kPlbertCtxBytes,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("failed to initialize ggml context for kitten_plbert_encode");
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

            core::ModuleBuildContext build_ctx{ctx, "kitten.plbert"};
            auto hidden = plbert_last_hidden_state(build_ctx, ids, attn_mask, position_ids, token_type_ids, *weights);
            if (project_hidden) {
                hidden = modules::LinearModule({weights->bert_encoder.in_features, weights->bert_encoder.out_features,
                                                weights->bert_encoder.use_bias})
                             .build(build_ctx, hidden, {weights->bert_encoder.weight, weights->bert_encoder.bias});
            }
            output = hidden.tensor;
            ggml_set_output(output);
            if (output->view_src != nullptr) {
                ggml_set_output(output->view_src);
            }
            graph = ggml_new_graph_custom(ctx, 4096, false);
            ggml_build_forward_expand(graph, output);

            core::set_backend_threads(backend, n_threads);
            const double alloc_ms = measure_ms([&]() {
                gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
                    !ggml_gallocr_alloc_graph(gallocr, graph)) {
                    throw std::runtime_error("failed to allocate Kitten PL-BERT graph tensors");
                }
            });
            if (engine::core::uses_host_graph_plan(backend)) {
                plan = engine::core::create_backend_graph_plan_if_host(backend, graph);
                if (!plan) {
                    throw std::runtime_error("failed to create Kitten PL-BERT graph plan");
                }
            }
            position_ids_host.assign(static_cast<size_t>(token_count), 0);
            token_type_ids_host.assign(static_cast<size_t>(token_count), 0);
            for (int64_t t = 0; t < token_count; ++t) {
                position_ids_host[static_cast<size_t>(t)] = static_cast<int32_t>(t);
            }
            engine::debug::timing_log_scalar("kitten.graph.build.plbert_alloc_ms", alloc_ms);
        } catch (...) {
            if (plan) {
                engine::core::free_backend_graph_plan(backend, plan);
            }
            core::release_backend_graph_resources(backend, graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
            ggml_free(ctx);
            ctx = nullptr;
            throw;
        }
    }

    ~PlbertSession() {
        if (plan) {
            engine::core::free_backend_graph_plan(backend, plan);
        }
        core::release_backend_graph_resources(backend, graph);
        if (gallocr) {
            ggml_gallocr_free(gallocr);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    bool matches(const std::shared_ptr<const KittenWeights> &weights_in, int64_t token_count_in, bool project_hidden_in,
                 int n_threads_in, bool use_device_backend_in) const {
        return weights.get() == weights_in.get() && token_count == token_count_in &&
               project_hidden == project_hidden_in && n_threads == n_threads_in &&
               use_device_backend == use_device_backend_in;
    }

    std::vector<float> run(const std::vector<int32_t> &input_ids) {
        if (input_ids.empty() || static_cast<int64_t>(input_ids.size()) > token_count) {
            throw std::runtime_error("Kitten PL-BERT input length exceeds prepared capacity");
        }
        std::vector<int32_t> padded_ids(static_cast<size_t>(token_count), 0);
        std::memcpy(padded_ids.data(), input_ids.data(), static_cast<size_t>(input_ids.size()) * sizeof(int32_t));
        std::vector<int32_t> valid_mask(static_cast<size_t>(token_count), 0);
        std::fill_n(valid_mask.begin(), input_ids.size(), 1);
        std::vector<ggml_fp16_t> attn_mask_host;
        const double mask_build_ms =
            measure_ms([&]() { attn_mask_host = build_attention_mask(valid_mask, token_count, 1); });

        const double input_upload_ms = measure_ms([&]() {
            core::write_tensor_i32(
                core::wrap_tensor(ids, core::TensorShape::from_dims({1, token_count}), GGML_TYPE_I32), padded_ids);
            ggml_backend_tensor_set(attn_mask, attn_mask_host.data(), 0, ggml_nbytes(attn_mask));
            core::write_tensor_i32(
                core::wrap_tensor(position_ids, core::TensorShape::from_dims({1, token_count}), GGML_TYPE_I32),
                position_ids_host);
            core::write_tensor_i32(
                core::wrap_tensor(token_type_ids, core::TensorShape::from_dims({1, token_count}), GGML_TYPE_I32),
                token_type_ids_host);
        });
        core::set_backend_threads(backend, n_threads);
        ggml_status status = GGML_STATUS_SUCCESS;
        const double compute_ms =
            measure_ms([&]() { status = engine::core::compute_backend_graph(backend, graph, plan); });
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("kitten_plbert_encode graph compute failed: ") +
                                     ggml_status_to_string(status));
        }
        std::vector<float> full;
        const double output_read_ms = measure_ms([&]() { full = core::read_tensor_f32(output); });
        const int64_t hidden_dim = output->ne[0];
        std::vector<float> result(static_cast<size_t>(hidden_dim * static_cast<int64_t>(input_ids.size())));
        const double output_slice_ms = measure_ms([&]() {
            for (int64_t row = 0; row < static_cast<int64_t>(input_ids.size()); ++row) {
                std::memcpy(result.data() + static_cast<ptrdiff_t>(row * hidden_dim),
                            full.data() + static_cast<ptrdiff_t>(row * hidden_dim),
                            static_cast<size_t>(hidden_dim) * sizeof(float));
            }
        });
        engine::debug::timing_log_scalar("kitten.plbert.mask_build_ms", mask_build_ms);
        engine::debug::timing_log_scalar("kitten.plbert.input_upload_ms", input_upload_ms);
        engine::debug::timing_log_scalar("kitten.plbert.compute_ms", compute_ms);
        engine::debug::timing_log_scalar("kitten.plbert.output_read_ms", output_read_ms);
        engine::debug::timing_log_scalar("kitten.plbert.output_slice_ms", output_slice_ms);
        return result;
    }
};

struct KittenPlbertRuntime::Impl {
    std::shared_ptr<const KittenWeights> weights;
    ggml_backend_t backend = nullptr;
    int n_threads = 1;
    bool use_device_backend = false;
    int64_t fixed_token_capacity = 0;
    std::unique_ptr<PlbertSession> session;

    Impl(std::shared_ptr<const KittenWeights> weights_in, ggml_backend_t backend_in, int n_threads_in,
         bool use_device_backend_in, int64_t fixed_token_capacity_in)
        : weights(std::move(weights_in)), backend(backend_in), n_threads(std::max(1, n_threads_in)),
          use_device_backend(use_device_backend_in), fixed_token_capacity(fixed_token_capacity_in) {}

    PlbertSession &session_for(int64_t token_count, bool project_hidden) {
        const int64_t effective_token_count = fixed_token_capacity > 0 ? fixed_token_capacity : token_count;
        if (token_count > effective_token_count) {
            throw std::runtime_error("Kitten PL-BERT request length exceeds fixed token capacity");
        }
        if (session &&
            session->matches(weights, effective_token_count, project_hidden, n_threads, use_device_backend)) {
            return *session;
        }
        session.reset();
        const double build_ms = measure_ms([&]() {
            session = std::make_unique<PlbertSession>(weights, backend, effective_token_count, project_hidden,
                                                      n_threads, use_device_backend);
        });
        engine::debug::timing_log_scalar("kitten.graph.build.plbert_ms", build_ms);
        return *session;
    }
};

KittenPlbertRuntime::KittenPlbertRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend,
                                         int n_threads, bool use_device_backend, int64_t fixed_token_capacity)
    : impl_(std::make_unique<Impl>(std::move(weights), backend, n_threads, use_device_backend, fixed_token_capacity)) {
    if (!impl_->weights) {
        throw std::runtime_error("Kitten weights are null");
    }
}

KittenPlbertRuntime::~KittenPlbertRuntime() = default;

void KittenPlbertRuntime::prepare(bool project_hidden) {
    if (impl_->fixed_token_capacity <= 0) {
        throw std::runtime_error("kitten_plbert prepare requires a fixed token capacity");
    }
    (void)impl_->session_for(impl_->fixed_token_capacity, project_hidden);
}

std::vector<float> KittenPlbertRuntime::encode(const std::vector<int32_t> &input_ids, bool project_hidden) {
    if (input_ids.empty()) {
        throw std::runtime_error("kitten_plbert_encode requires non-empty input_ids");
    }
    if (static_cast<int64_t>(input_ids.size()) > impl_->weights->context_length) {
        throw std::runtime_error("kitten_plbert_encode input length exceeds PL-BERT context length");
    }
    return impl_->session_for(static_cast<int64_t>(input_ids.size()), project_hidden).run(input_ids);
}

} // namespace engine::models::kitten_tts
