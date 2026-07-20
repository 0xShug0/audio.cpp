#include "engine/models/ace_step/condition_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "helper_utils.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace {

namespace modules = engine::modules;
namespace binding = modules::binding;

using Clock = std::chrono::steady_clock;

struct AceStepPackedConditioning {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

class GgmlContextDeleter {
public:
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
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
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = modules::SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    auto output = heads.front();
    for (size_t i = 1; i < heads.size(); ++i) {
        output = modules::ConcatModule({1}).build(ctx, output, heads[i]);
    }
    return output;
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask) {
    return modules::ScaledDotProductAttentionModule({
        dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
}

core::TensorValue encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const AceStepConditionEncoderLayerWeights & weights,
    const AceStepDiffusionConfig & config,
    const std::optional<core::TensorValue> & attention_mask) {
    const int64_t dim = ace_step_diffusion_attention_head_dim(config, "ACE-Step condition encoder");
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
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(ctx, weights.input_norm));
    auto q = q_proj.build(ctx, attn_in, binding::linear_data(ctx, weights.q_proj));
    auto k = k_proj.build(ctx, attn_in, binding::linear_data(ctx, weights.k_proj));
    auto v = v_proj.build(ctx, attn_in, binding::linear_data(ctx, weights.v_proj));
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), binding::norm_data(ctx, weights.q_norm));
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), binding::norm_data(ctx, weights.k_norm));
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads =
        repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k), kv_repeats);
    auto v_heads =
        repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = add.build(ctx, input, o_proj.build(ctx, context, binding::linear_data(ctx, weights.o_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(ctx, weights.post_norm));
    auto gate =
        modules::LinearModule(binding::linear_config(
                                  config.hidden_size,
                                  config.intermediate_size,
                                  false))
            .build(ctx, ff_in, binding::linear_data(ctx, weights.gate_proj));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up =
        modules::LinearModule(binding::linear_config(
                                  config.hidden_size,
                                  config.intermediate_size,
                                  false))
            .build(ctx, ff_in, binding::linear_data(ctx, weights.up_proj));
    auto ff =
        modules::LinearModule(binding::linear_config(
                                  config.intermediate_size,
                                  config.hidden_size,
                                  false))
            .build(ctx, modules::MulModule{}.build(ctx, gate, up), binding::linear_data(ctx, weights.down_proj));
    return add.build(ctx, x, ff);
}

std::vector<float> build_padding_attention_mask_values(int64_t tokens, int64_t valid_tokens) {
    if (tokens <= 0 || valid_tokens <= 0 || valid_tokens > tokens) {
        throw std::runtime_error("ACE-Step padding attention mask requires 0 < valid_tokens <= tokens");
    }
    std::vector<float> values(static_cast<size_t>(tokens * tokens), 0.0F);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t q = 0; q < valid_tokens; ++q) {
        for (int64_t k = valid_tokens; k < tokens; ++k) {
            values[static_cast<size_t>(q * tokens + k)] = neg_inf;
        }
    }
    return values;
}

AceStepPackedConditioning single_batch_conditioning(
    const AceStepTextConditioning & conditioning,
    int64_t padded_tokens) {
    if (conditioning.tokens < 0 || conditioning.hidden_size <= 0 || padded_tokens < conditioning.tokens) {
        throw std::runtime_error("ACE-Step conditioning shape is invalid");
    }
    if (static_cast<int64_t>(conditioning.values.size()) != conditioning.tokens * conditioning.hidden_size) {
        throw std::runtime_error("ACE-Step conditioning values do not match token/hidden shape");
    }
    AceStepPackedConditioning out;
    out.batch = 1;
    out.tokens = padded_tokens;
    out.hidden_size = conditioning.hidden_size;
    out.values.assign(static_cast<size_t>(padded_tokens * conditioning.hidden_size), 0.0F);
    std::copy(conditioning.values.begin(), conditioning.values.end(), out.values.begin());
    out.attention_mask.assign(static_cast<size_t>(padded_tokens), 0);
    std::fill_n(out.attention_mask.begin(), static_cast<size_t>(conditioning.tokens), 1);
    return out;
}

AceStepPackedConditioning single_batch_conditioning(const AceStepTextConditioning & conditioning) {
    return single_batch_conditioning(conditioning, conditioning.tokens);
}

AceStepPackedConditioning pack_sequences(
    const AceStepPackedConditioning & first,
    const AceStepPackedConditioning & second) {
    if (first.batch != second.batch || first.hidden_size != second.hidden_size) {
        throw std::runtime_error("ACE-Step pack_sequences requires matching batch and hidden size");
    }
    if (static_cast<int64_t>(first.attention_mask.size()) != first.batch * first.tokens ||
        static_cast<int64_t>(second.attention_mask.size()) != second.batch * second.tokens) {
        throw std::runtime_error("ACE-Step pack_sequences attention mask shape is invalid");
    }
    if (static_cast<int64_t>(first.values.size()) != first.batch * first.tokens * first.hidden_size ||
        static_cast<int64_t>(second.values.size()) != second.batch * second.tokens * second.hidden_size) {
        throw std::runtime_error("ACE-Step pack_sequences tensor shape is invalid");
    }

    AceStepPackedConditioning out;
    out.batch = first.batch;
    out.tokens = first.tokens + second.tokens;
    out.hidden_size = first.hidden_size;
    out.values.assign(static_cast<size_t>(out.batch * out.tokens * out.hidden_size), 0.0F);
    out.attention_mask.assign(static_cast<size_t>(out.batch * out.tokens), 0);

    for (int64_t b = 0; b < out.batch; ++b) {
        std::vector<int64_t> order;
        order.reserve(static_cast<size_t>(out.tokens));
        for (int64_t i = 0; i < first.tokens; ++i) {
            order.push_back(i);
        }
        for (int64_t i = 0; i < second.tokens; ++i) {
            order.push_back(first.tokens + i);
        }
        std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
            const int32_t lhs_mask = lhs < first.tokens
                ? first.attention_mask[static_cast<size_t>(b * first.tokens + lhs)]
                : second.attention_mask[static_cast<size_t>(b * second.tokens + (lhs - first.tokens))];
            const int32_t rhs_mask = rhs < first.tokens
                ? first.attention_mask[static_cast<size_t>(b * first.tokens + rhs)]
                : second.attention_mask[static_cast<size_t>(b * second.tokens + (rhs - first.tokens))];
            return lhs_mask > rhs_mask;
        });

        int64_t valid = 0;
        for (int64_t out_index = 0; out_index < out.tokens; ++out_index) {
            const int64_t src_index = order[static_cast<size_t>(out_index)];
            const bool from_first = src_index < first.tokens;
            const int64_t local_index = from_first ? src_index : (src_index - first.tokens);
            const auto & src = from_first ? first : second;
            const int32_t mask = src.attention_mask[static_cast<size_t>(b * src.tokens + local_index)];
            valid += mask != 0 ? 1 : 0;
            const float * src_ptr = src.values.data()
                + static_cast<size_t>((b * src.tokens + local_index) * src.hidden_size);
            float * dst_ptr = out.values.data()
                + static_cast<size_t>((b * out.tokens + out_index) * out.hidden_size);
            std::copy_n(src_ptr, static_cast<size_t>(out.hidden_size), dst_ptr);
        }
        std::fill_n(
            out.attention_mask.begin() + static_cast<std::ptrdiff_t>(b * out.tokens),
            static_cast<size_t>(valid),
            1);
    }

    return out;
}

AceStepEncoderConditioning to_encoder_conditioning(const AceStepPackedConditioning & packed) {
    if (packed.batch != 1) {
        throw std::runtime_error("ACE-Step encoder conditioning currently supports only batch size 1");
    }
    AceStepEncoderConditioning out;
    out.values = packed.values;
    out.attention_mask = packed.attention_mask;
    out.tokens = packed.tokens;
    out.hidden_size = packed.hidden_size;
    return out;
}

}  // namespace

class AceStepConditionEncoderRuntime::Impl {
public:
    class TextProjectorGraph {
    public:
        TextProjectorGraph(
            ggml_backend_t backend,
            int threads,
            std::shared_ptr<const AceStepAssets> assets,
            std::shared_ptr<const AceStepConditionEncoderWeights> weights,
            int64_t tokens)
            : backend_(backend),
              threads_(threads),
              assets_(std::move(assets)),
              weights_(std::move(weights)),
              tokens_(tokens) {
            build();
        }

        ~TextProjectorGraph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        bool can_run(int64_t tokens) const noexcept {
            return tokens_ >= tokens;
        }

        AceStepTextConditioning run(const AceStepTextConditioning & input) const {
            if (input.tokens <= 0 || input.tokens > tokens_ ||
                input.hidden_size != assets_->config.diffusion.text_hidden_dim) {
                throw std::runtime_error("ACE-Step text projector input shape mismatch");
            }
            std::vector<float> padded(
                static_cast<size_t>(tokens_ * assets_->config.diffusion.text_hidden_dim),
                0.0F);
            std::copy(input.values.begin(), input.values.end(), padded.begin());
            core::write_tensor_f32(input_value_, padded);
            core::set_backend_threads(backend_, threads_);
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step text projector graph compute failed");
            }
            AceStepTextConditioning out;
            out.tokens = input.tokens;
            out.hidden_size = assets_->config.diffusion.hidden_size;
            std::vector<float> full;
            core::read_tensor_f32_into(output_, full);
            out.values.assign(
                full.begin(),
                full.begin() + static_cast<std::ptrdiff_t>(input.tokens * out.hidden_size));
            return out;
        }

    private:
        void build() {
            const auto & config = assets_->config.diffusion;
            ggml_init_params params{8ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step text projector ggml context initialization failed");
            }
            core::ModuleBuildContext build_ctx{ctx_.get()};
            input_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, tokens_, config.text_hidden_dim}));
            input_value_ = input_;
            auto projected = modules::LinearModule(binding::linear_config(
                                                      config.text_hidden_dim,
                                                      config.hidden_size,
                                                      false))
                                 .build(build_ctx, input_, binding::linear_data(build_ctx, weights_->text_projector));
            output_ = projected.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
            ggml_build_forward_expand(graph_, output_);
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("ACE-Step text projector backend buffer allocation failed");
            }
        }

        ggml_backend_t backend_ = nullptr;
        int threads_ = 1;
        std::shared_ptr<const AceStepAssets> assets_;
        std::shared_ptr<const AceStepConditionEncoderWeights> weights_;
        int64_t tokens_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        core::TensorValue input_value_;
        core::TensorValue input_;
        ggml_tensor * output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    class LyricEncoderGraph {
    public:
        LyricEncoderGraph(
            ggml_backend_t backend,
            int threads,
            std::shared_ptr<const AceStepAssets> assets,
            std::shared_ptr<const AceStepConditionEncoderWeights> weights,
            int64_t tokens)
            : backend_(backend),
              threads_(threads),
              assets_(std::move(assets)),
              weights_(std::move(weights)),
              tokens_(tokens) {
            build();
        }

        ~LyricEncoderGraph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        bool can_run(int64_t tokens) const noexcept {
            return tokens_ >= tokens;
        }

        AceStepTextConditioning run(const AceStepTextConditioning & input) const {
            if (input.tokens <= 0 || input.tokens > tokens_ ||
                input.hidden_size != assets_->config.diffusion.text_hidden_dim) {
                throw std::runtime_error("ACE-Step lyric encoder input shape mismatch");
            }
            std::vector<float> padded(
                static_cast<size_t>(tokens_ * assets_->config.diffusion.text_hidden_dim),
                0.0F);
            std::copy(input.values.begin(), input.values.end(), padded.begin());
            core::write_tensor_f32(input_value_, padded);
            core::write_tensor_f16(
                padding_mask_value_,
                build_padding_attention_mask_values(tokens_, input.tokens));
            core::set_backend_threads(backend_, threads_);
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step lyric encoder graph compute failed");
            }
            AceStepTextConditioning out;
            out.tokens = input.tokens;
            out.hidden_size = assets_->config.diffusion.hidden_size;
            std::vector<float> full;
            core::read_tensor_f32_into(output_, full);
            out.values.assign(
                full.begin(),
                full.begin() + static_cast<std::ptrdiff_t>(input.tokens * out.hidden_size));
            return out;
        }

    private:
        void build() {
            const auto & config = assets_->config.diffusion;
            ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step lyric encoder ggml context initialization failed");
            }
            core::ModuleBuildContext build_ctx{ctx_.get()};

            input_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, tokens_, config.text_hidden_dim}));
            input_value_ = input_;
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, tokens_);
            auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({tokens_}), GGML_TYPE_I32);
            sliding_mask_value_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, 1, tokens_, tokens_}));
            padding_mask_value_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, 1, tokens_, tokens_}));
            auto hidden = modules::LinearModule({config.text_hidden_dim, config.hidden_size, true})
                              .build(
                                  build_ctx,
                                  input_,
                                  {weights_->lyric_embed_weight, weights_->lyric_embed_bias});

            const auto sliding_mask = core::wrap_tensor(
                sliding_mask_value_.tensor,
                core::TensorShape::from_dims({1, 1, tokens_, tokens_}),
                GGML_TYPE_F16);
            const auto padding_mask = core::wrap_tensor(
                padding_mask_value_.tensor,
                core::TensorShape::from_dims({1, 1, tokens_, tokens_}),
                GGML_TYPE_F16);
            for (int64_t i = 0; i < config.num_lyric_encoder_hidden_layers; ++i) {
                const std::optional<core::TensorValue> mask =
                    (config.layer_types[static_cast<size_t>(i)] == "sliding_attention")
                    ? std::optional<core::TensorValue>(
                          modules::AddModule{}.build(build_ctx, sliding_mask, padding_mask))
                    : std::optional<core::TensorValue>(padding_mask);
                hidden = encoder_layer(build_ctx, hidden, positions, weights_->lyric_layers[static_cast<size_t>(i)], config, mask);
            }
            hidden = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                         .build(build_ctx, hidden, binding::norm_data(build_ctx, weights_->lyric_norm));

            output_ = hidden.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            ggml_build_forward_expand(graph_, output_);
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("ACE-Step lyric encoder backend buffer allocation failed");
            }

            std::vector<int32_t> position_values(static_cast<size_t>(tokens_), 0);
            for (int64_t i = 0; i < tokens_; ++i) {
                position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
            const std::vector<float> sliding_mask_values =
                ace_step_bidirectional_sliding_mask_values(
                    tokens_,
                    config.sliding_window,
                    "ACE-Step");
            std::vector<ggml_fp16_t> sliding_mask_f16(sliding_mask_values.size());
            for (size_t i = 0; i < sliding_mask_values.size(); ++i) {
                sliding_mask_f16[i] = ggml_fp32_to_fp16(sliding_mask_values[i]);
            }
            ggml_backend_tensor_set(
                sliding_mask_value_.tensor,
                sliding_mask_f16.data(),
                0,
                sliding_mask_f16.size() * sizeof(ggml_fp16_t));
        }

        ggml_backend_t backend_ = nullptr;
        int threads_ = 1;
        std::shared_ptr<const AceStepAssets> assets_;
        std::shared_ptr<const AceStepConditionEncoderWeights> weights_;
        int64_t tokens_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        core::TensorValue input_;
        core::TensorValue input_value_;
        ggml_tensor * positions_ = nullptr;
        core::TensorValue sliding_mask_value_;
        core::TensorValue padding_mask_value_;
        ggml_tensor * output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    class TimbreEncoderGraph {
    public:
        TimbreEncoderGraph(
            ggml_backend_t backend,
            int threads,
            std::shared_ptr<const AceStepAssets> assets,
            std::shared_ptr<const AceStepConditionEncoderWeights> weights,
            int64_t frames)
            : backend_(backend),
              threads_(threads),
              assets_(std::move(assets)),
              weights_(std::move(weights)),
              frames_(frames) {
            build();
        }

        ~TimbreEncoderGraph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        bool can_run(int64_t frames) const noexcept {
            return frames_ >= frames;
        }

        AceStepTextConditioning run_one(const std::vector<float> & packed_reference, int64_t frames) const {
            const auto & config = assets_->config.diffusion;
            if (frames <= 0 || frames > frames_ ||
                static_cast<int64_t>(packed_reference.size()) != frames * config.timbre_hidden_dim) {
                throw std::runtime_error("ACE-Step timbre encoder input shape mismatch");
            }
            std::vector<float> padded(
                static_cast<size_t>(frames_ * config.timbre_hidden_dim),
                0.0F);
            std::copy(packed_reference.begin(), packed_reference.end(), padded.begin());
            core::write_tensor_f32(input_value_, padded);
            core::write_tensor_f16(
                padding_mask_value_,
                build_padding_attention_mask_values(frames_, frames));
            core::set_backend_threads(backend_, threads_);
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step timbre encoder graph compute failed");
            }
            AceStepTextConditioning out;
            out.tokens = 1;
            out.hidden_size = config.hidden_size;
            core::read_tensor_f32_into(output_, out.values);
            return out;
        }

    private:
        void build() {
            const auto & config = assets_->config.diffusion;
            ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step timbre encoder ggml context initialization failed");
            }
            core::ModuleBuildContext build_ctx{ctx_.get()};

            input_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, frames_, config.timbre_hidden_dim}));
            input_value_ = input_;
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, frames_);
            auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({frames_}), GGML_TYPE_I32);
            sliding_mask_value_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, 1, frames_, frames_}));
            padding_mask_value_ = core::make_tensor(
                build_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, 1, frames_, frames_}));
            auto hidden = modules::LinearModule({config.timbre_hidden_dim, config.hidden_size, true})
                              .build(
                                  build_ctx,
                                  input_,
                                  {weights_->timbre_embed_weight, weights_->timbre_embed_bias});
            const auto sliding_mask = core::wrap_tensor(
                sliding_mask_value_.tensor,
                core::TensorShape::from_dims({1, 1, frames_, frames_}),
                GGML_TYPE_F16);
            const auto padding_mask = core::wrap_tensor(
                padding_mask_value_.tensor,
                core::TensorShape::from_dims({1, 1, frames_, frames_}),
                GGML_TYPE_F16);
            for (int64_t i = 0; i < config.num_timbre_encoder_hidden_layers; ++i) {
                const std::optional<core::TensorValue> mask =
                    (config.layer_types[static_cast<size_t>(i)] == "sliding_attention")
                    ? std::optional<core::TensorValue>(
                          modules::AddModule{}.build(build_ctx, sliding_mask, padding_mask))
                    : std::optional<core::TensorValue>(padding_mask);
                hidden = encoder_layer(build_ctx, hidden, positions, weights_->timbre_layers[static_cast<size_t>(i)], config, mask);
            }
            hidden = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                         .build(build_ctx, hidden, binding::norm_data(build_ctx, weights_->timbre_norm));
            hidden = modules::SliceModule({1, 0, 1}).build(build_ctx, hidden);
            hidden = core::reshape_tensor(build_ctx, hidden, core::TensorShape::from_dims({1, config.hidden_size}));

            output_ = hidden.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            ggml_build_forward_expand(graph_, output_);
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("ACE-Step timbre encoder backend buffer allocation failed");
            }

            std::vector<int32_t> position_values(static_cast<size_t>(frames_), 0);
            for (int64_t i = 0; i < frames_; ++i) {
                position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
            const std::vector<float> sliding_mask_values =
                ace_step_bidirectional_sliding_mask_values(
                    frames_,
                    config.sliding_window,
                    "ACE-Step");
            std::vector<ggml_fp16_t> sliding_mask_f16(sliding_mask_values.size());
            for (size_t i = 0; i < sliding_mask_values.size(); ++i) {
                sliding_mask_f16[i] = ggml_fp32_to_fp16(sliding_mask_values[i]);
            }
            ggml_backend_tensor_set(
                sliding_mask_value_.tensor,
                sliding_mask_f16.data(),
                0,
                sliding_mask_f16.size() * sizeof(ggml_fp16_t));
        }

        ggml_backend_t backend_ = nullptr;
        int threads_ = 1;
        std::shared_ptr<const AceStepAssets> assets_;
        std::shared_ptr<const AceStepConditionEncoderWeights> weights_;
        int64_t frames_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        core::TensorValue input_;
        core::TensorValue input_value_;
        ggml_tensor * positions_ = nullptr;
        core::TensorValue sliding_mask_value_;
        core::TensorValue padding_mask_value_;
        ggml_tensor * output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          threads_(std::max(1, execution.config().threads)),
          weights_(dit_weights_runtime->condition_encoder_weights()) {
        if (assets_ == nullptr) {
            throw std::runtime_error("ACE-Step condition encoder requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step condition encoder backend initialization failed");
        }
    }

    ~Impl() {
        timbre_encoder_.reset();
        lyric_encoder_.reset();
        text_projector_.reset();
    }

    AceStepEncoderConditioning encode(
        const AceStepTextConditioning & text_hidden_states,
        const AceStepTextConditioning & lyric_token_embeddings,
        const std::vector<float> & refer_audio_acoustic_hidden_states_packed,
        int64_t refer_audio_count,
        int64_t refer_audio_frames,
        const std::vector<int32_t> & refer_audio_order_mask) const {
        const auto total_start = Clock::now();
        if (text_hidden_states.hidden_size != assets_->config.diffusion.text_hidden_dim) {
            throw std::runtime_error("ACE-Step condition encoder text hidden size mismatch");
        }
        if (lyric_token_embeddings.hidden_size != assets_->config.diffusion.text_hidden_dim) {
            throw std::runtime_error("ACE-Step condition encoder lyric hidden size mismatch");
        }
        const auto project_text_start = Clock::now();
        auto text_projected = project_text(text_hidden_states);
        engine::debug::timing_log_scalar(
            "ace_step.condition_encoder.project_text_ms",
            engine::debug::elapsed_ms(project_text_start, Clock::now()));
        const auto encode_lyrics_start = Clock::now();
        auto lyric_hidden = encode_lyrics(lyric_token_embeddings);
        engine::debug::timing_log_scalar(
            "ace_step.condition_encoder.encode_lyrics_ms",
            engine::debug::elapsed_ms(encode_lyrics_start, Clock::now()));
        const auto encode_timbre_start = Clock::now();
        auto timbre = encode_timbre(
            refer_audio_acoustic_hidden_states_packed,
            refer_audio_count,
            refer_audio_frames,
            refer_audio_order_mask);
        engine::debug::timing_log_scalar(
            "ace_step.condition_encoder.encode_timbre_ms",
            engine::debug::elapsed_ms(encode_timbre_start, Clock::now()));

        const auto pack_start = Clock::now();
        auto packed = pack_sequences(
            single_batch_conditioning(lyric_hidden),
            timbre);
        packed = pack_sequences(
            packed,
            single_batch_conditioning(text_projected));
        AceStepEncoderConditioning out = to_encoder_conditioning(packed);
        engine::debug::timing_log_scalar(
            "ace_step.condition_encoder.pack_ms",
            engine::debug::elapsed_ms(pack_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.condition_encoder.tokens", out.tokens);
        engine::debug::timing_log_scalar("ace_step.condition_encoder.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    void release_runtime_graphs() const {
        text_projector_.reset();
        lyric_encoder_.reset();
        timbre_encoder_.reset();
    }

private:
    AceStepTextConditioning project_text(const AceStepTextConditioning & input) const {
        if (!text_projector_ || !text_projector_->can_run(input.tokens)) {
            text_projector_.reset();
            text_projector_ = std::make_unique<TextProjectorGraph>(
                backend_,
                threads_,
                assets_,
                weights_,
                input.tokens);
        }
        return text_projector_->run(input);
    }

    AceStepTextConditioning encode_lyrics(const AceStepTextConditioning & input) const {
        if (!lyric_encoder_ || !lyric_encoder_->can_run(input.tokens)) {
            lyric_encoder_.reset();
            lyric_encoder_ = std::make_unique<LyricEncoderGraph>(
                backend_,
                threads_,
                assets_,
                weights_,
                input.tokens);
        }
        return lyric_encoder_->run(input);
    }

    AceStepPackedConditioning encode_timbre(
        const std::vector<float> & packed_values,
        int64_t refer_audio_count,
        int64_t refer_audio_frames,
        const std::vector<int32_t> & refer_audio_order_mask) const {
        const auto & config = assets_->config.diffusion;
        if (refer_audio_count <= 0 || refer_audio_frames <= 0) {
            throw std::runtime_error("ACE-Step timbre encoder requires positive reference count and frame count");
        }
        if (static_cast<int64_t>(refer_audio_order_mask.size()) != refer_audio_count) {
            throw std::runtime_error("ACE-Step timbre encoder order mask size mismatch");
        }
        if (static_cast<int64_t>(packed_values.size()) != refer_audio_count * refer_audio_frames * config.timbre_hidden_dim) {
            throw std::runtime_error("ACE-Step timbre encoder packed values shape mismatch");
        }
        if (!timbre_encoder_ || !timbre_encoder_->can_run(refer_audio_frames)) {
            timbre_encoder_.reset();
            timbre_encoder_ = std::make_unique<TimbreEncoderGraph>(
                backend_,
                threads_,
                assets_,
                weights_,
                refer_audio_frames);
        }
        for (int32_t order : refer_audio_order_mask) {
            if (order != 0) {
                throw std::runtime_error("ACE-Step timbre encoder currently supports only single-batch reference ordering");
            }
        }
        AceStepPackedConditioning out;
        out.batch = 1;
        out.tokens = refer_audio_count;
        out.hidden_size = config.hidden_size;
        out.values.assign(static_cast<size_t>(refer_audio_count * config.hidden_size), 0.0F);
        out.attention_mask.assign(static_cast<size_t>(refer_audio_count), 1);
        for (int64_t i = 0; i < refer_audio_count; ++i) {
            const float * src = packed_values.data()
                + static_cast<size_t>(i * refer_audio_frames * config.timbre_hidden_dim);
            std::vector<float> one(src, src + static_cast<std::ptrdiff_t>(refer_audio_frames * config.timbre_hidden_dim));
            auto encoded = timbre_encoder_->run_one(one, refer_audio_frames);
            std::copy(
                encoded.values.begin(),
                encoded.values.end(),
                out.values.begin() + static_cast<std::ptrdiff_t>(i * config.hidden_size));
        }
        return out;
    }

    std::shared_ptr<const AceStepAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const AceStepConditionEncoderWeights> weights_;
    mutable std::unique_ptr<TextProjectorGraph> text_projector_;
    mutable std::unique_ptr<LyricEncoderGraph> lyric_encoder_;
    mutable std::unique_ptr<TimbreEncoderGraph> timbre_encoder_;
};

AceStepConditionEncoderRuntime::AceStepConditionEncoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const AceStepAssets> assets,
    std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime)
    : impl_(std::make_unique<Impl>(execution, std::move(assets), std::move(dit_weights_runtime))) {}

AceStepConditionEncoderRuntime::~AceStepConditionEncoderRuntime() = default;

AceStepConditionEncoderRuntime::AceStepConditionEncoderRuntime(AceStepConditionEncoderRuntime &&) noexcept = default;

AceStepConditionEncoderRuntime & AceStepConditionEncoderRuntime::operator=(AceStepConditionEncoderRuntime &&) noexcept =
    default;

AceStepEncoderConditioning AceStepConditionEncoderRuntime::encode(
    const AceStepTextConditioning & text_hidden_states,
    const AceStepTextConditioning & lyric_token_embeddings,
    const std::vector<float> & refer_audio_acoustic_hidden_states_packed,
    int64_t refer_audio_count,
    int64_t refer_audio_frames,
    const std::vector<int32_t> & refer_audio_order_mask) const {
    return impl_->encode(
        text_hidden_states,
        lyric_token_embeddings,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_count,
        refer_audio_frames,
        refer_audio_order_mask);
}

void AceStepConditionEncoderRuntime::release_runtime_graphs() const {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::ace_step
