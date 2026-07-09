#include "engine/framework/modules/attention/qwen_causal_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

void validate_config(const QwenCausalDecoderConfig & config) {
    if (config.stack.hidden_size <= 0 || config.logits_size <= 0) {
        throw std::runtime_error("QwenCausalDecoderConfig requires positive hidden and logits sizes");
    }
    if (config.stack.layers <= 0) {
        throw std::runtime_error("QwenCausalDecoderConfig requires a positive layer count");
    }
}

void validate_steps(int64_t steps, const char * label) {
    if (steps <= 0) {
        throw std::runtime_error(std::string(label) + " requires positive step count");
    }
}

}  // namespace

QwenCausalDecoderModule::QwenCausalDecoderModule(QwenCausalDecoderConfig config)
    : config_(std::move(config)) {
    validate_config(config_);
}

const QwenCausalDecoderConfig & QwenCausalDecoderModule::config() const noexcept {
    return config_;
}

QwenCausalDecoderOutputs QwenCausalDecoderModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenCausalDecoderWeights & weights,
    const std::optional<QwenDecoderStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (input.shape.rank != 3 || input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenCausalDecoderModule input shape must be [batch, steps, hidden]");
    }

    auto stack = QwenDecoderStackModule(config_.stack)
                     .build(ctx, input, positions, weights.stack, prefix_state, attention_mask);
    auto hidden_sequence = RMSNormModule({config_.stack.hidden_size, config_.stack.rms_norm_eps, true, false})
                               .build(ctx, stack.output, weights.final_norm);
    core::TensorValue hidden = hidden_sequence;
    if (config_.logits_mode == QwenCausalDecoderLogitsMode::LastStep) {
        const int64_t steps = hidden_sequence.shape.dims[1];
        hidden = SliceModule({1, steps - 1, 1}).build(ctx, hidden_sequence);
    }

    const auto logits = LinearModule({
                            config_.stack.hidden_size,
                            config_.logits_size,
                            config_.use_lm_head_bias,
                            config_.lm_head_precision,
                        })
                            .build(ctx, hidden, weights.lm_head);
    return {std::move(stack.output), hidden, logits, std::move(stack.state)};
}

QwenCausalDecoderStaticCacheOutputs QwenCausalDecoderModule::build_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const QwenCausalDecoderWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const std::optional<core::TensorValue> & cache_slot) const {
    if (graph == nullptr) {
        throw std::runtime_error("QwenCausalDecoderModule static-cache build requires a graph");
    }
    validate_steps(cache_steps, "QwenCausalDecoderModule static-cache build");
    if (input.shape.rank != 3 || input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenCausalDecoderModule static-cache input shape must be [batch, steps, hidden]");
    }
    if (input.shape.dims[0] != 1 || input.shape.dims[1] != 1) {
        throw std::runtime_error("QwenCausalDecoderModule static-cache build currently supports single-token decode");
    }
    if (static_cast<int64_t>(weights.stack.layers.size()) != config_.stack.layers) {
        throw std::runtime_error("QwenCausalDecoderWeights layer count does not match config");
    }

    const int64_t step_elems = config_.stack.num_key_value_heads * config_.stack.head_dim;
    std::vector<core::TensorValue> cache_keys;
    std::vector<core::TensorValue> cache_values;
    cache_keys.reserve(weights.stack.layers.size());
    cache_values.reserve(weights.stack.layers.size());

    auto x = input;
    const QwenDecoderLayerModule layer_module(qwen_decoder_layer_config_from_stack(config_.stack));
    for (const auto & layer : weights.stack.layers) {
        cache_keys.push_back(core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, cache_steps, config_.stack.num_key_value_heads, config_.stack.head_dim})));
        cache_values.push_back(core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, cache_steps, config_.stack.num_key_value_heads, config_.stack.head_dim})));
        auto out = layer_module.build_with_static_cache_tail(
            ctx,
            graph,
            x,
            positions,
            layer,
            cache_keys.back(),
            cache_values.back(),
            cache_slot,
            attention_mask);
        x = out.output;
    }

    auto hidden = RMSNormModule({config_.stack.hidden_size, config_.stack.rms_norm_eps, true, false})
                      .build(ctx, x, weights.final_norm);
    const auto logits = LinearModule({
                            config_.stack.hidden_size,
                            config_.logits_size,
                            config_.use_lm_head_bias,
                            config_.lm_head_precision,
                        })
                            .build(ctx, hidden, weights.lm_head);
    return {
        std::move(x),
        hidden,
        logits,
        runtime::TransformerKVCache(cache_steps, step_elems, std::move(cache_keys), std::move(cache_values)),
    };
}

std::vector<int32_t> qwen_position_ids(int64_t steps, int64_t offset) {
    validate_steps(steps, "qwen_position_ids");
    std::vector<int32_t> out(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(offset + i);
    }
    return out;
}

std::vector<ggml_fp16_t> qwen_causal_prefill_mask_values(int64_t batch_size, int64_t steps) {
    if (batch_size <= 0) {
        throw std::runtime_error("qwen_causal_prefill_mask_values requires positive batch size");
    }
    validate_steps(steps, "qwen_causal_prefill_mask_values");
    const auto masked = ggml_fp32_to_fp16(-INFINITY);
    const auto visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> one(static_cast<size_t>(steps * steps), masked);
    for (int64_t row = 0; row < steps; ++row) {
        const size_t row_offset = static_cast<size_t>(row * steps);
        for (int64_t col = 0; col <= row; ++col) {
            one[row_offset + static_cast<size_t>(col)] = visible;
        }
    }
    if (batch_size == 1) {
        return one;
    }
    std::vector<ggml_fp16_t> out;
    out.reserve(static_cast<size_t>(batch_size) * one.size());
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        out.insert(out.end(), one.begin(), one.end());
    }
    return out;
}

void write_qwen_causal_prefill_mask(
    ggml_tensor * tensor,
    int64_t batch_size,
    int64_t steps) {
    if (tensor == nullptr) {
        throw std::runtime_error("write_qwen_causal_prefill_mask requires a tensor");
    }
    auto values = qwen_causal_prefill_mask_values(batch_size, steps);
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
}

void write_qwen_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot) {
    if (tensor == nullptr) {
        throw std::runtime_error("write_qwen_cached_step_mask requires a tensor");
    }
    validate_steps(mask_steps, "write_qwen_cached_step_mask");
    if (visible_prefix_steps < 0 || visible_prefix_steps > mask_steps) {
        throw std::runtime_error("write_qwen_cached_step_mask visible prefix is out of range");
    }
    if (current_slot < 0 || current_slot >= mask_steps) {
        throw std::runtime_error("write_qwen_cached_step_mask current slot is out of range");
    }
    const auto masked = ggml_fp32_to_fp16(-INFINITY);
    const auto visible = ggml_fp32_to_fp16(0.0F);
    if (scratch.size() != static_cast<size_t>(mask_steps)) {
        scratch.resize(static_cast<size_t>(mask_steps));
    }
    std::fill(scratch.begin(), scratch.end(), masked);
    for (int64_t i = 0; i < visible_prefix_steps; ++i) {
        scratch[static_cast<size_t>(i)] = visible;
    }
    scratch[static_cast<size_t>(current_slot)] = visible;
    ggml_backend_tensor_set(tensor, scratch.data(), 0, scratch.size() * sizeof(ggml_fp16_t));
}

}  // namespace engine::modules
