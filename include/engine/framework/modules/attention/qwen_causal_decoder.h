#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/qwen_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/kv_cache.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <ggml.h>

namespace engine::modules {

enum class QwenCausalDecoderLogitsMode {
    LastStep,
    AllSteps,
};

struct QwenCausalDecoderConfig {
    QwenDecoderStackConfig stack;
    int64_t logits_size = 0;
    QwenCausalDecoderLogitsMode logits_mode = QwenCausalDecoderLogitsMode::LastStep;
    bool use_lm_head_bias = false;
    ggml_prec lm_head_precision = GGML_PREC_DEFAULT;
};

struct QwenCausalDecoderWeights {
    QwenDecoderStackWeights stack;
    NormWeights final_norm;
    LinearWeights lm_head;
};

struct QwenCausalDecoderOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    QwenDecoderStackState state;
};

struct QwenCausalDecoderStaticCacheOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    runtime::TransformerKVCache cache;
};

class QwenCausalDecoderModule {
public:
    explicit QwenCausalDecoderModule(QwenCausalDecoderConfig config);

    const QwenCausalDecoderConfig & config() const noexcept;

    QwenCausalDecoderOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenCausalDecoderWeights & weights,
        const std::optional<QwenDecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    QwenCausalDecoderStaticCacheOutputs build_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenCausalDecoderWeights & weights,
        int64_t cache_steps,
        const core::TensorValue & attention_mask,
        const std::optional<core::TensorValue> & cache_slot = std::nullopt) const;

private:
    QwenCausalDecoderConfig config_;
};

std::vector<int32_t> qwen_position_ids(int64_t steps, int64_t offset = 0);

std::vector<ggml_fp16_t> qwen_causal_prefill_mask_values(int64_t batch_size, int64_t steps);

void write_qwen_causal_prefill_mask(
    ggml_tensor * tensor,
    int64_t batch_size,
    int64_t steps);

void write_qwen_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot);

}  // namespace engine::modules
