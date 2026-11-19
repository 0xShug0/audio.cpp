#pragma once

// Granite text-decoder configuration and generation over the shared runtime.
// Embedding x12 is applied after audio injection; attention scale is 1/128,
// residual branches are scaled by 0.22, and the untied LM head logits by 1/8.

#include "engine/community_models/granite_speech5/prompt.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/runtime/greedy_causal_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::granite_speech5 {

// Granite text-decoder binding constants (see header doc for provenance).
inline constexpr int64_t kGraniteSpeech5DecoderLayers = 40;
inline constexpr int64_t kGraniteSpeech5DecoderKvHeads = 4;
inline constexpr float kGraniteSpeech5EmbedScale = 12.0F;
inline constexpr float kGraniteSpeech5AttentionScale = 0.0078125F;
inline constexpr float kGraniteSpeech5ResidualScale = 0.22F;
inline constexpr float kGraniteSpeech5LogitDivisor = 8.0F;
inline constexpr float kGraniteSpeech5YarnFreqScale = 0.25F;  // 1 / factor
inline constexpr float kGraniteSpeech5YarnExtFactor = 1.0F;   // interpolation blend
// YaRN magnitude factor: ggml derives 0.1 * ln(1 / freq_scale) + 1 ~= 1.13863
// from freq_scale when ext_factor != 0 (external/ggml/src/ggml-cpu/ops.cpp
// rope_yarn: mscale *= 1 + 0.1 * log(1 / freq_scale)), matching HF
// _compute_yarn_parameters get_mscale(factor). Passing 0.1*ln(4)+1 here
// would apply the factor twice, so the config carries 1.0.
inline constexpr float kGraniteSpeech5YarnAttnFactor = 1.0F;
inline constexpr float kGraniteSpeech5YarnBetaFast = 32.0F;
inline constexpr float kGraniteSpeech5YarnBetaSlow = 1.0F;
inline constexpr int32_t kGraniteSpeech5YarnOrigContext = 4096;

// Shared CausalDecoderConfig for the Granite text decoder (also mirrored
// into the greedy spec by make_granite_speech5_greedy_decoder_spec).
modules::CausalDecoderConfig make_granite_speech5_causal_decoder_config(
    core::BackendType backend = core::BackendType::Cpu);

// GreedyCausalDecoderSpec for generate(..., reuse_graphs=true): identical
// math to the config above, plus tensor names and EOS/budget.
runtime::GreedyCausalDecoderSpec make_granite_speech5_greedy_decoder_spec(
    core::BackendType backend = core::BackendType::Cpu);

// Prompt budget gate: expanded prompt length + max_new_tokens must fit the
// 16384 model limit. Throws std::runtime_error otherwise (also on empty
// prompt or non-positive budget).
void granite_speech5_require_generation_budget(
    int64_t prompt_len,
    int64_t max_new_tokens,
    int64_t max_position_embeddings = kGraniteSpeech5MaxPositionEmbeddings);

}  // namespace engine::community_models::granite_speech5
