#pragma once

// granite_speech_2b text decoder (Task C2): Granite weight binding and
// generation over the shared decoder stack.
//
// Two entry points share one spec/config binding (values below are verbatim
// from the pinned checkpoint config.json text_config + generation_config):
//   - GraniteSpeech2BDecoderRuntime: thin wrapper calling
//     GreedyCausalDecoderRuntime::generate(..., reuse_graphs=true) with one
//     complete prompt per call and zero retained prefix steps. This is the
//     path Task D sessions will use.
//   - GraniteSpeech2BDecoder: logits-capable decoder over
//     CausalDecoderRuntime (device embedding lookup, host audio injection,
//     bounded-block prefill into the token-decode cache, single-token forced
//     decode). Used by the parity probe prefill/generate stages and the C2
//     unit tests that compare all-100353-logit vectors, forced-step margins,
//     and cache-path equivalence.
//
// Binding values (C2 table verbatim):
//   - language_model.model.embed_tokens + 40 language_model.model.layers +
//     final RMS norm [2048] + untied language_model.lm_head [100353,2048];
//     per layer Q [2048,2048], K/V [512,2048] each, out [2048,2048],
//     gate/up [4096,2048] each, down [2048,4096], 2x RMSNorm [2048];
//     RMS eps 1e-5; no Q/K norm; bias-free; GQA 4 query heads per KV head.
//   - Embed x12 AFTER audio substitution, exactly once per generated-token
//     lookup, at the common DecoderHiddenModule/static-cache entry boundary.
//   - Attn scale 0.0078125 pre-softmax on plain and accelerated paths, causal
//     mask. Residual 0.22 after attention out-proj and after MLP down.
//   - RoPE: 128 channels NeoX half-pair, base 10000, YaRN factor 4
//     (freq_scale 1/4), orig context 4096, limit 16384, beta 32/1,
//     interpolation blend (ext_factor 1), magnitude factor
//     0.1*ln(4)+1 ~= 1.13863. Upstream: transformers 4.57.0
//     modeling_granite.py GraniteRotaryEmbedding + modeling_rope_utils.py
//     _compute_yarn_parameters/get_mscale.
//   - Logits /8. Stop on EOS (100257) or budget.
//
// Non-flash CPU reference policy: ManualRepeat prefill/static attention
// modes with flash attention disallowed.

#include "engine/community_models/granite_speech_2b/prompt.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/runtime/greedy_causal_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::granite_speech_2b {

// Granite text-decoder binding constants (see header doc for provenance).
inline constexpr int64_t kGraniteSpeech2BDecoderLayers = 40;
inline constexpr int64_t kGraniteSpeech2BDecoderKvHeads = 4;
inline constexpr float kGraniteSpeech2BEmbedScale = 12.0F;
inline constexpr float kGraniteSpeech2BAttentionScale = 0.0078125F;
inline constexpr float kGraniteSpeech2BResidualScale = 0.22F;
inline constexpr float kGraniteSpeech2BLogitDivisor = 8.0F;
inline constexpr float kGraniteSpeech2BYarnFreqScale = 0.25F;  // 1 / factor
inline constexpr float kGraniteSpeech2BYarnExtFactor = 1.0F;   // interpolation blend
// YaRN magnitude factor: ggml derives 0.1 * ln(1 / freq_scale) + 1 ~= 1.13863
// from freq_scale when ext_factor != 0 (external/ggml/src/ggml-cpu/ops.cpp
// rope_yarn: mscale *= 1 + 0.1 * log(1 / freq_scale)), matching HF
// _compute_yarn_parameters get_mscale(factor). Passing 0.1*ln(4)+1 here
// would apply the factor twice, so the config carries 1.0.
inline constexpr float kGraniteSpeech2BYarnAttnFactor = 1.0F;
inline constexpr float kGraniteSpeech2BYarnBetaFast = 32.0F;
inline constexpr float kGraniteSpeech2BYarnBetaSlow = 1.0F;
inline constexpr int32_t kGraniteSpeech2BYarnOrigContext = 4096;

// Shared CausalDecoderConfig for the Granite text decoder (also mirrored
// into the greedy spec by make_granite_speech_2b_greedy_decoder_spec).
modules::CausalDecoderConfig make_granite_speech_2b_causal_decoder_config();

// GreedyCausalDecoderSpec for generate(..., reuse_graphs=true): identical
// math to the config above, plus tensor names and EOS/budget.
runtime::GreedyCausalDecoderSpec make_granite_speech_2b_greedy_decoder_spec();

// Prompt budget gate: expanded prompt length + max_new_tokens must fit the
// 16384 model limit. Throws std::runtime_error otherwise (also on empty
// prompt or non-positive budget).
void granite_speech_2b_require_generation_budget(
    int64_t prompt_len,
    int64_t max_new_tokens,
    int64_t max_position_embeddings = kGraniteSpeech2BMaxPositionEmbeddings);

// Thin generation wrapper over the shared greedy runtime (Task D path).
class GraniteSpeech2BDecoderRuntime {
public:
    GraniteSpeech2BDecoderRuntime(
        std::shared_ptr<const assets::TensorSource> weights_source,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes = 1024ull * 1024ull * 1024ull,
        size_t decode_graph_arena_bytes = 1024ull * 1024ull * 1024ull,
        size_t weight_context_bytes = 8ull * 1024ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::F32);
    ~GraniteSpeech2BDecoderRuntime();

    GraniteSpeech2BDecoderRuntime(const GraniteSpeech2BDecoderRuntime &) = delete;
    GraniteSpeech2BDecoderRuntime & operator=(const GraniteSpeech2BDecoderRuntime &) = delete;

    // One complete prompt per call, zero retained prefix steps. EOS IDs are
    // excluded from the return (mirrors the shared greedy runtime); at most
    // max_new_tokens IDs are returned.
    std::vector<int32_t> generate(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Logits-capable decoder over CausalDecoderRuntime (probe + C2 units).
class GraniteSpeech2BDecoder {
public:
    GraniteSpeech2BDecoder(
        std::shared_ptr<const assets::TensorSource> weights_source,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes = 2ull * 1024ull * 1024ull * 1024ull,
        size_t decode_graph_arena_bytes = 1024ull * 1024ull * 1024ull,
        size_t weight_context_bytes = 8ull * 1024ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::F32);
    ~GraniteSpeech2BDecoder();

    GraniteSpeech2BDecoder(const GraniteSpeech2BDecoder &) = delete;
    GraniteSpeech2BDecoder & operator=(const GraniteSpeech2BDecoder &) = delete;

    struct PrefillResult {
        int64_t steps = 0;
        // Merged token-major [steps, 2048] embeddings AFTER audio
        // substitution, BEFORE the x12 scale (matches merged_prefill.f32).
        std::vector<float> merged;
        // Host-side x12 of merged (matches scaled_prefill.f32); the graph
        // applies the same factor once at the hidden-module entry, which
        // last-logit parity proves.
        std::vector<float> scaled;
        // Last-position logits (already /8).
        std::vector<float> last_logits;
    };

    // Device embedding lookup + host audio injection + bounded-block prefill
    // into the token-decode cache (keep_prefix_steps = 0: the full prompt is
    // always recomputed). cache capacity covers steps + max_new_tokens.
    PrefillResult prefill(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens,
        int64_t chunk_steps = 64);

    // Single forced-decode step continuing the prefilled cache. Returns the
    // full [100353] logit vector (already /8).
    std::vector<float> decode_token(int32_t token);

    // Greedy continuation with EOS/budget stop (EOS excluded, mirrors the
    // shared greedy runtime loop).
    std::vector<int32_t> generate_greedy(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::granite_speech_2b
