#pragma once

// The LFM3/LFM2 layer library shared by all four stacks: backbone, encoder
// (LFM2 names), depth decoder and codec decoder.

#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/norm_modules.h"

#include <ggml-cpp.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {
class WeightStores;  // src/community_models/lfm2_audio/weight_stores.h
}  // namespace engine::community_models::lfm2_audio

namespace engine::community_models::lfm3_audio {

using lfm2_audio::WeightStores;

namespace blocks {

// Every LFM3-Audio matmul asks for F32 precision. Only CUDA reads it: above 8 rows it keeps F16 weights out of
// cuBLAS's F16-output GEMM. BF16 and quantized weights run the same kernels either way.
inline constexpr ggml_prec kMatmulPrecision = GGML_PREC_F32;

enum class LayerKind { Conv, FullAttention, SlidingAttention };
enum class Names { Lfm3, Lfm2 };  // Lfm2: self_attn.out_proj, q_layernorm/k_layernorm, no gate, no post_* norms

struct StackConfig {
    int64_t hidden = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0, conv_kernel = 0, sliding_window = 0;
    float eps = 1e-5f;
    std::vector<LayerKind> layers;
    bool sandwich_norm = false;  // post_operator_norm, post_ffn_norm
    bool output_gate = false;    // sigmoid(gate_proj(hn)) per head, before o_proj
    bool causal = true;          // false: depth decoder (no mask at all)
    float full_theta = 1e6f, sliding_theta = 1e6f;
    int64_t full_rotary_dims = 0;  // 0 = head_dim; else zero-tail freq_factors (backbone: 64 of 128)
    Names names = Names::Lfm3;
};

struct ConvWeights {
    core::TensorValue in_proj, out_proj, kernel_t;  // kernel_t: F32 [K, hidden] (row k contiguous)
    core::TensorValue kernel;                       // F32 [hidden, K] (HF layout), for ssm_conv
};

struct AttentionWeights {
    core::TensorValue q, k, v, o;
    modules::NormWeights q_norm, k_norm;
    std::optional<core::TensorValue> gate;
};

struct LayerWeights {
    LayerKind kind;
    modules::NormWeights operator_norm, ffn_norm;
    std::optional<modules::NormWeights> post_operator_norm, post_ffn_norm;
    ConvWeights conv;
    AttentionWeights attention;
    core::TensorValue w1, w2, w3;
};

struct StackWeights {
    std::vector<LayerWeights> layers;
    modules::NormWeights final_norm;
    std::optional<core::TensorValue> full_freq_factors;
};

// prefix e.g. "lfm.model.", "encoder.lfm.lfm.", "depthformer.model.", "decoder.lfm."; final norm "<prefix>embedding_norm.weight".
// `backend` is the stores' backend (WeightStores does not expose it); it only feeds the BF16 probe.
StackWeights load_stack(WeightStores & stores, ggml_backend_t backend, const assets::TensorSource & source,
                        const std::string & prefix, const StackConfig & config, const Lfm3WeightOptions & options);

// A [rows, cols] matmul weight in the storage lfm3_storage_type picks. The source may add trailing unit dims
// (the checkpoint's encoder lin_out is a Conv1d [768, 4096, 1]); any other shape must match.
core::TensorValue load_matmul(WeightStores & stores, ggml_backend_t backend, const assets::TensorSource & source,
                              const std::string & name, const std::vector<int64_t> & shape, const Lfm3WeightOptions & options);

// A no_alloc context with room for one graph of up to `nodes` nodes.
ggml_context_ptr graph_context(size_t nodes, const std::string & what);
// Runs `graph` with the execution's thread count and waits for it.
void compute(core::ExecutionContext & execution, ggml_cgraph * graph, const std::string & what);

// Persistent per-layer state for stateful stacks (backbone, codec).
struct LayerState {
    core::TensorValue key_cache, value_cache;  // attention: logical [1, capacity, kv_heads, head_dim], F16 or F32
    core::TensorValue conv_state;              // conv: F32 logical [K-1, hidden]
};

// The layers' state in one backend buffer, zeroed: K/V caches of `capacity` rows of `cache_type`, F32 conv tails.
class StackState {
public:
    StackState(core::ExecutionContext & execution, const StackConfig & config, int64_t capacity, ggml_type cache_type,
               const std::string & what);
    [[nodiscard]] int64_t capacity() const noexcept { return capacity_; }
    void clear();

    std::vector<LayerState> layers;

private:
    int64_t capacity_ = 0;
    ggml_context_ptr ctx_;
    ggml_backend_buffer_ptr buffer_;
};

struct StepInputs {
    core::TensorValue x;                             // [1, S, hidden]
    core::TensorValue positions;                     // I32 [S] (rope)
    std::optional<core::TensorValue> slots;          // I64 [S] cache rows to write; stateful only
    int64_t kv_rows = 0;                             // stateful: attend over cache rows [0, kv_rows)
    std::optional<core::TensorValue> full_mask;      // F16 [kv, S] (kv = kv_rows stateful, S stateless)
    std::optional<core::TensorValue> sliding_mask;   // F16 [kv, S]
};

// state == nullptr: stateless whole sequence (encoder, depth); conv starts from zeros.
// Returns the final-normed hidden [1, S, hidden]. Writes K/V rows and the new conv tails into `graph`.
core::TensorValue build_stack(core::ModuleBuildContext & ctx, ggml_cgraph * graph, const StackConfig & config,
                              const StackWeights & weights, const StepInputs & in, std::vector<LayerState> * state,
                              const Lfm3WeightOptions & options);

// Every capacity and kv_rows of a stateful stack is a multiple of this (CUDA's FA vector kernel needs K rows % 256).
inline constexpr int64_t kKvRowAlignment = 256;

// Building blocks, exposed for unit tests:
// y[t] = sum_k padded[t + k] * kernel_t[k], returned as [1, S, D]; ctx.backend_type and steps pick the lowering.
// `kernel` (optional, [D, K]) saves the ssm_conv lowering a transpose of kernel_t.
core::TensorValue causal_conv(core::ModuleBuildContext & ctx, const core::TensorValue & padded /*[S+K-1, D]*/,
                              const core::TensorValue & kernel_t /*[K, D]*/, int64_t steps,
                              const core::TensorValue * kernel = nullptr);
core::TensorValue gated_attention(core::ModuleBuildContext & ctx, const StackConfig & config, LayerKind kind,
                                  const AttentionWeights & w, const core::TensorValue & normed, const StepInputs & in,
                                  LayerState * state, const StackWeights & stack, const Lfm3WeightOptions & options);
// 0 visible, -inf hidden. Visible iff k <= q && (window == 0 || q - k < window).
// q_pos: absolute positions of the queries; keys are absolute rows [0, kv).
std::vector<ggml_fp16_t> window_mask(const std::vector<int32_t> & q_pos, int64_t kv, int64_t window);

}  // namespace blocks

}  // namespace engine::community_models::lfm3_audio
