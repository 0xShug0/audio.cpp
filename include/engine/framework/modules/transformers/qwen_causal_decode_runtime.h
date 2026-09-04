#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/runtime/kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

enum class QwenCausalDecodeOutputMode {
    Logits,
    Hidden,
};

struct QwenCausalDecodeRuntimeConfig {
    std::string trace_name = "qwen_causal_decode";
    QwenCausalDecoderConfig decoder;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    QwenCausalDecodeOutputMode output_mode = QwenCausalDecodeOutputMode::Logits;
    bool return_hidden = false;
    std::optional<ggml_type> readback_round_type;
    std::vector<int32_t> logits_readback_token_ids;
    int64_t sliding_window = 0;
    bool evict_cuda_graph_cache_on_release = false;
};

struct QwenCausalDecodeRuntimeWeights {
    core::TensorValue token_embedding;
    QwenDecoderStackWeights stack;
    NormWeights final_norm;
    std::optional<LinearWeights> lm_head;
};

struct QwenCausalPrefillResult {
    std::vector<float> logits;
    std::vector<float> hidden;
    runtime::TransformerKVState state;
};

struct QwenCausalBatchedPrefillResult {
    std::vector<float> logits;
    std::vector<float> hidden;
    runtime::TransformerBatchedKVState state;
};

struct QwenCausalDecodeStepResult {
    std::vector<float> logits;
    std::vector<float> hidden;
};

class QwenCausalDecodeRuntime {
public:
    QwenCausalDecodeRuntime(
        core::ExecutionContext & execution,
        QwenCausalDecodeRuntimeConfig config,
        QwenCausalDecodeRuntimeWeights weights);
    ~QwenCausalDecodeRuntime();

    QwenCausalDecodeRuntime(const QwenCausalDecodeRuntime &) = delete;
    QwenCausalDecodeRuntime & operator=(const QwenCausalDecodeRuntime &) = delete;

    QwenCausalPrefillResult prefill_tokens(const std::vector<int32_t> & token_ids);
    QwenCausalPrefillResult prefill_embeddings(const std::vector<float> & embeddings, int64_t steps);
    // 固定 graph 的 padded prefill：graph 按 padded_steps 建一次并复用（避免因
    // 不同 steps 重建 prefill graph 破坏 CUDA pool 逆序约束）。embeddings 必须
    // 是 padded_steps × hidden（padding 零），只有前 valid_steps 参与位置/mask。
    // 返回 state 只含 valid_steps（截断）。若 padded_steps < 当前已建 steps 则复用。
    QwenCausalPrefillResult prefill_embeddings_padded(
        const std::vector<float> & embeddings,
        int64_t padded_steps,
        int64_t valid_steps);

    QwenCausalBatchedPrefillResult prefill_tokens_batched(
        const std::vector<int32_t> & token_ids,
        int64_t batch_size,
        int64_t steps);
    QwenCausalBatchedPrefillResult prefill_embeddings_batched(
        const std::vector<float> & embeddings,
        int64_t batch_size,
        int64_t steps);

    void start_decode_tokens(const runtime::TransformerKVState & state, int64_t required_cache_steps);
    void start_decode_embeddings(const runtime::TransformerKVState & state, int64_t required_cache_steps);
    QwenCausalDecodeStepResult decode_token(int32_t token);
    QwenCausalDecodeStepResult decode_embedding(const std::vector<float> & embedding);

    void start_decode_tokens_batched(
        const runtime::TransformerBatchedKVState & state,
        int64_t required_cache_steps);
    void start_decode_embeddings_batched(
        const runtime::TransformerBatchedKVState & state,
        int64_t required_cache_steps);
    QwenCausalDecodeStepResult decode_tokens_batched(const std::vector<int32_t> & tokens);
    // 每步 batched decode。active_mask（可选，长度==batch_size）：只有置 1 的行才
    // 真正前进一步、mask 才暴露其前缀；置 0 的行全 -inf（不读自身 stale KV）、
    // 不 advance —— 非活跃行彻底 inert，杜绝"冻结行携带上一请求 stale KV 参与
    // decode"导致的跨请求串音。
    QwenCausalDecodeStepResult decode_embeddings_batched(
        const std::vector<float> & embeddings,
        int64_t batch_size,
        const std::vector<uint8_t> & active_mask = {});
    // 冻结/重置某 batch 行的解码位置（非活跃行 end=0，mask 全 -inf）。
    void set_batched_member_end(int64_t batch, int64_t end);

    // Snapshot of the batched decode KV cache (host vectors), suitable for
    // replication and re-import via start_decode_*_batched with a different
    // batch size — the runtime rebuilds its decode graphs for the new batch.
    runtime::TransformerBatchedKVState export_batched_decode_state() const;

    int64_t decode_cache_steps() const noexcept;
    int64_t decode_current_end() const noexcept;
    int64_t decode_valid_steps() const noexcept;
    void release_runtime_graphs();

    // [DIAG] 每 batch 行的当前解码结束位置（member_ends_ 拷贝；未启动则空）。
    std::vector<int64_t> batched_member_ends() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules
