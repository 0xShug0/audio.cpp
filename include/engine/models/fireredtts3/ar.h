#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/models/fireredtts3/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::fireredtts3 {

class FireRedArRuntime {
public:
    FireRedArRuntime(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        bool instruct);
    ~FireRedArRuntime();

    FireRedArRuntime(const FireRedArRuntime &) = delete;
    FireRedArRuntime & operator=(const FireRedArRuntime &) = delete;

    std::vector<float> token_embedding(const std::vector<int32_t> & token_ids);
    std::vector<float> speaker_llm(const std::vector<float> & speaker);
    std::vector<float> speaker_dit(const std::vector<float> & speaker);
    std::vector<float> patch_encode(const std::vector<float> & latents);
    std::vector<float> dit_head(const std::vector<float> & hidden, int64_t rows);
    float stop(const std::vector<float> & hidden);
    std::vector<float> text_logits(const std::vector<float> & hidden);

    engine::modules::QwenCausalPrefillResult prefill_embeddings(const std::vector<float> & embeddings, int64_t steps);
    engine::modules::QwenCausalPrefillResult prefill_embeddings_padded(
        const std::vector<float> & embeddings, int64_t padded_steps, int64_t valid_steps);
    void start_decode_embeddings(const engine::runtime::TransformerKVState & state, int64_t required_cache_steps);
    engine::modules::QwenCausalDecodeStepResult decode_embedding(const std::vector<float> & embedding);

    // --- batch（多 slot 并发推理）passthroughs ---
    void start_decode_embeddings_batched(
        const engine::runtime::TransformerBatchedKVState & state, int64_t required_cache_steps);
    engine::modules::QwenCausalDecodeStepResult decode_embeddings_batched(
        const std::vector<float> & embeddings, int64_t batch_size,
        const std::vector<uint8_t> & active_mask = {});
    engine::runtime::TransformerBatchedKVState export_batched_decode_state() const;
    // 冻结/重置某 batch 行的解码位置：非活跃行应保持 end=0（mask 全 -inf，不参与 attention），
    // 避免 run_batched_decode_step 对空行 advance_member 导致其位置递增、mask 污染活跃行。
    void set_batched_member_end(int64_t batch, int64_t end);
    // [DIAG] 当前 batched decode 各行的解码结束位置（未启动则空）。
    std::vector<int64_t> batched_member_ends() const;

    void release_graphs();
    void release_backbone_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fireredtts3
