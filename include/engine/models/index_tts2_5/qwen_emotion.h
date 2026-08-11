#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/index_tts2_5/assets.h"

#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25QwenEmotionWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue token_embedding;
    engine::modules::QwenDecoderStackWeights decoder;
    engine::modules::NormWeights final_norm;
};

struct IndexTTS25EmotionVector {
    std::vector<float> values;
};

std::shared_ptr<const IndexTTS25QwenEmotionWeights> load_index_tts2_5_qwen_emotion_weights(
    const IndexTTS25Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType storage_type,
    size_t weight_context_bytes);

class IndexTTS25QwenEmotionTokenizer {
public:
    explicit IndexTTS25QwenEmotionTokenizer(std::shared_ptr<const IndexTTS25Assets> assets);

    std::vector<int32_t> encode_chat_prompt(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const;
    int32_t eos_token_id() const noexcept;
    int32_t think_end_token_id() const noexcept;

private:
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t eos_token_id_ = 151643;
    int32_t think_end_token_id_ = 151668;
};

class IndexTTS25QwenEmotionRuntime {
public:
    IndexTTS25QwenEmotionRuntime(
        std::shared_ptr<const IndexTTS25Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~IndexTTS25QwenEmotionRuntime();

    IndexTTS25QwenEmotionRuntime(const IndexTTS25QwenEmotionRuntime &) = delete;
    IndexTTS25QwenEmotionRuntime & operator=(const IndexTTS25QwenEmotionRuntime &) = delete;

    IndexTTS25EmotionVector infer(const std::string & text, int64_t max_new_tokens = 256);
    void release_graphs();

private:
    class PrefillGraph;
    class DecodeGraph;

    std::shared_ptr<const IndexTTS25Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t prefill_graph_arena_bytes_ = 0;
    size_t decode_graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS25QwenEmotionWeights> weights_;
    IndexTTS25QwenEmotionTokenizer tokenizer_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
};

}  // namespace engine::models::index_tts2_5
