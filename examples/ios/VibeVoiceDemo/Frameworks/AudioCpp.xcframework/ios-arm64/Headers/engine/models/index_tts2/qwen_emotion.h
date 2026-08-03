#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/index_tts2/assets.h"

#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2QwenEmotionWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue token_embedding;
    engine::modules::QwenDecoderStackWeights decoder;
    engine::modules::NormWeights final_norm;
};

struct IndexTTS2EmotionVector {
    std::vector<float> values;
};

std::shared_ptr<const IndexTTS2QwenEmotionWeights> load_index_tts2_qwen_emotion_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType storage_type,
    size_t weight_context_bytes);

class IndexTTS2QwenEmotionTokenizer {
public:
    explicit IndexTTS2QwenEmotionTokenizer(std::shared_ptr<const IndexTTS2Assets> assets);

    std::vector<int32_t> encode_chat_prompt(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const;
    int32_t eos_token_id() const noexcept;
    int32_t think_end_token_id() const noexcept;

private:
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t eos_token_id_ = 151643;
    int32_t think_end_token_id_ = 151668;
};

class IndexTTS2QwenEmotionRuntime {
public:
    IndexTTS2QwenEmotionRuntime(
        std::shared_ptr<const IndexTTS2Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~IndexTTS2QwenEmotionRuntime();

    IndexTTS2QwenEmotionRuntime(const IndexTTS2QwenEmotionRuntime &) = delete;
    IndexTTS2QwenEmotionRuntime & operator=(const IndexTTS2QwenEmotionRuntime &) = delete;

    IndexTTS2EmotionVector infer(const std::string & text, int64_t max_new_tokens = 256);
    void release_graphs();

private:
    class PrefillGraph;
    class DecodeGraph;

    std::shared_ptr<const IndexTTS2Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t prefill_graph_arena_bytes_ = 0;
    size_t decode_graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights_;
    IndexTTS2QwenEmotionTokenizer tokenizer_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
};

}  // namespace engine::models::index_tts2
