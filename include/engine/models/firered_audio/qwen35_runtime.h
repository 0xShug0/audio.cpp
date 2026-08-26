#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/firered_audio/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::firered_audio {

struct FireRedAudioBackboneForwardResult {
    std::vector<float> hidden;
    int64_t steps = 0;
};

class FireRedAudioQwen35Runtime {
public:
    FireRedAudioQwen35Runtime(
        std::shared_ptr<const FireRedAudioAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~FireRedAudioQwen35Runtime();

    FireRedAudioQwen35Runtime(const FireRedAudioQwen35Runtime &) = delete;
    FireRedAudioQwen35Runtime & operator=(const FireRedAudioQwen35Runtime &) = delete;

    std::vector<float> token_embedding(const std::vector<int32_t> & token_ids);
    FireRedAudioBackboneForwardResult forward_embeddings(const std::vector<float> & embeddings, int64_t steps);
    std::vector<float> lm_head(const std::vector<float> & hidden);
    void release_graphs();

    class DecodeSession {
    public:
        virtual ~DecodeSession() = default;
        virtual void reset() = 0;
        virtual FireRedAudioBackboneForwardResult prefill_embeddings(
            const std::vector<float> & embeddings,
            int64_t steps) = 0;
        virtual std::vector<float> run_embedding_step(const std::vector<float> & embedding) = 0;
        virtual int64_t valid_steps() const noexcept = 0;
    };

    std::unique_ptr<DecodeSession> create_decode_session(int64_t cache_steps);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::firered_audio
