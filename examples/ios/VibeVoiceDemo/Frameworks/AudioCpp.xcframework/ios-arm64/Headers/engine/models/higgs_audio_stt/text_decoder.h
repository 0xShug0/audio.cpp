#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/higgs_audio_stt/assets.h"
#include "engine/models/higgs_audio_stt/types.h"

#include <cstddef>
#include <functional>
#include <memory>

namespace engine::models::higgs_audio_stt {

using HiggsAudioSTTTokenCallback = std::function<void(const HiggsAudioSTTGeneratedTokens &)>;

class HiggsAudioSTTTextDecoderRuntime {
public:
    struct Impl;

    HiggsAudioSTTTextDecoderRuntime(
        std::shared_ptr<const HiggsAudioSTTAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~HiggsAudioSTTTextDecoderRuntime();

    HiggsAudioSTTGeneratedTokens generate(
        const HiggsAudioSTTPrompt & prompt,
        const HiggsAudioSTTAudioEmbeddings & audio_embeddings,
        const HiggsAudioSTTGenerationOptions & options,
        const HiggsAudioSTTTokenCallback & token_callback = {});

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_audio_stt
