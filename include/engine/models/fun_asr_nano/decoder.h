#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/types.h"

#include <cstddef>
#include <functional>
#include <memory>

namespace engine::models::fun_asr_nano {

using FunAsrNanoTokenCallback =
    std::function<void(const FunAsrNanoGeneratedTokens &)>;

class FunAsrNanoDecoderRuntime {
public:
  struct Impl;

  FunAsrNanoDecoderRuntime(std::shared_ptr<const FunAsrNanoAssets> assets,
                           core::ExecutionContext &execution,
                           size_t prefill_graph_arena_bytes,
                           size_t decode_graph_arena_bytes,
                           size_t weight_context_bytes,
                           assets::TensorStorageType weight_storage_type);
  ~FunAsrNanoDecoderRuntime();

  FunAsrNanoGeneratedTokens
  generate(const FunAsrNanoPrompt &prompt,
           const FunAsrNanoAdaptorEmbeddings &audio_embeddings,
           const FunAsrNanoGenerationOptions &options,
           const FunAsrNanoTokenCallback &token_callback = {});

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::fun_asr_nano
