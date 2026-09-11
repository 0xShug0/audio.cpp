#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace engine::models::fun_asr_nano {

class FunAsrNanoEncoderRuntime {
public:
  FunAsrNanoEncoderRuntime(std::shared_ptr<const FunAsrNanoAssets> assets,
                           engine::core::ExecutionContext &execution_context,
                           size_t graph_arena_bytes,
                           engine::assets::TensorStorageType weight_storage =
                               engine::assets::TensorStorageType::F32);
  ~FunAsrNanoEncoderRuntime();

  FunAsrNanoEncoderRuntime(const FunAsrNanoEncoderRuntime &) = delete;
  FunAsrNanoEncoderRuntime &
  operator=(const FunAsrNanoEncoderRuntime &) = delete;
  FunAsrNanoEncoderRuntime(FunAsrNanoEncoderRuntime &&) noexcept;
  FunAsrNanoEncoderRuntime &operator=(FunAsrNanoEncoderRuntime &&) noexcept;

  void prepare_capacity(int64_t frames);
  FunAsrNanoEncoderEmbeddings encode(const FunAsrNanoAudioFeatures &features,
                                     bool capture_stages = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::fun_asr_nano
