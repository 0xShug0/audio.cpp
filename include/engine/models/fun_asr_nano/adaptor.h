#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::fun_asr_nano {

class FunAsrNanoAdaptorRuntime {
public:
  FunAsrNanoAdaptorRuntime(std::shared_ptr<const FunAsrNanoAssets> assets,
                           engine::core::ExecutionContext &execution_context,
                           size_t graph_arena_bytes,
                           engine::assets::TensorStorageType weight_storage =
                               engine::assets::TensorStorageType::F32);
  ~FunAsrNanoAdaptorRuntime();

  FunAsrNanoAdaptorRuntime(const FunAsrNanoAdaptorRuntime &) = delete;
  FunAsrNanoAdaptorRuntime &
  operator=(const FunAsrNanoAdaptorRuntime &) = delete;
  FunAsrNanoAdaptorRuntime(FunAsrNanoAdaptorRuntime &&) noexcept;
  FunAsrNanoAdaptorRuntime &operator=(FunAsrNanoAdaptorRuntime &&) noexcept;

  void prepare_capacity(int64_t valid_frames);
  FunAsrNanoAdaptorEmbeddings
  adapt(const FunAsrNanoEncoderEmbeddings &encoder_embeddings,
        const std::vector<int32_t> &mask, bool capture_stages = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::fun_asr_nano
