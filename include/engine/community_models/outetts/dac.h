#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/outetts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::outetts {

class OuteTTSDacDecoder final {
public:
  struct EncodedReference {
    std::vector<float> samples;
    std::vector<int32_t> codebook1;
    std::vector<int32_t> codebook2;
  };

  OuteTTSDacDecoder(std::shared_ptr<const OuteTTSAssets> assets,
                    core::ExecutionContext &execution_context,
                    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull,
                    size_t graph_context_bytes = 1536ull * 1024ull * 1024ull,
                    assets::TensorStorageType weight_storage_type =
                        assets::TensorStorageType::F32);
  ~OuteTTSDacDecoder();

  runtime::AudioBuffer decode(const std::vector<int32_t> &codebook1,
                              const std::vector<int32_t> &codebook2);

  EncodedReference encode_reference(const runtime::AudioBuffer &audio);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::outetts
