#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/voxcpm1/assets.h"
#include "engine/community_models/voxcpm1/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::community_models::voxcpm1 {

struct VoxCPM1AudioVAEDecoderConfig {
  size_t weight_context_bytes = 768ull * 1024ull * 1024ull;
  size_t graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t encoder_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  int64_t latent_frame_capacity = 0;
  int64_t encoder_sample_capacity = 240000;
  engine::assets::TensorStorageType weight_storage_type =
      engine::assets::TensorStorageType::F32;
};

class VoxCPM1AudioVAEDecoderRuntime final {
public:
  VoxCPM1AudioVAEDecoderRuntime(
      std::shared_ptr<const VoxCPM1Assets> assets,
      engine::core::ExecutionContext &execution_context,
      VoxCPM1AudioVAEDecoderConfig config = {});
  ~VoxCPM1AudioVAEDecoderRuntime();

  runtime::AudioBuffer decode_features(const std::vector<float> &features,
                                       int64_t patches);
  VoxCPM1EncodedPrompt encode_prompt_audio(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);
  void release_runtime_memory();
  void release_encoder_graph();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::community_models::voxcpm1
