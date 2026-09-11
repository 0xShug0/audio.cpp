#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/irodori_tts/assets.h"
#include "engine/models/irodori_tts/condition_encoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::irodori_tts {

struct IrodoriRfSampleRequest {
  const IrodoriConditionOutput *conditions = nullptr;
  const std::vector<uint8_t> *text_mask = nullptr;
  IrodoriCaptionCondition caption;
  IrodoriSpeakerCondition speaker;
  IrodoriGenerationOptions generation;
};

struct IrodoriRfSampleResult {
  std::vector<float> latent;
  int64_t latent_steps = 0;
  int64_t target_samples = 0;
};

struct IrodoriRfSampleTiming {
  double context_cond_ms = 0.0;
  double context_cfg_ms = 0.0;
  double step_cond_ms = 0.0;
  double step_cfg_ms = 0.0;
};

class IrodoriRfSampler {
public:
  IrodoriRfSampler(std::shared_ptr<const IrodoriTTSAssets> assets,
                   core::ExecutionContext &execution_context,
                   size_t graph_arena_bytes, size_t weight_context_bytes,
                   assets::TensorStorageType weight_storage_type,
                   bool mem_saver);
  ~IrodoriRfSampler();

  IrodoriRfSampleResult sample(
      const IrodoriRfSampleRequest &request,
      IrodoriRfSampleTiming *timing = nullptr);
  void release_graphs();
  int64_t context_graph_rebuilds() const noexcept;
  int64_t step_graph_rebuilds() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::irodori_tts
