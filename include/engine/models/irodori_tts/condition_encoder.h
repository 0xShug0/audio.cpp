#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/irodori_tts/assets.h"
#include "engine/models/irodori_tts/types.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::irodori_tts {

struct IrodoriSpeakerCondition {
  std::vector<float> state;
  std::vector<uint8_t> mask;
  int64_t tokens = 0;
  bool has_speaker = false;
};

struct IrodoriCaptionCondition {
  std::vector<int32_t> token_ids;
  std::vector<uint8_t> mask;
  bool has_caption = false;
};

struct IrodoriConditionOutput {
  std::vector<float> text_state;
  std::vector<float> caption_state;
  float predicted_log_frames = 0.0F;
};

struct IrodoriSelfAttentionWeights {
  modules::LinearWeights wq;
  modules::LinearWeights wk;
  modules::LinearWeights wv;
  modules::LinearWeights wo;
  modules::LinearWeights gate;
  core::TensorValue q_norm;
  core::TensorValue k_norm;
};

struct IrodoriTextBlockWeights {
  core::TensorValue attention_norm;
  IrodoriSelfAttentionWeights attention;
  core::TensorValue mlp_norm;
  modules::LinearWeights mlp_w1;
  modules::LinearWeights mlp_w2;
  modules::LinearWeights mlp_w3;
};

struct IrodoriDurationBlockWeights {
  core::TensorValue norm;
  modules::LinearWeights mlp_w1;
  modules::LinearWeights mlp_w2;
  modules::LinearWeights mlp_w3;
  modules::LinearWeights modulation;
  modules::LinearWeights caption_modulation;
};

struct IrodoriDurationWeights {
  core::TensorValue null_speaker;
  core::TensorValue null_caption;
  modules::LinearWeights token_input_proj;
  std::vector<IrodoriDurationBlockWeights> token_blocks;
  core::TensorValue token_out_norm;
  modules::LinearWeights token_out_proj;
};

struct IrodoriConditionEncoderWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  core::TensorValue text_embedding;
  std::vector<IrodoriTextBlockWeights> text_blocks;
  core::TensorValue text_norm;
  modules::LinearWeights speaker_in_proj;
  std::vector<IrodoriTextBlockWeights> speaker_blocks;
  core::TensorValue speaker_norm;
  core::TensorValue caption_embedding;
  std::vector<IrodoriTextBlockWeights> caption_blocks;
  core::TensorValue caption_norm;
  IrodoriDurationWeights duration;
};

struct IrodoriEncodedConditions {
  std::vector<float> text_state;
  std::vector<int32_t> text_mask;
  std::vector<float> speaker_state;
  std::vector<int32_t> speaker_mask;
  std::vector<float> caption_state;
  std::vector<int32_t> caption_mask;
  int64_t text_tokens = 0;
  int64_t text_dim = 0;
  int64_t speaker_tokens = 0;
  int64_t speaker_dim = 0;
  int64_t caption_tokens = 0;
  int64_t caption_dim = 0;
  float predicted_log_frames = 0.0F;
};

IrodoriConditionEncoderWeights load_irodori_condition_encoder_weights(
    const IrodoriAssets &assets, ggml_backend_t backend,
    core::BackendType backend_type, size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

core::TensorValue build_irodori_text_encoder(
    core::ModuleBuildContext &ctx, const core::TensorValue &input_ids,
    const core::TensorValue &text_mask, const core::TensorValue &attention_mask,
    const core::TensorValue &positions,
    const IrodoriConditionEncoderWeights &weights,
    const IrodoriModelConfig &config);

core::TensorValue build_irodori_reference_latent_encoder(
    core::ModuleBuildContext &ctx, const core::TensorValue &ref_latent,
    const core::TensorValue &ref_mask, const core::TensorValue &attention_mask,
    const core::TensorValue &positions,
    const IrodoriConditionEncoderWeights &weights,
    const IrodoriModelConfig &config);

core::TensorValue build_irodori_caption_encoder(
    core::ModuleBuildContext &ctx, const core::TensorValue &input_ids,
    const core::TensorValue &caption_mask,
    const core::TensorValue &attention_mask, const core::TensorValue &positions,
    const IrodoriConditionEncoderWeights &weights,
    const IrodoriModelConfig &config);

core::TensorValue build_irodori_duration_predictor(
    core::ModuleBuildContext &ctx, const core::TensorValue &text_state,
    const core::TensorValue &text_mask, const core::TensorValue &speaker_state,
    const core::TensorValue &has_speaker,
    const core::TensorValue &caption_state,
    const core::TensorValue &caption_mask, const core::TensorValue &has_caption,
    const IrodoriConditionEncoderWeights &weights,
    const IrodoriModelConfig &config);

class IrodoriConditionEncoder {
public:
  IrodoriConditionEncoder(std::shared_ptr<const IrodoriAssets> assets,
                          core::ExecutionContext &execution_context,
                          size_t graph_arena_bytes, size_t weight_context_bytes,
                          assets::TensorStorageType weight_storage_type);
  ~IrodoriConditionEncoder();

  IrodoriConditionOutput run(const std::vector<int32_t> &token_ids,
                             const std::vector<uint8_t> &token_mask,
                             const IrodoriCaptionCondition &caption,
                             const IrodoriSpeakerCondition &speaker);
  IrodoriSpeakerCondition
  encode_speaker_reference(const std::vector<float> &ref_latent,
                           int64_t ref_tokens);
  void release_graphs();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::irodori_tts
