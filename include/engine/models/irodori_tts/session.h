#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/irodori_tts/assets.h"
#include "engine/models/irodori_tts/tokenizer_text.h"
#include "engine/models/irodori_tts/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::irodori_tts {

class IrodoriCodec;
class IrodoriConditionEncoder;
class IrodoriRfSampler;

class IrodoriTTSSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession {
public:
  IrodoriTTSSession(runtime::TaskSpec task, runtime::SessionOptions options,
                    std::shared_ptr<const IrodoriAssets> assets);
  ~IrodoriTTSSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

private:
  IrodoriRequest make_request(const runtime::TaskRequest &request) const;

  runtime::TaskSpec task_;
  std::shared_ptr<const IrodoriAssets> assets_;
  IrodoriTextTokenizer tokenizer_;
  size_t condition_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
  size_t rf_graph_arena_bytes_ = 768ull * 1024ull * 1024ull;
  size_t codec_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
  size_t condition_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
  size_t rf_weight_context_bytes_ = 768ull * 1024ull * 1024ull;
  size_t codec_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
  assets::TensorStorageType weight_storage_type_ =
      assets::TensorStorageType::Native;
  assets::TensorStorageType codec_weight_storage_type_ =
      assets::TensorStorageType::Native;
  std::unique_ptr<IrodoriConditionEncoder> condition_encoder_;
  std::unique_ptr<IrodoriRfSampler> rf_sampler_;
  std::unique_ptr<IrodoriCodec> codec_;
  uint64_t cached_reference_key_ = 0;
  int cached_reference_sample_rate_ = 0;
  int cached_reference_channels_ = 0;
  size_t cached_reference_samples_ = 0;
  std::vector<float> cached_reference_speaker_state_;
  std::vector<uint8_t> cached_reference_speaker_mask_;
  int64_t cached_reference_speaker_tokens_ = 0;
  bool cached_reference_speaker_has_speaker_ = false;
  bool cached_reference_valid_ = false;
};

} // namespace engine::models::irodori_tts
