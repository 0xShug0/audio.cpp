#pragma once

#include "engine/community_models/sense_asr/assets.h"
#include "engine/community_models/sense_asr/encoder.h"
#include "engine/community_models/sense_asr/frontend.h"
#include "engine/community_models/sense_asr/types.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::runtime {
class ILoadedVoiceModel;
}

namespace engine::models::silero_vad {
class SileroVADLoadedModel;
}

namespace engine::community_models::sense_asr {

std::shared_ptr<runtime::IVoiceModelLoader> make_sense_asr_loader();

class SenseAsrSession final : public runtime::RuntimeSessionBase,
                              public runtime::IOfflineVoiceTaskSession,
                              public runtime::IStreamingVoiceTaskSession {
public:
  SenseAsrSession(
      runtime::TaskSpec task, runtime::SessionOptions options,
      std::shared_ptr<const SenseAsrAssets> assets,
      std::shared_ptr<const engine::model_spec::ModelContract> contract);
  ~SenseAsrSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
  runtime::StreamingPolicy streaming_policy() const override;
  void start_stream(const runtime::TaskRequest &request) override;
  void set_stream_event_sink(runtime::StreamEventCallback sink) override;
  void reset() override;
  runtime::StreamEvent
  process_audio_chunk(const runtime::AudioChunk &chunk) override;
  runtime::TaskResult finish_stream() override;
  runtime::TaskResult finalize() override;

private:
  struct AudioChunkPlan {
    runtime::TimeSpan source_span;
  };

  struct AsrRequest {
    runtime::AudioBuffer audio;
    SenseAsrTranscriptionOptions transcription;
  };

  AsrRequest make_request(const runtime::TaskRequest &request) const;
  std::vector<AudioChunkPlan>
  audio_chunk_plan(const runtime::TaskRequest &request);
  runtime::IOfflineVoiceTaskSession &vad_session();
  runtime::TaskResult run_single(const AsrRequest &request);
  runtime::StreamEvent process_available_stream_chunks(bool final);
  runtime::StreamEvent
  process_one_stream_chunk(const runtime::AudioBuffer &audio);

  runtime::TaskSpec task_;
  std::shared_ptr<const SenseAsrAssets> assets_;
  std::shared_ptr<const engine::model_spec::ModelContract> contract_;
  size_t encoder_graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
  assets::TensorStorageType weight_storage_type_ =
      assets::TensorStorageType::Native;
  SenseAsrFrontend frontend_;
  SenseAsrEncoderRuntime encoder_;
  std::filesystem::path vad_model_path_;
  std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
  std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
  runtime::TaskRequest streaming_request_;
  runtime::TaskResult streaming_result_;
  runtime::AudioBuffer streaming_audio_;
  size_t streaming_audio_offset_values_ = 0;
  std::string streaming_text_;
  size_t streaming_published_bytes_ = 0;
  int64_t streaming_windows_processed_ = 0;
  runtime::StreamEventCallback stream_event_sink_;
  bool stream_started_ = false;
  std::chrono::steady_clock::time_point stream_wall_start_{};
};

} // namespace engine::community_models::sense_asr
