#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/fun_asr_nano/adaptor.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/decoder.h"
#include "engine/models/fun_asr_nano/encoder.h"
#include "engine/models/fun_asr_nano/frontend.h"
#include "engine/models/fun_asr_nano/prompt.h"
#include "engine/models/fun_asr_nano/tokenizer_text.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::fun_asr_nano {

class FunAsrNanoSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession {
public:
  FunAsrNanoSession(runtime::TaskSpec task, runtime::SessionOptions options,
                    std::shared_ptr<const FunAsrNanoAssets> assets);
  ~FunAsrNanoSession() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

private:
  struct AudioChunkPlan {
    runtime::TimeSpan source_span;
  };

  struct AsrRequest {
    runtime::AudioBuffer audio;
    FunAsrNanoPromptRequest prompt;
    FunAsrNanoGenerationOptions generation;
  };

  AsrRequest make_request(const runtime::TaskRequest &request) const;
  std::vector<AudioChunkPlan>
  audio_chunk_plan(const runtime::TaskRequest &request) const;
  runtime::TaskResult run_single(const AsrRequest &request);

  runtime::TaskSpec task_;
  std::shared_ptr<const FunAsrNanoAssets> assets_;
  size_t encoder_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
  size_t adaptor_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
  size_t decoder_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
  size_t decoder_decode_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
  size_t decoder_weight_context_bytes_ = 32ull * 1024ull * 1024ull;
  assets::TensorStorageType encoder_weight_storage_type_ =
      assets::TensorStorageType::Native;
  assets::TensorStorageType adaptor_weight_storage_type_ =
      assets::TensorStorageType::Native;
  assets::TensorStorageType decoder_weight_storage_type_ =
      assets::TensorStorageType::Native;
  FunAsrNanoTextTokenizer tokenizer_;
  FunAsrNanoFrontend frontend_;
  FunAsrNanoEncoderRuntime encoder_;
  FunAsrNanoAdaptorRuntime adaptor_;
  FunAsrNanoPromptBuilder prompt_builder_;
  FunAsrNanoDecoderRuntime decoder_;
};

} // namespace engine::models::fun_asr_nano
