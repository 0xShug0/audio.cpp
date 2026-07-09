#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/voxcpm2/assets.h"
#include "engine/models/voxcpm2/audiovae.h"
#include "engine/models/voxcpm2/generator.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::voxcpm2 {

class VoxCPM2SessionBase : public runtime::RuntimeSessionBase {
public:
  VoxCPM2SessionBase(runtime::TaskSpec task, runtime::SessionOptions options,
                     std::shared_ptr<const VoxCPM2Assets> assets);
  ~VoxCPM2SessionBase() override;

protected:
  std::string family_impl() const;
  runtime::VoiceTaskKind task_kind_impl() const;
  runtime::RunMode run_mode_impl() const;
  void prepare_impl(const runtime::SessionPreparationRequest &request);

  struct EncodedPromptCacheEntry {
    std::string prompt_text;
    std::optional<runtime::AudioBuffer> prompt_audio;
    std::optional<runtime::AudioBuffer> reference_audio;
    VoxCPM2EncodedPrompt encoded;
  };

  VoxCPM2GenerationOptions
  generation_options_from_request(const runtime::TaskRequest &request) const;
  void validate_request(const runtime::TaskRequest &request) const;
  const VoxCPM2EncodedPrompt *encoded_prompt_for_request(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);

  runtime::TaskResult run_offline_request(const runtime::TaskRequest &request);
  runtime::TaskResult run_streaming_request(
      const runtime::TaskRequest &request,
      const runtime::StreamEventCallback &stream_event_sink = nullptr);
  void release_request_runtime_memory();

  runtime::TaskSpec task_;
  std::shared_ptr<const VoxCPM2Assets> assets_;
  VoxCPM2FeatureGeneratorConfig generator_config_;
  VoxCPM2AudioVAEDecoderConfig decoder_config_;
  std::unique_ptr<VoxCPM2FeatureGeneratorRuntime> generator_;
  std::unique_ptr<VoxCPM2AudioVAEDecoderRuntime> decoder_;
  std::optional<EncodedPromptCacheEntry> encoded_prompt_cache_;
};

class VoxCPM2OfflineSession final : public VoxCPM2SessionBase,
                                    public runtime::IOfflineVoiceTaskSession {
public:
  VoxCPM2OfflineSession(runtime::TaskSpec task, runtime::SessionOptions options,
                        std::shared_ptr<const VoxCPM2Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
};

class VoxCPM2StreamingSession final : public VoxCPM2SessionBase,
                                      public runtime::IStreamingVoiceTaskSession {
public:
  VoxCPM2StreamingSession(runtime::TaskSpec task, runtime::SessionOptions options,
                          std::shared_ptr<const VoxCPM2Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::StreamingPolicy streaming_policy() const override;
  void start_stream(const runtime::TaskRequest &request) override;
  std::optional<runtime::StreamEvent> next_stream_event() override;
  void set_stream_event_sink(runtime::StreamEventCallback sink) override;
  runtime::TaskResult finish_stream() override;
  void reset() override;
  runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &chunk) override;
  runtime::TaskResult finalize() override;

private:
  runtime::TaskResult result_;
  size_t next_chunk_index_ = 0;
  bool started_ = false;
  runtime::StreamEventCallback stream_event_sink_;
};

} // namespace engine::models::voxcpm2
