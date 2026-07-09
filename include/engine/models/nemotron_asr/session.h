#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/nemotron_asr/assets.h"
#include "engine/models/nemotron_asr/decoder.h"
#include "engine/models/nemotron_asr/encoder.h"
#include "engine/models/nemotron_asr/frontend.h"
#include "engine/models/nemotron_asr/weights.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::models::nemotron_asr {

class NemotronASRSessionBase : public runtime::RuntimeSessionBase {
public:
    NemotronASRSessionBase(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const NemotronAssets> assets);
    ~NemotronASRSessionBase() override;

protected:
    std::string family_impl() const;
    runtime::VoiceTaskKind task_kind_impl() const;
    runtime::RunMode run_mode_impl() const;

    int64_t prompt_id_for_request(const runtime::TaskRequest & request) const;
    int64_t lookahead_for_options(const std::unordered_map<std::string, std::string> & options) const;
    NemotronDecodeOptions decode_options_for_request(const runtime::TaskRequest & request) const;
    NemotronDecodedText run_streaming_audio(
        const runtime::AudioBuffer & audio,
        int64_t prompt_id,
        int64_t lookahead,
        const NemotronDecodeOptions & decode_options,
        const NemotronTextDeltaCallback & on_text_delta = nullptr);

    runtime::TaskSpec task_;
    std::shared_ptr<const NemotronAssets> assets_;
    std::shared_ptr<const NemotronWeights> weights_;
    size_t weight_context_bytes_ = 3072ull * 1024ull * 1024ull;
    size_t encoder_graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t decoder_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    bool mem_saver_ = false;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    NemotronFrontend frontend_;
    std::unique_ptr<NemotronEncoderRuntime> encoder_;
    std::unique_ptr<NemotronDecoderRuntime> decoder_;
    runtime::AudioBuffer streaming_audio_;
    std::string streaming_language_;
    std::unordered_map<std::string, std::string> streaming_options_;
};

class NemotronASROfflineSession final
    : public NemotronASRSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    NemotronASROfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const NemotronAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
};

class NemotronASRStreamingSession final
    : public NemotronASRSessionBase
    , public runtime::IStreamingVoiceTaskSession {
public:
    NemotronASRStreamingSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const NemotronAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

private:
    runtime::StreamEventCallback stream_event_sink_;
};

}  // namespace engine::models::nemotron_asr
