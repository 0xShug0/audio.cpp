#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/models/nemotron_asr/assets.h"
#include "engine/models/nemotron_asr/decoder.h"
#include "engine/models/nemotron_asr/encoder.h"
#include "engine/models/nemotron_asr/frontend.h"
#include "engine/models/nemotron_asr/masked_asr.h"
#include "engine/models/nemotron_asr/speaker_tagging.h"
#include "engine/models/nemotron_asr/weights.h"

#include <chrono>
#include <optional>
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
        std::shared_ptr<const NemotronASRAssets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract);
    ~NemotronASRSessionBase() override;

protected:
    std::string family_impl() const;
    runtime::VoiceTaskKind task_kind_impl() const;
    runtime::RunMode run_mode_impl() const;

    int64_t prompt_id_for_request(const runtime::TaskRequest & request) const;
    int64_t lookahead_for_options(const std::unordered_map<std::string, std::string> & options) const;
    NemotronDecodeOptions decode_options_for_request(const runtime::TaskRequest & request) const;
    void validate_request_options(const std::unordered_map<std::string, std::string> & options) const;
    std::optional<SpeakerTaggingOptions> speaker_tagging_for_options(
        const std::unordered_map<std::string, std::string> & options) const;
    // Masked mode picks its lookahead from the diarizer file (plan D6) and warns
    // when the masks cannot be NeMo-exact.
    int64_t masked_lookahead(
        const std::unordered_map<std::string, std::string> & options,
        const SpeakerProbabilities & probabilities) const;
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::shared_ptr<const NemotronASRAssets> assets_;
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
        std::shared_ptr<const NemotronASRAssets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract);

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
        std::shared_ptr<const NemotronASRAssets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract);

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
    runtime::StreamEvent process_available_chunks(bool flush_tail);
    void process_feature_chunk(const NemotronFrontendFeatures & features);
    runtime::StreamEvent publish_stream_update();
    // Attribution mode: attribute the words that can no longer grow.
    std::vector<runtime::SpeakerTurn> attribute_stream_words(bool final);

    runtime::StreamEventCallback stream_event_sink_;
    runtime::PartialTextPublisher partials_;
    NemotronEncoderStreamState encoder_stream_state_;
    NemotronDecoderStreamState decoder_stream_state_;
    std::vector<float> streaming_waveform_;
    int64_t streaming_waveform_base_ = 0;
    int64_t received_samples_ = 0;
    int64_t next_chunk_start_ = 0;
    int64_t prompt_id_ = 0;
    int64_t lookahead_ = 0;
    int64_t chunks_processed_ = 0;
    bool first_chunk_processed_ = false;
    bool stream_started_ = false;
    bool finalized_ = false;
    std::chrono::steady_clock::time_point stream_wall_start_{};
    std::optional<SpeakerTaggingOptions> stream_tagging_;
    std::optional<SpeakerProbabilities> stream_speaker_probabilities_;
    std::unique_ptr<SpeakerSegmentBuilder> stream_segments_;
    std::unique_ptr<MaskedSpeakerStreams> masked_streams_;
    size_t stream_attributed_words_ = 0;
};

}  // namespace engine::models::nemotron_asr
