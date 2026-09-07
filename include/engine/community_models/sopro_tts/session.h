#pragma once

#include "engine/community_models/sopro_tts/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::sopro_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_sopro_tts_loader();

class SoproAcousticRuntime;
class SoproReferenceBuilder;
class SoproSemanticEncoderRuntime;
class SoproSemanticLMRuntime;
class SoproSpeakerEncoderRuntime;
class SoproTextTokenizer;
class SoproVocoderRuntime;

// Everything one synthesis run carries between its text segments: the parsed
// options, the encoded reference voice, the LM carry-over prompt and the RNG.
// Offline builds one and drains it in a loop; streaming keeps it alive across
// next_stream_event calls, so both paths consume the seed in the same order.
struct SoproSynthesisState;

class SoproTTSSession final : public runtime::RuntimeSessionBase,
                              public runtime::IOfflineVoiceTaskSession,
                              public runtime::IStreamingVoiceTaskSession {
public:
    SoproTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SoproTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SoproTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    SoproRequestOptions parse_options(const runtime::TaskRequest & request) const;
    // Validates the request, encodes the reference voice and splits the text.
    std::unique_ptr<SoproSynthesisState> begin_synthesis(const runtime::TaskRequest & request);
    // Runs one text segment through the LM, the acoustic head and the vocoder.
    // Returns the raw 24 kHz waveform before any levelling, or an empty vector
    // when the segment generated nothing.
    std::vector<float> synthesize_segment(SoproSynthesisState & state);

    runtime::TaskSpec task_;
    std::shared_ptr<const SoproTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::string default_language_;

    std::unique_ptr<SoproTextTokenizer> tokenizer_;
    std::unique_ptr<SoproSpeakerEncoderRuntime> speaker_encoder_;
    std::unique_ptr<SoproSemanticEncoderRuntime> semantic_encoder_;
    std::unique_ptr<SoproVocoderRuntime> vocoder_;
    std::unique_ptr<SoproSemanticLMRuntime> semantic_lm_;
    std::unique_ptr<SoproAcousticRuntime> acoustic_;
    std::unique_ptr<SoproReferenceBuilder> reference_builder_;

    std::unique_ptr<SoproSynthesisState> stream_state_;
    std::vector<runtime::AudioBuffer> stream_chunks_;
};

}  // namespace engine::community_models::sopro_tts
