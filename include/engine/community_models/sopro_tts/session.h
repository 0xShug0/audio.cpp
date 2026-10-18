#pragma once

#include "engine/community_models/sopro_tts/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
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
class SoproVocosRuntime;

// Everything one synthesis run carries between its text segments: the parsed
// options, the encoded reference voice, the LM carry-over prompt and the RNG.
// Offline builds one and drains it in a loop; streaming keeps it alive across
// next_stream_event calls.
struct SoproSynthesisState;
struct SoproPromptState;
struct SoproVoice;

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
    // A reference clip as the voice cache sees it: its 24 kHz samples and the
    // crop length, hashed together.
    struct VoiceKey {
        uint64_t sample_count = 0;
        uint64_t hash = 0;
        bool operator==(const VoiceKey & other) const noexcept {
            return sample_count == other.sample_count && hash == other.hash;
        }
    };

    SoproRequestOptions parse_options(const runtime::TaskRequest & request) const;
    // Validates the request, prepares the reference voice and splits the text.
    std::unique_ptr<SoproSynthesisState> begin_synthesis(const runtime::TaskRequest & request);
    // SoproTTS.prepare_reference, served from the voice cache when it holds the clip.
    std::shared_ptr<SoproVoice> prepare_voice(const std::vector<float> & audio24, float ref_seconds);
    // SoproTTS._prompt_state: the voice's streaming prompt for `steps`, solved
    // once and kept on the voice.
    const SoproPromptState & prompt_state(SoproVoice & voice, int64_t steps);
    // Runs one text segment through the LM, the acoustic head and the vocoder.
    // Returns the raw 24 kHz waveform before any levelling, or an empty vector
    // when the segment generated nothing.
    std::vector<float> synthesize_segment(SoproSynthesisState & state);
    // SoproTTS._stream_segment, split into the steps one pull event needs.
    void begin_stream_segment(SoproSynthesisState & state);
    std::optional<std::vector<float>> gate_stream_audio(SoproSynthesisState & state, std::vector<float> audio) const;
    std::vector<float> finish_stream_segment(SoproSynthesisState & state);

    runtime::TaskSpec task_;
    std::shared_ptr<const SoproTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::string default_language_;

    std::unique_ptr<SoproTextTokenizer> tokenizer_;
    std::unique_ptr<SoproSpeakerEncoderRuntime> speaker_encoder_;
    std::unique_ptr<SoproSemanticEncoderRuntime> semantic_encoder_;
    std::unique_ptr<SoproVocosRuntime> vocoder_;
    std::unique_ptr<SoproSemanticLMRuntime> semantic_lm_;
    std::unique_ptr<SoproAcousticRuntime> acoustic_;
    std::unique_ptr<SoproReferenceBuilder> reference_builder_;
    runtime::CacheSlots<VoiceKey, std::shared_ptr<SoproVoice>> voice_cache_;

    std::unique_ptr<SoproSynthesisState> stream_state_;
    std::vector<runtime::AudioBuffer> stream_chunks_;
};

}  // namespace engine::community_models::sopro_tts
