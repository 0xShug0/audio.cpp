#pragma once

// LFM2.5-Audio sessions. ASR: audio -> FastConformer -> adapter -> LFM2
// backbone, prompted with the ASR system prompt from liquid-audio's README
// and decoded greedily like LFM2AudioModel.generate_sequential
// (model/lfm2_audio.py). TTS: text -> backbone -> depthformer, one audio
// frame per step, -> detokenizer -> 24 kHz audio. S2S: audio in as for ASR,
// a reply of text and audio out as for TTS, interleaved, after the earlier
// turns of the conversation that the request carries (chat.h).

#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/chat.h"
#include "engine/community_models/lfm2_audio/depthformer.h"
#include "engine/community_models/lfm2_audio/detokenizer.h"
#include "engine/community_models/lfm2_audio/interleaved.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/community_models/lfm2_audio/tts.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm2_audio_loader();

class Lfm2AudioSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Lfm2AudioSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm2AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm2AudioSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    // Whether the last run() cut a chunk's transcript off at max_tokens.
    [[nodiscard]] bool reached_max_tokens() const;

private:
    struct RequestOptions {
        int64_t max_tokens = 512;
    };

    RequestOptions parse_request_options(const runtime::TaskRequest & request) const;
    std::vector<runtime::TimeSpan> plan_chunks(const runtime::TaskRequest & request, const std::vector<float> & samples);
    runtime::IOfflineVoiceTaskSession & vad_session();
    std::string transcribe(const std::vector<float> & samples, const runtime::TimeSpan & span, const RequestOptions & options);

    runtime::TaskSpec task_;
    std::shared_ptr<const Lfm2AudioAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    bool cpu_repack_;  // lfm2_audio.cpu_repack, for the runtimes below
    std::shared_ptr<const Lfm2AudioComponents> components_;
    Lfm2TextTokenizer tokenizer_;
    Lfm2AudioFeatureExtractor features_;
    Lfm2FastConformerEncoderRuntime encoder_;
    Lfm2BackboneRuntime backbone_;
    std::string language_;
    Lfm2AsrPrompt prompt_;
    std::filesystem::path vad_model_path_;
    double max_pass_seconds_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
    bool reached_max_tokens_ = false;
};

// Streaming TTS pulls events: each carries the audio of the next
// stream_frames_per_event frames, decoded as they come, the way liquid-audio's
// demo decodes each frame; together they are the offline speech.
class Lfm2AudioTtsSession final : public runtime::RuntimeSessionBase,
                                  public runtime::IOfflineVoiceTaskSession,
                                  public runtime::IStreamingVoiceTaskSession {
public:
    Lfm2AudioTtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm2AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm2AudioTtsSession() override;

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

    // Whether max_tokens cut a text chunk's speech off in the last run(), or
    // in the stream since start_stream().
    [[nodiscard]] bool reached_max_tokens() const;

private:
    struct RequestOptions {
        std::string system_prompt;
        Lfm2SpeechOptions speech;
        int64_t text_chunk_size = 0;
        text::TextChunkMode text_chunk_mode = text::TextChunkMode::Default;
        int64_t stream_frames_per_event = 0;
        uint64_t seed = 0;
        std::vector<std::string> texts;  // the text chunks, one turn each
    };

    struct Stream;

    RequestOptions parse_request(const runtime::TaskRequest & request) const;
    std::unique_ptr<Lfm2SpeechGenerator> start_turn(const RequestOptions & options, size_t turn);
    std::vector<float> speak(const RequestOptions & options, size_t turn);

    runtime::TaskSpec task_;
    std::shared_ptr<const Lfm2AudioAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    bool cpu_repack_;  // lfm2_audio.cpu_repack, for the runtimes below
    std::shared_ptr<const Lfm2AudioComponents> components_;
    std::shared_ptr<const Lfm2AudioOutputComponents> output_;
    Lfm2TextTokenizer tokenizer_;
    Lfm2BackboneRuntime backbone_;
    Lfm2DepthformerRuntime depthformer_;
    Lfm2DetokenizerRuntime detokenizer_;
    std::string language_;
    std::unique_ptr<Stream> stream_;
    bool reached_max_tokens_ = false;
};

// Speech-to-speech (s2s): a spoken user turn in, a reply of interleaved text
// and audio out (LFM2AudioModel.generate_interleaved) under the system prompt
// of liquid-audio's chat demo, which text_input replaces when given. A
// request's input artifacts carry the conversation's earlier turns, which
// the prompt replays in full as liquid-audio does, and with return_codes its
// result carries the reply to send back with the next turn. What a request
// gives depends on that request alone; to save time on the next turn, for
// each of up to lfm2_audio.conversation_cache_slots conversations (four by
// default), the session keeps the questions of its last request with history
// or return_codes, with their encoder output, and the backbone the full
// prefill blocks of its last turn with history. Streaming takes the user's
// audio in chunks and, once it has all come, pulls the reply as events: the
// audio of the next stream_frames_per_event frames and the text written since
// the last event.
class Lfm2AudioChatSession final : public runtime::RuntimeSessionBase,
                                   public runtime::IOfflineVoiceTaskSession,
                                   public runtime::IStreamingVoiceTaskSession {
public:
    Lfm2AudioChatSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm2AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm2AudioChatSession() override;

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

    // Whether max_tokens cut the reply off, in the last run() or in the
    // stream since start_stream().
    [[nodiscard]] bool reached_max_tokens() const;

    // Of the last reply's prompt: the steps the backbone restored from the
    // turns before rather than ran (Lfm2BackboneRuntime::resumed_prefill_steps),
    // and the earlier questions that took the encoder output kept from them.
    [[nodiscard]] int64_t resumed_prefill_steps() const;
    [[nodiscard]] int64_t reused_questions() const;

private:
    struct RequestOptions {
        std::string system_prompt;
        Lfm2InterleavedOptions reply;
        int64_t stream_frames_per_event = 0;
        bool return_codes = false;
        std::vector<Lfm2ConversationTurn> history;
        int64_t text_steps = 0;  // with history: prompt steps besides the questions' audio
    };

    // A question as the encoder took it, mono 16 kHz, and what it gave. Neither
    // changes once encoded, so a request that takes a kept question shares
    // both with the table before rather than copying them.
    struct EncodedQuestion {
        std::shared_ptr<const std::vector<float>> samples;
        std::shared_ptr<const Lfm2AudioEmbeddings> encoded;
    };

    struct Stream;

    RequestOptions parse_request(const runtime::TaskRequest & request) const;
    std::unique_ptr<Lfm2InterleavedGenerator> start_reply(const RequestOptions & options, const runtime::AudioBuffer & audio);
    void keep_questions(std::vector<EncodedQuestion> questions);
    [[nodiscard]] Lfm2ReplyCheckpoint reply_checkpoint() const;

    runtime::TaskSpec task_;
    std::shared_ptr<const Lfm2AudioAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    bool cpu_repack_;  // lfm2_audio.cpu_repack, for the runtimes below
    // lfm2_audio.conversation_cache_slots, for the backbone below and questions_
    size_t conversation_slots_;
    std::shared_ptr<const Lfm2AudioComponents> components_;
    std::shared_ptr<const Lfm2AudioOutputComponents> output_;
    Lfm2TextTokenizer tokenizer_;
    Lfm2AudioFeatureExtractor features_;
    Lfm2FastConformerEncoderRuntime encoder_;
    Lfm2BackboneRuntime backbone_;
    Lfm2DepthformerRuntime depthformer_;
    Lfm2DetokenizerRuntime detokenizer_;
    std::string language_;
    double max_pass_seconds_;
    std::unique_ptr<Stream> stream_;
    bool reached_max_tokens_ = false;
    // The questions of the last request with history or return_codes of each
    // of up to conversation_slots_ conversations, the most recent first.
    std::vector<std::vector<EncodedQuestion>> questions_;
    int64_t reused_questions_ = 0;
};

}  // namespace engine::community_models::lfm2_audio
