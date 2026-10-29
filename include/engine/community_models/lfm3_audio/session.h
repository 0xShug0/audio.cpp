#pragma once

// LFM3-Audio sessions. ASR: audio -> log-mel -> encoder -> backbone, greedy
// text; with the chat request option the spoken input is answered instead of
// transcribed. TTS: text -> backbone -> MaskGIT depth decoder, one audio
// frame per step, -> codec -> 24 kHz audio. S2S: a spoken question in, an
// interleaved text and audio reply out, after the earlier turns the request
// carries (chat.h).

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm3_audio_loader();

// The ASR max_tokens default for a pass of `seconds`: 8 tokens per second, at least 512 (long single passes need
// about 4 per second); chat 512, or 2048 when thinking, which takes tens to over a thousand tokens.
int64_t lfm3_asr_default_max_tokens(double seconds, bool chat, bool thinking);

// A thinking chat's answer: the tokens after the FIRST </think> (upstream: text.partition("</think>")[2]); nullopt without one.
std::optional<std::vector<int32_t>> lfm3_thinking_answer(const std::vector<int32_t> & tokens, int32_t think_end);

// Appends a chunk's trimmed transcript to `text`, after a space unless Chinese or Japanese text meets at the join
// or the chunk starts with ASCII punctuation.
void append_lfm3_chunk_text(std::string & text, std::string chunk);

class Lfm3AsrSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Lfm3AsrSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm3AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm3AsrSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    // Whether the last run() cut a chunk's text off at max_tokens.
    [[nodiscard]] bool reached_max_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Lfm3TtsSession final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession,
                             public runtime::IStreamingVoiceTaskSession {
public:
    Lfm3TtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm3AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm3TtsSession() override;

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

    // Whether max_tokens cut a text chunk's speech off in the last run(), or in the stream since start_stream().
    [[nodiscard]] bool reached_max_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Lfm3ChatSession final : public runtime::RuntimeSessionBase,
                              public runtime::IOfflineVoiceTaskSession,
                              public runtime::IStreamingVoiceTaskSession {
public:
    Lfm3ChatSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Lfm3AudioAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Lfm3ChatSession() override;

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

    // Whether max_tokens cut the reply off, in the last run() or in the stream since start_stream().
    [[nodiscard]] bool reached_max_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
