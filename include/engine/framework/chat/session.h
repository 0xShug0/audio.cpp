#pragma once

#include "engine/framework/chat/reply.h"
#include <chrono>

namespace engine::chat {

struct DuplexSource {
    int sample_rate = 0;
    int64_t frame_samples = 0;
    int64_t codebooks = 0;
    std::function<std::vector<int32_t>(const runtime::AudioBuffer &)> encode;
    std::function<std::vector<int32_t>(const runtime::AudioBuffer &, bool)> encode_stream;
    std::function<void()> reset_codec;
    std::function<bool(const int32_t *, int32_t *)> predict;
    std::function<runtime::AudioBuffer(const std::vector<int32_t> &, int64_t)> decode;
    std::function<runtime::AudioBuffer(const std::vector<int32_t> &, int64_t)> decode_stream;
};

// One conversation execution; reuse Session to create a fresh execution while
// retaining the model's graph caches. Owns input progression, pending
// codes and decoded output; the model supplies exactly one causal frame step.
class Duplex final {
public:
    explicit Duplex(DuplexSource source);
    ~Duplex();
    void begin(const std::optional<runtime::AudioBuffer> & complete_input);
    runtime::StreamEvent push(const runtime::AudioChunk & chunk);
    runtime::StreamEvent end_input();
    runtime::TaskResult finalize();
    runtime::TaskResult interrupt();
    runtime::TaskResult run(const runtime::AudioBuffer & input);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Exchange {
    runtime::TaskRequest user;
    runtime::TaskResult assistant;
    bool complete = false;
    std::vector<ReplyStep> steps;
};

// Prepared mathematical inputs for one reply. The model validates options and
// constructs its prompt here; the session collects the user's audio and drives
// the reply. A binding must not implement its own generation or event loop.
enum class HistoryMode { Conversation, Request };

struct UserTurn {
    double max_audio_seconds = 0;
    int output_sample_rate = 0;
    std::string language;
    std::function<std::unique_ptr<Reply>(const runtime::AudioBuffer &, const std::vector<ConversationTurn> &)> reply;
    std::function<std::unique_ptr<Duplex>()> duplex;
    std::optional<HistoryFormat> history_format;
    int64_t max_reply_steps = 0;
    bool return_codes = false;
    // Pure prompt mathematics: a lower bound on prompt steps from the validated
    // history. The owner checks it before accepting audio or running inference.
    std::function<int64_t(const std::vector<ConversationTurn> &)> history_prefix_steps;
};

using OpenTurn = std::function<UserTurn(const runtime::TaskRequest &, HistoryMode)>;

// C++ conversation owner. Explicit end_turn starts a turn-based response;
// polling publishes independently consumable text/audio. Ending early commits
// only the generated prefix. Model inference remains serialized by the caller.
class Session final {
public:
    Session(std::string model_name, OpenTurn open);
    ~Session();
    void begin(const runtime::TaskRequest & request, HistoryMode history = HistoryMode::Conversation);
    void push_audio(const runtime::AudioChunk & chunk);
    void end_turn();
    // Close the input stream and publish any codec tail through poll(), then
    // finalize() collects the full result and releases the current execution.
    void end_input();
    std::optional<runtime::StreamEvent> poll();
    runtime::TaskResult finalize();
    runtime::TaskResult run(const runtime::TaskRequest & request, HistoryMode history = HistoryMode::Conversation);
    void interrupt();
    // Discard the current execution; completed conversation history is kept.
    void cancel();
    void reset();
    void clear_history();
    const std::vector<Exchange> & history() const;
    bool reply_limited() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void begin_user(const runtime::TaskRequest & request, bool streaming, HistoryMode history);
};

// Shared capability for composed and inherited model integrations. The borrowed
// conversation is valid for the lifetime of its owning model session.
class IConversationSession {
public:
    virtual ~IConversationSession() = default;
    virtual Session & conversation() = 0;
};

// Compatibility adapter for the existing task API. Models bind inference once;
// they do not reimplement buffering, reply progression or stream lifecycle.
// Request artifacts remain authoritative for this stateless legacy interface.
class TaskSession : public runtime::IOfflineVoiceTaskSession, public runtime::IStreamingVoiceTaskSession, public IConversationSession {
public:
    TaskSession() = default;
    TaskSession(const TaskSession &) = delete;
    TaskSession & operator=(const TaskSession &) = delete;
    TaskSession(TaskSession &&) = delete;
    TaskSession & operator=(TaskSession &&) = delete;
    Session & conversation() final;
    runtime::TaskResult run(const runtime::TaskRequest & request) final;
    runtime::StreamingPolicy streaming_policy() const final;
    void start_stream(const runtime::TaskRequest & request) final;
    std::optional<runtime::StreamEvent> next_stream_event() final;
    void set_stream_event_sink(runtime::StreamEventCallback) final;
    runtime::TaskResult finish_stream() final;
    void reset() final;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) final;
    runtime::TaskResult finalize() final;

protected:
    bool reply_limited() const;
    void bind_chat(std::string model_name, OpenTurn open,
        runtime::StreamingPolicy policy = {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::PullEvents, 0, 0.1});

private:
    std::unique_ptr<Session> chat_;
    runtime::StreamingPolicy policy_;
    std::chrono::steady_clock::time_point stream_start_;
};

}  // namespace engine::chat
