#pragma once

#include "engine/framework/chat/history.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace engine::chat {

using AudioCodes = std::vector<std::vector<int32_t>>;

// next() is one inference step. decode_text() decodes only the supplied new
// IDs, not a cumulative transcript; it may end inside a UTF-8 character.
// history(), when supplied, borrows immutable source-owned steps until the
// binding is released. Captures must keep that source alive.
struct ReplySource {
    std::function<std::optional<ReplyStep>()> next;
    std::function<bool()> ended;
    std::function<std::string(const std::vector<int32_t> &)> decode_text;
    std::function<const std::vector<ReplyStep> &()> history;
};

// Bind inference operations, never a generation loop. The reply owns codec
// progression: begin once, push newly generated frames, then flush exactly once.
struct ReplyCodec {
    int sample_rate = 0;
    int channels = 1;
    std::function<std::vector<float>(const AudioCodes &)> decode;
    std::function<void()> begin;
    std::function<std::vector<float>(const AudioCodes &)> push;
    std::function<std::vector<float>()> finish;
};

struct ReplyConfig {
    int32_t end_of_audio = -1;
    size_t frames_per_event = 1;
    std::string language;
    int64_t max_steps = 0;
    bool require_audio = true;
    // Whole-reply text EOS IDs, not interleaving/audio phase markers.
    std::vector<int32_t> eos_tokens = {};
    // Otherwise collect text until frames_per_event or source exhaustion.
    bool publish_text_immediately = true;
};

// The factory prepares a text source. Its bindings are destroyed after text
// decoding and before synthesis, allowing stage-owned device storage to be freed.
struct TextSpeechSource {
    std::function<ReplySource()> text_source;
    std::function<std::vector<float>(const std::string &)> synthesize;
    int sample_rate = 0;
    std::string language;
    int64_t max_steps = 0;
    std::vector<int32_t> eos_tokens = {};
};

// Session-owned reply execution. Offline decoding retains the whole-sequence
// codec path; streaming groups audio frames and follows the text publication policy.
// Neither path changes the generator's sampling or recurrent state updates.
class Reply final {
public:
    Reply(ReplySource source, ReplyCodec codec, ReplyConfig config);
    explicit Reply(TextSpeechSource source);
    ~Reply();
    Reply(const Reply &) = delete;
    Reply & operator=(const Reply &) = delete;

    void retain_history();
    void enable_history(HistoryFormat format, bool emit_artifact);
    void run();
    runtime::TaskResult finalize();
    std::optional<runtime::StreamEvent> next();
    runtime::TaskResult result() const;
    bool done() const;
    bool reached_eos() const;
    size_t audio_frames() const;
    size_t text_tokens() const;
    const std::vector<ReplyStep> & steps() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::chat
