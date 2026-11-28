#pragma once

#include "engine/framework/runtime/session.h"
#include <functional>

#include <unordered_map>

namespace engine::chat {

// One causal inference result. A result contains either a text token or one
// complete audio-code frame. Protocol markers remain the generator's concern.
struct ReplyStep {
    int32_t token = -1;
    std::vector<int32_t> codes;
};



struct ConversationTurn {
    runtime::AudioBuffer question;
    std::vector<ReplyStep> reply;
};

// Description of a checkpoint's existing token-history wire format. Identity
// and tensor dimensions are data; parsing, validation and retention are shared.
struct HistoryFormat {
    std::string model_name;
    std::string question_id;
    std::string reply_id;
    std::string format;
    int64_t codebooks = 0;
    int64_t audio_vocab_size = 0;
    int64_t text_vocab_size = 0;
    int64_t context_steps = 0;
    std::unordered_map<std::string, std::string> checkpoint_tags;
    std::function<int64_t(const runtime::AudioBuffer &)> question_steps;
};

runtime::VoiceArtifact encode_reply(const HistoryFormat & format, const std::vector<ReplyStep> & steps, bool ended);
std::vector<ReplyStep> decode_reply(const HistoryFormat & format, const runtime::VoiceArtifact & artifact);
std::vector<ConversationTurn> decode_conversation(const HistoryFormat & format,
    const std::vector<runtime::VoiceArtifact> & artifacts, int64_t max_reply_steps);
void require_conversation_room(const HistoryFormat & format, int64_t prompt_steps, bool lower_bound, int64_t max_reply_steps);

}  // namespace engine::chat
