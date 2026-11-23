#include "engine/community_models/lfm2_audio/chat.h"

#include "engine/community_models/lfm2_audio/asr_inputs.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

// Never looked up: audio embeddings and frames take these positions.
constexpr int32_t kPlaceholderId = 0;
// A frame in a reply payload starts with this.
constexpr int32_t kFrameMarker = -1;

void append(std::vector<int32_t> & out, const std::vector<int32_t> & ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

void append_audio(Lfm2Prompt & prompt, int64_t tokens) {
    if (tokens <= 0) {
        throw std::runtime_error("LFM2-Audio conversation turn needs its question's audio");
    }

    for (int64_t i = 0; i < tokens; ++i) {
        prompt.audio_positions.push_back(static_cast<int32_t>(prompt.input_ids.size()));
        prompt.input_ids.push_back(kPlaceholderId);
    }
}


}  // namespace

Lfm2Prompt make_lfm2_chat_prompt(
    const Lfm2TextTokenizer & tokenizer,
    const std::string & system_prompt,
    const std::vector<Lfm2ChatTurn> & history,
    int64_t question_tokens) {
    // The first turn's pieces, which also check the vocabulary's markup
    // tokens; ChatState encodes each piece on its own.
    const auto first = make_lfm2_spoken_prompt(tokenizer, system_prompt);
    const auto end_turn = tokenizer.encode("<|im_end|>\n");
    const auto user_turn = tokenizer.encode("<|im_start|>user\n");

    Lfm2Prompt out;
    out.input_ids = first.prefix;
    size_t codebooks = 0;
    for (const auto & turn : history) {
        append_audio(out, turn.question_tokens);
        append(out.input_ids, first.suffix);
        for (const auto & step : turn.reply) {
            if (step.codes.empty()) {
                if (step.token < 0) {
                    throw std::runtime_error("LFM2-Audio reply step has neither a token nor codes");
                }

                out.input_ids.push_back(step.token);
                continue;
            }

            codebooks = codebooks == 0 ? step.codes.size() : codebooks;
            if (step.token != kFrameMarker || step.codes.size() != codebooks) {
                throw std::runtime_error("LFM2-Audio reply frames must all have one code per codebook and no token");
            }

            out.frame_positions.push_back(static_cast<int32_t>(out.input_ids.size()));
            out.input_ids.push_back(kPlaceholderId);
            out.frame_codes.insert(out.frame_codes.end(), step.codes.begin(), step.codes.end());
        }

        append(out.input_ids, end_turn);
        append(out.input_ids, user_turn);
    }

    append_audio(out, question_tokens);
    append(out.input_ids, first.suffix);
    return out;
}

chat::HistoryFormat lfm2_history_format(const Lfm2ReplyCheckpoint & checkpoint) {
    return {"LFM2-Audio", kLfm2QuestionArtifactId, kLfm2ReplyArtifactId, kLfm2ReplyFormat,
        checkpoint.codebooks, checkpoint.audio_vocab_size, checkpoint.text_vocab_size,
        kLfm2MaxConversationSteps, {{"language", checkpoint.language}}, {}};
}

runtime::VoiceArtifact make_lfm2_reply_artifact(const std::vector<Lfm2ReplyStep> & reply, bool ended, const Lfm2ReplyCheckpoint & checkpoint) {
    return chat::encode_reply(lfm2_history_format(checkpoint), reply, ended);
}
std::vector<Lfm2ReplyStep> read_lfm2_reply_artifact(const runtime::VoiceArtifact & artifact, const Lfm2ReplyCheckpoint & checkpoint) {
    return chat::decode_reply(lfm2_history_format(checkpoint), artifact);
}
void require_lfm2_conversation_room(int64_t prompt_steps, bool at_least, int64_t max_tokens) {
    chat::require_conversation_room(lfm2_history_format({}), prompt_steps, at_least, max_tokens);
}
std::vector<Lfm2ConversationTurn> read_lfm2_conversation(const std::vector<runtime::VoiceArtifact> & artifacts,
    const Lfm2ReplyCheckpoint & checkpoint, int64_t max_tokens) {
    return chat::decode_conversation(lfm2_history_format(checkpoint), artifacts, max_tokens);
}

int64_t lfm2_chat_prompt_text_steps(
    const Lfm2TextTokenizer & tokenizer, const std::string & system_prompt, const std::vector<Lfm2ConversationTurn> & history) {
    // make_lfm2_chat_prompt's pieces.
    const auto first = make_lfm2_spoken_prompt(tokenizer, system_prompt);
    const auto next_user = tokenizer.encode("<|im_end|>\n").size() + tokenizer.encode("<|im_start|>user\n").size();
    auto steps = static_cast<int64_t>(first.prefix.size() + first.suffix.size());
    for (const auto & turn : history) {
        steps += static_cast<int64_t>(first.suffix.size() + turn.reply.size() + next_user);
    }

    return steps;
}

}  // namespace engine::community_models::lfm2_audio
