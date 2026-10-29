#pragma once

// LFM3-Audio conversations. Each S2S request carries the earlier turns back:
// every question as a WAV artifact and every reply as the raw step sequence
// the model generated (lfm3_audio.reply/1). Replaying that exact text/audio
// slot sequence equals HF's continued KV cache; the chat template's
// audio_out items would lose the interleaving.

#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/community_models/lfm3_audio/tokenizer.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {

struct Lfm3Step {  // codes non-empty <=> frame (EOS frame = all 2016)
    int32_t token = -1;
    std::vector<int32_t> codes;
};

inline constexpr const char * kLfm3QuestionArtifactId = "lfm3_audio.question";
inline constexpr const char * kLfm3ReplyArtifactId = "lfm3_audio.reply";
inline constexpr const char * kLfm3ReplyFormat = "lfm3_audio.reply/1";
inline constexpr int64_t kLfm3MaxConversationSteps = 8192;

struct Lfm3ReplyCheckpoint {
    int64_t codebooks = 12, audio_vocab_size = 2017, text_vocab_size = 128000;
};

// Payload: little-endian int32 per step. A text step is its token id (>= 0); a frame is -1 followed by
// `codebooks` codes, each in [0, audio_vocab_size). Meta keys: format, codebooks, audio_vocab_size,
// text_vocab_size, steps, ended.
runtime::VoiceArtifact make_lfm3_reply_artifact(const std::vector<Lfm3Step> & reply, bool ended, const Lfm3ReplyCheckpoint &);
std::vector<Lfm3Step> read_lfm3_reply_artifact(const runtime::VoiceArtifact &, const Lfm3ReplyCheckpoint &);

struct Lfm3ConversationTurn {
    runtime::AudioBuffer question;
    std::vector<Lfm3Step> reply;
};

// The <|audio|> slots the encoder gives a question, counted from its length before any resampling; 0 for no audio.
int64_t lfm3_question_tokens(const runtime::AudioBuffer & question);

// Counts every question's audio and every reply's steps against the room, before anything is encoded.
std::vector<Lfm3ConversationTurn> read_lfm3_conversation(
    const std::vector<runtime::VoiceArtifact> &, const Lfm3ReplyCheckpoint &, int64_t max_tokens);  // throws InvalidRequest/Capacity

// <bos>[<|im_start|>system\n{sys}<|im_end|>\n] then per turn
//   <|im_start|>user\n<|audio|>*Nq<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>{reply steps}[<|im_end|> if last step != im_end]\n
// then <|im_start|>user\n<|audio|>*N<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>
// Text steps go in as their ids, frames as <|audio_out|> + Lfm3Prompt::frames. Before an appended <|im_end|> (no `\n`
// precedes it), a reply whose last frame is not an EOS frame gets one (end_of_audio x codebooks; Lfm3DepthConfig::end_of_audio()).
Lfm3Prompt make_lfm3_s2s_prompt(const Lfm3Tokenizer &, const std::string & system_prompt,
                                const std::vector<int64_t> & history_question_tokens,
                                const std::vector<std::vector<Lfm3Step>> & history_replies, int64_t question_tokens,
                                int32_t end_of_audio);
void require_lfm3_conversation_room(int64_t prompt_steps, int64_t max_tokens);  // CapacityError past kLfm3MaxConversationSteps

}  // namespace engine::community_models::lfm3_audio
