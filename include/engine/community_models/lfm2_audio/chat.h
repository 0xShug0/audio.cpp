#pragma once

// LFM2.5-Audio conversations: the prompt of a turn after earlier ones, as
// liquid-audio's ChatState (processor.py) holds it in the README's multi-turn
// example, and the artifacts a request carries its earlier turns in.
// liquid-audio prefills the whole conversation again for every turn, so a
// turn's reply depends only on the system prompt, the earlier questions and
// replies, the new question and the request's options. A client keeps the
// conversation and sends it back with each turn.

#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/interleaved.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {

// An earlier turn as the prompt holds it: the audio positions its question
// took and every step its reply yielded, the frame that ends the audio and
// frames with end-of-audio for another codebook included, as ChatState keeps
// them.
struct Lfm2ChatTurn {
    int64_t question_tokens = 0;
    std::vector<Lfm2ReplyStep> reply;
};

// The prompt of a turn after `history`, as ChatState holds it when the
// README's multi-turn example generates the reply:
//   <|startoftext|><|im_start|>system\n{system prompt}<|im_end|>\n<|im_start|>user\n
// then for each earlier turn
//   {question}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n<|im_start|>user\n
// and the new question, {question}<|im_end|>\n<|im_start|>assistant\n.
// The system prompt comes once, at the start. demo/chat.py, as written, adds
// the system turn again after the open user turn of every turn: chat.text is
// [1, n], so its len(chat.text) == 1 always holds. This prompt does not. A
// reply goes in step by step: its token ids, never its text encoded again,
// and its frames as the backbone takes frames. Without history this is
// make_lfm2_spoken_prompt's prompt with the audio in.
Lfm2Prompt make_lfm2_chat_prompt(
    const Lfm2TextTokenizer & tokenizer,
    const std::string & system_prompt,
    const std::vector<Lfm2ChatTurn> & history,
    int64_t question_tokens);

// The most steps a conversation's prompt and reply may take together, well
// within the 128000-token context, so that a conversation keeps to a few
// hundred MiB of decode cache. The client keeps the conversation and leaves
// out its oldest turns past this; nothing here leaves out any itself.
inline constexpr int64_t kLfm2MaxConversationSteps = 8192;

// Throws CapacityError when a prompt of `prompt_steps` steps and a reply of
// max_tokens come to more than kLfm2MaxConversationSteps. `at_least` says
// that the prompt takes at least that many, the audio of some of its
// questions not counted yet.
void require_lfm2_conversation_room(int64_t prompt_steps, bool at_least, int64_t max_tokens);

// A request's earlier turns are its input artifacts, in conversation order:
// for each turn a question, the user's audio as WAV bytes (kind Custom), then
// the reply artifact that turn's result returned, unchanged.
inline constexpr const char * kLfm2QuestionArtifactId = "lfm2_audio.question";
inline constexpr const char * kLfm2ReplyArtifactId = "lfm2_audio.reply";
inline constexpr const char * kLfm2ReplyFormat = "lfm2_audio.reply/1";

// What a reply artifact records of the checkpoint that wrote it, and must
// match where it is read: a reply carries token ids and codes, so it replays
// on any quantization of the same checkpoint.
struct Lfm2ReplyCheckpoint {
    std::string language;
    int64_t codebooks = 0;
    int64_t audio_vocab_size = 0;  // codes per codebook, end-of-audio included
    int64_t text_vocab_size = 0;
};

// Kind AcousticTokens, id lfm2_audio.reply. The payload is little-endian
// int32: a text step is its token id, and a frame is -1 followed by one code
// per codebook. The meta holds format (lfm2_audio.reply/1), the checkpoint's
// language, codebooks, audio_vocab_size and text_vocab_size, steps (text
// tokens and frames together) and ended ("false" when max_tokens, or a stream
// finished early, cut the reply off).
runtime::VoiceArtifact make_lfm2_reply_artifact(
    const std::vector<Lfm2ReplyStep> & reply, bool ended, const Lfm2ReplyCheckpoint & checkpoint);

// The steps of a reply artifact. Throws InvalidRequestError unless its kind,
// format and meta are the ones make_lfm2_reply_artifact writes for
// `checkpoint`, with no other meta, and its payload holds `steps` whole steps
// of tokens and codes in range.
std::vector<Lfm2ReplyStep> read_lfm2_reply_artifact(const runtime::VoiceArtifact & artifact, const Lfm2ReplyCheckpoint & checkpoint);

// An earlier turn as a request carries it.
struct Lfm2ConversationTurn {
    runtime::AudioBuffer question;  // as the WAV holds it
    std::vector<Lfm2ReplyStep> reply;
};

// The earlier turns in a request's input artifacts: a question and then a
// reply for each turn, in order. Throws InvalidRequestError on any other
// artifact, on one out of turn, on a question without its reply, and on a
// question or reply that does not read. The replies are counted from their
// meta before their payloads are read, and once they leave no room for
// max_tokens this throws CapacityError, as require_lfm2_conversation_room,
// without reading further.
std::vector<Lfm2ConversationTurn> read_lfm2_conversation(
    const std::vector<runtime::VoiceArtifact> & artifacts, const Lfm2ReplyCheckpoint & checkpoint, int64_t max_tokens);

// The steps of make_lfm2_chat_prompt's prompt for a turn after `history`
// other than its questions' audio, which only the encoder counts: the chat
// markup, the system prompt and the replies.
int64_t lfm2_chat_prompt_text_steps(
    const Lfm2TextTokenizer & tokenizer, const std::string & system_prompt, const std::vector<Lfm2ConversationTurn> & history);

}  // namespace engine::community_models::lfm2_audio
