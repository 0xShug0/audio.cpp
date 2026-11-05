#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include "engine/framework/runtime/session.h"

namespace engine::community_models::lfm3_audio {

struct Lfm3AudioAssets;

struct Lfm3TokenIds {  // resolved by content via find_token_id; throws if any is missing
    int32_t bos, im_start, im_end, nl /*"\n" = 207*/, text_end, audio_start, mixed_start, audio, audio_out, think, think_end;
};

class Lfm3Tokenizer {
public:
    explicit Lfm3Tokenizer(const Lfm3AudioAssets & assets);
    explicit Lfm3Tokenizer(const std::filesystem::path & tokenizer_json);  // Llama3 pre-tokenizer, ignore_merges honoured
    explicit Lfm3Tokenizer(const tokenizers::LlamaBpeGgufVocabulary & vocabulary);  // the same from tokenizer.ggml.*
    [[nodiscard]] std::vector<int32_t> encode(const std::string & text) const;  // parse_special = true (HF semantics)
    // HF decode(skip_special_tokens=True): drops CONTROL ids (is_control_token_id) and <|audio_out|>,
    // keeps <think>/</think> and tool tags. Implemented as filter + decode(ids, false), because vendor decode_bpe
    // skip=true also drops USER_DEFINED. skip_special = false is HF decode(skip_special_tokens=False).
    [[nodiscard]] std::string decode(const std::vector<int32_t> & ids, bool skip_special = true) const;
    [[nodiscard]] const Lfm3TokenIds & ids() const noexcept;

private:
    explicit Lfm3Tokenizer(tokenizers::LlamaBpeTokenizerSpec spec);

    std::shared_ptr<const tokenizers::LlamaBpeTokenizer> bpe_;
    Lfm3TokenIds ids_;
};

// The encoder output fills the <|audio|> ids in order; their count must equal the audio tokens (checked by backbone).
struct Lfm3Prompt {
    std::vector<int32_t> ids;                  // full sequence incl. BOS from the template
    std::vector<std::vector<int32_t>> frames;  // codes for each <|audio_out|> id, in order (history replay)
};

struct Lfm3Voice {
    std::string id, language_name /*"American English"*/, gender /*"female"*/;
};

const std::vector<Lfm3Voice> & lfm3_tts_voices();  // 38 voices
const Lfm3Voice & lfm3_tts_voice(const std::string & voice_id, const std::string & language);  // empty id: language default, female
std::string lfm3_tts_system_prompt(const Lfm3Voice & voice);       // "Perform TTS in American English using a female voice."
std::string lfm3_asr_system_prompt(const std::string & language);  // "auto" -> "Perform ASR."; ISO -> "Perform ASR in <Name>."
inline constexpr const char * kLfm3S2sSystemPrompt = "Respond with interleaved text and audio.";

// Each builder renders the chat_template text and tokenizes it with encode().
Lfm3Prompt make_lfm3_asr_prompt(const Lfm3Tokenizer &, const std::string & system_prompt, int64_t audio_tokens);
Lfm3Prompt make_lfm3_chat_prompt(const Lfm3Tokenizer &, int64_t audio_tokens, bool thinking);  // no system; "<think></think>" or "<think>"
Lfm3Prompt make_lfm3_tts_prompt(const Lfm3Tokenizer &, const std::string & system_prompt, const std::string & text);  // ends <|audio_start|>
// S2S is in chat.h (history).
// InvalidRequestError if ids tokenized from request text hold <|audio|> or <|audio_out|>, which only audio fills.
void require_lfm3_no_audio_placeholders(const std::vector<int32_t> & ids, const Lfm3TokenIds & token_ids);

enum class Lfm3Mode { Text, Audio, Interleaved };
Lfm3Mode lfm3_initial_mode(const Lfm3Prompt & prompt, const Lfm3TokenIds & ids, int32_t end_of_audio);

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
