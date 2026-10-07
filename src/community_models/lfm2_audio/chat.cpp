#include "engine/community_models/lfm2_audio/chat.h"

#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/errors.h"

#include <cstddef>
#include <exception>
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

void put_i32(std::vector<std::byte> & out, int32_t value) {
    const auto bits = static_cast<uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::byte>((bits >> shift) & 0xffu));
    }
}

int32_t get_i32(const std::vector<std::byte> & in, size_t offset) {
    uint32_t bits = 0;
    for (int i = 0; i < 4; ++i) {
        bits |= static_cast<uint32_t>(in[offset + static_cast<size_t>(i)]) << (8 * i);
    }

    return static_cast<int32_t>(bits);
}

// Whether a reply step is a text token or a frame of `codebooks` codes, all
// in range; `what` names the step for the error.
void check_step(const Lfm2ReplyStep & step, const Lfm2ReplyCheckpoint & checkpoint, const std::string & what) {
    if (step.codes.empty()) {
        if (step.token < 0 || step.token >= checkpoint.text_vocab_size) {
            throw std::runtime_error(what + " has token id " + std::to_string(step.token) + ", outside the " +
                                     std::to_string(checkpoint.text_vocab_size) + "-token vocabulary");
        }

        return;
    }

    if (step.token != kFrameMarker || static_cast<int64_t>(step.codes.size()) != checkpoint.codebooks) {
        throw std::runtime_error(what + " is a frame without one code per codebook (" + std::to_string(checkpoint.codebooks) + ")");
    }

    for (const int32_t code : step.codes) {
        if (code < 0 || code >= checkpoint.audio_vocab_size) {
            throw std::runtime_error(what + " has audio code " + std::to_string(code) + ", outside the codebook of " +
                                     std::to_string(checkpoint.audio_vocab_size));
        }
    }
}

const char * const kMetaKeys[] = {"format", "language", "codebooks", "audio_vocab_size", "text_vocab_size", "steps", "ended"};

// A meta value written as a decimal integer, nothing else.
int64_t meta_count(const std::string & value, const std::string & what) {
    if (value.empty() || value.size() > 18 || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error(what + " is not a whole number: \"" + value + "\"");
    }

    return std::stoll(value);
}

// A reply artifact's kind and meta, checked against the checkpoint, and the
// steps its meta says it holds; `where` names it for the errors.
int64_t reply_steps(const runtime::VoiceArtifact & artifact, const Lfm2ReplyCheckpoint & checkpoint, const std::string & where) {
    if (artifact.kind != runtime::ArtifactKind::AcousticTokens) {
        throw std::runtime_error(where + " must be of kind acoustic_tokens");
    }

    const auto & meta = artifact.meta;
    const auto field = [&](const char * key) -> const std::string & {
        const auto it = meta.find(key);
        if (it == meta.end()) {
            throw std::runtime_error(where + " has no meta " + key);
        }

        return it->second;
    };

    // The format first, since another version can have other meta.
    if (field("format") != kLfm2ReplyFormat) {
        throw std::runtime_error(where + " has format " + field("format") + "; this build reads " + kLfm2ReplyFormat);
    }

    for (const auto & [key, value] : meta) {
        bool known = false;
        for (const char * expected : kMetaKeys) {
            known = known || key == expected;
        }

        if (!known) {
            throw std::runtime_error(where + " has meta " + key + ", which " + kLfm2ReplyFormat + " does not have");
        }
    }

    if (field("language") != checkpoint.language) {
        throw std::runtime_error(where + " comes from a " + field("language") + " checkpoint, and this one is " + checkpoint.language);
    }

    const std::pair<const char *, int64_t> sizes[] = {
        {"codebooks", checkpoint.codebooks}, {"audio_vocab_size", checkpoint.audio_vocab_size}, {"text_vocab_size", checkpoint.text_vocab_size}};
    for (const auto & [key, expected] : sizes) {
        if (meta_count(field(key), where + " meta " + key) != expected) {
            throw std::runtime_error(where + " has " + key + " " + field(key) + ", and this checkpoint " + std::to_string(expected));
        }
    }

    const auto steps = meta_count(field("steps"), where + " meta steps");
    if (field("ended") != "true" && field("ended") != "false") {
        throw std::runtime_error(where + " meta ended must be true or false, not \"" + field("ended") + "\"");
    }

    return steps;
}

// The payload of a reply artifact whose meta says it holds `steps` steps,
// read no further than those.
std::vector<Lfm2ReplyStep> reply_payload(
    const runtime::VoiceArtifact & artifact, const Lfm2ReplyCheckpoint & checkpoint, const std::string & where, int64_t steps) {
    const auto & payload = artifact.payload;
    if (payload.size() % 4 != 0) {
        throw std::runtime_error(where + " payload is " + std::to_string(payload.size()) + " bytes, not whole int32 values");
    }

    const size_t values = payload.size() / 4;
    std::vector<Lfm2ReplyStep> out;
    for (size_t at = 0; at < values;) {
        if (static_cast<int64_t>(out.size()) == steps) {
            throw std::runtime_error(where + " holds more than the " + std::to_string(steps) + " steps its meta says");
        }

        Lfm2ReplyStep step;
        step.token = get_i32(payload, 4 * at++);
        if (step.token == kFrameMarker) {
            if (values - at < static_cast<size_t>(checkpoint.codebooks)) {
                throw std::runtime_error(where + " payload ends inside step " + std::to_string(out.size() + 1));
            }

            for (int64_t codebook = 0; codebook < checkpoint.codebooks; ++codebook) {
                step.codes.push_back(get_i32(payload, 4 * at++));
            }
        }

        check_step(step, checkpoint, where + " step " + std::to_string(out.size() + 1));
        out.push_back(std::move(step));
    }

    if (static_cast<int64_t>(out.size()) != steps) {
        throw std::runtime_error(where + " holds " + std::to_string(out.size()) + " steps, and its meta says " + std::to_string(steps));
    }

    return out;
}

std::vector<Lfm2ReplyStep> read_reply(const runtime::VoiceArtifact & artifact, const Lfm2ReplyCheckpoint & checkpoint, const std::string & where) {
    return reply_payload(artifact, checkpoint, where, reply_steps(artifact, checkpoint, where));
}

runtime::AudioBuffer read_question(const runtime::VoiceArtifact & artifact, const std::string & name) {
    const std::string where = "LFM2-Audio " + name;
    if (artifact.kind != runtime::ArtifactKind::Custom) {
        throw std::runtime_error(where + " must be of kind custom");
    }

    audio::WavData wav;
    try {
        wav = audio::read_wav_f32(std::string_view(reinterpret_cast<const char *>(artifact.payload.data()), artifact.payload.size()));
    } catch (const std::exception & error) {
        throw std::runtime_error(where + " does not read as a WAV file: " + error.what());
    }

    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
        throw std::runtime_error(where + " holds no audio");
    }

    return runtime::AudioBuffer{wav.sample_rate, wav.channels, std::move(wav.samples)};
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

runtime::VoiceArtifact make_lfm2_reply_artifact(
    const std::vector<Lfm2ReplyStep> & reply, bool ended, const Lfm2ReplyCheckpoint & checkpoint) {
    std::vector<std::byte> payload;
    for (size_t i = 0; i < reply.size(); ++i) {
        const auto & step = reply[i];
        check_step(step, checkpoint, "LFM2-Audio reply step " + std::to_string(i + 1));
        put_i32(payload, step.token);
        for (const int32_t code : step.codes) {
            put_i32(payload, code);
        }
    }

    return runtime::make_voice_artifact(
        runtime::ArtifactKind::AcousticTokens,
        kLfm2ReplyArtifactId,
        std::move(payload),
        {{"format", kLfm2ReplyFormat},
         {"language", checkpoint.language},
         {"codebooks", std::to_string(checkpoint.codebooks)},
         {"audio_vocab_size", std::to_string(checkpoint.audio_vocab_size)},
         {"text_vocab_size", std::to_string(checkpoint.text_vocab_size)},
         {"steps", std::to_string(reply.size())},
         {"ended", ended ? "true" : "false"}});
}

std::vector<Lfm2ReplyStep> read_lfm2_reply_artifact(const runtime::VoiceArtifact & artifact, const Lfm2ReplyCheckpoint & checkpoint) {
    if (artifact.id != kLfm2ReplyArtifactId) {
        throw std::runtime_error("LFM2-Audio artifact " + artifact.id + " is not an " + kLfm2ReplyArtifactId + " artifact");
    }

    return read_reply(artifact, checkpoint, std::string("LFM2-Audio ") + kLfm2ReplyArtifactId + " artifact");
}

void require_lfm2_conversation_room(int64_t prompt_steps, bool at_least, int64_t max_tokens) {
    // Written so that nothing overflows, as max_tokens can be any int64.
    if (prompt_steps > kLfm2MaxConversationSteps || max_tokens > kLfm2MaxConversationSteps - prompt_steps) {
        throw runtime::CapacityError("LFM2-Audio conversation needs " + std::string(at_least ? "at least " : "") + std::to_string(prompt_steps) +
                                     " prompt steps plus max_tokens=" + std::to_string(max_tokens) + ", more than the " +
                                     std::to_string(kLfm2MaxConversationSteps) +
                                     " a conversation may take; leave out its oldest turns, or lower max_tokens");
    }
}

std::vector<Lfm2ConversationTurn> read_lfm2_conversation(
    const std::vector<runtime::VoiceArtifact> & artifacts, const Lfm2ReplyCheckpoint & checkpoint, int64_t max_tokens) {
    std::vector<Lfm2ConversationTurn> turns;
    bool awaiting_reply = false;
    int64_t steps = 0;  // the replies' so far, never over kLfm2MaxConversationSteps
    for (size_t i = 0; i < artifacts.size(); ++i) {
        const auto & artifact = artifacts[i];
        const auto name = "input artifact " + std::to_string(i + 1) + " (" + artifact.id + ")";
        if (artifact.id == kLfm2QuestionArtifactId) {
            if (awaiting_reply) {
                throw std::runtime_error("LFM2-Audio " + name + " follows a question; each question needs its reply after it");
            }

            turns.push_back({read_question(artifact, name), {}});
            awaiting_reply = true;
        } else if (artifact.id == kLfm2ReplyArtifactId) {
            if (!awaiting_reply) {
                throw std::runtime_error("LFM2-Audio " + name + " has no question before it");
            }

            // A reply past the room left is never read, however long its
            // payload.
            const auto where = "LFM2-Audio " + name;
            const auto count = reply_steps(artifact, checkpoint, where);
            require_lfm2_conversation_room(steps + count, true, max_tokens);
            steps += count;
            turns.back().reply = reply_payload(artifact, checkpoint, where, count);
            awaiting_reply = false;
        } else {
            throw std::runtime_error("LFM2-Audio s2s takes " + std::string(kLfm2QuestionArtifactId) + " and " + kLfm2ReplyArtifactId +
                                     " artifacts only, not " + name);
        }
    }

    if (awaiting_reply) {
        throw std::runtime_error("LFM2-Audio input artifact " + std::to_string(artifacts.size()) + " (" + kLfm2QuestionArtifactId +
                                 ") has no reply after it; the new question goes in audio_input");
    }

    return turns;
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
