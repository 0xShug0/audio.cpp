#include "engine/community_models/lfm3_audio/chat.h"

#include "engine/community_models/lfm3_audio/features.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

// A frame in a reply payload starts with this.
constexpr int32_t kFrameMarker = -1;

constexpr int64_t kQuestionSampleRate = 16000;

void append(std::vector<int32_t> & out, const std::vector<int32_t> & ids) {
    out.insert(out.end(), ids.begin(), ids.end());
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

// A bad step read from a request is the caller's error (InvalidRequestError),
// one about to be written is the model's (std::runtime_error).
template <typename Error>
void check_step(const Lfm3Step & step, const Lfm3ReplyCheckpoint & checkpoint, const std::string & what) {
    if (step.codes.empty()) {
        if (step.token < 0 || step.token >= checkpoint.text_vocab_size) {
            throw Error(what + " has token id " + std::to_string(step.token) + ", outside the " +
                        std::to_string(checkpoint.text_vocab_size) + "-token vocabulary");
        }

        return;
    }

    if (step.token != kFrameMarker || static_cast<int64_t>(step.codes.size()) != checkpoint.codebooks) {
        throw Error(what + " is a frame without one code per codebook (" + std::to_string(checkpoint.codebooks) + ")");
    }

    for (const int32_t code : step.codes) {
        if (code < 0 || code >= checkpoint.audio_vocab_size) {
            throw Error(what + " has audio code " + std::to_string(code) + ", outside the codebook of " +
                        std::to_string(checkpoint.audio_vocab_size));
        }
    }
}

const char * const kMetaKeys[] = {"format", "codebooks", "audio_vocab_size", "text_vocab_size", "steps", "ended"};

int64_t meta_count(const std::string & value, const std::string & what) {
    if (value.empty() || value.size() > 18 || value.find_first_not_of("0123456789") != std::string::npos) {
        throw runtime::InvalidRequestError(what + " is not a whole number: \"" + value + "\"");
    }

    return std::stoll(value);
}

// A reply artifact's kind and meta, checked against the checkpoint, and the
// steps its meta says it holds.
int64_t reply_steps(const runtime::VoiceArtifact & artifact, const Lfm3ReplyCheckpoint & checkpoint, const std::string & where) {
    if (artifact.kind != runtime::ArtifactKind::AcousticTokens) {
        throw runtime::InvalidRequestError(where + " must be of kind acoustic_tokens");
    }

    const auto & meta = artifact.meta;
    const auto field = [&](const char * key) -> const std::string & {
        const auto it = meta.find(key);
        if (it == meta.end()) {
            throw runtime::InvalidRequestError(where + " has no meta " + key);
        }

        return it->second;
    };

    // The format first, since another version can have other meta.
    if (field("format") != kLfm3ReplyFormat) {
        throw runtime::InvalidRequestError(where + " has format " + field("format") + "; this build reads " + kLfm3ReplyFormat);
    }

    for (const auto & [key, value] : meta) {
        bool known = false;
        for (const char * expected : kMetaKeys) {
            known = known || key == expected;
        }

        if (!known) {
            throw runtime::InvalidRequestError(where + " has meta " + key + ", which " + kLfm3ReplyFormat + " does not have");
        }
    }

    const std::pair<const char *, int64_t> sizes[] = {
        {"codebooks", checkpoint.codebooks}, {"audio_vocab_size", checkpoint.audio_vocab_size}, {"text_vocab_size", checkpoint.text_vocab_size}};
    for (const auto & [key, expected] : sizes) {
        if (meta_count(field(key), where + " meta " + key) != expected) {
            throw runtime::InvalidRequestError(where + " has " + key + " " + field(key) + ", and this checkpoint " + std::to_string(expected));
        }
    }

    const auto steps = meta_count(field("steps"), where + " meta steps");
    if (field("ended") != "true" && field("ended") != "false") {
        throw runtime::InvalidRequestError(where + " meta ended must be true or false, not \"" + field("ended") + "\"");
    }

    return steps;
}

std::vector<Lfm3Step> reply_payload(
    const runtime::VoiceArtifact & artifact, const Lfm3ReplyCheckpoint & checkpoint, const std::string & where, int64_t steps) {
    const auto & payload = artifact.payload;
    if (payload.size() % 4 != 0) {
        throw runtime::InvalidRequestError(where + " payload is " + std::to_string(payload.size()) + " bytes, not whole int32 values");
    }

    const size_t values = payload.size() / 4;
    std::vector<Lfm3Step> out;
    for (size_t at = 0; at < values;) {
        if (static_cast<int64_t>(out.size()) == steps) {
            throw runtime::InvalidRequestError(where + " holds more than the " + std::to_string(steps) + " steps its meta says");
        }

        Lfm3Step step;
        step.token = get_i32(payload, 4 * at++);
        if (step.token == kFrameMarker) {
            if (values - at < static_cast<size_t>(checkpoint.codebooks)) {
                throw runtime::InvalidRequestError(where + " payload ends inside step " + std::to_string(out.size() + 1));
            }

            for (int64_t codebook = 0; codebook < checkpoint.codebooks; ++codebook) {
                step.codes.push_back(get_i32(payload, 4 * at++));
            }
        }

        check_step<runtime::InvalidRequestError>(step, checkpoint, where + " step " + std::to_string(out.size() + 1));
        out.push_back(std::move(step));
    }

    if (static_cast<int64_t>(out.size()) != steps) {
        throw runtime::InvalidRequestError(where + " holds " + std::to_string(out.size()) + " steps, and its meta says " + std::to_string(steps));
    }

    return out;
}

runtime::AudioBuffer read_question(const runtime::VoiceArtifact & artifact, const std::string & name) {
    const std::string where = "LFM3-Audio " + name;
    if (artifact.kind != runtime::ArtifactKind::Custom) {
        throw runtime::InvalidRequestError(where + " must be of kind custom");
    }

    audio::WavData wav;
    try {
        wav = audio::read_wav_f32(std::string_view(reinterpret_cast<const char *>(artifact.payload.data()), artifact.payload.size()));
    } catch (const std::exception & error) {
        throw runtime::InvalidRequestError(where + " does not read as a WAV file: " + error.what());
    }

    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
        throw runtime::InvalidRequestError(where + " holds no audio");
    }

    return runtime::AudioBuffer{wav.sample_rate, wav.channels, std::move(wav.samples)};
}

// `at_least` says that the prompt takes at least `prompt_steps`, the audio
// and markup of its turns not counted yet.
void require_room(int64_t prompt_steps, bool at_least, int64_t max_tokens) {
    // Written so that nothing overflows, as max_tokens can be any int64.
    if (prompt_steps > kLfm3MaxConversationSteps || max_tokens > kLfm3MaxConversationSteps - prompt_steps) {
        throw runtime::CapacityError("LFM3-Audio conversation needs " + std::string(at_least ? "at least " : "") + std::to_string(prompt_steps) +
                                     " prompt steps plus max_tokens=" + std::to_string(max_tokens) + ", more than the " +
                                     std::to_string(kLfm3MaxConversationSteps) +
                                     " a conversation may take; leave out its oldest turns, or lower max_tokens");
    }
}

}  // namespace

runtime::VoiceArtifact make_lfm3_reply_artifact(const std::vector<Lfm3Step> & reply, bool ended, const Lfm3ReplyCheckpoint & checkpoint) {
    std::vector<std::byte> payload;
    for (size_t i = 0; i < reply.size(); ++i) {
        const auto & step = reply[i];
        check_step<std::runtime_error>(step, checkpoint, "LFM3-Audio reply step " + std::to_string(i + 1));
        put_i32(payload, step.token);
        for (const int32_t code : step.codes) {
            put_i32(payload, code);
        }
    }

    return runtime::make_voice_artifact(
        runtime::ArtifactKind::AcousticTokens,
        kLfm3ReplyArtifactId,
        std::move(payload),
        {{"format", kLfm3ReplyFormat},
         {"codebooks", std::to_string(checkpoint.codebooks)},
         {"audio_vocab_size", std::to_string(checkpoint.audio_vocab_size)},
         {"text_vocab_size", std::to_string(checkpoint.text_vocab_size)},
         {"steps", std::to_string(reply.size())},
         {"ended", ended ? "true" : "false"}});
}

std::vector<Lfm3Step> read_lfm3_reply_artifact(const runtime::VoiceArtifact & artifact, const Lfm3ReplyCheckpoint & checkpoint) {
    if (artifact.id != kLfm3ReplyArtifactId) {
        throw runtime::InvalidRequestError("LFM3-Audio artifact " + artifact.id + " is not an " + kLfm3ReplyArtifactId + " artifact");
    }

    const std::string where = std::string("LFM3-Audio ") + kLfm3ReplyArtifactId + " artifact";
    return reply_payload(artifact, checkpoint, where, reply_steps(artifact, checkpoint, where));
}

int64_t lfm3_question_tokens(const runtime::AudioBuffer & question) {
    if (question.sample_rate <= 0 || question.channels <= 0 || question.samples.empty()) {
        return 0;
    }

    // lfm3_audio_mono_16k gives ceil(n * 16000 / rate) samples, with soxr and with the torchaudio fallback.
    const auto frames = static_cast<int64_t>(question.samples.size() / static_cast<size_t>(question.channels));
    const int64_t samples = (frames * kQuestionSampleRate + question.sample_rate - 1) / question.sample_rate;
    return lfm3_audio_tokens(lfm3_feature_frames(samples));
}

std::vector<Lfm3ConversationTurn> read_lfm3_conversation(
    const std::vector<runtime::VoiceArtifact> & artifacts, const Lfm3ReplyCheckpoint & checkpoint, int64_t max_tokens) {
    std::vector<Lfm3ConversationTurn> turns;
    bool awaiting_reply = false;
    int64_t steps = 0;  // the questions' and replies' so far, never over kLfm3MaxConversationSteps
    for (size_t i = 0; i < artifacts.size(); ++i) {
        const auto & artifact = artifacts[i];
        const auto name = "input artifact " + std::to_string(i + 1) + " (" + artifact.id + ")";
        if (artifact.id == kLfm3QuestionArtifactId) {
            if (awaiting_reply) {
                throw runtime::InvalidRequestError("LFM3-Audio " + name + " follows a question; each question needs its reply after it");
            }

            turns.push_back({read_question(artifact, name), {}});
            const auto count = lfm3_question_tokens(turns.back().question);
            require_room(steps + count, true, max_tokens);
            steps += count;
            awaiting_reply = true;
        } else if (artifact.id == kLfm3ReplyArtifactId) {
            if (!awaiting_reply) {
                throw runtime::InvalidRequestError("LFM3-Audio " + name + " has no question before it");
            }

            // A reply past the room left is never read, however long its payload.
            const auto where = "LFM3-Audio " + name;
            const auto count = reply_steps(artifact, checkpoint, where);
            require_room(steps + count, true, max_tokens);
            steps += count;
            turns.back().reply = reply_payload(artifact, checkpoint, where, count);
            awaiting_reply = false;
        } else {
            throw runtime::InvalidRequestError("LFM3-Audio s2s takes " + std::string(kLfm3QuestionArtifactId) + " and " + kLfm3ReplyArtifactId +
                                               " artifacts only, not " + name);
        }
    }

    if (awaiting_reply) {
        throw runtime::InvalidRequestError("LFM3-Audio input artifact " + std::to_string(artifacts.size()) + " (" + kLfm3QuestionArtifactId +
                                           ") has no reply after it; the new question goes in audio_input");
    }

    return turns;
}

Lfm3Prompt make_lfm3_s2s_prompt(const Lfm3Tokenizer & tokenizer, const std::string & system_prompt,
                                const std::vector<int64_t> & history_question_tokens,
                                const std::vector<std::vector<Lfm3Step>> & history_replies, int64_t question_tokens,
                                int32_t end_of_audio) {
    if (history_question_tokens.size() != history_replies.size()) {
        throw std::runtime_error("LFM3-Audio conversation has " + std::to_string(history_question_tokens.size()) + " earlier questions and " +
                                 std::to_string(history_replies.size()) + " replies");
    }

    const auto & ids = tokenizer.ids();
    const auto open = tokenizer.encode("<|startoftext|>" + (system_prompt.empty() ? std::string() : "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"));
    require_lfm3_no_audio_placeholders(open, ids);  // the system prompt can come from the request
    const auto user = tokenizer.encode("<|im_start|>user\n");
    const auto answer = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>");

    Lfm3Prompt out;
    out.ids = open;
    const auto ask = [&](int64_t tokens) {
        if (tokens <= 0) {
            throw std::runtime_error("LFM3-Audio conversation turn needs its question's audio");
        }

        append(out.ids, user);
        out.ids.insert(out.ids.end(), static_cast<size_t>(tokens), ids.audio);
        append(out.ids, answer);
    };
    const auto add_frame = [&](const std::vector<int32_t> & codes) {
        out.ids.push_back(ids.audio_out);
        out.frames.push_back(codes);
    };

    for (size_t turn = 0; turn < history_replies.size(); ++turn) {
        ask(history_question_tokens[turn]);
        const auto & reply = history_replies[turn];
        const std::vector<int32_t> * last_frame = nullptr;
        for (size_t i = 0; i < reply.size(); ++i) {
            const auto & step = reply[i];
            const auto what = "LFM3-Audio reply " + std::to_string(turn + 1) + " step " + std::to_string(i + 1);
            if (!step.codes.empty()) {
                if (step.token != kFrameMarker || (last_frame != nullptr && step.codes.size() != last_frame->size())) {
                    throw std::runtime_error(what + " is a frame with a token or another codebook count");
                }

                add_frame(step.codes);
                last_frame = &step.codes;
                continue;
            }

            if (step.token < 0) {
                throw std::runtime_error(what + " has neither a token nor codes");
            }

            // The placeholders take their rows from audio and frames; as text they would shift both.
            if (step.token == ids.audio || step.token == ids.audio_out) {
                throw runtime::InvalidRequestError(what + " is an audio placeholder token, which a reply never holds as text");
            }

            out.ids.push_back(step.token);
        }

        // A cut reply is closed as the template closes a turn, and its audio
        // as the processor closes a clip: with an end-of-audio frame.
        if (reply.empty() || reply.back().token != ids.im_end) {
            if (last_frame != nullptr && std::find(last_frame->begin(), last_frame->end(), end_of_audio) == last_frame->end()) {
                add_frame(std::vector<int32_t>(last_frame->size(), end_of_audio));
            }

            out.ids.push_back(ids.im_end);
        }

        out.ids.push_back(ids.nl);
    }

    ask(question_tokens);
    return out;
}

void require_lfm3_conversation_room(int64_t prompt_steps, int64_t max_tokens) {
    require_room(prompt_steps, false, max_tokens);
}

}  // namespace engine::community_models::lfm3_audio
