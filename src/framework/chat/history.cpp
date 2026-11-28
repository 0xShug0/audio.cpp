#include "engine/framework/chat/history.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/errors.h"

#include <exception>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::chat {
namespace {
constexpr int32_t kFrameMarker = -1;
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
// in range; `what` names the step for the error. A step read from a request
// is the caller's error (InvalidRequestError), one about to be written is
// the model's (std::runtime_error).
template <typename Error>
void check_step(const ReplyStep & step, const HistoryFormat & checkpoint, const std::string & what) {
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

// A meta value written as a decimal integer, nothing else.
int64_t meta_count(const std::string & value, const std::string & what) {
    if (value.empty() || value.size() > 18 || value.find_first_not_of("0123456789") != std::string::npos) {
        throw runtime::InvalidRequestError(what + " is not a whole number: \"" + value + "\"");
    }

    return std::stoll(value);
}

// A reply artifact's kind and meta, checked against the checkpoint, and the
// steps its meta says it holds; `where` names it for the errors.
int64_t reply_steps(const runtime::VoiceArtifact & artifact, const HistoryFormat & checkpoint, const std::string & where) {
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
    if (field("format") != checkpoint.format) {
        throw runtime::InvalidRequestError(where + " has format " + field("format") + "; this build reads " + checkpoint.format);
    }

    for (const auto & [key, value] : meta) {
        bool known = checkpoint.checkpoint_tags.count(key) != 0;
        for (const char * expected : kMetaKeys) {
            known = known || key == expected;
        }

        if (!known) {
            throw runtime::InvalidRequestError(where + " has meta " + key + ", which " + checkpoint.format + " does not have");
        }
    }

    for (const auto & [key, expected] : checkpoint.checkpoint_tags) {
        if (field(key.c_str()) != expected) {
            if (key == "language") {
                throw runtime::InvalidRequestError(where + " comes from a " + field(key.c_str()) + " checkpoint, and this one is " + expected);
            }
            throw runtime::InvalidRequestError(where + " has incompatible checkpoint field " + key);
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

// The payload of a reply artifact whose meta says it holds `steps` steps,
// read no further than those.
std::vector<ReplyStep> reply_payload(
    const runtime::VoiceArtifact & artifact, const HistoryFormat & checkpoint, const std::string & where, int64_t steps) {
    const auto & payload = artifact.payload;
    if (payload.size() % 4 != 0) {
        throw runtime::InvalidRequestError(where + " payload is " + std::to_string(payload.size()) + " bytes, not whole int32 values");
    }

    const size_t values = payload.size() / 4;
    std::vector<ReplyStep> out;
    for (size_t at = 0; at < values;) {
        if (static_cast<int64_t>(out.size()) == steps) {
            throw runtime::InvalidRequestError(where + " holds more than the " + std::to_string(steps) + " steps its meta says");
        }

        ReplyStep step;
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

runtime::AudioBuffer read_question(const runtime::VoiceArtifact & artifact, const std::string & where) {
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

}  // namespace

runtime::VoiceArtifact encode_reply(const HistoryFormat & checkpoint, const std::vector<ReplyStep> & reply, bool ended) {
    std::vector<std::byte> payload;
    for (size_t i = 0; i < reply.size(); ++i) {
        const auto & step = reply[i];
        check_step<std::runtime_error>(step, checkpoint, checkpoint.model_name + " reply step " + std::to_string(i + 1));
        put_i32(payload, step.token);
        for (const int32_t code : step.codes) {
            put_i32(payload, code);
        }
    }

    auto artifact = runtime::make_voice_artifact(
        runtime::ArtifactKind::AcousticTokens,
        checkpoint.reply_id,
        std::move(payload),
        {{"format", checkpoint.format},
         {"codebooks", std::to_string(checkpoint.codebooks)},
         {"audio_vocab_size", std::to_string(checkpoint.audio_vocab_size)},
         {"text_vocab_size", std::to_string(checkpoint.text_vocab_size)},
         {"steps", std::to_string(reply.size())},
         {"ended", ended ? "true" : "false"}});
    artifact.meta.insert(checkpoint.checkpoint_tags.begin(), checkpoint.checkpoint_tags.end());
    return artifact;
}

std::vector<ReplyStep> decode_reply(const HistoryFormat & checkpoint, const runtime::VoiceArtifact & artifact) {
    if (artifact.id != checkpoint.reply_id) {
        throw runtime::InvalidRequestError(checkpoint.model_name + " artifact " + artifact.id + " is not an " + checkpoint.reply_id + " artifact");
    }

    const auto where = checkpoint.model_name + " " + checkpoint.reply_id + " artifact";
    return reply_payload(artifact, checkpoint, where, reply_steps(artifact, checkpoint, where));
}

void require_conversation_room(const HistoryFormat & checkpoint, int64_t prompt_steps, bool at_least, int64_t max_tokens) {
    // Written so that nothing overflows, as max_tokens can be any int64.
    if (prompt_steps > checkpoint.context_steps || max_tokens > checkpoint.context_steps - prompt_steps) {
        throw runtime::CapacityError(checkpoint.model_name + " conversation needs " + std::string(at_least ? "at least " : "") + std::to_string(prompt_steps) +
                                     " prompt steps plus max_tokens=" + std::to_string(max_tokens) + ", more than the " +
                                     std::to_string(checkpoint.context_steps) +
                                     " a conversation may take; leave out its oldest turns, or lower max_tokens");
    }
}

std::vector<ConversationTurn> decode_conversation(const HistoryFormat & checkpoint, const std::vector<runtime::VoiceArtifact> & artifacts, int64_t max_tokens) {
    std::vector<ConversationTurn> turns;
    bool awaiting_reply = false;
    int64_t steps = 0;  // the replies' so far, never over checkpoint.context_steps
    for (size_t i = 0; i < artifacts.size(); ++i) {
        const auto & artifact = artifacts[i];
        const auto name = "input artifact " + std::to_string(i + 1) + " (" + artifact.id + ")";
        if (artifact.id == checkpoint.question_id) {
            if (awaiting_reply) {
                throw runtime::InvalidRequestError(checkpoint.model_name + " " + name + " follows a question; each question needs its reply after it");
            }

            turns.push_back({read_question(artifact, checkpoint.model_name + " " + name), {}});
            if (checkpoint.question_steps) {
                const auto count = checkpoint.question_steps(turns.back().question);
                require_conversation_room(checkpoint, steps + count, true, max_tokens);
                steps += count;
            }
            awaiting_reply = true;
        } else if (artifact.id == checkpoint.reply_id) {
            if (!awaiting_reply) {
                throw runtime::InvalidRequestError(checkpoint.model_name + " " + name + " has no question before it");
            }

            // A reply past the room left is never read, however long its
            // payload.
            const auto where = checkpoint.model_name + " " + name;
            const auto count = reply_steps(artifact, checkpoint, where);
            require_conversation_room(checkpoint, steps + count, true, max_tokens);
            steps += count;
            turns.back().reply = reply_payload(artifact, checkpoint, where, count);
            awaiting_reply = false;
        } else {
            throw runtime::InvalidRequestError(checkpoint.model_name + " s2s takes " + std::string(checkpoint.question_id) + " and " + checkpoint.reply_id +
                                               " artifacts only, not " + name);
        }
    }

    if (awaiting_reply) {
        throw runtime::InvalidRequestError(checkpoint.model_name + " input artifact " + std::to_string(artifacts.size()) + " (" + checkpoint.question_id +
                                           ") has no reply after it; the new question goes in audio_input");
    }

    return turns;
}

}  // namespace engine::chat
