// Conversations: the prompt of a turn after earlier ones, which must be the
// sequence liquid-audio's ChatState holds, the artifacts a request carries
// the earlier turns in, and the S2S session, offline and streaming, on a
// synthetic package.
#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/community_models/lfm2_audio/chat.h"
#include "engine/community_models/lfm2_audio/session.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
namespace runtime = engine::runtime;
using engine::test::require;
using engine::test::require_eq;

// The message of the CapacityError `fn` throws.
template <typename Fn>
std::string capacity_message(Fn && fn, const std::string & label) {
    try {
        fn();
    } catch (const runtime::CapacityError & error) {
        return error.what();
    }

    require(false, label + " must throw CapacityError");
    return {};
}

// `fn` throws CapacityError, with `needle` in its message.
template <typename Fn>
void require_capacity_error(Fn && fn, const std::string & needle, const std::string & label) {
    const auto message = capacity_message(fn, label);
    require(message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
}

// `fn` throws InvalidRequestError, which a server answers with 400, with
// `needle` in its message: a request the caller has to fix.
template <typename Fn>
void require_request_error(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
    } catch (const runtime::InvalidRequestError & error) {
        const std::string message = error.what();
        require(message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
        return;
    }

    require(false, label + " must throw InvalidRequestError");
}

// `fn` throws, with `needle` in its message, but not InvalidRequestError: a
// fault of the code that called it, not of a request.
template <typename Fn>
void require_internal_error(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
    } catch (const runtime::InvalidRequestError & error) {
        require(false, label + " threw InvalidRequestError: " + error.what());
    } catch (const std::exception & error) {
        const std::string message = error.what();
        require(message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
        return;
    }

    require(false, label + " must throw");
}

lfm2::Lfm2TextVocabulary text_vocabulary() {
    const auto vocab = lfm2_audio_test::byte_level_vocabulary();
    lfm2::Lfm2TextVocabulary out;
    out.tokens = vocab.tokens;
    out.token_types = vocab.types;
    out.merges = vocab.merges;
    out.pre_tokenizer = "lfm2";
    return out;
}

int32_t id_of(const lfm2::Lfm2TextVocabulary & vocab, const std::string & token) {
    for (size_t i = 0; i < vocab.tokens.size(); ++i) {
        if (vocab.tokens[i] == token) {
            return static_cast<int32_t>(i);
        }
    }

    throw std::runtime_error("no token " + token);
}

// The prompt's token strings, space-separated, with <audio> and <frame> where
// the audio and the frames go.
std::string spell(const lfm2::Lfm2TextVocabulary & vocab, const lfm2::Lfm2Prompt & prompt) {
    std::string out;
    size_t audio = 0;
    size_t frame = 0;
    for (size_t i = 0; i < prompt.input_ids.size(); ++i) {
        std::string token = vocab.tokens.at(static_cast<size_t>(prompt.input_ids[i]));
        if (audio < prompt.audio_positions.size() && prompt.audio_positions[audio] == static_cast<int32_t>(i)) {
            token = "<audio>";
            ++audio;
        } else if (frame < prompt.frame_positions.size() && prompt.frame_positions[frame] == static_cast<int32_t>(i)) {
            token = "<frame>";
            ++frame;
        }

        out += (out.empty() ? "" : " ") + token;
    }

    return out;
}

lfm2::Lfm2ReplyStep text(int32_t token) {
    return {token, {}};
}

lfm2::Lfm2ReplyStep frame(std::vector<int32_t> codes) {
    return {-1, std::move(codes)};
}

const std::string kSystem = "Be brief.";
const std::string kOpening = "<|startoftext|> <|im_start|> s y s t e m Ċ B e Ġ b r i e f . <|im_end|> Ċ <|im_start|> u s e r Ċ";
const std::string kAnswer = "<|im_end|> Ċ <|im_start|> assistant Ċ";
const std::string kNextUser = "<|im_end|> Ċ <|im_start|> u s e r Ċ";

// The first turn is the prompt S2S has always had.
void test_first_turn() {
    const auto vocab = text_vocabulary();
    const lfm2::Lfm2TextTokenizer tokenizer(vocab);
    const auto prompt = lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {}, 3);
    const auto spoken = lfm2::make_lfm2_spoken_prompt(tokenizer, kSystem).with_audio(3);
    require(prompt.input_ids == spoken.input_ids && prompt.audio_positions == spoken.audio_positions, "the first turn is the spoken prompt");
    require(prompt.frame_positions.empty() && prompt.frame_codes.empty(), "the first turn has no frames");
    require_eq(spell(vocab, prompt), kOpening + " <audio> <audio> <audio> " + kAnswer, "the first turn");
}

// ChatState after the README's turns: the system prompt once; each reply step
// by step, its ids as generated (here " A" as two byte tokens, which the
// tokenizer would merge into one), the end-of-audio frame and a frame with
// end-of-audio in another codebook kept; <|im_end|>\n after each reply, then
// the next user turn.
void test_later_turns() {
    const auto vocab = text_vocabulary();
    const lfm2::Lfm2TextTokenizer tokenizer(vocab);
    const int32_t end = 8;
    const std::vector<lfm2::Lfm2ReplyStep> first = {text(id_of(vocab, "Ġ")), text(id_of(vocab, "A")), text(id_of(vocab, "<|text_end|>")),
                                                    frame({1, 2}), frame({3, end}), frame({end, end})};
    // Cut off at max_tokens: no <|text_end|> and no end-of-audio frame.
    const std::vector<lfm2::Lfm2ReplyStep> second = {text(id_of(vocab, "o")), frame({4, 5}), text(id_of(vocab, "k"))};

    const auto one = lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{2, first}}, 1);
    require_eq(spell(vocab, one),
        kOpening + " <audio> <audio> " + kAnswer + " Ġ A <|text_end|> <frame> <frame> <frame> " + kNextUser + " <audio> " + kAnswer,
        "one earlier turn");
    require(one.frame_codes == std::vector<int32_t>{1, 2, 3, end, end, end}, "the frames' codes, in order");

    const auto two = lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{2, first}, {1, second}}, 2);
    require_eq(spell(vocab, two),
        kOpening + " <audio> <audio> " + kAnswer + " Ġ A <|text_end|> <frame> <frame> <frame> " + kNextUser + " <audio> " + kAnswer +
            " o <frame> k " + kNextUser + " <audio> <audio> " + kAnswer,
        "two earlier turns");
    require(two.frame_codes == std::vector<int32_t>{1, 2, 3, end, end, end, 4, 5}, "both replies' codes, in order");
    require_eq(two.audio_positions.size(), size_t{5}, "every question's audio");

    // The steps besides the questions' audio, counted without the prompt.
    const auto text_steps = [&](const lfm2::Lfm2Prompt & prompt) {
        return static_cast<int64_t>(prompt.input_ids.size() - prompt.audio_positions.size());
    };
    const auto first_turn = lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {}, 3);
    std::vector<lfm2::Lfm2ConversationTurn> asked(2);
    asked[0].reply = first;
    asked[1].reply = second;
    require_eq(lfm2::lfm2_chat_prompt_text_steps(tokenizer, kSystem, {}), text_steps(first_turn), "a first turn's text steps");
    const std::vector<lfm2::Lfm2ConversationTurn> asked_once(asked.begin(), asked.begin() + 1);
    require_eq(lfm2::lfm2_chat_prompt_text_steps(tokenizer, kSystem, asked_once), text_steps(one), "text steps after one turn");
    require_eq(lfm2::lfm2_chat_prompt_text_steps(tokenizer, kSystem, asked), text_steps(two), "text steps after two turns");

    // An empty reply still closes its turn.
    require_eq(spell(vocab, lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{1, {}}}, 1)),
        kOpening + " <audio> " + kAnswer + " " + kNextUser + " <audio> " + kAnswer, "an empty reply");

    require_internal_error([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {}, 0); }, "needs its question's audio",
        "a question without audio");
    require_internal_error([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{0, first}}, 1); }, "needs its question's audio",
        "an earlier question without audio");
    require_internal_error([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{1, {frame({1, 2}), frame({1, 2, 3})}}}, 1); },
        "one code per codebook", "frames of different sizes");
    require_internal_error([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{1, {lfm2::Lfm2ReplyStep{}}}}, 1); },
        "neither a token nor codes", "a step with nothing in it");
}

const lfm2::Lfm2ReplyCheckpoint kCheckpoint{"en", 2, 9, 265};

std::vector<lfm2::Lfm2ReplyStep> sample_reply() {
    return {text(76), text(6), frame({0, 7}), frame({3, 8}), frame({8, 8})};
}

bool same_steps(const std::vector<lfm2::Lfm2ReplyStep> & a, const std::vector<lfm2::Lfm2ReplyStep> & b) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].token != b[i].token || a[i].codes != b[i].codes) {
            return false;
        }
    }

    return true;
}

std::vector<uint8_t> bytes_of(const runtime::VoiceArtifact & artifact) {
    std::vector<uint8_t> out;
    for (const auto byte : artifact.payload) {
        out.push_back(static_cast<uint8_t>(byte));
    }

    return out;
}

void test_reply_artifact() {
    const auto reply = sample_reply();
    const auto artifact = lfm2::make_lfm2_reply_artifact(reply, true, kCheckpoint);
    require(artifact.kind == runtime::ArtifactKind::AcousticTokens && artifact.id == "lfm2_audio.reply", "kind and id");
    const std::unordered_map<std::string, std::string> meta = {{"format", "lfm2_audio.reply/1"}, {"language", "en"}, {"codebooks", "2"},
        {"audio_vocab_size", "9"}, {"text_vocab_size", "265"}, {"steps", "5"}, {"ended", "true"}};
    require(artifact.meta == meta, "meta");
    // Little-endian int32: 76, 6, then -1 and two codes per frame.
    const std::vector<uint8_t> head = {76, 0, 0, 0, 6, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 7, 0, 0, 0};
    const auto payload = bytes_of(artifact);
    require(payload.size() == 4 * (2 + 3 * 3) && std::equal(head.begin(), head.end(), payload.begin()), "payload");
    require(same_steps(lfm2::read_lfm2_reply_artifact(artifact, kCheckpoint), reply), "the reply reads back");
    require_eq(lfm2::make_lfm2_reply_artifact(reply, false, kCheckpoint).meta.at("ended"), std::string("false"), "a cut reply");
    require(lfm2::read_lfm2_reply_artifact(lfm2::make_lfm2_reply_artifact({}, false, kCheckpoint), kCheckpoint).empty(), "an empty reply");

    // Only replies the checkpoint could have written. Writing one it could
    // not is the model's fault; reading one is the request's.
    require_internal_error([&] { (void)lfm2::make_lfm2_reply_artifact({frame({1, 2, 3})}, true, kCheckpoint); }, "one code per codebook",
        "writing a frame of three codes");
    require_internal_error([&] { (void)lfm2::make_lfm2_reply_artifact({frame({1, 9})}, true, kCheckpoint); }, "audio code 9",
        "writing a code outside the codebook");
    require_internal_error([&] { (void)lfm2::make_lfm2_reply_artifact({text(265)}, true, kCheckpoint); }, "token id 265",
        "writing a token outside the vocabulary");

    const auto rejects = [&](const std::function<void(runtime::VoiceArtifact &)> & edit, const std::string & needle, const std::string & label) {
        auto changed = artifact;
        edit(changed);
        require_request_error([&] { (void)lfm2::read_lfm2_reply_artifact(changed, kCheckpoint); }, needle, label);
    };
    rejects([](auto & a) { a.id = "lfm2_audio.question"; }, "not an lfm2_audio.reply artifact", "another id");
    rejects([](auto & a) { a.kind = runtime::ArtifactKind::Custom; }, "kind acoustic_tokens", "another kind");
    rejects([](auto & a) { a.meta["format"] = "lfm2_audio.reply/2"; }, "format lfm2_audio.reply/2; this build reads lfm2_audio.reply/1",
        "another format version");
    rejects([](auto & a) { a.meta.erase("format"); }, "no meta format", "no format");
    rejects([](auto & a) { a.meta["language"] = "ja"; }, "comes from a ja checkpoint, and this one is en", "another language");
    rejects([](auto & a) { a.meta["codebooks"] = "8"; }, "has codebooks 8, and this checkpoint 2", "other codebooks");
    rejects([](auto & a) { a.meta["audio_vocab_size"] = "2049"; }, "audio_vocab_size 2049", "another codebook size");
    rejects([](auto & a) { a.meta["text_vocab_size"] = "65536"; }, "text_vocab_size 65536", "another vocabulary");
    rejects([](auto & a) { a.meta["codebooks"] = "two"; }, "not a whole number", "codebooks in words");
    rejects([](auto & a) { a.meta["steps"] = "4"; }, "holds more than the 4 steps its meta says", "fewer steps in the meta");
    rejects([](auto & a) { a.meta["steps"] = "6"; }, "holds 5 steps, and its meta says 6", "more steps in the meta");
    rejects([](auto & a) { a.meta["steps"] = "-5"; }, "not a whole number", "a negative step count");
    rejects([](auto & a) { a.meta["ended"] = "yes"; }, "ended must be true or false", "ended neither true nor false");
    rejects([](auto & a) { a.meta.erase("ended"); }, "no meta ended", "no ended");
    rejects([](auto & a) { a.meta["mime"] = "application/octet-stream"; }, "has meta mime", "an unknown meta key");
    rejects([](auto & a) { a.meta["format"] = "lfm2_audio.reply/2"; a.meta["mime"] = "application/octet-stream"; }, "format lfm2_audio.reply/2",
        "another format version with other meta");
    rejects([](auto & a) { a.payload.resize(a.payload.size() - 4); }, "ends inside step 5", "a payload cut inside a frame");
    rejects([](auto & a) { a.payload.resize(a.payload.size() - 1); }, "not whole int32 values", "a payload cut inside a value");
    rejects([](auto & a) { a.payload[0] = std::byte{0xfe}; a.payload[1] = a.payload[2] = a.payload[3] = std::byte{0xff}; }, "token id -2",
        "a negative token other than the frame marker");
    rejects([](auto & a) { a.payload[1] = std::byte{1}; }, "token id 332, outside the 265-token vocabulary", "a token outside the vocabulary");
    rejects([](auto & a) { a.payload[16] = std::byte{9}; }, "audio code 9, outside the codebook of 9", "a code outside the codebook");
    rejects([](auto & a) { a.payload[12] = std::byte{0xff}; a.payload[13] = a.payload[14] = a.payload[15] = std::byte{0xff}; },
        "audio code -1", "a negative code");
}

// A WAV file of 16-bit samples.
std::vector<std::byte> wav_bytes(int sample_rate, int channels, const std::vector<int16_t> & samples) {
    std::vector<std::byte> out;
    const auto put = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) {
            out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xffu));
        }
    };
    const auto tag = [&](const char * text) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::byte>(text[i]));
        }
    };
    const auto data = static_cast<uint32_t>(samples.size() * 2);
    tag("RIFF");
    put(36 + data, 4);
    tag("WAVE");
    tag("fmt ");
    put(16, 4);
    put(1, 2);
    put(static_cast<uint32_t>(channels), 2);
    put(static_cast<uint32_t>(sample_rate), 4);
    put(static_cast<uint32_t>(sample_rate * channels * 2), 4);
    put(static_cast<uint32_t>(channels * 2), 2);
    put(16, 2);
    tag("data");
    put(data, 4);
    for (const int16_t sample : samples) {
        put(static_cast<uint16_t>(sample), 2);
    }

    return out;
}

runtime::VoiceArtifact question(std::vector<int16_t> samples, int sample_rate = 16000, int channels = 1) {
    return runtime::make_voice_artifact(runtime::ArtifactKind::Custom, "lfm2_audio.question", wav_bytes(sample_rate, channels, samples));
}

// The room a conversation's prompt and max_tokens have, and what they need
// past it.
void test_conversation_room() {
    lfm2::require_lfm2_conversation_room(8192 - 1024, false, 1024);
    lfm2::require_lfm2_conversation_room(0, true, 8192);
    const auto over = [](int64_t prompt_steps, bool at_least, int64_t max_tokens, const std::string & needle, const std::string & label) {
        require_capacity_error([&] { lfm2::require_lfm2_conversation_room(prompt_steps, at_least, max_tokens); }, needle, label);
    };
    over(8192 - 1023, false, 1024,
        "LFM2-Audio conversation needs 7169 prompt steps plus max_tokens=1024, more than the 8192 a conversation may take; leave out its "
        "oldest turns, or lower max_tokens",
        "one step over");
    over(8192 - 1023, true, 1024, "needs at least 7169 prompt steps", "one step over, counting");
    over(9000, false, 1, "needs 9000 prompt steps plus max_tokens=1", "a prompt over the limit alone");
    over(10, false, std::numeric_limits<int64_t>::max(), "plus max_tokens=9223372036854775807", "the largest max_tokens");
}

constexpr int64_t kMaxTokens = 1024;

void test_conversation() {
    const auto reply = lfm2::make_lfm2_reply_artifact(sample_reply(), true, kCheckpoint);
    const auto cut = lfm2::make_lfm2_reply_artifact({text(77), frame({1, 1})}, false, kCheckpoint);
    require(lfm2::read_lfm2_conversation({}, kCheckpoint, kMaxTokens).empty(), "no artifacts, no earlier turns");

    const auto turns =
        lfm2::read_lfm2_conversation({question({100, -200, 300}), reply, question({4, 5, 6, 7}, 24000, 2), cut}, kCheckpoint, kMaxTokens);
    require_eq(turns.size(), size_t{2}, "two earlier turns");
    require(turns[0].question.sample_rate == 16000 && turns[0].question.channels == 1 &&
                turns[0].question.samples == std::vector<float>{100.0f / 32768, -200.0f / 32768, 300.0f / 32768},
        "the first question's samples");
    require(turns[1].question.sample_rate == 24000 && turns[1].question.channels == 2 && turns[1].question.samples.size() == 4,
        "the second question as its WAV has it");
    require(same_steps(turns[0].reply, sample_reply()) && same_steps(turns[1].reply, {text(77), frame({1, 1})}), "the replies");

    const auto rejects = [&](const std::vector<runtime::VoiceArtifact> & artifacts, const std::string & needle, const std::string & label) {
        require_request_error([&] { (void)lfm2::read_lfm2_conversation(artifacts, kCheckpoint, kMaxTokens); }, needle, label);
    };
    const auto q = question({1, 2});
    rejects({reply}, "input artifact 1 (lfm2_audio.reply) has no question before it", "a reply first");
    rejects({q}, "input artifact 1 (lfm2_audio.question) has no reply after it; the new question goes in audio_input", "a question alone");
    rejects({q, q, reply}, "input artifact 2 (lfm2_audio.question) follows a question", "two questions in a row");
    rejects({q, reply, reply}, "input artifact 3 (lfm2_audio.reply) has no question before it", "two replies in a row");
    rejects({q, reply, q}, "input artifact 3 (lfm2_audio.question) has no reply after it", "a last question without a reply");
    auto other = q;
    other.id = "voice.state";
    rejects({other}, "takes lfm2_audio.question and lfm2_audio.reply artifacts only, not input artifact 1 (voice.state)", "an unknown artifact");
    rejects({q, reply, other, reply}, "not input artifact 3 (voice.state)", "an unknown artifact between turns");

    auto wrong_kind = q;
    wrong_kind.kind = runtime::ArtifactKind::AcousticTokens;
    rejects({wrong_kind, reply}, "input artifact 1 (lfm2_audio.question) must be of kind custom", "a question of another kind");
    auto not_wav = q;
    not_wav.payload.resize(10);
    rejects({not_wav, reply}, "input artifact 1 (lfm2_audio.question) does not read as a WAV file: invalid WAV RIFF header",
        "a question that is not a WAV");
    rejects({question({}), reply}, "input artifact 1 (lfm2_audio.question) does not read as a WAV file: incomplete WAV file",
        "a WAV without samples");
    auto ja = reply;
    ja.meta["language"] = "ja";
    rejects({q, ja}, "input artifact 2 (lfm2_audio.reply) comes from a ja checkpoint", "a reply of another checkpoint");

    // Replies are counted from their meta, and one past the room max_tokens
    // leaves is not read: this payload would not read.
    const auto capacity = [&](const std::vector<runtime::VoiceArtifact> & artifacts, int64_t max_tokens, const std::string & needle,
                              const std::string & label) {
        require_capacity_error([&] { (void)lfm2::read_lfm2_conversation(artifacts, kCheckpoint, max_tokens); }, needle, label);
    };
    auto long_reply = reply;
    long_reply.meta["steps"] = "8000";
    long_reply.payload.resize(2);
    capacity({q, long_reply}, kMaxTokens, "needs at least 8000 prompt steps plus max_tokens=1024, more than the 8192", "a reply too long to read");
    require_request_error([&] { (void)lfm2::read_lfm2_conversation({q, long_reply}, kCheckpoint, 1); }, "payload is 2 bytes, not whole int32 values",
        "the same reply with room for it");
    // The replies add up: 5 steps each, with room for 9.
    require_eq(lfm2::read_lfm2_conversation({q, reply}, kCheckpoint, 8192 - 9).size(), size_t{1}, "one reply in the room");
    capacity({q, reply, q, reply}, 8192 - 9, "needs at least 10 prompt steps plus max_tokens=8183", "two replies past the room");
    capacity({q, reply}, std::numeric_limits<int64_t>::max(), "needs at least 5 prompt steps plus max_tokens=9223372036854775807",
        "the largest max_tokens");
}

// The S2S package. Its backbone's layers are random, but they write only to
// dimensions kLayerDims and up, and so do the audio embeddings, so what a
// step's own input puts below them reaches the text logits and the
// depthformer unchanged:
// - dimensions 0-3 hold the text chain \n -> h -> i -> <|text_end|>, each
//   link scoring the next highest (as in lfm2_audio_session_test), and no
//   other token has an embedding, so every reply writes "hi";
// - the depthformer's first codebook picks code 1 after <|text_end|>, and
//   after a frame whose first code is c the code at the next angle, which
//   dimensions 5 and 6 hold, until end-of-audio after code kSpokenFrames;
// - the end-of-audio frame sets dimension 4, which only <|im_end|> reads, so
//   the reply ends there: 3 text steps and kSpokenFrames + 1 frames.
// The second codebook is sampled from the layers' dimensions and never picks
// end-of-audio, so every frame but the last speaks. The end-of-audio frame
// also sets dimension 7, which a head of the last attention layer averages
// over the conversation so far into kHistoryOut, which the second codebook
// reads: only earlier replies hold that frame, so earlier turns change its
// codes.
namespace s2s {

constexpr int64_t kTextEndDim = 3;
constexpr int64_t kEndDim = 4;
constexpr int64_t kCosDim = 5;
constexpr int64_t kSinDim = 6;
constexpr int64_t kHistoryDim = 7;
constexpr int64_t kLayerDims = 8;
constexpr int64_t kHistoryOut = 15;
constexpr int32_t kEnd = 8;  // end-of-audio, after codes 0-7
constexpr int64_t kSpokenFrames = 7;
constexpr int64_t kReplySteps = 3 + kSpokenFrames + 1;
constexpr int64_t kSamplesPerFrame = 1920;
constexpr double kAngle = 40.0 * 3.14159265358979323846 / 180.0;
const std::string kModel = "Model-F16.gguf";

float & at(lfm2_audio_test::Tensor & tensor, int64_t row, int64_t col) {
    return tensor.values[static_cast<size_t>(row * tensor.shape.back() + col)];
}

void write_package(const std::filesystem::path & root) {
    lfm2_audio_test::BackboneShape shape;
    shape.context = 16384;  // room for max_tokens past what a conversation may take
    const int64_t d = shape.hidden;
    const auto vocab = lfm2_audio_test::byte_level_vocabulary();
    auto backbone = lfm2_audio_test::backbone_tensors(shape, static_cast<int64_t>(vocab.tokens.size()), lfm2_audio_test::random_fill(21));
    for (auto & [name, tensor] : backbone) {
        if (name.find("attn_output") != std::string::npos || name.find("out_proj") != std::string::npos ||
            name.find("ffn_down") != std::string::npos) {
            std::fill(tensor.values.begin(), tensor.values.begin() + kLayerDims * tensor.shape.back(), 0.0f);
        }
    }

    auto & table = backbone.at("token_embd.weight");
    std::fill(table.values.begin(), table.values.end(), 0.0f);
    const auto set = [&](const std::string & token, int64_t dim, float value) { at(table, lfm2_audio_test::token_id(vocab, token), dim) = value; };
    float scale = 0.25f;
    const std::vector<std::string> chain = {"Ċ", "h", "i", "<|text_end|>"};
    for (size_t k = 0; k < chain.size(); ++k, scale *= 4.0f) {
        if (k > 0) {
            set(chain[k], static_cast<int64_t>(k) - 1, scale);
        }

        set(chain[k], static_cast<int64_t>(k), scale);
    }

    set("<|im_end|>", kEndDim, 4.0f);

    // Query heads 0 and 1 of blk.3 share a key head of zeros, so they average
    // their values over every position so far, and their only value is
    // kHistoryDim.
    const int64_t head = shape.head_dim();
    auto & keys = backbone.at("blk.3.attn_k.weight");
    auto & values = backbone.at("blk.3.attn_v.weight");
    auto & output = backbone.at("blk.3.attn_output.weight");
    for (int64_t row = 0; row < head; ++row) {
        for (int64_t col = 0; col < d; ++col) {
            at(keys, row, col) = 0.0f;
            at(values, row, col) = row == 0 && col == kHistoryDim ? 1.0f : 0.0f;
        }
    }

    for (int64_t row = 0; row < d; ++row) {
        for (int64_t col = 0; col < 2 * head; ++col) {
            at(output, row, col) = row == kHistoryOut && col == 0 ? 400.0f : 0.0f;
        }
    }

    lfm2_audio_test::write_backbone(root / kModel, shape, vocab, backbone);

    // Frames come back in through the mmproj's a.position_embd.
    lfm2_audio_test::Random random(23);
    lfm2_audio_test::DepthformerShape depth;
    lfm2_audio_test::EncoderShape encoder;
    encoder.output = d;
    auto mmproj = lfm2_audio_test::encoder_tensors(encoder, lfm2_audio_test::random_fill(11, 0.2f));
    // Audio embeddings take the layers' dimensions only, like everything else
    // but the inputs above.
    for (const char * name : {"mm.a.mlp.3.weight", "mm.a.mlp.3.bias"}) {
        auto & adapter = mmproj.at(name);
        const int64_t row_size = adapter.shape.size() == 1 ? 1 : adapter.shape.back();
        std::fill(adapter.values.begin(), adapter.values.begin() + kLayerDims * row_size, 0.0f);
    }

    lfm2_audio_test::Tensor frames{{depth.codebooks * depth.audio_vocab, d}, {}};
    frames.values.assign(static_cast<size_t>(depth.codebooks * depth.audio_vocab * d), 0.0f);
    for (int64_t codebook = 0; codebook < depth.codebooks; ++codebook) {
        for (int32_t code = 0; code <= kEnd; ++code) {
            const int64_t row = codebook * depth.audio_vocab + code;
            if (code == kEnd) {
                at(frames, row, kEndDim) = 1.0f;
                at(frames, row, kHistoryDim) = 30.0f;
                continue;
            }

            for (int64_t dim = kLayerDims; dim < d; ++dim) {
                at(frames, row, dim) = random.uniform(1.0f);
            }

            if (codebook == 0) {
                at(frames, row, kCosDim) = static_cast<float>(std::cos(code * kAngle));
                at(frames, row, kSinDim) = static_cast<float>(std::sin(code * kAngle));
            }
        }
    }

    mmproj["a.position_embd.weight"] = frames;
    lfm2_audio_test::write_mmproj(root / ("mmproj-" + kModel), encoder, mmproj);

    // An identity depthformer: zero layers, so each codebook's logits come
    // from its slice of depth_linear and the code before it. Depth dimension
    // 0 reads <|text_end|>, 1 and 2 the angle, 3-6 the layers' dimensions
    // (3 mostly kHistoryOut), and 7 is constant.
    lfm2_audio_test::DetokenizerShape detokenizer;
    detokenizer.lfm.context = 4096;
    auto vocoder = lfm2_audio_test::vocoder_tensors(depth, d, detokenizer, [](const std::string & name, size_t count) {
        return std::vector<float>(count, name.find("norm") != std::string::npos ? 1.0f : 0.0f);
    });
    auto & linear = vocoder.at("depth_linear.weight");
    at(linear, 0, kTextEndDim) = 1.0f;
    at(linear, 1, kCosDim) = 1.0f;
    at(linear, 2, kSinDim) = 1.0f;
    const int64_t second = depth.hidden;  // the second codebook's rows
    for (int64_t row = 3; row < 7; ++row) {
        for (int64_t col = kLayerDims; col < d; ++col) {
            at(linear, second + row, col) = random.uniform(4.0f);
        }
    }

    at(linear, second + 3, kHistoryOut) = 10.0f;
    vocoder.at("depth_linear.bias").values[static_cast<size_t>(second + 7)] = 16.0f;
    auto & first_logits = vocoder.at("depth_embeddings.0.to_logits.weight");
    at(first_logits, 1, 0) = 50.0f;
    for (int32_t code = 1; code <= kSpokenFrames; ++code) {
        const int32_t next = code == kSpokenFrames ? kEnd : code + 1;
        at(first_logits, next, 1) = static_cast<float>(50.0 * std::cos(code * kAngle));
        at(first_logits, next, 2) = static_cast<float>(50.0 * std::sin(code * kAngle));
    }

    auto & first_codes = vocoder.at("depth_embeddings.0.embedding.weight");
    auto & second_logits = vocoder.at("depth_embeddings.1.to_logits.weight");
    for (int32_t code = 0; code <= kEnd; ++code) {
        for (int64_t dim = 3; dim < 7; ++dim) {
            at(first_codes, code, dim) = random.uniform(1.0f);
            at(second_logits, code, dim) = code == kEnd ? 0.0f : random.uniform(1.5f);
        }

        at(second_logits, code, 7) = code == kEnd ? -4.0f : 2.0f;
    }

    auto & code_embedding = vocoder.at("emb.emb.weight");
    code_embedding.values = random.uniform(code_embedding.values.size(), 0.3f);
    lfm2_audio_test::write_vocoder(root / ("vocoder-" + kModel), depth, vocoder, 4, 3);

    // Small log-magnitudes keep the speech near full scale.
    auto detokenizer_weights = lfm2_audio_test::detokenizer_tensors(detokenizer, lfm2_audio_test::random_fill(41));
    for (const char * name : {"dense_2.weight", "dense_2.bias"}) {
        for (auto & value : detokenizer_weights.at(name).values) {
            value *= 0.1f;
        }
    }

    lfm2_audio_test::write_detokenizer(root / ("tokenizer-" + kModel), detokenizer, detokenizer_weights);
}

struct Package {
    std::filesystem::path root;
    Package() : root(lfm2_audio_test::fresh_directory("audiocpp_lfm2_audio_chat_test")) { write_package(root); }
    ~Package() { std::filesystem::remove_all(root); }
};

std::unique_ptr<runtime::IVoiceTaskSession> open(
    const Package & package,
    runtime::VoiceTaskKind task = runtime::VoiceTaskKind::SpeechToSpeech,
    runtime::RunMode mode = runtime::RunMode::Offline,
    std::unordered_map<std::string, std::string> session_options = {}) {
    runtime::ModelLoadRequest load;
    load.model_path = package.root;
    load.family_hint = "lfm2_audio";
    runtime::SessionOptions options;
    options.backend = {engine::core::BackendType::Cpu, 0, 2};
    options.options = std::move(session_options);
    auto session = lfm2::make_lfm2_audio_loader()->load(load)->create_task_session({task, mode}, options);
    session->prepare({});
    return session;
}

runtime::IOfflineVoiceTaskSession & offline(runtime::IVoiceTaskSession & session) {
    return dynamic_cast<runtime::IOfflineVoiceTaskSession &>(session);
}

runtime::IStreamingVoiceTaskSession & streaming(runtime::IVoiceTaskSession & session) {
    return dynamic_cast<runtime::IStreamingVoiceTaskSession &>(session);
}

// Starts a stream for `request` and sends its audio in chunks of 1600
// frames.
void send_question(runtime::IStreamingVoiceTaskSession & s2s, const runtime::TaskRequest & request) {
    s2s.start_stream(request);
    const auto & input = *request.audio_input;
    const auto chunk_samples = static_cast<size_t>(1600 * input.channels);
    for (size_t start = 0; start < input.samples.size(); start += chunk_samples) {
        runtime::AudioChunk chunk;
        chunk.sample_rate = input.sample_rate;
        chunk.channels = input.channels;
        chunk.samples.assign(input.samples.begin() + static_cast<std::ptrdiff_t>(start),
            input.samples.begin() + static_cast<std::ptrdiff_t>(std::min(input.samples.size(), start + chunk_samples)));
        (void)s2s.process_audio_chunk(chunk);
    }
}

// A question, and the same samples as an lfm2_audio.question artifact: 16-bit
// samples, so that the WAV holds exactly the audio the turn was asked with.
struct Question {
    runtime::AudioBuffer audio;
    runtime::VoiceArtifact artifact;
};

Question question(double seconds, double hz, int sample_rate = 16000, int channels = 1) {
    std::vector<int16_t> pcm;
    const auto frames = static_cast<size_t>(std::llround(seconds * sample_rate));
    for (size_t i = 0; i < frames; ++i) {
        const double value = 0.3 * std::sin(2.0 * 3.14159265358979323846 * hz * static_cast<double>(i) / sample_rate);
        for (int c = 0; c < channels; ++c) {
            pcm.push_back(static_cast<int16_t>(std::lround(value * 32767.0)));
        }
    }

    Question out;
    out.audio = {sample_rate, channels, {}};
    for (const int16_t sample : pcm) {
        out.audio.samples.push_back(static_cast<float>(sample) / 32768.0f);
    }

    out.artifact = runtime::make_voice_artifact(runtime::ArtifactKind::Custom, "lfm2_audio.question", wav_bytes(sample_rate, channels, pcm));
    return out;
}

runtime::TaskRequest ask(const Question & q, const std::string & seed, std::vector<runtime::VoiceArtifact> history = {}, bool return_codes = true) {
    runtime::TaskRequest request;
    request.audio_input = q.audio;
    request.options["seed"] = seed;
    if (return_codes) {
        request.options["return_codes"] = "true";
    }

    request.input_artifacts = std::move(history);
    return request;
}

struct Reply {
    std::string text;
    std::vector<float> audio;
    std::vector<runtime::VoiceArtifact> artifacts;

    [[nodiscard]] const runtime::VoiceArtifact & artifact() const {
        require(artifacts.size() == 1, "the result has one artifact, not " + std::to_string(artifacts.size()));
        return artifacts.front();
    }
};

Reply reply_of(const runtime::TaskResult & result) {
    require(result.text_output.has_value() && result.audio_output.has_value() && result.audio_output->sample_rate == 24000,
        "a reply has text and 24 kHz audio");
    require(!result.artifact_output.has_value(), "the reply artifact is one of the output artifacts");
    return {result.text_output->text, result.audio_output->samples, result.output_artifacts};
}

bool same(const Reply & a, const Reply & b) {
    if (a.text != b.text || a.audio != b.audio || a.artifacts.size() != b.artifacts.size()) {
        return false;
    }

    for (size_t i = 0; i < a.artifacts.size(); ++i) {
        if (a.artifacts[i].payload != b.artifacts[i].payload || a.artifacts[i].meta != b.artifacts[i].meta) {
            return false;
        }
    }

    return true;
}

const lfm2::Lfm2ReplyCheckpoint kPackage{"en", 2, 9, 265};

std::vector<lfm2::Lfm2ReplyStep> steps_of(const Reply & reply) {
    return lfm2::read_lfm2_reply_artifact(reply.artifact(), kPackage);
}

// "hi", <|text_end|>, the frames that speak and the end-of-audio frame.
void require_whole_reply(const Reply & reply, const std::string & label) {
    const auto vocab = lfm2_audio_test::byte_level_vocabulary();
    require_eq(reply.text, std::string("hi"), label + " text");
    require_eq(reply.audio.size(), static_cast<size_t>(kSpokenFrames * kSamplesPerFrame), label + " speech");
    const auto steps = steps_of(reply);
    require_eq(steps.size(), static_cast<size_t>(kReplySteps), label + " steps");
    require(steps[0].token == lfm2_audio_test::token_id(vocab, "h") && steps[1].token == lfm2_audio_test::token_id(vocab, "i") &&
                steps[2].token == lfm2_audio_test::token_id(vocab, "<|text_end|>"),
        label + " starts with its text");
    for (int64_t frame = 0; frame < kSpokenFrames; ++frame) {
        const auto & codes = steps[static_cast<size_t>(3 + frame)].codes;
        require(codes.size() == 2 && codes[0] == frame + 1 && codes[1] >= 0 && codes[1] < kEnd, label + " frame " + std::to_string(frame));
    }

    require(steps.back().codes == std::vector<int32_t>{kEnd, kEnd}, label + " keeps the end-of-audio frame");
    require_eq(reply.artifact().meta.at("ended"), std::string("true"), label + " ended");
}

// The second codebook's codes, which carry what the conversation changes.
std::vector<int32_t> second_codes(const Reply & reply) {
    std::vector<int32_t> out;
    for (const auto & step : steps_of(reply)) {
        if (!step.codes.empty()) {
            out.push_back(step.codes[1]);
        }
    }

    return out;
}

// The steps of a second turn's prompt, after `first`, other than the
// questions' audio.
int64_t text_steps_after(const Reply & first) {
    const auto vocab = text_vocabulary();
    const lfm2::Lfm2TextTokenizer tokenizer(vocab);
    std::vector<lfm2::Lfm2ConversationTurn> asked(1);
    asked[0].reply = steps_of(first);
    return lfm2::lfm2_chat_prompt_text_steps(tokenizer, lfm2::kLfm2ChatSystemPrompt, asked);
}

// What the session writes to std::cerr while it lives.
class CapturedStderr {
public:
    CapturedStderr() : previous_(std::cerr.rdbuf(text_.rdbuf())) {}
    ~CapturedStderr() { std::cerr.rdbuf(previous_); }
    CapturedStderr(const CapturedStderr &) = delete;
    CapturedStderr & operator=(const CapturedStderr &) = delete;

    [[nodiscard]] std::string text() const { return text_.str(); }

private:
    std::ostringstream text_;
    std::streambuf * previous_;
};

struct Turns {
    Question q1 = question(1.0, 440.0);
    Question q2 = question(0.7, 660.0, 24000, 2);  // resampled and mixed down, in history too
    Question q3 = question(0.5, 300.0);
    Reply plain;   // turn 1 without return_codes
    Reply first;   // turn 1
    Reply second;  // turn 2, after turn 1
    Reply third;   // turn 3, after both

    [[nodiscard]] std::vector<runtime::VoiceArtifact> after_first() const { return {q1.artifact, first.artifact()}; }
    [[nodiscard]] std::vector<runtime::VoiceArtifact> after_second() const {
        return {q1.artifact, first.artifact(), q2.artifact, second.artifact()};
    }
};

// Turn by turn, each with the history the turns before returned. A turn's
// reply depends on its request alone, so each repeats to the bit, in any
// order, and the first turn is the reply a request without history gets.
Turns test_offline_turns(const Package & package) {
    auto session = open(package);
    auto & s2s = offline(*session);
    Turns turns;
    turns.plain = reply_of(s2s.run(ask(turns.q1, "7", {}, false)));
    require(turns.plain.artifacts.empty(), "no reply artifact without return_codes");
    require_eq(turns.plain.text, std::string("hi"), "turn 1 text");

    turns.first = reply_of(s2s.run(ask(turns.q1, "7")));
    require(turns.first.text == turns.plain.text && turns.first.audio == turns.plain.audio, "return_codes leaves the reply as it is");
    require_whole_reply(turns.first, "turn 1");
    const auto & meta = turns.first.artifact().meta;
    require(meta.at("format") == "lfm2_audio.reply/1" && meta.at("language") == "en" && meta.at("codebooks") == "2" &&
                meta.at("audio_vocab_size") == "9" && meta.at("text_vocab_size") == "265" && meta.at("steps") == std::to_string(kReplySteps),
        "the reply artifact's meta");

    turns.second = reply_of(s2s.run(ask(turns.q2, "9", turns.after_first())));
    require_whole_reply(turns.second, "turn 2");
    const auto alone = reply_of(s2s.run(ask(turns.q2, "9")));
    require_whole_reply(alone, "turn 2 without history");
    const auto join_codes = [](const std::vector<int32_t> & codes) {
        std::string out;
        for (const int32_t code : codes) {
            out += (out.empty() ? "" : ",") + std::to_string(code);
        }

        return out;
    };
    require(second_codes(alone) != second_codes(turns.second) && alone.audio != turns.second.audio,
        "the earlier turn changes the reply: " + join_codes(second_codes(alone)) + " alone, " + join_codes(second_codes(turns.second)) +
            " after turn 1");

    turns.third = reply_of(s2s.run(ask(turns.q3, "11", turns.after_second())));
    require_whole_reply(turns.third, "turn 3");

    require(same(reply_of(s2s.run(ask(turns.q2, "9", turns.after_first()))), turns.second), "turn 2 again");
    require(same(reply_of(s2s.run(ask(turns.q1, "7", {}, false))), turns.plain), "turn 1 again, after later turns");
    require(same(reply_of(s2s.run(ask(turns.q3, "11", turns.after_second()))), turns.third), "turn 3 again");
    auto fresh = open(package);
    require(same(reply_of(offline(*fresh).run(ask(turns.q2, "9", turns.after_first()))), turns.second), "turn 2 in a fresh session");

    // A reply cut off at max_tokens is returned as it is, not ended, and
    // replays as history like any other.
    auto cut_request = ask(turns.q2, "9", turns.after_first());
    cut_request.options["max_tokens"] = "6";
    Reply cut;
    {
        const CapturedStderr captured;
        cut = reply_of(s2s.run(cut_request));
        require(captured.text().find("reached max_tokens=6") != std::string::npos, "the cut reply warns: " + captured.text());
    }

    require(dynamic_cast<const lfm2::Lfm2AudioChatSession &>(*session).reached_max_tokens(), "the session reports the cut");
    const auto cut_steps = steps_of(cut);
    require(cut_steps.size() == 6 && cut.artifact().meta.at("steps") == "6" && cut.artifact().meta.at("ended") == "false",
        "the cut reply's artifact holds its 6 steps, not ended");
    require_whole_reply(reply_of(s2s.run(ask(turns.q3, "11", {turns.q1.artifact, turns.first.artifact(), turns.q2.artifact, cut.artifact()}))),
        "a turn after a cut reply");
    return turns;
}

// Streamed, a turn returns the reply artifact the offline turn returns, and
// its audio up to the detokenizer's arithmetic. A stream finished early
// returns the steps so far, not ended.
void test_streamed_turns(const Package & package, const Turns & turns) {
    auto session = open(package, runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Streaming);
    auto & s2s = streaming(*session);
    const auto stream = [&](const runtime::TaskRequest & request, size_t max_events = std::numeric_limits<size_t>::max()) {
        send_question(s2s, request);
        Reply events;
        for (size_t i = 0; i < max_events; ++i) {
            auto event = s2s.next_stream_event();
            if (!event.has_value()) {
                break;
            }

            if (event->partial_text.has_value()) {
                events.text += event->partial_text->text;
            }

            if (event->audio_output.has_value()) {
                events.audio.insert(events.audio.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
            }
        }

        auto finished = reply_of(s2s.finish_stream());
        require(finished.text == events.text && finished.audio == events.audio, "a stream's result is its events");
        return finished;
    };
    const auto close_to = [](const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() != b.size()) {
            return false;
        }

        double difference = 0.0;
        double energy = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            difference += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
            energy += static_cast<double>(b[i]) * b[i];
        }

        return std::sqrt(difference / energy) < 1e-3;
    };

    const auto second = stream(ask(turns.q2, "9", turns.after_first()));
    require(second.text == turns.second.text && second.artifact().payload == turns.second.artifact().payload &&
                second.artifact().meta == turns.second.artifact().meta,
        "streamed turn 2 returns the offline reply artifact");
    require(close_to(second.audio, turns.second.audio), "streamed turn 2 speaks the offline audio");
    require(same(stream(ask(turns.q2, "9", turns.after_first())), second), "streamed turn 2 again");

    const auto first = stream(ask(turns.q1, "7", {}, false));
    require(first.artifacts.empty() && first.text == turns.plain.text && close_to(first.audio, turns.plain.audio),
        "streamed turn 1 without return_codes");
    const auto third = stream(ask(turns.q3, "11", turns.after_second()));
    require(third.artifact().payload == turns.third.artifact().payload, "streamed turn 3 returns the offline reply artifact");

    // The first event carries the text and the first frame, the second the
    // next frame.
    const auto early = stream(ask(turns.q2, "9", turns.after_first()), 2);
    const auto early_steps = steps_of(early);
    const auto all_steps = steps_of(turns.second);
    require(early_steps.size() == 5 && early.artifact().meta.at("ended") == "false", "a stream finished early has the steps so far, not ended");
    for (size_t i = 0; i < early_steps.size(); ++i) {
        require(early_steps[i].token == all_steps[i].token && early_steps[i].codes == all_steps[i].codes,
            "the early steps start the reply, step " + std::to_string(i));
    }

    const auto unstarted = stream(ask(turns.q2, "9", turns.after_first()), 0);
    require(steps_of(unstarted).empty() && unstarted.artifact().meta.at("ended") == "false", "a stream finished before its reply");

    auto unknown = ask(turns.q2, "9", turns.after_first());
    unknown.input_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::Custom, "voice.state", "x"));
    require_request_error([&] { s2s.start_stream(unknown); }, "not input artifact 3 (voice.state)", "a stream with an unknown artifact");

    // So does a conversation whose replies and markup alone pass the limit.
    const auto text_steps = text_steps_after(turns.first);
    auto over = ask(turns.q2, "9", turns.after_first());
    over.options["max_tokens"] = std::to_string(8192 - text_steps + 1);
    require_capacity_error([&] { s2s.start_stream(over); }, "needs at least " + std::to_string(text_steps) + " prompt steps",
        "a stream past the limit");
}

void test_rejects(const Package & package, const Turns & turns) {
    auto session = open(package);
    auto & s2s = offline(*session);
    const auto rejects = [&](runtime::TaskRequest request, const std::string & needle, const std::string & label) {
        require_request_error([&] { (void)s2s.run(request); }, needle, label);
    };

    auto unknown = ask(turns.q2, "9", turns.after_first());
    unknown.input_artifacts.insert(unknown.input_artifacts.begin(), runtime::make_text_artifact(runtime::ArtifactKind::Custom, "voice.state", "x"));
    rejects(unknown, "takes lfm2_audio.question and lfm2_audio.reply artifacts only, not input artifact 1 (voice.state)", "an unknown artifact");
    rejects(ask(turns.q2, "9", {turns.first.artifact(), turns.q1.artifact}), "input artifact 1 (lfm2_audio.reply) has no question before it",
        "a turn out of order");
    auto ja = turns.first.artifact();
    ja.meta["language"] = "ja";
    rejects(ask(turns.q2, "9", {turns.q1.artifact, ja}), "comes from a ja checkpoint, and this one is en", "a reply of the Japanese checkpoint");
    auto maybe = ask(turns.q2, "9");
    maybe.options["return_codes"] = "maybe";
    rejects(maybe, "return_codes must be true/false", "return_codes=maybe");

    // A conversation's prompt and max_tokens may take 8192 steps, checked as
    // soon as the request shows they pass it: on the replies, on the text
    // with its markup, after each earlier question is encoded, and on the
    // whole prompt. What each error says the prompt needs sets the next
    // max_tokens, up to a turn exactly at the limit.
    const auto with_max_tokens = [&](int64_t max_tokens) {
        auto request = ask(turns.q2, "9", turns.after_first());
        request.options["max_tokens"] = std::to_string(max_tokens);
        return request;
    };
    const auto needs = [&](int64_t max_tokens, const std::string & how, const std::string & label) {
        const std::string prefix = "LFM2-Audio conversation needs " + how;
        const std::string suffix = " prompt steps plus max_tokens=" + std::to_string(max_tokens) +
                                   ", more than the 8192 a conversation may take; leave out its oldest turns, or lower max_tokens";
        try {
            (void)s2s.run(with_max_tokens(max_tokens));
        } catch (const runtime::CapacityError & error) {
            const std::string message = error.what();
            const bool shaped = message.size() > prefix.size() + suffix.size() && message.rfind(prefix, 0) == 0 &&
                                message.compare(message.size() - suffix.size(), suffix.size(), suffix) == 0;
            const auto number = shaped ? message.substr(prefix.size(), message.size() - prefix.size() - suffix.size()) : std::string();
            require(shaped && number.find_first_not_of("0123456789") == std::string::npos, label + ": " + message);
            return static_cast<int64_t>(std::stoll(number));
        }

        require(false, label + " must throw CapacityError");
        return int64_t{0};
    };
    require_eq(needs(8192, "at least ", "the replies past the limit"), kReplySteps, "what the replies need");
    (void)needs(std::numeric_limits<int64_t>::max(), "at least ", "the largest max_tokens");
    const auto text_steps = text_steps_after(turns.first);
    require_eq(needs(8192 - text_steps + 1, "at least ", "the text past the limit"), text_steps, "what the text needs");
    const auto encoded = needs(8192 - text_steps, "at least ", "the earlier question past the limit");
    require(encoded > text_steps, "the earlier question's audio counts");
    const auto prompt = needs(8192 - encoded, "", "the prompt past the limit");
    require(prompt > encoded, "the new question's audio counts");
    require_whole_reply(reply_of(s2s.run(with_max_tokens(8192 - prompt))), "a turn at the limit");

    // A first turn keeps the context's limit.
    auto long_first = ask(turns.q1, "7");
    long_first.options["max_tokens"] = "8192";
    require_whole_reply(reply_of(s2s.run(long_first)), "a first turn with max_tokens=8192");

    // Earlier questions are held to the one-pass limit too.
    auto limited = open(package, runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Offline, {{"lfm2_audio.max_pass_seconds", "1"}});
    const auto long_question = question(1.5, 440.0);
    try {
        (void)offline(*limited).run(ask(turns.q2, "9", {long_question.artifact, turns.first.artifact()}));
        require(false, "an earlier question over the limit must throw");
    } catch (const runtime::CapacityError & error) {
        const std::string message = error.what();
        require(message.find("the question of earlier turn 1 is 1.5 s; leave that turn out") != std::string::npos,
            "an earlier question over the limit: " + message);
    }

    // History is for S2S.
    auto asr = open(package, runtime::VoiceTaskKind::Asr);
    auto heard = ask(turns.q1, "7", {}, false);
    heard.options.erase("seed");
    heard.options["return_codes"] = "true";
    require_request_error([&] { (void)offline(*asr).run(heard); }, "ASR does not take request option return_codes", "ASR with return_codes");
    heard.options.erase("return_codes");
    heard.input_artifacts = turns.after_first();
    require_request_error([&] { (void)offline(*asr).run(heard); }, "ASR takes no conversation history, so no lfm2_audio.question artifact",
        "ASR with history");

    auto tts = open(package, runtime::VoiceTaskKind::Tts);
    runtime::TaskRequest spoken;
    spoken.text_input = runtime::Transcript{"hi", "en"};
    spoken.options["return_codes"] = "true";
    require_request_error([&] { (void)offline(*tts).run(spoken); }, "TTS does not take request option return_codes", "TTS with return_codes");
    spoken.options.clear();
    spoken.input_artifacts = {turns.first.artifact()};
    require_request_error([&] { (void)offline(*tts).run(spoken); }, "TTS takes no conversation history, so no lfm2_audio.reply artifact",
        "TTS with history");
}

// A turn run to the end, offline or streamed in chunks of 1600 frames.
Reply run_turn(runtime::IVoiceTaskSession & session, runtime::RunMode mode, const runtime::TaskRequest & request) {
    if (mode == runtime::RunMode::Offline) {
        return reply_of(offline(session).run(request));
    }

    auto & s2s = streaming(session);
    send_question(s2s, request);
    while (s2s.next_stream_event().has_value()) {
    }

    return reply_of(s2s.finish_stream());
}

const lfm2::Lfm2AudioChatSession & chat(const runtime::IVoiceTaskSession & session) {
    return dynamic_cast<const lfm2::Lfm2AudioChatSession &>(session);
}

// A later turn's prompt starts with the turn before's. The session gives
// the earlier questions the encoder output it kept from the request before,
// and the backbone restores the prefill blocks the turn before ran, which
// must leave each turn's bytes those of a fresh session. Questions of 10, 12,
// 14 and 6 s take the prompts past 256 steps at turn 2 and past 512 at
// turn 3, so turn 3 resumes from turn 2's first block and turn 4 from turn
// 3's first two; turn 1 runs one-shot and keeps no blocks. The session keeps
// one conversation, so another one in between replaces what it kept.
void test_turns_reuse(const Package & package, runtime::RunMode mode) {
    const std::string how = mode == runtime::RunMode::Offline ? "offline" : "streamed";
    auto session = open(package, runtime::VoiceTaskKind::SpeechToSpeech, mode, {{"lfm2_audio.conversation_cache_slots", "1"}});
    const auto fresh = [&](const runtime::TaskRequest & request) {
        auto other = open(package, runtime::VoiceTaskKind::SpeechToSpeech, mode);
        return run_turn(*other, mode, request);
    };
    const std::vector<Question> questions = {question(10.0, 220.0), question(12.0, 330.0, 24000, 2), question(14.0, 495.0), question(6.0, 770.0)};
    const std::vector<int64_t> resumed = {0, 0, 256, 512};
    std::vector<runtime::VoiceArtifact> history;
    std::vector<Reply> replies;
    runtime::TaskRequest last;
    for (size_t turn = 0; turn < questions.size(); ++turn) {
        const auto request = ask(questions[turn], std::to_string(20 + turn), history);
        const std::string label = how + " turn " + std::to_string(turn + 1);
        auto reply = run_turn(*session, mode, request);
        require_eq(chat(*session).resumed_prefill_steps(), resumed[turn], label + ": prefill steps resumed");
        require_eq(chat(*session).reused_questions(), static_cast<int64_t>(turn), label + ": earlier questions reused");
        require_whole_reply(reply, label);
        require(same(reply, fresh(request)), label + " gives a fresh session's bytes");
        history.push_back(questions[turn].artifact);
        history.push_back(reply.artifact());
        replies.push_back(std::move(reply));
        last = request;
    }

    // Another conversation in between replaces what the session kept, and
    // the first one's next turn, run cold, gives the same bytes.
    const auto other_first = question(9.0, 610.0);
    const auto other = run_turn(*session, mode, ask(other_first, "40"));
    const auto other_second = ask(question(8.0, 905.0), "41", {other_first.artifact, other.artifact()});
    require(same(run_turn(*session, mode, other_second), fresh(other_second)), how + ": another conversation's turn 2");
    require_eq(chat(*session).reused_questions(), int64_t{1}, how + ": the other conversation's question reused");
    require(same(run_turn(*session, mode, last), replies.back()), how + ": turn 4 after another conversation");
    require_eq(chat(*session).resumed_prefill_steps(), int64_t{0}, how + ": turn 4 after another conversation resumes nothing");
    require_eq(chat(*session).reused_questions(), int64_t{0}, how + ": turn 4 after another conversation reuses nothing");
    require(same(run_turn(*session, mode, last), replies.back()), how + ": turn 4 again");
    require_eq(chat(*session).resumed_prefill_steps(), int64_t{512}, how + ": turn 4 again resumes");
    require_eq(chat(*session).reused_questions(), int64_t{3}, how + ": turn 4 again reuses");

    // Another conversation's first turn alone runs one-shot: it replaces the
    // kept questions but leaves the kept blocks, so the first conversation's
    // next turn encodes its questions again and still resumes.
    require(same(run_turn(*session, mode, ask(other_first, "40")), other), how + ": another conversation's turn 1 again");
    require(same(run_turn(*session, mode, last), replies.back()), how + ": turn 4 after another conversation's first turn");
    require_eq(chat(*session).resumed_prefill_steps(), int64_t{512}, how + ": turn 4 after another conversation's first turn resumes");
    require_eq(chat(*session).reused_questions(), int64_t{0}, how + ": turn 4 after another conversation's first turn reuses nothing");

    // Turn 4 over the limit fails as on a fresh session, with the same
    // message, after each earlier question and on the whole prompt, and
    // leaves what the session kept.
    const auto vocab = text_vocabulary();
    const lfm2::Lfm2TextTokenizer tokenizer(vocab);
    std::vector<lfm2::Lfm2ConversationTurn> asked(3);
    for (size_t turn = 0; turn < asked.size(); ++turn) {
        asked[turn].reply = steps_of(replies[turn]);
    }

    auto cold = open(package, runtime::VoiceTaskKind::SpeechToSpeech, mode);
    auto max_tokens = 8192 - lfm2::lfm2_chat_prompt_text_steps(tokenizer, lfm2::kLfm2ChatSystemPrompt, asked);
    for (int check = 1;; ++check) {
        require(check <= 4, how + ": four checks past the text");
        auto over = last;
        over.options["max_tokens"] = std::to_string(max_tokens);
        const std::string label = how + ": turn 4 with max_tokens=" + std::to_string(max_tokens);
        const auto message = capacity_message([&] { (void)run_turn(*session, mode, over); }, label);
        require_eq(message, capacity_message([&] { (void)run_turn(*cold, mode, over); }, label + " on a fresh session"), label);
        const std::string at_least = "LFM2-Audio conversation needs at least ";
        if (message.rfind(at_least, 0) != 0) {
            require_eq(check, 4, how + ": the whole prompt is the fourth check");
            break;
        }

        max_tokens = 8192 - static_cast<int64_t>(std::stoll(message.substr(at_least.size())));
    }

    require(same(run_turn(*session, mode, last), replies.back()), how + ": turn 4 after it went over the limit");
    require_eq(chat(*session).resumed_prefill_steps(), int64_t{512}, how + ": turn 4 after it went over the limit resumes");
    require_eq(chat(*session).reused_questions(), int64_t{3}, how + ": turn 4 after it went over the limit reuses");

    // A request without history or return_codes keeps nothing, and leaves
    // what the session kept.
    const auto plain = ask(question(5.0, 440.0), "50", {}, false);
    require(same(run_turn(*session, mode, plain), fresh(plain)), how + ": a request without history or return_codes");
    require(same(run_turn(*session, mode, last), replies.back()), how + ": turn 4 after a request without history");
    require_eq(chat(*session).resumed_prefill_steps(), int64_t{512}, how + ": turn 4 after a request without history resumes");
    require_eq(chat(*session).reused_questions(), int64_t{3}, how + ": turn 4 after a request without history reuses");
}

// A conversation's questions, the seed of its first turn (one more each turn
// after), and its system prompt, the default when empty.
struct Conversation {
    std::vector<Question> questions;
    int seed = 0;
    std::string system_prompt;
};

// A turn's reply, and the prefill steps and earlier questions the session
// took for it from the turns before.
struct Taken {
    Reply reply;
    int64_t resumed = 0;
    int64_t reused = 0;
};

// Turn `turn` of `c`, after the replies `before` of its turns before.
runtime::TaskRequest turn_request(const Conversation & c, size_t turn, const std::vector<Taken> & before) {
    std::vector<runtime::VoiceArtifact> history;
    for (size_t k = 0; k < turn; ++k) {
        history.push_back(c.questions[k].artifact);
        history.push_back(before[k].reply.artifact());
    }

    auto request = ask(c.questions[turn], std::to_string(c.seed + static_cast<int>(turn)), std::move(history));
    if (!c.system_prompt.empty()) {
        request.text_input = runtime::Transcript{c.system_prompt, "en"};
    }

    return request;
}

// The conversations on one session, taking turns: the first turn of each,
// then the second of each, and so on, each turn after the replies this
// session gave its own conversation.
std::vector<std::vector<Taken>> take_turns(
    runtime::IVoiceTaskSession & session, runtime::RunMode mode, const std::vector<Conversation> & conversations) {
    std::vector<std::vector<Taken>> out(conversations.size());
    for (size_t turn = 0;; ++turn) {
        bool asked = false;
        for (size_t c = 0; c < conversations.size(); ++c) {
            if (turn < conversations[c].questions.size()) {
                const auto request = turn_request(conversations[c], turn, out[c]);
                auto reply = run_turn(session, mode, request);
                out[c].push_back({std::move(reply), chat(session).resumed_prefill_steps(), chat(session).reused_questions()});
                asked = true;
            }
        }

        if (!asked) {
            return out;
        }
    }
}

// lfm2_audio.conversation_cache_slots: an S2S session keeps the questions and
// prefill blocks of that many conversations, so conversations taking turns on
// it each take what they would take on a session of their own, and give the
// same bytes.
void test_conversations_kept(const Package & package, runtime::RunMode mode) {
    const std::string how = mode == runtime::RunMode::Offline ? "offline" : "streamed";
    const auto with_slots = [&](const std::string & slots) {
        return open(package, runtime::VoiceTaskKind::SpeechToSpeech, mode, {{"lfm2_audio.conversation_cache_slots", slots}});
    };
    const auto alone = [&](const Conversation & c) {
        auto session = open(package, runtime::VoiceTaskKind::SpeechToSpeech, mode);
        return take_turns(*session, mode, {c}).front();
    };
    const auto require_replies = [&](const std::vector<Taken> & turns, const std::vector<Taken> & expected, const std::string & label) {
        require_eq(turns.size(), expected.size(), label + " turns");
        for (size_t k = 0; k < turns.size(); ++k) {
            require(same(turns[k].reply, expected[k].reply), label + " turn " + std::to_string(k + 1) + " gives its bytes alone");
        }
    };
    const auto require_taken = [&](const Taken & turn, int64_t resumed, int64_t reused, const std::string & label) {
        require_eq(turn.resumed, resumed, label + ": prefill steps resumed");
        require_eq(turn.reused, reused, label + ": earlier questions reused");
    };

    // A is test_turns_reuse's conversation; B resumes from its turn 3 too.
    const Conversation a{{question(10.0, 220.0), question(12.0, 330.0, 24000, 2), question(14.0, 495.0), question(6.0, 770.0)}, 20, ""};
    const Conversation b{{question(11.0, 260.0), question(13.0, 390.0), question(12.0, 560.0), question(5.0, 840.0)}, 60, ""};
    const auto a_alone = alone(a);
    const auto b_alone = alone(b);
    const std::vector<int64_t> a_resumed = {0, 0, 256, 512};
    for (size_t k = 0; k < a_alone.size(); ++k) {
        require_taken(a_alone[k], a_resumed[k], static_cast<int64_t>(k), how + ": A alone, turn " + std::to_string(k + 1));
    }

    require(b_alone[2].resumed > 0 && b_alone[3].resumed > b_alone[2].resumed, how + ": B alone resumes from its turn 3");

    // With two slots, each turn takes what it takes alone. With one, each
    // conversation's turn drops the other's.
    auto two = with_slots("2");
    const auto both = take_turns(*two, mode, {a, b});
    auto one = with_slots("1");
    const auto pushed = take_turns(*one, mode, {a, b});
    for (size_t i = 0; i < 2; ++i) {
        const auto & expected = i == 0 ? a_alone : b_alone;
        const std::string name = how + (i == 0 ? ": A" : ": B");
        require_replies(both[i], expected, name + " with two slots");
        require_replies(pushed[i], expected, name + " with one slot");
        for (size_t k = 0; k < expected.size(); ++k) {
            const std::string turn = ", turn " + std::to_string(k + 1);
            require_taken(both[i][k], expected[k].resumed, expected[k].reused, name + " with two slots" + turn);
            require_taken(pushed[i][k], 0, 0, name + " with one slot" + turn);
        }
    }

    // A third conversation then drops what was least recently used: its first
    // turn A's questions (a first turn keeps no blocks), its second B's
    // blocks.
    const Conversation c{{question(9.0, 650.0), question(8.0, 930.0)}, 80, ""};
    const auto c_alone = alone(c);
    const auto step = [&](runtime::IVoiceTaskSession & session, const runtime::TaskRequest & request, const Reply & expected,
                          int64_t resumed, int64_t reused, const std::string & label) {
        const auto reply = run_turn(session, mode, request);
        require(same(reply, expected), label + " gives its bytes alone");
        require_taken({reply, chat(session).resumed_prefill_steps(), chat(session).reused_questions()}, resumed, reused, label);
    };
    step(*two, turn_request(c, 0, c_alone), c_alone[0].reply, 0, 0, how + ": C's turn 1");
    step(*two, turn_request(a, 3, a_alone), a_alone[3].reply, 512, 0, how + ": A's turn 4 after C's turn 1");
    step(*two, turn_request(c, 1, c_alone), c_alone[1].reply, 0, 1, how + ": C's turn 2");
    step(*two, turn_request(b, 3, b_alone), b_alone[3].reply, 0, 0, how + ": B's turn 4 after C's turn 2");

    // No slots keep nothing.
    auto none = with_slots("0");
    const auto kept_nothing = take_turns(*none, mode, {a}).front();
    require_replies(kept_nothing, a_alone, how + ": A with no slots");
    for (size_t k = 0; k < kept_nothing.size(); ++k) {
        require_taken(kept_nothing[k], 0, 0, how + ": A with no slots, turn " + std::to_string(k + 1));
    }

    // One system prompt of 300 tokens fills the first block of both
    // conversations. Each still keeps its own blocks after it; B's turn 2
    // takes that block, which A's turn 2 kept.
    std::string system_prompt;
    while (system_prompt.size() < 300) {
        system_prompt += "zebra lamp ";
    }

    system_prompt.resize(300);
    auto long_a = a;
    auto long_b = b;
    long_a.system_prompt = system_prompt;
    long_b.system_prompt = system_prompt;
    const auto long_a_alone = alone(long_a);
    const auto long_b_alone = alone(long_b);
    require(long_a_alone[2].resumed >= 512, how + ": A with a long system prompt resumes more than its first block");
    require_eq(long_b_alone[1].resumed, int64_t{0}, how + ": B with a long system prompt resumes nothing at turn 2");
    auto shared = with_slots("2");
    const auto long_both = take_turns(*shared, mode, {long_a, long_b});
    for (size_t i = 0; i < 2; ++i) {
        const auto & expected = i == 0 ? long_a_alone : long_b_alone;
        const std::string name = how + (i == 0 ? ": A" : ": B") + " with a long system prompt";
        require_replies(long_both[i], expected, name);
        for (size_t k = 0; k < expected.size(); ++k) {
            const auto resumed = i == 1 && k == 1 ? int64_t{256} : expected[k].resumed;
            require_taken(long_both[i][k], resumed, expected[k].reused, name + ", turn " + std::to_string(k + 1));
        }
    }

    // Conversations that ask the same clips, first and at turn 3: each turn
    // still takes the encoder output of all its earlier questions.
    const Conversation same_a{{question(9.0, 350.0), question(12.0, 470.0), question(7.0, 610.0), question(6.0, 720.0)}, 90, ""};
    const Conversation same_b{{question(9.0, 350.0), question(11.0, 530.0), question(7.0, 610.0), question(5.0, 880.0)}, 95, ""};
    const auto same_a_alone = alone(same_a);
    const auto same_b_alone = alone(same_b);
    auto clips = with_slots("2");
    const auto same_both = take_turns(*clips, mode, {same_a, same_b});
    for (size_t i = 0; i < 2; ++i) {
        const auto & expected = i == 0 ? same_a_alone : same_b_alone;
        const std::string name = how + (i == 0 ? ": A" : ": B") + " with shared clips";
        require_replies(same_both[i], expected, name);
        for (size_t k = 0; k < expected.size(); ++k) {
            const std::string label = name + ", turn " + std::to_string(k + 1);
            require_eq(same_both[i][k].reused, static_cast<int64_t>(k), label + ": earlier questions reused");
            require(same_both[i][k].resumed >= expected[k].resumed, label + ": resumes at least what it does alone");
        }
    }

    // Without the option a session keeps four conversations: four taking
    // turns each take what they take alone, and then a fifth one's first turn
    // drops the questions of the least recently used, A's.
    auto defaulted = open(package, runtime::VoiceTaskKind::SpeechToSpeech, mode);
    const std::vector<std::vector<Taken>> four_alone = {a_alone, b_alone, same_a_alone, same_b_alone};
    const auto four = take_turns(*defaulted, mode, {a, b, same_a, same_b});
    for (size_t i = 0; i < four.size(); ++i) {
        const std::string name = how + ": conversation " + std::to_string(i + 1) + " of four without the option";
        require_replies(four[i], four_alone[i], name);
        for (size_t k = 0; k < four_alone[i].size(); ++k) {
            const std::string label = name + ", turn " + std::to_string(k + 1);
            require_eq(four[i][k].reused, four_alone[i][k].reused, label + ": earlier questions reused");
            require(four[i][k].resumed >= four_alone[i][k].resumed, label + ": resumes at least what it does alone");
        }
    }

    step(*defaulted, turn_request(c, 0, c_alone), c_alone[0].reply, 0, 0, how + ": C's turn 1 after four conversations");
    step(*defaulted, turn_request(a, 3, a_alone), a_alone[3].reply, 512, 0, how + ": A's turn 4 after C's turn 1, without the option");
}

// The option's values: a count of at least 0, which ASR and TTS take and
// ignore.
void test_conversation_cache_slots_option(const Package & package) {
    const auto open_with = [&](runtime::VoiceTaskKind task, const std::string & slots) {
        return open(package, task, runtime::RunMode::Offline, {{"lfm2_audio.conversation_cache_slots", slots}});
    };
    lfm2_audio_test::require_throws_with([&] { (void)open_with(runtime::VoiceTaskKind::SpeechToSpeech, "-1"); },
                                         "lfm2_audio.conversation_cache_slots must be non-negative", "a negative slot count");
    lfm2_audio_test::require_throws_with([&] { (void)open_with(runtime::VoiceTaskKind::SpeechToSpeech, "2x"); },
                                         "lfm2_audio.conversation_cache_slots must be an integer", "a slot count that is not a number");
    (void)open_with(runtime::VoiceTaskKind::Asr, "2");
    (void)open_with(runtime::VoiceTaskKind::Tts, "2");
}

}  // namespace s2s

}  // namespace

int main() {
    try {
        test_first_turn();
        test_later_turns();
        test_reply_artifact();
        test_conversation_room();
        test_conversation();
        const s2s::Package package;
        const auto turns = s2s::test_offline_turns(package);
        s2s::test_streamed_turns(package, turns);
        s2s::test_rejects(package, turns);
        s2s::test_turns_reuse(package, runtime::RunMode::Offline);
        s2s::test_turns_reuse(package, runtime::RunMode::Streaming);
        s2s::test_conversations_kept(package, runtime::RunMode::Offline);
        s2s::test_conversations_kept(package, runtime::RunMode::Streaming);
        s2s::test_conversation_cache_slots_option(package);
        std::cout << "lfm2_audio_chat_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_chat_test: " << error.what() << '\n';
        return 1;
    }
}
