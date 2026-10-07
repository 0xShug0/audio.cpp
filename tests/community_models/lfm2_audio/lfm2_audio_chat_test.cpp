// Conversations: the prompt of a turn after earlier ones, which must be the
// sequence liquid-audio's ChatState holds, and the artifacts a request
// carries the earlier turns in.
#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/community_models/lfm2_audio/chat.h"
#include "engine/framework/runtime/errors.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
namespace runtime = engine::runtime;
using engine::test::require;
using engine::test::require_eq;
using lfm2_audio_test::require_throws_with;

// `fn` throws CapacityError, with `needle` in its message.
template <typename Fn>
void require_capacity_error(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
    } catch (const runtime::CapacityError & error) {
        const std::string message = error.what();
        require(message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
        return;
    }

    require(false, label + " must throw CapacityError");
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

    require_throws_with([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {}, 0); }, "needs its question's audio", "a question without audio");
    require_throws_with([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{0, first}}, 1); }, "needs its question's audio",
        "an earlier question without audio");
    require_throws_with([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{1, {frame({1, 2}), frame({1, 2, 3})}}}, 1); },
        "one code per codebook", "frames of different sizes");
    require_throws_with([&] { (void)lfm2::make_lfm2_chat_prompt(tokenizer, kSystem, {{1, {lfm2::Lfm2ReplyStep{}}}}, 1); },
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

    // Only replies the checkpoint could have written.
    require_throws_with([&] { (void)lfm2::make_lfm2_reply_artifact({frame({1, 2, 3})}, true, kCheckpoint); }, "one code per codebook",
        "writing a frame of three codes");
    require_throws_with([&] { (void)lfm2::make_lfm2_reply_artifact({frame({1, 9})}, true, kCheckpoint); }, "audio code 9",
        "writing a code outside the codebook");
    require_throws_with([&] { (void)lfm2::make_lfm2_reply_artifact({text(265)}, true, kCheckpoint); }, "token id 265",
        "writing a token outside the vocabulary");

    const auto rejects = [&](const std::function<void(runtime::VoiceArtifact &)> & edit, const std::string & needle, const std::string & label) {
        auto changed = artifact;
        edit(changed);
        require_throws_with([&] { (void)lfm2::read_lfm2_reply_artifact(changed, kCheckpoint); }, needle, label);
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
        require_throws_with([&] { (void)lfm2::read_lfm2_conversation(artifacts, kCheckpoint, kMaxTokens); }, needle, label);
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
    require_throws_with([&] { (void)lfm2::read_lfm2_conversation({q, long_reply}, kCheckpoint, 1); }, "payload is 2 bytes, not whole int32 values",
        "the same reply with room for it");
    // The replies add up: 5 steps each, with room for 9.
    require_eq(lfm2::read_lfm2_conversation({q, reply}, kCheckpoint, 8192 - 9).size(), size_t{1}, "one reply in the room");
    capacity({q, reply, q, reply}, 8192 - 9, "needs at least 10 prompt steps plus max_tokens=8183", "two replies past the room");
    capacity({q, reply}, std::numeric_limits<int64_t>::max(), "needs at least 5 prompt steps plus max_tokens=9223372036854775807",
        "the largest max_tokens");
}

}  // namespace

int main() {
    try {
        test_first_turn();
        test_later_turns();
        test_reply_artifact();
        test_conversation_room();
        test_conversation();
        std::cout << "lfm2_audio_chat_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_chat_test: " << error.what() << '\n';
        return 1;
    }
}
