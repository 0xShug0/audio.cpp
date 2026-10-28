// LFM3-Audio conversations: the reply artifact a client sends back, the
// request artifacts that carry earlier turns, the room a conversation has,
// and the S2S prompt that replays earlier replies step by step, on a
// synthetic tokenizer.
#include "engine/community_models/lfm3_audio/chat.h"
#include "engine/community_models/lfm3_audio/features.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session.h"
#include "lfm3_audio_test_tokenizer.h"
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

namespace lfm3 = engine::community_models::lfm3_audio;
namespace runtime = engine::runtime;
using engine::test::require;
using engine::test::require_eq;
using lfm3_audio_test::join;
using lfm3_audio_test::require_error;
using lfm3_audio_test::require_ids;
using lfm3_audio_test::require_request_error;

// `fn` throws, but not InvalidRequestError: a fault of the caller's code.
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

lfm3::Lfm3Step text(int32_t token) {
    return {token, {}};
}

lfm3::Lfm3Step frame(std::vector<int32_t> codes) {
    return {-1, std::move(codes)};
}

bool same_steps(const std::vector<lfm3::Lfm3Step> & a, const std::vector<lfm3::Lfm3Step> & b) {
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

const lfm3::Lfm3ReplyCheckpoint kCheckpoint{2, 9, 265};

std::vector<lfm3::Lfm3Step> sample_reply() {
    return {text(76), text(6), frame({0, 7}), frame({3, 8}), frame({8, 8})};
}

void test_reply_artifact() {
    const auto reply = sample_reply();
    const auto artifact = lfm3::make_lfm3_reply_artifact(reply, true, kCheckpoint);
    require(artifact.kind == runtime::ArtifactKind::AcousticTokens && artifact.id == "lfm3_audio.reply", "kind and id");
    const std::unordered_map<std::string, std::string> meta = {{"format", "lfm3_audio.reply/1"}, {"codebooks", "2"}, {"audio_vocab_size", "9"},
        {"text_vocab_size", "265"}, {"steps", "5"}, {"ended", "true"}};
    require(artifact.meta == meta, "meta");
    // Little-endian int32: 76, 6, then -1 and two codes per frame.
    const std::vector<uint8_t> head = {76, 0, 0, 0, 6, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 7, 0, 0, 0};
    std::vector<uint8_t> payload;
    for (const auto byte : artifact.payload) {
        payload.push_back(static_cast<uint8_t>(byte));
    }
    require(payload.size() == 4 * (2 + 3 * 3) && std::equal(head.begin(), head.end(), payload.begin()), "payload");
    require(same_steps(lfm3::read_lfm3_reply_artifact(artifact, kCheckpoint), reply), "the reply reads back");
    require_eq(lfm3::make_lfm3_reply_artifact(reply, false, kCheckpoint).meta.at("ended"), std::string("false"), "a cut reply");
    require(lfm3::read_lfm3_reply_artifact(lfm3::make_lfm3_reply_artifact({}, false, kCheckpoint), kCheckpoint).empty(), "an empty reply");

    // The checkpoint's real sizes round-trip too: 12 codes, end-of-audio 2016.
    const lfm3::Lfm3ReplyCheckpoint real;
    const std::vector<lfm3::Lfm3Step> long_reply = {text(127999), frame(std::vector<int32_t>(12, 2016)), frame({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2015})};
    require(same_steps(lfm3::read_lfm3_reply_artifact(lfm3::make_lfm3_reply_artifact(long_reply, true, real), real), long_reply),
        "a reply of the real checkpoint reads back");

    // Only replies the checkpoint could have written. Writing one it could
    // not is the model's fault; reading one is the request's.
    require_internal_error([&] { (void)lfm3::make_lfm3_reply_artifact({frame({1, 2, 3})}, true, kCheckpoint); }, "one code per codebook",
        "writing a frame of three codes");
    require_internal_error([&] { (void)lfm3::make_lfm3_reply_artifact({frame({1, 9})}, true, kCheckpoint); }, "audio code 9",
        "writing a code outside the codebook");
    require_internal_error([&] { (void)lfm3::make_lfm3_reply_artifact({text(265)}, true, kCheckpoint); }, "token id 265",
        "writing a token outside the vocabulary");
    require_internal_error([&] { (void)lfm3::make_lfm3_reply_artifact({lfm3::Lfm3Step{3, {1, 2}}}, true, kCheckpoint); }, "one code per codebook",
        "writing a frame with a token");

    const auto rejects = [&](const std::function<void(runtime::VoiceArtifact &)> & edit, const std::string & needle, const std::string & label) {
        auto changed = artifact;
        edit(changed);
        require_request_error([&] { (void)lfm3::read_lfm3_reply_artifact(changed, kCheckpoint); }, needle, label);
    };
    rejects([](auto & a) { a.id = "lfm3_audio.question"; }, "not an lfm3_audio.reply artifact", "another id");
    rejects([](auto & a) { a.kind = runtime::ArtifactKind::Custom; }, "kind acoustic_tokens", "another kind");
    rejects([](auto & a) { a.meta["format"] = "lfm3_audio.reply/2"; }, "format lfm3_audio.reply/2; this build reads lfm3_audio.reply/1",
        "another format version");
    rejects([](auto & a) { a.meta.erase("format"); }, "no meta format", "no format");
    rejects([](auto & a) { a.meta["codebooks"] = "8"; }, "has codebooks 8, and this checkpoint 2", "other codebooks");
    rejects([](auto & a) { a.meta["audio_vocab_size"] = "2049"; }, "audio_vocab_size 2049", "another codebook size");
    rejects([](auto & a) { a.meta["text_vocab_size"] = "65536"; }, "text_vocab_size 65536", "another vocabulary");
    rejects([](auto & a) { a.meta["codebooks"] = "two"; }, "not a whole number", "codebooks in words");
    rejects([](auto & a) { a.meta["codebooks"] = "2.0"; }, "not a whole number", "codebooks as a decimal");
    rejects([](auto & a) { a.meta["steps"] = "4"; }, "holds more than the 4 steps its meta says", "fewer steps in the meta");
    rejects([](auto & a) { a.meta["steps"] = "6"; }, "holds 5 steps, and its meta says 6", "more steps in the meta");
    rejects([](auto & a) { a.meta["steps"] = "-5"; }, "not a whole number", "a negative step count");
    rejects([](auto & a) { a.meta["steps"] = "99999999999999999999"; }, "not a whole number", "a step count past int64");
    rejects([](auto & a) { a.meta["ended"] = "yes"; }, "ended must be true or false", "ended neither true nor false");
    rejects([](auto & a) { a.meta.erase("ended"); }, "no meta ended", "no ended");
    rejects([](auto & a) { a.meta["mime"] = "application/octet-stream"; }, "has meta mime", "an unknown meta key");
    rejects([](auto & a) { a.meta["format"] = "lfm3_audio.reply/2"; a.meta["mime"] = "application/octet-stream"; }, "format lfm3_audio.reply/2",
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
    return runtime::make_voice_artifact(runtime::ArtifactKind::Custom, "lfm3_audio.question", wav_bytes(sample_rate, channels, samples));
}

void test_conversation_room() {
    lfm3::require_lfm3_conversation_room(8192 - 1024, 1024);
    lfm3::require_lfm3_conversation_room(0, 8192);
    const auto over = [](int64_t prompt_steps, int64_t max_tokens, const std::string & needle, const std::string & label) {
        require_error<runtime::CapacityError>([&] { lfm3::require_lfm3_conversation_room(prompt_steps, max_tokens); }, needle, label);
    };
    over(8192 - 1023, 1024,
        "LFM3-Audio conversation needs 7169 prompt steps plus max_tokens=1024, more than the 8192 a conversation may take; leave out its "
        "oldest turns, or lower max_tokens",
        "one step over");
    over(9000, 1, "needs 9000 prompt steps plus max_tokens=1", "a prompt over the limit alone");
    over(10, std::numeric_limits<int64_t>::max(), "plus max_tokens=9223372036854775807", "the largest max_tokens");
}

constexpr int64_t kMaxTokens = 1024;

void test_conversation() {
    const auto reply = lfm3::make_lfm3_reply_artifact(sample_reply(), true, kCheckpoint);
    const auto cut = lfm3::make_lfm3_reply_artifact({text(77), frame({1, 1})}, false, kCheckpoint);
    require(lfm3::read_lfm3_conversation({}, kCheckpoint, kMaxTokens).empty(), "no artifacts, no earlier turns");

    const auto turns =
        lfm3::read_lfm3_conversation({question({100, -200, 300}), reply, question({4, 5, 6, 7}, 24000, 2), cut}, kCheckpoint, kMaxTokens);
    require_eq(turns.size(), size_t{2}, "two earlier turns");
    require(turns[0].question.sample_rate == 16000 && turns[0].question.channels == 1 &&
                turns[0].question.samples == std::vector<float>{100.0f / 32768, -200.0f / 32768, 300.0f / 32768},
        "the first question's samples");
    require(turns[1].question.sample_rate == 24000 && turns[1].question.channels == 2 && turns[1].question.samples.size() == 4,
        "the second question as its WAV has it");
    require(same_steps(turns[0].reply, sample_reply()) && same_steps(turns[1].reply, {text(77), frame({1, 1})}), "the replies");

    const auto rejects = [&](const std::vector<runtime::VoiceArtifact> & artifacts, const std::string & needle, const std::string & label) {
        require_request_error([&] { (void)lfm3::read_lfm3_conversation(artifacts, kCheckpoint, kMaxTokens); }, needle, label);
    };
    const auto q = question({1, 2});
    rejects({reply}, "input artifact 1 (lfm3_audio.reply) has no question before it", "a reply first");
    rejects({q}, "input artifact 1 (lfm3_audio.question) has no reply after it; the new question goes in audio_input", "a question alone");
    rejects({q, q, reply}, "input artifact 2 (lfm3_audio.question) follows a question", "two questions in a row");
    rejects({q, reply, reply}, "input artifact 3 (lfm3_audio.reply) has no question before it", "two replies in a row");
    rejects({q, reply, q}, "input artifact 3 (lfm3_audio.question) has no reply after it", "a last question without a reply");
    auto other = q;
    other.id = "lfm2_audio.question";
    rejects({other}, "takes lfm3_audio.question and lfm3_audio.reply artifacts only, not input artifact 1 (lfm2_audio.question)",
        "an LFM2-Audio artifact");
    rejects({q, reply, other, reply}, "not input artifact 3 (lfm2_audio.question)", "an unknown artifact between turns");

    auto wrong_kind = q;
    wrong_kind.kind = runtime::ArtifactKind::AcousticTokens;
    rejects({wrong_kind, reply}, "input artifact 1 (lfm3_audio.question) must be of kind custom", "a question of another kind");
    auto not_wav = q;
    not_wav.payload.resize(10);
    rejects({not_wav, reply}, "input artifact 1 (lfm3_audio.question) does not read as a WAV file", "a question that is not a WAV");
    rejects({question({}), reply}, "input artifact 1 (lfm3_audio.question) does not read as a WAV file", "a WAV without samples");
    auto bad_reply = reply;
    bad_reply.meta["codebooks"] = "12";
    rejects({q, bad_reply}, "input artifact 2 (lfm3_audio.reply) has codebooks 12", "a reply of another checkpoint");

    // Replies are counted from their meta, and one past the room max_tokens
    // leaves is not read: this payload would not read.
    const auto capacity = [&](const std::vector<runtime::VoiceArtifact> & artifacts, int64_t max_tokens, const std::string & needle,
                              const std::string & label) {
        require_error<runtime::CapacityError>([&] { (void)lfm3::read_lfm3_conversation(artifacts, kCheckpoint, max_tokens); }, needle, label);
    };
    auto long_reply = reply;
    long_reply.meta["steps"] = "8000";
    long_reply.payload.resize(2);
    capacity({q, long_reply}, kMaxTokens, "needs at least 8001 prompt steps plus max_tokens=1024, more than the 8192", "a reply too long to read");
    require_request_error([&] { (void)lfm3::read_lfm3_conversation({q, long_reply}, kCheckpoint, 1); }, "payload is 2 bytes, not whole int32 values",
        "the same reply with room for it");
    // The turns add up: 1 audio token and 5 reply steps each, with room for 11.
    require_eq(lfm3::read_lfm3_conversation({q, reply}, kCheckpoint, 8192 - 11).size(), size_t{1}, "one turn in the room");
    capacity({q, reply, q, reply}, 8192 - 11, "needs at least 12 prompt steps plus max_tokens=8181", "two turns past the room");

    // Questions are counted from their length at 16 kHz, before anything is encoded or read further:
    // 300 samples at 1 Hz are 300 s, 3750 tokens, and three of them leave no room even for max_tokens=1.
    const auto slow = question(std::vector<int16_t>(300, 1), 1);
    const auto empty = lfm3::make_lfm3_reply_artifact({}, false, kCheckpoint);
    require_eq(lfm3::read_lfm3_conversation({slow, empty, slow, empty}, kCheckpoint, 1).size(), size_t{2}, "two long questions in the room");
    capacity({slow, empty, slow, empty, slow, empty}, 1, "needs at least 11250 prompt steps plus max_tokens=1", "three long questions past the room");
    capacity({slow, empty, slow, empty, slow, not_wav}, 1, "needs at least 11250 prompt steps", "a question past the room before its reply");
}

// The token count must equal what the encoder makes of lfm3_audio_mono_16k's output.
void test_question_tokens() {
    const auto tokens = [](int sample_rate, int channels, size_t frames) {
        return lfm3::lfm3_question_tokens(runtime::AudioBuffer{sample_rate, channels, std::vector<float>(frames * static_cast<size_t>(channels), 0.1f)});
    };
    require_eq(lfm3::lfm3_question_tokens(runtime::AudioBuffer{16000, 1, {}}), int64_t{0}, "no audio");
    require_eq(lfm3::lfm3_question_tokens(runtime::AudioBuffer{0, 1, {0.1f}}), int64_t{0}, "no sample rate");
    require_eq(tokens(16000, 1, 1), int64_t{1}, "one sample, one feature frame");
    require_eq(tokens(16000, 1, 400 + 160 * 8), int64_t{2}, "nine feature frames");
    require_eq(tokens(16000, 2, 400 + 160 * 7), int64_t{1}, "eight feature frames, in stereo");
    for (const int rate : {8000, 22050, 24000, 44100, 48000}) {
        const size_t frames = static_cast<size_t>(rate) * 3 + 7;  // 3 s and a few samples, not a whole number at 16 kHz
        const runtime::AudioBuffer audio{rate, 1, std::vector<float>(frames, 0.1f)};
        const auto samples = static_cast<int64_t>(lfm3::lfm3_audio_mono_16k(audio).size());
        require_eq(lfm3::lfm3_question_tokens(audio), lfm3::lfm3_audio_tokens(lfm3::lfm3_feature_frames(samples)),
            "the resampled length at " + std::to_string(rate) + " Hz");
    }
}

constexpr int32_t kEoa = 2016;

// The upstream S2S prompt layout (as the chat template renders it), written
// out by hand for the synthetic tokenizer: the system block, then per turn
// the user's audio, the assistant header with <|mixed_start|>, the reply
// steps, <|im_end|> and \n.
void test_s2s_prompt() {
    const lfm3::Lfm3Tokenizer tokenizer(lfm3_audio_test::write_tokenizer(lfm3_audio_test::fresh_directory("lfm3_audio_chat_test_s2s"), true));
    const auto & id = tokenizer.ids();
    const std::vector<int32_t> nl = {id.nl};
    const auto open = join({{id.bos, id.im_start}, tokenizer.encode("system"), nl, tokenizer.encode("Be brief."), {id.im_end}, nl});
    const auto ask = [&](size_t audio) {
        return join({{id.im_start}, tokenizer.encode("user"), nl, std::vector<int32_t>(audio, id.audio), {id.im_end}, nl, {id.im_start},
                     tokenizer.encode("assistant"), nl, {id.mixed_start}});
    };
    const int32_t a = 'a';
    const int32_t b = 'b';
    const std::vector<int32_t> eos_frame(3, kEoa);
    const std::vector<lfm3::Lfm3Step> ended = {text(a), text(id.text_end), frame({1, 2, 3}), frame({4, kEoa, 5}), text(id.im_end)};
    const std::vector<lfm3::Lfm3Step> cut_in_audio = {text(b), frame({6, 7, 8})};
    const std::vector<lfm3::Lfm3Step> cut_after_eos = {text(b), frame(eos_frame)};
    const std::vector<lfm3::Lfm3Step> cut_in_text = {text(a), frame({1, 1, 1}), text(b)};
    const std::vector<lfm3::Lfm3Step> text_only = {text(a), text(b)};

    const auto build = [&](std::vector<int64_t> questions, std::vector<std::vector<lfm3::Lfm3Step>> replies, int64_t audio) {
        return lfm3::make_lfm3_s2s_prompt(tokenizer, "Be brief.", questions, replies, audio, kEoa);
    };

    // A finished turn replays as generated, end-of-audio frame included.
    auto prompt = build({2}, {ended}, 3);
    require_ids(prompt.ids, join({open, ask(2), {a, id.text_end, id.audio_out, id.audio_out, id.im_end}, nl, ask(3)}), "a finished turn");
    require(prompt.frames == std::vector<std::vector<int32_t>>{{1, 2, 3}, {4, kEoa, 5}}, "its frames, in order");

    // Cut in its audio: the clip is closed with an end-of-audio frame, then the turn.
    prompt = build({1}, {cut_in_audio}, 1);
    require_ids(prompt.ids, join({open, ask(1), {b, id.audio_out, id.audio_out, id.im_end}, nl, ask(1)}), "a turn cut in its audio");
    require(prompt.frames == std::vector<std::vector<int32_t>>{{6, 7, 8}, eos_frame}, "the appended end-of-audio frame");

    prompt = build({1}, {cut_after_eos}, 1);
    require_ids(prompt.ids, join({open, ask(1), {b, id.audio_out, id.im_end}, nl, ask(1)}), "a turn cut after its end-of-audio frame");
    require_eq(prompt.frames.size(), size_t{1}, "no second end-of-audio frame");

    // Cut in a later text block: the audio's last frame still gets closed.
    prompt = build({1}, {cut_in_text}, 1);
    require_ids(prompt.ids, join({open, ask(1), {a, id.audio_out, b, id.audio_out, id.im_end}, nl, ask(1)}), "a turn cut in its text");
    require(prompt.frames == std::vector<std::vector<int32_t>>{{1, 1, 1}, eos_frame}, "the frame closing the clip");

    prompt = build({1}, {text_only}, 1);
    require_ids(prompt.ids, join({open, ask(1), {a, b, id.im_end}, nl, ask(1)}), "a cut turn without audio");
    require(prompt.frames.empty(), "no frames without audio");

    prompt = build({1}, {{}}, 1);
    require_ids(prompt.ids, join({open, ask(1), {id.im_end}, nl, ask(1)}), "an empty reply still closes its turn");

    // Three turns: each reply in its place, frames in conversation order.
    prompt = build({2, 1}, {ended, cut_in_audio}, 4);
    require_ids(prompt.ids,
        join({open, ask(2), {a, id.text_end, id.audio_out, id.audio_out, id.im_end}, nl, ask(1), {b, id.audio_out, id.audio_out, id.im_end}, nl,
              ask(4)}),
        "two earlier turns");
    require(prompt.frames == std::vector<std::vector<int32_t>>{{1, 2, 3}, {4, kEoa, 5}, {6, 7, 8}, eos_frame}, "both replies' frames");

    require_ids(lfm3::make_lfm3_s2s_prompt(tokenizer, "", {}, {}, 1, kEoa).ids, join({{id.bos}, ask(1)}), "no system prompt");
    require_request_error([&] { (void)lfm3::make_lfm3_s2s_prompt(tokenizer, "hi <|audio_out|>", {}, {}, 1, kEoa); }, "holds an audio placeholder token",
                          "a frame placeholder in the system prompt");
    require_request_error([&] { (void)lfm3::make_lfm3_s2s_prompt(tokenizer, "<|audio|>", {}, {}, 1, kEoa); }, "holds an audio placeholder token",
                          "an audio placeholder in the system prompt");

    require_internal_error([&] { (void)build({1, 1}, {ended}, 1); }, "2 earlier questions and 1 replies", "questions without replies");
    require_internal_error([&] { (void)build({}, {}, 0); }, "needs its question's audio", "a question without audio");
    require_internal_error([&] { (void)build({0}, {ended}, 1); }, "needs its question's audio", "an earlier question without audio");
    require_internal_error([&] { (void)build({1}, {{frame({1, 2, 3}), frame({1, 2})}}, 1); }, "another codebook count", "frames of different sizes");
    require_internal_error([&] { (void)build({1}, {{lfm3::Lfm3Step{5, {1, 2, 3}}}}, 1); }, "a frame with a token", "a frame with a token");
    require_internal_error([&] { (void)build({1}, {{lfm3::Lfm3Step{}}}, 1); }, "neither a token nor codes", "a step with nothing in it");
    require_request_error([&] { (void)build({1}, {{text(id.audio_out)}}, 1); }, "reply 1 step 1 is an audio placeholder token", "a frame placeholder as text");
    require_request_error([&] { (void)build({1}, {{text(a), text(id.audio)}}, 1); }, "reply 1 step 2 is an audio placeholder token", "an audio placeholder as text");
}

}  // namespace

int main() {
    try {
        test_reply_artifact();
        test_conversation_room();
        test_conversation();
        test_question_tokens();
        test_s2s_prompt();
        std::cout << "lfm3_audio_chat_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_chat_test: " << error.what() << '\n';
        return 1;
    }
}
