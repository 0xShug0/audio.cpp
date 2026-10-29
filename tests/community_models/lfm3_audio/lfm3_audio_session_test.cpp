// The LFM3-Audio sessions end to end on a synthetic package (CPU). Every
// backbone layer writes zero (w2, conv out_proj and o_proj are zero), so each
// position's logits come from its own token embedding alone, and the
// embeddings chain <think> -> "x" -> </think> -> "\n" -> "o" -> "k" ->
// <|im_end|>: greedy text after any of those tokens follows the chain. That
// makes ASR ("ok"), chat and thinking chat predictable while the encoder,
// prefill, decode, depth decoder and codec all run. The codec has 1680 codes
// per codebook, not the checkpoint's 2016, so a replayed cut reply's EOS frame
// must come from the config.
#include "engine/community_models/lfm3_audio/chat.h"
#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/community_models/lfm3_audio/features.h"
#include "engine/community_models/lfm3_audio/session.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "lfm3_audio_test_package.h"
#include "lfm3_audio_test_tokenizer.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
namespace runtime = engine::runtime;
using engine::test::require;
using engine::test::require_eq;
using lfm3_audio_test::require_error;
using lfm3_audio_test::require_throws_with;

const std::unordered_map<std::string, std::string> kF32 = {{"lfm3_audio.weight_type", "f32"}};
const std::vector<std::string> kChain = {"<think>", "x", "</think>", "\xc4\x8a" /* "\n" byte token */, "o", "k", "<|im_end|>"};
const std::filesystem::path kRepoRoot = ENGINE_REPO_ROOT;

lfm3_audio_test::Package write_session_package(const std::vector<std::string> & chain = kChain, const std::string & name = "lfm3_audio_session_test") {
    lfm3_audio_test::PackageShape shape;
    shape.fsq_levels = {8, 7, 6, 5};
    return lfm3_audio_test::write_package(name, shape, [&](lfm3_audio_test::Package & out) {
        for (auto & [tensor_name, tensor] : out.model) {
            if (tensor_name.rfind("lfm.model.", 0) != 0) continue;
            const bool zero = tensor_name.find("feed_forward.w2") != std::string::npos || tensor_name.find("conv.out_proj") != std::string::npos ||
                              tensor_name.find("self_attn.o_proj") != std::string::npos || tensor_name == "lfm.model.embed_tokens.weight";
            if (zero) std::fill(tensor.values.begin(), tensor.values.end(), 0.0f);
            if (tensor_name == "lfm.model.embedding_norm.weight") std::fill(tensor.values.begin(), tensor.values.end(), 1.0f);
        }

        // row(c_k) = 4^k (e_{k-1} + e_k): after c_k, c_{k+1} scores 4^(k+1), c_k 2 * 4^k, c_{k-1} 4^(k-1), others 0.
        auto & table = out.model.at("lfm.model.embed_tokens.weight");
        const int64_t hidden = shape.backbone.hidden;
        float scale = 1.0f;
        for (size_t k = 0; k < chain.size(); ++k, scale *= 4.0f) {
            float * row = table.values.data() + out.ids.at(chain[k]) * hidden;
            if (k > 0) row[k - 1] = scale;
            row[k] = scale;
        }
    });
}

struct Fixture {
    lfm3_audio_test::Package package;
    std::unique_ptr<runtime::ILoadedVoiceModel> model;
};

template <typename Session>
std::unique_ptr<Session> open(const Fixture & f, runtime::VoiceTaskKind kind, runtime::RunMode mode,
                              std::unordered_map<std::string, std::string> session_options = {}) {
    runtime::SessionOptions options;
    options.backend = {engine::core::BackendType::Cpu, 0, 2};
    options.options = std::move(session_options);
    auto session = f.model->create_task_session({kind, mode}, options);
    session->prepare({});
    auto * typed = dynamic_cast<Session *>(session.get());
    require(typed != nullptr, "unexpected session type");
    session.release();
    return std::unique_ptr<Session>(typed);
}

runtime::AudioBuffer tone(double seconds, int sample_rate = 16000) {
    runtime::AudioBuffer audio{sample_rate, 1, {}};
    const auto frames = static_cast<size_t>(std::llround(seconds * sample_rate));
    for (size_t i = 0; i < frames; ++i) {
        audio.samples.push_back(static_cast<float>(0.3 * std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(i) / sample_rate)));
    }
    return audio;
}

runtime::TaskRequest audio_request(runtime::AudioBuffer audio, std::unordered_map<std::string, std::string> options = {}) {
    runtime::TaskRequest out;
    out.audio_input = std::move(audio);
    out.options = std::move(options);
    return out;
}

runtime::TaskRequest text_request(const std::string & text, std::unordered_map<std::string, std::string> options = {}) {
    runtime::TaskRequest out;
    out.text_input = runtime::Transcript{text, ""};
    out.options = std::move(options);
    return out;
}

double rel_diff(const std::vector<float> & a, const std::vector<float> & b) {
    require_eq(a.size(), b.size(), "audio lengths");
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        den += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

template <typename Fn>
std::string cerr_of(Fn && fn) {
    std::ostringstream captured;
    auto * previous = std::cerr.rdbuf(captured.rdbuf());
    try {
        fn();
    } catch (...) {
        std::cerr.rdbuf(previous);
        throw;
    }
    std::cerr.rdbuf(previous);
    return captured.str();
}

std::string text_of(const runtime::TaskResult & result) {
    require(result.text_output.has_value(), "result has text");
    return result.text_output->text;
}

// Returns the events' audio and text, and the finish_stream result.
std::pair<runtime::TaskResult, runtime::TaskResult> stream(runtime::IStreamingVoiceTaskSession & session, const runtime::TaskRequest & request) {
    session.start_stream(request);
    if (request.audio_input.has_value()) {
        const auto & audio = *request.audio_input;
        for (size_t at = 0; at < audio.samples.size(); at += 1600) {
            const auto end = std::min(audio.samples.size(), at + 1600);
            session.process_audio_chunk({audio.sample_rate, 1, static_cast<int64_t>(at), {audio.samples.begin() + at, audio.samples.begin() + end}});
        }
    }
    runtime::TaskResult events;
    events.audio_output = runtime::AudioBuffer{24000, 1, {}};
    events.text_output = runtime::Transcript{"", ""};
    while (auto event = session.next_stream_event()) {
        if (event->audio_output) runtime::append_audio_buffer(*events.audio_output, *event->audio_output);
        if (event->partial_text) events.text_output->text += event->partial_text->text;
    }
    return {events, session.finish_stream()};
}

void test_asr(const Fixture & f) {
    auto asr = open<lfm3::Lfm3AsrSession>(f, runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline);
    require_eq(text_of(asr->run(audio_request(tone(1.0)))), std::string("ok"), "ASR transcript");
    require(!asr->reached_max_tokens(), "ASR ended on <|im_end|>");
    require_eq(text_of(asr->run(audio_request(tone(1.0), {{"language", "en"}}))), std::string("ok"), "ASR language=en");
    // 24 kHz stereo input goes through the 16 kHz mono conversion.
    runtime::AudioBuffer stereo{24000, 2, {}};
    for (const float v : tone(1.0, 24000).samples) stereo.samples.insert(stereo.samples.end(), {v, v});
    require_eq(text_of(asr->run(audio_request(stereo))), std::string("ok"), "ASR 24 kHz stereo");

    const auto cut = asr->run(audio_request(tone(1.0), {{"max_tokens", "1"}}));
    require_eq(text_of(cut), std::string("o"), "ASR cut at max_tokens");
    require(asr->reached_max_tokens(), "ASR reports the cut");
    require_eq(text_of(asr->run(audio_request(tone(1.0), {{"max_tokens", ""}}))), std::string("ok"), "empty max_tokens keeps the default");

    // Chat answers start with "\n", which is stripped; thinking keeps only what follows </think>.
    require_eq(text_of(asr->run(audio_request(tone(1.0), {{"chat", "true"}}))), std::string("ok"), "chat answer");
    require_eq(asr->run(audio_request(tone(1.0), {{"language", "fr"}})).text_output->language, std::string("fr"), "a transcript's language");
    require(asr->run(audio_request(tone(1.0), {{"chat", "true"}, {"language", "fr"}})).text_output->language.empty(),
            "a chat answer names no language");
    require_throws_with([&] { asr->run(audio_request(tone(1.0), {{"chat", "true"}, {"audio_chunk_duration_sec", "0.5"}})); },
                        "audio_chunk_duration_sec must be at least 1", "a bad chunk length with chat");
    require_eq(text_of(asr->run(audio_request(tone(1.0), {{"chat", "true"}, {"enable_thinking", "true"}}))), std::string("ok"),
               "thinking chat answer");
    std::string thought;
    const auto warned = cerr_of([&] {
        thought = text_of(asr->run(audio_request(tone(1.0), {{"chat", "true"}, {"enable_thinking", "true"}, {"max_tokens", "1"}})));
    });
    require_eq(thought, std::string(), "thinking without </think> gives no answer");
    require(asr->reached_max_tokens(), "thinking cut reports max_tokens");
    require(warned.find("without closing </think>, so it is empty; raise max_tokens") != std::string::npos, "thinking cut warning");
    // The answer starts after the first </think>; a later one is part of the answer, as upstream's partition.
    require(lfm3::lfm3_thinking_answer({1, 9, 2, 9, 3}, 9) == std::vector<int32_t>({2, 9, 3}), "answer after the first </think>");
    require(!lfm3::lfm3_thinking_answer({1, 2}, 9).has_value(), "no </think>, no answer");

    require_throws_with([&] { asr->run(audio_request(tone(1.0), {{"enable_thinking", "true"}})); }, "needs chat=true", "thinking without chat");
    require_throws_with([&] { asr->run(audio_request(tone(1.0), {{"language", "xx"}})); }, "does not know language", "unknown language");
    require_throws_with([&] { asr->run(audio_request(tone(1.0), {{"temperature", "0.5"}})); }, "does not take request option temperature",
                        "TTS option in ASR");
    require_throws_with([&] { asr->run(audio_request(tone(1.0), {{"chat", "maybe"}})); }, "chat", "bad bool");
    auto history = audio_request(tone(1.0));
    history.input_artifacts.push_back(runtime::make_voice_artifact(runtime::ArtifactKind::Custom, lfm3::kLfm3QuestionArtifactId, {}));
    require_throws_with([&] { asr->run(history); }, "takes no conversation history", "history in ASR");
    // 8192 steps at most: the prompt plus max_tokens.
    require_error<runtime::CapacityError>([&] { asr->run(audio_request(tone(1.0), {{"max_tokens", "9000"}})); }, "9000", "max_tokens past 8192");

    require_eq(lfm3::lfm3_asr_default_max_tokens(10.0, false, false), int64_t{512}, "short pass default");
    require_eq(lfm3::lfm3_asr_default_max_tokens(100.0, false, false), int64_t{800}, "long pass default scales");
    require_eq(lfm3::lfm3_asr_default_max_tokens(100.0, true, false), int64_t{512}, "chat default");
    require_eq(lfm3::lfm3_asr_default_max_tokens(100.0, true, true), int64_t{2048}, "thinking default");

    // max_pass_duration_sec turns longer single passes away before encoding.
    auto limited = open<lfm3::Lfm3AsrSession>(f, runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline, {{"lfm3_audio.max_pass_duration_sec", "1"}});
    for (const auto & options : std::vector<std::unordered_map<std::string, std::string>>{{{"audio_chunk_mode", "none"}}, {{"chat", "true"}}}) {
        require_error<runtime::CapacityError>([&] { limited->run(audio_request(tone(1.5), options)); }, "max_pass_duration_sec",
                                              "a 1.5 s pass past a 1 s limit");
    }
    // 1 s fixed chunks: the 0.5 s tail joins the second, so two transcripts.
    require_eq(text_of(asr->run(audio_request(tone(2.5), {{"audio_chunk_mode", "fixed"}, {"audio_chunk_duration_sec", "1"}}))),
               std::string("ok ok"), "fixed chunks join their transcripts");
    // A space between chunks in every script that spaces its words; none inside Chinese or Japanese text.
    const auto joined = [](std::initializer_list<const char *> chunks) {
        std::string text;
        for (const char * chunk : chunks) lfm3::append_lfm3_chunk_text(text, chunk);
        return text;
    };
    require_eq(joined({"ok", ", ok.", " "}), std::string("ok, ok."), "ASCII punctuation attaches, a blank chunk adds nothing");
    require_eq(joined({"C'était l'été", "Il pleut"}), std::string("C'était l'été Il pleut"), "French");
    require_eq(joined({"όλα καλά", "ευχαριστώ"}), std::string("όλα καλά ευχαριστώ"), "Greek");
    require_eq(joined({"안녕하세요", "감사합니다"}), std::string("안녕하세요 감사합니다"), "Korean");
    require_eq(joined({"今日は", "晴れです。", "OK"}), std::string("今日は晴れです。OK"), "Japanese");
    require_eq(joined({"ok", "你好"}), std::string("ok你好"), "Chinese after a Latin word");
    // The bundled Silero VAD splits speech, 3 s of silence, then the speech again at the pause: two transcripts.
    auto vad = open<lfm3::Lfm3AsrSession>(f, runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline,
                                          {{"lfm3_audio.vad_model_path", (kRepoRoot / "assets/framework/models/silero_vad").string()}});
    const auto wav = engine::audio::read_wav_f32(kRepoRoot / "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
    runtime::AudioBuffer speech{wav.sample_rate, 1, wav.samples};
    speech.samples.insert(speech.samples.end(), static_cast<size_t>(3 * wav.sample_rate), 0.0f);
    speech.samples.insert(speech.samples.end(), wav.samples.begin(), wav.samples.end());
    require_eq(text_of(vad->run(audio_request(speech, {{"audio_chunk_mode", "vad"}}))), std::string("ok ok"), "VAD chunks split at the pause");

    require_throws_with([&] { open<lfm3::Lfm3AsrSession>(f, runtime::VoiceTaskKind::Asr, runtime::RunMode::Streaming); }, "offline only",
                        "streaming ASR");
    require_throws_with([&] { open<lfm3::Lfm3AsrSession>(f, runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline, {{"lfm3_audio.weight_type", "f8"}}); },
                        "weight_type", "bad weight_type");
    std::cout << "  ASR: transcript, chat, thinking, cuts, options, limits, VAD chunks ok\n";
}

void test_thinking_ends_inside() {
    Fixture f{write_session_package({"<think>", "x", "<|im_end|>"}, "lfm3_audio_session_test_think"), nullptr};
    runtime::ModelLoadRequest load;
    load.model_path = f.package.root;
    f.model = lfm3::make_lfm3_audio_loader()->load(load);
    auto asr = open<lfm3::Lfm3AsrSession>(f, runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline);
    std::string answer = "unset";
    const auto warned = cerr_of([&] { answer = text_of(asr->run(audio_request(tone(1.0), {{"chat", "true"}, {"enable_thinking", "true"}}))); });
    require_eq(answer, std::string(), "an answer that ends inside <think> is empty");
    require(!asr->reached_max_tokens(), "it ended on <|im_end|>, not max_tokens");
    require(warned.find("ended inside <think> without closing </think>, so it is empty") != std::string::npos, "ended-inside warning");
    require(warned.find("max_tokens") == std::string::npos, "the warning does not mention max_tokens");
    std::cout << "  ASR: thinking that ends inside <think> ok\n";
}

void test_tts(const Fixture & f) {
    // F32 (reference) weights: native BF16 CPU kernels round differently once a codec chunk passes ~60 rows,
    // which offline decoding reaches and per-frame streaming does not (~1e-2 rel on these random weights).
    auto offline = open<lfm3::Lfm3TtsSession>(f, runtime::VoiceTaskKind::Tts, runtime::RunMode::Offline, kF32);
    auto streaming = open<lfm3::Lfm3TtsSession>(f, runtime::VoiceTaskKind::Tts, runtime::RunMode::Streaming, kF32);
    const auto request = text_request("the user", {{"seed", "5"}, {"max_tokens", "6"}});
    const auto audio = offline->run(request).audio_output->samples;
    require(!audio.empty(), "TTS audio");
    require(offline->run(request).audio_output->samples == audio, "same seed, same audio");

    for (const char * frames : {"1", "4"}) {
        auto req = request;
        req.options["stream_frames_per_event"] = frames;
        const auto [events, result] = stream(*streaming, req);
        const double rel = rel_diff(events.audio_output->samples, audio);
        require(rel <= 1e-5, "TTS streaming (" + std::string(frames) + " frames per event) vs offline rel " + std::to_string(rel));
        require(result.audio_output->samples == events.audio_output->samples, "finish_stream returns the events' audio");
        require_eq(streaming->reached_max_tokens(), offline->reached_max_tokens(), "streaming and offline agree on the cut");
    }

    auto voiced = request;
    voiced.voice = runtime::VoiceCondition{runtime::VoiceReference{std::nullopt, std::string("german_male")}, std::nullopt};
    require(!offline->run(voiced).audio_output->samples.empty(), "voice german_male");
    require(!offline->run(text_request("the user", {{"seed", "5"}, {"max_tokens", "4"}, {"language", "ja"}})).audio_output->samples.empty(),
            "language default voice");
    voiced.voice->speaker->cached_voice_id = "klingon_female";
    require_throws_with([&] { offline->run(voiced); }, "has no voice klingon_female", "unknown voice");
    voiced.voice->speaker->audio = tone(1.0);
    require_throws_with([&] { offline->run(voiced); }, "does not clone", "reference audio");
    require_throws_with([&] { offline->run(text_request(" ")); }, "requires text_input", "empty text");
    require_throws_with([&] { offline->run(text_request("hi", {{"chat", "true"}})); }, "does not take request option chat", "ASR option in TTS");
    require_throws_with([&] { offline->run(text_request("hi", {{"repetition_penalty", "0"}})); }, "repetition_penalty", "penalty 0");
    // A placeholder in a later chunk fails the request before any chunk is spoken or streamed.
    const auto placeholder = text_request("the user. the user. say <|audio_out|>.", {{"max_tokens", "3"}, {"text_chunk_size", "10"}});
    require_throws_with([&] { offline->run(placeholder); }, "holds an audio placeholder token", "placeholder in TTS text");
    require_throws_with([&] { streaming->start_stream(placeholder); }, "holds an audio placeholder token", "placeholder in streamed TTS text");

    // Long text: one turn per chunk, seeds seed + i.
    const auto chunked = offline->run(text_request("the user. the user. the user.", {{"seed", "5"}, {"max_tokens", "3"}, {"text_chunk_size", "10"}}));
    require(chunked.audio_output->samples.size() > offline->run(text_request("the user.", {{"seed", "5"}, {"max_tokens", "3"}})).audio_output->samples.size(),
            "chunked text speaks every chunk");
    std::cout << "  TTS: offline == streaming, seeds, voices, chunks, options ok\n";
}

std::vector<std::byte> wav_bytes(const runtime::AudioBuffer & audio) {
    const auto path = std::filesystem::temp_directory_path() / "lfm3_audio_session_test_question.wav";
    engine::audio::write_pcm16_wav(path, audio.sample_rate, audio.channels, audio.samples);
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::byte> out(bytes.size());
    std::transform(bytes.begin(), bytes.end(), out.begin(), [](char c) { return static_cast<std::byte>(c); });
    return out;
}

void test_s2s(const Fixture & f) {
    auto offline = open<lfm3::Lfm3ChatSession>(f, runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Offline, kF32);
    auto streaming = open<lfm3::Lfm3ChatSession>(f, runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Streaming, kF32);
    const auto question = tone(1.0);
    const auto request = audio_request(question, {{"seed", "3"}, {"max_tokens", "12"}, {"return_codes", "true"}});
    const auto reply = offline->run(request);
    require(!reply.audio_output->samples.empty(), "S2S audio");
    require_eq(reply.output_artifacts.size(), size_t{1}, "return_codes gives the reply artifact");
    const auto & artifact = reply.output_artifacts.front();
    require_eq(artifact.id, std::string(lfm3::kLfm3ReplyArtifactId), "reply artifact id");
    const auto vocab = static_cast<int32_t>(f.package.shape.audio_vocab_size());
    const lfm3::Lfm3ReplyCheckpoint checkpoint{12, vocab, 320};
    const auto steps = lfm3::read_lfm3_reply_artifact(artifact, checkpoint);
    require(steps.size() <= 12, "reply within max_tokens");
    require_eq(artifact.meta.at("ended"), std::string(offline->reached_max_tokens() ? "false" : "true"), "ended matches the cut");

    const auto [events, result] = stream(*streaming, request);
    require_eq(text_of(events), text_of(reply), "S2S streaming text == offline");
    require_eq(text_of(result), text_of(reply), "finish_stream text");
    require_eq(result.output_artifacts.size(), size_t{1}, "streaming reply artifact");
    require(result.output_artifacts.front().payload == artifact.payload, "streaming reply artifact == offline");
    const double rel = rel_diff(events.audio_output->samples, reply.audio_output->samples);
    require(rel <= 1e-5, "S2S streaming vs offline audio rel " + std::to_string(rel));

    // Turn 2 replays turn 1 from the artifacts; the reply was cut at max_tokens, so its replay closes the audio
    // with an EOS frame, then <|im_end|> and "\n".
    require(offline->reached_max_tokens(), "turn 1 is cut at max_tokens inside its first audio block");
    const auto dump_dir = lfm3_audio_test::fresh_directory("lfm3_audio_session_test_dumps");
    lfm3_audio_test::set_env("LFM3_AUDIO_DUMP_DIR", dump_dir.string());
    auto turn2 = audio_request(tone(0.5), {{"seed", "4"}, {"max_tokens", "8"}});
    turn2.input_artifacts = {runtime::make_voice_artifact(runtime::ArtifactKind::Custom, lfm3::kLfm3QuestionArtifactId, wav_bytes(question)), artifact};
    const auto second = offline->run(turn2);
    lfm3_audio_test::set_env("LFM3_AUDIO_DUMP_DIR", "");
    require(!second.audio_output->samples.empty(), "turn 2 audio");

    const lfm3::Lfm3Tokenizer tokenizer(f.package.root / "tokenizer.json");
    const auto tokens = [](double seconds) { return lfm3::lfm3_audio_tokens(lfm3::lfm3_feature_frames(static_cast<int64_t>(std::llround(seconds * 16000)))); };
    const auto expected = lfm3::make_lfm3_s2s_prompt(tokenizer, lfm3::kLfm3S2sSystemPrompt, {tokens(1.0)}, {steps}, tokens(0.5), vocab - 1);
    std::vector<int64_t> shape;
    const auto dumped = lfm3::lfm3_read_npy_i32(dump_dir / "prompt_ids.npy", &shape);
    require(dumped == expected.ids, "turn 2 prompt replays turn 1");
    const auto & ids = tokenizer.ids();
    const std::vector<int32_t> closing = {ids.audio_out, ids.im_end, ids.nl, ids.im_start};
    require(std::search(dumped.begin(), dumped.end(), closing.begin(), closing.end()) != dumped.end(),
            "the cut reply is closed by <|audio_out|> <|im_end|> \\n before the next user turn");
    require(expected.frames.back() == std::vector<int32_t>(12, vocab - 1), "the closing frame is the EOS frame");
    for (const char * name : {"wave16k", "features", "downsample", "encoder", "projector", "prompt_ids", "prefill_hidden", "steps",
                              "maskgit_orders", "depth_logits_f0_p0", "codes", "codec_spectrum", "audio24k"}) {
        require(std::filesystem::exists(dump_dir / (std::string(name) + ".npy")), std::string("dump ") + name);
    }

    // A stream finished after its first event: its reply artifact is not ended, and replays in the next turn.
    streaming->start_stream(request);
    streaming->process_audio_chunk({question.sample_rate, 1, 0, question.samples});
    (void)streaming->next_stream_event();
    auto early = turn2;
    early.input_artifacts.back() = streaming->finish_stream().output_artifacts.at(0);
    require_eq(early.input_artifacts.back().meta.at("ended"), std::string("false"), "an early finish is not ended");
    require(!offline->run(early).audio_output->samples.empty(), "a reply finished early replays");
    streaming->start_stream(request);
    require_error<runtime::InvalidRequestError>([&] { (void)streaming->next_stream_event(); }, "received no audio", "a stream without audio");

    require_throws_with([&] { offline->run(audio_request(question, {{"max_audio_tail_frames", "-1"}})); }, "max_audio_tail_frames", "negative tail");
    require_throws_with([&] { offline->run(audio_request(question, {{"text_chunk_size", "10"}})); }, "does not take request option text_chunk_size",
                        "TTS option in s2s");
    require_throws_with([&] { offline->run(audio_request(question, {{"language", "de"}})); }, "takes no language", "s2s language");
    auto bad_system = audio_request(question);
    bad_system.text_input = runtime::Transcript{"Respond <|audio|> now.", ""};
    require_throws_with([&] { streaming->start_stream(bad_system); }, "holds an audio placeholder token", "placeholder in the streamed system prompt");
    auto lonely = audio_request(question);
    lonely.input_artifacts = {turn2.input_artifacts.front()};
    require_throws_with([&] { offline->run(lonely); }, "has no reply after it", "question without reply");
    require_error<runtime::CapacityError>([&] { offline->run(audio_request(question, {{"max_tokens", "9000"}})); }, "9000", "max_tokens past 8192");
    std::cout << "  S2S: offline == streaming, reply artifact, 2-turn replay, early finish, dumps, options ok\n";
}

}  // namespace

int main() {
    try {
        Fixture f{write_session_package(), nullptr};
        runtime::ModelLoadRequest load;
        load.model_path = f.package.root;
        load.family_hint = "lfm3_audio";
        const auto loader = lfm3::make_lfm3_audio_loader();
        require(loader->can_load(load), "the loader claims the package");
        require_eq(runtime::make_default_registry().inspect(load).metadata.family, std::string("lfm3_audio"),
                   "the explicit family selects lfm3_audio");
        f.model = loader->load(load);
        test_asr(f);
        test_thinking_ends_inside();
        test_tts(f);
        test_s2s(f);
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_session_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "lfm3_audio sessions: ok\n";
    return 0;
}
