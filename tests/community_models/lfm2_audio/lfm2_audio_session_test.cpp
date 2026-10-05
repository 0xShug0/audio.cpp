// The ASR session end to end on a synthetic package. The backbone's layers
// are all zero, so each position's logits come from its own embedding alone,
// and the embeddings are built so that greedy decoding after the prompt's
// final "\n" spells a fixed word and then emits <|im_end|>. That makes the
// transcript predictable while the encoder, prefill and decode all run.
#include "engine/community_models/lfm2_audio/session.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
namespace runtime = engine::runtime;
using engine::test::require;
using engine::test::require_eq;
using lfm2_audio_test::byte_level_vocabulary;
using lfm2_audio_test::byte_tokens;
using lfm2_audio_test::require_throws_with;
using lfm2_audio_test::token_id;

// Chain c0 -> c1 -> ... where c0 = "\n" and the last link is <|im_end|>:
// row(c_k) = 4^k (e_{k-1} + e_k). The query c_k scores c_{k+1} at 4^(k+1),
// itself at 2 * 4^k, and every other row at most 4^(k-1) (others are zero).
lfm2_audio_test::TensorMap backbone_weights(
    const lfm2_audio_test::BackboneShape & shape,
    const lfm2_audio_test::TextVocab & vocab,
    const std::string & word,
    const std::string & stop_token) {
    auto tensors = lfm2_audio_test::backbone_tensors(shape, static_cast<int64_t>(vocab.tokens.size()),
        [](const std::string & name, size_t count) {
            const bool is_norm = name.find("norm") != std::string::npos;
            return std::vector<float>(count, is_norm ? 1.0f : 0.0f);
        });
    // One byte token per byte of the word, so non-ASCII characters span
    // several tokens. The word's bytes must all differ.
    const auto bytes = byte_tokens();
    std::vector<std::string> chain = {"Ċ"};
    for (const char ch : word) chain.push_back(bytes[static_cast<unsigned char>(ch)]);
    chain.push_back(stop_token);
    auto & table = tensors.at("token_embd.weight");
    float scale = 1.0f;
    for (size_t k = 0; k < chain.size(); ++k, scale *= 4.0f) {
        float * row = table.values.data() + token_id(vocab, chain[k]) * shape.hidden;
        if (k > 0) row[k - 1] = scale;
        row[k] = scale;
    }

    return tensors;
}

struct PackageOptions {
    std::vector<std::string> languages = {"en"};
    bool nan_adapter = false;
    std::string stop_token = "<|im_end|>";
    std::vector<std::string> missing_tokens;
};

// Writes Model-F16.gguf, which transcribes everything as `word`, and its
// mmproj into `root`.
void write_package(const std::filesystem::path & root, const std::string & word, const std::string & model_name = "Model-F16.gguf",
                   const PackageOptions & options = {}) {
    lfm2_audio_test::BackboneShape shape;
    shape.context = 2048;  // room for the default 512-token budget
    auto vocab = byte_level_vocabulary();
    for (const auto & token : options.missing_tokens) {
        const auto index = static_cast<std::ptrdiff_t>(token_id(vocab, token));
        vocab.tokens.erase(vocab.tokens.begin() + index);
        vocab.types.erase(vocab.types.begin() + index);
    }

    lfm2_audio_test::write_backbone(root / model_name, shape, vocab, backbone_weights(shape, vocab, word, options.stop_token), options.languages);

    lfm2_audio_test::EncoderShape encoder;
    encoder.output = shape.hidden;
    auto encoder_weights = lfm2_audio_test::encoder_tensors(encoder, lfm2_audio_test::random_fill(11, 0.2f));
    if (options.nan_adapter) {
        encoder_weights.at("mm.a.mlp.3.bias").values[0] = std::numeric_limits<float>::quiet_NaN();
    }

    lfm2_audio_test::write_mmproj(root / ("mmproj-" + model_name), encoder, encoder_weights);
}

runtime::ModelLoadRequest load_request(const std::filesystem::path & root) {
    runtime::ModelLoadRequest request;
    request.model_path = root;
    request.family_hint = "lfm2_audio";
    return request;
}

const std::filesystem::path kRepoRoot = ENGINE_REPO_ROOT;
const std::filesystem::path kVadModel = kRepoRoot / "assets" / "framework" / "models" / "silero_vad";

std::unique_ptr<runtime::IOfflineVoiceTaskSession> open_session(
    const std::filesystem::path & root, std::unordered_map<std::string, std::string> session_options = {}) {
    session_options.emplace("lfm2_audio.vad_model_path", kVadModel.string());
    const auto model = lfm2::make_lfm2_audio_loader()->load(load_request(root));
    runtime::SessionOptions options;
    options.backend = {engine::core::BackendType::Cpu, 0, 2};
    options.options = std::move(session_options);
    auto session = model->create_task_session({runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline}, options);
    session->prepare({});
    auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
    require(offline != nullptr, "the ASR session must be offline");
    session.release();
    return std::unique_ptr<runtime::IOfflineVoiceTaskSession>(offline);
}

runtime::AudioBuffer tone(double seconds, int sample_rate = 16000, int channels = 1) {
    runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = channels;
    const auto frames = static_cast<size_t>(std::llround(seconds * sample_rate));
    for (size_t i = 0; i < frames; ++i) {
        const auto value = static_cast<float>(0.3 * std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(i) / sample_rate));
        for (int c = 0; c < channels; ++c) audio.samples.push_back(value);
    }

    return audio;
}

runtime::TaskRequest request(runtime::AudioBuffer audio, std::unordered_map<std::string, std::string> options = {}) {
    runtime::TaskRequest out;
    out.audio_input = std::move(audio);
    out.options = std::move(options);
    return out;
}

std::string transcribe(runtime::IOfflineVoiceTaskSession & session, const runtime::TaskRequest & task_request) {
    const auto result = session.run(task_request);
    require(result.text_output.has_value(), "ASR must return a transcript");
    return result.text_output->text;
}

struct Package {
    std::filesystem::path root;
    explicit Package(const std::string & name) : root(lfm2_audio_test::fresh_directory(name)) {}
    ~Package() { std::filesystem::remove_all(root); }
};

void test_transcribes(const Package & package) {
    auto session = open_session(package.root);
    const auto result = session->run(request(tone(1.0)));
    require(result.text_output.has_value(), "ASR must return a transcript");
    require_eq(result.text_output->text, std::string("hi"), "transcript");
    require_eq(result.text_output->language, std::string("en"), "transcript language");
    // The session is reusable.
    require_eq(transcribe(*session, request(tone(0.4))), std::string("hi"), "second request");
}

void test_audio_inputs(const Package & package) {
    auto session = open_session(package.root);
    require_eq(transcribe(*session, request(tone(1.0, 44100, 2))), std::string("hi"), "44.1 kHz stereo");
    require_eq(transcribe(*session, request(tone(1.0, 8000))), std::string("hi"), "8 kHz mono");
    require_eq(transcribe(*session, request(tone(0.01))), std::string("hi"), "10 ms");

    require_throws_with([&] { (void)session->run(runtime::TaskRequest{}); }, "audio_input", "a request without audio");
    require_throws_with([&] { (void)session->run(request(tone(0.0))); }, "non-empty audio", "empty audio");
    require_throws_with([&] { (void)session->run(request(tone(0.005))); }, "10 ms", "5 ms of audio");

    auto nan_sample = tone(1.0);
    nan_sample.samples[100] = std::numeric_limits<float>::quiet_NaN();
    require_throws_with([&] { (void)session->run(request(nan_sample)); }, "non-finite samples", "a NaN sample");

    auto inf_sample = tone(1.0, 44100);
    inf_sample.samples[100] = std::numeric_limits<float>::infinity();
    require_throws_with([&] { (void)session->run(request(inf_sample)); }, "non-finite samples", "an Inf sample");

    auto no_channels = tone(1.0);
    no_channels.channels = 0;
    require_throws_with([&] { (void)session->run(request(no_channels)); }, "non-empty audio", "zero channels");
}

void test_request_options(const Package & package) {
    auto session = open_session(package.root);
    require_eq(transcribe(*session, request(tone(1.0), {{"max_tokens", "3"}})), std::string("hi"), "max_tokens=3");
    require_eq(transcribe(*session, request(tone(1.0), {{"language", "en"}})), std::string("hi"), "language=en");
    require_eq(transcribe(*session, request(tone(1.0), {{"language", "auto"}})), std::string("hi"), "language=auto");

    const auto rejects = [&](std::unordered_map<std::string, std::string> options, const std::string & needle) {
        require_throws_with([&] { (void)session->run(request(tone(1.0), options)); }, needle,
            "request option " + options.begin()->first + "=" + options.begin()->second);
    };

    rejects({{"language", "ja"}}, "transcribes en");
    rejects({{"max_tokens", "0"}}, "max_tokens");
    rejects({{"max_tokens", "5x"}}, "max_tokens");

    // The framework's integer parser reports "stoll: no conversion" here,
    // without the option name, so only the rejection is checked.
    lfm2_audio_test::require_throws([&] { (void)session->run(request(tone(1.0), {{"max_tokens", "many"}})); },
        "request option max_tokens=many");

    rejects({{"audio_chunk_mode", "sometimes"}}, "audio_chunk_mode");
    rejects({{"audio_chunk_seconds", "0"}}, "audio_chunk_seconds");
    rejects({{"audio_chunk_seconds", "-3"}}, "audio_chunk_seconds");
    rejects({{"audio_chunk_seconds", "0.5"}}, "audio_chunk_seconds");

    rejects({{"temperature", "0.7"}}, "temperature");
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

size_t occurrences(const std::string & text, const std::string & needle) {
    size_t found = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) {
        ++found;
    }

    return found;
}

// "hi" and <|im_end|> take three tokens. A transcript that reaches max_tokens
// is cut off there and kept, as liquid-audio keeps what it generated, with a
// warning; in chunked audio each chunk is cut on its own and the rest go on.
void test_max_tokens(const Package & package) {
    auto session = open_session(package.root);
    const auto & asr = dynamic_cast<const lfm2::Lfm2AudioSession &>(*session);
    const auto run = [&](const runtime::TaskRequest & task_request, std::string & warnings) {
        const CapturedStderr captured;
        auto text = transcribe(*session, task_request);
        warnings = captured.text();
        return text;
    };

    std::string warnings;
    require_eq(run(request(tone(1.0), {{"max_tokens", "3"}}), warnings), std::string("hi"), "max_tokens=3");
    require(warnings.empty() && !asr.reached_max_tokens(), "a whole transcript does not warn");

    require_eq(run(request(tone(1.0), {{"max_tokens", "2"}}), warnings), std::string("hi"), "max_tokens=2");
    require(asr.reached_max_tokens(), "max_tokens=2 cuts the transcript");
    require(occurrences(warnings, "[warning][lfm2_audio]") == 1, "one warning: " + warnings);
    require(warnings.find("0.0-1.0 s reached max_tokens=2") != std::string::npos && warnings.find("raise max_tokens") != std::string::npos,
        "the warning says where and what to raise: " + warnings);

    require_eq(run(request(tone(1.0), {{"max_tokens", "1"}}), warnings), std::string("h"), "max_tokens=1");
    const std::unordered_map<std::string, std::string> cut_chunks = {
        {"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}, {"max_tokens", "1"}};
    require_eq(run(request(tone(3.0), cut_chunks), warnings), std::string("h h h"), "three chunks cut at max_tokens=1");
    require(occurrences(warnings, "[warning][lfm2_audio]") == 3 && warnings.find("2.0-3.0 s") != std::string::npos,
        "a warning per cut chunk: " + warnings);
    require_throws_with([&] { (void)session->run(request(tone(1.0), {{"max_tokens", "0"}})); }, "max_tokens", "max_tokens=0");
    require(!asr.reached_max_tokens(), "a failed request cuts nothing");

    require_eq(run(request(tone(1.0)), warnings), std::string("hi"), "the next request");
    require(warnings.empty() && !asr.reached_max_tokens(), "the next request starts uncut");
}

// Byte-level tokens need not end on a character boundary, so max_tokens can
// cut a transcript inside a character. The transcript then ends at the last
// whole character, with the same warning, and each chunk is trimmed before the
// chunks are joined.
void test_max_tokens_inside_a_character() {
    // é, 日 and 😀 take two, three and four tokens after the one-token "a".
    for (const std::string character : {"\xC3\xA9", "\xE6\x97\xA5", "\xF0\x9F\x98\x80"}) {
        const auto bytes = std::to_string(character.size()) + "-byte";
        const Package package("audiocpp_lfm2_audio_session_cut_" + std::to_string(character.size()) + "_test");
        write_package(package.root, "a" + character);
        auto session = open_session(package.root);
        const auto & asr = dynamic_cast<const lfm2::Lfm2AudioSession &>(*session);
        const auto run = [&](std::unordered_map<std::string, std::string> options, double seconds, std::string & warnings) {
            const CapturedStderr captured;
            auto text = transcribe(*session, request(tone(seconds), std::move(options)));
            warnings = captured.text();
            return text;
        };

        std::string warnings;
        for (size_t kept = 1; kept < character.size(); ++kept) {
            const auto max_tokens = std::to_string(1 + kept);
            const auto label = bytes + " character cut after byte " + std::to_string(kept);
            require_eq(run({{"max_tokens", max_tokens}}, 1.0, warnings), std::string("a"), label);
            require(asr.reached_max_tokens() && occurrences(warnings, "reached max_tokens=" + max_tokens) == 1,
                label + " still warns: " + warnings);
        }

        // A cut right after the character, or none, leaves the text as it is.
        const auto whole = std::to_string(1 + character.size());
        require_eq(run({{"max_tokens", whole}}, 1.0, warnings), "a" + character, bytes + " character cut after it");
        require(asr.reached_max_tokens(), bytes + " character cut after it is still cut");
        require_eq(run({}, 1.0, warnings), "a" + character, bytes + " character uncut");
        require(warnings.empty() && !asr.reached_max_tokens(), bytes + " character uncut does not warn");

        // The synthetic model writes the same tokens for every chunk. A chunk
        // cut inside the character joins the next as "a" would, with a space
        // before the next ASCII word, rather than leaving the bytes mid-text.
        const std::unordered_map<std::string, std::string> fixed_1s = {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}};
        auto cut_chunks = fixed_1s;
        cut_chunks.emplace("max_tokens", "2");
        require_eq(run(cut_chunks, 3.0, warnings), std::string("a a a"), bytes + " character cut in three chunks");
        require(occurrences(warnings, "[warning][lfm2_audio]") == 3, bytes + " character: a warning per cut chunk: " + warnings);
        const auto whole_chunk = "a" + character;
        require_eq(run(fixed_1s, 3.0, warnings), whole_chunk + whole_chunk + whole_chunk, bytes + " character in three uncut chunks");
    }
}

// The trim does not depend on a cut: a transcript the model itself ends inside
// a character also ends at the last whole character, without a warning.
void test_ends_inside_a_character() {
    const Package package("audiocpp_lfm2_audio_session_lead_byte_test");
    write_package(package.root, "a\xC3");
    auto session = open_session(package.root);
    const auto & asr = dynamic_cast<const lfm2::Lfm2AudioSession &>(*session);
    const CapturedStderr captured;
    require_eq(transcribe(*session, request(tone(1.0))), std::string("a"), "a transcript that ends inside a character");
    require(!asr.reached_max_tokens(), "a transcript that ends inside a character is not cut");
    require_eq(transcribe(*session, request(tone(3.0), {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}})),
        std::string("a a a"), "three chunks that end inside a character");
    require(!asr.reached_max_tokens() && captured.text().empty(), "no chunk is cut: " + captured.text());
}

void test_chunking(const Package & package) {
    auto session = open_session(package.root);
    const auto audio = tone(3.0);
    require_eq(transcribe(*session, request(audio)), std::string("hi"), "default chunks");
    require_eq(transcribe(*session, request(audio, {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}})),
        std::string("hi hi hi"), "fixed 1 s chunks");
    require_eq(transcribe(*session, request(audio, {{"audio_chunk_mode", "none"}, {"audio_chunk_seconds", "1"}})),
        std::string("hi"), "no chunks");
    // A tail under a second joins the previous chunk; alone, 3 ms would fail
    // the features and half a second is at best a cut-off word.
    const std::unordered_map<std::string, std::string> fixed_1s = {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}};
    require_eq(transcribe(*session, request(tone(2.003), fixed_1s)), std::string("hi hi"), "a 3 ms tail");
    require_eq(transcribe(*session, request(tone(2.5), fixed_1s)), std::string("hi hi"), "a 0.5 s tail");
    // auto takes input that fits one chunk whole, as liquid-audio does.
    require_eq(transcribe(*session, request(tone(0.5), {{"audio_chunk_seconds", "1"}})), std::string("hi"),
        "auto on audio shorter than one chunk");
}

// 3.5 s of LibriSpeech speech followed by `silence_seconds` of zeros, and the
// same again when `twice`.
runtime::AudioBuffer speech(double silence_seconds, bool twice = false) {
    const auto wav = engine::audio::read_wav_f32(
        kRepoRoot / "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
    runtime::AudioBuffer out{wav.sample_rate, 1, {}};
    for (int i = 0; i < (twice ? 2 : 1); ++i) {
        out.samples.insert(out.samples.end(), wav.samples.begin(), wav.samples.end());
        out.samples.insert(out.samples.end(), static_cast<size_t>(silence_seconds * wav.sample_rate), 0.0f);
    }

    return out;
}

// Longer input is split at pauses, and silence between the speech is not
// transcribed; the synthetic model says "hi" once per chunk.
void test_vad_chunking(const Package & package) {
    auto session = open_session(package.root);
    const auto two_utterances = speech(3.0, true);
    require_eq(transcribe(*session, request(two_utterances, {{"audio_chunk_mode", "vad"}})), std::string("hi hi"),
        "vad on two utterances");
    require_eq(transcribe(*session, request(two_utterances, {{"audio_chunk_seconds", "5"}})), std::string("hi hi"),
        "auto on input longer than a chunk");
    require_eq(transcribe(*session, request(speech(30.0))), std::string("hi"), "a long silent tail");
    require_eq(transcribe(*session, request(tone(35.0, 16000) , {{"audio_chunk_mode", "vad"}})), std::string(""),
        "a tone without speech");
    auto silence = tone(35.0);
    std::fill(silence.samples.begin(), silence.samples.end(), 0.0f);
    require_eq(transcribe(*session, request(silence)), std::string(""), "35 s of silence");

    // Without the model, auto falls back to fixed chunks and vad fails.
    auto no_vad = open_session(package.root, {{"lfm2_audio.vad_model_path", (package.root / "missing").string()}});
    require_eq(transcribe(*no_vad, request(tone(3.0), {{"audio_chunk_seconds", "1"}})), std::string("hi hi hi"),
        "auto without the VAD model");
    require_throws_with([&] { (void)no_vad->run(request(tone(3.0), {{"audio_chunk_mode", "vad"}})); }, "missing",
        "vad without the VAD model");
}

void test_selects_backbone() {
    const Package package("audiocpp_lfm2_audio_session_select_test");
    write_package(package.root, "hi", "Model-F16.gguf");
    write_package(package.root, "yo", "Model-Q8_0.gguf");
    require_throws_with([&] { (void)open_session(package.root); }, "several backbone GGUFs", "two backbones");
    require_eq(transcribe(*open_session(package.root, {{"lfm2_audio.model_gguf", "Model-Q8_0.gguf"}}), request(tone(1.0))),
        std::string("yo"), "Model-Q8_0.gguf");
    require_eq(transcribe(*open_session(package.root, {{"lfm2_audio.model_gguf", "Model-F16.gguf"}}), request(tone(1.0))),
        std::string("hi"), "Model-F16.gguf");
    require_throws_with(
        [&] { (void)open_session(package.root, {{"lfm2_audio.model_gguf", "Model-Q4_0.gguf"}}); }, "Model-Q4_0.gguf",
        "a missing backbone");
    require_throws_with([&] { (void)open_session(package.root, {{"lfm2_audio.voice", "x"}}); }, "lfm2_audio.voice",
        "an unknown session option");
}

void test_japanese_checkpoint() {
    const Package package("audiocpp_lfm2_audio_session_ja_test");
    PackageOptions options;
    options.languages = {"ja"};
    write_package(package.root, "hi", "Model-F16.gguf", options);
    auto session = open_session(package.root);
    const auto result = session->run(request(tone(1.0), {{"language", "ja"}}));
    require_eq(result.text_output->language, std::string("ja"), "transcript language");
    require_throws_with([&] { (void)session->run(request(tone(1.0), {{"language", "en"}})); }, "transcribes ja",
        "language=en on a Japanese checkpoint");
}

void test_checkpoint_metadata() {
    const auto open_with = [](const std::string & name, const PackageOptions & options) {
        const Package package(name);
        write_package(package.root, "hi", "Model-F16.gguf", options);
        return open_session(package.root)->run(request(tone(1.0))).text_output->language;
    };
    PackageOptions options;
    options.languages = {};
    require_eq(open_with("audiocpp_lfm2_audio_no_language_test", options), std::string("en"), "no general.languages");

    options.languages = {"de"};
    require_throws_with([&] { (void)open_with("audiocpp_lfm2_audio_de_test", options); }, "general.languages = [de]",
        "an unknown language");
    options.languages = {"en", "ja"};
    require_throws_with([&] { (void)open_with("audiocpp_lfm2_audio_en_ja_test", options); }, "general.languages = [en, ja]",
        "two languages");

    options = {};
    options.missing_tokens = {"<|im_start|>"};
    require_throws_with([&] { (void)open_with("audiocpp_lfm2_audio_no_im_start_test", options); }, "<|im_start|>",
        "a vocabulary without <|im_start|>");
}

// "日" is three byte tokens; decoding must put the character back together,
// and chunks of Japanese text join without spaces.
void test_multibyte_transcript() {
    const Package package("audiocpp_lfm2_audio_session_bytes_test");
    PackageOptions options;
    options.languages = {"ja"};
    write_package(package.root, "日", "Model-F16.gguf", options);
    auto session = open_session(package.root);
    require_eq(transcribe(*session, request(tone(1.0))), std::string("日"), "a three-byte character");
    require_eq(transcribe(*session, request(tone(3.0), {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}})),
        std::string("日日日"), "Japanese chunks");
}

// Bytes that can never make a character come out as U+FFFD, as liquid-audio
// decodes them: here a Thai character left open before " ay", as S2S replies
// in Thai script have it. Each chunk's transcript is decoded on its own. A
// transcript that ends inside a character ends at the last whole one instead
// (test_ends_inside_a_character).
void test_ill_formed_transcript() {
    const Package package("audiocpp_lfm2_audio_session_ill_formed_test");
    write_package(package.root, "\xE0\xB8 ay");
    auto session = open_session(package.root);
    require_eq(transcribe(*session, request(tone(1.0))), std::string("\xEF\xBF\xBD ay"), "a character left open mid-text");
    require_eq(transcribe(*session, request(tone(3.0), {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}})),
        std::string("\xEF\xBF\xBD ay\xEF\xBF\xBD ay\xEF\xBF\xBD ay"), "chunks with a character left open mid-text");
}

// <|audio_start|> would switch liquid-audio to audio output, so it ends the
// transcript just like <|im_end|>.
void test_audio_start_stops() {
    const Package package("audiocpp_lfm2_audio_session_audio_start_test");
    PackageOptions options;
    options.stop_token = "<|audio_start|>";
    write_package(package.root, "hi", "Model-F16.gguf", options);
    require_eq(transcribe(*open_session(package.root), request(tone(1.0))), std::string("hi"), "<|audio_start|>");
}

// NaN audio embeddings would reach the logits; the request must fail rather
// than return the empty transcript that argmax over NaN decodes to.
void test_numeric_failure_is_an_error() {
    const Package package("audiocpp_lfm2_audio_session_nan_test");
    PackageOptions options;
    options.nan_adapter = true;
    write_package(package.root, "hi", "Model-F16.gguf", options);
    auto session = open_session(package.root);
    require_throws_with([&] { (void)session->run(request(tone(1.0))); }, "non-finite audio embeddings", "NaN adapter output");
}

void test_loader(const Package & package) {
    const auto loader = lfm2::make_lfm2_audio_loader();
    require(loader->can_load(load_request(package.root)), "the loader must accept the package");
    auto other_family = load_request(package.root);
    other_family.family_hint = "vibeasr";
    require(!loader->can_load(other_family), "the loader must decline another family");

    // An incomplete package is still claimed, so loading it says what is missing.
    const Package broken("audiocpp_lfm2_audio_session_broken_test");
    write_package(broken.root, "hi");
    std::filesystem::remove(broken.root / "mmproj-Model-F16.gguf");
    auto no_hint = load_request(broken.root);
    no_hint.family_hint.reset();
    require(loader->can_load(no_hint), "a package without an mmproj is still LFM2-Audio's");
    require_throws_with([&] { (void)loader->load(no_hint); }, "no mmproj GGUF", "loading a package without an mmproj");

    const Package unrelated("audiocpp_lfm2_audio_session_unrelated_test");
    lfm2_audio_test::GgufWriter other;
    other.set("general.architecture", "llama");
    other.add("weight", lfm2_audio_test::Tensor{{4}, {0.0f, 0.0f, 0.0f, 0.0f}});
    other.write(unrelated.root / "model.gguf");
    auto unrelated_request = load_request(unrelated.root);
    unrelated_request.family_hint.reset();
    require(!loader->can_load(unrelated_request), "a directory without LFM2-Audio files");

    // Streaming is for speech output, and speech output needs the vocoder and
    // detokenizer files, which this package does not have.
    const auto model = loader->load(load_request(package.root));
    runtime::SessionOptions options;
    options.backend = {engine::core::BackendType::Cpu, 0, 2};
    require_throws_with([&] { (void)model->create_task_session({runtime::VoiceTaskKind::Asr, runtime::RunMode::Streaming}, options); },
        "ASR runs offline only", "a streaming ASR session");
    require_throws_with([&] { (void)model->create_task_session({runtime::VoiceTaskKind::Tts, runtime::RunMode::Offline}, options); },
        "cannot pick the vocoder GGUF", "a TTS session without the vocoder file");
    require_throws_with(
        [&] { (void)model->create_task_session({runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Streaming}, options); },
        "cannot pick the vocoder GGUF", "a streaming s2s session without the vocoder file");
    require_throws_with([&] { (void)model->create_task_session({runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline}, options); },
        "supports the asr, tts and s2s tasks", "another task");
}

}  // namespace

int main() {
    try {
        const Package package("audiocpp_lfm2_audio_session_test");
        write_package(package.root, "hi");
        test_transcribes(package);
        test_audio_inputs(package);
        test_request_options(package);
        test_max_tokens(package);
        test_max_tokens_inside_a_character();
        test_ends_inside_a_character();
        test_chunking(package);
        test_vad_chunking(package);
        test_selects_backbone();
        test_japanese_checkpoint();
        test_checkpoint_metadata();
        test_multibyte_transcript();
        test_ill_formed_transcript();
        test_audio_start_stops();
        test_numeric_failure_is_an_error();
        test_loader(package);
        std::cout << "lfm2_audio_session_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_session_test: " << error.what() << '\n';
        return 1;
    }
}
