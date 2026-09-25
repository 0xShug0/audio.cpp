// The ASR session end to end on a synthetic package. The backbone's layers
// are all zero, so each position's logits come from its own embedding alone,
// and the embeddings are built so that greedy decoding after the prompt's
// final "\n" spells a fixed word and then emits <|im_end|>. That makes the
// transcript predictable while the encoder, prefill and decode all run.
#include "engine/community_models/lfm2_audio/session.h"
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

std::unique_ptr<runtime::IOfflineVoiceTaskSession> open_session(
    const std::filesystem::path & root, std::unordered_map<std::string, std::string> session_options = {}) {
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

    // "hi" and <|im_end|> take three tokens; a cut-off transcript is an error.
    rejects({{"max_tokens", "2"}}, "max_tokens");
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

void test_chunking(const Package & package) {
    auto session = open_session(package.root);
    const auto audio = tone(3.0);
    require_eq(transcribe(*session, request(audio)), std::string("hi"), "default chunks");
    require_eq(transcribe(*session, request(audio, {{"audio_chunk_seconds", "1"}})), std::string("hi hi hi"),
        "1 s chunks");
    require_eq(transcribe(*session, request(audio, {{"audio_chunk_mode", "fixed"}, {"audio_chunk_seconds", "1"}})),
        std::string("hi hi hi"), "fixed 1 s chunks");
    require_eq(transcribe(*session, request(audio, {{"audio_chunk_mode", "none"}, {"audio_chunk_seconds", "1"}})),
        std::string("hi"), "no chunks");
    // A tail under a second joins the previous chunk; alone, 3 ms would fail
    // the features and half a second is at best a cut-off word.
    require_eq(transcribe(*session, request(tone(2.003), {{"audio_chunk_seconds", "1"}})), std::string("hi hi"),
        "a 3 ms tail");
    require_eq(transcribe(*session, request(tone(2.5), {{"audio_chunk_seconds", "1"}})), std::string("hi hi"),
        "a 0.5 s tail");
    require_eq(transcribe(*session, request(tone(0.5), {{"audio_chunk_seconds", "1"}})), std::string("hi"),
        "audio shorter than one chunk");
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
    require_eq(transcribe(*session, request(tone(3.0), {{"audio_chunk_seconds", "1"}})), std::string("日日日"),
        "Japanese chunks");
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

    const Package broken("audiocpp_lfm2_audio_session_broken_test");
    write_package(broken.root, "hi");
    std::filesystem::remove(broken.root / "mmproj-Model-F16.gguf");
    require(!loader->can_load(load_request(broken.root)), "a package without an mmproj");
    require_throws_with([&] { (void)loader->load(load_request(broken.root)); }, "mmproj", "loading a package without an mmproj");
}

}  // namespace

int main() {
    try {
        const Package package("audiocpp_lfm2_audio_session_test");
        write_package(package.root, "hi");
        test_transcribes(package);
        test_audio_inputs(package);
        test_request_options(package);
        test_chunking(package);
        test_selects_backbone();
        test_japanese_checkpoint();
        test_checkpoint_metadata();
        test_multibyte_transcript();
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
