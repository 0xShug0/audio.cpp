// LFM2.5-Audio speech-to-speech (s2s) on Liquid's published GGUFs against
// liquid-audio 1.3.0 (LFM2.5-Audio-1.5B revision
// c362a0625dfe45aa588dce5f0ada28a7e5707628, fp32), for assets/resources/c.wav
// ("Oh hi, I'm Linashi Mumei from Whole Life English Council. ...") under the
// chat system prompt:
// - the prompt ChatState builds around the audio, and the interleave blocks;
// - generate_interleaved's first text block and first audio block, greedy:
//   the text logits, the backbone output after the text and the first
//   frame's depthformer logits;
// - replies through the registry, offline and streamed, whose audio says what
//   their text says, checked by transcribing it back with the ASR task.
//
// --model is the directory of LiquidAI/LFM2.5-Audio-1.5B-GGUF (default
// models/LFM2.5-Audio-1.5B-GGUF, where the lfm2_audio_1_5b_* packages install);
// --model-gguf picks the backbone, and the mmproj-, vocoder- and tokenizer-
// files of the same name come with it. Skips with 125 when the files are not
// there.
#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/depthformer.h"
#include "engine/community_models/lfm2_audio/interleaved.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

constexpr const char * kAudio = "assets/resources/c.wav";

// ChatState's text around the audio, and where the audio goes.
const std::vector<int32_t> kPromptIds = {1, 6, 24131, 708, 3104, 4168, 916, 1251, 799, 17927, 3304, 810,
                                         14052, 523, 7, 708, 6, 6423, 708, 7, 708, 6, 64015, 708};
constexpr int64_t kAudioTokens = 95;
constexpr int32_t kFirstAudioPosition = 19;

struct Top {
    int32_t id;
    float logit;
    float second;
};

// The first text block: each greedy token, its logit and the runner-up's.
const Top kTextBlock[] = {
    {550, 16.97331f, 16.84728f}, {10109, 15.12635f, 14.32001f}, {26508, 9.75035f, 9.49224f},
    {1010, 14.96412f, 13.30494f}, {4343, 14.45033f, 14.36696f}, {9368, 14.02111f, 12.57555f},
};
constexpr const char * kTextBlockText = "I’m glad you’re interested";

struct Point {
    int64_t index;
    float value;
};

// Backbone output after the text block, the first frame's depthformer input.
const Point kHiddenPoints[] = {{0, 0.064191f}, {17, -0.032250f}, {511, 0.283188f}, {1024, 0.048182f}, {1500, -0.114832f}, {2047, -0.027834f}};

const Top kFirstFrame[] = {
    {1049, 22.11196f, 9.33686f}, {811, 17.91690f, 15.03631f}, {1626, 19.42575f, 12.59472f}, {290, 20.41790f, 14.60610f},
    {1335, 17.07018f, 15.24617f}, {1350, 19.51537f, 9.71587f}, {666, 16.72897f, 9.72017f}, {1630, 17.04280f, 13.54772f},
};

// The first audio block, greedy. Frame 10 holds a near-tie (0.015).
const std::vector<std::vector<int32_t>> kAudioBlock = {
    {1049, 811, 1626, 290, 1335, 1350, 666, 1630}, {127, 1470, 457, 1422, 481, 1509, 1978, 1533},
    {1880, 91, 1029, 1229, 457, 1030, 43, 1177},   {972, 1050, 1697, 104, 340, 1030, 825, 1744},
    {972, 1050, 1559, 104, 457, 11, 819, 1533},    {972, 1050, 457, 290, 457, 1572, 825, 2008},
    {972, 1050, 1559, 546, 457, 1572, 43, 2008},   {972, 1050, 457, 290, 340, 1030, 43, 945},
    {395, 1050, 1559, 546, 1736, 1720, 666, 1648}, {395, 1050, 1697, 104, 1736, 1030, 1978, 2008},
    {1140, 875, 325, 1478, 143, 1556, 1603, 1691}, {1268, 257, 1365, 1414, 35, 565, 1700, 158},
};

// Greedy picks are compared until the reference's top two come closer than
// this; past such a near-tie the weights' rounding decides.
constexpr float kNearTie = 0.05f;

// A spoken reply transcribed back matches its text but for the names the
// model makes up ("Al-Ula" comes back as "Aula"): 4-7% at the median of 30 to
// 60 seeds per backend. Now and then the ASR answers the reply, paraphrases it
// or runs on to max_tokens instead of transcribing it (liquid-audio fp32 went
// over 0.2 on 5 of 60 seeds), and a reply can fail to end its audio within
// max_tokens. So the check takes the median of three replies, and a reply or
// transcript that fails counts as a miss.
constexpr double kMaxReplyWordErrors = 0.2;
const std::vector<std::string> kReplySeeds = {"1234", "1235", "1236"};

std::filesystem::path repo_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }

    return fallback;
}

std::string join(const std::vector<int32_t> & ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        out << (i == 0 ? "" : ",") << ids[i];
    }

    return out.str();
}

class Checks {
public:
    void expect(bool ok, const std::string & label, const std::string & detail = "") {
        std::cout << (ok ? "PASS " : "FAIL ") << label << (ok || detail.empty() ? "" : ": " + detail) << "\n";
        failures_ += ok ? 0 : 1;
    }

    void expect_close(double actual, double expected, double tolerance, const std::string & label) {
        std::ostringstream detail;
        detail << label << ": got " << actual << ", expected " << expected << " +- " << tolerance;
        expect(std::fabs(actual - expected) <= tolerance, detail.str());
    }

    [[nodiscard]] int failures() const { return failures_; }

private:
    int failures_ = 0;
};

int32_t argmax(const std::vector<float> & values) {
    return static_cast<int32_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

float margin(std::vector<float> values) {
    std::partial_sort(values.begin(), values.begin() + 2, values.end(), std::greater<float>());
    return values[0] - values[1];
}

// Lower-case words; ’ is an apostrophe and anything else neither a letter nor
// a digit separates words.
std::vector<std::string> words(const std::string & text) {
    std::vector<std::string> out;
    std::string word;
    for (size_t i = 0; i < text.size(); ++i) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (text.compare(i, 3, "\xE2\x80\x99") == 0) {
            word += '\'';
            i += 2;
        } else if (std::isalnum(byte) != 0 || byte == '\'') {
            word += static_cast<char>(std::tolower(byte));
        } else if (!word.empty()) {
            out.push_back(word);
            word.clear();
        }
    }

    if (!word.empty()) {
        out.push_back(word);
    }

    return out;
}

double word_error_rate(const std::vector<std::string> & reference, const std::vector<std::string> & hypothesis) {
    std::vector<size_t> row(hypothesis.size() + 1);
    for (size_t j = 0; j < row.size(); ++j) {
        row[j] = j;
    }

    for (size_t i = 1; i <= reference.size(); ++i) {
        size_t diagonal = row[0];
        row[0] = i;
        for (size_t j = 1; j <= hypothesis.size(); ++j) {
            const size_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (reference[i - 1] == hypothesis[j - 1] ? 0 : 1)});
            diagonal = above;
        }
    }

    return reference.empty() ? 0.0 : static_cast<double>(row.back()) / static_cast<double>(reference.size());
}

void check_prompt(const lfm2::Lfm2AudioComponents & components, const lfm2::Lfm2AudioOutputComponents & output, Checks & checks) {
    const lfm2::Lfm2TextTokenizer tokenizer(components.vocabulary);
    const auto prompt = lfm2::make_lfm2_spoken_prompt(tokenizer, lfm2::kLfm2ChatSystemPrompt).with_audio(kAudioTokens);
    std::vector<int32_t> text_ids;
    for (size_t i = 0; i < prompt.input_ids.size(); ++i) {
        if (std::find(prompt.audio_positions.begin(), prompt.audio_positions.end(), static_cast<int32_t>(i)) == prompt.audio_positions.end()) {
            text_ids.push_back(prompt.input_ids[i]);
        }
    }

    checks.expect(text_ids == kPromptIds, "chat prompt around the audio", "got " + join(text_ids));
    checks.expect(!prompt.audio_positions.empty() && prompt.audio_positions.front() == kFirstAudioPosition, "audio after the user turn opens");

    // The English vocoder does not record its blocks; liquid-audio's
    // config.json has 6 and 12.
    checks.expect(output.interleave.text_steps == 6 && output.interleave.audio_steps == 12, "interleaved blocks of 6 text tokens and 12 frames",
        std::to_string(output.interleave.text_steps) + "/" + std::to_string(output.interleave.audio_steps));
}

// F16 weights keep the text logits within 0.15 and the backbone output
// within 0.01 of fp32 on every backend measured.
void check_stage_numbers(
    const lfm2::Lfm2AudioComponents & components,
    const lfm2::Lfm2AudioOutputComponents & output,
    const engine::core::BackendConfig & backend,
    Checks & checks) {
    engine::core::ExecutionContext execution(backend);
    const lfm2::Lfm2TextTokenizer tokenizer(components.vocabulary);
    lfm2::Lfm2FastConformerEncoderRuntime encoder(components.mmproj, components.encoder, execution);
    lfm2::Lfm2BackboneRuntime backbone(
        components.model, components.backbone, execution, components.mmproj, output.depthformer.codebooks, output.depthformer.audio_vocab_size);
    lfm2::Lfm2DepthformerRuntime depthformer(output.vocoder, output.depthformer, execution);

    const auto wav = engine::audio::read_wav_f32(repo_path(kAudio));
    const auto samples = lfm2::lfm2_audio_mono_16k({wav.sample_rate, wav.channels, wav.samples});
    const auto audio = encoder.encode(lfm2::Lfm2AudioFeatureExtractor(components.encoder.n_mels, backend.threads).extract(samples));
    checks.expect(audio.tokens == kAudioTokens, "audio tokens", std::to_string(audio.tokens));
    const auto prompt = lfm2::make_lfm2_spoken_prompt(tokenizer, lfm2::kLfm2ChatSystemPrompt).with_audio(audio.tokens);

    // The text block, greedy: text logits in, the next token back.
    auto out = backbone.start(prompt, audio, 64);
    std::vector<int32_t> block;
    for (size_t i = 0; i < std::size(kTextBlock); ++i) {
        const auto & top = kTextBlock[i];
        const int32_t token = argmax(out);
        checks.expect_close(out[static_cast<size_t>(top.id)], top.logit, 0.15, "text step " + std::to_string(i) + " top logit");
        if (token != top.id) {
            checks.expect(top.logit - top.second < kNearTie, "text step " + std::to_string(i) + " differs only at a near-tie", std::to_string(token));
            return;
        }

        block.push_back(token);
        const bool last = i + 1 == std::size(kTextBlock);
        out = backbone.step_text(token, last ? lfm2::Lfm2StepOutput::Hidden : lfm2::Lfm2StepOutput::Logits);
    }

    checks.expect(tokenizer.decode(block) == kTextBlockText, "text block", tokenizer.decode(block));
    for (const auto & point : kHiddenPoints) {
        checks.expect_close(out[static_cast<size_t>(point.index)], point.value, 0.01, "backbone output [" + std::to_string(point.index) + "]");
    }

    // The audio block, greedy.
    std::vector<std::vector<float>> frame_logits;
    const auto greedy = [&](int64_t, std::vector<float> & values) {
        frame_logits.push_back(values);
        return argmax(values);
    };

    size_t compared = 0;
    for (size_t frame = 0; frame < kAudioBlock.size(); ++frame) {
        if (frame > 0) {
            out = backbone.step_audio(kAudioBlock[frame - 1], lfm2::Lfm2StepOutput::Hidden);
        }

        frame_logits.clear();
        const auto codes = depthformer.frame(out, greedy);
        if (frame == 0) {
            for (size_t codebook = 0; codebook < std::size(kFirstFrame); ++codebook) {
                const auto & top = kFirstFrame[codebook];
                checks.expect_close(frame_logits[codebook][static_cast<size_t>(top.id)], top.logit, 0.15,
                    "first frame, codebook " + std::to_string(codebook) + " top logit");
            }
        }

        if (codes != kAudioBlock[frame]) {
            const bool tie = std::any_of(frame_logits.begin(), frame_logits.end(), [](const auto & values) { return margin(values) < kNearTie; });
            checks.expect(tie, "greedy frame " + std::to_string(frame) + " differs only at a near-tie", join(codes));
            break;
        }

        ++compared;
    }

    std::cout << "greedy frames identical: " << compared << " of " << kAudioBlock.size() << "\n";
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> open_session(
    engine::runtime::ILoadedVoiceModel & model,
    engine::runtime::VoiceTaskKind task,
    const std::string & model_gguf,
    const engine::core::BackendConfig & backend,
    engine::runtime::RunMode mode = engine::runtime::RunMode::Offline) {
    engine::runtime::SessionOptions options;
    options.backend = backend;
    options.options["lfm2_audio.model_gguf"] = model_gguf;
    auto session = model.create_task_session({task, mode}, options);
    if (auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get())) {
        offline->prepare({});
    }

    return session;
}

engine::runtime::TaskResult run(engine::runtime::IVoiceTaskSession & session, const engine::runtime::TaskRequest & request) {
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(&session);
    if (offline == nullptr) {
        throw std::runtime_error("the session is not an IOfflineVoiceTaskSession");
    }

    return offline->run(request);
}

// Replies through the registry, and their audio back through ASR.
void check_replies(
    const std::filesystem::path & model_dir,
    const std::filesystem::path & spec_override,
    const std::string & model_gguf,
    const engine::core::BackendConfig & backend,
    bool full_precision,
    Checks & checks) {
    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_dir;
    load_request.model_spec_override = spec_override;
    load_request.family_hint = "lfm2_audio";
    auto model = registry.load(load_request);
    auto s2s = open_session(*model, engine::runtime::VoiceTaskKind::SpeechToSpeech, model_gguf, backend);
    auto asr = open_session(*model, engine::runtime::VoiceTaskKind::Asr, model_gguf, backend);

    const auto wav = engine::audio::read_wav_f32(repo_path(kAudio));
    // The README's sampling, temperature 1.0 and top-k 4, by default.
    const auto question = [&](const std::string & seed = kReplySeeds[0]) {
        engine::runtime::TaskRequest request;
        request.audio_input = engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
        request.options["seed"] = seed;
        return request;
    };

    const auto reply = run(*s2s, question());
    const bool has_audio = reply.audio_output.has_value() && reply.audio_output->sample_rate == 24000 && !reply.audio_output->samples.empty();
    const std::string text = reply.text_output.has_value() ? reply.text_output->text : "";
    checks.expect(has_audio, "the reply is 24 kHz audio");
    checks.expect(!words(text).empty() && text.find("<|") == std::string::npos, "the reply has text without markup", text);
    // Text is greedy, and the first block comes before any audio is sampled.
    // Quantized weights can pick other words for it: Q4_0 does, and Q8_0 is
    // within 0.1 of a tie on some of its tokens.
    if (full_precision) {
        checks.expect(text.rfind(kTextBlockText, 0) == 0, "the reply starts with the text block");
    }

    std::cout << "reply: " << text << "\n";
    if (has_audio) {
        const double seconds = static_cast<double>(reply.audio_output->samples.size()) / 24000.0;
        std::cout << "reply speech: " << seconds << " s\n";
        checks.expect(seconds > 2.0 && seconds < 80.0, "reply length", std::to_string(seconds));

        std::vector<double> errors;
        std::string heard_texts;
        const auto round_trip = [&](const std::string & seed, const engine::runtime::TaskResult * given) {
            std::string spoken_text;
            std::string heard_text;
            double error = std::numeric_limits<double>::infinity();
            try {
                const auto spoken = given != nullptr ? *given : run(*s2s, question(seed));
                spoken_text = spoken.text_output.has_value() ? spoken.text_output->text : "";
                engine::runtime::TaskRequest transcribe;
                transcribe.audio_input = spoken.audio_output;
                const auto heard = run(*asr, transcribe);
                heard_text = heard.text_output.has_value() ? heard.text_output->text : "";
                if (!words(spoken_text).empty()) {
                    error = word_error_rate(words(spoken_text), words(heard_text));
                }
            } catch (const std::exception & failure) {
                heard_text = std::string("(failed: ") + failure.what() + ")";
            }

            errors.push_back(error);
            std::cout << "seed " << seed << " word error rate " << error << "\n  wrote: " << spoken_text << "\n  heard: " << heard_text << "\n";
            heard_texts += (heard_texts.empty() ? "" : " | ") + heard_text;
        };

        round_trip(kReplySeeds[0], &reply);
        for (size_t i = 1; i < kReplySeeds.size(); ++i) {
            round_trip(kReplySeeds[i], nullptr);
        }

        std::sort(errors.begin(), errors.end());
        const double median = errors[errors.size() / 2];
        checks.expect(median <= kMaxReplyWordErrors, "the replies say what they write, median word error rate " + std::to_string(median), heard_texts);
    }

    const auto rejects = [&](const std::function<void(engine::runtime::TaskRequest &)> & edit, const std::string & needle, const std::string & label) {
        auto request = question();
        edit(request);
        std::string message = "<no error>";
        try {
            (void)run(*s2s, request);
        } catch (const std::exception & error) {
            message = error.what();
        }

        checks.expect(message.find(needle) != std::string::npos, "rejects " + label, message);
    };

    rejects([](auto & r) { r.audio_input.reset(); }, "requires audio_input", "a request without audio");
    rejects([](auto & r) { r.voice = engine::runtime::VoiceCondition{engine::runtime::VoiceReference{std::nullopt, std::string("uk_male")}, std::nullopt}; },
        "takes no voice", "a voice");
    rejects([](auto & r) { r.options["text_chunk_size"] = "64"; }, "does not take request option text_chunk_size", "a TTS option");
    rejects([](auto & r) { r.options["language"] = "ja"; }, "speaks en", "another language");

    // Streamed with the same seed: the same reply, its audio decoded a frame
    // at a time. The other sessions go first, so one set of weights is loaded.
    s2s.reset();
    asr.reset();
    auto streaming_session = open_session(*model, engine::runtime::VoiceTaskKind::SpeechToSpeech, model_gguf, backend, engine::runtime::RunMode::Streaming);
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(streaming_session.get());
    checks.expect(streaming != nullptr, "the s2s session streams");
    if (streaming == nullptr || !has_audio) {
        return;
    }

    checks.expect(streaming->streaming_policy().input == engine::runtime::StreamingInputKind::AudioChunks, "the stream takes audio chunks");
    auto request = question();
    streaming->start_stream(request);
    const auto & input = *request.audio_input;
    for (size_t start = 0; start < input.samples.size(); start += 4800) {
        engine::runtime::AudioChunk chunk;
        chunk.sample_rate = input.sample_rate;
        chunk.channels = input.channels;
        chunk.samples.assign(input.samples.begin() + static_cast<std::ptrdiff_t>(start),
            input.samples.begin() + static_cast<std::ptrdiff_t>(std::min(input.samples.size(), start + 4800)));
        (void)streaming->process_audio_chunk(chunk);
    }

    std::vector<float> streamed;
    std::string streamed_text;
    size_t events = 0;
    while (auto event = streaming->next_stream_event()) {
        ++events;
        if (event->audio_output.has_value()) {
            streamed.insert(streamed.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
        }

        if (event->partial_text.has_value()) {
            streamed_text += event->partial_text->text;
        }
    }

    const auto finished = streaming->finish_stream();
    checks.expect(finished.audio_output.has_value() && finished.audio_output->samples == streamed, "the stream's audio is its events");
    checks.expect(finished.text_output.has_value() && finished.text_output->text == streamed_text, "the stream's text is its events");
    checks.expect(streamed_text == text, "the streamed reply has the offline text", streamed_text);
    const auto & offline = reply.audio_output->samples;
    checks.expect(events == offline.size() / 1920 + 1, "one event per frame and one for the tail", std::to_string(events) + " events");
    checks.expect(streamed.size() == offline.size(), "the streamed reply has the offline length", std::to_string(streamed.size()));
    if (streamed.size() == offline.size()) {
        double difference = 0.0;
        double energy = 0.0;
        for (size_t i = 0; i < streamed.size(); ++i) {
            difference += (static_cast<double>(streamed[i]) - offline[i]) * (static_cast<double>(streamed[i]) - offline[i]);
            energy += static_cast<double>(offline[i]) * offline[i];
        }

        // The detokenizer runs in graphs of other sizes. Measured with F16 on
        // this 12 s reply: 7.5e-4 on the CPU, 4.4e-2 on CUDA, whose F16 kernels
        // round differently for each size. Quantized weights widen it: 0.15
        // with Q4_0 on CUDA. A stream that lost context would be off by the
        // signal itself.
        checks.expect_close(std::sqrt(difference / energy), 0.0, full_precision ? 0.1 : 0.3, "streamed against offline reply, relative RMS difference");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_dir = arg_value(argc, argv, "--model", repo_path("models/LFM2.5-Audio-1.5B-GGUF").string());
    const std::string model_gguf = arg_value(argc, argv, "--model-gguf", "LFM2.5-Audio-1.5B-F16.gguf");
    const std::filesystem::path spec_override =
        arg_value(argc, argv, "--model-spec-override", repo_path("model_specs").string());
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    const int threads = std::atoi(arg_value(argc, argv, "--threads", "4").c_str());
    if (backend_name != "cpu" && backend_name != "best") {
        std::cerr << "FAIL: --backend must be cpu or best\n";
        return kExitFail;
    }

    for (const auto & prefix : {"", "mmproj-", "vocoder-", "tokenizer-"}) {
        if (!engine::io::is_existing_file(model_dir / (std::string(prefix) + model_gguf))) {
            std::fprintf(
                stderr,
                "SKIP: test_lfm2_audio_s2s needs %s and its mmproj-, vocoder- and tokenizer- files in '%s'.\n"
                "      python3 tools/model_manager_v2.py install lfm2_audio_1_5b_f16 --models-root models\n",
                model_gguf.c_str(),
                model_dir.string().c_str());
            return kExitSkip;
        }
    }

    engine::core::BackendConfig backend;
    backend.type = backend_name == "cpu" ? engine::core::BackendType::Cpu : engine::core::BackendType::BestAvailable;
    backend.threads = threads > 0 ? threads : 1;

    // Quantized weights move the stage numbers and the greedy text by design;
    // their replies are still checked.
    const bool full_precision = model_gguf.find("-F16.") != std::string::npos || model_gguf.find("-F32.") != std::string::npos;
    Checks checks;
    try {
        {
            const auto assets = lfm2::load_lfm2_audio_assets(model_dir);
            const auto components = lfm2::load_lfm2_audio_components(*assets, model_gguf, "");
            const auto output = lfm2::load_lfm2_audio_output_components(*assets, *components, "", "");
            check_prompt(*components, *output, checks);
            if (full_precision) {
                check_stage_numbers(*components, *output, backend, checks);
            } else {
                std::cout << "SKIP stage numbers: " << model_gguf << " is quantized\n";
            }
        }

        check_replies(model_dir, spec_override, model_gguf, backend, full_precision, checks);
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    std::cout << (checks.failures() == 0 ? "PASS" : "FAIL") << ": " << checks.failures() << " failed checks\n";
    return checks.failures() == 0 ? kExitPass : kExitFail;
}
