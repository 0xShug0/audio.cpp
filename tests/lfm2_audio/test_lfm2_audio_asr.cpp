// LFM2.5-Audio ASR on Liquid's published GGUFs against liquid-audio 1.3.0
// (LFM2.5-Audio-1.5B revision c362a0625dfe45aa588dce5f0ada28a7e5707628,
// fp32 on CPU):
// - the prompt ChatState builds and the tokenizer on tricky text;
// - for F16/F32 GGUFs, the numbers at each stage for one clip: features,
//   adapter output, prompt length, the first step's log-probabilities and the
//   greedy tokens;
// - end-to-end transcripts of the bundled 16 kHz LibriSpeech clips through the
//   registry.
//
// --model is the directory of LiquidAI/LFM2.5-Audio-1.5B-GGUF (default
// models/LFM2.5-Audio-1.5B-GGUF, where the lfm2_audio_1_5b_* packages install);
// --model-gguf picks the backbone. Skips with 125 when the files are not there.
#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
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

// ChatState's ids before and after the audio. The EN and JP checkpoints share
// one tokenizer.json, so the EN GGUF checks both prompts.
const std::vector<int32_t> kPrefixEn = {1, 6, 24131, 708, 8173, 1199, 11866, 559, 523, 7, 708, 6, 6423, 708};
const std::vector<int32_t> kPrefixJa = {1, 6, 24131, 708, 8173, 1199, 11866, 559, 797, 41035, 3391, 523, 7, 708, 6, 6423, 708};
const std::vector<int32_t> kSuffix = {7, 708, 6, 64015, 708};

// The HF tokenizer's ids for text the Llama 3 split, the byte fallback and the
// added tokens all have to get right.
struct TokenizerCase {
    const char * text;
    std::vector<int32_t> ids;
};

const TokenizerCase kTokenizerCases[] = {
    {"Concord returned to its place amidst the tents.", {2414, 38107, 7444, 811, 1352, 2380, 34309, 820, 779, 766, 1104, 523}},
    {"I'm from the cutter, lying off the coast.", {550, 6217, 988, 779, 4169, 886, 521, 21926, 1612, 779, 7457, 523}},
    {"DON'T you think it's 12345 or 3.14159?", {545, 2332, 32640, 1010, 2539, 936, 1090, 730, 10293, 2637, 933, 730, 528, 523, 13888, 5599, 540}},
    {"  leading spaces,\ttabs\nand\n\nnewlines  ", {730, 5845, 12914, 521, 15153, 7954, 708, 905, 509, 3514, 10890, 767}},
    {"私はあの先生に日本語を習っています。", {27982, 2318, 1084, 29122, 1262, 62506, 1414, 22611, 16664, 1147}},
    {"東京タワー 333m、1958年", {20374, 3028, 44908, 730, 21141, 586, 1096, 2081, 533, 2387}},
    {"python and Mathias write pythonic code", {64014, 810, 730, 64011, 5773, 730, 64014, 793, 5214}},
    {"<|im_start|>user\nhello<|im_end|>\n", {6, 6423, 708, 52572, 7, 708}},
    {"emoji 😀 and café naïve", {39724, 3966, 54342, 732, 810, 35499, 2116, 6838, 1124}},
    {"e.g. U.S.A. -- well...", {578, 3748, 523, 1003, 2420, 4921, 523, 3977, 1714, 2597}},
};

// Stage numbers for librispeech_test_clean_6930-75918-0000.
constexpr int64_t kMelFrames = 351;

struct MelPoint {
    int64_t bin;
    int64_t frame;
    float value;
};

const MelPoint kMelPoints[] = {
    {3, 10, -0.776207f},
    {20, 55, -1.004136f},
    {47, 120, 0.977365f},
    {64, 200, -0.496402f},
    {100, 300, 0.568487f},
    {127, 340, -0.784139f},
};

const float kAdapterRowNorms[] = {
    12.88882f, 11.90990f, 10.57765f, 10.04871f, 9.53111f, 9.51655f, 8.99367f, 7.00964f,
    5.10602f, 4.34605f, 4.37357f, 4.79834f, 6.42083f, 8.72848f, 8.65136f, 9.82372f,
    6.65207f, 3.71349f, 5.37129f, 10.63032f, 8.83541f, 6.05571f, 6.55437f, 5.96310f,
    7.63172f, 3.18646f, 3.85922f, 7.92687f, 7.93696f, 6.90148f, 4.28253f, 4.31148f,
    6.76868f, 8.85598f, 7.16025f, 8.98978f, 5.32113f, 3.86787f, 4.42683f, 7.94297f,
    7.70101f, 8.85497f, 10.78221f, 5.20597f,
};

constexpr size_t kPromptLength = 63;

struct TopToken {
    int32_t id;
    double logprob;
};

const TopToken kTop5[] = {
    {2414, -0.39871},
    {49084, -1.13826},
    {544, -5.81231},
    {44570, -6.85481},
    {1098, -7.21757},
};

const std::vector<int32_t> kTokens = {2414, 38107, 7444, 811, 1352, 2380, 34309, 820, 779, 766, 1104, 523};

// End-to-end transcripts, liquid-audio's greedy generate_sequential.
struct Case {
    const char * audio;
    const char * expected;
};

const Case kCases[] = {
    {"assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav",
     "Concord returned to its place amidst the tents."},
    {"assets/asr_validation/librispeech/librispeech_test_other_7902-96591-0000.wav",
     "I'm from the cutter lying off the coast."},
};

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

// Newlines and tabs spelled out, so each check prints on one line.
std::string printable(const std::string & text) {
    std::string out;
    for (const char ch : text) {
        out += ch == '\n' ? "\\n" : ch == '\t' ? "\\t" : std::string(1, ch);
    }

    return out;
}

class Checks {
public:
    void expect(bool ok, const std::string & label, const std::string & detail = "") {
        std::cout << (ok ? "PASS " : "FAIL ") << label << (ok || detail.empty() ? "" : ": " + detail) << "\n";
        failures_ += ok ? 0 : 1;
    }

    void expect_ids(const std::vector<int32_t> & actual, const std::vector<int32_t> & expected, const std::string & label) {
        expect(actual == expected, label, "got " + join(actual) + ", expected " + join(expected));
    }

    void expect_close(double actual, double expected, double tolerance, const std::string & label) {
        // Printed on a pass too, so drift towards the tolerance is visible.
        std::ostringstream detail;
        detail << label << ": got " << actual << ", expected " << expected << " +- " << tolerance;
        expect(std::fabs(actual - expected) <= tolerance, detail.str());
    }

    [[nodiscard]] int failures() const { return failures_; }

private:
    int failures_ = 0;
};

void check_prompt_and_tokenizer(const lfm2::Lfm2AudioComponents & components, Checks & checks) {
    const lfm2::Lfm2TextTokenizer tokenizer(components.vocabulary);
    const auto en = lfm2::make_lfm2_asr_prompt(tokenizer, "en");
    const auto ja = lfm2::make_lfm2_asr_prompt(tokenizer, "ja");
    checks.expect_ids(en.prefix, kPrefixEn, "English prompt before the audio");
    checks.expect_ids(ja.prefix, kPrefixJa, "Japanese prompt before the audio");
    checks.expect_ids(en.suffix, kSuffix, "prompt after the audio");
    checks.expect_ids(en.stop_token_ids, {7, 128}, "stop tokens");

    for (const auto & test_case : kTokenizerCases) {
        const auto ids = tokenizer.encode(test_case.text);
        checks.expect_ids(ids, test_case.ids, "tokenize \"" + printable(test_case.text) + "\"");
        // Markup is dropped on decode; everything else comes back unchanged.
        const std::string text = test_case.text;
        if (text.find("<|") == std::string::npos) {
            checks.expect(tokenizer.decode(ids) == text, "decode \"" + printable(text) + "\"", printable(tokenizer.decode(ids)));
        }
    }
}

// F16 weights keep the adapter within 0.4% and the first step's
// log-probabilities within 0.06 on every backend measured.
void check_stage_numbers(
    const lfm2::Lfm2AudioComponents & components, const engine::core::BackendConfig & backend, Checks & checks) {
    engine::core::ExecutionContext execution(backend);
    const lfm2::Lfm2TextTokenizer tokenizer(components.vocabulary);
    const auto prompt = lfm2::make_lfm2_asr_prompt(tokenizer, "en");
    const auto wav = engine::audio::read_wav_f32(repo_path(kCases[0].audio));
    const auto samples = lfm2::lfm2_audio_mono_16k({wav.sample_rate, wav.channels, wav.samples});

    const auto features = lfm2::Lfm2AudioFeatureExtractor(components.encoder.n_mels, backend.threads).extract(samples);
    checks.expect(features.frames == kMelFrames, "feature frames", std::to_string(features.frames));
    if (features.frames == kMelFrames) {
        for (const auto & point : kMelPoints) {
            checks.expect_close(features.values[static_cast<size_t>(point.bin * features.frames + point.frame)], point.value, 2e-3,
                "feature [" + std::to_string(point.bin) + "][" + std::to_string(point.frame) + "]");
        }
    }

    lfm2::Lfm2FastConformerEncoderRuntime encoder(components.mmproj, components.encoder, execution);
    const auto audio = encoder.encode(features);
    const auto rows = static_cast<int64_t>(std::size(kAdapterRowNorms));
    checks.expect(audio.tokens == rows, "adapter rows", std::to_string(audio.tokens));
    if (audio.tokens == rows) {
        double worst = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            const auto begin = audio.values.begin() + r * audio.hidden_size;
            const double norm = std::sqrt(std::inner_product(begin, begin + audio.hidden_size, begin, 0.0));
            worst = std::max(worst, std::fabs(norm / kAdapterRowNorms[r] - 1.0));
        }

        checks.expect_close(worst, 0.0, 1e-2, "adapter row norms, worst relative error");
    }

    // The runtime keeps its graph between chunks: the same chunk again, and
    // again after a chunk of another length, gives the same adapter output.
    checks.expect(encoder.encode(features).values == audio.values, "adapter output from the kept graph");
    const std::vector<float> head(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2));
    (void)encoder.encode(lfm2::Lfm2AudioFeatureExtractor(components.encoder.n_mels, backend.threads).extract(head));
    checks.expect(encoder.encode(features).values == audio.values, "adapter output after a chunk of another length");

    lfm2::Lfm2BackboneRuntime backbone(components.model, components.backbone, execution);
    const auto request = prompt.with_audio(audio.tokens);
    checks.expect(request.input_ids.size() == kPromptLength, "prompt length", std::to_string(request.input_ids.size()));
    const auto result = backbone.generate(request, audio, {128, prompt.stop_token_ids});

    const auto & logits = result.prefill_logits;
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float value : logits) {
        sum += std::exp(static_cast<double>(value) - max_logit);
    }

    const double log_norm = max_logit + std::log(sum);
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    for (size_t i = 0; i < std::size(kTop5); ++i) {
        checks.expect(order[i] == kTop5[i].id, "first step top-" + std::to_string(i + 1) + " token", std::to_string(order[i]));
        checks.expect_close(logits[kTop5[i].id] - log_norm, kTop5[i].logprob, 0.1,
            "first step log-probability of token " + std::to_string(kTop5[i].id));
    }

    checks.expect(result.stopped, "generation ends on a stop token");
    checks.expect_ids(result.tokens, kTokens, "greedy tokens");
}

void check_transcripts(
    const std::filesystem::path & model_dir,
    const std::filesystem::path & spec_override,
    const std::string & model_gguf,
    const engine::core::BackendConfig & backend,
    Checks & checks) {
    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_dir;
    load_request.model_spec_override = spec_override;
    load_request.family_hint = "lfm2_audio";
    auto model = registry.load(load_request);

    engine::runtime::SessionOptions session_options;
    session_options.backend = backend;
    session_options.options["lfm2_audio.model_gguf"] = model_gguf;
    auto session = model->create_task_session(
        {engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline}, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    checks.expect(offline != nullptr, "the session is an IOfflineVoiceTaskSession");
    if (offline == nullptr) {
        return;
    }

    offline->prepare({});
    for (const auto & test_case : kCases) {
        const auto wav = engine::audio::read_wav_f32(repo_path(test_case.audio));
        engine::runtime::TaskRequest request;
        request.audio_input = engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
        const auto result = offline->run(request);
        const std::string actual = result.text_output.has_value() ? result.text_output->text : "<no transcript>";
        checks.expect(actual == test_case.expected, "transcript of " + std::filesystem::path(test_case.audio).filename().string(),
            "got \"" + actual + "\", expected \"" + test_case.expected + "\"");
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

    if (!engine::io::is_existing_file(model_dir / model_gguf) ||
        !engine::io::is_existing_file(model_dir / ("mmproj-" + model_gguf))) {
        std::fprintf(
            stderr,
            "SKIP: test_lfm2_audio_asr needs %s and its mmproj- file in '%s'.\n"
            "      python3 tools/model_manager_v2.py install lfm2_audio_1_5b_f16 --models-root models\n",
            model_gguf.c_str(),
            model_dir.string().c_str());
        return kExitSkip;
    }

    engine::core::BackendConfig backend;
    backend.type = backend_name == "cpu" ? engine::core::BackendType::Cpu : engine::core::BackendType::BestAvailable;
    backend.threads = threads > 0 ? threads : 1;

    Checks checks;
    try {
        {
            const auto components = lfm2::load_lfm2_audio_components(*lfm2::load_lfm2_audio_assets(model_dir), model_gguf, "");
            check_prompt_and_tokenizer(*components, checks);
            // Quantized weights move these numbers by design; their transcripts
            // are still checked below.
            const bool full_precision = model_gguf.find("-F16.") != std::string::npos || model_gguf.find("-F32.") != std::string::npos;
            if (full_precision) {
                check_stage_numbers(*components, backend, checks);
            } else {
                std::cout << "SKIP stage numbers: " << model_gguf << " is quantized\n";
            }
        }

        check_transcripts(model_dir, spec_override, model_gguf, backend, checks);
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    std::cout << (checks.failures() == 0 ? "PASS" : "FAIL") << ": " << checks.failures() << " failed checks\n";
    return checks.failures() == 0 ? kExitPass : kExitFail;
}
