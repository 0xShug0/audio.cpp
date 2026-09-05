// End-to-end probe for the ported VibeASR pipeline: I8_S VAE encoder -> ternary
// I2_S Qwen2 decoder -> transcript.
//
// The package ships two GGUFs, so --model points at the LM GGUF and the spec is
// resolved from the repo (same convention as minimax_h3). Skips with 125 when
// the checkpoint is not installed.
//
// Upstream: https://github.com/microsoft/VibeASR.cpp

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

// LibriSpeech test-clean 6930-75918-0000, transcribed by VibeASR.cpp's own
// asr_infer --greedy on the same two GGUFs.
const char * kExpectedText = "Concord returned to its place amidst the tents.";

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

std::string normalize_text(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || std::isspace(static_cast<unsigned char>(ch))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return engine::io::trim_ascii_whitespace(std::move(out));
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_path = arg_value(
        argc, argv, "--model", repo_path("models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf").string());
    const std::filesystem::path spec_override = arg_value(
        argc, argv, "--model-spec-override", repo_path("model_specs").string());
    const std::filesystem::path audio_path = arg_value(
        argc, argv, "--audio",
        repo_path("assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav").string());
    const int threads = std::atoi(arg_value(argc, argv, "--threads", "4").c_str());

    if (!engine::io::is_existing_file(model_path) || !engine::io::is_existing_file(audio_path)) {
        std::fprintf(
            stderr,
            "SKIP: test_vibeasr_asr needs the LM GGUF at '%s' and audio at '%s'.\n"
            "      Fetch huggingface.co/microsoft/VibeVoice-ASR-BitNet and run\n"
            "      tools/community_models/convert_vibeasr_gguf.py --in-place on both GGUFs.\n",
            model_path.string().c_str(),
            audio_path.string().c_str());
        return kExitSkip;
    }

    try {
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.model_spec_override = spec_override;
        load_request.family_hint = "vibeasr";
        auto model = registry.load(load_request);

        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Asr,
            engine::runtime::RunMode::Offline,
        };
        engine::runtime::SessionOptions session_options;
        session_options.backend.threads = threads > 0 ? threads : 1;

        auto session = model->create_task_session(task, session_options);
        auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            std::cerr << "FAIL: VibeASR session is not an IOfflineVoiceTaskSession\n";
            return kExitFail;
        }

        const auto wav = engine::audio::read_wav_f32(audio_path);
        engine::runtime::AudioBuffer audio;
        audio.sample_rate = wav.sample_rate;
        audio.channels = wav.channels;
        audio.samples = wav.samples;

        offline->prepare(engine::runtime::build_preparation_request(audio));

        engine::runtime::TaskRequest request;
        request.audio_input = audio;
        const auto result = offline->run(request);

        if (!result.text_output.has_value()) {
            std::cerr << "FAIL: VibeASR produced no text output\n";
            return kExitFail;
        }
        const std::string actual = result.text_output->text;
        std::cout << "transcript: " << actual << "\n";
        std::cout << "expected:   " << kExpectedText << "\n";

        // Raw equality pins punctuation and casing against the reference decode;
        // the normalized compare is only there to localize a failure.
        if (actual != kExpectedText) {
            if (normalize_text(actual) == normalize_text(kExpectedText)) {
                std::cerr << "FAIL: transcript differs only in punctuation or casing\n";
            } else {
                std::cerr << "FAIL: transcript mismatch\n";
            }
            return kExitFail;
        }

        std::cout << "PASS: VibeASR end-to-end transcription matches VibeASR.cpp\n";
        return kExitPass;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }
}
