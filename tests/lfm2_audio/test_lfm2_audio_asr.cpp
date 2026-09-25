// End-to-end LFM2.5-Audio ASR on Liquid's published GGUFs: registry load, the
// bundled 16 kHz LibriSpeech clips, and greedy transcripts compared with
// liquid-audio's.
//
// --model is the directory of LiquidAI/LFM2.5-Audio-1.5B-GGUF (default
// models/lfm2_audio); --model-gguf picks the backbone. Skips with 125 when the
// files are not there.
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

// liquid-audio 1.3.0, LFM2.5-Audio-1.5B revision
// c362a0625dfe45aa588dce5f0ada28a7e5707628, fp32 on CPU, greedy
// generate_sequential with "Perform ASR.". The F16 and Q8_0 GGUFs produce the
// same tokens.
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

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_dir = arg_value(argc, argv, "--model", repo_path("models/lfm2_audio").string());
    const std::string model_gguf = arg_value(argc, argv, "--model-gguf", "LFM2.5-Audio-1.5B-F16.gguf");
    const std::filesystem::path spec_override =
        arg_value(argc, argv, "--model-spec-override", repo_path("model_specs").string());
    const std::string backend = arg_value(argc, argv, "--backend", "cpu");
    const int threads = std::atoi(arg_value(argc, argv, "--threads", "4").c_str());
    if (backend != "cpu" && backend != "best") {
        std::cerr << "FAIL: --backend must be cpu or best\n";
        return kExitFail;
    }

    if (!engine::io::is_existing_file(model_dir / model_gguf) ||
        !engine::io::is_existing_file(model_dir / ("mmproj-" + model_gguf))) {
        std::fprintf(
            stderr,
            "SKIP: test_lfm2_audio_asr needs %s and its mmproj- file in '%s'.\n"
            "      hf download LiquidAI/LFM2.5-Audio-1.5B-GGUF --local-dir models/lfm2_audio\n",
            model_gguf.c_str(),
            model_dir.string().c_str());
        return kExitSkip;
    }

    try {
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_dir;
        load_request.model_spec_override = spec_override;
        load_request.family_hint = "lfm2_audio";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type =
            backend == "cpu" ? engine::core::BackendType::Cpu : engine::core::BackendType::BestAvailable;
        session_options.backend.threads = threads > 0 ? threads : 1;
        session_options.options["lfm2_audio.model_gguf"] = model_gguf;
        auto session = model->create_task_session(
            {engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline}, session_options);
        auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            std::cerr << "FAIL: the LFM2-Audio session is not an IOfflineVoiceTaskSession\n";
            return kExitFail;
        }

        offline->prepare({});

        int failures = 0;
        for (const auto & test_case : kCases) {
            const auto wav = engine::audio::read_wav_f32(repo_path(test_case.audio));
            engine::runtime::TaskRequest request;
            request.audio_input = engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
            const auto result = offline->run(request);
            const std::string actual = result.text_output.has_value() ? result.text_output->text : "<no transcript>";
            const bool pass = actual == test_case.expected;
            std::cout << (pass ? "PASS " : "FAIL ") << std::filesystem::path(test_case.audio).filename().string() << "\n"
                      << "  transcript: " << actual << "\n"
                      << "  expected:   " << test_case.expected << "\n";
            failures += pass ? 0 : 1;
        }

        return failures == 0 ? kExitPass : kExitFail;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }
}
