#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
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

const char * kExpectedText = "concord returned to its place amidst the tents";

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

std::string normalize_text(std::string text) {
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
        argc, argv, "--model", repo_path("granite5asr").string());
    const std::filesystem::path audio_path = arg_value(
        argc, argv, "--audio", repo_path("assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav").string());
    const std::string weight_type = arg_value(argc, argv, "--weight-type", "");

    const bool model_available =
        engine::io::is_existing_file(model_path) ||
        engine::io::is_existing_file(model_path / "config.json");
    if (!model_available || !engine::io::is_existing_file(audio_path)) {
        std::fprintf(
            stderr,
            "SKIP: test_granite5asr_golden_transcription requires model weights at '%s' "
            "and test audio at '%s'.\n",
            model_path.string().c_str(),
            audio_path.string().c_str());
        return kExitSkip;
    }

    try {
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "granite5asr";
        auto model = registry.load(load_request);

        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Asr,
            engine::runtime::RunMode::Offline,
        };

        engine::runtime::SessionOptions session_options;
        if (!weight_type.empty()) {
            session_options.options["granite5asr.weight_type"] = weight_type;
        }

        auto session = model->create_task_session(task, session_options);
        auto * offline_session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
        if (!offline_session) {
            std::cerr << "FAIL: session is not an IOfflineVoiceTaskSession\n";
            return kExitFail;
        }

        const auto wav_data = engine::audio::read_wav_f32(audio_path);
        engine::runtime::AudioBuffer audio;
        audio.sample_rate = wav_data.sample_rate;
        audio.channels = wav_data.channels;
        audio.samples = wav_data.samples;

        const auto prep = engine::runtime::build_preparation_request(audio);
        offline_session->prepare(prep);

        engine::runtime::TaskRequest request;
        request.audio_input = audio;
        const auto result = offline_session->run(request);

        if (!result.text_output.has_value()) {
            std::cerr << "FAIL: Granite 5 ASR produced no text output\n";
            return kExitFail;
        }

        const std::string actual_raw = result.text_output->text;
        const std::string actual = normalize_text(actual_raw);
        const std::string expected = normalize_text(kExpectedText);

        std::cout << "Raw transcript:      " << actual_raw << "\n";
        std::cout << "Normalized actual:   " << actual << "\n";
        std::cout << "Normalized expected: " << expected << "\n";

        if (actual != expected) {
            std::cerr << "FAIL: transcript mismatch!\n";
            return kExitFail;
        }

        std::cout << "PASS: Granite 5 ASR golden transcription verified successfully.\n";
        return kExitPass;
    } catch (const std::exception & ex) {
        std::cerr << "FAIL: exception: " << ex.what() << "\n";
        return kExitFail;
    }
}
