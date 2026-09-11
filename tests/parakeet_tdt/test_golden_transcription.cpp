// Golden-transcription regression test for the Parakeet-TDT 0.6B v3 offline
// full-context and bounded-window long-form ASR paths.
//
// This is a cheap, deterministic stand-in for the numerical parity work done
// while debugging the encoder/frontend correctness bugs fixed in 4a10c48 and
// 3bf2d12: it runs the full model end to end against a fixed test clip and
// asserts the transcribed text matches the known-correct NeMo reference
// output exactly. It catches output-changing regressions cheaply, but it did
// not catch every numerical bug found during development: greedy decoding
// produced the same text for one folded-bias error despite measurable
// encoder drift.
//
// This is *not* a substitute for the numerical layer-by-layer parity
// comparison against NeMo used to diagnose those bugs (see
// the numerical parity harness in tests/parakeet_tdt/parity/ for that). It only
// catches regressions that are large enough to flip the final decoded text
// for this one clip; a subtle per-layer numerical drift that doesn't change
// the greedy-decoded token sequence would slip through. It is deliberately
// cheap enough to run on every change to this model, unlike the NeMo-based
// comparison, which needs a NeMo install and real model weights on top of
// what this test needs.
//
// Requires the real model weights and the checked-in fixture audio
// (tests/parakeet_tdt/assets/2086-149220-0033.wav, see that directory's
// README.md for provenance). --model accepts either the installed
// safetensors directory or a standalone GGUF. If the default model directory
// isn't present (e.g. a fresh checkout without models downloaded), this test
// SKIPs rather than failing, via SKIP_RETURN_CODE configured in CMakeLists.txt.

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

const char * kExpectedText =
    "Well, I don't wish to see it any more, observed Phoebe, turning away "
    "her eyes. It is certainly very like the old portrait.";

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
    const std::filesystem::path model_path = arg_value(
        argc, argv, "--model", repo_path("models/parakeet-tdt-0.6b-v3").string());
    const std::filesystem::path audio_path = arg_value(
        argc, argv, "--audio", repo_path("tests/parakeet_tdt/assets/2086-149220-0033.wav").string());
    const std::string weight_type = arg_value(argc, argv, "--weight-type", "");

    const bool model_available =
        engine::io::is_existing_file(model_path) ||
        engine::io::is_existing_file(model_path / "config.json");
    if (!model_available ||
        !engine::io::is_existing_file(audio_path)) {
        std::fprintf(
            stderr,
            "SKIP: parakeet_golden_transcription_test requires model weights at '%s' "
            "and test audio at '%s'; a model directory or standalone GGUF is accepted.\n",
            model_path.string().c_str(),
            audio_path.string().c_str());
        return kExitSkip;
    }

    try {
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "parakeet_tdt";
        auto model = registry.load(load_request);

        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Asr,
            engine::runtime::RunMode::Offline,
        };
        engine::runtime::SessionOptions session_options;
        session_options.backend.type = engine::core::BackendType::Cpu;
        if (!weight_type.empty()) {
            session_options.options["parakeet_tdt.matmul_weight_type"] = weight_type;
        }

        auto session_base = model->create_task_session(task, session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            std::fprintf(stderr, "FAIL: Parakeet TDT did not produce an offline ASR session\n");
            return kExitFail;
        }

        const auto wav = engine::audio::read_wav_f32(audio_path);
        if (wav.channels != 1) {
            std::fprintf(stderr, "FAIL: fixture audio must be mono\n");
            return kExitFail;
        }

        engine::runtime::TaskRequest request;
        request.audio_input = engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};

        session->prepare(engine::runtime::build_preparation_request(request));
        const auto result = session->run(request);

        const std::string actual = result.text_output.has_value() ? result.text_output->text : "";
        if (actual != kExpectedText) {
            std::fprintf(
                stderr,
                "FAIL: transcription mismatch\n  expected: \"%s\"\n  actual:   \"%s\"\n",
                kExpectedText,
                actual.c_str());
            return kExitFail;
        }

        if (result.word_timestamps.empty()) {
            std::fprintf(stderr, "FAIL: expected word timestamps\n");
            return kExitFail;
        }

        std::string reconstructed;
        int64_t previous_end = 0;
        const int64_t audio_samples = static_cast<int64_t>(wav.samples.size());
        for (const auto & timestamp : result.word_timestamps) {
            if (timestamp.word.empty() ||
                timestamp.word.find("\xE2\x96\x81") != std::string::npos) {
                std::fprintf(stderr, "FAIL: timestamp contains a token piece instead of a word\n");
                return kExitFail;
            }
            if (timestamp.span.start_sample < previous_end ||
                timestamp.span.end_sample < timestamp.span.start_sample ||
                timestamp.span.end_sample > audio_samples) {
                std::fprintf(stderr, "FAIL: word timestamp spans are not monotonic and bounded\n");
                return kExitFail;
            }
            if (!reconstructed.empty()) {
                reconstructed += ' ';
            }
            reconstructed += timestamp.word;
            previous_end = timestamp.span.end_sample;
        }
        if (reconstructed != actual) {
            std::fprintf(
                stderr,
                "FAIL: word timestamps do not reconstruct the transcript\n"
                "  expected: \"%s\"\n  actual:   \"%s\"\n",
                actual.c_str(),
                reconstructed.c_str());
            return kExitFail;
        }

        session_base.reset();
        session_options.options["parakeet_tdt.matmul_weight_type"] =
            weight_type.empty() ? "q8_0" : weight_type;
        session_options.options["parakeet_tdt.offline_mode"] = "long_form";
        session_options.options["parakeet_tdt.left_context_sec"] = "2";
        session_options.options["parakeet_tdt.right_context_sec"] = "1";
        auto long_form_base = model->create_task_session(task, session_options);
        auto* long_form =
            dynamic_cast<engine::runtime::IOfflineVoiceTaskSession*>(long_form_base.get());
        if (long_form == nullptr) {
            std::fprintf(stderr, "FAIL: Parakeet TDT did not produce a long-form session\n");
            return kExitFail;
        }
        long_form->prepare(engine::runtime::build_preparation_request(request));
        const auto long_form_result = long_form->run(request);
        const std::string long_form_text =
            long_form_result.text_output.has_value()
                ? long_form_result.text_output->text
                : "";
        if (long_form_text != kExpectedText) {
            std::fprintf(stderr, "FAIL: long-form transcription mismatch\n");
            return kExitFail;
        }

        std::printf(
            "PASS: full-context and long-form transcription plus word timestamps "
            "match expected reference output\n");
        return kExitPass;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return kExitFail;
    }
}
