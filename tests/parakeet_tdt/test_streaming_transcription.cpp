#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
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
constexpr const char* kExpectedText =
    "Well, I don't wish to see it any more, observed Phoebe, turning away "
    "her eyes. It is certainly very like the old portrait.";

std::filesystem::path repo_path(const std::string& relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

std::string arg_value(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

engine::runtime::TaskResult run_stream(
    engine::runtime::IStreamingVoiceTaskSession& session,
    const engine::audio::WavData& wav,
    bool& saw_partial) {
    engine::runtime::TaskRequest stream_request;
    session.start_stream(stream_request);
    constexpr size_t chunk_samples = 8000;
    for (size_t offset = 0; offset < wav.samples.size(); offset += chunk_samples) {
        const size_t count = std::min(chunk_samples, wav.samples.size() - offset);
        engine::runtime::AudioChunk chunk;
        chunk.sample_rate = wav.sample_rate;
        chunk.channels = 1;
        chunk.start_sample = static_cast<int64_t>(offset);
        chunk.samples.assign(
            wav.samples.begin() + static_cast<std::ptrdiff_t>(offset),
            wav.samples.begin() + static_cast<std::ptrdiff_t>(offset + count));
        const auto event = session.process_audio_chunk(chunk);
        saw_partial = saw_partial ||
            (event.partial_text.has_value() && !event.partial_text->text.empty());
    }
    return session.finalize();
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path model_path = arg_value(
        argc, argv, "--model", repo_path("models/parakeet-tdt-0.6b-v3").string());
    const std::filesystem::path audio_path = arg_value(
        argc, argv, "--audio", repo_path("tests/parakeet_tdt/assets/2086-149220-0033.wav").string());
    const std::string weight_type = arg_value(argc, argv, "--weight-type", "q8_0");
    const bool model_available =
        engine::io::is_existing_file(model_path) ||
        engine::io::is_existing_file(model_path / "config.json");
    if (!model_available ||
        !engine::io::is_existing_file(audio_path)) {
        std::fprintf(stderr, "SKIP: Parakeet streaming test requires model weights\n");
        return kExitSkip;
    }

    try {
        const auto wav = engine::audio::read_wav_f32(audio_path);
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "parakeet_tdt";
        auto model = registry.load(load_request);
        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Asr,
            engine::runtime::RunMode::Streaming,
        };
        engine::runtime::SessionOptions options;
        options.backend.type = engine::core::BackendType::Cpu;
        options.backend.threads = 12;
        options.options["parakeet_tdt.matmul_weight_type"] = weight_type;
        options.options["parakeet_tdt.audio_chunk_duration_sec"] = "2";
        options.options["parakeet_tdt.left_context_sec"] = "2";
        options.options["parakeet_tdt.right_context_sec"] = "1";

        auto require_rejected_session_option = [&](
                                                   const engine::runtime::TaskSpec& invalid_task,
                                                   const std::string& key,
                                                   const std::string& value) {
            auto invalid_options = options;
            invalid_options.options[key] = value;
            bool rejected = false;
            try {
                (void)model->create_task_session(invalid_task, invalid_options);
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error(
                    "invalid session option was accepted: " + key + "=" + value);
            }
        };
        require_rejected_session_option(
            task,
            "parakeet_tdt.full_context_max_duration_sec",
            "30");
        require_rejected_session_option(
            task,
            "parakeet_tdt.audio_chunk_duration_sec",
            "0.0001");
        require_rejected_session_option(
            task,
            "parakeet_tdt.audio_chunk_threshold_sec",
            "0.0001");
        require_rejected_session_option(
            task,
            "parakeet_tdt.offline_mode",
            "typo");
        require_rejected_session_option(
            {
                engine::runtime::VoiceTaskKind::Asr,
                engine::runtime::RunMode::Offline,
            },
            "parakeet_tdt.streaming_attention_mode",
            "typo");

        auto base = model->create_task_session(task, options);
        auto* session =
            dynamic_cast<engine::runtime::IStreamingVoiceTaskSession*>(base.get());
        if (session == nullptr) {
            throw std::runtime_error("model did not create a streaming session");
        }
        engine::runtime::TaskRequest prepare_request;
        prepare_request.audio_input = engine::runtime::AudioBuffer{
            wav.sample_rate,
            wav.channels,
            wav.samples,
        };
        session->prepare(engine::runtime::build_preparation_request(prepare_request));

        session->start_stream({});
        bool discontinuity_failed = false;
        try {
            (void)session->process_audio_chunk({
                wav.sample_rate,
                1,
                1,
                {},
            });
        } catch (const std::exception&) {
            discontinuity_failed = true;
        }
        if (!discontinuity_failed) {
            throw std::runtime_error("non-contiguous stream chunk did not fail");
        }

        bool saw_partial = false;
        const auto first = run_stream(*session, wav, saw_partial);
        if (!saw_partial) {
            throw std::runtime_error("stream produced no partial result before finalize");
        }
        const std::string first_text =
            first.text_output.has_value() ? first.text_output->text : "";
        if (first_text != kExpectedText) {
            throw std::runtime_error("buffered-streaming final transcript mismatch");
        }

        bool repeated_finalize_failed = false;
        try {
            (void)session->finalize();
        } catch (const std::exception&) {
            repeated_finalize_failed = true;
        }
        if (!repeated_finalize_failed) {
            throw std::runtime_error("repeated finalize did not fail");
        }
        bool post_finalize_chunk_failed = false;
        try {
            (void)session->process_audio_chunk({
                wav.sample_rate,
                1,
                static_cast<int64_t>(wav.samples.size()),
                {},
            });
        } catch (const std::exception&) {
            post_finalize_chunk_failed = true;
        }
        if (!post_finalize_chunk_failed) {
            throw std::runtime_error("post-finalize chunk did not fail");
        }

        bool second_saw_partial = false;
        const auto second = run_stream(*session, wav, second_saw_partial);
        const std::string second_text =
            second.text_output.has_value() ? second.text_output->text : "";
        if (second_text != first_text || !second_saw_partial) {
            throw std::runtime_error("reset stream did not reproduce the first result");
        }

        std::printf("PASS: buffered streaming partials, finalize, and reset are stable\n");
        return kExitPass;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        return kExitFail;
    }
}
