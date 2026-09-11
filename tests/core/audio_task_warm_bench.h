#pragma once

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace engine::tools {

enum class AudioTaskOutputKind {
    Asr,
    Vad,
    Speaker,
};

struct AudioTaskBenchConfig {
    const char * family = "";
    const char * default_model = "";
    runtime::VoiceTaskKind task = runtime::VoiceTaskKind::Asr;
    AudioTaskOutputKind output_kind = AudioTaskOutputKind::Asr;
};

inline std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

inline int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

inline std::vector<std::pair<std::string, std::string>> session_option_args(int argc, char ** argv) {
    std::vector<std::pair<std::string, std::string>> out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != "--session-option") {
            continue;
        }
        const std::string option = argv[i + 1];
        const size_t sep = option.find('=');
        if (sep == std::string::npos || sep == 0) {
            throw std::runtime_error("invalid audio warmbench --session-option: " + option);
        }
        out.emplace_back(option.substr(0, sep), option.substr(sep + 1));
    }
    return out;
}

inline engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

inline std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

inline runtime::AudioBuffer read_audio_buffer(const std::filesystem::path & path) {
    const auto wav = audio::read_wav_f32(path);
    runtime::AudioBuffer buffer;
    buffer.sample_rate = wav.sample_rate;
    buffer.channels = wav.channels;
    buffer.samples = wav.samples;
    return buffer;
}

inline io::json::Value number(double value) {
    return io::json::Value::make_number(value);
}

inline io::json::Value string(std::string value) {
    return io::json::Value::make_string(std::move(value));
}

inline io::json::Value word_timestamps_json(const std::vector<runtime::WordTimestamp> & words) {
    io::json::Value::Array out;
    out.reserve(words.size());
    for (const auto & word : words) {
        out.push_back(io::json::Value::make_object({
            {"start_sample", number(static_cast<double>(word.span.start_sample))},
            {"end_sample", number(static_cast<double>(word.span.end_sample))},
            {"word", string(word.word)},
            {"confidence", number(word.confidence)},
        }));
    }
    return io::json::Value::make_array(std::move(out));
}

inline io::json::Value speech_segments_json(const std::vector<runtime::SpeechSegment> & segments) {
    io::json::Value::Array out;
    out.reserve(segments.size());
    for (const auto & segment : segments) {
        out.push_back(io::json::Value::make_object({
            {"start_sample", number(static_cast<double>(segment.span.start_sample))},
            {"end_sample", number(static_cast<double>(segment.span.end_sample))},
            {"confidence", number(segment.confidence)},
        }));
    }
    return io::json::Value::make_array(std::move(out));
}

inline io::json::Value speaker_turns_json(const std::vector<runtime::SpeakerTurn> & turns) {
    io::json::Value::Array out;
    out.reserve(turns.size());
    for (const auto & turn : turns) {
        out.push_back(io::json::Value::make_object({
            {"start_sample", number(static_cast<double>(turn.span.start_sample))},
            {"end_sample", number(static_cast<double>(turn.span.end_sample))},
            {"speaker_id", string(turn.speaker_id)},
            {"confidence", number(turn.confidence)},
        }));
    }
    return io::json::Value::make_array(std::move(out));
}

inline io::json::Value result_step_json(
    const AudioTaskBenchConfig & config,
    const runtime::TaskResult & result,
    int request_index,
    const std::filesystem::path & audio_path,
    double wall_ms) {
    io::json::Value::Object step{
        {"request_index", number(request_index)},
        {"audio", string(audio_path.string())},
        {"metrics", io::json::Value::make_object({{"wall_ms", number(wall_ms)}})},
    };
    switch (config.output_kind) {
    case AudioTaskOutputKind::Asr:
        step.emplace(
            "text_output",
            string(result.text_output.has_value() ? result.text_output->text : ""));
        step.emplace("word_timestamps", word_timestamps_json(result.word_timestamps));
        break;
    case AudioTaskOutputKind::Vad:
        step.emplace("speech_segments", speech_segments_json(result.speech_segments));
        break;
    case AudioTaskOutputKind::Speaker:
        {
        const runtime::VoiceArtifact * speaker_artifact = nullptr;
        for (const auto & artifact : result.output_artifacts) {
            if (artifact.kind == runtime::ArtifactKind::SpeakerEmbedding || artifact.kind == runtime::ArtifactKind::Custom) {
                speaker_artifact = &artifact;
                break;
            }
        }
        const auto & meta = speaker_artifact != nullptr ? speaker_artifact->meta : std::unordered_map<std::string, std::string>{};
        const auto label = meta.find("label");
        const auto index = meta.find("index");
        const auto score = meta.find("score");
        step.emplace("label", string(label != meta.end() ? label->second : ""));
        step.emplace(
            "index",
            number(index != meta.end() ? std::stod(index->second) : -1.0));
        step.emplace(
            "score",
            number(score != meta.end() ? std::stod(score->second) : 0.0));
        }
        break;
    }
    return io::json::Value::make_object(std::move(step));
}

inline void set_process_env(const char * key, const std::string & value) {
#if defined(_WIN32)
    if (_putenv_s(key, value.c_str()) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#else
    if (setenv(key, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#endif
}

inline int run_audio_task_warm_bench(int argc, char ** argv, const AudioTaskBenchConfig & config) {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", config.default_model);
    const std::string audio_sequence_value = arg_value(argc, argv, "--audio-sequence", "");
    const std::filesystem::path audio_path = arg_value(argc, argv, "--audio", "resources/sample.wav");
    const std::filesystem::path warmup_audio_path = arg_value(argc, argv, "--warmup-audio", audio_path.string());
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    const int device = int_arg(argc, argv, "--device", 0);
    const int threads = int_arg(argc, argv, "--threads", 8);
    const int warmup = int_arg(argc, argv, "--warmup", 1);
    const int iterations = int_arg(argc, argv, "--iterations", 1);
    const std::filesystem::path timing_path = arg_value(
        argc,
        argv,
        "--timing-file",
        (std::filesystem::path("build/logs/warmbench") / (std::string(config.family) + "_timing.log")).string());

    if (!timing_path.parent_path().empty()) {
        std::filesystem::create_directories(timing_path.parent_path());
    }
    set_process_env("ENGINE_TRACE_ENABLED", "0");
    set_process_env("ENGINE_TIMING_ENABLED", "1");
    set_process_env("ENGINE_TIMING_FILE", timing_path.string());

    auto registry = runtime::make_default_registry();
    runtime::ModelLoadRequest load_request;
    load_request.model_path = model_path;
    load_request.family_hint = config.family;
    auto model = registry.load(load_request);

    runtime::SessionOptions session_options;
    session_options.backend.type = parse_backend(backend_name);
    session_options.backend.device = device;
    session_options.backend.threads = threads;
    for (const auto & [key, value] : session_option_args(argc, argv)) {
        session_options.options[key] = value;
    }

    runtime::TaskSpec task_spec;
    task_spec.task = config.task;
    task_spec.mode = runtime::RunMode::Offline;
    auto session_base = model->create_task_session(task_spec, session_options);
    auto * session = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session_base.get());
    if (session == nullptr) {
        throw std::runtime_error(std::string(config.family) + " did not create an offline task session");
    }

    const auto warmup_audio = read_audio_buffer(warmup_audio_path);
    session->prepare(runtime::build_preparation_request(warmup_audio));
    runtime::TaskRequest request;
    request.audio_input = warmup_audio;
    for (int i = 0; i < warmup; ++i) {
        (void) session->run(request);
    }

    std::vector<std::filesystem::path> request_paths;
    if (!audio_sequence_value.empty()) {
        for (const auto & item : split_csv(audio_sequence_value)) {
            request_paths.emplace_back(item);
        }
    } else {
        request_paths.push_back(audio_path);
    }
    if (request_paths.empty()) {
        throw std::runtime_error("audio request sequence is empty");
    }

    io::json::Value::Array steps;
    steps.reserve(request_paths.size());
    for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
        const auto audio = read_audio_buffer(request_paths[request_index]);
        runtime::TaskRequest run_request;
        run_request.audio_input = audio;
        runtime::TaskResult last_result;
        double total_ms = 0.0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            last_result = session->run(run_request);
            const auto ended = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
        }
        const double wall_ms = total_ms / static_cast<double>(iterations);
        std::cout << "average[" << request_index << "]\n";
        std::cout << config.family << ".wall_ms=" << wall_ms << "\n";
        steps.push_back(result_step_json(
            config,
            last_result,
            static_cast<int>(request_index),
            request_paths[request_index],
            wall_ms));
    }

    const auto summary = io::json::Value::make_object({
        {"family", string(config.family)},
        {"backend", string(backend_name)},
        {"sequence_steps", io::json::Value::make_array(std::move(steps))},
    });
    std::cout << "summary_json=" << io::json::stringify(summary) << "\n";
    return 0;
}

}  // namespace engine::tools
