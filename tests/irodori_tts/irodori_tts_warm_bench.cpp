#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

std::vector<std::pair<std::string, std::string>> parse_session_options(int argc, char ** argv) {
    std::vector<std::pair<std::string, std::string>> out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != "--session-option") {
            continue;
        }
        const std::string option = argv[i + 1];
        const size_t eq = option.find('=');
        if (eq == std::string::npos || eq == 0) {
            throw std::runtime_error("invalid Irodori-TTS --session-option: " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    throw std::runtime_error("Irodori-TTS warmbench is CUDA-only");
}

std::string optional_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::string{} : value->as_string();
}

std::string required_text(const engine::io::json::Value & object) {
    std::string value = optional_string(object, "text");
    if (value.empty()) {
        value = optional_string(object, "prompt");
    }
    if (value.empty()) {
        throw std::runtime_error("Irodori-TTS warmbench request missing text");
    }
    return value;
}

std::string option_text(const engine::io::json::Value & value) {
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(value.as_number());
    }
    return value.as_string();
}

void set_optional_option(
    engine::runtime::TaskRequest & request,
    const engine::io::json::Value & object,
    const std::string & key) {
    const auto * value = object.find(key);
    if (value != nullptr && !value->is_null()) {
        request.options[key] = option_text(*value);
    }
}

engine::runtime::AudioBuffer read_audio_buffer(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

engine::runtime::TaskRequest make_request(const engine::io::json::Value & object) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{required_text(object), "ja"};
    set_optional_option(request, object, "caption");
    set_optional_option(request, object, "no_ref");
    set_optional_option(request, object, "num_steps");
    set_optional_option(request, object, "cfg_scale_text");
    set_optional_option(request, object, "cfg_scale_caption");
    set_optional_option(request, object, "cfg_scale_speaker");
    set_optional_option(request, object, "cfg_guidance_mode");
    set_optional_option(request, object, "cfg_min_t");
    set_optional_option(request, object, "cfg_max_t");
    set_optional_option(request, object, "duration_scale");
    if (const auto * value = object.find("seconds"); value != nullptr && !value->is_null()) {
        request.options["duration_seconds"] = option_text(*value);
    }
    set_optional_option(request, object, "duration_seconds");
    set_optional_option(request, object, "seed");
    set_optional_option(request, object, "trim_tail");
    const std::string ref_wav = optional_string(object, "ref_wav");
    if (!ref_wav.empty()) {
        request.voice = engine::runtime::VoiceCondition{};
        request.voice->speaker = engine::runtime::VoiceReference{};
        request.voice->speaker->audio = read_audio_buffer(ref_wav);
    }
    return request;
}

std::vector<engine::runtime::TaskRequest> parse_requests(const std::string & request_sequence_json) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("Irodori-TTS warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> requests;
    for (const auto & item : root.as_array()) {
        requests.push_back(make_request(item));
    }
    if (requests.empty()) {
        throw std::runtime_error("Irodori-TTS warmbench request sequence is empty");
    }
    return requests;
}

engine::io::json::Value number(double value) {
    return engine::io::json::Value::make_number(value);
}

engine::io::json::Value string(std::string value) {
    return engine::io::json::Value::make_string(std::move(value));
}

engine::io::json::Value audio_summary_json(const engine::runtime::AudioBuffer & audio) {
    if (audio.samples.empty()) {
        throw std::runtime_error("Irodori-TTS warmbench received empty audio output");
    }
    double sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    float min_value = audio.samples.front();
    float max_value = audio.samples.front();
    for (const float sample : audio.samples) {
        sum += static_cast<double>(sample);
        abs_sum += std::abs(static_cast<double>(sample));
        sq_sum += static_cast<double>(sample) * static_cast<double>(sample);
        min_value = std::min(min_value, sample);
        max_value = std::max(max_value, sample);
    }
    const auto channels = std::max(1, audio.channels);
    const double frames = static_cast<double>(audio.samples.size() / static_cast<size_t>(channels));
    const double count = static_cast<double>(audio.samples.size());
    return engine::io::json::Value::make_object({
        {"sample_rate", number(static_cast<double>(audio.sample_rate))},
        {"channels", number(static_cast<double>(audio.channels))},
        {"samples", number(count)},
        {"frames", number(frames)},
        {"duration_sec", number(audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0)},
        {"sum", number(sum)},
        {"mean_abs", number(abs_sum / count)},
        {"rms", number(std::sqrt(sq_sum / count))},
        {"min", number(min_value)},
        {"max", number(max_value)},
    });
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    int request_index,
    double wall_ms,
    const std::filesystem::path & audio_path) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("Irodori-TTS warmbench expected audio output");
    }
    engine::io::json::Value::Object stem{
        {"name", string("audio")},
        {"summary", audio_summary_json(*result.audio_output)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", string(audio_path.string()));
    }
    return engine::io::json::Value::make_object({
        {"request_index", number(static_cast<double>(request_index))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", number(wall_ms)}})},
    });
}

void set_env_required(const char * key, const std::string & value) {
    if (setenv(key, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/Irodori-TTS-500M-v3");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 0);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        const std::string request_sequence_json = arg_value(argc, argv, "--request-sequence-json", "");
        const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path =
            arg_value(argc, argv, "--timing-file", "/tmp/irodori_tts_warm_bench_timing.log");
        set_env_required("MINITTS_TIMING_ENABLED", "1");
        set_env_required("MINITTS_TIMING_FILE", timing_path.string());
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "irodori_tts";
        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions options;
        options.backend.type = parse_backend(backend_name);
        options.backend.device = device;
        options.backend.threads = threads;
        for (const auto & [key, value] : parse_session_options(argc, argv)) {
            options.options.insert_or_assign(key, value);
        }

        auto session_base = model->create_task_session(
            {engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline},
            options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("loaded Irodori-TTS session is not offline-capable");
        }

        const auto requests = parse_requests(request_sequence_json);
        session->prepare(engine::runtime::build_preparation_request(requests.front()));
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(requests.front());
        }
        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }

        engine::io::json::Value::Array steps;
        steps.reserve(requests.size());
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            engine::runtime::TaskResult last_result;
            double total_ms = 0.0;
            for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                last_result = session->run(requests[request_index]);
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            if (!last_result.audio_output.has_value()) {
                throw std::runtime_error("Irodori-TTS warmbench expected audio output");
            }
            const double wall_ms = total_ms / static_cast<double>(std::max(1, iterations));
            std::filesystem::path audio_path;
            if (!output_dir.empty()) {
                audio_path = output_dir / ("audio_" + std::to_string(request_index) + ".wav");
                engine::audio::write_pcm16_wav(
                    audio_path,
                    last_result.audio_output->sample_rate,
                    last_result.audio_output->channels,
                    last_result.audio_output->samples);
            }
            std::cout << "irodori_tts.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(last_result, static_cast<int>(request_index), wall_ms, audio_path));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("irodori_tts")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "irodori_tts_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
