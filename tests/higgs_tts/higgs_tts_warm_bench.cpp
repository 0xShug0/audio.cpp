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
#include <fstream>
#include <iostream>
#include <optional>
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

bool has_flag(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
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
            throw std::runtime_error("invalid Higgs TTS --session-option: " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    throw std::runtime_error("Higgs TTS warmbench is CUDA-only");
}

std::filesystem::path resolve_path(const std::string & path_text) {
    const std::filesystem::path path(path_text);
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::current_path() / path;
}

std::string optional_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::string{} : value->as_string();
}

std::string required_string(const engine::io::json::Value & object, const std::string & key) {
    const auto value = optional_string(object, key);
    if (value.empty()) {
        throw std::runtime_error("Higgs TTS warmbench request missing field: " + key);
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
    const std::string & source,
    const std::string & target) {
    const auto * value = object.find(source);
    if (value != nullptr && !value->is_null()) {
        request.options[target] = option_text(*value);
    }
}

std::optional<engine::runtime::AudioBuffer> read_reference_audio(
    const engine::io::json::Value & object) {
    auto reference_path = optional_string(object, "reference_audio");
    if (reference_path.empty()) {
        reference_path = optional_string(object, "voice_ref");
    }
    if (reference_path.empty()) {
        return std::nullopt;
    }
    const auto wav = engine::audio::read_wav_f32(resolve_path(reference_path));
    return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

engine::runtime::TaskRequest make_request(const engine::io::json::Value & object) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{required_string(object, "text"), ""};
    if (auto reference_audio = read_reference_audio(object); reference_audio.has_value()) {
        request.voice = engine::runtime::VoiceCondition{};
        request.voice->speaker = engine::runtime::VoiceReference{};
        request.voice->speaker->audio = std::move(*reference_audio);
        request.options["reference_text"] = required_string(object, "reference_text");
    }
    set_optional_option(request, object, "max_tokens", "max_tokens");
    set_optional_option(request, object, "temperature", "temperature");
    set_optional_option(request, object, "top_p", "top_p");
    set_optional_option(request, object, "top_k", "top_k");
    set_optional_option(request, object, "repetition_penalty", "repetition_penalty");
    set_optional_option(request, object, "seed", "seed");
    return request;
}

std::vector<engine::runtime::TaskRequest> parse_requests(const std::string & request_sequence_json) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("Higgs TTS warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> requests;
    for (const auto & item : root.as_array()) {
        requests.push_back(make_request(item));
    }
    if (requests.empty()) {
        throw std::runtime_error("Higgs TTS warmbench request sequence is empty");
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
        throw std::runtime_error("Higgs TTS warmbench received empty audio output");
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
        throw std::runtime_error("Higgs TTS warmbench expected audio output");
    }
    engine::io::json::Value::Object stem{
        {"name", string("audio")},
        {"summary", audio_summary_json(*result.audio_output)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", string(audio_path.string()));
    }
    const auto & audio = *result.audio_output;
    const double frames = static_cast<double>(
        audio.samples.size() / static_cast<size_t>(std::max(1, audio.channels)));
    const double duration_sec = audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0;
    const double rtf = duration_sec > 0.0 ? wall_ms / 1000.0 / duration_sec : 0.0;
    return engine::io::json::Value::make_object({
        {"request_index", number(static_cast<double>(request_index))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", number(wall_ms)},
            {"rtf", number(rtf)},
        })},
    });
}

void write_timing(const std::filesystem::path & path, const std::vector<std::string> & lines) {
    if (path.empty()) {
        return;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open timing file: " + path.string());
    }
    for (const auto & line : lines) {
        out << line << "\n";
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/higgs-audio-v3-tts-4b");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 0);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        std::string request_sequence_json = arg_value(argc, argv, "--request-sequence-json", "");
        const std::filesystem::path request_sequence_file =
            arg_value(argc, argv, "--request-sequence-file", "");
        if (request_sequence_json.empty() && !request_sequence_file.empty()) {
            std::ifstream input(request_sequence_file, std::ios::binary);
            if (!input) {
                throw std::runtime_error(
                    "failed to open Higgs TTS request sequence: " + request_sequence_file.string());
            }
            request_sequence_json.assign(
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>());
        }
        const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path =
            arg_value(argc, argv, "--timing-file", "/tmp/higgs_tts_warm_bench_timing.log");
        const std::filesystem::path log_path = arg_value(argc, argv, "--log-file", "");
        if (has_flag(argc, argv, "--enable-trace")) {
            if (log_path.empty()) {
                throw std::runtime_error("Higgs TTS --enable-trace requires --log-file");
            }
            engine::debug::configure_logging(engine::debug::LoggingConfig{true, log_path.string()});
        }

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "higgs_tts";
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
            throw std::runtime_error("loaded Higgs TTS session is not offline-capable");
        }

        const auto requests = parse_requests(request_sequence_json);
        session->prepare(engine::runtime::build_preparation_request(requests.front()));
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(requests.front());
        }
        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }

        std::vector<std::string> timing_lines{"higgs_tts.cpp.model_load_excluded=1"};
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
                throw std::runtime_error("Higgs TTS warmbench expected audio output");
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
            timing_lines.push_back(
                "higgs_tts.cpp.request_" + std::to_string(request_index) + ".wall_ms=" + std::to_string(wall_ms));
            const auto & audio = *last_result.audio_output;
            const double frames = static_cast<double>(
                audio.samples.size() / static_cast<size_t>(std::max(1, audio.channels)));
            const double duration_sec = audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0;
            const double rtf = duration_sec > 0.0 ? wall_ms / 1000.0 / duration_sec : 0.0;
            timing_lines.push_back(
                "higgs_tts.cpp.request_" + std::to_string(request_index) + ".rtf=" + std::to_string(rtf));
            std::cout << "higgs_tts.cpp.request=" << request_index
                      << " wall_ms=" << wall_ms
                      << " rtf=" << rtf << "\n";
            steps.push_back(step_json(last_result, static_cast<int>(request_index), wall_ms, audio_path));
        }

        write_timing(timing_path, timing_lines);
        const auto summary = engine::io::json::Value::make_object({
            {"family", string("higgs_tts")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "higgs_tts_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
