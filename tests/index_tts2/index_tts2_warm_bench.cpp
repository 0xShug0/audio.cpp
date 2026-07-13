#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    throw std::runtime_error("IndexTTS2 warmbench backend must be cuda or cpu");
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
            throw std::runtime_error("invalid IndexTTS2 --session-option: " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
}

std::string option_text(const engine::io::json::Value & value) {
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(value.as_number());
    }
    if (value.is_array()) {
        std::string out;
        const auto & items = value.as_array();
        for (size_t i = 0; i < items.size(); ++i) {
            if (i != 0) {
                out += ",";
            }
            if (!items[i].is_number()) {
                throw std::runtime_error("IndexTTS2 warmbench option arrays must contain only numbers");
            }
            out += engine::io::json::stringify_number(items[i].as_number());
        }
        return out;
    }
    return value.as_string();
}

std::string required_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        throw std::runtime_error("IndexTTS2 warmbench request missing " + key);
    }
    return value->as_string();
}

std::string optional_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::string{} : value->as_string();
}

void set_optional_option(engine::runtime::TaskRequest & request, const engine::io::json::Value & object, const std::string & key) {
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
    const std::string language = optional_string(object, "language");
    request.text_input = engine::runtime::Transcript{required_string(object, "text"), language.empty() ? "en" : language};
    engine::runtime::VoiceCondition voice;
    engine::runtime::VoiceReference speaker;
    speaker.audio = read_audio_buffer(required_string(object, "voice_ref"));
    voice.speaker = std::move(speaker);
    request.voice = std::move(voice);

    const auto emotion_audio = optional_string(object, "audio");
    if (!emotion_audio.empty()) {
        request.audio_input = read_audio_buffer(emotion_audio);
    }

    set_optional_option(request, object, "emotion_alpha");
    set_optional_option(request, object, "emotion_vector");
    set_optional_option(request, object, "use_emotion_text");
    set_optional_option(request, object, "emotion_text");
    set_optional_option(request, object, "use_random_emotion");
    set_optional_option(request, object, "interval_silence_ms");
    set_optional_option(request, object, "text_chunk_size");
    set_optional_option(request, object, "do_sample");
    set_optional_option(request, object, "top_p");
    set_optional_option(request, object, "top_k");
    set_optional_option(request, object, "temperature");
    set_optional_option(request, object, "length_penalty");
    set_optional_option(request, object, "num_beams");
    set_optional_option(request, object, "repetition_penalty");
    set_optional_option(request, object, "max_tokens");
    set_optional_option(request, object, "seed");
    return request;
}

std::vector<engine::runtime::TaskRequest> parse_requests(const std::string & request_sequence_json) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("IndexTTS2 warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> requests;
    for (const auto & item : root.as_array()) {
        requests.push_back(make_request(item));
    }
    if (requests.empty()) {
        throw std::runtime_error("IndexTTS2 warmbench request sequence is empty");
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
        throw std::runtime_error("IndexTTS2 warmbench received empty audio output");
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
    const int channels = std::max(1, audio.channels);
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

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/IndexTTS-2");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 0);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        const std::string request_sequence_json = arg_value(argc, argv, "--request-sequence-json", "");
        const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path = arg_value(argc, argv, "--timing-file", "/tmp/index_tts2_warm_bench_timing.log");
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "index_tts2";
        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(load_request);

        engine::runtime::TaskSpec task;
        task.task = engine::runtime::VoiceTaskKind::Tts;
        task.mode = engine::runtime::RunMode::Offline;
        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & [key, value] : parse_session_options(argc, argv)) {
            session_options.options[key] = value;
        }
        auto requests = parse_requests(request_sequence_json);
        auto session_base = model->create_task_session(task, session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("IndexTTS2 model did not create an offline voice task session");
        }
        engine::runtime::SessionPreparationRequest preparation;
        preparation.text = requests.front().text_input;
        preparation.voice = requests.front().voice;
        preparation.audio = requests.front().audio_input.has_value()
            ? std::optional<engine::runtime::AudioPreparationContract>({
                  requests.front().audio_input->sample_rate,
                  requests.front().audio_input->channels,
                  static_cast<int64_t>(requests.front().audio_input->samples.size()),
              })
            : std::nullopt;
        preparation.options = requests.front().options;
        session->prepare(preparation);

        std::vector<engine::io::json::Value> steps;
        std::vector<std::string> timing_lines;
        timing_lines.push_back("index_tts2.backend " + backend_name);
        timing_lines.push_back("index_tts2.model_root " + model_path.string());
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(requests.front());
        }

        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            double total_ms = 0.0;
            engine::runtime::TaskResult last_result;
            for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
                const auto start = Clock::now();
                last_result = session->run(requests[request_index]);
                const auto end = Clock::now();
                const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
                total_ms += wall_ms;
                timing_lines.push_back("index_tts2.wall_ms " + engine::io::json::stringify_number(wall_ms));
            }
            if (!last_result.audio_output.has_value()) {
                throw std::runtime_error("IndexTTS2 warmbench expected audio output");
            }
            const double avg_ms = total_ms / static_cast<double>(std::max(1, iterations));
            std::filesystem::path audio_path;
            if (!output_dir.empty()) {
                std::filesystem::create_directories(output_dir);
                audio_path = output_dir / ("request_" + std::to_string(request_index) + ".wav");
                engine::audio::write_pcm16_wav(
                    audio_path,
                    last_result.audio_output->sample_rate,
                    last_result.audio_output->channels,
                    last_result.audio_output->samples);
            }
            engine::io::json::Value::Object step{
                {"request_index", number(static_cast<double>(request_index))},
                {"stems", engine::io::json::Value::make_array({
                    engine::io::json::Value::make_object({
                        {"name", string("audio")},
                        {"summary", audio_summary_json(*last_result.audio_output)},
                        {"audio", string(audio_path.string())},
                    }),
                })},
                {"metrics", engine::io::json::Value::make_object({{"wall_ms", number(avg_ms)}})},
            };
            steps.push_back(engine::io::json::Value::make_object(std::move(step)));
            std::cout << "index_tts2.wall_ms=" << avg_ms << "\n";
        }

        if (!timing_path.empty()) {
            std::filesystem::create_directories(timing_path.parent_path());
            std::ofstream timing(timing_path, std::ios::app);
            for (const auto & line : timing_lines) {
                timing << line << "\n";
            }
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("index_tts2")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "index_tts2_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
