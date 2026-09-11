#include "engine/framework/audio/wav_writer.h"

#include "../core/audio_task_warm_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct NeuTTSBenchRequest {
    std::string id;
    engine::runtime::TaskRequest request;
};

std::string option_text(const engine::io::json::Value & value) {
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(value.as_number());
    }
    return value.as_string();
}

std::string optional_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::string{} : value->as_string();
}

std::string required_string(const engine::io::json::Value & object, const std::string & key) {
    const auto value = optional_string(object, key);
    if (value.empty()) {
        throw std::runtime_error("NeuTTS warmbench request missing field: " + key);
    }
    return value;
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

NeuTTSBenchRequest make_request(const engine::io::json::Value & object, const std::string & fallback_id) {
    NeuTTSBenchRequest out;
    out.id = optional_string(object, "id");
    if (out.id.empty()) {
        out.id = fallback_id;
    }
    out.request.text_input = engine::runtime::Transcript{required_string(object, "text"), "en"};
    set_optional_option(out.request, object, "speaker", "speaker");
    set_optional_option(out.request, object, "emotion", "emotion");
    set_optional_option(out.request, object, "seed", "seed");
    set_optional_option(out.request, object, "temperature", "temperature");
    set_optional_option(out.request, object, "top_k", "top_k");
    set_optional_option(out.request, object, "max_tokens", "max_tokens");
    set_optional_option(out.request, object, "min_tokens", "min_tokens");
    set_optional_option(out.request, object, "text_chunk_mode", "text_chunk_mode");
    set_optional_option(out.request, object, "text_chunk_size", "text_chunk_size");
    return out;
}

std::vector<NeuTTSBenchRequest> parse_requests(const std::string & request_sequence_json) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("NeuTTS warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<NeuTTSBenchRequest> requests;
    int index = 0;
    for (const auto & item : root.as_array()) {
        requests.push_back(make_request(item, "request_" + std::to_string(index++)));
    }
    if (requests.empty()) {
        throw std::runtime_error("NeuTTS warmbench request sequence is empty");
    }
    return requests;
}

std::optional<NeuTTSBenchRequest> parse_warmup_request(const std::string & request_json) {
    if (request_json.empty()) {
        return std::nullopt;
    }
    return make_request(engine::io::json::parse(request_json), "warmup");
}

engine::io::json::Value audio_summary_json(const engine::runtime::AudioBuffer & audio) {
    if (audio.samples.empty()) {
        throw std::runtime_error("NeuTTS warmbench received empty audio output");
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
        {"sample_rate", engine::tools::number(static_cast<double>(audio.sample_rate))},
        {"channels", engine::tools::number(static_cast<double>(audio.channels))},
        {"samples", engine::tools::number(count)},
        {"frames", engine::tools::number(frames)},
        {"duration_sec", engine::tools::number(audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0)},
        {"sum", engine::tools::number(sum)},
        {"mean_abs", engine::tools::number(abs_sum / count)},
        {"rms", engine::tools::number(std::sqrt(sq_sum / count))},
        {"min", engine::tools::number(min_value)},
        {"max", engine::tools::number(max_value)},
    });
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    const NeuTTSBenchRequest & request,
    int request_index,
    double wall_ms,
    const std::filesystem::path & audio_path) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("NeuTTS warmbench expected audio output");
    }
    engine::io::json::Value::Object stem{
        {"name", engine::tools::string("audio")},
        {"summary", audio_summary_json(*result.audio_output)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", engine::tools::string(audio_path.string()));
    }
    const auto & audio = *result.audio_output;
    const double frames = static_cast<double>(
        audio.samples.size() / static_cast<size_t>(std::max(1, audio.channels)));
    const double duration_sec = audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0;
    const double rtf = duration_sec > 0.0 ? wall_ms / 1000.0 / duration_sec : 0.0;
    return engine::io::json::Value::make_object({
        {"request_index", engine::tools::number(static_cast<double>(request_index))},
        {"id", engine::tools::string(request.id)},
        {"text_length", engine::tools::number(static_cast<double>(request.request.text_input->text.size()))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", engine::tools::number(wall_ms)},
            {"rtf", engine::tools::number(rtf)},
        })},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = engine::tools::arg_value(argc, argv, "--model", "models/NeuTTS-2E");
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cuda");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 0);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::filesystem::path output_dir = engine::tools::arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "/tmp/neutts_warm_bench_timing.log");
        const auto warmup_request =
            parse_warmup_request(engine::tools::arg_value(argc, argv, "--warmup-request-json", ""));
        if (backend_name != "cuda") {
            throw std::runtime_error("NeuTTS warmbench is CUDA-only");
        }

        engine::tools::set_process_env("ENGINE_TRACE_ENABLED", "0");
        engine::tools::set_process_env("ENGINE_TIMING_ENABLED", "1");
        engine::tools::set_process_env("ENGINE_TIMING_FILE", timing_path.string());
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "neutts";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions options;
        options.backend.type = engine::tools::parse_backend(backend_name);
        options.backend.device = device;
        options.backend.threads = threads;
        for (const auto & [key, value] : engine::tools::session_option_args(argc, argv)) {
            options.options.insert_or_assign(key, value);
        }

        auto session_base = model->create_task_session(
            {engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline},
            options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("loaded NeuTTS session is not offline-capable");
        }

        const auto requests = parse_requests(engine::tools::arg_value(argc, argv, "--request-sequence-json", ""));
        const auto & prepare_request = warmup_request.has_value() ? warmup_request->request : requests.front().request;
        session->prepare(engine::runtime::build_preparation_request(prepare_request));
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(prepare_request);
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
                last_result = session->run(requests[request_index].request);
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            if (!last_result.audio_output.has_value()) {
                throw std::runtime_error("NeuTTS warmbench expected audio output");
            }
            const double wall_ms = total_ms / static_cast<double>(std::max(1, iterations));
            std::filesystem::path audio_path;
            if (!output_dir.empty()) {
                audio_path = output_dir / (requests[request_index].id + ".wav");
                const auto & audio = *last_result.audio_output;
                engine::audio::write_pcm16_wav(audio_path, audio.sample_rate, audio.channels, audio.samples);
            }
            std::cout << "neutts.request[" << request_index << "].id=" << requests[request_index].id << "\n";
            std::cout << "neutts.request[" << request_index << "].wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(
                last_result,
                requests[request_index],
                static_cast<int>(request_index),
                wall_ms,
                audio_path));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", engine::tools::string("neutts")},
            {"backend", engine::tools::string(backend_name)},
            {"mode", engine::tools::string("offline")},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "neutts_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
