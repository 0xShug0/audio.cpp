#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"

#include "../core/audio_task_warm_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

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

engine::runtime::AudioBuffer read_reference_audio(const engine::io::json::Value & object) {
    const auto path = engine::io::json::require_string(object, "reference_audio");
    if (path.empty()) {
        throw std::runtime_error("Confucius4-TTS warmbench reference_audio is empty");
    }
    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    return {wav.sample_rate, wav.channels, wav.samples};
}

engine::runtime::TaskRequest make_request(const engine::io::json::Value & object) {
    const auto text = engine::io::json::require_string(object, "text");
    const auto language = engine::io::json::require_string(object, "language");
    if (text.empty() || language.empty()) {
        throw std::runtime_error("Confucius4-TTS warmbench request requires text and language");
    }
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{text, language};
    request.voice = engine::runtime::VoiceCondition{};
    request.voice->speaker = engine::runtime::VoiceReference{};
    request.voice->speaker->audio = read_reference_audio(object);
    if (const auto * cached_voice_id = object.find("cached_voice_id");
        cached_voice_id != nullptr && !cached_voice_id->is_null()) {
        request.voice->speaker->cached_voice_id = cached_voice_id->as_string();
    }
    set_optional_option(request, object, "seed", "seed");
    set_optional_option(request, object, "temperature", "temperature");
    set_optional_option(request, object, "top_p", "top_p");
    set_optional_option(request, object, "top_k", "top_k");
    set_optional_option(request, object, "num_beams", "num_beams");
    set_optional_option(request, object, "repetition_penalty", "repetition_penalty");
    set_optional_option(request, object, "max_tokens", "max_tokens");
    set_optional_option(request, object, "num_inference_steps", "num_inference_steps");
    set_optional_option(request, object, "guidance_scale", "guidance_scale");
    set_optional_option(request, object, "text_chunk_size", "text_chunk_size");
    set_optional_option(request, object, "text_chunk_mode", "text_chunk_mode");
    set_optional_option(request, object, "cross_fade_duration", "cross_fade_duration");
    return request;
}

std::vector<engine::runtime::TaskRequest> parse_requests(const std::string & request_sequence_json) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("Confucius4-TTS warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> requests;
    for (const auto & item : root.as_array()) {
        requests.push_back(make_request(item));
    }
    if (requests.empty()) {
        throw std::runtime_error("Confucius4-TTS warmbench request sequence is empty");
    }
    return requests;
}

engine::io::json::Value audio_summary_json(const engine::runtime::AudioBuffer & audio) {
    if (audio.samples.empty()) {
        throw std::runtime_error("Confucius4-TTS warmbench received empty audio output");
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
    int request_index,
    int text_length,
    double wall_ms,
    const std::filesystem::path & audio_path) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("Confucius4-TTS warmbench expected audio output");
    }
    engine::io::json::Value::Object stem{
        {"name", engine::tools::string("audio")},
        {"summary", audio_summary_json(*result.audio_output)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", engine::tools::string(audio_path.string()));
    }
    return engine::io::json::Value::make_object({
        {"request_index", engine::tools::number(static_cast<double>(request_index))},
        {"text_length", engine::tools::number(static_cast<double>(text_length))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", engine::tools::number(wall_ms)}})},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = engine::tools::arg_value(argc, argv, "--model", "models/Confucius4-TTS/audio_cpp");
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cuda");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup_count = engine::tools::int_arg(argc, argv, "--warmup", 0);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string run_mode = engine::tools::arg_value(argc, argv, "--run-mode", "offline");
        if (iterations != 1) {
            throw std::runtime_error("Confucius4-TTS warmbench records raw per-request timing; --iterations must be 1");
        }
        const auto requests = parse_requests(engine::tools::arg_value(argc, argv, "--request-sequence-json", ""));
        const std::filesystem::path output_dir = engine::tools::arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path = engine::tools::arg_value(argc, argv, "--timing-file", "");
        if (timing_path.empty()) {
            throw std::runtime_error("Confucius4-TTS warmbench requires --timing-file from warmbench.py");
        }
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "confucius4_tts";
        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions options;
        options.backend.type = engine::tools::parse_backend(backend_name);
        options.backend.device = device;
        options.backend.threads = threads;
        for (const auto & [key, value] : engine::tools::session_option_args(argc, argv)) {
            options.options[key] = value;
        }
        const bool streaming = run_mode == "streaming";
        if (!streaming && run_mode != "offline") {
            throw std::runtime_error("unsupported Confucius4-TTS warmbench run mode: " + run_mode);
        }
        auto session_base = model->create_task_session(
            {engine::runtime::VoiceTaskKind::VoiceCloning,
             streaming ? engine::runtime::RunMode::Streaming : engine::runtime::RunMode::Offline},
            options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        auto * streaming_session = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session_base.get());
        if (streaming) {
            if (streaming_session == nullptr) {
                throw std::runtime_error("loaded Confucius4-TTS session is not streaming-capable");
            }
        } else if (session == nullptr) {
            throw std::runtime_error("loaded Confucius4-TTS session is not offline-capable");
        }

        session_base->prepare(engine::runtime::build_preparation_request(requests.front()));
        for (int i = 0; i < warmup_count; ++i) {
            if (streaming) {
                streaming_session->start_stream(requests.front());
                while (streaming_session->next_stream_event().has_value()) {
                }
                (void)streaming_session->finish_stream();
            } else {
                (void)session->run(requests.front());
            }
        }
        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }

        engine::io::json::Value::Array steps;
        steps.reserve(requests.size());
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            const auto & request = requests[request_index];
            const int text_length = static_cast<int>(request.text_input->text.size());
            const auto started = std::chrono::steady_clock::now();
            engine::runtime::TaskResult result;
            if (streaming) {
                streaming_session->start_stream(request);
                while (streaming_session->next_stream_event().has_value()) {
                }
                result = streaming_session->finish_stream();
            } else {
                result = session->run(request);
            }
            const auto ended = std::chrono::steady_clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(ended - started).count();
            if (!result.audio_output.has_value()) {
                throw std::runtime_error("Confucius4-TTS warmbench expected audio output");
            }
            std::filesystem::path audio_path;
            if (!output_dir.empty()) {
                audio_path = output_dir / ("audio_" + std::to_string(request_index) + ".wav");
                engine::audio::write_pcm16_wav(
                    audio_path,
                    result.audio_output->sample_rate,
                    result.audio_output->channels,
                    result.audio_output->samples);
            }
            std::cout << "confucius4_tts.request[" << request_index << "].length=" << text_length << "\n";
            std::cout << "confucius4_tts.request[" << request_index << "].wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(result, static_cast<int>(request_index), text_length, wall_ms, audio_path));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", engine::tools::string("confucius4_tts")},
            {"backend", engine::tools::string(backend_name)},
            {"mode", engine::tools::string(run_mode)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "confucius4_tts_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
