#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct RequestCase {
    std::string name;
    std::string text;
    std::unordered_map<std::string, std::string> options;
};

std::string arg_value(
    int argc,
    char ** argv,
    const std::string & name,
    const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<std::string> arg_values(
    int argc,
    char ** argv,
    const std::string & name) {
    std::vector<std::string> out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            out.emplace_back(argv[i + 1]);
        }
    }
    return out;
}

int int_arg(
    int argc,
    char ** argv,
    const std::string & name,
    int fallback) {
    return std::stoi(
        arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(
    const std::string & value) {
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

double audio_seconds(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0.0;
    }
    return static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.sample_rate * audio.channels);
}

std::string scalar_option(
    const engine::io::json::Value & value) {
    if (value.is_string()) {
        return value.as_string();
    }
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(
            value.as_number());
    }
    throw std::runtime_error(
        "GLM-TTS warm-bench options must be scalar values");
}

std::vector<RequestCase> load_requests(
    const std::filesystem::path & path,
    const std::string & fallback_text) {
    if (path.empty()) {
        return {RequestCase{
            "request",
            fallback_text,
            {
                {"top_k", "25"},
                {"top_p", "0.8"},
                {"temperature", "1.0"},
                {"seed", "0"},
                {"max_tokens", "256"},
            }}};
    }
    const auto root = engine::io::json::parse_file(path);
    const auto & items = root.require("requests").as_array();
    if (items.empty()) {
        throw std::runtime_error(
            "GLM-TTS request file has no requests");
    }
    std::vector<RequestCase> out;
    out.reserve(items.size());
    for (size_t index = 0; index < items.size(); ++index) {
        const auto & item = items[index];
        RequestCase request;
        request.name = engine::io::json::optional_string(
            item, "name", "request_" + std::to_string(index));
        request.text =
            engine::io::json::require_string(item, "text");
        for (const char * name : {
                 "max_tokens",
                 "temperature",
                 "top_k",
                 "top_p",
                 "seed",
                 "num_inference_steps",
                 "flow_guidance_scale"}) {
            if (const auto * value = item.find(name);
                value != nullptr && !value->is_null()) {
                request.options[name] = scalar_option(*value);
            }
        }
        if (request.options.find("top_k") ==
            request.options.end()) {
            request.options["top_k"] = "25";
        }
        if (request.options.find("top_p") ==
            request.options.end()) {
            request.options["top_p"] = "0.8";
        }
        if (request.options.find("temperature") ==
            request.options.end()) {
            request.options["temperature"] = "1.0";
        }
        if (request.options.find("seed") ==
            request.options.end()) {
            request.options["seed"] = "0";
        }
        if (request.options.find("max_tokens") ==
            request.options.end()) {
            request.options["max_tokens"] = "512";
        }
        out.push_back(std::move(request));
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model_path =
        arg_value(argc, argv, "--model", "");
    const std::filesystem::path voice_path =
        arg_value(argc, argv, "--voice-ref", "");
    const std::filesystem::path output_dir =
        arg_value(
            argc,
            argv,
            "--audio-out-dir",
            "build/logs/warmbench/glm_tts_audio");
    const std::filesystem::path log_file =
        arg_value(
            argc,
            argv,
            "--log-file",
            "build/logs/warmbench/glm_tts.log");
    const std::string reference_text =
        arg_value(argc, argv, "--reference-text", "");
    const std::string text =
        arg_value(
            argc,
            argv,
            "--text",
            "Hello from G L M T T S.");
    const std::filesystem::path request_file =
        arg_value(argc, argv, "--request-file", "");
    const int iterations =
        int_arg(argc, argv, "--iterations", 2);
    const int hold_seconds =
        int_arg(argc, argv, "--hold-seconds", 0);
    if (model_path.empty() || voice_path.empty() ||
        reference_text.empty()) {
        throw std::runtime_error(
            "--model, --voice-ref, and --reference-text are required");
    }
    if (iterations <= 0) {
        throw std::runtime_error("--iterations must be positive");
    }
    if (hold_seconds < 0) {
        throw std::runtime_error(
            "--hold-seconds must be non-negative");
    }
    const auto requests = load_requests(request_file, text);

    std::filesystem::create_directories(output_dir);
    if (!log_file.parent_path().empty()) {
        std::filesystem::create_directories(log_file.parent_path());
    }
    engine::debug::configure_logging(
        engine::debug::LoggingConfig{true, log_file.string()});

    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_path;
    load_request.family_hint = "glm_tts";
    auto model = registry.load(load_request);

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = parse_backend(
        arg_value(argc, argv, "--backend", "cuda"));
    session_options.backend.device =
        int_arg(argc, argv, "--device", 0);
    session_options.backend.threads =
        int_arg(argc, argv, "--threads", 8);
    session_options.options["glm_tts.mem_saver"] =
        arg_value(argc, argv, "--mem-saver", "false");
    session_options.options["glm_tts.reference_cache_slots"] =
        arg_value(argc, argv, "--reference-cache-slots", "1");
    for (const auto & option :
         arg_values(argc, argv, "--session-option")) {
        const size_t equals = option.find('=');
        if (equals == std::string::npos || equals == 0) {
            throw std::runtime_error(
                "invalid --session-option: " + option);
        }
        session_options.options[option.substr(0, equals)] =
            option.substr(equals + 1);
    }

    auto session_base = model->create_task_session(
        {engine::runtime::VoiceTaskKind::VoiceCloning,
         engine::runtime::RunMode::Offline},
        session_options);
    auto * session =
        dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
            session_base.get());
    if (session == nullptr) {
        throw std::runtime_error(
            "GLM-TTS did not create an offline session");
    }
    session->prepare(
        engine::runtime::SessionPreparationRequest{});

    const auto wav = engine::audio::read_wav_f32(voice_path);
    engine::runtime::AudioBuffer reference_audio{
        wav.sample_rate, wav.channels, wav.samples};
    for (const auto & request_case : requests) {
        for (int iteration = 0;
             iteration < iterations;
             ++iteration) {
            engine::runtime::TaskRequest request;
            request.text_input = engine::runtime::Transcript{
                request_case.text, "en"};
            request.voice = engine::runtime::VoiceCondition{};
            request.voice->speaker =
                engine::runtime::VoiceReference{};
            request.voice->speaker->audio = reference_audio;
            request.options = request_case.options;
            request.options["reference_text"] = reference_text;

            const auto started =
                std::chrono::steady_clock::now();
            auto result = session->run(request);
            const double wall_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started)
                    .count();
            if (!result.audio_output.has_value()) {
                throw std::runtime_error(
                    "GLM-TTS produced no audio");
            }
            const auto output_path =
                output_dir /
                (request_case.name + "_" +
                 std::to_string(iteration + 1) + ".wav");
            engine::audio::write_pcm16_wav(
                output_path,
                result.audio_output->sample_rate,
                result.audio_output->channels,
                result.audio_output->samples);
            const double seconds =
                audio_seconds(*result.audio_output);
            std::cout
                << "request=" << request_case.name << "\n"
                << "iteration=" << iteration + 1 << "\n"
                << "wall_ms=" << wall_ms << "\n"
                << "audio_seconds=" << seconds << "\n"
                << "rtf="
                << (seconds > 0.0
                        ? wall_ms / 1000.0 / seconds
                        : 0.0)
                << "\n"
                << "audio_out=" << output_path.string()
                << "\n";
        }
    }
    std::cout << "log_out=" << log_file.string() << "\n";
    if (hold_seconds > 0) {
        std::cout
            << "holding_session_seconds="
            << hold_seconds << "\n";
        std::cout.flush();
        std::this_thread::sleep_for(
            std::chrono::seconds(hold_seconds));
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "glm_tts_warm_bench failed: "
        << error.what() << "\n";
    return 1;
}
