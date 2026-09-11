#include "../core/audio_task_warm_bench.h"

#include "engine/framework/debug/trace.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::string> split_csv_keep_empty(const std::string & value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        out.push_back(item);
    }
    return out;
}

std::vector<std::pair<std::string, std::string>> parse_key_value_options(
    int argc,
    char ** argv,
    const std::string & name) {
    std::vector<std::pair<std::string, std::string>> out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != name) {
            continue;
        }
        const std::string option = argv[i + 1];
        const size_t eq = option.find('=');
        if (eq == std::string::npos || eq == 0) {
            throw std::runtime_error("invalid Higgs Audio STT " + name + ": " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
}

std::string repeated_sequence_arg(const std::vector<std::string> & values, size_t index, const std::string & fallback) {
    return index < values.size() && !values[index].empty() ? values[index] : fallback;
}

engine::io::json::Value number(double value) {
    return engine::io::json::Value::make_number(value);
}

engine::io::json::Value string(std::string value) {
    return engine::io::json::Value::make_string(std::move(value));
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    int request_index,
    const std::filesystem::path & audio_path,
    double wall_ms,
    const std::string & enable_thinking,
    const std::string & max_tokens) {
    return engine::io::json::Value::make_object({
        {"request_index", number(request_index)},
        {"audio", string(audio_path.string())},
        {"language", string(result.text_output.has_value() ? result.text_output->language : "")},
        {"text_output", string(result.text_output.has_value() ? result.text_output->text : "")},
        {"word_timestamps", engine::io::json::Value::make_array({})},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", number(wall_ms)},
            {"enable_thinking", string(enable_thinking)},
            {"max_tokens", string(max_tokens)},
        })},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path =
            engine::tools::arg_value(argc, argv, "--model", "models/higgs-audio-v3-stt");
        const std::string audio_sequence_value = engine::tools::arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path = engine::tools::arg_value(argc, argv, "--audio", "resources/sample_16k.wav");
        const std::filesystem::path warmup_audio_path =
            engine::tools::arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cuda");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 0);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string prompt = engine::tools::arg_value(
            argc,
            argv,
            "--prompt",
            "Transcribe the speech. Output only the spoken words in lowercase with no punctuation.");
        const std::string max_tokens = engine::tools::arg_value(argc, argv, "--max-tokens", "1024");
        const std::string enable_thinking = engine::tools::arg_value(argc, argv, "--enable-thinking", "true");
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "/tmp/higgs_audio_stt_warm_bench_timing.log");

        setenv("MINITTS_TRACE_ENABLED", "0", 1);
        setenv("MINITTS_TIMING_ENABLED", "1", 1);
        setenv("MINITTS_TIMING_FILE", timing_path.c_str(), 1);
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "higgs_audio_stt";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = engine::tools::parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & [key, value] : parse_key_value_options(argc, argv, "--session-option")) {
            session_options.options.emplace(key, value);
        }
        const auto request_option_overrides = parse_key_value_options(argc, argv, "--request-option");

        auto session_base = model->create_task_session(
            engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline},
            session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("Higgs Audio STT did not create an offline task session");
        }

        const auto warmup_audio = engine::tools::read_audio_buffer(warmup_audio_path);
        session->prepare(engine::runtime::build_preparation_request(warmup_audio));
        engine::runtime::TaskRequest warmup_request;
        warmup_request.audio_input = warmup_audio;
        warmup_request.text_input = engine::runtime::Transcript{prompt, "en"};
        warmup_request.options["max_tokens"] = max_tokens;
        warmup_request.options["enable_thinking"] = enable_thinking;
        for (const auto & [key, value] : request_option_overrides) {
            warmup_request.options[key] = value;
        }
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(warmup_request);
        }

        std::vector<std::filesystem::path> request_paths;
        if (!audio_sequence_value.empty()) {
            for (const auto & item : engine::tools::split_csv(audio_sequence_value)) {
                request_paths.emplace_back(item);
            }
        } else {
            request_paths.push_back(audio_path);
        }
        const auto prompt_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--prompt-sequence", ""));
        const auto max_tokens_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--max-tokens-sequence", ""));
        const auto enable_thinking_sequence =
            split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--enable-thinking-sequence", ""));

        engine::io::json::Value::Array steps;
        steps.reserve(request_paths.size());
        for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
            const auto audio = engine::tools::read_audio_buffer(request_paths[request_index]);
            const std::string request_prompt = repeated_sequence_arg(prompt_sequence, request_index, prompt);
            const std::string request_max_tokens = repeated_sequence_arg(max_tokens_sequence, request_index, max_tokens);
            const std::string request_enable_thinking =
                repeated_sequence_arg(enable_thinking_sequence, request_index, enable_thinking);
            engine::runtime::TaskRequest run_request;
            run_request.audio_input = audio;
            run_request.text_input = engine::runtime::Transcript{request_prompt, "en"};
            run_request.options["max_tokens"] = request_max_tokens;
            run_request.options["enable_thinking"] = request_enable_thinking;
            for (const auto & [key, value] : request_option_overrides) {
                run_request.options[key] = value;
            }
            engine::runtime::TaskResult last_result;
            double total_ms = 0.0;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                last_result = session->run(run_request);
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            const double wall_ms = total_ms / static_cast<double>(iterations);
            std::cout << "average[" << request_index << "]\n";
            std::cout << "higgs_audio_stt.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(
                last_result,
                static_cast<int>(request_index),
                request_paths[request_index],
                wall_ms,
                request_enable_thinking,
                request_max_tokens));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("higgs_audio_stt")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "higgs_audio_stt_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
