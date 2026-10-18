#include "../core/audio_task_warm_bench.h"

#include <chrono>
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

std::string repeated_arg(int argc, char ** argv, const std::string & name, size_t index, const std::string & fallback) {
    size_t seen = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            if (seen == index) {
                return argv[i + 1];
            }
            ++seen;
        }
    }
    return fallback;
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
            throw std::runtime_error("invalid MMS forced aligner --session-option: " + option);
        }
        out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
    }
    return out;
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
    const std::string & requested_language,
    double wall_ms) {
    return engine::io::json::Value::make_object({
        {"request_index", number(request_index)},
        {"audio", string(audio_path.string())},
        {"requested_language", string(requested_language)},
        {"text_output", string(result.text_output.has_value() ? result.text_output->text : "")},
        {"language", string(result.text_output.has_value() ? result.text_output->language : "")},
        {"word_timestamps", engine::tools::word_timestamps_json(result.word_timestamps)},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", number(wall_ms)}})},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path =
            engine::tools::arg_value(argc, argv, "--model", "models/mms-300m-1130-forced-aligner");
        const std::string audio_sequence_value = engine::tools::arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path = engine::tools::arg_value(argc, argv, "--audio", "resources/sample.wav");
        const std::filesystem::path warmup_audio_path =
            engine::tools::arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cpu");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 1);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string default_language = engine::tools::arg_value(argc, argv, "--language", "eng");
        const std::string default_transcript = engine::tools::arg_value(argc, argv, "--transcript", "");
        const std::string warmup_language = engine::tools::arg_value(argc, argv, "--warmup-language", default_language);
        const std::string warmup_transcript = engine::tools::arg_value(argc, argv, "--warmup-transcript", default_transcript);
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "/tmp/mms_forced_aligner_warm_bench_timing.log");

        engine::tools::set_process_env("ENGINE_TRACE_ENABLED", "0");
        engine::tools::set_process_env("ENGINE_TIMING_ENABLED", "1");
        engine::tools::set_process_env("ENGINE_TIMING_FILE", timing_path.string());

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "mms_forced_aligner";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = engine::tools::parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & [key, value] : parse_session_options(argc, argv)) {
            session_options.options.emplace(key, value);
        }

        auto session_base = model->create_task_session(
            engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Alignment, engine::runtime::RunMode::Offline},
            session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("MMS forced aligner did not create an offline task session");
        }

        const auto warmup_audio = engine::tools::read_audio_buffer(warmup_audio_path);
        engine::runtime::TaskRequest warmup_prepare;
        warmup_prepare.audio_input = warmup_audio;
        warmup_prepare.text_input = engine::runtime::Transcript{warmup_transcript, warmup_language};
        session->prepare(engine::runtime::build_preparation_request(warmup_prepare));
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(warmup_prepare);
        }

        std::vector<std::filesystem::path> request_paths;
        if (!audio_sequence_value.empty()) {
            for (const auto & item : engine::tools::split_csv(audio_sequence_value)) {
                request_paths.emplace_back(item);
            }
        } else {
            request_paths.push_back(audio_path);
        }
        const auto request_languages = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--language-sequence", ""));
        const auto request_transcripts = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--transcript-sequence", ""));

        engine::io::json::Value::Array steps;
        steps.reserve(request_paths.size());
        for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
            const std::string language = request_index < request_languages.size() && !request_languages[request_index].empty()
                ? request_languages[request_index]
                : repeated_arg(argc, argv, "--request-language", request_index, default_language);
            const std::string transcript = request_index < request_transcripts.size()
                ? request_transcripts[request_index]
                : repeated_arg(argc, argv, "--request-transcript", request_index, default_transcript);
            const auto audio = engine::tools::read_audio_buffer(request_paths[request_index]);
            engine::runtime::TaskRequest run_request;
            run_request.audio_input = audio;
            run_request.text_input = engine::runtime::Transcript{transcript, language};
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
            std::cout << "mms_forced_aligner.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(
                last_result,
                static_cast<int>(request_index),
                request_paths[request_index],
                language,
                wall_ms));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("mms_forced_aligner")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "mms_forced_aligner_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
