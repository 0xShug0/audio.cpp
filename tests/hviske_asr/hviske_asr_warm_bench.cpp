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
            throw std::runtime_error("invalid Hviske ASR " + name + ": " + option);
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
    const std::string & requested_language,
    double wall_ms) {
    return engine::io::json::Value::make_object({
        {"request_index", number(request_index)},
        {"audio", string(audio_path.string())},
        {"requested_language", string(requested_language)},
        {"language", string(result.text_output.has_value() ? result.text_output->language : "")},
        {"text_output", string(result.text_output.has_value() ? result.text_output->text : "")},
        {"word_timestamps", engine::tools::word_timestamps_json(result.word_timestamps)},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", number(wall_ms)}})},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path =
            engine::tools::arg_value(argc, argv, "--model", "models/hviske-v5.3");
        const std::string audio_sequence_value = engine::tools::arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path = engine::tools::arg_value(argc, argv, "--audio", "resources/sample.wav");
        const std::filesystem::path warmup_audio_path =
            engine::tools::arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cpu");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 1);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string default_language = engine::tools::arg_value(argc, argv, "--language", "da");
        const std::string warmup_language = engine::tools::arg_value(argc, argv, "--warmup-language", default_language);
        const std::string punctuation = engine::tools::arg_value(argc, argv, "--punctuation", "true");
        const std::string max_tokens = engine::tools::arg_value(argc, argv, "--max-tokens", "256");
        const std::string num_beams = engine::tools::arg_value(argc, argv, "--num-beams", "1");
        const std::string length_penalty = engine::tools::arg_value(argc, argv, "--length-penalty", "1.0");
        const std::string do_sample = engine::tools::arg_value(argc, argv, "--do-sample", "false");
        const std::string temperature = engine::tools::arg_value(argc, argv, "--temperature", "1.0");
        const std::string top_k = engine::tools::arg_value(argc, argv, "--top-k", "50");
        const std::string top_p = engine::tools::arg_value(argc, argv, "--top-p", "1.0");
        const std::string seed = engine::tools::arg_value(argc, argv, "--seed", "0");
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "/tmp/hviske_asr_warm_bench_timing.log");

        setenv("MINITTS_TRACE_ENABLED", "0", 1);
        setenv("MINITTS_TIMING_ENABLED", "1", 1);
        setenv("MINITTS_TIMING_FILE", timing_path.c_str(), 1);
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "hviske_asr";
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
            throw std::runtime_error("Hviske ASR did not create an offline task session");
        }

        const auto warmup_audio = engine::tools::read_audio_buffer(warmup_audio_path);
        session->prepare(engine::runtime::build_preparation_request(warmup_audio));
        engine::runtime::TaskRequest warmup_request;
        warmup_request.audio_input = warmup_audio;
        warmup_request.text_input = engine::runtime::Transcript{"", warmup_language};
        warmup_request.options["punctuation"] = punctuation;
        warmup_request.options["max_tokens"] = max_tokens;
        warmup_request.options["num_beams"] = num_beams;
        warmup_request.options["length_penalty"] = length_penalty;
        warmup_request.options["do_sample"] = do_sample;
        warmup_request.options["temperature"] = temperature;
        warmup_request.options["top_k"] = top_k;
        warmup_request.options["top_p"] = top_p;
        warmup_request.options["seed"] = seed;
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
        const auto request_languages = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--language-sequence", ""));
        const auto max_tokens_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--max-tokens-sequence", ""));
        const auto num_beams_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--num-beams-sequence", ""));
        const auto length_penalty_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--length-penalty-sequence", ""));
        const auto do_sample_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--do-sample-sequence", ""));
        const auto temperature_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--temperature-sequence", ""));
        const auto top_k_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--top-k-sequence", ""));
        const auto top_p_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--top-p-sequence", ""));
        const auto seed_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--seed-sequence", ""));

        engine::io::json::Value::Array steps;
        steps.reserve(request_paths.size());
        for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
            const std::string language = request_index < request_languages.size() && !request_languages[request_index].empty()
                ? request_languages[request_index]
                : repeated_arg(argc, argv, "--request-language", request_index, default_language);
            const auto audio = engine::tools::read_audio_buffer(request_paths[request_index]);
            engine::runtime::TaskRequest run_request;
            run_request.audio_input = audio;
            run_request.text_input = engine::runtime::Transcript{"", language};
            run_request.options["punctuation"] = punctuation;
            run_request.options["max_tokens"] = repeated_sequence_arg(max_tokens_sequence, request_index, max_tokens);
            run_request.options["num_beams"] = repeated_sequence_arg(num_beams_sequence, request_index, num_beams);
            run_request.options["length_penalty"] = repeated_sequence_arg(length_penalty_sequence, request_index, length_penalty);
            run_request.options["do_sample"] = repeated_sequence_arg(do_sample_sequence, request_index, do_sample);
            run_request.options["temperature"] = repeated_sequence_arg(temperature_sequence, request_index, temperature);
            run_request.options["top_k"] = repeated_sequence_arg(top_k_sequence, request_index, top_k);
            run_request.options["top_p"] = repeated_sequence_arg(top_p_sequence, request_index, top_p);
            run_request.options["seed"] = repeated_sequence_arg(seed_sequence, request_index, seed);
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
            std::cout << "hviske_asr.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(
                last_result,
                static_cast<int>(request_index),
                request_paths[request_index],
                language,
                wall_ms));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("hviske_asr")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "hviske_asr_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
