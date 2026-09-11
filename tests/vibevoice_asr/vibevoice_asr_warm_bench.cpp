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
            throw std::runtime_error("invalid VibeVoice-ASR " + name + ": " + option);
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
    const std::string & context,
    const std::string & max_tokens,
    const std::string & temperature,
    const std::string & top_p,
    const std::string & top_k,
    const std::string & num_beams,
    const std::string & seed) {
    return engine::io::json::Value::make_object({
        {"request_index", number(request_index)},
        {"audio", string(audio_path.string())},
        {"language", string(result.text_output.has_value() ? result.text_output->language : "")},
        {"text_output", string(result.text_output.has_value() ? result.text_output->text : "")},
        {"word_timestamps", engine::io::json::Value::make_array({})},
        {"speech_segments", engine::tools::speech_segments_json(result.speech_segments)},
        {"speaker_turns", engine::tools::speaker_turns_json(result.speaker_turns)},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", number(wall_ms)},
            {"context", string(context.empty() ? "none" : context)},
            {"max_tokens", string(max_tokens)},
            {"temperature", string(temperature)},
            {"top_p", string(top_p)},
            {"top_k", string(top_k)},
            {"num_beams", string(num_beams)},
            {"seed", string(seed)},
        })},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path =
            engine::tools::arg_value(argc, argv, "--model", "models/VibeVoice-ASR");
        const std::string audio_sequence_value = engine::tools::arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path = engine::tools::arg_value(argc, argv, "--audio", "resources/sample_16k.wav");
        const std::filesystem::path warmup_audio_path =
            engine::tools::arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cuda");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 0);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string context = engine::tools::arg_value(argc, argv, "--context", "");
        const std::string max_tokens = engine::tools::arg_value(argc, argv, "--max-tokens", "32768");
        const std::string temperature = engine::tools::arg_value(argc, argv, "--temperature", "0.0");
        const std::string top_p = engine::tools::arg_value(argc, argv, "--top-p", "1.0");
        const std::string top_k = engine::tools::arg_value(argc, argv, "--top-k", "50");
        const std::string num_beams = engine::tools::arg_value(argc, argv, "--num-beams", "1");
        const std::string seed = engine::tools::arg_value(argc, argv, "--seed", "1234");
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "/tmp/vibevoice_asr_warm_bench_timing.log");

        setenv("MINITTS_TRACE_ENABLED", "0", 1);
        setenv("MINITTS_TIMING_ENABLED", "1", 1);
        setenv("MINITTS_TIMING_FILE", timing_path.c_str(), 1);
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "vibevoice_asr";
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
            throw std::runtime_error("VibeVoice-ASR did not create an offline task session");
        }

        std::vector<std::filesystem::path> request_paths;
        if (!audio_sequence_value.empty()) {
            for (const auto & item : engine::tools::split_csv(audio_sequence_value)) {
                request_paths.emplace_back(item);
            }
        } else {
            request_paths.push_back(audio_path);
        }
        if (request_paths.empty()) {
            throw std::runtime_error("VibeVoice-ASR request sequence is empty");
        }

        std::vector<engine::runtime::AudioBuffer> request_audio_buffers;
        request_audio_buffers.reserve(request_paths.size());
        for (const auto & path : request_paths) {
            request_audio_buffers.push_back(engine::tools::read_audio_buffer(path));
        }

        const auto warmup_audio = engine::tools::read_audio_buffer(warmup_audio_path);
        int64_t prepare_samples = static_cast<int64_t>(warmup_audio.samples.size());
        for (const auto & audio : request_audio_buffers) {
            if (audio.sample_rate == warmup_audio.sample_rate &&
                audio.channels == warmup_audio.channels &&
                static_cast<int64_t>(audio.samples.size()) > prepare_samples) {
                prepare_samples = static_cast<int64_t>(audio.samples.size());
            }
        }
        engine::runtime::SessionPreparationRequest prepare_request;
        prepare_request.audio = engine::runtime::AudioPreparationContract{
            warmup_audio.sample_rate,
            warmup_audio.channels,
            prepare_samples,
        };
        prepare_request.text = engine::runtime::Transcript{context, "auto"};
        session->prepare(prepare_request);

        engine::runtime::TaskRequest warmup_request;
        warmup_request.audio_input = warmup_audio;
        warmup_request.text_input = engine::runtime::Transcript{context, "auto"};
        warmup_request.options["max_tokens"] = max_tokens;
        warmup_request.options["temperature"] = temperature;
        warmup_request.options["top_p"] = top_p;
        warmup_request.options["top_k"] = top_k;
        warmup_request.options["num_beams"] = num_beams;
        warmup_request.options["seed"] = seed;
        for (const auto & [key, value] : request_option_overrides) {
            warmup_request.options[key] = value;
        }
        for (int i = 0; i < warmup; ++i) {
            (void) session->run(warmup_request);
        }

        const auto context_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--context-sequence", ""));
        const auto max_tokens_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--max-tokens-sequence", ""));
        const auto temperature_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--temperature-sequence", ""));
        const auto top_p_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--top-p-sequence", ""));
        const auto top_k_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--top-k-sequence", ""));
        const auto num_beams_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--num-beams-sequence", ""));
        const auto seed_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--seed-sequence", ""));

        engine::io::json::Value::Array steps;
        steps.reserve(request_paths.size());
        for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
            const std::string request_context = repeated_sequence_arg(context_sequence, request_index, context);
            const std::string request_max_tokens = repeated_sequence_arg(max_tokens_sequence, request_index, max_tokens);
            const std::string request_temperature = repeated_sequence_arg(temperature_sequence, request_index, temperature);
            const std::string request_top_p = repeated_sequence_arg(top_p_sequence, request_index, top_p);
            const std::string request_top_k = repeated_sequence_arg(top_k_sequence, request_index, top_k);
            const std::string request_num_beams = repeated_sequence_arg(num_beams_sequence, request_index, num_beams);
            const std::string request_seed = repeated_sequence_arg(seed_sequence, request_index, seed);
            engine::runtime::TaskRequest run_request;
            run_request.audio_input = request_audio_buffers[request_index];
            run_request.text_input = engine::runtime::Transcript{request_context, "auto"};
            run_request.options["max_tokens"] = request_max_tokens;
            run_request.options["temperature"] = request_temperature;
            run_request.options["top_p"] = request_top_p;
            run_request.options["top_k"] = request_top_k;
            run_request.options["num_beams"] = request_num_beams;
            run_request.options["seed"] = request_seed;
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
            std::cout << "vibevoice_asr.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(
                last_result,
                static_cast<int>(request_index),
                request_paths[request_index],
                wall_ms,
                request_context,
                request_max_tokens,
                request_temperature,
                request_top_p,
                request_top_k,
                request_num_beams,
                request_seed));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("vibevoice_asr")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "vibevoice_asr_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
