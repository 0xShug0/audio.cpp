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
            throw std::runtime_error("invalid Nemotron ASR " + name + ": " + option);
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
    double wall_ms,
    int64_t lookahead_tokens,
    bool streaming) {
    return engine::io::json::Value::make_object({
        {"request_index", number(request_index)},
        {"audio", string(audio_path.string())},
        {"requested_language", string(requested_language)},
        {"language", string(result.text_output.has_value() ? result.text_output->language : "")},
        {"text_output", string(result.text_output.has_value() ? result.text_output->text : "")},
        {"word_timestamps", engine::tools::word_timestamps_json(result.word_timestamps)},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", number(wall_ms)},
            {"lookahead_tokens", number(static_cast<double>(lookahead_tokens))},
            {"streaming", number(streaming ? 1.0 : 0.0)},
        })},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path =
            engine::tools::arg_value(argc, argv, "--model", "models/nemotron-3.5-asr-streaming-0.6b");
        const std::string audio_sequence_value = engine::tools::arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path = engine::tools::arg_value(argc, argv, "--audio", "resources/sample_16k.wav");
        const std::filesystem::path warmup_audio_path =
            engine::tools::arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cpu");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 0);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string default_language = engine::tools::arg_value(argc, argv, "--language", "en-US");
        const std::string warmup_language = engine::tools::arg_value(argc, argv, "--warmup-language", default_language);
        const std::string lookahead_tokens = engine::tools::arg_value(argc, argv, "--lookahead-tokens", "3");
        const std::string max_tokens = engine::tools::arg_value(argc, argv, "--max-tokens", "256");
        const std::string streaming = engine::tools::arg_value(argc, argv, "--streaming", "false");
        const std::string keep_language_tags = engine::tools::arg_value(argc, argv, "--keep-language-tags", "false");
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "/tmp/nemotron_asr_warm_bench_timing.log");

        setenv("MINITTS_TRACE_ENABLED", "0", 1);
        setenv("MINITTS_TIMING_ENABLED", "1", 1);
        setenv("MINITTS_TIMING_FILE", timing_path.c_str(), 1);
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "nemotron_asr";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = engine::tools::parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & [key, value] : parse_key_value_options(argc, argv, "--session-option")) {
            session_options.options.emplace(key, value);
        }
        const auto request_option_overrides = parse_key_value_options(argc, argv, "--request-option");

        const bool streaming_session =
            streaming == "true" || streaming == "1" || streaming == "yes" || streaming == "on";
        auto session_base = model->create_task_session(
            engine::runtime::TaskSpec{
                engine::runtime::VoiceTaskKind::Asr,
                streaming_session ? engine::runtime::RunMode::Streaming : engine::runtime::RunMode::Offline},
            session_options);
        auto * offline_session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        auto * stream_session = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session_base.get());
        if (streaming_session) {
            if (stream_session == nullptr) {
                throw std::runtime_error("Nemotron ASR did not create a streaming task session");
            }
        } else if (offline_session == nullptr) {
            throw std::runtime_error("Nemotron ASR did not create an offline task session");
        }

        std::vector<std::filesystem::path> request_paths;
        if (!audio_sequence_value.empty()) {
            for (const auto & item : engine::tools::split_csv(audio_sequence_value)) {
                request_paths.emplace_back(item);
            }
        } else {
            request_paths.push_back(audio_path);
        }
        std::vector<engine::runtime::AudioBuffer> request_audio_buffers;
        request_audio_buffers.reserve(request_paths.size());
        for (const auto & path : request_paths) {
            request_audio_buffers.push_back(engine::tools::read_audio_buffer(path));
        }

        const auto warmup_audio = engine::tools::read_audio_buffer(warmup_audio_path);
        int64_t prepare_samples = static_cast<int64_t>(warmup_audio.samples.size());
        for (const auto & candidate : request_audio_buffers) {
            if (candidate.sample_rate == warmup_audio.sample_rate &&
                candidate.channels == warmup_audio.channels &&
                static_cast<int64_t>(candidate.samples.size()) > prepare_samples) {
                prepare_samples = static_cast<int64_t>(candidate.samples.size());
            }
        }
        const auto request_languages = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--language-sequence", ""));
        const auto lookahead_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--lookahead-tokens-sequence", ""));
        const auto max_tokens_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--max-tokens-sequence", ""));
        const auto streaming_sequence = split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--streaming-sequence", ""));
        const auto keep_language_tags_sequence =
            split_csv_keep_empty(engine::tools::arg_value(argc, argv, "--keep-language-tags-sequence", ""));
        int64_t prepare_lookahead = std::stoll(lookahead_tokens);
        for (const auto & item : lookahead_sequence) {
            if (!item.empty()) {
                prepare_lookahead = std::max<int64_t>(prepare_lookahead, std::stoll(item));
            }
        }
        engine::runtime::SessionPreparationRequest prepare_request;
        prepare_request.audio = engine::runtime::AudioPreparationContract{
            warmup_audio.sample_rate,
            warmup_audio.channels,
            prepare_samples,
        };
        prepare_request.options["lookahead_tokens"] = std::to_string(prepare_lookahead);
        prepare_request.options["streaming"] = streaming;
        prepare_request.options["max_tokens"] = max_tokens;
        prepare_request.options["keep_language_tags"] = keep_language_tags;
        prepare_request.text = engine::runtime::Transcript{"", warmup_language};
        session_base->prepare(prepare_request);

        engine::runtime::TaskRequest warmup_request;
        warmup_request.audio_input = warmup_audio;
        warmup_request.text_input = engine::runtime::Transcript{"", warmup_language};
        warmup_request.options["lookahead_tokens"] = lookahead_tokens;
        warmup_request.options["max_tokens"] = max_tokens;
        warmup_request.options["streaming"] = streaming;
        warmup_request.options["keep_language_tags"] = keep_language_tags;
        for (const auto & [key, value] : request_option_overrides) {
            warmup_request.options[key] = value;
        }
        for (int i = 0; i < warmup; ++i) {
            if (streaming_session) {
                stream_session->reset();
                stream_session->process_audio_chunk({
                    warmup_audio.sample_rate,
                    warmup_audio.channels,
                    0,
                    warmup_audio.samples,
                });
                (void) stream_session->finalize();
            } else {
                (void) offline_session->run(warmup_request);
            }
        }

        engine::io::json::Value::Array steps;
        steps.reserve(request_paths.size());
        for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
            const std::string language = request_index < request_languages.size() && !request_languages[request_index].empty()
                ? request_languages[request_index]
                : repeated_arg(argc, argv, "--request-language", request_index, default_language);
            const std::string request_lookahead = repeated_sequence_arg(lookahead_sequence, request_index, lookahead_tokens);
            const std::string request_streaming = repeated_sequence_arg(streaming_sequence, request_index, streaming);
            const std::string request_keep_language_tags =
                repeated_sequence_arg(keep_language_tags_sequence, request_index, keep_language_tags);
            engine::runtime::TaskResult last_result;
            double total_ms = 0.0;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                if (streaming_session) {
                    stream_session->reset();
                    const auto & audio = request_audio_buffers[request_index];
                    stream_session->process_audio_chunk({
                        audio.sample_rate,
                        audio.channels,
                        0,
                        audio.samples,
                    });
                    last_result = stream_session->finalize();
                } else {
                    engine::runtime::TaskRequest run_request;
                    run_request.audio_input = request_audio_buffers[request_index];
                    run_request.text_input = engine::runtime::Transcript{"", language};
                    run_request.options["lookahead_tokens"] = request_lookahead;
                    run_request.options["max_tokens"] = repeated_sequence_arg(max_tokens_sequence, request_index, max_tokens);
                    run_request.options["streaming"] = request_streaming;
                    run_request.options["keep_language_tags"] = request_keep_language_tags;
                    for (const auto & [key, value] : request_option_overrides) {
                        run_request.options[key] = value;
                    }
                    last_result = offline_session->run(run_request);
                }
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            const double wall_ms = total_ms / static_cast<double>(iterations);
            std::cout << "average[" << request_index << "]\n";
            std::cout << "nemotron_asr.wall_ms=" << wall_ms << "\n";
            const int64_t request_lookahead_i64 = std::stoll(request_lookahead);
            const bool request_streaming_bool =
                request_streaming == "true" || request_streaming == "1" || request_streaming == "yes" || request_streaming == "on";
            steps.push_back(step_json(
                last_result,
                static_cast<int>(request_index),
                request_paths[request_index],
                language,
                wall_ms,
                request_lookahead_i64,
                request_streaming_bool));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("nemotron_asr")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "nemotron_asr_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
