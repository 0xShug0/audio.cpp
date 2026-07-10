#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
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

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & value) {
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

bool starts_with(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

double parse_timing_value(const std::string &, const std::string & value) {
    return std::stod(value);
}

std::unordered_map<std::string, double> parse_timing_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open timing log: " + path.string());
    }

    std::unordered_map<std::string, double> metrics;
    std::string line;
    while (std::getline(input, line)) {
        if (!starts_with(line, "[TIMING")) {
            continue;
        }
        const auto prefix_end = line.find("] ");
        if (prefix_end == std::string::npos) {
            continue;
        }
        const auto first_space = line.find(' ', prefix_end + 2);
        if (first_space == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(prefix_end + 2, first_space - (prefix_end + 2));
        const std::string value = line.substr(first_space + 1);
        if (key == "audio.stft_impl") {
            continue;
        }
        metrics[key] = parse_timing_value(key, value);
    }
    return metrics;
}

void clear_file(const std::filesystem::path & path) {
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to clear timing log: " + path.string());
    }
}

std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<std::string> repeated_arg_values(int argc, char ** argv, const std::string & name) {
    std::vector<std::string> values;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            values.emplace_back(argv[i + 1]);
        }
    }
    return values;
}

std::pair<std::string, std::string> split_key_value(const std::string & value) {
    const auto pos = value.find('=');
    if (pos == std::string::npos || pos == 0) {
        throw std::runtime_error("session option must be key=value: " + value);
    }
    return {value.substr(0, pos), value.substr(pos + 1)};
}

const std::vector<std::string> & ordered_keys() {
    static const std::vector<std::string> keys = {
        "parakeet.frontend_ms",
        "parakeet.pre_encode_ms",
        "parakeet.encoder_ms",
        "parakeet.longform.attention_ms",
        "parakeet.longform.non_attention_ms",
        "parakeet.decoder_ms",
        "parakeet.buffered_capacity_frames",
        "parakeet.buffered_valid_frames",
        "parakeet.timestamps_ms",
        "parakeet.transcribe_wall_ms",
    };
    return keys;
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += ch;
            break;
        }
    }
    return out;
}

void print_word_timestamps_plain(const std::vector<engine::runtime::WordTimestamp> & items) {
    std::cout << "word_timestamps=[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        const auto & item = items[i];
        std::cout << "{"
                  << "\"start_sample\":" << item.span.start_sample << ","
                  << "\"end_sample\":" << item.span.end_sample << ","
                  << "\"word\":\"" << json_escape(item.word) << "\","
                  << "\"confidence\":" << std::fixed << std::setprecision(6) << item.confidence
                  << "}";
    }
    std::cout << "]\n";
}

void print_word_timestamps_json(const std::vector<engine::runtime::WordTimestamp> & items) {
    std::cout << "\"word_timestamps\":[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        const auto & item = items[i];
        std::cout << "{"
                  << "\"start_sample\":" << item.span.start_sample << ","
                  << "\"end_sample\":" << item.span.end_sample << ","
                  << "\"word\":\"" << json_escape(item.word) << "\","
                  << "\"confidence\":" << std::fixed << std::setprecision(6) << item.confidence
                  << "}";
    }
    std::cout << "]";
}

void print_diagnostics_json() {
    std::cout << "\"diagnostics\":{}";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/parakeet-tdt-0.6b-v3");
        const std::string audio_sequence_value = arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path =
            arg_value(argc, argv, "--audio", "build/assets/parakeet/2086-149220-0033_5s.wav");
        const std::filesystem::path warmup_audio_path =
            arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 1);
        const int iterations = int_arg(argc, argv, "--iterations", 5);
        const std::string run_mode = arg_value(argc, argv, "--run-mode", "offline");
        const std::string encoder_variant = arg_value(argc, argv, "--encoder-variant", "full_context");
        const std::string graph_capacity_mode =
            arg_value(argc, argv, "--graph-capacity-mode", run_mode == "streaming" ? "fixed" : "tiered");
        const std::string decoder_algorithm = arg_value(argc, argv, "--decoder-algorithm", "greedy_duration_loop");
        const std::string full_context_capacity_frames = arg_value(argc, argv, "--full-context-capacity-frames", "");
        const std::string long_context_capacity_frames = arg_value(argc, argv, "--long-context-capacity-frames", "");
        const std::string streaming_chunk_secs = arg_value(argc, argv, "--streaming-chunk-secs", "");
        const std::string streaming_left_context_secs = arg_value(argc, argv, "--streaming-left-context-secs", "");
        const std::string streaming_right_context_secs = arg_value(argc, argv, "--streaming-right-context-secs", "");
        const int chunk_size = int_arg(argc, argv, "--chunk-size", 32000);
        const std::filesystem::path timing_path =
            arg_value(argc, argv, "--timing-file", "/tmp/parakeet_warm_bench_timing.log");

        setenv("ENGINE_TRACE_ENABLED", "0", 1);
        setenv("ENGINE_TIMING_ENABLED", "1", 1);
        setenv("ENGINE_TIMING_FILE", timing_path.c_str(), 1);

        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(model_path);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        session_options.options["decoder_algorithm"] = decoder_algorithm;
        session_options.options["encoder_variant"] = encoder_variant;
        if (run_mode == "streaming") {
            session_options.options["streaming_graph_capacity_mode"] = graph_capacity_mode;
            if (!streaming_chunk_secs.empty()) {
                session_options.options["chunk_secs"] = streaming_chunk_secs;
            }
            if (!streaming_left_context_secs.empty()) {
                session_options.options["left_context_secs"] = streaming_left_context_secs;
            }
            if (!streaming_right_context_secs.empty()) {
                session_options.options["right_context_secs"] = streaming_right_context_secs;
            }
        } else {
            session_options.options["offline_graph_capacity_mode"] = graph_capacity_mode;
            if (!full_context_capacity_frames.empty()) {
                session_options.options["full_context_capacity_frames"] = full_context_capacity_frames;
            }
            if (!long_context_capacity_frames.empty()) {
                session_options.options["long_context_capacity_frames"] = long_context_capacity_frames;
            }
        }
        for (const auto & option : repeated_arg_values(argc, argv, "--session-option")) {
            auto [key, value] = split_key_value(option);
            session_options.options[std::move(key)] = std::move(value);
        }

        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Asr,
            run_mode == "streaming" ? engine::runtime::RunMode::Streaming : engine::runtime::RunMode::Offline,
        };
        const std::string family_name = model->metadata().family;
        std::vector<std::filesystem::path> audio_paths;
        if (!audio_sequence_value.empty()) {
            for (const auto & item : split_csv(audio_sequence_value)) {
                audio_paths.emplace_back(item);
            }
        }
        if (audio_paths.empty()) {
            audio_paths.push_back(audio_path);
        }
        std::vector<engine::audio::WavData> wavs;
        wavs.reserve(audio_paths.size());
        for (const auto & path : audio_paths) {
            const auto wav = engine::audio::read_wav_f32(path);
            if (wav.channels != 1) {
                throw std::runtime_error("parakeet_warm_bench requires mono WAV input");
            }
            wavs.push_back(wav);
        }
        const auto warmup_wav = engine::audio::read_wav_f32(warmup_audio_path);
        if (warmup_wav.channels != 1) {
            throw std::runtime_error("parakeet_warm_bench requires mono WAV warmup input");
        }
        engine::runtime::TaskRequest warmup_request;
        warmup_request.audio_input = engine::runtime::AudioBuffer{
            warmup_wav.sample_rate,
            warmup_wav.channels,
            warmup_wav.samples,
        };

        struct SequenceStepSummary {
            std::string audio_path;
            engine::runtime::TaskResult result;
            std::unordered_map<std::string, double> metrics;
        };

        engine::runtime::TaskResult last_result;
        std::vector<SequenceStepSummary> last_sequence_steps;
        double streaming_last_wall_ms = 0.0;
        std::function<std::vector<SequenceStepSummary>()> run_once;
        std::function<void()> warmup_once;
        std::unique_ptr<engine::runtime::IVoiceTaskSession> streaming_session_base;
        std::unique_ptr<engine::runtime::IVoiceTaskSession> offline_session_base;
        if (run_mode == "streaming") {
            streaming_session_base = model->create_task_session(task, session_options);
            streaming_session_base->prepare(engine::runtime::build_preparation_request(warmup_request));
            auto * session = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(streaming_session_base.get());
            if (session == nullptr) {
                throw std::runtime_error("loaded model did not produce a streaming ASR session");
            }
            auto run_streaming_once = [session, &wavs, &audio_paths, chunk_size, &streaming_last_wall_ms]() {
                std::vector<SequenceStepSummary> steps;
                steps.reserve(wavs.size());
                streaming_last_wall_ms = 0.0;
                for (size_t wav_index = 0; wav_index < wavs.size(); ++wav_index) {
                    session->reset();
                    const auto started = std::chrono::steady_clock::now();
                    const auto & wav = wavs[wav_index];
                    const auto & samples = wav.samples;
                    const size_t chunk_size_samples = static_cast<size_t>(chunk_size);
                    const size_t lead_samples = chunk_size_samples * 2;
                    auto send_chunk = [&](size_t offset, size_t available) {
                        std::vector<float> chunk(available, 0.0f);
                        std::copy(
                            samples.begin() + static_cast<ptrdiff_t>(offset),
                            samples.begin() + static_cast<ptrdiff_t>(offset + available),
                            chunk.begin());
                        (void)session->process_audio_chunk({
                            wav.sample_rate,
                            1,
                            static_cast<int64_t>(offset),
                            std::move(chunk),
                        });
                    };

                    size_t offset = 0;
                    if (samples.size() <= lead_samples) {
                        send_chunk(0, samples.size());
                    } else {
                        send_chunk(0, lead_samples);
                        offset = lead_samples;
                        while (offset + lead_samples < samples.size()) {
                            send_chunk(offset, chunk_size_samples);
                            offset += chunk_size_samples;
                        }
                        if (offset < samples.size()) {
                            send_chunk(offset, samples.size() - offset);
                        }
                    }
                    auto result = session->finalize();
                    const auto ended = std::chrono::steady_clock::now();
                    const double step_wall_ms = std::chrono::duration<double, std::milli>(ended - started).count();
                    streaming_last_wall_ms += step_wall_ms;
                    SequenceStepSummary step;
                    step.audio_path = audio_paths[wav_index].string();
                    step.result = std::move(result);
                    step.metrics["parakeet.transcribe_wall_ms"] = step_wall_ms;
                    steps.push_back(std::move(step));
                }
                return steps;
            };
            warmup_once = [run_streaming_once]() mutable {
                (void)run_streaming_once();
            };
            run_once = [run_streaming_once]() mutable {
                return run_streaming_once();
            };
        } else {
            offline_session_base = model->create_task_session(task, session_options);
            engine::runtime::TaskRequest prepare_request;
            prepare_request.audio_input = engine::runtime::AudioBuffer{
                warmup_wav.sample_rate,
                warmup_wav.channels,
                warmup_wav.samples,
            };
            offline_session_base->prepare(engine::runtime::build_preparation_request(prepare_request));
            auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(offline_session_base.get());
            if (session == nullptr) {
                throw std::runtime_error("loaded model did not produce an offline ASR session");
            }
            std::vector<engine::runtime::TaskRequest> requests;
            requests.reserve(wavs.size());
            for (const auto & wav : wavs) {
                engine::runtime::TaskRequest request;
                request.audio_input = engine::runtime::AudioBuffer{
                    wav.sample_rate,
                    wav.channels,
                    wav.samples,
                };
                requests.push_back(std::move(request));
            }
            warmup_once = [session, warmup_request, timing_path]() {
                clear_file(timing_path);
                (void)session->run(warmup_request);
            };
            run_once = [session, requests, audio_paths, timing_path]() {
                std::vector<SequenceStepSummary> steps;
                steps.reserve(requests.size());
                for (size_t i = 0; i < requests.size(); ++i) {
                    clear_file(timing_path);
                    const auto step_started = std::chrono::steady_clock::now();
                    auto result = session->run(requests[i]);
                    const auto step_ended = std::chrono::steady_clock::now();
                    SequenceStepSummary step;
                    step.audio_path = audio_paths[i].string();
                    step.result = std::move(result);
                    step.metrics = parse_timing_file(timing_path);
                    step.metrics["parakeet.transcribe_wall_ms"] =
                        std::chrono::duration<double, std::milli>(step_ended - step_started).count();
                    steps.push_back(std::move(step));
                }
                return steps;
            };
        }

        std::map<std::string, double> sums;
        std::vector<std::unordered_map<std::string, double>> per_run;
        per_run.reserve(static_cast<size_t>(std::max(1, iterations)));

        for (int i = 0; i < std::max(0, warmup); ++i) {
            warmup_once();
        }

        for (int i = 0; i < std::max(1, iterations); ++i) {
            const auto started = std::chrono::steady_clock::now();
            last_sequence_steps = run_once();
            const auto ended = std::chrono::steady_clock::now();
            std::unordered_map<std::string, double> metrics;
            for (const auto & step : last_sequence_steps) {
                for (const auto & [key, value] : step.metrics) {
                    metrics[key] += value;
                }
                last_result = step.result;
            }
            if (run_mode == "streaming") {
                (void)started;
                (void)ended;
                metrics["parakeet.transcribe_wall_ms"] = streaming_last_wall_ms;
            } else {
                metrics["parakeet.transcribe_wall_ms"] =
                    std::chrono::duration<double, std::milli>(ended - started).count();
            }
            per_run.push_back(metrics);
            for (const auto & [key, value] : metrics) {
                sums[key] += value;
            }
        }

        std::cout << "family=" << family_name << "\n";
        std::cout << "backend=" << backend_name << "\n";
        std::cout << "threads=" << threads << "\n";
        std::cout << "warmup=" << warmup << "\n";
        std::cout << "iterations=" << iterations << "\n";
        std::cout << "run_mode=" << run_mode << "\n";
        std::cout << "encoder_variant=" << encoder_variant << "\n";
        std::cout << "decoder_algorithm=" << decoder_algorithm << "\n";
        if (last_result.text_output.has_value()) {
            std::cout << "text_output=" << last_result.text_output->text << "\n";
        }
        print_word_timestamps_plain(last_result.word_timestamps);
        if (last_sequence_steps.size() > 1) {
            std::cout << "sequence_steps=" << last_sequence_steps.size() << "\n";
            for (size_t i = 0; i < last_sequence_steps.size(); ++i) {
                const auto & step = last_sequence_steps[i];
                std::cout << "sequence_step[" << i << "].audio=" << step.audio_path << "\n";
                if (step.result.text_output.has_value()) {
                    std::cout << "sequence_step[" << i << "].text_output=" << step.result.text_output->text << "\n";
                }
            }
        }

        for (size_t run_idx = 0; run_idx < per_run.size(); ++run_idx) {
            std::cout << "run=" << (run_idx + 1) << "\n";
            for (const auto & key : ordered_keys()) {
                const auto it = per_run[run_idx].find(key);
                if (it != per_run[run_idx].end()) {
                    std::cout << key << "=" << std::fixed << std::setprecision(6) << it->second << "\n";
                }
            }
        }

        std::cout << "average\n";
        for (const auto & key : ordered_keys()) {
            const auto it = sums.find(key);
            if (it != sums.end()) {
                std::cout << key << "="
                          << std::fixed
                          << std::setprecision(6)
                          << (it->second / static_cast<double>(std::max(1, iterations)))
                          << "\n";
            }
        }

        std::cout << "summary_json={";
        std::cout << "\"family\":\"" << json_escape(family_name) << "\",";
        std::cout << "\"backend\":\"" << json_escape(backend_name) << "\",";
        std::cout << "\"device\":" << device << ",";
        std::cout << "\"threads\":" << threads << ",";
        std::cout << "\"warmup\":" << warmup << ",";
        std::cout << "\"iterations\":" << iterations << ",";
        std::cout << "\"warmup_audio\":\"" << json_escape(warmup_audio_path.string()) << "\",";
        std::cout << "\"encoder_variant\":\"" << json_escape(encoder_variant) << "\",";
        std::cout << "\"decoder_algorithm\":\"" << json_escape(decoder_algorithm) << "\",";
        std::cout << "\"audio_sequence\":[";
        for (size_t i = 0; i < audio_paths.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << "\"" << json_escape(audio_paths[i].string()) << "\"";
        }
        std::cout << "],";
        std::cout << "\"text_output\":\""
                  << json_escape(last_result.text_output.has_value() ? last_result.text_output->text : "")
                  << "\",";
        print_word_timestamps_json(last_result.word_timestamps);
        std::cout << ",";
        print_diagnostics_json();
        std::cout << ",";
        std::cout << "\"runs\":[";
        for (size_t run_idx = 0; run_idx < per_run.size(); ++run_idx) {
            if (run_idx != 0) {
                std::cout << ",";
            }
            std::cout << "{";
            bool first_metric = true;
            for (const auto & key : ordered_keys()) {
                const auto it = per_run[run_idx].find(key);
                if (it == per_run[run_idx].end()) {
                    continue;
                }
                if (!first_metric) {
                    std::cout << ",";
                }
                first_metric = false;
                std::cout << "\"" << json_escape(key) << "\":" << std::fixed << std::setprecision(6) << it->second;
            }
            std::cout << "}";
        }
        std::cout << "],";
        std::cout << "\"sequence_steps\":[";
        for (size_t i = 0; i < last_sequence_steps.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            const auto & step = last_sequence_steps[i];
            std::cout << "{";
            std::cout << "\"audio\":\"" << json_escape(step.audio_path) << "\",";
            std::cout << "\"text_output\":\""
                      << json_escape(step.result.text_output.has_value() ? step.result.text_output->text : "")
                      << "\",";
            print_word_timestamps_json(step.result.word_timestamps);
            std::cout << ",";
            print_diagnostics_json();
            std::cout << ",\"metrics\":{";
            bool first_step_metric = true;
            for (const auto & key : ordered_keys()) {
                const auto it = step.metrics.find(key);
                if (it == step.metrics.end()) {
                    continue;
                }
                if (!first_step_metric) {
                    std::cout << ",";
                }
                first_step_metric = false;
                std::cout << "\"" << json_escape(key) << "\":" << std::fixed << std::setprecision(6) << it->second;
            }
            std::cout << "}}";
        }
        std::cout << "],";
        std::cout << "\"average\":{";
        bool first_average = true;
        for (const auto & key : ordered_keys()) {
            const auto it = sums.find(key);
            if (it == sums.end()) {
                continue;
            }
            if (!first_average) {
                std::cout << ",";
            }
            first_average = false;
            std::cout << "\""
                      << json_escape(key)
                      << "\":"
                      << std::fixed
                      << std::setprecision(6)
                      << (it->second / static_cast<double>(std::max(1, iterations)));
        }
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "parakeet_warm_bench failed: " << e.what() << "\n";
        return 1;
    }
}
