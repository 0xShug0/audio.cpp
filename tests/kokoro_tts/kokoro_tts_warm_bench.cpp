#include "engine/framework/debug/trace.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstdlib>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char * kFixedWarmupText =
    "This is a fixed warmup request for the speech session benchmark.";
constexpr const char * kCaseCatalogPath = "tests/kokoro_tts/kokoro_tts_warm_bench_cases.txt";

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<std::string> arg_values(int argc, char ** argv, const std::string & name) {
    std::vector<std::string> values;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            values.push_back(argv[i + 1]);
        }
    }
    return values;
}

std::unordered_map<std::string, std::string> parse_key_value_args(
    int argc,
    char ** argv,
    const std::string & name) {
    std::unordered_map<std::string, std::string> values;
    for (const std::string & entry : arg_values(argc, argv, name)) {
        const auto equals = entry.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 >= entry.size()) {
            throw std::runtime_error("expected key=value for " + name + ", got: " + entry);
        }
        values[entry.substr(0, equals)] = entry.substr(equals + 1);
    }
    return values;
}

std::string trim_ascii(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while (!value.empty() && is_space(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && is_space(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::unordered_map<std::string, std::vector<std::string>> load_case_catalog(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open Kokoro warm bench case catalog: " + path.string());
    }
    std::unordered_map<std::string, std::vector<std::string>> cases;
    std::string current_case;
    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = trim_ascii(line);
        if (trimmed.empty() || (!trimmed.empty() && trimmed.front() == '#')) {
            continue;
        }
        if (trimmed.size() >= 2 && trimmed.front() == '[' && trimmed.back() == ']') {
            current_case = trimmed.substr(1, trimmed.size() - 2);
            cases[current_case];
            continue;
        }
        if (current_case.empty()) {
            throw std::runtime_error("Kokoro warm bench case catalog entry is missing a [case] header");
        }
        cases[current_case].push_back(trimmed);
    }
    return cases;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

float float_arg(int argc, char ** argv, const std::string & name, float fallback) {
    return std::stof(arg_value(argc, argv, name, std::to_string(fallback)));
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

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string timing_key(const std::string & line) {
    const auto prefix_end = line.find("] ");
    if (prefix_end == std::string::npos) {
        return {};
    }
    const auto first_space = line.find(' ', prefix_end + 2);
    if (first_space == std::string::npos) {
        return {};
    }
    return line.substr(prefix_end + 2, first_space - (prefix_end + 2));
}

bool is_ms_timing_line(const std::string & line) {
    return starts_with(line, "[TIMING") && ends_with(timing_key(line), "_ms");
}

bool is_bench_evidence_line(const std::string & line) {
    if (!starts_with(line, "[TIMING")) {
        return false;
    }
    const auto key = timing_key(line);
    return ends_with(key, "_ms");
}

std::string timestamp_seconds_local() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif
    std::ostringstream out;
    out << std::put_time(&local_tm, "%Y%m%d-%H%M%S");
    return out.str();
}

void set_process_env(const char * key, const std::string & value) {
#if defined(_WIN32)
    if (_putenv_s(key, value.c_str()) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#else
    if (setenv(key, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#endif
}

std::unordered_map<std::string, double> parse_timing_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open timing log: " + path.string());
    }
    std::unordered_map<std::string, double> metrics;
    std::string line;
    while (std::getline(input, line)) {
        if (!is_ms_timing_line(line)) {
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
        try {
            metrics[key] = std::stod(value);
        } catch (...) {
        }
    }
    return metrics;
}

std::vector<std::string> read_timing_lines(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open timing log: " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (is_bench_evidence_line(line)) {
            lines.push_back(line);
        }
    }
    return lines;
}

void clear_file(const std::filesystem::path & path) {
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to clear timing log: " + path.string());
    }
}

void write_sectioned_timing_log(
    const std::filesystem::path & path,
    const std::vector<std::pair<std::string, std::vector<std::string>>> & sections) {
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write timing log: " + path.string());
    }
    for (size_t i = 0; i < sections.size(); ++i) {
        output << "[" << sections[i].first << "]\n";
        for (const std::string & line : sections[i].second) {
            output << line << "\n";
        }
        if (i + 1 < sections.size()) {
            output << "\n";
        }
    }
}

std::string summary_json(const engine::runtime::TaskResult & result) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("Kokoro warm bench expected audio_output");
    }
    const auto & audio = *result.audio_output;

    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float min_value = 0.0f;
    float max_value = 0.0f;
    if (!audio.samples.empty()) {
        min_value = audio.samples.front();
        max_value = audio.samples.front();
    }
    for (float sample : audio.samples) {
        sum += sample;
        sum_abs += std::abs(sample);
        sum_sq += static_cast<double>(sample) * static_cast<double>(sample);
        min_value = std::min(min_value, sample);
        max_value = std::max(max_value, sample);
    }

    std::ostringstream out;
    out << "{";
    out << "\"sample_rate\":" << audio.sample_rate << ",";
    out << "\"channels\":" << audio.channels << ",";
    out << "\"samples\":" << audio.samples.size() << ",";
    out << "\"sum\":" << std::fixed << std::setprecision(9) << sum << ",";
    out << "\"mean_abs\":" << (audio.samples.empty() ? 0.0 : sum_abs / static_cast<double>(audio.samples.size())) << ",";
    out << "\"rms\":" << (audio.samples.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(audio.samples.size()))) << ",";
    out << "\"min\":" << min_value << ",";
    out << "\"max\":" << max_value << ",";
    out << "\"first_samples\":[";
    const size_t preview = std::min<size_t>(32, audio.samples.size());
    for (size_t i = 0; i < preview; ++i) {
        if (i != 0) {
            out << ",";
        }
        out << audio.samples[i];
    }
    out << "]";
    out << "}";
    return out.str();
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/kokoro-82m-v1_0-ggml");
        std::vector<std::string> texts = arg_values(argc, argv, "--text");
        const std::vector<std::string> case_names = arg_values(argc, argv, "--case-name");
        if (!case_names.empty()) {
            const auto case_catalog = load_case_catalog(kCaseCatalogPath);
            for (const std::string & case_name : case_names) {
                const auto it = case_catalog.find(case_name);
                if (it == case_catalog.end()) {
                    throw std::runtime_error("unknown Kokoro warm bench case: " + case_name);
                }
                texts.insert(texts.end(), it->second.begin(), it->second.end());
            }
        }
        if (texts.empty()) {
            texts.push_back("Hello world. This is a warm benchmark for the Kokoro framework session.");
        }
        const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 1);
        const std::string warmup_text = arg_value(argc, argv, "--warmup-text", kFixedWarmupText);
        const int iterations = int_arg(argc, argv, "--iterations", 5);
        const std::string voice_id = arg_value(argc, argv, "--voice-id", "af_heart");
        const std::string style_language = arg_value(argc, argv, "--style-language", "");
        const float speaking_rate = float_arg(argc, argv, "--speaking-rate", 1.0f);
        const std::string graph_capacity_mode = arg_value(argc, argv, "--graph-capacity-mode", "fixed");
        const int max_input_tokens = int_arg(argc, argv, "--max-input-tokens", 512);
        const int rng_seed = int_arg(argc, argv, "--rng-seed", 0);
        const auto session_option_overrides = parse_key_value_args(argc, argv, "--session-option");
        const std::string artifact_stamp = arg_value(argc, argv, "--artifact-stamp", timestamp_seconds_local());
        const std::filesystem::path audio_out_path =
            arg_value(argc, argv, "--audio-out", "kokoro_cpp_audio.wav");
        const std::string audio_out_dir_arg = arg_value(argc, argv, "--audio-out-dir", "");
        const std::filesystem::path audio_out_dir =
            audio_out_dir_arg.empty() ? std::filesystem::path{} : std::filesystem::path(audio_out_dir_arg);
        const std::string timing_file_arg = arg_value(argc, argv, "--timing-file", "");
        const std::filesystem::path timing_path = timing_file_arg.empty()
            ? (std::filesystem::path("build/logs/parity/kokoro_tts") /
               ("kokoro_tts_cpp_" + backend_name + "-" + artifact_stamp + ".log"))
            : std::filesystem::path(timing_file_arg);

        if (!timing_path.parent_path().empty()) {
            std::filesystem::create_directories(timing_path.parent_path());
        }

        set_process_env("ENGINE_TRACE_ENABLED", "0");
        set_process_env("ENGINE_TIMING_ENABLED", "1");
        set_process_env("ENGINE_TIMING_FILE", timing_path.string());

        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(model_path);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        session_options.options["offline_graph_capacity_mode"] = graph_capacity_mode;
        session_options.options["max_input_tokens"] = std::to_string(max_input_tokens);
        session_options.options["kokoro_rng_seed"] = std::to_string(rng_seed);
        for (const auto & [key, value] : session_option_overrides) {
            session_options.options[key] = value;
        }

        const engine::runtime::TaskSpec task{
            engine::runtime::VoiceTaskKind::Tts,
            engine::runtime::RunMode::Offline,
        };

        auto session_base = model->create_task_session(task, session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("loaded Kokoro session is not an offline TTS session");
        }

        std::vector<engine::runtime::TaskRequest> requests;
        requests.reserve(texts.size());
        for (const std::string & text : texts) {
            engine::runtime::TaskRequest request;
            request.text_input = engine::runtime::Transcript{text, ""};
            request.voice = engine::runtime::VoiceCondition{};
            request.voice->speaker = engine::runtime::VoiceReference{};
            request.voice->speaker->cached_voice_id = voice_id;
            request.voice->style = engine::runtime::StyleCondition{};
            request.voice->style->speaking_rate = speaking_rate;
            if (!style_language.empty()) {
                request.voice->style->language = style_language;
            }
            requests.push_back(std::move(request));
        }

        engine::runtime::TaskRequest warmup_request;
        warmup_request.text_input = engine::runtime::Transcript{warmup_text, ""};
        warmup_request.voice = engine::runtime::VoiceCondition{};
        warmup_request.voice->speaker = engine::runtime::VoiceReference{};
        warmup_request.voice->speaker->cached_voice_id = voice_id;
        warmup_request.voice->style = engine::runtime::StyleCondition{};
        warmup_request.voice->style->speaking_rate = speaking_rate;
        if (!style_language.empty()) {
            warmup_request.voice->style->language = style_language;
        }

        clear_file(timing_path);
        session_base->prepare(engine::runtime::build_preparation_request(warmup_request));
        std::vector<std::string> prepare_lines = read_timing_lines(timing_path);
        std::vector<std::pair<std::string, std::vector<std::string>>> log_sections;
        log_sections.push_back({"prepare", prepare_lines});

        const int warmup_passes = std::max(0, warmup);

        std::vector<std::map<std::string, double>> sums(requests.size());
        std::vector<engine::runtime::TaskResult> last_results(requests.size());
        for (int i = 0; i < warmup_passes; ++i) {
            clear_file(timing_path);
            (void) session->run(warmup_request);
            auto lines = read_timing_lines(timing_path);
            log_sections.push_back({
                "warmup" + std::to_string(i + 1),
                std::move(lines),
            });
        }

        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            for (int i = 0; i < iterations; ++i) {
                clear_file(timing_path);
                last_results[request_index] = session->run(requests[request_index]);
                const auto metrics = parse_timing_file(timing_path);
                auto lines = read_timing_lines(timing_path);
                log_sections.push_back({
                    "iteration" + std::to_string(i + 1) + ".request" + std::to_string(request_index + 1),
                    std::move(lines),
                });
                for (const auto & [key, value] : metrics) {
                    sums[request_index][key] += value;
                }
            }
        }

        write_sectioned_timing_log(timing_path, log_sections);

        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            std::cout << "text[" << request_index << "]=" << texts[request_index] << "\n";
            std::cout << "summary_json[" << request_index << "]="
                      << summary_json(last_results[request_index]) << "\n";
            if (requests.size() == 1 && request_index == 0) {
                std::cout << "text=" << texts[request_index] << "\n";
                std::cout << "summary_json=" << summary_json(last_results[request_index]) << "\n";
            }
        }
        std::cout << "timing_out=" << timing_path.string() << "\n";
        if (!audio_out_dir.empty()) {
            std::filesystem::create_directories(audio_out_dir);
            for (size_t request_index = 0; request_index < last_results.size(); ++request_index) {
                const auto & request_result = last_results[request_index];
                if (!request_result.audio_output.has_value()) {
                    throw std::runtime_error("Kokoro warm bench expected audio_output for per-request WAV write");
                }
                const auto request_audio_out = audio_out_dir / ("request_" + std::to_string(request_index) + ".wav");
                engine::audio::write_pcm16_wav(
                    request_audio_out,
                    request_result.audio_output->sample_rate,
                    request_result.audio_output->channels,
                    request_result.audio_output->samples);
                std::cout << "audio_out[" << request_index << "]=" << request_audio_out.string() << "\n";
            }
        }
        const auto & last_result = last_results.back();
        if (!last_result.audio_output.has_value()) {
            throw std::runtime_error("Kokoro warm bench expected audio_output for WAV write");
        }
        engine::audio::write_pcm16_wav(
            audio_out_path,
            last_result.audio_output->sample_rate,
            last_result.audio_output->channels,
            last_result.audio_output->samples);
        std::cout << "audio_out=" << audio_out_path.string() << "\n";
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            std::cout << "average[" << request_index << "]\n";
            for (const auto & [key, sum] : sums[request_index]) {
                std::cout << key << "=" << (sum / static_cast<double>(std::max(1, iterations))) << "\n";
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "kokoro_tts_warm_bench failed: " << e.what() << "\n";
        return 1;
    }
}
