#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char * kDefaultWarmupText =
    "This is a fixed warmup request for the MOSS-TTS-Nano session benchmark.";

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

std::string timing_line_scalar(const std::string & timestamp, const std::string & key, const std::string & value) {
    return "[TIMING ts=" + timestamp + "] " + key + " " + value;
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

void append_request_evidence_lines(
    std::vector<std::string> & lines,
    const std::string & request_text,
    double wall_ms) {
    const std::string ts = timestamp_seconds_local();
    lines.push_back(timing_line_scalar(ts, "moss_tts_nano.request_char_count", std::to_string(request_text.size())));
    {
        std::ostringstream wall_stream;
        wall_stream << std::fixed << std::setprecision(6) << wall_ms;
        lines.push_back(timing_line_scalar(ts, "moss_tts_nano.request_wall_ms", wall_stream.str()));
    }
}

std::string summary_json(const engine::runtime::TaskResult & result, const std::string & request_text) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("MOSS-TTS-Nano warm bench expected audio_output");
    }
    const auto & audio = *result.audio_output;
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float min_value = 0.0F;
    float max_value = 0.0F;
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
    out << "\"samples\":" << (audio.channels <= 0 ? 0 : audio.samples.size() / static_cast<size_t>(audio.channels)) << ",";
    out << "\"sum\":" << std::fixed << std::setprecision(9) << sum << ",";
    out << "\"mean_abs\":" << (audio.samples.empty() ? 0.0 : sum_abs / static_cast<double>(audio.samples.size())) << ",";
    out << "\"rms\":" << (audio.samples.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(audio.samples.size()))) << ",";
    out << "\"min\":" << min_value << ",";
    out << "\"max\":" << max_value << ",";
    out << "\"request_char_count\":" << request_text.size() << ",";
    out << "\"first_samples\":[";
    const size_t first_count = std::min<size_t>(32, audio.samples.size());
    for (size_t i = 0; i < first_count; ++i) {
        if (i != 0) {
            out << ",";
        }
        out << audio.samples[i];
    }
    out << "]}";
    return out.str();
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/MOSS-TTS-Nano-100M");
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    const int device = int_arg(argc, argv, "--device", 0);
    const int threads = int_arg(argc, argv, "--threads", 8);
    const int warmup = int_arg(argc, argv, "--warmup", 1);
    const int iterations = int_arg(argc, argv, "--iterations", 1);
    const std::string warmup_text = arg_value(argc, argv, "--warmup-text", kDefaultWarmupText);
    const std::vector<std::string> texts = arg_values(argc, argv, "--text");
    const std::filesystem::path clone_audio_path = arg_value(argc, argv, "--clone-audio", "resources/sample.wav");
    const std::filesystem::path audio_out = arg_value(argc, argv, "--audio-out", "moss_tts_nano_cpp_audio.wav");
    const std::filesystem::path audio_out_dir = arg_value(argc, argv, "--audio-out-dir", "");
    const std::filesystem::path timing_path = arg_value(argc, argv, "--timing-file", "/tmp/moss_tts_nano_warm_bench.log");
    const auto request_options = parse_key_value_args(argc, argv, "--request-option");

    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_path;
    load_request.family_hint = "moss_tts_nano";
    auto model = registry.load(load_request);

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = parse_backend(backend_name);
    session_options.backend.device = device;
    session_options.backend.threads = threads;
    for (const std::string & option : arg_values(argc, argv, "--session-option")) {
        const auto equals = option.find('=');
        if (equals == std::string::npos || equals == 0) {
            throw std::runtime_error("invalid --session-option: " + option);
        }
        session_options.options[option.substr(0, equals)] = option.substr(equals + 1);
    }

    engine::runtime::TaskSpec task;
    task.task = engine::runtime::VoiceTaskKind::Tts;
    task.mode = engine::runtime::RunMode::Offline;
    auto session_base = model->create_task_session(task, session_options);
    auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
    if (session == nullptr) {
        throw std::runtime_error("MOSS-TTS-Nano warm bench expected an offline voice task session");
    }

    const auto clone_wav = engine::audio::read_wav_f32(clone_audio_path);
    engine::runtime::AudioBuffer clone_audio{
        clone_wav.sample_rate,
        clone_wav.channels,
        clone_wav.samples,
    };

    engine::runtime::SessionPreparationRequest prepare_request;
    prepare_request.text = engine::runtime::Transcript{warmup_text, ""};
    prepare_request.voice = engine::runtime::VoiceCondition{};
    prepare_request.voice->speaker = engine::runtime::VoiceReference{};
    prepare_request.voice->speaker->audio = clone_audio;
    session->prepare(prepare_request);

    std::vector<std::pair<std::string, std::vector<std::string>>> timing_sections;
    timing_sections.reserve(static_cast<size_t>(warmup + texts.size() * std::max(1, iterations)));

    for (int warmup_index = 0; warmup_index < warmup; ++warmup_index) {
        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{warmup_text, ""};
        request.voice = engine::runtime::VoiceCondition{};
        request.voice->speaker = engine::runtime::VoiceReference{};
        request.voice->speaker->audio = clone_audio;
        request.options = request_options;
        const auto started = std::chrono::steady_clock::now();
        auto result = session->run(request);
        const auto wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::vector<std::string> lines;
        append_request_evidence_lines(lines, warmup_text, wall_ms);
        timing_sections.push_back({"warmup" + std::to_string(warmup_index + 1), std::move(lines)});
        std::cout << "warmup_text[" << warmup_index << "]=" << warmup_text << "\n";
        std::cout << "warmup_summary_json[" << warmup_index << "]=" << summary_json(result, warmup_text) << "\n";
    }

    if (texts.empty()) {
        throw std::runtime_error("MOSS-TTS-Nano warm bench requires at least one --text");
    }

    std::vector<double> wall_sums(texts.size(), 0.0);
    std::vector<engine::runtime::TaskResult> last_results(texts.size());
    for (size_t request_index = 0; request_index < texts.size(); ++request_index) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            engine::runtime::TaskRequest request;
            request.text_input = engine::runtime::Transcript{texts[request_index], ""};
            request.voice = engine::runtime::VoiceCondition{};
            request.voice->speaker = engine::runtime::VoiceReference{};
            request.voice->speaker->audio = clone_audio;
            request.options = request_options;
            const auto started = std::chrono::steady_clock::now();
            last_results[request_index] = session->run(request);
            const double wall_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            wall_sums[request_index] += wall_ms;
            std::vector<std::string> lines;
            append_request_evidence_lines(lines, texts[request_index], wall_ms);
            timing_sections.push_back({
                "iteration" + std::to_string(iteration + 1) + ".request" + std::to_string(request_index + 1),
                std::move(lines),
            });
        }
    }

    write_sectioned_timing_log(timing_path, timing_sections);

    if (!audio_out_dir.empty()) {
        std::filesystem::create_directories(audio_out_dir);
    }
    for (size_t request_index = 0; request_index < texts.size(); ++request_index) {
        std::cout << "text[" << request_index << "]=" << texts[request_index] << "\n";
        std::cout << "summary_json[" << request_index << "]="
                  << summary_json(last_results[request_index], texts[request_index]) << "\n";
        if (last_results[request_index].audio_output.has_value() && !audio_out_dir.empty()) {
            const auto request_path = audio_out_dir / ("request_" + std::to_string(request_index) + ".wav");
            engine::audio::write_pcm16_wav(
                request_path,
                last_results[request_index].audio_output->sample_rate,
                last_results[request_index].audio_output->channels,
                last_results[request_index].audio_output->samples);
            std::cout << "audio_out[" << request_index << "]=" << request_path.string() << "\n";
        }
    }
    std::cout << "timing_out=" << timing_path.string() << "\n";
    if (last_results.back().audio_output.has_value()) {
        std::filesystem::create_directories(audio_out.parent_path());
        engine::audio::write_pcm16_wav(
            audio_out,
            last_results.back().audio_output->sample_rate,
            last_results.back().audio_output->channels,
            last_results.back().audio_output->samples);
        std::cout << "audio_out=" << audio_out.string() << "\n";
    }
    for (size_t request_index = 0; request_index < texts.size(); ++request_index) {
        std::cout << "average[" << request_index << "]\n";
        std::cout << "moss_tts_nano.request_wall_ms=" << (wall_sums[request_index] / static_cast<double>(iterations)) << "\n";
    }
    return 0;
} catch (const std::exception & exc) {
    std::cerr << "moss_tts_nano_warm_bench failed: " << exc.what() << "\n";
    return 1;
}
