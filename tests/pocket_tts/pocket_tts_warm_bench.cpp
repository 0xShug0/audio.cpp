#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/pocket_tts/assets.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char * kDefaultWarmupText =
    "At sunrise the studio monitors clicked on, and the first calibration phrase rolled across the room with steady timing.";
constexpr const char * kCaseCatalogPath = "tests/pocket_tts/pocket_tts_warm_bench_cases.txt";
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
        throw std::runtime_error("failed to open Pocket TTS warm bench case catalog: " + path.string());
    }
    std::unordered_map<std::string, std::vector<std::string>> cases;
    std::string current_case;
    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = trim_ascii(line);
        if (trimmed.empty() || trimmed.front() == '#') {
            continue;
        }
        if (trimmed.size() >= 2 && trimmed.front() == '[' && trimmed.back() == ']') {
            current_case = trimmed.substr(1, trimmed.size() - 2);
            cases[current_case];
            continue;
        }
        if (current_case.empty()) {
            throw std::runtime_error("Pocket TTS warm bench case catalog entry is missing a [case] header");
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

std::string json_escape(const std::string & value);

std::string timing_line_scalar(const std::string & timestamp, const std::string & key, const std::string & value) {
    return "[TIMING ts=" + timestamp + "] " + key + " " + value;
}

std::unordered_map<std::string, std::string> parse_timing_entries(const std::vector<std::string> & lines) {
    std::unordered_map<std::string, std::string> entries;
    for (const std::string & line : lines) {
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
        entries[key] = line.substr(first_space + 1);
    }
    return entries;
}

void append_parity_evidence_lines(
    std::vector<std::string> & lines,
    const std::string & mode,
    const std::optional<uint32_t> & noise_seed,
    const std::optional<int> & noise_steps,
    const std::optional<int64_t> & noise_latent_dim,
    const std::optional<std::string> & noise_hash,
    const std::optional<std::filesystem::path> & noise_path) {
    const std::string ts = timestamp_seconds_local();
    lines.push_back(timing_line_scalar(
        ts,
        "pocket_tts.test_noise_mode",
        mode == "parity" ? "\"shared_schedule_file_default\"" : "\"api_default_random\""));
    if (mode != "parity") {
        return;
    }
    lines.push_back(timing_line_scalar(ts, "pocket_tts.test_noise_seed", std::to_string(*noise_seed)));
    lines.push_back(timing_line_scalar(ts, "pocket_tts.test_noise_steps", std::to_string(*noise_steps)));
    lines.push_back(timing_line_scalar(ts, "pocket_tts.test_noise_latent_dim", std::to_string(*noise_latent_dim)));
    lines.push_back(timing_line_scalar(ts, "pocket_tts.test_noise_hash", "\"" + *noise_hash + "\""));
    lines.push_back(timing_line_scalar(ts, "pocket_tts.test_noise_file", "\"" + noise_path->string() + "\""));
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

std::vector<float> load_noise_schedule_file(const std::filesystem::path & path, int64_t latent_dim) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open Pocket TTS noise schedule: " + path.string());
    }
    const size_t value_count = static_cast<size_t>(input.tellg()) / sizeof(float);
    if (value_count == 0 || value_count % static_cast<size_t>(latent_dim) != 0) {
        throw std::runtime_error("invalid Pocket TTS noise schedule shape: " + path.string());
    }
    std::vector<float> values(value_count);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(value_count * sizeof(float)));
    return values;
}

std::string fnv1a64_hex(const std::vector<float> & values) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(values.data());
    const size_t byte_count = values.size() * sizeof(float);
    uint64_t hash_value = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < byte_count; ++i) {
        hash_value ^= bytes[i];
        hash_value *= 0x100000001B3ULL;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash_value;
    return out.str();
}

std::unordered_map<std::string, double> parse_timing_metrics(const std::vector<std::string> & lines) {
    std::unordered_map<std::string, double> metrics;
    for (const auto & [key, value] : parse_timing_entries(lines)) {
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
        if (starts_with(line, "[TIMING")) {
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

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
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
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::string summary_json(
    const engine::runtime::TaskResult & result,
    const std::unordered_map<std::string, std::string> & evidence,
    const std::string & mode,
    const std::optional<uint32_t> & noise_seed,
    const std::optional<int> & noise_steps,
    const std::optional<int64_t> & noise_latent_dim,
    const std::optional<std::string> & noise_hash,
    const std::optional<std::filesystem::path> & noise_path) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("Pocket TTS warm bench expected audio_output");
    }
    const auto & audio = *result.audio_output;
    auto require_evidence = [&](const std::string & key) -> const std::string & {
        const auto it = evidence.find(key);
        if (it == evidence.end()) {
            throw std::runtime_error("missing Pocket TTS timing evidence: " + key);
        }
        return it->second;
    };

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
    out << "],";
    out << "\"request_char_count\":" << require_evidence("pocket_tts.request_char_count") << ",";
    out << "\"generated_steps\":" << require_evidence("pocket_tts.generated_steps") << ",";
    if (mode == "parity") {
        out << "\"test_noise_mode\":\"shared_schedule_file_default\",";
        out << "\"test_noise_seed\":" << *noise_seed << ",";
        out << "\"test_noise_steps\":" << *noise_steps << ",";
        out << "\"test_noise_latent_dim\":" << *noise_latent_dim << ",";
        out << "\"test_noise_hash\":\"" << *noise_hash << "\",";
        out << "\"test_noise_file\":\"" << json_escape(noise_path->string()) << "\"";
    } else {
        out << "\"test_noise_mode\":\"api_default_random\"";
    }
    out << "}";
    return out.str();
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/pocket-tts");
        std::vector<std::string> texts = arg_values(argc, argv, "--text");
        const std::vector<std::string> case_names = arg_values(argc, argv, "--case-name");
        if (!case_names.empty()) {
            const auto case_catalog = load_case_catalog(kCaseCatalogPath);
            for (const std::string & case_name : case_names) {
                const auto it = case_catalog.find(case_name);
                if (it == case_catalog.end()) {
                    throw std::runtime_error("unknown Pocket TTS warm bench case: " + case_name);
                }
                texts.insert(texts.end(), it->second.begin(), it->second.end());
            }
        }
        if (texts.empty()) {
            texts.push_back("We changed the benchmark request to a longer sentence so the Pocket TTS session exercises a larger prompt path during parity runs.");
        }
        const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
        const std::string mode = arg_value(argc, argv, "--mode", "parity");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 1);
        const int iterations = int_arg(argc, argv, "--iterations", 5);
        const std::string language = arg_value(argc, argv, "--language", "english");
        const std::string voice_id = arg_value(argc, argv, "--voice-id", "");
        const std::string warmup_text = arg_value(argc, argv, "--warmup-text", kDefaultWarmupText);
        const std::string voice_embedding_path = arg_value(argc, argv, "--voice-embedding-path", "");
        const std::string clone_audio_path = arg_value(argc, argv, "--clone-audio", "");
        const int max_steps = int_arg(argc, argv, "--max-steps", 0);
        const int frames_after_eos = int_arg(argc, argv, "--frames-after-eos", -1);
        const float temperature = float_arg(argc, argv, "--temperature", 0.7f);
        const float noise_clamp = float_arg(argc, argv, "--noise-clamp", -1.0f);
        const float eos_threshold = float_arg(argc, argv, "--eos-threshold", -4.0f);
        const int rng_seed = int_arg(argc, argv, "--seed", 1234);
        const std::string artifact_stamp = arg_value(argc, argv, "--artifact-stamp", timestamp_seconds_local());
        const std::string noise_file_arg = arg_value(argc, argv, "--noise-file", "");
        const auto session_option_overrides = parse_key_value_args(argc, argv, "--session-option");
        if (mode != "parity" && mode != "performance") {
            throw std::runtime_error("unsupported mode: " + mode);
        }
        if (mode == "parity" && noise_file_arg.empty()) {
            throw std::runtime_error(
                "parity mode requires --noise-file; run this through tests/warmbench.py for normal parity checks.");
        }
        const std::optional<std::filesystem::path> noise_path = noise_file_arg.empty()
            ? std::nullopt
            : std::make_optional(std::filesystem::absolute(std::filesystem::path(noise_file_arg)));
        const std::filesystem::path audio_out_path = arg_value(
            argc,
            argv,
            "--audio-out",
            "pocket_tts_cpp_" + backend_name + "_audio.wav");
        const std::string audio_out_dir_arg = arg_value(argc, argv, "--audio-out-dir", "");
        const std::filesystem::path audio_out_dir =
            audio_out_dir_arg.empty() ? std::filesystem::path{} : std::filesystem::path(audio_out_dir_arg);
        const std::string timing_file_arg = arg_value(argc, argv, "--timing-file", "");
        const std::filesystem::path timing_path = timing_file_arg.empty()
            ? (std::filesystem::path("build/logs/parity/pocket_tts") /
               ("pocket_tts_cpp_" + backend_name + "-" + artifact_stamp + ".log"))
            : std::filesystem::path(timing_file_arg);

        if (!timing_path.parent_path().empty()) {
            std::filesystem::create_directories(timing_path.parent_path());
        }

        set_process_env("ENGINE_TRACE_ENABLED", "0");
        set_process_env("ENGINE_TIMING_ENABLED", "1");
        set_process_env("ENGINE_TIMING_FILE", timing_path.string());

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.options["pocket_tts.language"] = language;
        auto model = registry.load(load_request);

        const auto manifest = engine::models::pocket_tts::load_pocket_tts_assets(
            model_path,
            language);
        std::optional<int64_t> noise_latent_dim = std::nullopt;
        std::optional<int> noise_steps = std::nullopt;
        std::optional<std::string> noise_hash = std::nullopt;
        if (mode == "parity") {
            noise_latent_dim = manifest->model_config.latent_dim;
            if (!std::filesystem::exists(*noise_path)) {
                throw std::runtime_error("Pocket TTS noise file does not exist: " + noise_path->string());
            }
            const std::vector<float> noise_schedule = load_noise_schedule_file(*noise_path, *noise_latent_dim);
            noise_steps = static_cast<int>(noise_schedule.size() / static_cast<size_t>(*noise_latent_dim));
            noise_hash = fnv1a64_hex(noise_schedule);
        }

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
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
            throw std::runtime_error("loaded Pocket TTS session is not an offline TTS session");
        }

        const bool use_prepared_voice =
            !clone_audio_path.empty() || !voice_id.empty() || !voice_embedding_path.empty();
        if (!use_prepared_voice) {
            throw std::runtime_error("Pocket TTS warm bench requires --voice-id, --clone-audio, or --voice-embedding-path");
        }

        auto make_request = [&](const std::string & text, bool include_voice, bool include_session_options) {
            engine::runtime::TaskRequest request;
            request.text_input = engine::runtime::Transcript{text, ""};
            if (include_voice) {
                request.voice = engine::runtime::VoiceCondition{};
                request.voice->speaker = engine::runtime::VoiceReference{};
                if (!clone_audio_path.empty()) {
                    const auto audio = engine::audio::read_wav_f32(clone_audio_path);
                    request.voice->speaker->audio = engine::runtime::AudioBuffer{
                        audio.sample_rate,
                        audio.channels,
                        audio.samples,
                    };
                } else if (!voice_id.empty()) {
                    request.voice->speaker->cached_voice_id = voice_id;
                }
            }
            if (include_session_options && !voice_embedding_path.empty()) {
                request.options["pocket_tts.voice_embedding_path"] = voice_embedding_path;
            }
            if (include_session_options && max_steps > 0) {
                request.options["pocket_tts.max_steps"] = std::to_string(max_steps);
            }
            if (frames_after_eos >= 0) {
                request.options["pocket_tts.frames_after_eos"] = std::to_string(frames_after_eos);
            }
            if (include_session_options) {
                if (mode == "parity") {
                    request.options["pocket_tts.temperature"] = std::to_string(temperature);
                    request.options["pocket_tts.noise_clamp"] = std::to_string(noise_clamp);
                    request.options["pocket_tts.eos_threshold"] = std::to_string(eos_threshold);
                    request.options["pocket_tts.seed"] = std::to_string(rng_seed);
                    request.options["pocket_tts.noise_file"] = noise_path->string();
                }
            }
            return request;
        };

        std::vector<engine::runtime::TaskRequest> requests;
        requests.reserve(texts.size());
        for (const std::string & text : texts) {
            requests.push_back(make_request(text, false, false));
        }
        const auto warmup_request = make_request(warmup_text, use_prepared_voice, true);

        std::vector<std::pair<std::string, std::vector<std::string>>> log_sections;
        clear_file(timing_path);
        session_base->prepare(engine::runtime::build_preparation_request(warmup_request));
        std::vector<std::string> prepare_lines = read_timing_lines(timing_path);
        append_parity_evidence_lines(
            prepare_lines,
            mode,
            static_cast<uint32_t>(rng_seed),
            noise_steps,
            noise_latent_dim,
            noise_hash,
            noise_path);
        log_sections.insert(log_sections.begin(), {"prepare", std::move(prepare_lines)});

        const int warmup_passes = std::max(0, warmup);
        std::vector<std::map<std::string, double>> sums(requests.size());
        std::vector<engine::runtime::TaskResult> warmup_results;
        warmup_results.reserve(static_cast<size_t>(warmup_passes));
        std::vector<engine::runtime::TaskResult> last_results(requests.size());
        std::vector<std::unordered_map<std::string, std::string>> warmup_evidence;
        warmup_evidence.reserve(static_cast<size_t>(warmup_passes));
        std::vector<std::unordered_map<std::string, std::string>> last_evidence(requests.size());
        for (int i = 0; i < warmup_passes; ++i) {
            clear_file(timing_path);
            warmup_results.push_back(session->run(warmup_request));
            std::vector<std::string> lines = read_timing_lines(timing_path);
            append_parity_evidence_lines(
                lines,
                mode,
                static_cast<uint32_t>(rng_seed),
                noise_steps,
                noise_latent_dim,
                noise_hash,
                noise_path);
            warmup_evidence.push_back(parse_timing_entries(lines));
            log_sections.push_back({"warmup" + std::to_string(i + 1), std::move(lines)});
        }

        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            for (int i = 0; i < iterations; ++i) {
                clear_file(timing_path);
                last_results[request_index] = session->run(requests[request_index]);
                auto lines = read_timing_lines(timing_path);
                append_parity_evidence_lines(
                    lines,
                    mode,
                    static_cast<uint32_t>(rng_seed),
                    noise_steps,
                    noise_latent_dim,
                    noise_hash,
                    noise_path);
                const auto metrics = parse_timing_metrics(lines);
                log_sections.push_back({
                    "iteration" + std::to_string(i + 1) + ".request" + std::to_string(request_index + 1),
                    std::move(lines),
                });
                last_evidence[request_index] = parse_timing_entries(log_sections.back().second);
                for (const auto & [key, value] : metrics) {
                    sums[request_index][key] += value;
                }
            }
        }

        write_sectioned_timing_log(timing_path, log_sections);

        for (size_t warmup_index = 0; warmup_index < warmup_results.size(); ++warmup_index) {
            std::cout << "warmup_text[" << warmup_index << "]=" << warmup_text << "\n";
            std::cout << "warmup_summary_json[" << warmup_index << "]="
                      << summary_json(
                             warmup_results[warmup_index],
                             warmup_evidence[warmup_index],
                             mode,
                             static_cast<uint32_t>(rng_seed),
                             noise_steps,
                             noise_latent_dim,
                             noise_hash,
                             noise_path)
                      << "\n";
        }
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            std::cout << "text[" << request_index << "]=" << texts[request_index] << "\n";
            std::cout << "summary_json[" << request_index << "]="
                      << summary_json(
                             last_results[request_index],
                             last_evidence[request_index],
                             mode,
                             static_cast<uint32_t>(rng_seed),
                             noise_steps,
                             noise_latent_dim,
                             noise_hash,
                             noise_path)
                      << "\n";
            if (requests.size() == 1 && request_index == 0) {
                std::cout << "text=" << texts[request_index] << "\n";
                std::cout << "summary_json="
                          << summary_json(
                                 last_results[request_index],
                                 last_evidence[request_index],
                                 mode,
                                 static_cast<uint32_t>(rng_seed),
                                 noise_steps,
                                 noise_latent_dim,
                                 noise_hash,
                                 noise_path)
                          << "\n";
            }
        }
        std::cout << "timing_out=" << timing_path.string() << "\n";
        std::cout << "noise_file=" << (noise_path.has_value() ? noise_path->string() : "") << "\n";
        if (!audio_out_dir.empty()) {
            std::filesystem::create_directories(audio_out_dir);
            for (size_t request_index = 0; request_index < last_results.size(); ++request_index) {
                const auto & request_result = last_results[request_index];
                if (!request_result.audio_output.has_value()) {
                    throw std::runtime_error("Pocket TTS warm bench expected audio_output for per-request WAV write");
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
            throw std::runtime_error("Pocket TTS warm bench expected audio_output for WAV write");
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
        std::cerr << "pocket_tts_warm_bench failed: " << e.what() << "\n";
        return 1;
    }
}
