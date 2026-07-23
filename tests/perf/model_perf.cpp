#include "app/cli/request.h"

#include "engine/framework/audio/output.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using JsonValue = engine::io::json::Value;
using JsonObject = engine::io::json::Value::Object;

struct PerfCaseSpec {
    std::string id;
    std::string family;
    std::string case_id;
    std::string request_id;
    std::optional<std::string> warmup_request_id;
};

struct MeasurementSummary {
    std::vector<double> wall_ms;
    double average_wall_ms = 0.0;
    double rtf = 0.0;
    double duration_sec = 0.0;
};

struct CaseSummary {
    std::string id;
    std::string family;
    std::string case_id;
    std::string request_id;
    std::string task;
    std::string mode;
    MeasurementSummary cold;
    MeasurementSummary warm;
};

struct RunResult {
    engine::runtime::TaskResult result;
    double wall_ms = 0.0;
};

std::string optional_arg(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::optional<std::string> find_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

int parse_int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    if (const auto value = find_arg(argc, argv, name)) {
        return std::stoi(*value);
    }
    return fallback;
}

std::string quote_json(const std::string & value) {
    return engine::io::json::stringify_string(value);
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string make_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return out.str();
}

engine::core::BackendType parse_backend(const std::string & value) {
    const std::string lowered = lower_ascii(value);
    if (lowered == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (lowered == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (lowered == "hip" || lowered == "rocm") {
        return engine::core::BackendType::Hip;
    }
    if (lowered == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (lowered == "metal") {
        return engine::core::BackendType::Metal;
    }
    if (lowered == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

std::string shorten_text(std::string text, size_t limit) {
    if (text.size() <= limit) {
        return text;
    }
    size_t cut = text.rfind(' ', limit);
    if (cut == std::string::npos || cut < (limit / 2)) {
        cut = limit;
    }
    text.resize(cut);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.pop_back();
    }
    return text;
}

std::filesystem::path write_clipped_wav(
    const std::filesystem::path & source_path,
    const std::filesystem::path & temp_dir,
    double seconds) {
    const auto audio = minitts::cli::read_audio_buffer(source_path);
    const int64_t frame_limit = static_cast<int64_t>(seconds * static_cast<double>(audio.sample_rate));
    const int64_t sample_limit = std::max<int64_t>(
        1,
        std::min<int64_t>(
            static_cast<int64_t>(audio.samples.size()),
            frame_limit * static_cast<int64_t>(audio.channels)));
    std::vector<float> clipped(
        audio.samples.begin(),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(sample_limit));
    const auto target = temp_dir / source_path.filename();
    engine::audio::WavPcm16Sink().write(
        target,
        engine::audio::AudioBuffer{
            audio.sample_rate,
            audio.channels,
            std::move(clipped),
        });
    return target;
}

JsonValue clip_audio_path_field(
    const JsonValue & value,
    const std::filesystem::path & repo_root,
    const std::filesystem::path & temp_dir,
    double seconds) {
    const auto absolute = std::filesystem::path(value.as_string()).is_absolute()
        ? std::filesystem::path(value.as_string())
        : repo_root / value.as_string();
    return JsonValue::make_string(write_clipped_wav(absolute, temp_dir, seconds).string());
}

JsonValue derive_warmup_request(
    const JsonValue & request,
    const std::filesystem::path & repo_root,
    const std::filesystem::path & temp_dir) {
    JsonObject out = request.as_object();
    const auto shorten_field = [&](const char * key, size_t limit) {
        const auto it = out.find(key);
        if (it != out.end() && it->second.is_string()) {
            it->second = JsonValue::make_string(shorten_text(it->second.as_string(), limit));
        }
    };
    shorten_field("text", 80);
    shorten_field("target_text", 80);
    shorten_field("style_ref_text", 80);
    shorten_field("reference_text", 80);
    shorten_field("lyrics", 80);
    shorten_field("instruct", 64);

    const auto clip_field = [&](const char * key) {
        const auto it = out.find(key);
        if (it != out.end() && it->second.is_string()) {
            it->second = clip_audio_path_field(it->second, repo_root, temp_dir, 2.0);
        }
    };
    clip_field("audio");
    clip_field("voice_ref");
    clip_field("source_audio");
    clip_field("target_voice");
    clip_field("prosody_ref");
    clip_field("style_ref");

    const auto it_duration = out.find("duration_seconds");
    if (it_duration != out.end() && it_duration->second.is_number()) {
        it_duration->second = JsonValue::make_number(std::min(4.0, it_duration->second.as_number()));
    }

    const auto it_max_tokens = out.find("max_tokens");
    if (it_max_tokens != out.end() && it_max_tokens->second.is_number()) {
        it_max_tokens->second = JsonValue::make_number(std::min<int64_t>(64, it_max_tokens->second.as_i64()));
    }

    if (const auto options_it = out.find("options"); options_it != out.end() && options_it->second.is_object()) {
        JsonObject options = options_it->second.as_object();
        const auto reduce_numeric_option = [&](const char * key, double cap) {
            const auto it = options.find(key);
            if (it != options.end() && it->second.is_number()) {
                it->second = JsonValue::make_number(std::min(cap, it->second.as_number()));
            }
        };
        reduce_numeric_option("duration_seconds", 4.0);
        reduce_numeric_option("target_duration_seconds", 4.0);
        options_it->second = JsonValue::make_object(std::move(options));
    }

    return JsonValue::make_object(std::move(out));
}

const JsonValue & require_case(const JsonValue & root, const std::string & case_id) {
    for (const auto & item : root.require("cases").as_array()) {
        if (engine::io::json::require_string(item, "id") == case_id) {
            return item;
        }
    }
    throw std::runtime_error("perf harness could not find audiocpp_cli case: " + case_id);
}

const JsonValue & require_request(const JsonValue & case_value, const std::string & request_id) {
    for (const auto & request : case_value.require("requests").as_array()) {
        if (engine::io::json::require_string(request, "id") == request_id) {
            return request;
        }
    }
    throw std::runtime_error(
        "perf harness could not find request " + request_id + " in case " +
        engine::io::json::require_string(case_value, "id"));
}

std::unordered_map<std::string, std::string> json_string_map(const JsonValue * value) {
    std::unordered_map<std::string, std::string> out;
    if (value == nullptr || value->is_null()) {
        return out;
    }
    for (const auto & [key, child] : value->as_object()) {
        out.emplace(key, minitts::cli::json_option_string(child));
    }
    return out;
}

PerfCaseSpec parse_perf_case(const JsonValue & value) {
    PerfCaseSpec out;
    out.id = engine::io::json::require_string(value, "id");
    out.family = engine::io::json::require_string(value, "family");
    out.case_id = engine::io::json::require_string(value, "case_id");
    out.request_id = engine::io::json::require_string(value, "request_id");
    if (const auto * warmup_request_id = value.find("warmup_request_id"); warmup_request_id != nullptr && warmup_request_id->is_string()) {
        out.warmup_request_id = warmup_request_id->as_string();
    }
    return out;
}

engine::runtime::ModelLoadRequest build_load_request(
    const JsonValue & case_value,
    const std::filesystem::path & repo_root) {
    engine::runtime::ModelLoadRequest request;
    request.model_path = repo_root / engine::io::json::require_string(case_value, "model");
    request.family_hint = engine::io::json::require_string(case_value, "family");
    request.options = json_string_map(case_value.find("load_options"));
    return request;
}

engine::runtime::SessionOptions build_session_options(
    const JsonValue & case_value,
    engine::core::BackendType backend_type,
    int device,
    int threads) {
    engine::runtime::SessionOptions options;
    options.backend.type = backend_type;
    options.backend.device = device;
    options.backend.threads = threads;
    options.options = json_string_map(case_value.find("session_options"));
    return options;
}

engine::runtime::TaskSpec build_task_spec(const JsonValue & case_value) {
    return engine::runtime::TaskSpec{
        engine::runtime::parse_voice_task_kind(engine::io::json::require_string(case_value, "task")),
        engine::runtime::parse_run_mode(engine::io::json::require_string(case_value, "mode")),
    };
}

double audio_duration_sec(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0.0;
    }
    return static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.sample_rate * audio.channels);
}

double result_duration_sec(
    const engine::runtime::TaskRequest & request,
    const engine::runtime::TaskResult & result) {
    if (result.audio_output.has_value()) {
        return audio_duration_sec(*result.audio_output);
    }
    if (!result.named_audio_outputs.empty()) {
        return audio_duration_sec(result.named_audio_outputs.front().audio);
    }
    if (request.audio_input.has_value()) {
        return audio_duration_sec(*request.audio_input);
    }
    return 0.0;
}

RunResult run_impl(
    engine::runtime::IVoiceTaskSession & session,
    const engine::runtime::TaskSpec & task_spec,
    const engine::runtime::TaskRequest & request,
    bool prepare_first) {
    const auto started = std::chrono::steady_clock::now();
    if (prepare_first) {
        session.prepare(engine::runtime::build_preparation_request(request));
    }
    if (task_spec.mode == engine::runtime::RunMode::Offline) {
        auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(&session);
        if (offline == nullptr) {
            throw std::runtime_error("selected model does not support offline mode");
        }
        auto result = offline->run(request);
        const auto ended = std::chrono::steady_clock::now();
        return {
            std::move(result),
            std::chrono::duration<double, std::milli>(ended - started).count(),
        };
    }

    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(&session);
    if (streaming == nullptr) {
        throw std::runtime_error("selected model does not support streaming mode");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("streaming perf request requires request.audio_input");
    }
    streaming->reset();
    const auto & audio = *request.audio_input;
    constexpr int kChunkSamples = 512;
    for (size_t offset = 0; offset < audio.samples.size(); offset += static_cast<size_t>(kChunkSamples)) {
        const size_t available = std::min(static_cast<size_t>(kChunkSamples), audio.samples.size() - offset);
        std::vector<float> chunk(static_cast<size_t>(kChunkSamples), 0.0f);
        std::copy(
            audio.samples.begin() + static_cast<std::ptrdiff_t>(offset),
            audio.samples.begin() + static_cast<std::ptrdiff_t>(offset + available),
            chunk.begin());
        streaming->process_audio_chunk(engine::runtime::AudioChunk{
            audio.sample_rate,
            audio.channels,
            static_cast<int64_t>(offset / static_cast<size_t>(std::max(1, audio.channels))),
            std::move(chunk),
        });
    }
    auto result = streaming->finalize();
    const auto ended = std::chrono::steady_clock::now();
    return {
        std::move(result),
        std::chrono::duration<double, std::milli>(ended - started).count(),
    };
}

RunResult run_once(
    engine::runtime::IVoiceTaskSession & session,
    const engine::runtime::TaskSpec & task_spec,
    const engine::runtime::TaskRequest & request) {
    return run_impl(session, task_spec, request, true);
}

RunResult run_prepared_once(
    engine::runtime::IVoiceTaskSession & session,
    const engine::runtime::TaskSpec & task_spec,
    const engine::runtime::TaskRequest & request) {
    return run_impl(session, task_spec, request, false);
}

MeasurementSummary measure_cold(
    const engine::runtime::ModelRegistry & registry,
    const engine::runtime::ModelLoadRequest & load_request,
    const engine::runtime::TaskSpec & task_spec,
    const engine::runtime::SessionOptions & session_options,
    const engine::runtime::TaskRequest & request) {
    MeasurementSummary summary;
    for (int rep = 0; rep < 3; ++rep) {
        const auto started = std::chrono::steady_clock::now();
        auto model = registry.load(load_request);
        auto session = model->create_task_session(task_spec, session_options);
        auto run = run_once(*session, task_spec, request);
        const auto ended = std::chrono::steady_clock::now();
        summary.wall_ms.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
        summary.duration_sec = result_duration_sec(request, run.result);
    }
    summary.average_wall_ms =
        std::accumulate(summary.wall_ms.begin(), summary.wall_ms.end(), 0.0) /
        static_cast<double>(summary.wall_ms.size());
    if (summary.duration_sec > 0.0) {
        summary.rtf = (summary.average_wall_ms / 1000.0) / summary.duration_sec;
    }
    return summary;
}

MeasurementSummary measure_warm(
    const engine::runtime::ModelRegistry & registry,
    const engine::runtime::ModelLoadRequest & load_request,
    const engine::runtime::TaskSpec & task_spec,
    const engine::runtime::SessionOptions & session_options,
    const engine::runtime::TaskRequest & warmup_request,
    const engine::runtime::TaskRequest & request) {
    auto model = registry.load(load_request);
    auto session = model->create_task_session(task_spec, session_options);
    (void) run_once(*session, task_spec, warmup_request);
    session->prepare(engine::runtime::build_preparation_request(request));

    MeasurementSummary summary;
    for (int rep = 0; rep < 3; ++rep) {
        auto run = run_prepared_once(*session, task_spec, request);
        summary.wall_ms.push_back(run.wall_ms);
        summary.duration_sec = result_duration_sec(request, run.result);
    }
    summary.average_wall_ms =
        std::accumulate(summary.wall_ms.begin(), summary.wall_ms.end(), 0.0) /
        static_cast<double>(summary.wall_ms.size());
    if (summary.duration_sec > 0.0) {
        summary.rtf = (summary.average_wall_ms / 1000.0) / summary.duration_sec;
    }
    return summary;
}

std::string measurement_json(const MeasurementSummary & measurement) {
    std::ostringstream out;
    out << "{"
        << "\"wall_ms\":[";
    for (size_t i = 0; i < measurement.wall_ms.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << measurement.wall_ms[i];
    }
    out << "],\"average_wall_ms\":" << measurement.average_wall_ms
        << ",\"duration_sec\":" << measurement.duration_sec
        << ",\"rtf\":" << measurement.rtf
        << "}";
    return out.str();
}

std::string case_summary_json(const CaseSummary & summary) {
    std::ostringstream out;
    out << "{"
        << "\"id\":" << quote_json(summary.id)
        << ",\"family\":" << quote_json(summary.family)
        << ",\"case_id\":" << quote_json(summary.case_id)
        << ",\"request_id\":" << quote_json(summary.request_id)
        << ",\"task\":" << quote_json(summary.task)
        << ",\"mode\":" << quote_json(summary.mode)
        << ",\"cold\":" << measurement_json(summary.cold)
        << ",\"warm\":" << measurement_json(summary.warm)
        << "}";
    return out.str();
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        const std::filesystem::path repo_root = std::filesystem::current_path();
        const std::filesystem::path perf_cases_path = optional_arg(
            argc,
            argv,
            "--perf-cases",
            "tests/perf/model_perf_cases.json");
        const std::filesystem::path path_cases_path = optional_arg(
            argc,
            argv,
            "--request-cases",
            "tests/perf/model_perf_request_cases.json");
        const std::string backend_name = optional_arg(argc, argv, "--backend", "cuda");
        const int device = parse_int_arg(argc, argv, "--device", 0);
        const int threads = parse_int_arg(argc, argv, "--threads", 1);
        if (threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        const auto selected_id = find_arg(argc, argv, "--id");
        const auto summary_out_arg = find_arg(argc, argv, "--summary-out");
        const auto output_root = summary_out_arg.has_value()
            ? std::filesystem::path(*summary_out_arg).parent_path()
            : repo_root / "build" / "logs" / "perf" /
                ("cpp_" + lower_ascii(backend_name) + "_" + make_timestamp());
        std::filesystem::create_directories(output_root);

        const auto perf_root = engine::io::json::parse_file(perf_cases_path);
        const auto path_root = engine::io::json::parse_file(path_cases_path);
        auto registry = engine::runtime::make_default_registry();
        const auto backend_type = parse_backend(backend_name);

        std::vector<CaseSummary> summaries;
        for (const auto & perf_case_value : perf_root.require("cases").as_array()) {
            const auto perf_case = parse_perf_case(perf_case_value);
            if (selected_id.has_value() && *selected_id != perf_case.id) {
                continue;
            }
            const auto & case_value = require_case(path_root, perf_case.case_id);
            const auto & request_value = require_request(case_value, perf_case.request_id);
            const auto load_request = build_load_request(case_value, repo_root);
            const auto task_spec = build_task_spec(case_value);
            const auto session_options = build_session_options(case_value, backend_type, device, threads);
            const auto request = minitts::cli::build_request_from_json(request_value, repo_root);

            const auto temp_dir = output_root / (perf_case.id + "_warmup");
            std::filesystem::create_directories(temp_dir);
            engine::runtime::TaskRequest warmup_request;
            if (perf_case.warmup_request_id.has_value()) {
                warmup_request = minitts::cli::build_request_from_json(
                    require_request(case_value, *perf_case.warmup_request_id),
                    repo_root);
            } else {
                warmup_request = minitts::cli::build_request_from_json(
                    derive_warmup_request(request_value, repo_root, temp_dir),
                    repo_root);
            }

            CaseSummary summary;
            summary.id = perf_case.id;
            summary.family = perf_case.family;
            summary.case_id = perf_case.case_id;
            summary.request_id = perf_case.request_id;
            summary.task = engine::runtime::to_string(task_spec.task);
            summary.mode = engine::runtime::to_string(task_spec.mode);
            summary.cold = measure_cold(registry, load_request, task_spec, session_options, request);
            summary.warm = measure_warm(registry, load_request, task_spec, session_options, warmup_request, request);
            summaries.push_back(summary);

            std::cout << perf_case.id
                      << " cold_avg_ms=" << summary.cold.average_wall_ms
                      << " cold_rtf=" << summary.cold.rtf
                      << " warm_avg_ms=" << summary.warm.average_wall_ms
                      << " warm_rtf=" << summary.warm.rtf
                      << std::endl;
        }

        std::ostringstream summary_json;
        summary_json << "{\"backend\":" << quote_json(backend_name)
                     << ",\"device\":" << device
                     << ",\"threads\":" << threads
                     << ",\"cases\":[";
        for (size_t i = 0; i < summaries.size(); ++i) {
            if (i != 0) {
                summary_json << ",";
            }
            summary_json << case_summary_json(summaries[i]);
        }
        summary_json << "]}";

        const auto summary_path = summary_out_arg.has_value()
            ? std::filesystem::path(*summary_out_arg)
            : output_root / "summary.json";
        std::ofstream(summary_path) << summary_json.str() << "\n";
        std::cout << "summary_json=" << summary_path.string() << std::endl;
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "model_perf failed: " << ex.what() << "\n";
        return 1;
    }
}
