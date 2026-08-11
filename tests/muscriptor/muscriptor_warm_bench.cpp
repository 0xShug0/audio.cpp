#include "../core/audio_task_warm_bench.h"

#include "engine/framework/io/json.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string sequence_value(const std::vector<std::string> & values, size_t index, const std::string & fallback) {
    return index < values.size() ? values[index] : fallback;
}

std::vector<std::string> split_csv_keep_empty(const std::string & value) {
    if (!value.empty() && value.front() == '[') {
        const auto parsed = engine::io::json::parse(value);
        std::vector<std::string> out;
        out.reserve(parsed.as_array().size());
        for (const auto & item : parsed.as_array()) {
            out.push_back(item.as_string());
        }
        return out;
    }
    std::vector<std::string> out;
    std::string item;
    for (const char ch : value) {
        if (ch == ',') {
            out.push_back(item);
            item.clear();
        } else {
            item.push_back(ch);
        }
    }
    if (!value.empty()) {
        out.push_back(item);
    }
    return out;
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    int request_index,
    const std::filesystem::path & audio_path,
    const std::string & instruments,
    const std::string & use_sampling,
    const std::string & temperature,
    const std::string & guidance_scale,
    const std::string & batch_size,
    const std::string & num_beams,
    const std::string & prelude_forcing,
    const std::string & seed,
    double wall_ms) {
    return engine::io::json::Value::make_object({
        {"request_index", engine::tools::number(request_index)},
        {"audio", engine::tools::string(audio_path.string())},
        {"instruments", engine::tools::string(instruments)},
        {"use_sampling", engine::tools::string(use_sampling)},
        {"temperature", engine::tools::string(temperature)},
        {"guidance_scale", engine::tools::string(guidance_scale)},
        {"batch_size", engine::tools::string(batch_size)},
        {"num_beams", engine::tools::string(num_beams)},
        {"prelude_forcing", engine::tools::string(prelude_forcing)},
        {"seed", engine::tools::string(seed)},
        {"text_output", engine::tools::string(result.text_output.has_value() ? result.text_output->text : "")},
        {"word_timestamps", engine::io::json::Value::make_array({})},
        {"metrics", engine::io::json::Value::make_object({{"wall_ms", engine::tools::number(wall_ms)}})},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = engine::tools::arg_value(argc, argv, "--model", "models/muscriptor-small");
        const std::string audio_sequence_value = engine::tools::arg_value(argc, argv, "--audio-sequence", "");
        const std::filesystem::path audio_path =
            engine::tools::arg_value(argc, argv, "--audio", "build/logs/muscriptor/input/headache_by_lost_deposit_1min_16k.wav");
        const std::filesystem::path warmup_audio_path =
            engine::tools::arg_value(argc, argv, "--warmup-audio", audio_path.string());
        const std::string backend_name = engine::tools::arg_value(argc, argv, "--backend", "cpu");
        const int device = engine::tools::int_arg(argc, argv, "--device", 0);
        const int threads = engine::tools::int_arg(argc, argv, "--threads", 8);
        const int warmup = engine::tools::int_arg(argc, argv, "--warmup", 1);
        const int iterations = engine::tools::int_arg(argc, argv, "--iterations", 1);
        const std::string instruments = engine::tools::arg_value(argc, argv, "--instruments", "");
        const std::string instruments_sequence = engine::tools::arg_value(argc, argv, "--instruments-sequence", "");
        const std::string use_sampling = engine::tools::arg_value(argc, argv, "--use-sampling", "false");
        const std::string use_sampling_sequence = engine::tools::arg_value(argc, argv, "--use-sampling-sequence", "");
        const std::string temperature = engine::tools::arg_value(argc, argv, "--temperature", "1.0");
        const std::string temperature_sequence = engine::tools::arg_value(argc, argv, "--temperature-sequence", "");
        const std::string guidance_scale = engine::tools::arg_value(argc, argv, "--cfg-coef", "1.0");
        const std::string guidance_scale_sequence = engine::tools::arg_value(argc, argv, "--cfg-coef-sequence", "");
        const std::string batch_size = engine::tools::arg_value(argc, argv, "--batch-size", "0");
        const std::string batch_size_sequence = engine::tools::arg_value(argc, argv, "--batch-size-sequence", "");
        const std::string num_beams = engine::tools::arg_value(argc, argv, "--beam-size", "1");
        const std::string num_beams_sequence = engine::tools::arg_value(argc, argv, "--beam-size-sequence", "");
        const std::string prelude_forcing = engine::tools::arg_value(argc, argv, "--prelude-forcing", "true");
        const std::string prelude_forcing_sequence = engine::tools::arg_value(argc, argv, "--prelude-forcing-sequence", "");
        const std::string seed = engine::tools::arg_value(argc, argv, "--seed", "1234");
        const std::string seed_sequence = engine::tools::arg_value(argc, argv, "--seed-sequence", "");
        const std::filesystem::path timing_path =
            engine::tools::arg_value(argc, argv, "--timing-file", "build/logs/warmbench/muscriptor_timing.log");

        if (!timing_path.parent_path().empty()) {
            std::filesystem::create_directories(timing_path.parent_path());
        }
        engine::tools::set_process_env("ENGINE_TIMING_ENABLED", "1");
        engine::tools::set_process_env("ENGINE_TIMING_FILE", timing_path.string());
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "muscriptor";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = engine::tools::parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & [key, value] : engine::tools::session_option_args(argc, argv)) {
            session_options.options[key] = value;
        }

        auto session_base = model->create_task_session(
            engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline},
            session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("muscriptor did not create an offline task session");
        }

        const auto warmup_audio = engine::tools::read_audio_buffer(warmup_audio_path);
        session->prepare(engine::runtime::build_preparation_request(warmup_audio));
        engine::runtime::TaskRequest warmup_request;
        warmup_request.audio_input = warmup_audio;
        warmup_request.options["instruments"] = instruments;
        warmup_request.options["use_sampling"] = use_sampling;
        warmup_request.options["temperature"] = temperature;
        warmup_request.options["guidance_scale"] = guidance_scale;
        if (batch_size != "0") {
            warmup_request.options["batch_size"] = batch_size;
        }
        warmup_request.options["num_beams"] = num_beams;
        warmup_request.options["prelude_forcing"] = prelude_forcing;
        warmup_request.options["seed"] = seed;
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
        const auto instrument_values = split_csv_keep_empty(instruments_sequence);
        const auto use_sampling_values = split_csv_keep_empty(use_sampling_sequence);
        const auto temperature_values = split_csv_keep_empty(temperature_sequence);
        const auto guidance_scale_values = split_csv_keep_empty(guidance_scale_sequence);
        const auto batch_size_values = split_csv_keep_empty(batch_size_sequence);
        const auto num_beams_values = split_csv_keep_empty(num_beams_sequence);
        const auto prelude_forcing_values = split_csv_keep_empty(prelude_forcing_sequence);
        const auto seed_values = split_csv_keep_empty(seed_sequence);

        engine::io::json::Value::Array steps;
        steps.reserve(request_paths.size());
        for (size_t request_index = 0; request_index < request_paths.size(); ++request_index) {
            const auto request_audio = engine::tools::read_audio_buffer(request_paths[request_index]);
            const auto request_instruments = sequence_value(instrument_values, request_index, instruments);
            const auto request_use_sampling = sequence_value(use_sampling_values, request_index, use_sampling);
            const auto request_temperature = sequence_value(temperature_values, request_index, temperature);
            const auto request_guidance_scale = sequence_value(guidance_scale_values, request_index, guidance_scale);
            const auto request_batch_size = sequence_value(batch_size_values, request_index, batch_size);
            const auto request_num_beams = sequence_value(num_beams_values, request_index, num_beams);
            const auto request_prelude_forcing = sequence_value(prelude_forcing_values, request_index, prelude_forcing);
            const auto request_seed = sequence_value(seed_values, request_index, seed);
            engine::runtime::TaskRequest request;
            request.audio_input = request_audio;
            request.options["instruments"] = request_instruments;
            request.options["use_sampling"] = request_use_sampling;
            request.options["temperature"] = request_temperature;
            request.options["guidance_scale"] = request_guidance_scale;
            if (request_batch_size != "0") {
                request.options["batch_size"] = request_batch_size;
            }
            request.options["num_beams"] = request_num_beams;
            request.options["prelude_forcing"] = request_prelude_forcing;
            request.options["seed"] = request_seed;
            engine::runtime::TaskResult last_result;
            double total_ms = 0.0;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                last_result = session->run(request);
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            const double wall_ms = total_ms / static_cast<double>(iterations);
            std::cout << "average[" << request_index << "]\n";
            std::cout << "muscriptor.wall_ms=" << wall_ms << "\n";
            steps.push_back(step_json(
                last_result,
                static_cast<int>(request_index),
                request_paths[request_index],
                request_instruments,
                request_use_sampling,
                request_temperature,
                request_guidance_scale,
                request_batch_size,
                request_num_beams,
                request_prelude_forcing,
                request_seed,
                wall_ms));
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", engine::tools::string("muscriptor")},
            {"backend", engine::tools::string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "muscriptor_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
