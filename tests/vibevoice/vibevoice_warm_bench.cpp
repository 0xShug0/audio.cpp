#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/models/vibevoice/assets.h"
#include "engine/models/vibevoice/connector.h"
#include "engine/models/vibevoice/decoder.h"
#include "engine/models/vibevoice/diffusion_head.h"
#include "engine/models/vibevoice/generator.h"
#include "engine/models/vibevoice/session.h"
#include "engine/models/vibevoice/tokenizer_audio.h"
#include "engine/models/vibevoice/tokenizer_text.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

bool has_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    throw std::runtime_error("VibeVoice warmbench unsupported backend: " + value);
}

void set_env_required(const char * key, const std::string & value) {
    if (setenv(key, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
}

std::filesystem::path resolve_path(const std::string & value) {
    auto path = std::filesystem::path(value);
    return path.is_absolute() ? path : std::filesystem::weakly_canonical(std::filesystem::current_path() / path);
}

std::string optional_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::string{} : value->as_string();
}

std::string required_string(const engine::io::json::Value & object, const std::string & key) {
    auto value = optional_string(object, key);
    if (value.empty()) {
        throw std::runtime_error("VibeVoice warmbench request missing required string field: " + key);
    }
    return value;
}

std::vector<std::string> required_string_array(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || !value->is_array()) {
        throw std::runtime_error("VibeVoice warmbench request missing required array field: " + key);
    }
    std::vector<std::string> out;
    for (const auto & item : value->as_array()) {
        out.push_back(item.as_string());
    }
    if (out.empty()) {
        throw std::runtime_error("VibeVoice warmbench request voice_samples must not be empty");
    }
    return out;
}

engine::runtime::AudioBuffer read_audio_buffer(const std::string & path_text) {
    const auto wav = engine::audio::read_wav_f32(resolve_path(path_text));
    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
        throw std::runtime_error("VibeVoice warmbench received empty voice sample: " + path_text);
    }
    return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

engine::models::vibevoice::VibeVoiceRequest make_request(
    const engine::io::json::Value & object,
    const engine::models::vibevoice::VibeVoiceAssets & assets,
    const std::string & prompt_noise_file,
    const std::string & noise_file) {
    engine::models::vibevoice::VibeVoiceRequest request;
    request.text = required_string(object, "text");
    request.generation.ddpm_inference_steps = assets.config.diffusion_head.ddpm_num_inference_steps;
    request.generation.max_new_tokens =
        engine::io::json::optional_i64(object, "max_new_tokens", request.generation.max_new_tokens);
    request.generation.max_length_times =
        engine::io::json::optional_f32(object, "max_length_times", request.generation.max_length_times);
    request.generation.ddpm_inference_steps =
        engine::io::json::optional_i64(object, "ddpm_steps", request.generation.ddpm_inference_steps);
    request.generation.cfg_scale =
        engine::io::json::optional_f32(object, "cfg_scale", request.generation.cfg_scale);
    request.generation.do_sample =
        engine::io::json::optional_bool(object, "do_sample", request.generation.do_sample);
    request.generation.temperature =
        engine::io::json::optional_f32(object, "temperature", request.generation.temperature);
    request.generation.top_k =
        engine::io::json::optional_i64(object, "top_k", request.generation.top_k);
    request.generation.top_p =
        engine::io::json::optional_f32(object, "top_p", request.generation.top_p);
    request.generation.seed =
        static_cast<uint32_t>(engine::io::json::optional_i64(object, "seed", request.generation.seed));
    request.generation.prompt_noise_file = prompt_noise_file.empty()
        ? optional_string(object, "prompt_noise_file")
        : prompt_noise_file;
    request.generation.diffusion_noise_file = noise_file.empty()
        ? optional_string(object, "diffusion_noise_file")
        : noise_file;
    if (request.generation.max_new_tokens < 0) {
        throw std::runtime_error("VibeVoice warmbench max_new_tokens must be non-negative");
    }
    if (request.generation.max_length_times <= 0.0F) {
        throw std::runtime_error("VibeVoice warmbench max_length_times must be positive");
    }
    if (request.generation.ddpm_inference_steps <= 0) {
        throw std::runtime_error("VibeVoice warmbench ddpm_steps must be positive");
    }
    if (request.generation.cfg_scale < 0.0F) {
        throw std::runtime_error("VibeVoice warmbench cfg_scale must be non-negative");
    }
    if (request.generation.temperature <= 0.0F) {
        throw std::runtime_error("VibeVoice warmbench temperature must be positive");
    }
    if (request.generation.top_k < 0) {
        throw std::runtime_error("VibeVoice warmbench top_k must be non-negative");
    }
    if (request.generation.top_p <= 0.0F || request.generation.top_p > 1.0F) {
        throw std::runtime_error("VibeVoice warmbench top_p must be in (0, 1]");
    }
    for (const auto & path : required_string_array(object, "voice_samples")) {
        request.speakers.push_back(engine::models::vibevoice::VibeVoiceSpeakerPrompt{read_audio_buffer(path)});
    }
    return request;
}

void set_optional_i64_option(
    engine::runtime::TaskRequest & request,
    const engine::io::json::Value & object,
    const std::string & json_key,
    const std::string & option_key) {
    if (object.find(json_key) != nullptr) {
        request.options[option_key] = std::to_string(engine::io::json::optional_i64(object, json_key, 0));
    }
}

void set_optional_f32_option(
    engine::runtime::TaskRequest & request,
    const engine::io::json::Value & object,
    const std::string & json_key,
    const std::string & option_key) {
    if (object.find(json_key) != nullptr) {
        request.options[option_key] = std::to_string(engine::io::json::optional_f32(object, json_key, 0.0F));
    }
}

void set_optional_bool_option(
    engine::runtime::TaskRequest & request,
    const engine::io::json::Value & object,
    const std::string & json_key,
    const std::string & option_key) {
    if (object.find(json_key) != nullptr) {
        request.options[option_key] = engine::io::json::optional_bool(object, json_key, false) ? "true" : "false";
    }
}

std::string join_voice_samples(const std::vector<std::string> & paths) {
    std::string joined;
    for (const auto & path : paths) {
        if (!joined.empty()) {
            joined += ",";
        }
        joined += path;
    }
    return joined;
}

engine::runtime::TaskRequest make_task_request(
    const engine::io::json::Value & object,
    const std::string & prompt_noise_file,
    const std::string & noise_file) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{required_string(object, "text"), ""};
    request.options["voice_samples"] = join_voice_samples(required_string_array(object, "voice_samples"));
    set_optional_i64_option(request, object, "max_new_tokens", "max_new_tokens");
    set_optional_f32_option(request, object, "max_length_times", "max_length_times");
    set_optional_i64_option(request, object, "ddpm_steps", "ddpm_steps");
    set_optional_f32_option(request, object, "cfg_scale", "cfg_scale");
    set_optional_bool_option(request, object, "do_sample", "do_sample");
    set_optional_f32_option(request, object, "temperature", "temperature");
    set_optional_i64_option(request, object, "top_k", "top_k");
    set_optional_f32_option(request, object, "top_p", "top_p");
    set_optional_i64_option(request, object, "seed", "seed");
    request.options["prompt_noise_file"] = prompt_noise_file.empty()
        ? optional_string(object, "prompt_noise_file")
        : prompt_noise_file;
    request.options["diffusion_noise_file"] = noise_file.empty()
        ? optional_string(object, "diffusion_noise_file")
        : noise_file;
    return request;
}

std::vector<engine::models::vibevoice::VibeVoiceRequest> parse_requests(
    const std::string & request_sequence_json,
    const engine::models::vibevoice::VibeVoiceAssets & assets,
    const std::string & prompt_noise_file,
    const std::string & noise_file) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("VibeVoice warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::models::vibevoice::VibeVoiceRequest> out;
    for (const auto & item : root.as_array()) {
        out.push_back(make_request(item, assets, prompt_noise_file, noise_file));
    }
    if (out.empty()) {
        throw std::runtime_error("VibeVoice warmbench request sequence is empty");
    }
    return out;
}

std::vector<engine::runtime::TaskRequest> parse_task_requests(
    const std::string & request_sequence_json,
    const std::string & prompt_noise_file,
    const std::string & noise_file) {
    if (request_sequence_json.empty()) {
        throw std::runtime_error("VibeVoice warmbench requires --request-sequence-json");
    }
    const auto root = engine::io::json::parse(request_sequence_json);
    std::vector<engine::runtime::TaskRequest> out;
    for (const auto & item : root.as_array()) {
        out.push_back(make_task_request(item, prompt_noise_file, noise_file));
    }
    if (out.empty()) {
        throw std::runtime_error("VibeVoice warmbench request sequence is empty");
    }
    return out;
}

engine::io::json::Value number(double value) {
    return engine::io::json::Value::make_number(value);
}

engine::io::json::Value string(std::string value) {
    return engine::io::json::Value::make_string(std::move(value));
}

engine::io::json::Value audio_summary_json(const engine::runtime::AudioBuffer & audio) {
    if (audio.samples.empty()) {
        throw std::runtime_error("VibeVoice warmbench received empty audio output");
    }
    double sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    float min_value = audio.samples.front();
    float max_value = audio.samples.front();
    for (const float sample : audio.samples) {
        sum += static_cast<double>(sample);
        abs_sum += std::abs(static_cast<double>(sample));
        sq_sum += static_cast<double>(sample) * static_cast<double>(sample);
        min_value = std::min(min_value, sample);
        max_value = std::max(max_value, sample);
    }
    const double count = static_cast<double>(audio.samples.size());
    return engine::io::json::Value::make_object({
        {"sample_rate", number(static_cast<double>(audio.sample_rate))},
        {"channels", number(static_cast<double>(audio.channels))},
        {"samples", number(static_cast<double>(audio.samples.size()))},
        {"duration_sec", number(audio.sample_rate > 0 ? static_cast<double>(audio.samples.size()) / audio.sample_rate : 0.0)},
        {"sum", number(sum)},
        {"mean_abs", number(abs_sum / count)},
        {"rms", number(std::sqrt(sq_sum / count))},
        {"min", number(min_value)},
        {"max", number(max_value)},
    });
}

engine::io::json::Value token_window_json(const std::vector<int32_t> & tokens) {
    engine::io::json::Value::Array head;
    const size_t head_count = std::min<size_t>(tokens.size(), 128);
    head.reserve(head_count);
    for (size_t i = 0; i < head_count; ++i) {
        head.push_back(number(static_cast<double>(tokens[i])));
    }
    engine::io::json::Value::Array tail;
    const size_t tail_count = std::min<size_t>(tokens.size(), 32);
    tail.reserve(tail_count);
    const size_t tail_start = tokens.size() - tail_count;
    for (size_t i = tail_start; i < tokens.size(); ++i) {
        tail.push_back(number(static_cast<double>(tokens[i])));
    }
    return engine::io::json::Value::make_object({
        {"head", engine::io::json::Value::make_array(std::move(head))},
        {"tail", engine::io::json::Value::make_array(std::move(tail))},
        {"count", number(static_cast<double>(tokens.size()))},
    });
}

engine::io::json::Value step_json(
    const engine::models::vibevoice::VibeVoiceResult & result,
    int request_index,
    double wall_ms,
    const std::filesystem::path & audio_path) {
    engine::io::json::Value::Object stem{
        {"name", string("audio")},
        {"summary", audio_summary_json(result.audio)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", string(audio_path.string()));
    }
    return engine::io::json::Value::make_object({
        {"request_index", number(static_cast<double>(request_index))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", number(wall_ms)},
            {"generated_tokens", number(static_cast<double>(result.generated_tokens.size()))},
            {"generated_token_ids", token_window_json(result.generated_tokens)},
        })},
    });
}

engine::io::json::Value step_json(
    const engine::runtime::TaskResult & result,
    int request_index,
    double wall_ms,
    const std::filesystem::path & audio_path) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("VibeVoice warmbench expected audio output");
    }
    engine::io::json::Value::Object stem{
        {"name", string("audio")},
        {"summary", audio_summary_json(*result.audio_output)},
    };
    if (!audio_path.empty()) {
        stem.emplace("audio", string(audio_path.string()));
    }
    return engine::io::json::Value::make_object({
        {"request_index", number(static_cast<double>(request_index))},
        {"stems", engine::io::json::Value::make_array({engine::io::json::Value::make_object(std::move(stem))})},
        {"metrics", engine::io::json::Value::make_object({
            {"wall_ms", number(wall_ms)},
        })},
    });
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/VibeVoice-1.5B");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int warmup = int_arg(argc, argv, "--warmup", 0);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        const std::string request_sequence_json = arg_value(argc, argv, "--request-sequence-json", "");
        const std::string prompt_noise_file = arg_value(argc, argv, "--prompt-noise-file", "");
        const std::string noise_file = arg_value(argc, argv, "--noise-file", "");
        const std::string log_file = arg_value(argc, argv, "--log-file", "");
        const bool batch = has_arg(argc, argv, "--batch");
        const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
        const std::filesystem::path timing_path =
            arg_value(argc, argv, "--timing-file", "/tmp/vibevoice_warm_bench_timing.log");
        engine::debug::configure_logging(engine::debug::LoggingConfig{!log_file.empty(), log_file});

        std::ofstream timing_log;
        if (!timing_path.empty()) {
            const auto parent = timing_path.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            timing_log.open(timing_path, std::ios::trunc);
            if (!timing_log) {
                throw std::runtime_error("failed to open VibeVoice warmbench timing file: " + timing_path.string());
            }
            timing_log << "vibevoice.model_path " << model_path.string() << "\n";
            timing_log << "vibevoice.backend " << backend_name << "\n";
            timing_log << "vibevoice.device " << device << "\n";
            timing_log << "vibevoice.threads " << threads << "\n";
            timing_log << "vibevoice.warmup " << warmup << "\n";
            timing_log << "vibevoice.iterations " << iterations << "\n";
        }

        set_env_required("MINITTS_TIMING_ENABLED", "1");
        set_env_required("MINITTS_TIMING_FILE", timing_path.string());

        const auto backend = parse_backend(backend_name);
        auto assets = engine::models::vibevoice::load_vibevoice_assets(model_path);

        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }

        engine::io::json::Value::Array steps;
        if (batch) {
            engine::models::vibevoice::VibeVoiceTextTokenizer text_tokenizer(assets);
            engine::models::vibevoice::VibeVoiceTokenizerWeightsRuntime audio_tokenizer(assets, backend, device, threads);
            engine::models::vibevoice::VibeVoiceConnectorWeightsRuntime connector(assets, backend, device, threads);
            engine::models::vibevoice::VibeVoiceQwen2WeightsRuntime decoder(assets, backend, device, threads);
            engine::models::vibevoice::VibeVoiceDiffusionHeadWeightsRuntime diffusion_head(assets, backend, device, threads);
            const auto requests = parse_requests(request_sequence_json, *assets, prompt_noise_file, noise_file);
            steps.reserve(requests.size());
            for (int i = 0; i < warmup; ++i) {
                (void) engine::models::vibevoice::generate_vibevoice(
                    requests.front(),
                    text_tokenizer,
                    audio_tokenizer,
                    connector,
                    decoder,
                    diffusion_head);
            }
            std::vector<engine::models::vibevoice::VibeVoiceResult> last_results;
            double total_ms = 0.0;
            for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
                const auto started = std::chrono::steady_clock::now();
                last_results = engine::models::vibevoice::generate_vibevoice_batch(
                    requests,
                    text_tokenizer,
                    audio_tokenizer,
                    connector,
                    decoder,
                    diffusion_head);
                const auto ended = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
            }
            const double wall_ms = total_ms / static_cast<double>(std::max(1, iterations));
            if (last_results.size() != requests.size()) {
                throw std::runtime_error("VibeVoice warmbench batch result count mismatch");
            }
            for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
                if (timing_log) {
                    timing_log << "vibevoice.request_index " << request_index << "\n";
                    timing_log << "vibevoice.synthesize_wall_ms " << wall_ms << "\n";
                }
                std::filesystem::path audio_path;
                if (!output_dir.empty()) {
                    audio_path = output_dir / ("audio_" + std::to_string(request_index) + ".wav");
                    const auto & audio = last_results[request_index].audio;
                    engine::audio::write_pcm16_wav(audio_path, audio.sample_rate, audio.channels, audio.samples);
                }
                std::cout << "vibevoice.wall_ms=" << wall_ms << "\n";
                steps.push_back(step_json(last_results[request_index], static_cast<int>(request_index), wall_ms, audio_path));
            }
        } else {
            engine::runtime::SessionOptions options;
            options.backend.type = backend;
            options.backend.device = device;
            options.backend.threads = threads;
            engine::models::vibevoice::VibeVoiceSession session(
                {engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline},
                options,
                assets);
            const auto requests = parse_task_requests(request_sequence_json, prompt_noise_file, noise_file);
            steps.reserve(requests.size());
            session.prepare(engine::runtime::build_preparation_request(requests.front()));
            for (int i = 0; i < warmup; ++i) {
                (void) session.run(requests.front());
            }
            for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
                engine::runtime::TaskResult last_result;
                double total_ms = 0.0;
                for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
                    const auto started = std::chrono::steady_clock::now();
                    last_result = session.run(requests[request_index]);
                    const auto ended = std::chrono::steady_clock::now();
                    total_ms += std::chrono::duration<double, std::milli>(ended - started).count();
                }
                if (!last_result.audio_output.has_value()) {
                    throw std::runtime_error("VibeVoice warmbench expected audio output");
                }
                const double wall_ms = total_ms / static_cast<double>(std::max(1, iterations));
                if (timing_log) {
                    timing_log << "vibevoice.request_index " << request_index << "\n";
                    timing_log << "vibevoice.synthesize_wall_ms " << wall_ms << "\n";
                }
                std::filesystem::path audio_path;
                if (!output_dir.empty()) {
                    audio_path = output_dir / ("audio_" + std::to_string(request_index) + ".wav");
                    const auto & audio = *last_result.audio_output;
                    engine::audio::write_pcm16_wav(audio_path, audio.sample_rate, audio.channels, audio.samples);
                }
                std::cout << "vibevoice.wall_ms=" << wall_ms << "\n";
                steps.push_back(step_json(last_result, static_cast<int>(request_index), wall_ms, audio_path));
            }
        }

        const auto summary = engine::io::json::Value::make_object({
            {"family", string("vibevoice")},
            {"backend", string(backend_name)},
            {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
        });
        std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "vibevoice_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
