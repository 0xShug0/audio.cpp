#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/voxcpm2/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string arg_value(int argc, char **argv, const std::string &name,
                      const std::string &fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

int int_arg(int argc, char **argv, const std::string &name, int fallback) {
  return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

std::vector<std::pair<std::string, std::string>>
parse_session_options(int argc, char **argv) {
  std::vector<std::pair<std::string, std::string>> out;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) != "--session-option") {
      continue;
    }
    const std::string option = argv[i + 1];
    const size_t eq = option.find('=');
    if (eq == std::string::npos || eq == 0) {
      throw std::runtime_error("invalid VoxCPM2 --session-option: " + option);
    }
    out.emplace_back(option.substr(0, eq), option.substr(eq + 1));
  }
  return out;
}

engine::core::BackendType parse_backend(const std::string &value) {
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
  throw std::runtime_error("unsupported VoxCPM2 warmbench backend: " + value);
}

std::filesystem::path resolve_path(const std::string &value) {
  std::filesystem::path path(value);
  if (path.is_absolute()) {
    return path;
  }
  return std::filesystem::current_path() / path;
}

engine::runtime::AudioBuffer read_audio(const std::string &value) {
  const auto wav = engine::audio::read_wav_f32(resolve_path(value));
  return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels,
                                      wav.samples};
}

std::string optional_string(const engine::io::json::Value &object,
                            const std::string &key) {
  const auto *value = object.find(key);
  return value == nullptr || value->is_null() ? std::string{}
                                              : value->as_string();
}

std::string required_string(const engine::io::json::Value &object,
                            const std::string &key) {
  const auto out = optional_string(object, key);
  if (out.empty()) {
    throw std::runtime_error("VoxCPM2 warmbench request missing field: " + key);
  }
  return out;
}

std::string option_text(const engine::io::json::Value &value) {
  if (value.is_bool()) {
    return value.as_bool() ? "true" : "false";
  }
  if (value.is_number()) {
    return engine::io::json::stringify_number(value.as_number());
  }
  return value.as_string();
}

void set_optional_option(engine::runtime::TaskRequest &request,
                         const engine::io::json::Value &object,
                         const std::string &source,
                         const std::string &target) {
  const auto *value = object.find(source);
  if (value != nullptr && !value->is_null()) {
    request.options[target] = option_text(*value);
  }
}

void reject_enabled_bool(const engine::io::json::Value &object,
                         const std::string &key,
                         const std::string &message) {
  const auto *value = object.find(key);
  if (value != nullptr && value->is_bool() && value->as_bool()) {
    throw std::runtime_error(message);
  }
}

engine::runtime::TaskRequest make_request(const engine::io::json::Value &object,
                                          const std::string &noise_file) {
  engine::runtime::TaskRequest request;
  request.text_input = engine::runtime::Transcript{required_string(object, "text"), ""};
  if (const auto prompt_wav = optional_string(object, "prompt_wav_path");
      !prompt_wav.empty()) {
    request.audio_input = read_audio(prompt_wav);
  }
  if (const auto reference_wav = optional_string(object, "reference_wav_path");
      !reference_wav.empty()) {
    request.voice = engine::runtime::VoiceCondition{};
    request.voice->speaker = engine::runtime::VoiceReference{};
    request.voice->speaker->audio = read_audio(reference_wav);
  }
  reject_enabled_bool(object, "normalize",
                      "VoxCPM2 C++ warmbench does not implement normalize");
  set_optional_option(request, object, "prompt_text", "prompt_text");
  set_optional_option(request, object, "cfg_value", "guidance_scale");
  set_optional_option(request, object, "inference_timesteps",
                      "num_inference_steps");
  set_optional_option(request, object, "min_len", "min_tokens");
  set_optional_option(request, object, "max_len", "max_tokens");
  set_optional_option(request, object, "retry_badcase", "retry_badcase");
  set_optional_option(request, object, "retry_badcase_max_times",
                      "retry_badcase_max_times");
  set_optional_option(request, object, "retry_badcase_ratio_threshold",
                      "retry_badcase_ratio_threshold");
  set_optional_option(request, object, "seed", "seed");
  set_optional_option(request, object, "denoise", "denoise");
  set_optional_option(request, object, "load_denoiser", "load_denoiser");
  if (!noise_file.empty()) {
    request.options["voxcpm2.cfm_noise_file"] = noise_file;
  }
  return request;
}

std::vector<engine::runtime::TaskRequest>
parse_requests(const std::string &request_sequence_json,
               const std::string &noise_file) {
  const auto root = engine::io::json::parse(request_sequence_json);
  std::vector<engine::runtime::TaskRequest> requests;
  for (const auto &item : root.as_array()) {
    requests.push_back(make_request(item, noise_file));
  }
  return requests;
}

engine::io::json::Value number(double value) {
  return engine::io::json::Value::make_number(value);
}

engine::io::json::Value string(std::string value) {
  return engine::io::json::Value::make_string(std::move(value));
}

engine::io::json::Value
audio_summary_json(const engine::runtime::AudioBuffer &audio) {
  if (audio.samples.empty()) {
    throw std::runtime_error("VoxCPM2 warmbench received empty audio output");
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
  const auto channels = std::max(1, audio.channels);
  const auto frames =
      static_cast<double>(audio.samples.size() / static_cast<size_t>(channels));
  const auto count = static_cast<double>(audio.samples.size());
  return engine::io::json::Value::make_object({
      {"sample_rate", number(static_cast<double>(audio.sample_rate))},
      {"channels", number(static_cast<double>(audio.channels))},
      {"samples", number(count)},
      {"frames", number(frames)},
      {"duration_sec",
       number(audio.sample_rate > 0 ? frames / audio.sample_rate : 0.0)},
      {"sum", number(sum)},
      {"mean_abs", number(abs_sum / count)},
      {"rms", number(std::sqrt(sq_sum / count))},
      {"min", number(min_value)},
      {"max", number(max_value)},
  });
}

engine::io::json::Value step_json(const engine::runtime::TaskResult &result,
                                  int request_index, double wall_ms,
                                  const std::filesystem::path &audio_path,
                                  int64_t chunk_count) {
  if (!result.audio_output.has_value()) {
    throw std::runtime_error("VoxCPM2 warmbench expected audio output");
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
      {"stems", engine::io::json::Value::make_array(
                    {engine::io::json::Value::make_object(std::move(stem))})},
      {"metrics", engine::io::json::Value::make_object(
                      {{"wall_ms", number(wall_ms)},
                       {"chunk_count", number(static_cast<double>(chunk_count))}})},
  });
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::filesystem::path model_path =
        arg_value(argc, argv, "--model", "models/VoxCPM2");
    const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
    const int device = int_arg(argc, argv, "--device", 0);
    const int threads = int_arg(argc, argv, "--threads", 8);
    const int warmup = int_arg(argc, argv, "--warmup", 0);
    const int iterations = int_arg(argc, argv, "--iterations", 1);
    const std::string run_mode = arg_value(argc, argv, "--run-mode", "offline");
    const std::string request_sequence_json =
        arg_value(argc, argv, "--request-sequence-json", "");
    const std::filesystem::path output_dir =
        arg_value(argc, argv, "--output-dir", "");
    const std::filesystem::path timing_path =
        arg_value(argc, argv, "--timing-file",
                  "/tmp/voxcpm2_warm_bench_timing.log");
    const std::string noise_file = arg_value(argc, argv, "--noise-file", "");
    if (request_sequence_json.empty()) {
      throw std::runtime_error("VoxCPM2 warmbench requires --request-sequence-json");
    }
    engine::debug::configure_logging(
        engine::debug::LoggingConfig{true, timing_path.string()});

    const auto session_option_overrides = parse_session_options(argc, argv);
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_path;
    load_request.family_hint = "voxcpm2";
    auto registry = engine::runtime::make_default_registry();
    auto model = registry.load(load_request);

    engine::runtime::SessionOptions options;
    options.backend.type = parse_backend(backend_name);
    options.backend.device = device;
    options.backend.threads = threads;
    options.options.insert_or_assign("voxcpm2.weight_type", "f32");
    for (const auto &[key, value] : session_option_overrides) {
      if (key == "voxcpm2.weight_type" && value != "f32") {
        throw std::runtime_error(
            "VoxCPM2 warmbench forces voxcpm2.weight_type=f32 for parity");
      }
      options.options.insert_or_assign(key, value);
    }

    const bool streaming = run_mode == "streaming";
    if (!streaming && run_mode != "offline") {
      throw std::runtime_error("unsupported VoxCPM2 warmbench run mode: " +
                               run_mode);
    }
    auto session_base = model->create_task_session(
        {engine::runtime::VoiceTaskKind::Tts,
         streaming ? engine::runtime::RunMode::Streaming
                   : engine::runtime::RunMode::Offline},
        options);
    auto *offline_session =
        dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
            session_base.get());
    auto *streaming_session =
        dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(
            session_base.get());
    if (streaming) {
      if (streaming_session == nullptr) {
        throw std::runtime_error(
            "loaded VoxCPM2 session is not streaming-capable");
      }
    } else if (offline_session == nullptr) {
      throw std::runtime_error("loaded VoxCPM2 session is not offline-capable");
    }

    const auto requests = parse_requests(request_sequence_json, noise_file);
    if (requests.empty()) {
      throw std::runtime_error("VoxCPM2 warmbench request sequence is empty");
    }
    session_base->prepare(engine::runtime::build_preparation_request(requests.front()));
    for (int i = 0; i < warmup; ++i) {
      if (streaming) {
        streaming_session->start_stream(requests.front());
        while (streaming_session->next_stream_event().has_value()) {
        }
        (void)streaming_session->finish_stream();
      } else {
        (void)offline_session->run(requests.front());
      }
    }

    if (!output_dir.empty()) {
      std::filesystem::create_directories(output_dir);
    }

    engine::io::json::Value::Array steps;
    steps.reserve(requests.size());
    for (size_t request_index = 0; request_index < requests.size();
         ++request_index) {
      engine::runtime::TaskResult last_result;
      double total_ms = 0.0;
      for (int iteration = 0; iteration < std::max(1, iterations); ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        if (streaming) {
          streaming_session->start_stream(requests[request_index]);
          while (streaming_session->next_stream_event().has_value()) {
          }
          last_result = streaming_session->finish_stream();
        } else {
          last_result = offline_session->run(requests[request_index]);
        }
        const auto ended = std::chrono::steady_clock::now();
        total_ms +=
            std::chrono::duration<double, std::milli>(ended - started).count();
      }
      if (!last_result.audio_output.has_value()) {
        throw std::runtime_error("VoxCPM2 warmbench expected audio output");
      }
      const double wall_ms =
          total_ms / static_cast<double>(std::max(1, iterations));
      std::filesystem::path audio_path;
      if (!output_dir.empty()) {
        audio_path = output_dir / ("audio_" + std::to_string(request_index) + ".wav");
        engine::audio::write_pcm16_wav(
            audio_path, last_result.audio_output->sample_rate,
            last_result.audio_output->channels, last_result.audio_output->samples);
      }
      std::cout << "voxcpm2.wall_ms=" << wall_ms << "\n";
      steps.push_back(
          step_json(last_result, static_cast<int>(request_index), wall_ms,
                    audio_path,
                    static_cast<int64_t>(
                        last_result.named_audio_outputs.size())));
    }

    const auto summary = engine::io::json::Value::make_object({
        {"family", string("voxcpm2")},
        {"backend", string(backend_name)},
        {"mode", string(run_mode)},
        {"sequence_steps", engine::io::json::Value::make_array(std::move(steps))},
    });
    std::cout << "summary_json=" << engine::io::json::stringify(summary) << "\n";
    engine::debug::reset_logging();
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "voxcpm2_warm_bench failed: " << ex.what() << "\n";
    return 1;
  }
}
