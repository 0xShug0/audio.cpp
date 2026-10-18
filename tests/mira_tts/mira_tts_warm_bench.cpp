#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct RequestCase {
  std::string name;
  std::string text;
  std::string language;
  std::filesystem::path voice_ref;
  std::unordered_map<std::string, std::string> options;
};

std::string arg_value(int argc, char **argv, const std::string &name,
                      const std::string &fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name)
      return argv[i + 1];
  }
  return fallback;
}

std::vector<std::string> arg_values(int argc, char **argv,
                                    const std::string &name) {
  std::vector<std::string> out;
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name)
      out.emplace_back(argv[i + 1]);
  }
  return out;
}

int int_arg(int argc, char **argv, const std::string &name, int fallback) {
  return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string &value) {
  if (value == "cpu")
    return engine::core::BackendType::Cpu;
  if (value == "cuda")
    return engine::core::BackendType::Cuda;
  if (value == "vulkan")
    return engine::core::BackendType::Vulkan;
  if (value == "best")
    return engine::core::BackendType::BestAvailable;
  throw std::runtime_error("unsupported backend: " + value);
}

std::string scalar_option(const engine::io::json::Value &value) {
  if (value.is_string())
    return value.as_string();
  if (value.is_bool())
    return value.as_bool() ? "true" : "false";
  if (value.is_number())
    return engine::io::json::stringify_number(value.as_number());
  throw std::runtime_error(
      "MiraTTS warm-bench options must be scalar values");
}

void copy_option_if_present(
    std::unordered_map<std::string, std::string> &options,
    const engine::io::json::Value &item, const std::string &name) {
  if (const auto *value = item.find(name);
      value != nullptr && !value->is_null()) {
    options[name] = scalar_option(*value);
  }
}

std::vector<RequestCase> load_requests(
    const std::filesystem::path &path,
    const std::filesystem::path &default_voice_ref,
    const std::unordered_map<std::string, std::string> &defaults) {
  const auto root = engine::io::json::parse_file(path);
  const auto &items = root.require("requests").as_array();
  if (items.empty())
    throw std::runtime_error("MiraTTS request file has no requests");

  std::vector<RequestCase> out;
  out.reserve(items.size());
  for (size_t index = 0; index < items.size(); ++index) {
    const auto &item = items[index];
    RequestCase request;
    request.name = engine::io::json::optional_string(
        item, "name", "request_" + std::to_string(index));
    request.text = engine::io::json::require_string(item, "text");
    request.language =
        engine::io::json::optional_string(item, "language", "en");
    request.voice_ref = engine::io::json::optional_string(
        item, "voice_ref", default_voice_ref.string());
    request.options = defaults;
    for (const char *name :
         {"max_tokens", "seed", "temperature", "top_k", "top_p",
          "min_p", "repetition_penalty", "text_chunk_size",
          "text_chunk_mode"}) {
      copy_option_if_present(request.options, item, name);
    }
    if (request.voice_ref.empty())
      throw std::runtime_error("MiraTTS request '" + request.name +
                               "' has no reference voice");
    out.push_back(std::move(request));
  }
  return out;
}

engine::runtime::AudioBuffer read_audio(const std::filesystem::path &path) {
  const auto wav = engine::audio::read_wav_f32(path);
  return {wav.sample_rate, wav.channels, wav.samples};
}

double audio_seconds(const engine::runtime::AudioBuffer &audio) {
  if (audio.sample_rate <= 0 || audio.channels <= 0)
    return 0.0;
  return static_cast<double>(audio.samples.size()) /
         static_cast<double>(audio.sample_rate * audio.channels);
}

std::string fnv1a64_hex(const engine::runtime::AudioBuffer &audio) {
  uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](const void *data, size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
      hash ^= static_cast<uint64_t>(bytes[i]);
      hash *= 1099511628211ULL;
    }
  };
  mix(&audio.sample_rate, sizeof(audio.sample_rate));
  mix(&audio.channels, sizeof(audio.channels));
  for (float sample : audio.samples) {
    uint32_t bits = 0;
    std::memcpy(&bits, &sample, sizeof(bits));
    mix(&bits, sizeof(bits));
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

engine::runtime::TaskRequest make_request(
    const RequestCase &request,
    std::unordered_map<std::string, engine::runtime::AudioBuffer>
        &audio_cache) {
  engine::runtime::TaskRequest out;
  out.text_input =
      engine::runtime::Transcript{request.text, request.language};
  out.options = request.options;
  const std::string key = request.voice_ref.lexically_normal().string();
  auto found = audio_cache.find(key);
  if (found == audio_cache.end())
    found = audio_cache.emplace(key, read_audio(request.voice_ref)).first;
  out.voice = engine::runtime::VoiceCondition{};
  out.voice->speaker = engine::runtime::VoiceReference{};
  out.voice->speaker->audio = found->second;
  return out;
}

engine::io::json::Value result_json(
    const RequestCase &request, int iteration, const std::string &mode,
    const engine::runtime::AudioBuffer &audio, double wall_ms,
    double first_event_ms, int event_count,
    const std::filesystem::path &output_path) {
  const double seconds = audio_seconds(audio);
  return engine::io::json::Value::make_object({
      {"name", engine::io::json::Value::make_string(request.name)},
      {"iteration", engine::io::json::Value::make_number(iteration)},
      {"mode", engine::io::json::Value::make_string(mode)},
      {"wall_ms", engine::io::json::Value::make_number(wall_ms)},
      {"audio_seconds", engine::io::json::Value::make_number(seconds)},
      {"rtf", engine::io::json::Value::make_number(
                  seconds > 0.0 ? wall_ms / 1000.0 / seconds : 0.0)},
      {"first_event_ms",
       first_event_ms >= 0.0
           ? engine::io::json::Value::make_number(first_event_ms)
           : engine::io::json::Value::make_null()},
      {"event_count", engine::io::json::Value::make_number(event_count)},
      {"sample_rate",
       engine::io::json::Value::make_number(audio.sample_rate)},
      {"channels", engine::io::json::Value::make_number(audio.channels)},
      {"samples",
       engine::io::json::Value::make_number(
           static_cast<double>(audio.samples.size()))},
      {"audio_hash", engine::io::json::Value::make_string(fnv1a64_hex(audio))},
      {"audio_out",
       engine::io::json::Value::make_string(output_path.string())},
  });
}

} // namespace

int main(int argc, char **argv) try {
  const std::filesystem::path model_path =
      arg_value(argc, argv, "--model", "models/MiraTTS/model.gguf");
  const std::filesystem::path request_file =
      arg_value(argc, argv, "--request-file", "");
  if (request_file.empty())
    throw std::runtime_error("MiraTTS warm bench requires --request-file");
  const std::filesystem::path default_voice_ref =
      arg_value(argc, argv, "--voice-ref", "");
  const std::filesystem::path output_dir =
      arg_value(argc, argv, "--audio-out-dir",
                "build/logs/warmbench/mira_tts_audio");
  const std::filesystem::path log_file =
      arg_value(argc, argv, "--log-file",
                "build/logs/warmbench/mira_tts.log");
  const std::filesystem::path summary_file =
      arg_value(argc, argv, "--summary-file",
                "build/logs/warmbench/mira_tts_summary.json");
  const std::filesystem::path spec_override =
      arg_value(argc, argv, "--model-spec-override", "");
  const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
  const std::string mode = arg_value(argc, argv, "--run-mode", "offline");
  const int device = int_arg(argc, argv, "--device", 0);
  const int threads = int_arg(argc, argv, "--threads", 8);
  const int iterations = int_arg(argc, argv, "--iterations", 1);
  const int hold_seconds = int_arg(argc, argv, "--hold-seconds", 0);
  if (mode != "offline" && mode != "streaming")
    throw std::runtime_error("--run-mode must be offline or streaming");
  if (iterations <= 0)
    throw std::runtime_error("--iterations must be positive");
  if (hold_seconds < 0)
    throw std::runtime_error("--hold-seconds must be non-negative");

  std::unordered_map<std::string, std::string> defaults;
  for (const auto &option : arg_values(argc, argv, "--request-option")) {
    const size_t equals = option.find('=');
    if (equals == std::string::npos || equals == 0)
      throw std::runtime_error("invalid --request-option: " + option);
    defaults[option.substr(0, equals)] = option.substr(equals + 1);
  }
  const auto requests =
      load_requests(request_file, default_voice_ref, defaults);

  std::filesystem::create_directories(output_dir);
  if (!log_file.parent_path().empty())
    std::filesystem::create_directories(log_file.parent_path());
  if (!summary_file.parent_path().empty())
    std::filesystem::create_directories(summary_file.parent_path());
  engine::debug::configure_logging(
      engine::debug::LoggingConfig{true, log_file.string()});

  auto registry = engine::runtime::make_default_registry();
  engine::runtime::ModelLoadRequest load_request;
  load_request.model_path = model_path;
  load_request.family_hint = "mira_tts";
  if (!spec_override.empty())
    load_request.model_spec_override = spec_override;
  auto model = registry.load(load_request);

  engine::runtime::SessionOptions session_options;
  session_options.backend.type = parse_backend(backend_name);
  session_options.backend.device = device;
  session_options.backend.threads = threads;
  for (const auto &option : arg_values(argc, argv, "--session-option")) {
    const size_t equals = option.find('=');
    if (equals == std::string::npos || equals == 0)
      throw std::runtime_error("invalid --session-option: " + option);
    session_options.options[option.substr(0, equals)] =
        option.substr(equals + 1);
  }

  const auto run_mode = mode == "streaming"
                            ? engine::runtime::RunMode::Streaming
                            : engine::runtime::RunMode::Offline;
  auto session_base = model->create_task_session(
      {engine::runtime::VoiceTaskKind::VoiceCloning, run_mode},
      session_options);
  auto *offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
      session_base.get());
  auto *streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(
      session_base.get());
  if (mode == "offline" && offline == nullptr)
    throw std::runtime_error("MiraTTS did not create an offline session");
  if (mode == "streaming" && streaming == nullptr)
    throw std::runtime_error("MiraTTS did not create a streaming session");

  std::unordered_map<std::string, engine::runtime::AudioBuffer> audio_cache;
  auto first_request = make_request(requests.front(), audio_cache);
  session_base->prepare(
      engine::runtime::build_preparation_request(first_request));

  engine::io::json::Value::Array results;
  for (const auto &request_case : requests) {
    for (int iteration = 1; iteration <= iterations; ++iteration) {
      auto request = make_request(request_case, audio_cache);
      const auto started = std::chrono::steady_clock::now();
      engine::runtime::TaskResult result;
      double first_event_ms = -1.0;
      int event_count = 0;
      if (mode == "streaming") {
        streaming->start_stream(request);
        while (auto event = streaming->next_stream_event()) {
          ++event_count;
          if (first_event_ms < 0.0) {
            first_event_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started)
                    .count();
          }
        }
        result = streaming->finish_stream();
      } else {
        result = offline->run(request);
      }
      const double wall_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
      if (!result.audio_output.has_value())
        throw std::runtime_error("MiraTTS produced no audio");
      const auto output_path =
          output_dir /
          (request_case.name + "_" + mode + "_" +
           std::to_string(iteration) + ".wav");
      engine::audio::write_pcm16_wav(
          output_path, result.audio_output->sample_rate,
          result.audio_output->channels, result.audio_output->samples);
      auto item = result_json(request_case, iteration, mode,
                              *result.audio_output, wall_ms, first_event_ms,
                              event_count, output_path);
      std::cout << "result_json=" << engine::io::json::stringify(item)
                << "\n";
      results.push_back(std::move(item));
    }
  }

  auto summary = engine::io::json::Value::make_object({
      {"family", engine::io::json::Value::make_string("mira_tts")},
      {"backend", engine::io::json::Value::make_string(backend_name)},
      {"mode", engine::io::json::Value::make_string(mode)},
      {"model", engine::io::json::Value::make_string(model_path.string())},
      {"results",
       engine::io::json::Value::make_array(std::move(results))},
  });
  {
    std::ofstream output(summary_file, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("failed to open summary file: " +
                               summary_file.string());
    output << engine::io::json::stringify(summary) << "\n";
  }
  std::cout << "summary_json=" << summary_file.string() << "\n";
  std::cout << "log_out=" << log_file.string() << "\n";
  if (hold_seconds > 0) {
    std::cout << "holding_session_seconds=" << hold_seconds << "\n";
    std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
  }
  engine::debug::reset_logging();
  return 0;
} catch (const std::exception &error) {
  std::cerr << "mira_tts_warm_bench failed: " << error.what() << "\n";
  return 1;
}
