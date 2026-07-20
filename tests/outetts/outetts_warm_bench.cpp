#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
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
  std::string reference_text;
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
      "OuteTTS warm-bench options must be strings, numbers, or booleans");
}

void copy_option_if_present(
    std::unordered_map<std::string, std::string> &options,
    const engine::io::json::Value &item, const std::string &name) {
  if (const auto *value = item.find(name);
      value != nullptr && !value->is_null()) {
    options[name] = scalar_option(*value);
  }
}

std::vector<RequestCase>
load_requests(const std::filesystem::path &path,
              const std::unordered_map<std::string, std::string> &defaults) {
  const auto root = engine::io::json::parse_file(path);
  const auto &items = root.require("requests").as_array();
  if (items.empty())
    throw std::runtime_error("OuteTTS request file has no requests");
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
    request.voice_ref =
        engine::io::json::optional_string(item, "voice_ref", "");
    request.reference_text =
        engine::io::json::optional_string(item, "reference_text", "");
    request.options = defaults;
    for (const char *name :
         {"max_tokens", "seed", "temperature", "top_k", "top_p", "min_p",
          "repetition_penalty", "repetition_window", "text_chunk_size",
          "text_chunk_mode", "reference_language"}) {
      copy_option_if_present(request.options, item, name);
    }
    if (!request.reference_text.empty())
      request.options["reference_text"] = request.reference_text;
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

engine::runtime::TaskRequest make_request(
    const RequestCase &request,
    std::unordered_map<std::string, engine::runtime::AudioBuffer>
        &audio_cache) {
  engine::runtime::TaskRequest out;
  out.text_input =
      engine::runtime::Transcript{request.text, request.language};
  out.options = request.options;
  if (!request.voice_ref.empty()) {
    const std::string key = request.voice_ref.string();
    auto found = audio_cache.find(key);
    if (found == audio_cache.end())
      found = audio_cache.emplace(key, read_audio(request.voice_ref)).first;
    out.voice = engine::runtime::VoiceCondition{};
    out.voice->speaker = engine::runtime::VoiceReference{};
    out.voice->speaker->audio = found->second;
  }
  return out;
}

} // namespace

int main(int argc, char **argv) try {
  const std::filesystem::path model_path =
      arg_value(argc, argv, "--model",
                "models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf");
  const std::filesystem::path request_file =
      arg_value(argc, argv, "--request-file", "");
  if (request_file.empty())
    throw std::runtime_error("OuteTTS warm bench requires --request-file");
  const std::filesystem::path output_dir =
      arg_value(argc, argv, "--audio-out-dir",
                "build/logs/warmbench/outetts_audio");
  const std::filesystem::path log_file =
      arg_value(argc, argv, "--log-file",
                arg_value(argc, argv, "--trace-file",
                          "build/logs/warmbench/outetts.log"));
  const std::string backend_name =
      arg_value(argc, argv, "--backend", "cpu");
  const int device = int_arg(argc, argv, "--device", 0);
  const int threads = int_arg(argc, argv, "--threads", 8);
  const int iterations = int_arg(argc, argv, "--iterations", 1);
  const int hold_seconds = int_arg(argc, argv, "--hold-seconds", 0);
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
  const auto requests = load_requests(request_file, defaults);

  std::filesystem::create_directories(output_dir);
  if (!log_file.parent_path().empty())
    std::filesystem::create_directories(log_file.parent_path());
  engine::debug::configure_logging(
      engine::debug::LoggingConfig{true, log_file.string()});

  auto registry = engine::runtime::make_default_registry();
  engine::runtime::ModelLoadRequest load_request;
  load_request.model_path = model_path;
  load_request.family_hint = "outetts";
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

  auto session_base = model->create_task_session(
      {engine::runtime::VoiceTaskKind::Tts,
       engine::runtime::RunMode::Offline},
      session_options);
  auto *session =
      dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
          session_base.get());
  if (session == nullptr)
    throw std::runtime_error(
        "OuteTTS did not create an offline TTS session");
  session->prepare(engine::runtime::SessionPreparationRequest{});

  std::unordered_map<std::string, engine::runtime::AudioBuffer> audio_cache;
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const auto started = std::chrono::steady_clock::now();
      auto result = session->run(
          make_request(requests[request_index], audio_cache));
      const double wall_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
      if (!result.audio_output.has_value())
        throw std::runtime_error("OuteTTS produced no audio");
      const double seconds = audio_seconds(*result.audio_output);
      const auto output_path =
          output_dir /
          (requests[request_index].name + "_" +
           std::to_string(iteration + 1) + ".wav");
      engine::audio::write_pcm16_wav(
          output_path, result.audio_output->sample_rate,
          result.audio_output->channels, result.audio_output->samples);
      std::cout << "request=" << requests[request_index].name << "\n";
      std::cout << "iteration=" << iteration + 1 << "\n";
      std::cout << "wall_ms=" << wall_ms << "\n";
      std::cout << "audio_seconds=" << seconds << "\n";
      std::cout << "rtf="
                << (seconds > 0.0 ? wall_ms / 1000.0 / seconds : 0.0)
                << "\n";
      std::cout << "audio_out=" << output_path.string() << "\n";
    }
  }
  std::cout << "log_out=" << log_file.string() << "\n";
  if (hold_seconds > 0) {
    std::cout << "holding_session_seconds=" << hold_seconds << "\n";
    std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
  }
  return 0;
} catch (const std::exception &error) {
  std::cerr << "outetts_warm_bench failed: " << error.what() << "\n";
  return 1;
}
