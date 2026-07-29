#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
  engine::core::BackendType backend = engine::core::BackendType::Cpu;
  std::filesystem::path model;
  std::filesystem::path audio;
};

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) {
      throw std::runtime_error("missing probe argument value");
    }
    const std::string option(argv[index]);
    const std::string value(argv[++index]);
    if (option == "--backend") {
      if (value == "cpu") {
        result.backend = engine::core::BackendType::Cpu;
      } else if (value == "cuda") {
        result.backend = engine::core::BackendType::Cuda;
      } else {
        throw std::runtime_error("unsupported probe backend: " + value);
      }
    } else if (option == "--model") {
      result.model = value;
    } else if (option == "--audio") {
      result.audio = value;
    } else {
      throw std::runtime_error("unknown probe argument: " + option);
    }
  }
  if (result.model.empty() || result.audio.empty()) {
    throw std::runtime_error("--model and --audio are required");
  }
  return result;
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_runtime_error(const std::string &needle,
                           const std::function<void()> &operation) {
  try {
    operation();
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find(needle) != std::string::npos) {
      return;
    }
    throw std::runtime_error("unexpected error: " + std::string(error.what()));
  }
  throw std::runtime_error("operation did not reject invalid input: " + needle);
}

bool has_option(const std::vector<engine::runtime::CliOptionInfo> &options,
                const std::string &name) {
  return std::any_of(options.begin(), options.end(),
                     [&](const auto &option) { return option.name == name; });
}

engine::runtime::AudioBuffer read_audio(const std::filesystem::path &path) {
  const auto wav = engine::audio::read_wav_f32(path);
  return {wav.sample_rate, wav.channels, wav.samples};
}

engine::runtime::AudioBuffer
make_resampled_stereo(const engine::runtime::AudioBuffer &source) {
  auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
      source.samples, source.sample_rate, source.channels, 8000);
  std::vector<float> stereo;
  stereo.reserve(mono.size() * 2);
  for (const float sample : mono) {
    stereo.push_back(sample);
    stereo.push_back(sample);
  }
  return {8000, 2, std::move(stereo)};
}

} // namespace

int main(int argc, char **argv) try {
  const auto arguments = parse_arguments(argc, argv);
  auto registry = engine::runtime::make_default_registry();
  engine::runtime::ModelLoadRequest load_request;
  load_request.model_path = arguments.model;
  load_request.family_hint = "fun_asr_nano";

  const auto inspection = registry.inspect(load_request);
  require(inspection.metadata.family == "fun_asr_nano",
          "loader family mismatch");
  require(inspection.capabilities.supported_tasks.size() == 1,
          "loader must advertise exactly one task");
  require(inspection.capabilities.supported_tasks.front().task ==
              engine::runtime::VoiceTaskKind::Asr,
          "loader must advertise ASR");
  require(inspection.capabilities.supported_tasks.front().modes ==
              std::vector<engine::runtime::RunMode>{
                  engine::runtime::RunMode::Offline},
          "loader must advertise offline mode only");
  require(!inspection.capabilities.supports_timestamps,
          "loader must not advertise timestamps");
  for (const char *name : {"language", "enable_itn", "max_tokens",
                           "audio_chunk_mode", "audio_chunk_seconds"}) {
    require(has_option(inspection.cli.request_options, name),
            std::string("missing request option: ") + name);
  }

  auto model = registry.load(load_request);
  require_runtime_error("offline", [&] {
    (void)model->create_task_session({engine::runtime::VoiceTaskKind::Asr,
                                      engine::runtime::RunMode::Streaming},
                                     {});
  });

  engine::runtime::SessionOptions session_options;
  session_options.backend = {arguments.backend, 0, 4};
  session_options.options["fun_asr_nano.weight_type"] = "f32";
  auto invalid_session_options = session_options;
  invalid_session_options.options["fun_asr_nano.unknown"] = "1";
  require_runtime_error("unknown Fun-ASR-Nano session option", [&] {
    (void)model->create_task_session({engine::runtime::VoiceTaskKind::Asr,
                                      engine::runtime::RunMode::Offline},
                                     invalid_session_options);
  });
  auto session_base = model->create_task_session(
      {engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline},
      session_options);
  auto *session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
      session_base.get());
  require(session != nullptr, "loader did not create an offline session");

  const auto audio = read_audio(arguments.audio);
  engine::runtime::TaskRequest request;
  request.audio_input = audio;
  request.options["language"] = "auto";
  request.options["enable_itn"] = "true";
  request.options["max_tokens"] = "1";
  request.options["audio_chunk_mode"] = "none";
  require_runtime_error("prepare", [&] { (void)session->run(request); });

  session->prepare(engine::runtime::build_preparation_request(request));
  const auto first = session->run(request);
  require(first.text_output.has_value(), "session omitted text output");
  require(first.word_timestamps.empty(), "session emitted timestamps");

  request.audio_input = make_resampled_stereo(audio);
  const auto resampled = session->run(request);
  require(resampled.text_output.has_value(),
          "resampled stereo request omitted text output");

  request.options["max_tokens"] = "0";
  require_runtime_error("max_tokens", [&] { (void)session->run(request); });
  request.options["max_tokens"] = "1";
  request.options["enable_itn"] = "maybe";
  require_runtime_error("enable_itn", [&] { (void)session->run(request); });
  request.options["enable_itn"] = "true";
  request.options["audio_chunk_mode"] = "vad";
  require_runtime_error("audio_chunk_mode",
                        [&] { (void)session->run(request); });
  request.options["audio_chunk_mode"] = "fixed";
  request.options["audio_chunk_seconds"] = "inf";
  require_runtime_error("audio_chunk_seconds",
                        [&] { (void)session->run(request); });
  const int saved_sample_rate = request.audio_input->sample_rate;
  request.audio_input->sample_rate = 1;
  request.options["audio_chunk_seconds"] = "9223372036854775808";
  require_runtime_error("audio_chunk_seconds is too large",
                        [&] { (void)session->run(request); });
  request.audio_input->sample_rate = saved_sample_rate;
  request.options.erase("audio_chunk_seconds");
  request.options["return_timestamps"] = "true";
  require_runtime_error("unknown Fun-ASR-Nano request option",
                        [&] { (void)session->run(request); });
  request.options.erase("return_timestamps");
  request.options["langauge"] = "en";
  require_runtime_error("unknown Fun-ASR-Nano request option",
                        [&] { (void)session->run(request); });

  session_base.reset();
  auto second_session = model->create_task_session(
      {engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline},
      session_options);
  require(dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
              second_session.get()) != nullptr,
          "loaded model could not create a second offline session");

  std::cout << "fun_asr_nano_session_probe passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fun_asr_nano_session_probe failed: " << error.what() << '\n';
  return 1;
}
