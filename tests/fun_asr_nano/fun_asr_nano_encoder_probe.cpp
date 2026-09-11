#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/encoder.h"
#include "engine/models/fun_asr_nano/frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t kGraphArenaBytes = 128 * 1024 * 1024;
constexpr float kAbsoluteTolerance = 2.0e-3F;
constexpr float kRelativeTolerance = 2.0e-3F;
constexpr const char *kTransformersCommit =
    "48e7f65fb274172e15aa88875d780c67c37606c7";
constexpr const char *kModelRevision =
    "854d88f94205cd17d2afdb24332130d86fbe654a";

struct Arguments {
  engine::core::BackendType backend = engine::core::BackendType::Cpu;
  std::filesystem::path model;
  std::filesystem::path reference;
  std::filesystem::path data;
};

Arguments parse_arguments(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + option);
    }
    const std::string value(argv[++index]);
    if (option == "--backend") {
      if (value == "cpu") {
        arguments.backend = engine::core::BackendType::Cpu;
      } else if (value == "cuda") {
        arguments.backend = engine::core::BackendType::Cuda;
      } else {
        throw std::runtime_error("unsupported encoder probe backend: " + value);
      }
    } else if (option == "--model") {
      arguments.model = value;
    } else if (option == "--reference") {
      arguments.reference = value;
    } else if (option == "--data") {
      arguments.data = value;
    } else {
      throw std::runtime_error("unknown encoder probe option: " + option);
    }
  }
  if (arguments.model.empty() || arguments.reference.empty() ||
      arguments.data.empty()) {
    throw std::runtime_error("--model, --reference, and --data are required");
  }
  return arguments;
}

std::vector<float> load_f32_slice(const std::filesystem::path &data_path,
                                  const engine::io::json::Value &descriptor) {
  const int64_t signed_offset = descriptor.require("offset_f32").as_i64();
  const int64_t signed_count = descriptor.require("count").as_i64();
  if (signed_offset < 0 || signed_count < 0 ||
      static_cast<uint64_t>(signed_count) >
          std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("invalid encoder reference data slice");
  }
  const uint64_t offset = static_cast<uint64_t>(signed_offset);
  const size_t count = static_cast<size_t>(signed_count);
  std::ifstream input(data_path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset * sizeof(float)));
  if (!input) {
    throw std::runtime_error("could not seek in encoder reference data");
  }
  std::vector<float> values(count);
  for (float &value : values) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char *>(bytes), 4);
    if (!input) {
      throw std::runtime_error("truncated encoder reference data");
    }
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    std::memcpy(&value, &bits, sizeof(value));
  }
  return values;
}

void require_close(const std::string &label, const std::vector<float> &actual,
                   const std::vector<float> &expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + " output size mismatch");
  }
  float maximum_absolute_error = 0.0F;
  float maximum_relative_error = 0.0F;
  size_t worst_index = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float absolute_error = std::abs(actual[index] - expected[index]);
    const float relative_error =
        absolute_error / std::max(std::abs(expected[index]), 1.0e-12F);
    if (absolute_error > maximum_absolute_error) {
      maximum_absolute_error = absolute_error;
      worst_index = index;
    }
    maximum_relative_error = std::max(maximum_relative_error, relative_error);
    const float limit =
        kAbsoluteTolerance + kRelativeTolerance * std::abs(expected[index]);
    if (absolute_error > limit) {
      throw std::runtime_error(
          label + " mismatch at " + std::to_string(index) + ": expected " +
          std::to_string(expected[index]) + ", got " +
          std::to_string(actual[index]) + ", absolute error " +
          std::to_string(absolute_error) + ", limit " + std::to_string(limit));
    }
  }
  std::cout << label << ": count=" << actual.size()
            << ", max_abs=" << maximum_absolute_error
            << ", max_rel=" << maximum_relative_error
            << ", worst_index=" << worst_index << '\n';
}

void require_runtime_error(const std::string &label, const std::string &message,
                           const std::function<void()> &operation) {
  try {
    operation();
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find(message) != std::string::npos) {
      return;
    }
    throw std::runtime_error(label +
                             " returned an unexpected error: " + error.what());
  }
  throw std::runtime_error(label + " did not reject invalid input");
}

} // namespace

int main(int argc, char **argv) try {
  const auto arguments = parse_arguments(argc, argv);
  const auto reference = engine::io::json::parse_file(arguments.reference);
  if (reference.require("schema_version").as_i64() != 1 ||
      reference.require("transformers_commit").as_string() !=
          kTransformersCommit ||
      reference.require("model_revision").as_string() != kModelRevision ||
      reference.require("weight_catalog_count").as_i64() != 1194 ||
      reference.require("data_format").as_string() != "little-endian-float32") {
    throw std::runtime_error("unexpected encoder reference metadata");
  }

  const auto input_shape =
      engine::io::json::require_i64_array(reference.require("input"), "shape");
  if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] <= 0 ||
      input_shape[2] != 560) {
    throw std::runtime_error("unexpected encoder reference input shape");
  }
  engine::models::fun_asr_nano::FunAsrNanoAudioFeatures features;
  features.values = load_f32_slice(arguments.data, reference.require("input"));
  features.frames = input_shape[1];
  features.feature_dim = input_shape[2];
  features.valid_frames = reference.require("valid_frames").as_i64();

  auto assets =
      engine::models::fun_asr_nano::load_fun_asr_nano_assets(arguments.model);
  engine::models::fun_asr_nano::FunAsrNanoFrontend frontend(
      assets->config.frontend);
  const auto frontend_output =
      frontend.extract(std::vector<float>(800, 0.0F), 16000);
  if (frontend_output.frames <= 0 || frontend_output.feature_dim != 560 ||
      frontend_output.valid_frames != frontend_output.frames ||
      static_cast<int64_t>(frontend_output.values.size()) !=
          frontend_output.frames * frontend_output.feature_dim) {
    throw std::runtime_error("unexpected encoder frontend contract");
  }
  bool rejected_sample_rate = false;
  try {
    static_cast<void>(frontend.extract(std::vector<float>(800, 0.0F), 8000));
  } catch (const std::runtime_error &) {
    rejected_sample_rate = true;
  }
  if (!rejected_sample_rate) {
    throw std::runtime_error(
        "Fun-ASR-Nano frontend accepted a non-16-kHz sample rate");
  }
  engine::core::ExecutionContext execution_context({arguments.backend, 0, 4});
  engine::models::fun_asr_nano::FunAsrNanoEncoderRuntime runtime(
      std::move(assets), execution_context, kGraphArenaBytes);
  const auto output = runtime.encode(features, true);
  if (output.frames != features.frames ||
      output.valid_frames != features.valid_frames ||
      output.hidden_size != 512 || output.stages.size() != 9) {
    throw std::runtime_error("unexpected encoder output contract");
  }

  std::unordered_map<std::string, std::vector<float>> actual_stages;
  for (const auto &stage : output.stages) {
    if (!actual_stages.emplace(stage.name, stage.values).second) {
      throw std::runtime_error("duplicate encoder stage: " + stage.name);
    }
  }
  const auto &expected_stages = reference.require("checkpoints");
  for (const char *name :
       {"stem", "main_layer_0", "main_layer_24", "main_layer_48",
        "main_layer_norm", "timestamp_layer_0", "timestamp_layer_10",
        "timestamp_layer_19", "final"}) {
    const auto found = actual_stages.find(name);
    if (found == actual_stages.end()) {
      throw std::runtime_error("missing encoder stage: " + std::string(name));
    }
    require_close(
        name, found->second,
        load_f32_slice(arguments.data, expected_stages.require(name)));
  }
  require_close(
      "final_output", output.values,
      load_f32_slice(arguments.data, expected_stages.require("final")));

  const auto cached_output = runtime.encode(features, false);
  if (!cached_output.stages.empty()) {
    throw std::runtime_error(
        "encoder graph cache returned unrequested stage outputs");
  }
  require_close("cached_output", cached_output.values, output.values);

  auto padded_features = features;
  --padded_features.valid_frames;
  require_runtime_error("encoder padding contract", "unpadded", [&] {
    static_cast<void>(runtime.encode(padded_features));
  });
  auto malformed_features = features;
  malformed_features.values.pop_back();
  require_runtime_error("encoder value count", "value count", [&] {
    static_cast<void>(runtime.encode(malformed_features));
  });
  require_runtime_error("encoder position capacity", "positional capacity",
                        [&] { runtime.prepare_capacity(2049); });
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fun_asr_nano_encoder_probe failed: " << error.what() << '\n';
  return 1;
}
