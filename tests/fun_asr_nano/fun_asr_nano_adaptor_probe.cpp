#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/models/fun_asr_nano/adaptor.h"
#include "engine/models/fun_asr_nano/assets.h"

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
#include <vector>

namespace {

constexpr size_t kGraphArenaBytes = 64 * 1024 * 1024;
constexpr float kAbsoluteTolerance = 2.0e-3F;
constexpr float kRelativeTolerance = 2.0e-3F;
constexpr const char *kTransformersCommit =
    "f9966442ac24fff57060774ce22e1884760f4a3b";
constexpr const char *kModelRevision =
    "854d88f94205cd17d2afdb24332130d86fbe654a";

struct Arguments {
  engine::core::BackendType backend = engine::core::BackendType::Cpu;
  std::filesystem::path model;
  std::filesystem::path reference;
  std::filesystem::path data;
};

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + option);
    }
    const std::string value(argv[++index]);
    if (option == "--backend") {
      if (value == "cpu") {
        result.backend = engine::core::BackendType::Cpu;
      } else if (value == "cuda") {
        result.backend = engine::core::BackendType::Cuda;
      } else {
        throw std::runtime_error("unsupported adaptor probe backend: " + value);
      }
    } else if (option == "--model") {
      result.model = value;
    } else if (option == "--reference") {
      result.reference = value;
    } else if (option == "--data") {
      result.data = value;
    } else {
      throw std::runtime_error("unknown adaptor probe option: " + option);
    }
  }
  if (result.model.empty() || result.reference.empty() || result.data.empty()) {
    throw std::runtime_error("--model, --reference, and --data are required");
  }
  return result;
}

std::vector<float> load_f32_slice(const std::filesystem::path &data_path,
                                  const engine::io::json::Value &descriptor) {
  const int64_t signed_offset = descriptor.require("offset_f32").as_i64();
  const int64_t signed_count = descriptor.require("count").as_i64();
  if (signed_offset < 0 || signed_count < 0 ||
      static_cast<uint64_t>(signed_count) >
          std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("invalid adaptor reference data slice");
  }
  std::ifstream input(data_path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(signed_offset * 4));
  std::vector<float> values(static_cast<size_t>(signed_count));
  for (float &value : values) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char *>(bytes), 4);
    if (!input) {
      throw std::runtime_error("truncated adaptor reference data");
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
      reference.require("weight_catalog_count").as_i64() != 36 ||
      reference.require("data_format").as_string() != "little-endian-float32") {
    throw std::runtime_error("unexpected adaptor reference metadata");
  }

  const auto input_shape =
      engine::io::json::require_i64_array(reference.require("input"), "shape");
  const auto mask_shape =
      engine::io::json::require_i64_array(reference.require("mask"), "shape");
  if (input_shape.size() != 3 || input_shape[0] != 2 || input_shape[1] != 4 ||
      input_shape[2] != 512 || mask_shape.size() != 2 || mask_shape[0] != 2 ||
      mask_shape[1] != 4) {
    throw std::runtime_error("unexpected adaptor reference input shape");
  }
  const auto input_values =
      load_f32_slice(arguments.data, reference.require("input"));
  const auto mask_values =
      load_f32_slice(arguments.data, reference.require("mask"));

  auto assets =
      engine::models::fun_asr_nano::load_fun_asr_nano_assets(arguments.model);
  engine::core::ExecutionContext execution_context({arguments.backend, 0, 4});
  engine::models::fun_asr_nano::FunAsrNanoAdaptorRuntime runtime(
      assets, execution_context, kGraphArenaBytes);

  std::unordered_map<std::string, std::vector<float>> actual_stages;
  std::vector<float> actual_packed;
  std::vector<engine::models::fun_asr_nano::FunAsrNanoAdaptorEmbeddings>
      per_utterance;
  for (int64_t batch = 0; batch < input_shape[0]; ++batch) {
    engine::models::fun_asr_nano::FunAsrNanoEncoderEmbeddings embeddings;
    const size_t row_offset = static_cast<size_t>(batch * 4 * 512);
    embeddings.values.assign(input_values.begin() + row_offset,
                             input_values.begin() + row_offset + 4 * 512);
    embeddings.frames = 4;
    embeddings.hidden_size = 512;
    std::vector<int32_t> mask(4);
    for (int64_t frame = 0; frame < 4; ++frame) {
      mask[static_cast<size_t>(frame)] = static_cast<int32_t>(
          mask_values[static_cast<size_t>(batch * 4 + frame)]);
    }
    embeddings.valid_frames =
        std::count(mask.begin(), mask.end(), static_cast<int32_t>(1));
    auto output = runtime.adapt(embeddings, mask, true);
    if (output.tokens != embeddings.valid_frames ||
        output.hidden_size != 1024 || output.stages.size() != 5) {
      throw std::runtime_error("unexpected adaptor output contract");
    }
    actual_packed.insert(actual_packed.end(), output.values.begin(),
                         output.values.end());
    for (const auto &stage : output.stages) {
      auto &values = actual_stages[stage.name];
      values.insert(values.end(), stage.values.begin(), stage.values.end());
    }
    per_utterance.push_back(std::move(output));
  }

  const auto &expected_stages = reference.require("checkpoints");
  for (const char *name :
       {"packed_valid", "linear_1", "linear_2", "block_0", "block_1"}) {
    const auto found = actual_stages.find(name);
    if (found == actual_stages.end()) {
      throw std::runtime_error(std::string("missing adaptor stage: ") + name);
    }
    require_close(
        name, found->second,
        load_f32_slice(arguments.data, expected_stages.require(name)));
  }
  require_close(
      "output", actual_packed,
      load_f32_slice(arguments.data, expected_stages.require("packed_valid")));

  engine::models::fun_asr_nano::FunAsrNanoEncoderEmbeddings padded;
  padded.values.assign(input_values.begin() + 4 * 512,
                       input_values.begin() + 8 * 512);
  padded.frames = 4;
  padded.valid_frames = 2;
  padded.hidden_size = 512;
  std::fill(padded.values.begin() + 2 * 512, padded.values.end(), -9999.0F);
  const std::vector<int32_t> padded_mask = {1, 1, 0, 0};
  const auto changed_padding = runtime.adapt(padded, padded_mask, false);
  if (changed_padding.values != per_utterance[1].values) {
    throw std::runtime_error("adaptor output depends on padded frame values");
  }
  const auto cached = runtime.adapt(padded, padded_mask, false);
  if (cached.values != changed_padding.values) {
    throw std::runtime_error("cached adaptor graph changed its output");
  }

  require_runtime_error("non-contiguous mask", "right-padded", [&] {
    static_cast<void>(runtime.adapt(padded, {1, 0, 1, 0}, false));
  });
  require_runtime_error("mask size", "mask size", [&] {
    static_cast<void>(runtime.adapt(padded, {1, 1}, false));
  });
  auto wrong_hidden = padded;
  wrong_hidden.hidden_size = 256;
  require_runtime_error("hidden size", "shape", [&] {
    static_cast<void>(runtime.adapt(wrong_hidden, padded_mask, false));
  });
  auto wrong_values = padded;
  wrong_values.values.pop_back();
  require_runtime_error("value count", "value count", [&] {
    static_cast<void>(runtime.adapt(wrong_values, padded_mask, false));
  });

  std::cout << "Fun-ASR-Nano adaptor parity passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fun_asr_nano_adaptor_probe: " << error.what() << '\n';
  return 1;
}
