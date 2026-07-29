#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/decoder.h"
#include "engine/models/fun_asr_nano/prompt.h"
#include "engine/models/fun_asr_nano/tokenizer_text.h"

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
#include <vector>

namespace {

constexpr size_t kPrefillGraphArenaBytes = 256 * 1024 * 1024;
constexpr size_t kDecodeGraphArenaBytes = 128 * 1024 * 1024;
constexpr size_t kWeightContextBytes = 32 * 1024 * 1024;
constexpr float kAbsoluteTolerance = 3.0e-3F;
constexpr float kRelativeTolerance = 3.0e-3F;
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
        throw std::runtime_error("unsupported decoder probe backend: " + value);
      }
    } else if (option == "--model") {
      result.model = value;
    } else if (option == "--reference") {
      result.reference = value;
    } else if (option == "--data") {
      result.data = value;
    } else {
      throw std::runtime_error("unknown decoder probe option: " + option);
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
    throw std::runtime_error("invalid decoder reference data slice");
  }
  std::ifstream input(data_path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(signed_offset * 4));
  std::vector<float> values(static_cast<size_t>(signed_count));
  for (float &value : values) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char *>(bytes), 4);
    if (!input) {
      throw std::runtime_error("truncated decoder reference data");
    }
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    std::memcpy(&value, &bits, sizeof(value));
  }
  return values;
}

void require_equal(const std::string &label, const std::vector<int32_t> &actual,
                   const std::vector<int64_t> &expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + " size mismatch");
  }
  for (size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      throw std::runtime_error(label + " mismatch at " + std::to_string(index));
    }
  }
}

int32_t argmax_index(const std::vector<float> &values) {
  if (values.empty()) {
    throw std::runtime_error("cannot select from empty decoder logits");
  }
  return static_cast<int32_t>(std::distance(
      values.begin(), std::max_element(values.begin(), values.end())));
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
      reference.require("text_weight_catalog_count").as_i64() != 310 ||
      reference.require("data_format").as_string() != "little-endian-float32") {
    throw std::runtime_error("unexpected decoder reference metadata");
  }

  auto assets =
      engine::models::fun_asr_nano::load_fun_asr_nano_assets(arguments.model);
  engine::models::fun_asr_nano::FunAsrNanoTextTokenizer tokenizer(assets);
  if (tokenizer.require_token_id("<|im_start|>") !=
          reference.require("im_start_token_id").as_i64() ||
      tokenizer.require_token_id("<|im_end|>") !=
          reference.require("im_end_token_id").as_i64() ||
      tokenizer.require_token_id("<|object_ref_start|>") !=
          reference.require("audio_token_id").as_i64()) {
    throw std::runtime_error(
        "Fun-ASR-Nano special token ids differ from reference");
  }

  engine::models::fun_asr_nano::FunAsrNanoPromptBuilder prompt_builder(assets);
  engine::models::fun_asr_nano::FunAsrNanoPromptRequest prompt_request;
  auto prompt = prompt_builder.build(prompt_request, 2);
  const auto &itn_reference = reference.require("itn");
  require_equal(
      "ITN prompt ids", prompt.input_ids,
      engine::io::json::require_i64_array(itn_reference, "input_ids"));
  require_equal("ITN audio positions", prompt.audio_token_positions,
                engine::io::json::require_i64_array(itn_reference,
                                                    "audio_token_positions"));

  prompt_request.enable_itn = false;
  const auto no_itn_prompt = prompt_builder.build(prompt_request, 2);
  const auto &no_itn_reference = reference.require("no_itn");
  require_equal(
      "non-ITN prompt ids", no_itn_prompt.input_ids,
      engine::io::json::require_i64_array(no_itn_reference, "input_ids"));
  require_equal("non-ITN audio positions", no_itn_prompt.audio_token_positions,
                engine::io::json::require_i64_array(no_itn_reference,
                                                    "audio_token_positions"));

  engine::models::fun_asr_nano::FunAsrNanoAdaptorEmbeddings audio_embeddings;
  audio_embeddings.values =
      load_f32_slice(arguments.data, reference.require("audio_embeddings"));
  audio_embeddings.tokens = 2;
  audio_embeddings.hidden_size = 1024;

  engine::core::ExecutionContext execution_context({arguments.backend, 0, 4});
  engine::models::fun_asr_nano::FunAsrNanoDecoderRuntime decoder(
      assets, execution_context, kPrefillGraphArenaBytes,
      kDecodeGraphArenaBytes, kWeightContextBytes,
      engine::assets::TensorStorageType::F32);
  engine::models::fun_asr_nano::FunAsrNanoGenerationOptions options;
  options.max_new_tokens = 3;
  options.capture_logits = true;
  const auto output = decoder.generate(prompt, audio_embeddings, options);
  const auto greedy_ids =
      engine::io::json::require_i64_array(reference, "greedy_token_ids");
  if (output.step_logits.size() != greedy_ids.size()) {
    throw std::runtime_error("decoder trace step count mismatch");
  }
  const auto &logit_references = reference.require("logits").as_array();
  if (logit_references.size() != output.step_logits.size()) {
    throw std::runtime_error("decoder reference logit count mismatch");
  }
  for (size_t step = 0; step < output.step_logits.size(); ++step) {
    require_close("decoder logits " + std::to_string(step),
                  output.step_logits[step],
                  load_f32_slice(arguments.data, logit_references[step]));
    if (argmax_index(output.step_logits[step]) != greedy_ids[step]) {
      throw std::runtime_error("decoder greedy token mismatch at step " +
                               std::to_string(step));
    }
  }
  std::vector<int64_t> expected_output(greedy_ids.begin(),
                                       greedy_ids.end() - 1);
  require_equal("generated non-EOS ids", output.token_ids, expected_output);
  if (tokenizer.decode(output.token_ids) !=
      reference.require("generated_text").as_string()) {
    throw std::runtime_error("decoder text differs from reference");
  }

  const auto cached = decoder.generate(prompt, audio_embeddings, options);
  if (cached.token_ids != output.token_ids ||
      cached.step_logits != output.step_logits) {
    throw std::runtime_error("cached decoder graphs changed their output");
  }

  auto wrong_audio = audio_embeddings;
  wrong_audio.tokens = 1;
  require_runtime_error("audio token count", "token count", [&] {
    static_cast<void>(decoder.generate(prompt, wrong_audio, options));
  });
  auto wrong_position = prompt;
  wrong_position.audio_token_positions[0] = 0;
  require_runtime_error("audio position", "cover every", [&] {
    static_cast<void>(
        decoder.generate(wrong_position, audio_embeddings, options));
  });
  auto duplicate_position = prompt;
  duplicate_position.audio_token_positions[1] =
      duplicate_position.audio_token_positions[0];
  require_runtime_error("duplicate audio position", "cover every", [&] {
    static_cast<void>(
        decoder.generate(duplicate_position, audio_embeddings, options));
  });
  auto wrong_mask = prompt;
  wrong_mask.attention_mask.pop_back();
  require_runtime_error("prompt mask", "unpadded prompt mask", [&] {
    static_cast<void>(decoder.generate(wrong_mask, audio_embeddings, options));
  });
  auto wrong_options = options;
  wrong_options.max_new_tokens = 0;
  require_runtime_error("generation length", "must be positive", [&] {
    static_cast<void>(
        decoder.generate(prompt, audio_embeddings, wrong_options));
  });
  wrong_options.max_new_tokens = std::numeric_limits<int64_t>::max();
  require_runtime_error("generation overflow", "max_position_embeddings", [&] {
    static_cast<void>(
        decoder.generate(prompt, audio_embeddings, wrong_options));
  });
  require_runtime_error("audio token length", "positive audio token", [&] {
    static_cast<void>(prompt_builder.build(prompt_request, 0));
  });

  std::cout << "Fun-ASR-Nano decoder parity passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fun_asr_nano_decoder_probe: " << error.what() << '\n';
  return 1;
}
