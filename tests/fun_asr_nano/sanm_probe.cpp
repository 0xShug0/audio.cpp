#include "engine/framework/core/backend.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/speech_encoders/sanm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 64 * 1024 * 1024;
constexpr size_t kGraphNodes = 4096;
constexpr float kCpuAbsoluteTolerance = 2.0e-4F;
constexpr float kCudaAbsoluteTolerance = 1.0e-3F;
constexpr float kRelativeTolerance = 2.0e-4F;
constexpr const char *kTransformersCommit =
    "48e7f65fb274172e15aa88875d780c67c37606c7";

struct Arguments {
  engine::core::BackendType backend = engine::core::BackendType::Cpu;
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
        throw std::runtime_error("unsupported SAN-M probe backend: " + value);
      }
    } else if (option == "--reference") {
      arguments.reference = value;
    } else if (option == "--data") {
      arguments.data = value;
    } else {
      throw std::runtime_error("unknown SAN-M probe option: " + option);
    }
  }
  if (arguments.reference.empty() || arguments.data.empty()) {
    throw std::runtime_error("--reference and --data are required");
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
    throw std::runtime_error("invalid SAN-M reference data slice");
  }
  const uint64_t offset = static_cast<uint64_t>(signed_offset);
  const size_t count = static_cast<size_t>(signed_count);
  if (offset >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) / 4) {
    throw std::runtime_error("SAN-M reference data offset is too large");
  }
  std::ifstream input(data_path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset * 4));
  if (!input) {
    throw std::runtime_error("could not seek in SAN-M reference data");
  }
  std::vector<float> values(count);
  for (float &value : values) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char *>(bytes), 4);
    if (!input) {
      throw std::runtime_error("truncated SAN-M reference data");
    }
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
  }
  return values;
}

engine::core::TensorShape
descriptor_shape(const engine::io::json::Value &descriptor) {
  const auto dimensions =
      engine::io::json::require_i64_array(descriptor, "shape");
  if (dimensions.empty() || dimensions.size() > engine::core::kMaxTensorRank) {
    throw std::runtime_error("SAN-M reference tensor rank is invalid");
  }
  engine::core::TensorShape shape;
  shape.rank = dimensions.size();
  for (size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] <= 0) {
      throw std::runtime_error(
          "SAN-M reference tensor dimension must be positive");
    }
    shape.dims[axis] = dimensions[axis];
  }
  return shape;
}

class GraphRunner {
public:
  explicit GraphRunner(engine::core::BackendType backend_type)
      : backend_(engine::core::init_backend({backend_type, 0, 4})) {
    if (backend_ == nullptr) {
      throw std::runtime_error("failed to initialize SAN-M probe backend");
    }
    engine::core::set_backend_threads(backend_, 4);
    ggml_init_params parameters{};
    parameters.mem_size = kGraphBytes;
    parameters.mem_buffer = nullptr;
    parameters.no_alloc = true;
    ggml_ = ggml_init(parameters);
    if (ggml_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize SAN-M probe graph context");
    }
    context_.ggml = ggml_;
    context_.module_instance_name = "fun_asr_nano_sanm_probe";
    context_.backend_type = backend_type;
  }

  ~GraphRunner() {
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
    }
    if (ggml_ != nullptr) {
      ggml_free(ggml_);
    }
    if (backend_ != nullptr) {
      ggml_backend_free(backend_);
    }
  }

  GraphRunner(const GraphRunner &) = delete;
  GraphRunner &operator=(const GraphRunner &) = delete;

  engine::core::ModuleBuildContext &context() { return context_; }

  engine::core::TensorValue make_f32(const engine::core::TensorShape &shape,
                                     std::vector<float> values) {
    if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
      throw std::runtime_error("SAN-M tensor data does not match its shape");
    }
    auto tensor = engine::core::make_tensor(context_, GGML_TYPE_F32, shape);
    pending_.push_back({tensor, std::move(values)});
    return tensor;
  }

  std::vector<float> run(const engine::core::TensorValue &output) {
    buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend_);
    if (buffer_ == nullptr) {
      throw std::runtime_error("failed to allocate SAN-M probe tensors");
    }
    for (const auto &pending : pending_) {
      engine::core::write_tensor_f32(pending.tensor, pending.values);
    }
    ggml_cgraph *graph = ggml_new_graph_custom(ggml_, kGraphNodes, false);
    ggml_build_forward_expand(graph, output.tensor);
    engine::core::validate_backend_graph_supported(backend_, graph,
                                                   "SAN-M probe");
    const auto status = engine::core::compute_backend_graph(
        backend_, graph, nullptr, "SAN-M probe");
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("SAN-M probe graph execution failed");
    }
    return engine::core::read_tensor_f32(output.tensor);
  }

private:
  struct PendingTensor {
    engine::core::TensorValue tensor;
    std::vector<float> values;
  };

  ggml_backend_t backend_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_context *ggml_ = nullptr;
  engine::core::ModuleBuildContext context_{};
  std::vector<PendingTensor> pending_;
};

class TensorLoader {
public:
  TensorLoader(GraphRunner &runner, std::filesystem::path data_path)
      : runner_(runner), data_path_(std::move(data_path)) {}

  engine::core::TensorValue load(const engine::io::json::Value &descriptor) {
    return runner_.make_f32(descriptor_shape(descriptor),
                            load_f32_slice(data_path_, descriptor));
  }

  engine::core::TensorValue load_named(const engine::io::json::Value &weights,
                                       const std::string &name) {
    return load(weights.require(name));
  }

  engine::modules::LinearWeights
  load_linear(const engine::io::json::Value &weights,
              const std::string &prefix) {
    return {
        load_named(weights, prefix + ".weight"),
        load_named(weights, prefix + ".bias"),
    };
  }

  engine::modules::NormWeights load_norm(const engine::io::json::Value &weights,
                                         const std::string &prefix) {
    return {
        load_named(weights, prefix + ".weight"),
        load_named(weights, prefix + ".bias"),
    };
  }

private:
  GraphRunner &runner_;
  std::filesystem::path data_path_;
};

engine::modules::SanmBlockWeightsView
load_weights(TensorLoader &loader, const engine::io::json::Value &weights) {
  return {
      loader.load_norm(weights, "self_attn_layer_norm"),
      loader.load_linear(weights, "self_attn.q_proj"),
      loader.load_linear(weights, "self_attn.k_proj"),
      loader.load_linear(weights, "self_attn.v_proj"),
      loader.load_linear(weights, "self_attn.out_proj"),
      loader.load_named(weights, "feedforward_sequential_memory.conv.weight"),
      loader.load_norm(weights, "final_layer_norm"),
      loader.load_linear(weights, "fc1"),
      loader.load_linear(weights, "fc2"),
  };
}

void require_close(const std::string &label, const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   engine::core::BackendType backend_type);

void run_layer_norm(const std::string &name,
                    const engine::io::json::Value &block,
                    const engine::io::json::Value &config_json,
                    const std::filesystem::path &data_path,
                    engine::core::BackendType backend_type) {
  GraphRunner runner(backend_type);
  TensorLoader loader(runner, data_path);
  const auto input = loader.load(block.require("input"));
  const auto weights =
      loader.load_norm(block.require("weights"), "self_attn_layer_norm");
  const auto output = engine::modules::sanm_layer_norm(
      runner.context(), input, weights,
      engine::io::json::require_f32(config_json, "layer_norm_eps"));
  const auto actual = runner.run(output);
  const auto expected = load_f32_slice(
      data_path, block.require("checkpoints").require("self_attn_layer_norm"));
  require_close(name + ".self_attn_layer_norm", actual, expected, backend_type);
}

void require_close(const std::string &label, const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   engine::core::BackendType backend_type) {
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
    const float absolute_tolerance =
        backend_type == engine::core::BackendType::Cuda ? kCudaAbsoluteTolerance
                                                        : kCpuAbsoluteTolerance;
    const float limit =
        absolute_tolerance + kRelativeTolerance * std::abs(expected[index]);
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

void run_block(const std::string &name, const engine::io::json::Value &block,
               const engine::io::json::Value &config_json,
               const std::filesystem::path &data_path,
               engine::core::BackendType backend_type) {
  GraphRunner runner(backend_type);
  TensorLoader loader(runner, data_path);
  const auto input = loader.load(block.require("input"));
  const auto weights = load_weights(loader, block.require("weights"));
  engine::modules::SanmBlockConfig config;
  config.input_size =
      name == "projection"
          ? engine::io::json::require_i64(config_json, "input_size")
          : engine::io::json::require_i64(config_json, "model_size");
  config.model_size = engine::io::json::require_i64(config_json, "model_size");
  config.num_heads = engine::io::json::require_i64(config_json, "num_heads");
  config.ffn_size = engine::io::json::require_i64(config_json, "ffn_size");
  config.fsmn_kernel_size =
      engine::io::json::require_i64(config_json, "fsmn_kernel_size");
  config.layer_norm_eps =
      engine::io::json::require_f32(config_json, "layer_norm_eps");
  config.attention_lowering =
      engine::modules::ScaledDotProductAttentionLowering::Explicit;

  const auto output = name == "projection"
                          ? engine::modules::sanm_projection_block(
                                runner.context(), input, weights, config)
                          : engine::modules::sanm_residual_block(
                                runner.context(), input, weights, config);
  const auto actual = runner.run(output);
  const auto expected = load_f32_slice(data_path, block.require("output"));
  require_close(name, actual, expected, backend_type);
}

void require_runtime_error(const std::string &label,
                           const std::string &expected_message,
                           const std::function<void()> &operation) {
  try {
    operation();
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find(expected_message) != std::string::npos) {
      return;
    }
    throw std::runtime_error(label +
                             " returned an unexpected error: " + error.what());
  }
  throw std::runtime_error(label + " did not reject the invalid configuration");
}

void run_config_contracts() {
  GraphRunner runner(engine::core::BackendType::Cpu);
  const auto input8 =
      runner.make_f32(engine::core::TensorShape::from_dims({1, 2, 8}),
                      std::vector<float>(16, 0.0F));
  const auto input7 =
      runner.make_f32(engine::core::TensorShape::from_dims({1, 2, 7}),
                      std::vector<float>(14, 0.0F));
  const engine::modules::SanmBlockWeightsView weights;
  engine::modules::SanmBlockConfig config{
      8, 8, 2, 16, 3,
  };

  auto bad_heads = config;
  bad_heads.num_heads = 3;
  require_runtime_error("SAN-M head divisibility", "divisible", [&] {
    static_cast<void>(engine::modules::sanm_projection_block(
        runner.context(), input8, weights, bad_heads));
  });

  auto even_kernel = config;
  even_kernel.fsmn_kernel_size = 4;
  require_runtime_error("SAN-M FSMN kernel", "odd", [&] {
    static_cast<void>(engine::modules::sanm_projection_block(
        runner.context(), input8, weights, even_kernel));
  });

  auto mismatched_residual = config;
  mismatched_residual.input_size = 7;
  require_runtime_error(
      "SAN-M residual dimensions", "input_size == model_size", [&] {
        static_cast<void>(engine::modules::sanm_residual_block(
            runner.context(), input7, weights, mismatched_residual));
      });
}

} // namespace

int main(int argc, char **argv) try {
  const auto arguments = parse_arguments(argc, argv);
  const auto reference = engine::io::json::parse_file(arguments.reference);
  if (reference.require("schema_version").as_i64() != 1 ||
      reference.require("transformers_commit").as_string() !=
          kTransformersCommit ||
      reference.require("data_format").as_string() != "little-endian-float32") {
    throw std::runtime_error("unexpected SAN-M reference metadata");
  }
  run_config_contracts();
  const auto &blocks = reference.require("blocks");
  const auto &config = reference.require("config");
  run_layer_norm("projection", blocks.require("projection"), config,
                 arguments.data, arguments.backend);
  run_layer_norm("residual", blocks.require("residual"), config, arguments.data,
                 arguments.backend);
  run_block("projection", blocks.require("projection"), config, arguments.data,
            arguments.backend);
  run_block("residual", blocks.require("residual"), config, arguments.data,
            arguments.backend);
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fun_asr_nano_sanm_probe failed: " << error.what() << '\n';
  return 1;
}
