#pragma once

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::core {
enum class BackendType;
}

namespace engine::runtime {

struct OptionMatch {
    std::string key;
    std::string value;
};

struct ComponentExecPlan {
    std::string impl;
    std::optional<core::BackendType> backend;
    std::unordered_map<std::string, std::string> artifacts;
    io::json::Value::Object params;
};

struct ComponentExecImplementation {
    std::string_view name;
    std::initializer_list<core::BackendType> backends;
    std::initializer_list<std::string_view> required_artifacts;
    std::initializer_list<std::string_view> optional_artifacts;
    std::initializer_list<std::string_view> params;
};

struct ComponentExecSpec {
    std::string_view name;
    std::string_view default_impl;
    std::initializer_list<ComponentExecImplementation> implementations;
};

// Parses one family.exec_plan JSON object. Omitted components/fields retain their
// defaults. Parameter values retain JSON types for model-local semantic validation.
std::unordered_map<std::string, ComponentExecPlan> parse_exec_plan(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    std::initializer_list<ComponentExecSpec> components);

// The compiled engines one `exec_plan` value declares: every artifact string, comma-separated
// ladders included, resolved against `model_root` the way a sidecar resolves it
// (`rknn::resolve_engine_path`) and priced by the target runtime, which reports 0 when it cannot
// say. A family uses this to answer IVoiceModelLoader::model_resources without re-deriving its
// component vocabulary, so plan parsing lives in one place: an absent option, a malformed plan and
// a non-string artifact all contribute nothing.
std::vector<ModelEngineRequirement> exec_plan_engine_requirements(
    const std::string & exec_plan_value,
    const std::filesystem::path & model_root);

std::optional<OptionMatch> find_option_match(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<std::string> find_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

bool parse_bool_option(const std::string & value, std::string_view key);

std::optional<int> parse_int_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<float> parse_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<float> parse_finite_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<float> parse_positive_finite_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<uint32_t> parse_u32_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<uint64_t> parse_u64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

std::optional<int64_t> parse_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys);

int64_t parse_positive_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    int64_t fallback);

size_t parse_size_mb_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    size_t fallback_bytes);

assets::TensorStorageType parse_tensor_storage_option(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    assets::TensorStorageType fallback,
    std::initializer_list<assets::TensorStorageType> allowed);

assets::TensorStorageType parse_tensor_storage_option(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view key,
    std::string_view fallback_key,
    assets::TensorStorageType fallback,
    std::initializer_list<assets::TensorStorageType> allowed);

uint32_t random_u32_seed();
uint64_t random_u64_seed();

}  // namespace engine::runtime
