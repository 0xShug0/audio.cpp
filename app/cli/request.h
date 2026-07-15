#pragma once

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace minitts::cli {

engine::runtime::AudioBuffer read_audio_buffer(const std::filesystem::path & path);
engine::runtime::AudioBuffer read_audio_buffer(std::istream & path);
engine::runtime::AudioBuffer read_audio_buffer(std::string_view input);
std::string json_option_string(const engine::io::json::Value & value);
std::unordered_map<std::string, std::string> json_options_map(const engine::io::json::Value * value);
std::optional<std::string> json_optional_string(
    const engine::io::json::Value & object,
    const std::string & key);
std::optional<float> json_optional_float(
    const engine::io::json::Value & object,
    const std::string & key);
engine::runtime::TaskRequest build_request_from_json(
    const engine::io::json::Value & value,
    const std::filesystem::path & base_dir);
engine::runtime::TaskRequest build_request_from_cli(int argc, char ** argv);

}  // namespace minitts::cli
