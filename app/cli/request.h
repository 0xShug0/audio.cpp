#pragma once

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace minitts::cli {

// Most artifact payload bytes one request may carry, inline and from files
// together: the server's default max_request_body_bytes (2 GiB).
inline constexpr uint64_t kMaxRequestArtifactBytes = 2ull * 1024ull * 1024ull * 1024ull;

// True when --audio selects raw PCM on stdin ("-") rather than a file path.
bool is_stdin_audio_source(std::string_view audio_arg);
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
// The request's `artifacts` array, in order, as TaskRequest::input_artifacts.
// Each entry has the shape the server writes a result's artifacts in: `id`,
// `kind`, `payload` (base64, or a data URI) and optional `meta` (string,
// number or boolean values). `path` may replace `payload`: the payload is then
// that file's bytes, resolved against base_dir. Throws
// engine::runtime::InvalidRequestError, which the server answers with 400.
std::vector<engine::runtime::VoiceArtifact> json_request_artifacts(
    const engine::io::json::Value * value,
    const std::filesystem::path & base_dir);
engine::runtime::TaskRequest build_request_from_json(
    const engine::io::json::Value & value,
    const std::filesystem::path & base_dir);
engine::runtime::TaskRequest build_request_from_cli(int argc, char ** argv);

}  // namespace minitts::cli
