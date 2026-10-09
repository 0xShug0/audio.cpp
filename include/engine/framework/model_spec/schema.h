#pragma once

#include "engine/framework/io/json.h"

#include <optional>
#include <string_view>

namespace engine::model_spec {

// Current typed schema. Numeric 2 is the integrator contract.
// Numeric 1 remains valid and rejects schema 2 fields.
inline constexpr int kModelSpecSchemaVersion = 2;
inline constexpr int kModelSpecSchemaVersionV1 = 1;

void validate_spec(const engine::io::json::Value & spec, std::string_view source_name);

// Method, path, encoding, and response type for a closed operation id.
// Empty when the id is not one of the operations schema 2 may name.
[[nodiscard]] std::optional<engine::io::json::Value> operation_surface(std::string_view operation);

// Event envelope for an operation whose task sets stream to true.
// Empty when that operation has no streaming response.
[[nodiscard]] std::optional<engine::io::json::Value> stream_response_surface(std::string_view operation);

}  // namespace engine::model_spec
