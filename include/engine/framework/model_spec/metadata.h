#pragma once

#include "engine/framework/runtime/model.h"

#include <optional>
#include <string_view>

namespace engine::model_spec {

[[nodiscard]] std::optional<runtime::CapabilitySet> advertised_capabilities(std::string_view family);
[[nodiscard]] std::optional<runtime::ModelCliInterface> cli_interface(std::string_view family);

}  // namespace engine::model_spec
