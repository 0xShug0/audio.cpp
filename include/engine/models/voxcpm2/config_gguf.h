#pragma once

#include "engine/models/voxcpm2/assets.h"
#include "engine/framework/assets/tensor_source.h"

#include <memory>
#include <optional>
#include <string>

namespace engine::models::voxcpm2 {

// Load VoxCPM1 config from GGUF metadata
VoxCPM2Config load_voxcpm1_config_from_gguf(const engine::assets::TensorSource & source);

// Check if GGUF has VoxCPM1 config metadata
bool has_voxcpm1_config_metadata(const engine::assets::TensorSource & source);

}  // namespace engine::models::voxcpm2