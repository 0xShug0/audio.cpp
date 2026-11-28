#pragma once

#include "engine/framework/assets/resource_bundle.h"

namespace engine::models::realtime_reuse {

struct Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
};

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path);

} // namespace engine::models::realtime_reuse
