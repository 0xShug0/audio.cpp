#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"

namespace engine::community_models::parakeet_tdt {

// Reads the original bark-levels-v3 container. Decoding is bounded to one tensor;
// no expanded checkpoint, Python interpreter or inference backend is required.
std::shared_ptr<const assets::TensorSource> open_bark_packed_source(const std::filesystem::path & path);
bool is_bark_packed_package(const std::filesystem::path & path);
std::shared_ptr<const struct ParakeetTDTAssets> load_bark_packed_assets(const std::filesystem::path & path);
std::shared_ptr<runtime::IVoiceModelLoader> wrap_bark_packed_loader(std::shared_ptr<runtime::IVoiceModelLoader> loader);

} // namespace engine::community_models::parakeet_tdt
