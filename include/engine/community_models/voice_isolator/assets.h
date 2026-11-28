#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::voice_isolator {

struct Assets {
    assets::ResourceBundle resources;
    // The model path when it names one checkpoint file of a safetensors directory.
    std::filesystem::path selected_file;
};

// One checkpoint: the architecture it describes itself with, and its tensors.
struct Checkpoint {
    std::string config_json;
    std::shared_ptr<const assets::TensorSource> tensors;
};

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path);

// A GGUF holds one checkpoint. A safetensors directory can hold the offline and
// the realtime one: a model path naming a file picks it, otherwise streaming
// takes the realtime checkpoint and offline prefers the offline one.
Checkpoint open_checkpoint(const Assets & assets, bool streaming);

}  // namespace engine::models::voice_isolator
