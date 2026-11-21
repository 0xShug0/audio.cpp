#include "engine/community_models/voice_isolator/assets.h"

#include "engine/framework/io/safetensors.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::voice_isolator {
namespace {

Checkpoint open_safetensors(const Assets & assets, const char * id) {
    const auto & path = assets.resources.require_file(std::string(id) + "_checkpoint");
    const auto index = io::load_safetensors_index(path);
    const auto config = index.metadata.find("config");
    if (config == index.metadata.end()) {
        throw std::runtime_error("Voice Isolator checkpoint has no 'config' metadata: " + path.string());
    }
    return {config->second, assets.resources.open_tensor_source(id)};
}

}  // namespace

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<Assets>();
    assets->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("voice_isolator"));
    if (path.extension() == ".safetensors") {
        assets->selected_file = std::filesystem::weakly_canonical(path);
    }
    return assets;
}

Checkpoint open_checkpoint(const Assets & assets, bool streaming) {
    const auto & resources = assets.resources;
    if (resources.has_tensor_source("weights")) {
        return {resources.read_text("config"), resources.open_tensor_source("weights")};
    }
    const bool has_offline = resources.has_tensor_source("offline");
    const bool has_realtime = resources.has_tensor_source("realtime");
    if (!assets.selected_file.empty()) {
        for (const char * id : {"offline", "realtime"}) {
            const auto * file = resources.find_file(std::string(id) + "_checkpoint");
            if (file != nullptr && std::filesystem::weakly_canonical(*file) == assets.selected_file) {
                return open_safetensors(assets, id);
            }
        }
        throw std::runtime_error("Voice Isolator expects denoiser.safetensors or denoiser_realtime.safetensors, got " +
            assets.selected_file.filename().string());
    }
    if (streaming ? has_realtime : !has_offline && has_realtime) {
        return open_safetensors(assets, "realtime");
    }
    if (!streaming && has_offline) {
        return open_safetensors(assets, "offline");
    }
    throw std::runtime_error(streaming
        ? "Voice Isolator streaming needs denoiser_realtime.safetensors in " + resources.model_root().string()
        : "Voice Isolator found neither denoiser.safetensors nor denoiser_realtime.safetensors in " +
            resources.model_root().string());
}

}  // namespace engine::models::voice_isolator
