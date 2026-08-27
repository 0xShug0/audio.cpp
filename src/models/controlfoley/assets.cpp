#include "engine/models/controlfoley/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::controlfoley {
namespace {

std::filesystem::path resolve_gguf_path(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        if (model_path.extension() != ".gguf") {
            throw std::runtime_error("ControlFoley requires a GGUF model file");
        }
        return std::filesystem::weakly_canonical(model_path);
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("ControlFoley model path does not exist: " + model_path.string());
    }
    const auto found = engine::assets::find_directory_gguf(model_path);
    if (!found.has_value()) {
        throw std::runtime_error("ControlFoley model directory must contain exactly one GGUF or model.gguf");
    }
    return std::filesystem::weakly_canonical(*found);
}

void validate_required_tensors(const engine::assets::TensorSource & source) {
    engine::assets::require_tensor_shape(source, "flow/latent_mean", {1, 1, 40});
    engine::assets::require_tensor_shape(source, "flow/latent_std", {1, 1, 40});
    engine::assets::require_tensor_shape(source, "flow/empty_string_feat", {77, 1024});
    engine::assets::require_tensor_shape(source, "flow/empty_clip_feat", {1, 1024});
    engine::assets::require_tensor_shape(source, "flow/empty_visual_feat", {1, 768});
    engine::assets::require_tensor_shape(source, "flow/empty_sync_feat", {1, 768});
    engine::assets::require_tensor_shape(source, "flow/empty_audio_feat", {1, 512});
    engine::assets::require_tensor_shape(source, "flow/empty_timbre_feat", {1, 1536});
    engine::assets::require_tensor_shape(source, "vae/decoder.conv_in.weight", {2048, 40, 3});
    engine::assets::require_tensor_shape(source, "bigvgan/conv_pre.weight_v", {1536, 128, 7});
}

}  // namespace

std::shared_ptr<const ControlFoleyAssets> load_controlfoley_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<ControlFoleyAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "controlfoley");
    assets->model_root = assets->resources.model_root();
    assets->gguf_path = resolve_gguf_path(model_path);
    assets->weights = engine::assets::open_tensor_source(assets->gguf_path);
    validate_required_tensors(*assets->weights);
    assets->flow_weights = assets->resources.open_tensor_source("flow");
    assets->vae_weights = assets->resources.open_tensor_source("vae");
    assets->bigvgan_weights = assets->resources.open_tensor_source("bigvgan");
    assets->open_clip_weights = assets->resources.open_tensor_source("open_clip");
    assets->cav_mae_weights = assets->resources.open_tensor_source("cav_mae");
    assets->synchformer_weights = assets->resources.open_tensor_source("synchformer");
    assets->clap_weights = assets->resources.open_tensor_source("clap");
    assets->mert_weights = assets->resources.open_tensor_source("mert");
    assets->musicgen_style_weights = assets->resources.open_tensor_source("musicgen_style");
    return assets;
}

}  // namespace engine::models::controlfoley
