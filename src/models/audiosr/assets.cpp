#include "engine/models/audiosr/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::audiosr {
namespace {

std::filesystem::path resolve_gguf_path(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        if (model_path.extension() != ".gguf") {
            throw std::runtime_error("AudioSR runtime requires a GGUF model file");
        }
        return std::filesystem::weakly_canonical(model_path);
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("AudioSR model path does not exist: " + model_path.string());
    }
    const auto found = engine::assets::find_directory_gguf(model_path);
    if (!found.has_value()) {
        throw std::runtime_error("AudioSR model directory must contain exactly one GGUF or model.gguf");
    }
    return std::filesystem::weakly_canonical(*found);
}

std::string variant_from_path(const std::filesystem::path & gguf_path) {
    const std::string name = gguf_path.filename().string();
    if (name.find("speech") != std::string::npos) {
        return "speech";
    }
    return "basic";
}

void validate_required_tensors(const engine::assets::TensorSource & source) {
    engine::assets::require_tensor_shape(source, "first_stage_model.encoder.conv_in.weight", {128, 1, 3, 3});
    engine::assets::require_tensor_shape(source, "first_stage_model.encoder.conv_out.weight", {32, 1024, 3, 3});
    engine::assets::require_tensor_shape(source, "first_stage_model.quant_conv.weight", {32, 32, 1, 1});
    engine::assets::require_tensor_shape(source, "first_stage_model.post_quant_conv.weight", {16, 16, 1, 1});
    engine::assets::require_tensor_shape(source, "first_stage_model.decoder.conv_in.weight", {1024, 16, 3, 3});
    engine::assets::require_tensor_shape(source, "first_stage_model.decoder.conv_out.weight", {1, 128, 3, 3});
    engine::assets::require_tensor_shape(source, "first_stage_model.vocoder.conv_pre.weight", {1536, 256, 7});
    engine::assets::require_tensor_shape(source, "first_stage_model.vocoder.conv_post.weight", {1, 48, 7});
    engine::assets::require_tensor_shape(source, "model.diffusion_model.input_blocks.0.0.weight", {128, 32, 3, 3});
    engine::assets::require_tensor_shape(source, "model.diffusion_model.time_embed.0.weight", {512, 128});
    engine::assets::require_tensor_shape(source, "model.diffusion_model.out.2.weight", {16, 128, 3, 3});
    engine::assets::require_tensor_shape(source, "cond_stage_models.0.vae.encoder.conv_in.weight", {128, 1, 3, 3});
}

}  // namespace

std::shared_ptr<const AudioSRAssets> load_audiosr_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<AudioSRAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "audiosr");
    assets->gguf_path = resolve_gguf_path(model_path);
    assets->model_root = assets->resources.model_root();
    assets->variant = variant_from_path(assets->gguf_path);
    assets->weights = assets->resources.open_tensor_source("weights");
    validate_required_tensors(*assets->weights);
    const auto scale_factor = assets->weights->require_f32("scale_factor");
    if (scale_factor.size() != 1) {
        throw std::runtime_error("AudioSR scale_factor tensor must contain one value");
    }
    assets->config.vae_scale_factor = scale_factor.front();
    assets->frontend_lowpass_sos = assets->weights->require_f32(
        "frontend/audiosr.frontend.lowpass_sos",
        {4, 1025, 4, 6});
    assets->frontend_lowpass_sos_valid = assets->weights->require_f32(
        "frontend/audiosr.frontend.lowpass_sos_valid",
        {1025});
    return assets;
}

}  // namespace engine::models::audiosr
