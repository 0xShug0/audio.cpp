#include "engine/models/vocalrender/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/voxcpm2/assets.h"
#include "engine/models/voxcpm2/loader.h"

#include <stdexcept>
#include <utility>

namespace engine::models::vocalrender {
namespace {

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {{runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}}};
    out.languages = {"zh"};
    out.supports_speaker_reference = true;
    return out;
}

runtime::ModelMetadata metadata(const voxcpm2::VoxCPM2Assets & assets) {
    runtime::ModelMetadata out;
    out.family = "vocalrender";
    out.variant = "VocalRender score-conditioned singing synthesis";
    out.description = "VocalRender renders interleaved lyrics, MIDI pitches, note values, tempo, and a 2-8 second singing reference into 48 kHz singing audio.";
    (void)assets;
    return out;
}

std::shared_ptr<const voxcpm2::VoxCPM2Assets> load_assets(
    const std::filesystem::path & model_path) {
    return voxcpm2::load_voxcpm2_assets(model_path, "vocalrender");
}

class VocalRenderLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "vocalrender"; }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    std::string advertised_instructions_policy() const override {
        return "none";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void)engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path("vocalrender"));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_assets(request.model_path);
        runtime::ModelInspection out;
        out.model_root = assets->resources.model_root();
        out.metadata = metadata(*assets);
        out.capabilities = capabilities();
        out.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            engine::model_spec::default_spec_path("vocalrender"),
            engine::model_spec::ResourceKind::Files);
        out.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            engine::model_spec::default_spec_path("vocalrender"),
            engine::model_spec::ResourceKind::Tensors);
        return out;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request) const override {
        auto assets = load_assets(request.model_path);
        auto model_metadata = metadata(*assets);
        auto model_capabilities = capabilities();
        return std::make_unique<voxcpm2::VoxCPM2LoadedModel>(
            std::move(model_metadata), std::move(model_capabilities), std::move(assets));
    }
};

} // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_vocalrender_loader() {
    return std::make_shared<VocalRenderLoader>();
}

} // namespace engine::models::vocalrender
