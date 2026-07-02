#include "engine/models/chatterbox/loader.h"

#include "engine/models/chatterbox/session.h"
#include "engine/models/chatterbox/text_tokenizer.h"

#include "engine/framework/io/filesystem.h"

#include <memory>
#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("Chatterbox expects a model directory: " + model_path.string());
    }
    return std::filesystem::weakly_canonical(model_path);
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(
        resolve_model_root(request.model_path),
        {"tokenizer.json", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(
        resolve_model_root(request.model_path),
        {
            "ve.safetensors",
            "t3_cfg.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "t3_mtl23ls_v3.safetensors",
            "s3gen.safetensors",
            "conds.pt",
        });
}

class ChatterboxLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "chatterbox";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            const auto assets = resolve_chatterbox_assets(root);
            (void) assets;
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto root = resolve_model_root(request.model_path);
        (void) resolve_chatterbox_assets(root);
        runtime::ModelInspection inspection;
        inspection.model_root = root;
        inspection.metadata.family = family();
        inspection.metadata.variant = root.filename().string();
        inspection.metadata.description = "Chatterbox voice cloning loaded from local assets.";
        inspection.metadata.config_candidates = {
            "tokenizer.json",
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json",
        };
        inspection.metadata.weight_candidates = {
            "ve.safetensors",
            "t3_cfg.safetensors",
            "t3_mtl23ls_v3.safetensors",
            "s3gen.safetensors",
            "conds.pt",
        };
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.languages = supported_chatterbox_language_codes();
        inspection.capabilities.supports_speaker_reference = true;
        inspection.capabilities.supports_style_condition = true;
        inspection.cli.session_options = {
            {
                "--session-option chatterbox.conditionals_cache_slots",
                "n",
                "Prepared voice-condition cache slots; default 1, set 0 to disable.",
            },
            {
                "--session-option chatterbox.mem_saver=true",
                "",
                "Free non-conditional runtime graphs after each request chunk; default false.",
            },
        };
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        const auto root = resolve_model_root(request.model_path);
        return load_chatterbox_model(root);
    }
};

}  // namespace

ChatterboxLoadedModel::ChatterboxLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ChatterboxAssetPaths> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Chatterbox loaded model requires assets");
    }
}

const runtime::ModelMetadata & ChatterboxLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ChatterboxLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ChatterboxLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("Chatterbox only supports VoiceTaskKind::VoiceCloning");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Chatterbox only supports offline mode");
    }
    return std::make_unique<ChatterboxSession>(task, options, assets_);
}

std::unique_ptr<ChatterboxLoadedModel> load_chatterbox_model(const std::filesystem::path & model_root) {
    const auto root = resolve_model_root(model_root);
    auto assets = std::make_shared<ChatterboxAssetPaths>(resolve_chatterbox_assets(root));

    runtime::ModelMetadata metadata;
    metadata.family = "chatterbox";
    metadata.variant = root.filename().string();
    metadata.description = "Chatterbox voice cloning loaded from local assets.";
    metadata.config_candidates = {
        "tokenizer.json",
        "grapheme_mtl_merged_expanded_v1.json",
        "Cangjie5_TC.json",
    };
    metadata.weight_candidates = {
        "ve.safetensors",
        "t3_cfg.safetensors",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.safetensors",
        "conds.pt",
    };

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    capabilities.languages = supported_chatterbox_language_codes();
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;

    return std::make_unique<ChatterboxLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_chatterbox_loader() {
    return std::make_shared<ChatterboxLoader>();
}

}  // namespace engine::models::chatterbox
