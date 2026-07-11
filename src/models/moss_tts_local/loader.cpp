#include "engine/models/moss_tts_local/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/moss_tts_local/assets.h"
#include "engine/models/moss_tts_local/session.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("MOSS-TTS-Local model path does not exist: " + model_path.string());
}

bool has_moss_tts_local_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json") &&
        engine::io::is_existing_file(root / "model.safetensors") &&
        engine::io::is_existing_file(root / "tokenizer.json");
}

runtime::CapabilitySet moss_tts_local_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"Auto"};
    capabilities.supports_speaker_reference = true;
    return capabilities;
}

runtime::ModelMetadata moss_tts_local_metadata() {
    runtime::ModelMetadata metadata;
    metadata.family = "moss_tts_local";
    metadata.variant = "MOSS-TTS-Local-Transformer-v1.5";
    metadata.description = "MOSS-TTS Local Transformer with the MOSS-Audio-Tokenizer-v2 codec.";
    metadata.config_candidates = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    };
    metadata.weight_candidates = {"model.safetensors"};
    return metadata;
}

std::vector<runtime::NamedAsset> discover_config_assets(const std::filesystem::path & root) {
    return runtime::discover_named_assets(
        root,
        {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        });
}

std::vector<runtime::NamedAsset> discover_weight_assets(const std::filesystem::path & root) {
    return runtime::discover_named_assets(root, {"model.safetensors"});
}

class MossTTSLocalLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MossTTSLocalLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MossTTSLocalAssets> assets)
        : metadata_(std::move(metadata)),
          capabilities_(std::move(capabilities)),
          assets_(std::move(assets)) {}

    const runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        if (task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("MOSS-TTS-Local only supports offline sessions");
        }
        if (task.task != runtime::VoiceTaskKind::Tts) {
            throw std::runtime_error("MOSS-TTS-Local only supports the Tts task");
        }
        return std::make_unique<MossTTSLocalSession>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MossTTSLocalAssets> assets_;
};

class MossTTSLocalLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moss_tts_local";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_moss_tts_local_assets(root) &&
                (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto root = resolve_model_root(request.model_path);
        const auto config = load_moss_tts_local_config(root);
        (void) config;
        runtime::ModelInspection inspection;
        inspection.model_root = root;
        inspection.metadata = moss_tts_local_metadata();
        inspection.capabilities = moss_tts_local_capabilities();
        inspection.cli.request_options = {
            {"voice_ref", "path", "Reference speaker WAV for zero-shot voice cloning."},
            {"reference_text", "text", "Transcript of the reference audio when known."},
            {"language", "code", "Optional language tag hint for multilingual synthesis."},
            {
                "moss_tts_local.reference_cache_slots",
                "n",
                "Prepared reference-voice cache slots; default 1, set 0 to disable.",
            },
        };
        inspection.discovered_configs = discover_config_assets(root);
        inspection.discovered_weights = discover_weight_assets(root);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        const auto root = resolve_model_root(request.model_path);
        return std::make_unique<MossTTSLocalLoadedModel>(
            moss_tts_local_metadata(), moss_tts_local_capabilities(), load_moss_tts_local_assets(root));
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_local_loader() {
    return std::make_shared<MossTTSLocalLoader>();
}

}  // namespace engine::models::moss_tts_local
