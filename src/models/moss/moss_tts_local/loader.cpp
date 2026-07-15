#include "engine/models/moss/moss_tts_local/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/moss/moss_tts_local/assets.h"
#include "engine/models/moss/moss_tts_local/session.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_tts_local {
namespace {

runtime::CapabilitySet moss_tts_local_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
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
    return metadata;
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
        if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) {
            throw std::runtime_error("MOSS-TTS-Local only supports the Tts and VoiceCloning tasks");
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
            (void) engine::assets::load_resource_bundle_from_package_spec(
                request.model_path,
                engine::assets::default_model_package_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_moss_tts_local_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = moss_tts_local_metadata();
        inspection.capabilities = moss_tts_local_capabilities();
        inspection.cli.request_options = {
            {"max_tokens", "n", "Maximum generated audio frames."},
            {"do_sample", "true|false", "Enable stochastic audio-token sampling."},
            {"temperature", "float", "Audio-token sampling temperature."},
            {"top_p", "float", "Audio-token nucleus sampling limit."},
            {"top_k", "n", "Audio-token top-k sampling limit."},
            {"repetition_penalty", "float", "Audio-token repetition penalty."},
            {"text_temperature", "float", "Text-gate sampling temperature."},
            {"text_top_p", "float", "Text-gate nucleus sampling limit."},
            {"text_top_k", "n", "Text-gate top-k sampling limit."},
            {"text_chunk_mode", "default|tag_aware|japanese|endline", "Long-form text chunking mode; default splitter."},
        };
        inspection.cli.session_options = {
            {"moss_tts_local.weight_type", "auto|native|f32|f16|bf16|q8_0", "Backbone weight storage type."},
            {"moss_tts_local.reference_cache_slots", "n", "Prepared reference-voice cache slots; default 1, set 0 to disable."},
        };
        const auto spec_path = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return std::make_unique<MossTTSLocalLoadedModel>(
            moss_tts_local_metadata(), moss_tts_local_capabilities(), load_moss_tts_local_assets(request.model_path));
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_local_loader() {
    return std::make_shared<MossTTSLocalLoader>();
}

}  // namespace engine::models::moss_tts_local
