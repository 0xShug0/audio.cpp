#include "engine/community_models/moss_voicegen/loader.h"

#include "engine/community_models/moss_voicegen/session.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_voicegen {
namespace {

runtime::ModelMetadata metadata(const MossVoiceGenAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "moss_voicegen";
    out.variant = std::to_string(assets.config.num_codebooks) + "vq";
    out.description = "MOSS-VoiceGenerator: speech in a voice designed from a written instruction.";
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
    };
    // The voice comes from the instruction; there is no reference-audio path in this model.
    out.supports_speaker_reference = false;
    out.supports_style_condition = true;
    out.languages = {"en", "zh"};
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"instruct", "<text>", "Voice description: timbre, age, pace, emotion."},
        {"language", "English|Chinese", "Full language name; the model does not understand codes like 'en'."},
        {"seed", "<int>", "Reproduces a take exactly. Note that it does not carry a voice across different text."},
        {"temperature", "<float>", "Audio sampling temperature (default 1.5)."},
        {"top_p", "<float>", "Audio nucleus sampling cutoff (default 0.6)."},
        {"top_k", "<int>", "Audio top-k (default 50)."},
        {"repetition_penalty", "<float>", "Audio repetition penalty (default 1.1)."},
        {"text_chunk_size", "<int>", "Maximum character budget per chunk (default 200)."},
        {"text_chunk_mode", "default|endline|tag_aware", "Text chunking mode (default 'default')."},
    };
    out.session_options = {
        {"moss_voicegen.weight_type", "native|f32|bf16|q8_0",
         "Weight storage; default bf16. f16 is rejected because this backbone produces NaN in it."},
    };
    return out;
}

class MossVoiceGenLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moss_voicegen";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_moss_voicegen_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities();
        inspection.cli = cli();
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_moss_voicegen_model(request.model_path);
    }
};

}  // namespace

MossVoiceGenLoadedModel::MossVoiceGenLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MossVoiceGenAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MossVoiceGenLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MossVoiceGenLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MossVoiceGenLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MOSS-VoiceGenerator only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("MOSS-VoiceGenerator only supports the vdes task");
    }
    return std::make_unique<MossVoiceGenSession>(task, options, assets_);
}

std::unique_ptr<MossVoiceGenLoadedModel> load_moss_voicegen_model(const std::filesystem::path & model_path) {
    auto assets = load_moss_voicegen_assets(model_path);
    return std::make_unique<MossVoiceGenLoadedModel>(metadata(*assets), capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_voicegen_loader() {
    return std::make_shared<MossVoiceGenLoader>();
}

}  // namespace engine::models::moss_voicegen
