#include "engine/community_models/moss_ttsd/loader.h"

#include "engine/community_models/moss_ttsd/session.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_ttsd {
namespace {

runtime::ModelMetadata metadata(const Assets & assets) {
    runtime::ModelMetadata out;
    out.family = "moss_ttsd";
    out.variant = std::to_string(assets.config.num_codebooks) + "vq";
    out.description = "MOSS-TTSD: 8B delay-pattern dialogue TTS with multi-speaker voice cloning.";
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    // Several speakers in one take is what this checkpoint is for -- the text is
    // tagged [S1]/[S2] and each tag can carry its own reference recording -- and
    // that is declared as `multi_speaker` in the model spec, where the capability
    // vocabulary lives. CapabilitySet carries no such field and this is not the
    // place to add one for a single family.
    out.languages = {"en", "zh"};
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"voice_samples", "path[,path...]",
         "Per-speaker reference WAVs, positional: the first is [S1], the second [S2]. Leave an entry empty to let that speaker be invented rather than cloned."},
        {"reference_text", "<text>",
         "What the reference recordings say, speaker-tagged. Prepended to the text so the continuation carries on from a transcript it has already spoken."},
        {"language", "English|Chinese", "Full language name; the model does not understand codes like 'en'."},
        {"instruct", "<text>", "Free-text direction for the delivery."},
        {"seed", "<int>", "Reproduces a take exactly."},
        {"temperature", "<float>", "Audio sampling temperature (default 1.5)."},
        {"top_p", "<float>", "Audio nucleus sampling cutoff (default 0.6)."},
        {"top_k", "<int>", "Audio top-k (default 50)."},
        {"repetition_penalty", "<float>", "Audio repetition penalty (default 1.1)."},
        {"max_frames", "<int>", "Ceiling on generated codec frames at 12.5/s. 0 derives one from the text length."},
    };
    out.session_options = {
        {"moss_ttsd.weight_type", "native|f32|bf16|q8_0",
         "Weight storage; default native, which keeps a quantised package quantised."},
    };
    return out;
}

class Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moss_ttsd";
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
        const auto assets = load_moss_ttsd_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities();
        inspection.cli = cli();
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec, engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec, engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_moss_ttsd_model(request.model_path);
    }
};

}  // namespace

LoadedModel::LoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Assets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & LoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & LoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> LoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MOSS-TTSD only supports offline sessions");
    }
    // One path serves both: cloning is tts with reference recordings attached.
    if (task.task != runtime::VoiceTaskKind::Tts
        && task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("MOSS-TTSD supports the tts and clone tasks");
    }
    return std::make_unique<MossTtsdSession>(task, options, assets_);
}

std::unique_ptr<LoadedModel> load_moss_ttsd_model(const std::filesystem::path & model_path) {
    auto assets = load_moss_ttsd_assets(model_path);
    return std::make_unique<LoadedModel>(metadata(*assets), capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_ttsd_loader() {
    return std::make_shared<Loader>();
}

}  // namespace engine::models::moss_ttsd
