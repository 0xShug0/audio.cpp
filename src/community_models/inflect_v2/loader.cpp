#include "engine/community_models/inflect_v2/loader.h"

#include "engine/community_models/inflect_v2/session.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::inflect_v2 {
namespace {

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.languages = {"en"};
    return out;
}

runtime::ModelMetadata metadata(const InflectV2Assets & assets) {
    runtime::ModelMetadata out;
    out.family = "inflect_v2";
    out.variant = assets.config.variant;
    out.description =
        "Community Inflect " + assets.config.variant +
        " English VITS speech synthesis with native GGML execution.";
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"speaking_rate", "0.5..2.0", "Speech speed multiplier; default 1.0."},
        {"variation", "0.0..1.0", "Acoustic latent variation; default 0.667."},
        {"seed", "n", "Acoustic noise seed; default 0."},
        {"text_chunk_mode", "word_budget", "Inflect punctuation-aware long-form chunking."},
        {"text_chunk_size", "n", "Maximum Unicode codepoints per chunk; default 280."},
    };
    out.session_options = {
        {"inflect_v2.espeak_library_path", "path", "Optional path to the eSpeak-ng shared library."},
        {"inflect_v2.espeak_data_path", "path", "Optional path to the eSpeak-ng data directory."},
    };
    return out;
}

class InflectV2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "inflect_v2"; }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void)model_spec::load_resource_bundle(
                request.model_path,
                model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(
        const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_inflect_v2_assets(request.model_path);
        runtime::ModelInspection out;
        out.model_root = assets->resources.model_root();
        out.metadata = metadata(*assets);
        out.capabilities = capabilities();
        out.cli = cli();
        const auto spec = model_spec::default_spec_path(family());
        out.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec,
            model_spec::ResourceKind::Files);
        out.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec,
            model_spec::ResourceKind::Tensors);
        return out;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request) const override {
        return load_inflect_v2_model(request.model_path);
    }
};

}  // namespace

InflectV2LoadedModel::InflectV2LoadedModel(
    runtime::ModelMetadata metadata_in,
    runtime::CapabilitySet capabilities_in,
    std::shared_ptr<const InflectV2Assets> assets)
    : metadata_(std::move(metadata_in)),
      capabilities_(std::move(capabilities_in)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Inflect v2 loaded model requires assets");
    }
}

const runtime::ModelMetadata & InflectV2LoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & InflectV2LoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
InflectV2LoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<InflectV2Session>(task, options, assets_);
}

std::unique_ptr<InflectV2LoadedModel> load_inflect_v2_model(
    const std::filesystem::path & model_path) {
    auto assets = load_inflect_v2_assets(model_path);
    return std::make_unique<InflectV2LoadedModel>(
        metadata(*assets),
        capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_inflect_v2_loader() {
    return std::make_shared<InflectV2Loader>();
}

}  // namespace engine::models::inflect_v2
