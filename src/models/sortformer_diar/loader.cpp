#include "engine/models/sortformer_diar/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/sortformer_diar/session.h"

#include <stdexcept>

namespace engine::models::sortformer_diar {

namespace {

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Diarization, {runtime::RunMode::Offline}},
    };
    out.supports_timestamps = true;
    return out;
}

runtime::ModelMetadata metadata(const SortformerAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "sortformer_diar";
    out.variant = assets.model_config.variant;
    out.description = "Sortformer offline diarization model loaded from local assets.";
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.session_options = {
        {"sortformer_diar.graph_context_mb", "n", "Inference graph arena size."},
        {"sortformer_diar.weight_context_mb", "n", "Weight context size."},
        {"sortformer_diar.weight_type", "native|f32|f16|bf16|q8_0", "All weight storage type; default f32."},
        {"sortformer_diar.matmul_weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type; defaults to sortformer_diar.weight_type."},
        {"sortformer_diar.conv_weight_type", "native|f32|f16|bf16|q8_0", "Convolution weight storage type; defaults to sortformer_diar.weight_type."},
    };
    return out;
}

class SortformerDiarLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "sortformer_diar";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Diarization, {runtime::RunMode::Offline}},
        };
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_sortformer_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities();
        inspection.cli = cli();
        const auto spec_path = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_sortformer_diar_model(request.model_path);
    }
};

}  // namespace

SortformerDiarLoadedModel::SortformerDiarLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const SortformerAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & SortformerDiarLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & SortformerDiarLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> SortformerDiarLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Diarization) {
        throw std::runtime_error("Sortformer diar only supports VoiceTaskKind::Diarization");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Sortformer diar only supports offline sessions");
    }
    return std::make_unique<SortformerDiarSession>(task, options, assets_);
}

std::unique_ptr<SortformerDiarLoadedModel> load_sortformer_diar_model(const std::filesystem::path & model_path) {
    auto assets = load_sortformer_assets(model_path);
    return std::make_unique<SortformerDiarLoadedModel>(metadata(*assets), capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sortformer_diar_loader() {
    return std::make_shared<SortformerDiarLoader>();
}

}  // namespace engine::models::sortformer_diar
