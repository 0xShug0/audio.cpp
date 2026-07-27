#include "engine/models/confucius4_tts/loader.h"

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/model_spec/package.h"
#include "engine/models/confucius4_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::confucius4_tts {
namespace {

constexpr const char * kFamily = "confucius4_tts";

std::shared_ptr<const engine::model_spec::ModelContract> contract() {
    auto out = engine::model_spec::model_contract(kFamily);
    if (!out.has_value()) {
        throw std::runtime_error("Confucius4-TTS requires a schema v1 model contract");
    }
    return std::make_shared<engine::model_spec::ModelContract>(std::move(*out));
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Confucius4-TTS loaded model requires a model contract");
    }
    return contract;
}

class ConfuciusLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return kFamily;
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return contract()->capabilities;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle_for_family(request.model_path, family());
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_confucius_assets(request.model_path);
        const auto model_contract = contract();
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = model_contract->metadata;
        inspection.capabilities = model_contract->capabilities;
        inspection.cli = model_contract->cli;
        const auto package_spec = engine::model_spec::default_package_spec_path(family());
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
        return load_confucius_model(request.model_path);
    }
};

}  // namespace

ConfuciusLoadedModel::ConfuciusLoadedModel(
    std::shared_ptr<const engine::model_spec::ModelContract> contract,
    std::shared_ptr<const ConfuciusAssets> assets)
    : contract_(require_contract(std::move(contract))),
      metadata_(contract_->metadata),
      capabilities_(contract_->capabilities),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & ConfuciusLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ConfuciusLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ConfuciusLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<ConfuciusSession>(task, options, assets_, contract_);
}

std::unique_ptr<ConfuciusLoadedModel> load_confucius_model(const std::filesystem::path & model_path) {
    auto assets = load_confucius_assets(model_path);
    return std::make_unique<ConfuciusLoadedModel>(
        contract(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader() {
    return std::make_shared<ConfuciusLoader>();
}

}  // namespace engine::models::confucius4_tts
