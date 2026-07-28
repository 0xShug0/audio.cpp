#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::runtime {

template <typename Assets>
struct SpecBackedVoiceModelConfig {
    std::string family;
    std::function<std::shared_ptr<const Assets>(const std::filesystem::path &)> load_assets;
    std::function<std::unique_ptr<IVoiceTaskSession>(
        const TaskSpec &,
        const SessionOptions &,
        std::shared_ptr<const Assets>,
        std::shared_ptr<const engine::model_spec::ModelContract>)> create_session;
};

inline std::shared_ptr<const engine::model_spec::ModelContract> require_model_contract(std::string family) {
    auto out = engine::model_spec::model_contract(family);
    if (!out.has_value()) {
        throw std::runtime_error(family + " requires a schema v1 model contract");
    }
    return std::make_shared<engine::model_spec::ModelContract>(std::move(*out));
}

template <typename Assets>
class SpecBackedLoadedVoiceModel final : public ILoadedVoiceModel {
public:
    explicit SpecBackedLoadedVoiceModel(
        SpecBackedVoiceModelConfig<Assets> config,
        std::shared_ptr<const engine::model_spec::ModelContract> contract,
        std::shared_ptr<const Assets> assets)
        : config_(std::move(config)),
          contract_(std::move(contract)),
          assets_(std::move(assets)) {
        if (contract_ == nullptr) {
            throw std::invalid_argument(config_.family + " loaded model requires a model contract");
        }
        if (assets_ == nullptr) {
            throw std::invalid_argument(config_.family + " loaded model requires assets");
        }
    }

    const ModelMetadata & metadata() const noexcept override {
        return contract_->metadata;
    }

    const CapabilitySet & capabilities() const noexcept override {
        return contract_->capabilities;
    }

    std::unique_ptr<IVoiceTaskSession> create_task_session(
        const TaskSpec & task,
        const SessionOptions & options) const override {
        return config_.create_session(task, options, assets_, contract_);
    }

private:
    SpecBackedVoiceModelConfig<Assets> config_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const Assets> assets_;
};

template <typename Assets>
class SpecBackedVoiceModelLoader final : public IVoiceModelLoader {
public:
    explicit SpecBackedVoiceModelLoader(SpecBackedVoiceModelConfig<Assets> config)
        : config_(std::move(config)) {
        if (config_.family.empty()) {
            throw std::invalid_argument("spec-backed model loader requires a family");
        }
        if (!config_.load_assets) {
            throw std::invalid_argument(config_.family + " spec-backed model loader requires an asset loader");
        }
        if (!config_.create_session) {
            throw std::invalid_argument(config_.family + " spec-backed model loader requires a session factory");
        }
    }

    std::string family() const override {
        return config_.family;
    }

    CapabilitySet advertised_capabilities() const override {
        return require_model_contract(config_.family)->capabilities;
    }

    bool can_load(const ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != config_.family) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle_for_family(request.model_path, config_.family);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    ModelInspection inspect(const ModelLoadRequest & request) const override {
        const auto assets = config_.load_assets(request.model_path);
        const auto contract = require_model_contract(config_.family);
        const auto package_spec = engine::model_spec::default_package_spec_path(config_.family);

        ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = contract->metadata;
        inspection.capabilities = contract->capabilities;
        inspection.cli = contract->cli;
        inspection.discovered_configs = discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const override {
        auto assets = config_.load_assets(request.model_path);
        auto contract = require_model_contract(config_.family);
        return std::make_unique<SpecBackedLoadedVoiceModel<Assets>>(
            config_,
            std::move(contract),
            std::move(assets));
    }

private:
    SpecBackedVoiceModelConfig<Assets> config_;
};

template <typename Assets>
std::shared_ptr<IVoiceModelLoader> make_spec_backed_voice_loader(SpecBackedVoiceModelConfig<Assets> config) {
    return std::make_shared<SpecBackedVoiceModelLoader<Assets>>(std::move(config));
}

}  // namespace engine::runtime
