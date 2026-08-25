#include "engine/models/controlfoley/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/controlfoley/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::controlfoley {
namespace {

constexpr const char * kFamily = "controlfoley";

std::shared_ptr<const ControlFoleyAssets> require_assets(
    std::shared_ptr<const ControlFoleyAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("ControlFoley session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("ControlFoley session requires a model contract");
    }
    return contract;
}

engine::assets::TensorStorageType weight_type(const engine::runtime::SessionOptions & options) {
    return engine::runtime::parse_tensor_storage_option(
        options.options,
        "controlfoley.weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
            engine::assets::TensorStorageType::BF16,
            engine::assets::TensorStorageType::Q8_0,
        });
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> create_controlfoley_session(
    const engine::runtime::TaskSpec & task,
    const engine::runtime::SessionOptions & options,
    std::shared_ptr<const ControlFoleyAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<ControlFoleySession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

ControlFoleySession::ControlFoleySession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const ControlFoleyAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : engine::runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    engine::runtime::validate_spec_backed_session_options(options_, *contract_, kFamily, "ControlFoley");
    if (task_.task != engine::runtime::VoiceTaskKind::AudioGeneration ||
        task_.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("ControlFoley supports only offline gen");
    }
    execution_ = std::make_unique<engine::core::ExecutionContext>(options_.backend);
    pipeline_ = std::make_unique<ControlFoleyPipelineRuntime>(
        assets_,
        *execution_,
        weight_type(options_));
}

ControlFoleySession::~ControlFoleySession() = default;

std::string ControlFoleySession::family() const {
    return kFamily;
}

engine::runtime::VoiceTaskKind ControlFoleySession::task_kind() const {
    return task_.task;
}

engine::runtime::RunMode ControlFoleySession::run_mode() const {
    return task_.mode;
}

void ControlFoleySession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "ControlFoley");
    mark_prepared();
}

engine::runtime::TaskResult ControlFoleySession::run(const engine::runtime::TaskRequest & request) {
    require_prepared("ControlFoley run");
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "ControlFoley");
    engine::runtime::TaskResult result;
    result.audio_output = pipeline_->run(request);
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_controlfoley_loader() {
    engine::runtime::SpecBackedVoiceModelConfig<ControlFoleyAssets> config;
    config.family = kFamily;
    config.load_assets = load_controlfoley_assets;
    config.create_session = create_controlfoley_session;
    return engine::runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::controlfoley
