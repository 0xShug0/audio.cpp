#include "engine/community_models/f5_tts/session.h"

#include "engine/framework/runtime/spec_backed_model.h"

#include <stdexcept>
#include <utility>

namespace engine::models::f5_tts {
namespace {

constexpr const char * kFamily = "f5_tts";

}  // namespace

std::shared_ptr<const F5TTSAssets> load_f5_tts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<F5TTSAssets>();
    assets->resources = assets::ResourceBundle(model_path);
    // M0 scaffolding: weight loading arrives with the DiT/Vocos milestones.
    return assets;
}

F5TTSSession::F5TTSSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const F5TTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : task_kind_(task.task),
      run_mode_(task.mode),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {
    (void) options;
    if (assets_ == nullptr) {
        throw std::runtime_error("F5-TTS session requires assets");
    }
    if (contract_ == nullptr) {
        throw std::runtime_error("F5-TTS session requires a model contract");
    }
}

std::string F5TTSSession::family() const noexcept {
    return kFamily;
}

runtime::VoiceTaskKind F5TTSSession::task_kind() const noexcept {
    return task_kind_;
}

runtime::RunMode F5TTSSession::run_mode() const noexcept {
    return run_mode_;
}

void F5TTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value() && !request.text->language.empty()) {
        // F5/Habibi infer language from the reference prompt; keep the
        // transcript for the M3 inference milestone.
    }
}

runtime::TaskResult F5TTSSession::run(const runtime::TaskRequest & request) {
    (void) request;
    // M0 scaffolding: inference is intentionally not implemented yet. Fail
    // loudly rather than returning silence so callers never mistake stub
    // output for generated speech.
    throw std::runtime_error(
        "F5-TTS community port is scaffolding only: inference is not implemented yet "
        "(see the milestone plan in docs/community_models/f5_tts.md)");
}

std::shared_ptr<runtime::IVoiceModelLoader> make_f5_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<F5TTSAssets> config;
    config.family = std::string(kFamily);
    config.load_assets = load_f5_tts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const F5TTSAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<F5TTSSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::f5_tts
