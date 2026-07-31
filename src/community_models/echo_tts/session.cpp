#include "engine/community_models/echo_tts/session.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::echo_tts {
namespace {

constexpr const char * kFamily = "echo_tts";
constexpr int kSampleRate = 44100;

struct EchoTtsAssets {
    assets::ResourceBundle resources;
};

std::shared_ptr<const EchoTtsAssets> load_echo_tts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<EchoTtsAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(
        model_path,
        kFamily);
    return assets;
}

}  // namespace

EchoTtsSession::EchoTtsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(std::move(options)),
      task_(task),
      contract_(std::move(contract)) {
    if (contract_ == nullptr) {
        throw std::runtime_error("Echo-TTS session requires a model contract");
    }
    if (task_.task != runtime::VoiceTaskKind::VoiceCloning ||
        task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Echo-TTS only supports offline voice cloning");
    }
}

EchoTtsSession::~EchoTtsSession() = default;

std::string EchoTtsSession::family() const { return kFamily; }
runtime::VoiceTaskKind EchoTtsSession::task_kind() const { return task_.task; }
runtime::RunMode EchoTtsSession::run_mode() const { return task_.mode; }

void EchoTtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

runtime::TaskResult EchoTtsSession::run(const runtime::TaskRequest & request) {
    (void)request;
    require_prepared("Echo-TTS run");

    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        kSampleRate,
        1,
        std::vector<float>(kSampleRate, 0.0F),
    };
    return result;
}

void EchoTtsSession::reset() {}

std::shared_ptr<runtime::IVoiceModelLoader> make_echo_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<EchoTtsAssets> config;
    config.family = kFamily;
    config.load_assets = load_echo_tts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const EchoTtsAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        (void)assets;
        return std::make_unique<EchoTtsSession>(
            task,
            options,
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::echo_tts
