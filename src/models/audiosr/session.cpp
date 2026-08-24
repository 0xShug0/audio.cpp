#include "engine/models/audiosr/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/audiosr/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::audiosr {
namespace {

constexpr const char * kFamily = "audiosr";

std::shared_ptr<const AudioSRAssets> require_assets(std::shared_ptr<const AudioSRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("AudioSR session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType weight_type(const engine::runtime::SessionOptions & options) {
    return engine::runtime::parse_tensor_storage_option(
        options.options,
        "audiosr.weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
            engine::assets::TensorStorageType::BF16,
            engine::assets::TensorStorageType::Q8_0,
        });
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("AudioSR session requires a model contract");
    }
    return contract;
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> create_audiosr_session(
    const engine::runtime::TaskSpec & task,
    const engine::runtime::SessionOptions & options,
    std::shared_ptr<const AudioSRAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<AudioSRSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

AudioSRSession::AudioSRSession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const AudioSRAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : engine::runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    engine::runtime::validate_spec_backed_session_options(options_, *contract_, kFamily, "AudioSR");
    if (task_.task != engine::runtime::VoiceTaskKind::AudioGeneration ||
        task_.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("AudioSR supports only offline gen");
    }
    execution_ = std::make_unique<engine::core::ExecutionContext>(options_.backend);
    pipeline_ = std::make_unique<AudioSRPipelineRuntime>(
        assets_,
        *execution_,
        weight_type(options_));
}

AudioSRSession::~AudioSRSession() = default;

std::string AudioSRSession::family() const {
    return kFamily;
}

engine::runtime::VoiceTaskKind AudioSRSession::task_kind() const {
    return task_.task;
}

engine::runtime::RunMode AudioSRSession::run_mode() const {
    return task_.mode;
}

void AudioSRSession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "AudioSR");
    mark_prepared();
}

engine::runtime::TaskResult AudioSRSession::run(const engine::runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    require_prepared("AudioSR run");
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "AudioSR");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("AudioSR requires --audio input");
    }
    engine::runtime::TaskResult result;
    result.audio_output = pipeline_->run(*request.audio_input, parse_audiosr_options(request.options));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_audiosr_loader() {
    engine::runtime::SpecBackedVoiceModelConfig<AudioSRAssets> config;
    config.family = kFamily;
    config.load_assets = load_audiosr_assets;
    config.create_session = create_audiosr_session;
    return engine::runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::audiosr
