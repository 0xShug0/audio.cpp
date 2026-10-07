#include "engine/community_models/whistle_asr/session.h"

#include "engine/framework/runtime/spec_backed_model.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::whistle_asr {
namespace {

runtime::SessionOptions require_cpu_offline(
    const runtime::TaskSpec & task, runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
        throw std::invalid_argument("Whistle currently supports offline ASR only");
    }
    if (options.backend.type != engine::core::BackendType::Cpu) {
        throw std::invalid_argument("Whistle currently supports CPU inference only");
    }
    if (!contract) {
        throw std::invalid_argument("Whistle needs a model specification");
    }
    runtime::validate_spec_backed_session_options(options, *contract, "whistle_asr", "Whistle");
    return options;
}

}  // namespace

WhistleAsrSession::WhistleAsrSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const WhistleAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(require_cpu_offline(task, options, contract)),
      task_(task),
      contract_(std::move(contract)),
      runtime_(std::move(assets), options.backend.threads) {}

std::string WhistleAsrSession::family() const { return "whistle_asr"; }
runtime::VoiceTaskKind WhistleAsrSession::task_kind() const { return task_.task; }
runtime::RunMode WhistleAsrSession::run_mode() const { return task_.mode; }

void WhistleAsrSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(
        request.options, request.option_arrays, *contract_, "Whistle");
    mark_prepared();
}

runtime::TaskResult WhistleAsrSession::run(const runtime::TaskRequest & request) {
    require_prepared("Whistle ASR run()");
    if (!request.audio_input) {
        throw std::invalid_argument("Whistle ASR requires audio input");
    }
    runtime::validate_spec_backed_request_options(
        request.options, request.option_arrays, *contract_, "Whistle");
    std::string language = request.text_input ? request.text_input->language : std::string();
    if (const auto hint = request.options.find("language"); hint != request.options.end()) {
        if (!language.empty() && language != hint->second) {
            throw std::invalid_argument("Whistle request contains conflicting language hints");
        }
        language = hint->second;
    }
    const auto transcript = runtime_.transcribe(*request.audio_input, language);
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{transcript.text, transcript.language};
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_whistle_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<WhistleAssets> config;
    config.family = "whistle_asr";
    config.load_assets = [](const std::filesystem::path & path) {
        return load_whistle_assets(path);
    };
    config.create_session = [](const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const WhistleAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<WhistleAsrSession>(
            task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::whistle_asr
