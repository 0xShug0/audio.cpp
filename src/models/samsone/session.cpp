#include "engine/models/samsone/session.h"

#include "engine/models/samsone/assets.h"
#include "engine/models/samsone/runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::samsone {
namespace {

constexpr const char * kFamily = "samsone";
constexpr const char * kDefaultInstruction = "describe the audio";

class SamsoneSession final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession {
public:
    SamsoneSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SamsoneAssets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(std::move(options)), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        if (task_.task != runtime::VoiceTaskKind::Asr || task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("SAMSONE supports offline audio understanding through the ASR task");
        }
        runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, "SAMSONE");
        runtime_ = std::make_unique<SamsoneRuntime>(assets_, execution_context());
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio.has_value()) {
            throw std::runtime_error("SAMSONE requires an audio preparation contract");
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("SAMSONE run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "SAMSONE");
        if (!request.audio_input.has_value()) {
            throw std::runtime_error("SAMSONE requires audio input");
        }
        const std::string instruction = runtime::find_option(request.options, {"instruct"}).value_or(kDefaultInstruction);
        const int64_t max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, 100);
        const auto started = std::chrono::steady_clock::now();
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{
            runtime_->generate(*request.audio_input, instruction, max_tokens, execution_context().config().threads),
            "en"};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const SamsoneAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<SamsoneRuntime> runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_samsone_loader() {
    runtime::SpecBackedVoiceModelConfig<SamsoneAssets> config;
    config.family = kFamily;
    config.load_assets = load_samsone_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const SamsoneAssets> assets,
                                std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<SamsoneSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::samsone
