#include "engine/models/smart_turn/session.h"

#include "engine/models/smart_turn/runtime.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace engine::models::smart_turn {
namespace {

struct Assets {
    assets::ResourceBundle resources;
};

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options,
            std::shared_ptr<const assets::TensorSource> source,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), contract_(std::move(contract)),
          runtime_(std::move(source), execution_context()) {}

    std::string family() const override { return "smart_turn"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::TurnDetection; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Smart Turn");
        if (!request.audio) throw std::runtime_error("Smart Turn preparation requires audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Smart Turn run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Smart Turn");
        if (!request.audio_input) throw std::runtime_error("Smart Turn requires audio_input");
        const auto & audio = *request.audio_input;
        if (audio.sample_rate != 16000 || audio.channels != 1) {
            throw std::runtime_error("Smart Turn requires 16000 Hz mono audio");
        }
        const float threshold = runtime::parse_finite_float_option(request.options, {"threshold"}).value_or(0.5f);
        if (threshold < 0.0f || threshold > 1.0f) {
            throw std::runtime_error("Smart Turn threshold must be between 0 and 1");
        }
        const auto started = std::chrono::steady_clock::now();
        const float probability = runtime_.completion_probability(audio.samples);
        runtime::TaskResult result;
        result.custom_schema_output = runtime::CustomSchemaOutput{
            "smart_turn.v1",
            io::json::Value::make_object({
                {"complete", io::json::Value::make_bool(probability > threshold)},
                {"probability", io::json::Value::make_number(probability)},
            }),
        };
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    std::shared_ptr<const model_spec::ModelContract> contract_;
    SmartTurnWhisperTinyRuntime runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_smart_turn_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "smart_turn";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<Assets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "smart_turn");
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::TurnDetection || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Smart Turn supports offline turn detection");
        }
        runtime::validate_spec_backed_session_options(options, *contract, "smart_turn", "Smart Turn");
        return std::make_unique<Session>(options, assets->resources.open_tensor_source("weights"), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::smart_turn
