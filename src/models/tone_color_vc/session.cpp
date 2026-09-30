#include "engine/models/tone_color_vc/session.h"
#include "engine/models/tone_color_vc/runtime.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <random>
#include <stdexcept>

namespace engine::models::tone_color_vc {
namespace {

struct Assets {
    assets::ResourceBundle resources;
};

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options, std::shared_ptr<const assets::TensorSource> source,
            std::shared_ptr<const model_spec::ModelContract> contract, const io::json::Value & config)
        : RuntimeSessionBase(options), contract_(std::move(contract)),
          runtime_(std::move(source), execution_context(), options.backend, config) {}

    std::string family() const override { return "tone_color_vc"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::VoiceConversion; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Tone Color VC");
        if (!request.audio) throw std::runtime_error("Tone Color VC requires source audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Tone Color VC run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Tone Color VC");
        if (!request.audio_input) throw std::runtime_error("Tone Color VC requires --audio");
        if (!request.voice || !request.voice->speaker || !request.voice->speaker->audio)
            throw std::runtime_error("Tone Color VC requires --voice-ref target speaker audio");
        const auto started = std::chrono::steady_clock::now();
        const auto temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(0.3f);
        const auto seed = runtime::parse_int_option(request.options, {"seed"}).value_or(1234);
        if (temperature < 0 || seed < -1) throw std::runtime_error("Invalid Tone Color VC temperature or seed");
        runtime::TaskResult result;
        result.audio_output = runtime_.convert(*request.audio_input, *request.voice->speaker->audio,
            temperature, seed == -1 ? std::random_device{}() : static_cast<uint32_t>(seed));
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    std::shared_ptr<const model_spec::ModelContract> contract_;
    ToneColorRuntime runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_tone_color_vc_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "tone_color_vc";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<Assets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "tone_color_vc");
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::VoiceConversion || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("Tone Color VC supports offline voice conversion");
        runtime::validate_spec_backed_session_options(options, *contract, "tone_color_vc", "Tone Color VC");
        return std::make_unique<Session>(options, assets->resources.open_tensor_source("weights"), std::move(contract),
            assets->resources.parse_json("config"));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::tone_color_vc
