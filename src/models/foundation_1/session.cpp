#include "engine/models/foundation_1/pipeline.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>

namespace engine::models::foundation_1 {
namespace {
class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const FoundationAssets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "foundation_1", "Foundation-1");
        if (task.task != runtime::VoiceTaskKind::AudioGeneration || task.mode != runtime::RunMode::Offline)
            throw runtime::InvalidRequestError("Foundation-1 supports offline music generation");
        pipeline_ = std::make_unique<FoundationPipeline>(std::move(assets), execution_context(),
            assets::parse_tensor_storage_type(runtime::find_option(options.options,
                {"foundation_1.weight_type", "weight_type"}).value_or("native")));
    }
    std::string family() const override { return "foundation_1"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Foundation-1 run");
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays,
                *contract_, "Foundation-1");
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        if (!request.text_input || request.text_input->text.empty())
            throw runtime::InvalidRequestError("Foundation-1 requires a text prompt");
        GenerationOptions options;
        options.duration_sec = runtime::parse_finite_float_option(request.options, {"duration_sec"}).value_or(10);
        options.num_steps = runtime::parse_i64_option(request.options, {"num_steps"}).value_or(100);
        options.guidance_scale = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(7);
        const auto seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(42);
        if (options.duration_sec < .1f || options.duration_sec > 20 || options.num_steps < 1 ||
            options.guidance_scale < 0 || seed < 0)
            throw runtime::InvalidRequestError("Foundation-1 requires duration_sec in [0.1,20], num_steps>0, guidance_scale>=0 and seed>=0");
        options.seed = static_cast<uint64_t>(seed);
        options.negative_prompt = runtime::find_option(request.options, {"negative_prompt"}).value_or("");
        debug::timing_log_scalar("foundation_1.seed", static_cast<double>(options.seed));
        const auto started = std::chrono::steady_clock::now();
        runtime::TaskResult result;
        result.audio_output = pipeline_->generate(request.text_input->text, options);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }
private:
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<FoundationPipeline> pipeline_;
};
} // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_foundation_1_loader() {
    runtime::SpecBackedVoiceModelConfig<FoundationAssets> config;
    config.family = "foundation_1";
    config.load_assets = load_foundation_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const FoundationAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
} // namespace engine::models::foundation_1
