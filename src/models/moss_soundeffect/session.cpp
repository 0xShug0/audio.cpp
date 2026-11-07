#include "engine/models/moss_soundeffect/session.h"

#include "engine/models/moss_soundeffect/assets.h"
#include "engine/models/moss_soundeffect/dit.h"
#include "engine/models/moss_soundeffect/vae.h"
#include "engine/models/moss_soundeffect/text_encoder.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/noise.h"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace engine::models::moss_soundeffect {
namespace {
constexpr const char * kFamily = "moss_soundeffect";

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
            std::shared_ptr<const Assets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "MOSS-SoundEffect v2");
        core::set_backend_round_bf16_fusion(execution_context().backend(), true);
        if (task.task != runtime::VoiceTaskKind::AudioGeneration || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("MOSS-SoundEffect v2 supports offline sound-effect generation");
        attention_ = core::parse_attention_preference(
            runtime::find_option(options.options, {"moss_soundeffect.attention"}).value_or("auto"),
            "moss_soundeffect.attention");
        storage_ = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"moss_soundeffect.weight_type"}).value_or("native"));
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest &) override {
        if (!text_) {
            auto & execution = execution_context();
            auto text = std::make_unique<TextEncoderRuntime>(execution, assets_, storage_);
            auto dit = std::make_unique<DitRuntime>(execution, assets_, storage_, attention_);
            auto vae = std::make_unique<VaeRuntime>(execution, assets_);
            text_ = std::move(text);
            dit_ = std::move(dit);
            vae_ = std::move(vae);
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("MOSS-SoundEffect v2 generation");
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "MOSS-SoundEffect v2");
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        if (!request.text_input || request.text_input->text.empty())
            throw runtime::InvalidRequestError("MOSS-SoundEffect v2 requires a text prompt");
        const auto start = std::chrono::steady_clock::now();
        if (!request.option_arrays.empty())
            throw runtime::InvalidRequestError("MOSS-SoundEffect v2 does not accept list-valued request options");
        const auto duration_text = runtime::find_option(request.options, {"duration_sec"}).value_or("10");
        // Use shared option validation, retaining the original double for decimal rounding.
        runtime::parse_positive_finite_float_option(request.options, {"duration_sec"});
        const double requested_duration = std::stod(duration_text);
        // Format the double directly: rounding an intermediate float changes decimal ties.
        std::ostringstream duration_suffix;
        duration_suffix << std::fixed << std::setprecision(1) << requested_duration;
        const double duration = std::stod(duration_suffix.str());
        if (duration < 0.1 || duration > assets_->max_duration_sec)
            throw runtime::InvalidRequestError("MOSS-SoundEffect v2 duration_sec must be between 0.1 and 30");
        const int steps = runtime::parse_int_option(request.options, {"num_steps"}).value_or(100);
        const float guidance = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(4.0F);
        const float shift = runtime::parse_positive_finite_float_option(request.options, {"sigma_shift"}).value_or(5.0F);
        const auto seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(0);
        if (steps < 1) throw runtime::InvalidRequestError("MOSS-SoundEffect v2 num_steps must be positive");
        auto prompt = request.text_input->text;
        const auto append = runtime::find_option(request.options, {"append_duration_suffix"});
        if (!append || runtime::parse_bool_option(*append, "append_duration_suffix")) {
            prompt += " duration: " + duration_suffix.str() + "s";
        }
        const auto text_start = std::chrono::steady_clock::now();
        const auto positive = text_->encode(prompt);
        const auto negative = text_->encode(runtime::find_option(request.options, {"negative_prompt"}).value_or(""));
        debug::timing_log_scalar("moss_soundeffect.text.total_ms", debug::elapsed_ms(text_start));
        dit_->prepare_context(positive, negative);
        const auto sample_start = std::chrono::steady_clock::now();
        const int64_t frames = int64_t(assets_->sample_rate) * assets_->max_duration_sec / assets_->vae.hop_length;
        auto noise = sampling::generate_normal_noise(static_cast<size_t>(assets_->dit.latent_dim * frames), seed);
        core::round_f32_to_bf16_in_place(noise);
        const auto latent = dit_->sample(noise, steps, guidance, shift);
        debug::timing_log_scalar("moss_soundeffect.dit.sample_ms", debug::elapsed_ms(sample_start));
        auto audio = vae_->decode(latent, static_cast<size_t>(duration * assets_->sample_rate));
        debug::timing_log_scalar("audio.duration_sec", duration);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    core::AttentionPreference attention_ = core::AttentionPreference::Auto;
    assets::TensorStorageType storage_ = assets::TensorStorageType::Native;
    std::unique_ptr<TextEncoderRuntime> text_;
    std::unique_ptr<DitRuntime> dit_;
    std::unique_ptr<VaeRuntime> vae_;
};
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_soundeffect_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = kFamily;
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}
