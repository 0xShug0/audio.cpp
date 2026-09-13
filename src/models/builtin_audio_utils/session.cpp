#include "engine/models/builtin_audio_utils/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/deepfilternet2.h"
#include "engine/framework/audio/flashsr.h"
#include "engine/framework/audio/gtcrn.h"
#include "engine/framework/audio/rnnoise.h"
#include "engine/framework/audio/utility_api.h"
#include "engine/framework/audio/zipenhancer.h"
#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace engine::models::builtin_audio_utils {
namespace {

using UtilityRuntime = std::variant<
    engine::audio::DeepFilterNet2Model,
    engine::audio::RnnoiseModel,
    engine::audio::ZipEnhancerModel,
    engine::audio::GTCRNModel,
    engine::audio::FlashSrModel>;

runtime::CapabilitySet builtin_audio_utils_capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::SpeechToSpeech, {runtime::RunMode::Offline}},
    };
    return out;
}

runtime::ModelMetadata builtin_audio_utils_metadata(const std::string & model_id) {
    const auto info = engine::audio::require_builtin_audio_utility(model_id);
    runtime::ModelMetadata out;
    out.family = "builtin_audio_utils";
    out.variant = model_id;
    out.description = info.kind == engine::audio::BuiltinAudioUtilityKind::SuperResolve
        ? "Built-in audio utility: super resolution."
        : "Built-in audio utility: denoise/enhance.";
    return out;
}

std::string operation_name(engine::audio::BuiltinAudioUtilityKind kind) {
    switch (kind) {
    case engine::audio::BuiltinAudioUtilityKind::Denoise:
        return "denoise";
    case engine::audio::BuiltinAudioUtilityKind::SuperResolve:
        return "super_resolve";
    }
    return "unknown";
}

UtilityRuntime load_utility_runtime(
    const std::string & model_id,
    const engine::audio::AudioUtilityPaths & paths) {
    const auto asset = engine::audio::resolve_builtin_audio_utility_asset(paths, model_id);
    if (model_id == "deepfilternet2") {
        return engine::audio::DeepFilterNet2Model::load_from_directory(asset, paths.backend);
    }
    if (model_id == "rnnoise") {
        return engine::audio::RnnoiseModel::load_from_safetensors(asset, paths.backend);
    }
    if (model_id == "zipenhancer") {
        return engine::audio::ZipEnhancerModel::load_from_directory(asset, paths.backend);
    }
    if (model_id == "gtcrn" || model_id == "gtcrn_streaming" ||
        model_id == "gtcrn_dns3" || model_id == "gtcrn_vctk") {
        return engine::audio::GTCRNModel::load_from_safetensors(asset, paths.backend);
    }
    if (model_id == "flashsr") {
        return engine::audio::FlashSrModel::load_from_directory(asset, paths.backend);
    }
    (void) engine::audio::require_builtin_audio_utility(model_id);
    throw std::runtime_error("unreachable builtin audio utility model: " + model_id);
}

runtime::AudioBuffer audio_buffer_from_mono(
    int sample_rate,
    std::vector<float> samples) {
    runtime::AudioBuffer output;
    output.sample_rate = sample_rate;
    output.channels = 1;
    output.samples = std::move(samples);
    return output;
}

class BuiltinAudioUtilsSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    BuiltinAudioUtilsSession(
        runtime::TaskSpec task,
        const runtime::SessionOptions & options,
        std::string model_id)
        : runtime::RuntimeSessionBase(options),
          task_(task),
          model_id_(std::move(model_id)),
          info_(engine::audio::require_builtin_audio_utility(model_id_)),
          runtime_(load_utility_runtime(
              model_id_,
              engine::audio::AudioUtilityPaths{
                  engine::audio::default_audio_utility_assets_root(),
                  options.backend})) {
        if (task_.task != runtime::VoiceTaskKind::SpeechToSpeech) {
            throw std::runtime_error("builtin_audio_utils only supports --task s2s");
        }
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("builtin_audio_utils only supports offline mode");
        }
    }

    std::string family() const override {
        return "builtin_audio_utils";
    }

    runtime::VoiceTaskKind task_kind() const override {
        return task_.task;
    }

    runtime::RunMode run_mode() const override {
        return task_.mode;
    }

    void prepare(const runtime::SessionPreparationRequest &) override {
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("builtin_audio_utils run");
        if (!request.audio_input.has_value()) {
            throw std::runtime_error("builtin_audio_utils requires --audio");
        }
        const auto & input = *request.audio_input;
        const auto prep_start = std::chrono::steady_clock::now();
        const auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            input.samples,
            input.sample_rate,
            input.channels,
            info_.input_sample_rate);
        engine::debug::timing_log_scalar(
            "builtin_audio_utils.audio_prepare_ms",
            engine::debug::elapsed_ms(prep_start));

        const auto compute_start = std::chrono::steady_clock::now();
        runtime::TaskResult result;
        if (model_id_ == "deepfilternet2") {
            const auto output = std::get<engine::audio::DeepFilterNet2Model>(runtime_).run_mono_48k(mono);
            result.audio_output = audio_buffer_from_mono(output.sample_rate, output.samples);
        } else if (model_id_ == "rnnoise") {
            const auto output = std::get<engine::audio::RnnoiseModel>(runtime_).process_mono_48k(mono);
            result.audio_output = audio_buffer_from_mono(output.sample_rate, output.samples);
        } else if (model_id_ == "zipenhancer") {
            const auto output = std::get<engine::audio::ZipEnhancerModel>(runtime_).denoise_mono_16k(mono);
            result.audio_output = audio_buffer_from_mono(output.sample_rate, output.samples);
        } else if (model_id_ == "gtcrn" || model_id_ == "gtcrn_streaming" ||
                   model_id_ == "gtcrn_dns3" || model_id_ == "gtcrn_vctk") {
            const auto output = std::get<engine::audio::GTCRNModel>(runtime_).denoise_mono_16k(mono);
            result.audio_output = audio_buffer_from_mono(output.sample_rate, output.samples);
        } else if (model_id_ == "flashsr") {
            const auto output = std::get<engine::audio::FlashSrModel>(runtime_).super_resolve_mono_16k(mono);
            result.audio_output = audio_buffer_from_mono(output.sample_rate, output.samples);
        } else {
            (void) engine::audio::require_builtin_audio_utility(model_id_);
        }
        engine::debug::timing_log_scalar(
            "builtin_audio_utils." + operation_name(info_.kind) + "_ms",
            engine::debug::elapsed_ms(compute_start));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::string model_id_;
    engine::audio::BuiltinAudioUtilityInfo info_;
    UtilityRuntime runtime_;
};

class BuiltinAudioUtilsLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "builtin_audio_utils";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return builtin_audio_utils_capabilities();
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (!request.family_hint.has_value() || *request.family_hint != family()) {
            return false;
        }
        return engine::audio::find_builtin_audio_utility(request.model_path.generic_string()).has_value();
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto model_id = request.model_path.generic_string();
        const auto info = engine::audio::require_builtin_audio_utility(model_id);
        runtime::ModelInspection inspection;
        inspection.model_root = engine::audio::default_audio_utility_assets_root();
        inspection.metadata = builtin_audio_utils_metadata(model_id);
        inspection.capabilities = builtin_audio_utils_capabilities();
        inspection.discovered_weights.push_back(runtime::NamedAsset{
            model_id,
            engine::audio::resolve_builtin_audio_utility_asset(
                engine::audio::AudioUtilityPaths{engine::audio::default_audio_utility_assets_root()},
                model_id)});
        inspection.cli.request_options.push_back(runtime::CliOptionInfo{
            "audio",
            "wav",
            "Input WAV audio to process.",
            true});
        inspection.cli.request_options.push_back(runtime::CliOptionInfo{
            "out",
            "wav",
            "Output WAV path.",
            true});
        inspection.cli.load_options.push_back(runtime::CliOptionInfo{
            "operation",
            operation_name(info.kind),
            "Selected built-in audio utility operation.",
            false,
            operation_name(info.kind)});
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return std::make_unique<BuiltinAudioUtilsLoadedModel>(request.model_path.generic_string());
    }
};

}  // namespace

BuiltinAudioUtilsLoadedModel::BuiltinAudioUtilsLoadedModel(std::string model_id)
    : model_id_(std::move(model_id)),
      metadata_(builtin_audio_utils_metadata(model_id_)),
      capabilities_(builtin_audio_utils_capabilities()) {
    (void) engine::audio::require_builtin_audio_utility(model_id_);
}

const runtime::ModelMetadata & BuiltinAudioUtilsLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & BuiltinAudioUtilsLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> BuiltinAudioUtilsLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<BuiltinAudioUtilsSession>(task, options, model_id_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_builtin_audio_utils_loader() {
    return std::make_shared<BuiltinAudioUtilsLoader>();
}

}  // namespace engine::models::builtin_audio_utils
