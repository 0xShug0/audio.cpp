#include "engine/models/audio_flamingo/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/audio_flamingo/audio_encoder.h"
#include "engine/models/audio_flamingo/decoder.h"
#include "engine/models/audio_flamingo/frontend.h"
#include "engine/models/audio_flamingo/projector.h"
#include "engine/models/audio_flamingo/tokenizer_text.h"

#include <chrono>
#include <stdexcept>

namespace engine::models::audio_flamingo {
namespace {
constexpr const char * kFamily = "audio_flamingo";
constexpr size_t kGraphBytes = 4 * 1024 * 1024;

class AudioFlamingoSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    AudioFlamingoSession(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const AudioFlamingoAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          tokenizer_(assets_), frontend_(assets_) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Audio Flamingo");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Audio Flamingo supports offline audio understanding through the ASR task");
        }
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"audio_flamingo.weight_type"}).value_or("native"));
        encoder_ = std::make_unique<AudioFlamingoAudioEncoderRuntime>(assets_, execution_context(), kGraphBytes, storage);
        projector_ = std::make_unique<AudioFlamingoAudioProjectorRuntime>(assets_, execution_context(), kGraphBytes, storage);
        decoder_ = std::make_unique<AudioFlamingoQwen2DecoderRuntime>(assets_, execution_context(), storage);
        assets_->model_weights->release_storage();
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio) {
            throw std::runtime_error("Audio Flamingo requires an audio preparation contract");
        }
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Audio Flamingo run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Audio Flamingo");
        if (!request.audio_input) {
            throw std::runtime_error("Audio Flamingo requires audio input");
        }
        AudioFlamingoGenerationOptions generation;
        generation.max_new_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, assets_->config.max_new_tokens);
        generation.repetition_penalty = runtime::parse_finite_float_option(request.options, {"repetition_penalty"}).value_or(1.0F);
        generation.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(1.0F);
        generation.top_p = runtime::parse_finite_float_option(request.options, {"top_p"}).value_or(1.0F);
        generation.top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(50);
        generation.seed = runtime::parse_int_option(request.options, {"seed"}).value_or(42);
        if (const auto sample = runtime::find_option(request.options, {"do_sample"})) {
            generation.do_sample = runtime::parse_bool_option(*sample, "do_sample");
        }
        const auto instruction = runtime::find_option(request.options, {"instruct"}).value_or(
            request.text_input && !request.text_input->text.empty() ? request.text_input->text : "Transcribe the input speech.");
        const auto started = std::chrono::steady_clock::now();
        auto stage = started;
        const auto features = frontend_.extract(*request.audio_input);
        debug::timing_log_scalar("audio_flamingo.frontend_ms", debug::elapsed_ms(stage));
        const auto prompt = tokenizer_.build_prompt(instruction, features);
        stage = std::chrono::steady_clock::now();
        const auto encoded = encoder_->encode(features);
        debug::timing_log_scalar("audio_flamingo.encoder_ms", debug::elapsed_ms(stage));
        stage = std::chrono::steady_clock::now();
        const auto projected = projector_->project(encoded, features, prompt);
        debug::timing_log_scalar("audio_flamingo.projector_ms", debug::elapsed_ms(stage));
        stage = std::chrono::steady_clock::now();
        const auto generated = decoder_->generate(prompt, projected, generation);
        debug::timing_log_scalar("audio_flamingo.decoder_ms", debug::elapsed_ms(stage));
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{tokenizer_.decode(generated.token_ids), ""};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const AudioFlamingoAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    AudioFlamingoTextTokenizer tokenizer_;
    AudioFlamingoFrontend frontend_;
    std::unique_ptr<AudioFlamingoAudioEncoderRuntime> encoder_;
    std::unique_ptr<AudioFlamingoAudioProjectorRuntime> projector_;
    std::unique_ptr<AudioFlamingoQwen2DecoderRuntime> decoder_;
};
}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_audio_flamingo_loader() {
    runtime::SpecBackedVoiceModelConfig<AudioFlamingoAssets> config;
    config.family = kFamily;
    config.load_assets = load_audio_flamingo_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const AudioFlamingoAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<AudioFlamingoSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace engine::models::audio_flamingo
