#include "engine/community_models/soulx_flashhead/session.h"
#include "engine/community_models/soulx_flashhead/pipeline.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/debug/profiler.h"

namespace engine::community_models::soulx_flashhead {
namespace {
constexpr const char * kFamily = "soulx_flashhead";
bool resolve_vae_cudnn(const runtime::SessionOptions & options, const Assets & assets) {
    const auto plan = runtime::parse_exec_plan(options.options, "soulx_flashhead.exec_plan",
        {{"vae", "ggml", {{"ggml", {options.backend.type}, {}, {}},
                           {"cudnn", {core::BackendType::Cuda}, {}, {}}}}});
    const auto & vae = plan.at("vae");
    const bool cudnn = vae.impl == "cudnn";
    if (cudnn && options.backend.type != core::BackendType::Cuda)
        throw std::runtime_error("soulx_flashhead.exec_plan.vae: cudnn requires the CUDA session backend");
    if (cudnn && assets.config.variant != "lite")
        throw std::runtime_error("soulx_flashhead.exec_plan.vae: cudnn requires the Lite LTX VAE");
    return cudnn;
}
class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
  public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Assets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)),
          pipeline_(execution_context(), assets, resolve_vae_cudnn(options, *assets)), block_samples_((32 - assets->config.stride[0]) * 16000 / 25),
          default_steps_(assets->config.variant == "pretrained" ? 20 : 4) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "SoulX-FlashHead");
        if (task.task != runtime::VoiceTaskKind::AudioGeneration)
            throw std::runtime_error("FlashHead supports the sfx task");
        mark_prepared();
    }
    std::string family() const override {
        return kFamily;
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
    runtime::StreamingPolicy streaming_policy() const override {
        return {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::FinalResult, block_samples_,
                block_samples_ / 16000.};
    }
    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("FlashHead start_stream");
        reset();
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_,
                                                      "SoulX-FlashHead");
        GenerateOptions options;
        const auto image = runtime::find_option(request.options, {"reference_image_file"});
        if (!image || image->empty())
            throw std::runtime_error("FlashHead requires reference_image_file");
        options.reference_image = *image;
        options.height = runtime::parse_int_option(request.options, {"height"}).value_or(512);
        options.width = runtime::parse_int_option(request.options, {"width"}).value_or(512);
        options.steps = runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(default_steps_);
        options.audio_guidance_scale =
            runtime::parse_float_option(request.options, {"audio_guidance_scale"}).value_or(3);
        options.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(42);
        options.color_strength =
            runtime::parse_float_option(request.options, {"color_correction_strength"}).value_or(1);
        started_ = std::chrono::steady_clock::now();
        pipeline_.begin(options);
        collect_artifacts_ = task_.mode == runtime::RunMode::Offline;
        active_ = true;
    }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        if (!active_)
            throw std::runtime_error("FlashHead stream has not started");
        if (chunk.sample_rate != 16000 || chunk.channels != 1)
            throw std::runtime_error("FlashHead streaming audio must be mono 16 kHz");
        received_samples_ += chunk.samples.size();
        return publish(pipeline_.push(chunk.samples, false), false);
    }
    void set_stream_event_sink(runtime::StreamEventCallback sink) override {
        sink_ = std::move(sink);
    }
    runtime::TaskResult finalize() override {
        if (!active_)
            throw std::runtime_error("FlashHead stream has not started");
        if (!received_samples_)
            throw std::runtime_error("FlashHead requires non-empty audio input");
        auto final = publish(pipeline_.push({}, true), true);
        active_ = false;
        if (!collect_artifacts_ && !sink_)
            artifacts_ = std::move(final.output_artifacts);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started_));
        runtime::TaskResult result;
        result.output_artifacts = std::move(artifacts_);
        return result;
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        if (!request.audio_input)
            throw std::runtime_error("FlashHead requires audio input");
        start_stream(request);
        collect_artifacts_ = true;
        const auto & input = *request.audio_input;
        if (input.samples.empty())
            throw std::runtime_error("FlashHead requires non-empty audio input");
        auto audio = engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            input.samples, input.sample_rate, input.channels, 16000);
        process_audio_chunk({16000, 1, 0, std::move(audio)});
        return finalize();
    }
    void reset() override {
        active_ = false;
        collect_artifacts_ = false;
        received_samples_ = 0;
        artifacts_.clear();
        pipeline_.reset();
    }

  private:
    runtime::StreamEvent publish(std::vector<VideoChunk> chunks, bool final) {
        runtime::StreamEvent event;
        event.is_final = final;
        for (auto & chunk : chunks) {
            event.output_artifacts.push_back(runtime::make_voice_artifact(
                runtime::ArtifactKind::Custom, "soulx_flashhead_video_" + std::to_string(chunk.index),
                std::move(chunk.rgb),
                {{"format", "rgb24"},
                 {"width", std::to_string(chunk.width)},
                 {"height", std::to_string(chunk.height)},
                 {"frames", std::to_string(chunk.frames)},
                 {"fps", "25"},
                 {"chunk_index", std::to_string(chunk.index)}}));
        }
        if (collect_artifacts_)
            artifacts_.insert(artifacts_.end(), event.output_artifacts.begin(), event.output_artifacts.end());
        if (final && sink_)
            sink_(event);
        return event;
    }
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    Pipeline pipeline_;
    int64_t block_samples_;
    int64_t default_steps_;
    size_t received_samples_ = 0;
    bool active_ = false;
    bool collect_artifacts_ = false;
    std::vector<runtime::VoiceArtifact> artifacts_;
    runtime::StreamEventCallback sink_;
    std::chrono::steady_clock::time_point started_;
};
} // namespace
std::shared_ptr<runtime::IVoiceModelLoader> make_soulx_flashhead_loader() {
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
} // namespace engine::community_models::soulx_flashhead
