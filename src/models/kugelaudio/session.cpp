#include "engine/models/kugelaudio/session.h"

#include "engine/models/kugelaudio/assets.h"
#include "engine/models/kugelaudio/generator.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <chrono>

namespace engine::models::kugelaudio {
namespace {

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
            std::shared_ptr<const ModelAssets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "kugelaudio", "KugelAudio");
        if (task.task != runtime::VoiceTaskKind::Tts) {
            throw std::runtime_error("KugelAudio supports TTS with preset voices, not voice cloning");
        }
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"weight_type"}).value_or("native"));
        generator_ = std::make_unique<Generator>(*assets_->weights,
            assets_->resources.require_file("tokenizer_json").parent_path(), execution_context(), storage,
            assets_->config.ar, assets_->config.diffusion, assets_->config.codec);
    }

    std::string family() const override { return "kugelaudio"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Tts; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "KugelAudio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("KugelAudio run requires an offline session");
        }
        return synthesize(request, false);
    }

    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::None;
        policy.output = runtime::StreamingOutputKind::FinalResult;
        return policy;
    }

    void start_stream(const runtime::TaskRequest & request) override {
        if (task_.mode != runtime::RunMode::Streaming) {
            throw std::runtime_error("KugelAudio start_stream requires a streaming session");
        }
        reset();
        stream_result_ = synthesize(request, true);
    }
    void set_stream_event_sink(runtime::StreamEventCallback sink) override { sink_ = std::move(sink); }
    void reset() override { stream_result_.reset(); }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &) override {
        throw std::runtime_error("KugelAudio streaming accepts text, not audio input");
    }
    runtime::TaskResult finalize() override {
        if (!stream_result_) {
            throw std::runtime_error("KugelAudio stream has not completed");
        }
        auto result = std::move(*stream_result_);
        reset();
        return result;
    }

private:
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("KugelAudio synthesis");
        const auto begin = std::chrono::steady_clock::now();
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "KugelAudio");
        if (!request.text_input || request.text_input->text.empty()) {
            throw std::runtime_error("KugelAudio requires non-empty text");
        }
        auto voice_id = runtime::find_option(request.options, {"voice_id"}).value_or("default");
        if (request.voice && request.voice->speaker) {
            if (request.voice->speaker->audio) {
                throw std::runtime_error("KugelAudio does not support reference-audio voice cloning");
            }
            if (request.voice->speaker->cached_voice_id) {
                voice_id = *request.voice->speaker->cached_voice_id;
            }
        }
        const auto voice = assets_->voices.find(voice_id);
        if (voice == assets_->voices.end()) {
            throw std::runtime_error("unknown KugelAudio voice_id: " + voice_id);
        }
        GenerationOptions options;
        options.max_tokens = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(2048);
        options.steps = static_cast<int>(runtime::parse_i64_option(request.options, {"num_inference_steps"})
            .value_or(assets_->config.inference_steps));
        options.guidance_scale = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(3.0F);
        options.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(1.0F);
        options.do_sample = runtime::parse_bool_option(
            runtime::find_option(request.options, {"do_sample"}).value_or("false"), "do_sample");
        const auto seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(1234);
        if (seed < -1) {
            throw std::runtime_error("KugelAudio seed must be -1 or non-negative");
        }
        options.seed = seed == -1 ? static_cast<uint32_t>(runtime::random_u64_seed()) : static_cast<uint32_t>(seed);
        const auto chunk_size = text::parse_text_chunk_size_override(request.options).value_or(300);
        const auto chunk_mode = text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::Default);
        const auto chunks = runtime::chunk_text_request(request, chunk_size, chunk_mode);
        debug::trace_log_scalar("kugelaudio.voice_id", voice_id);
        debug::trace_log_scalar("kugelaudio.seed", options.seed);
        debug::trace_log_scalar("kugelaudio.num_inference_steps", options.steps);
        debug::trace_log_scalar("kugelaudio.guidance_scale", options.guidance_scale);
        debug::trace_log_scalar("kugelaudio.text_chunk_count", static_cast<int64_t>(chunks.size()));
        runtime::AudioBuffer audio{24000, 1, {}};
        std::function<void(const std::vector<float> &)> on_audio;
        if (streaming) {
            on_audio = [this](const std::vector<float> & samples) {
                if (sink_) {
                    runtime::StreamEvent event;
                    event.audio_output = runtime::AudioBuffer{24000, 1, samples};
                    sink_(event);
                }
            };
        }
        for (const auto & chunk : chunks) {
            auto samples = generator_->generate(chunk.text_input->text, voice->second, options, on_audio);
            audio.samples.insert(audio.samples.end(), samples.begin(), samples.end());
            ++options.seed;
        }
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(begin));
        return result;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const ModelAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<Generator> generator_;
    runtime::StreamEventCallback sink_;
    std::optional<runtime::TaskResult> stream_result_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_kugelaudio_loader() {
    runtime::SpecBackedVoiceModelConfig<ModelAssets> config;
    config.family = "kugelaudio";
    config.load_assets = load_assets;
    config.create_session = [](const auto & task, const auto & options, auto assets, auto contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::kugelaudio
