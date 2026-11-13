#include "engine/models/kugelaudio/session.h"

#include "engine/models/kugelaudio/assets.h"
#include "engine/models/kugelaudio/generator.h"
#include "engine/models/kugelaudio/reference_encoder.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <cmath>

namespace engine::models::kugelaudio {
namespace {

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::StreamingTtsSessionBase {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
            std::shared_ptr<const ModelAssets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "kugelaudio", "KugelAudio");
        if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) {
            throw std::runtime_error("KugelAudio supports TTS and reference-audio voice cloning");
        }
        reference_encoder_path_ = runtime::find_option(options.options,
            {"kugelaudio.reference_encoder_path", "reference_encoder_path"}).value_or("");
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"kugelaudio.weight_type", "weight_type"}).value_or("native"));
        generator_ = std::make_unique<Generator>(*assets_->weights,
            assets_->resources.require_file("tokenizer_json").parent_path(), execution_context(), storage,
            assets_->config.ar, assets_->config.diffusion, assets_->config.codec);
    }

    std::string family() const override { return "kugelaudio"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "KugelAudio");
        // Assign preparation state only after reference validation/encoding has
        // succeeded. A failed request cannot publish a half-prepared voice.
        if (request.voice && request.voice->speaker && request.voice->speaker->audio)
            (void)reference_features(*request.voice->speaker->audio);
        prepared_voice_ = request.voice;
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("KugelAudio run requires an offline session");
        }
        return synthesize(request, false);
    }

private:
    const VoiceFeatures & reference_features(const runtime::AudioBuffer & audio) {
        auto waveform = prepare_reference_audio(audio);
        if (cached_reference_ && cached_reference_->waveform == waveform) {
            debug::trace_log_scalar("kugelaudio.reference_cache_hit", 1);
            return cached_reference_->features;
        }
        const auto config = parse_reference_encoder_config(
            assets_->resources.parse_json("config").require("acoustic_tokenizer_config"));
        auto source = assets_->weights;
        if (!reference_encoder_path_.empty()) source = assets::open_tensor_source(reference_encoder_path_);
        if (source->has_tensor("reference_encoder/model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight"))
            source = assets::make_prefixed_tensor_source(source, "reference_encoder/");
        if (!source->has_tensor("model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight"))
            throw std::runtime_error("This KugelAudio package omits the reference encoder. Supply "
                "kugelaudio.reference_encoder_path or convert with --include-reference-encoder");
        const auto begin = std::chrono::steady_clock::now();
        PreparedReference next;
        next.waveform = std::move(waveform);
        {
            // Only a completed voice feature vector is retained. Encoder graphs
            // and GPU weights are released before the autoregressive generation.
            ReferenceEncoder encoder(*source, execution_context(), config);
            next.features.mean = encoder.encode(next.waveform);
            next.features.std = config.fixed_std;
        }
        if (source != assets_->weights) source->release_storage();
        if (next.features.mean.empty() || next.features.mean.size() % config.latent_size)
            throw std::runtime_error("KugelAudio reference encoder returned invalid voice features");
        debug::timing_log_scalar("kugelaudio.reference_encode_ms", debug::elapsed_ms(begin));
        debug::trace_log_scalar("kugelaudio.reference_frames", int64_t(next.features.mean.size() / config.latent_size));
        debug::trace_log_scalar("kugelaudio.reference_cache_hit", 0);
        cached_reference_ = std::move(next);
        return cached_reference_->features;
    }

    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override {
        return synthesize(request, true);
    }

    void reset_stream_state() override {}

    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("KugelAudio synthesis");
        const auto begin = std::chrono::steady_clock::now();
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "KugelAudio");
        if (!request.text_input || request.text_input->text.empty()) {
            throw std::runtime_error("KugelAudio requires non-empty text");
        }
        auto voice_id = runtime::find_option(request.options, {"voice_id"}).value_or("default");
        const auto & condition = request.voice ? request.voice : prepared_voice_;
        const runtime::AudioBuffer * reference = nullptr;
        if (condition && condition->speaker) {
            if (condition->speaker->audio) reference = &*condition->speaker->audio;
            if (condition->speaker->cached_voice_id) {
                voice_id = *condition->speaker->cached_voice_id;
            }
        }
        if (task_.task == runtime::VoiceTaskKind::VoiceCloning && !reference)
            throw runtime::InvalidRequestError("KugelAudio voice cloning requires speaker reference audio");
        const VoiceFeatures * voice = nullptr;
        if (reference) voice = &reference_features(*reference);
        else {
            const auto preset = assets_->voices.find(voice_id);
            if (preset == assets_->voices.end()) throw std::runtime_error("unknown KugelAudio voice_id: " + voice_id);
            voice = &preset->second;
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
        debug::trace_log_scalar("kugelaudio.voice_id", reference ? "reference_audio" : voice_id);
        debug::trace_log_scalar("kugelaudio.seed", options.seed);
        debug::trace_log_scalar("kugelaudio.num_inference_steps", options.steps);
        debug::trace_log_scalar("kugelaudio.guidance_scale", options.guidance_scale);
        debug::trace_log_scalar("kugelaudio.text_chunk_count", static_cast<int64_t>(chunks.size()));
        runtime::AudioBuffer audio{24000, 1, {}};
        std::function<void(const std::vector<float> &)> on_audio;
        if (streaming) {
            on_audio = [this](const std::vector<float> & samples) {
                if (stream_event_sink()) {
                    runtime::StreamEvent event;
                    event.audio_output = runtime::AudioBuffer{24000, 1, samples};
                    stream_event_sink()(event);
                }
            };
        }
        for (const auto & chunk : chunks) {
            auto samples = generator_->generate(chunk.text_input->text, *voice, options, on_audio);
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
    struct PreparedReference {
        std::vector<float> waveform;
        VoiceFeatures features;
    };
    std::optional<PreparedReference> cached_reference_;
    std::optional<runtime::VoiceCondition> prepared_voice_;
    std::string reference_encoder_path_;
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
