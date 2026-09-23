#include "engine/models/moonshine_asr/session.h"

#include "engine/models/moonshine_asr/runtime.h"
#include "engine/models/moonshine_asr/assets.h"
#include "engine/models/moonshine_asr/weights.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/silero_vad/session.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::moonshine_asr {
namespace {

constexpr const char * kFamily = "moonshine_asr";
constexpr float kDefaultAudioChunkSeconds = 60.0F;

std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("Moonshine ASR requires audio with positive sample rate and channel count");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Moonshine ASR audio sample count must be divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

void append_transcript_text(runtime::TaskResult & output, const runtime::TaskResult & chunk_result) {
    if (!chunk_result.text_output.has_value() || chunk_result.text_output->text.empty()) {
        return;
    }
    if (!output.text_output.has_value()) {
        output.text_output = runtime::Transcript{"", chunk_result.text_output->language};
    }
    if (!output.text_output->text.empty()) {
        output.text_output->text += ' ';
    }
    output.text_output->text += chunk_result.text_output->text;
}

std::shared_ptr<const MoonshineAssets> require_assets(std::shared_ptr<const MoonshineAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Moonshine ASR session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Moonshine ASR session requires a model contract");
    }
    return contract;
}

runtime::SessionOptions require_supported_session_options(
    runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    const auto checked_contract = require_contract(contract);
    auto validation_options = options;
    // Older standalone GGUF packages embed a v1 contract that predates VAD
    // chunking. Keep those packages usable while still rejecting unrelated
    // unknown session options.
    if (checked_contract->session_option_keys.find("moonshine_asr.vad_model_path") ==
        checked_contract->session_option_keys.end()) {
        validation_options.options.erase("moonshine_asr.vad_model_path");
    }
    runtime::validate_spec_backed_session_options(validation_options, *checked_contract, kFamily, "Moonshine ASR");
    return options;
}

std::unordered_map<std::string, std::string> normalize_request_options(
    std::unordered_map<std::string, std::string> options,
    const engine::model_spec::ModelContract & contract) {
    auto validation_options = options;
    // Older standalone GGUF packages embed a v1 contract that predates
    // Moonshine audio chunking. Validate everything else against the embedded
    // contract, but allow the runtime to consume these local chunking controls.
    if (contract.request_option_keys.find("audio_chunk_mode") == contract.request_option_keys.end()) {
        validation_options.erase("audio_chunk_mode");
    }
    if (contract.request_option_keys.find("audio_chunk_duration_sec") == contract.request_option_keys.end()) {
        validation_options.erase("audio_chunk_duration_sec");
    }
    runtime::validate_spec_backed_request_options(validation_options, contract, "Moonshine ASR");
    return options;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_moonshine_asr_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MoonshineAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MoonshineSTTSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

runtime::TaskResult transcribe_moonshine_asr_with_chunking(
    const MoonshineAssets & assets,
    const MoonshineWeights & weights,
    const engine::core::ExecutionContext & execution_context,
    const runtime::AudioBuffer & audio,
    const std::unordered_map<std::string, std::string> & options,
    const MoonshineRuntimeConfig & runtime_config,
    runtime::IOfflineVoiceTaskSession * vad_session) {
    const auto mode = engine::audio::parse_audio_chunk_mode(options);
    if (mode == engine::audio::AudioChunkMode::None) {
        return transcribe_moonshine_asr(assets, weights, execution_context, audio, options, runtime_config);
    }
    if (mode == engine::audio::AudioChunkMode::QuietEnergy) {
        throw std::runtime_error("Moonshine ASR supports audio_chunk_mode=auto, fixed, vad, or none");
    }

    const float seconds =
        engine::audio::parse_audio_chunk_seconds_override(options).value_or(kDefaultAudioChunkSeconds);
    if (!std::isfinite(seconds) || seconds <= 0.0F) {
        throw std::runtime_error("Moonshine ASR audio_chunk_duration_sec must be positive");
    }
    const int64_t chunk_samples =
        static_cast<int64_t>(std::llround(static_cast<double>(seconds) * static_cast<double>(audio.sample_rate)));
    if (chunk_samples <= 0) {
        throw std::runtime_error("Moonshine ASR audio_chunk_duration_sec produced an empty chunk");
    }

    std::vector<runtime::TimeSpan> spans;
    if (mode == engine::audio::AudioChunkMode::Vad) {
        if (vad_session == nullptr) {
            throw std::runtime_error("Moonshine ASR VAD chunking requires a VAD session");
        }
        const auto vad_options = engine::audio::VadAudioChunkOptions{
            chunk_samples,
            static_cast<int64_t>(std::llround(0.5 * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.25 * static_cast<double>(audio.sample_rate))),
        };
        spans = engine::audio::plan_vad_audio_chunks(audio, *vad_session, vad_options);
    } else {
        const auto chunks = engine::audio::plan_audio_chunks(
            audio_frame_count(audio),
            engine::audio::AudioChunkSpec{
                chunk_samples,
                chunk_samples,
                engine::audio::AudioChunkPadMode::Zero,
                engine::audio::AudioChunkTailAlignment::Start,
                0,
            });
        spans.reserve(chunks.size());
        for (const auto & chunk : chunks) {
            spans.push_back(runtime::TimeSpan{
                chunk.output_start_sample,
                chunk.output_start_sample + chunk.valid_samples,
            });
        }
    }
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{"", "en"};
    for (const auto & span : spans) {
        auto chunk_options = options;
        chunk_options["audio_chunk_mode"] = "none";
        const auto chunk_audio = engine::audio::slice_audio_buffer(audio, span);
        const auto chunk_result = transcribe_moonshine_asr(
            assets,
            weights,
            execution_context,
            chunk_audio,
            chunk_options,
            runtime_config);
        append_transcript_text(result, chunk_result);
    }
    return result;
}

}  // namespace

MoonshineSTTSession::MoonshineSTTSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MoonshineAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(require_supported_session_options(std::move(options), contract)),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Moonshine ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Moonshine ASR supports offline and streaming sessions");
    }
    runtime_config_ = make_moonshine_runtime_config(
        assets_->config,
        execution_context().backend_type(),
        RuntimeSessionBase::options().options);
    vad_model_path_ = runtime::find_option(RuntimeSessionBase::options().options, {"moonshine_asr.vad_model_path"})
        .value_or(default_vad_model_path().string());
    weights_ = load_moonshine_asr_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        runtime_config_.encoder_weight_storage_type,
        runtime_config_.decoder_weight_storage_type,
        runtime_config_.conv_weight_storage_type,
        runtime_config_.weight_context_bytes);
}

MoonshineSTTSession::~MoonshineSTTSession() = default;

std::string MoonshineSTTSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MoonshineSTTSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MoonshineSTTSession::run_mode() const {
    return task_.mode;
}

void MoonshineSTTSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)normalize_request_options(request.options, *contract_);
    mark_prepared();
}

runtime::TaskResult MoonshineSTTSession::run(const runtime::TaskRequest & request) {
    require_prepared("run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Moonshine ASR run() requires an offline session");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Moonshine ASR requires audio input");
    }
    auto normalized_request = request;
    normalized_request.options = normalize_request_options(request.options, *contract_);
    return transcribe_moonshine_asr_with_chunking(
        *assets_,
        *weights_,
        execution_context(),
        *normalized_request.audio_input,
        normalized_request.options,
        runtime_config_,
        engine::audio::parse_audio_chunk_mode(normalized_request.options) == engine::audio::AudioChunkMode::Vad ? &vad_session() : nullptr);
}

runtime::StreamingPolicy MoonshineSTTSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = static_cast<int64_t>(assets_->config.encoder.sample_rate);
    policy.preferred_audio_chunk_seconds = 1.0;
    return policy;
}

void MoonshineSTTSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Moonshine ASR start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Moonshine ASR start_stream() requires a streaming session");
    }
    auto normalized_request = request;
    normalized_request.options = normalize_request_options(request.options, *contract_);
    reset();
    streaming_request_ = std::move(normalized_request);
    streaming_request_.audio_input = std::nullopt;
    streaming_audio_.sample_rate = static_cast<int>(assets_->config.encoder.sample_rate);
    streaming_audio_.channels = 1;
    stream_started_ = true;
}

void MoonshineSTTSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void MoonshineSTTSession::reset() {
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_request_ = runtime::TaskRequest{};
    stream_started_ = false;
}

runtime::StreamEvent MoonshineSTTSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Moonshine ASR process_audio_chunk() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Moonshine ASR process_audio_chunk() requires start_stream()");
    }
    if (chunk.sample_rate <= 0 || chunk.channels <= 0) {
        throw std::runtime_error("Moonshine ASR streaming chunk has invalid audio metadata");
    }
    if (streaming_audio_.samples.empty()) {
        streaming_audio_.sample_rate = chunk.sample_rate;
        streaming_audio_.channels = chunk.channels;
    } else if (streaming_audio_.sample_rate != chunk.sample_rate || streaming_audio_.channels != chunk.channels) {
        throw std::runtime_error("Moonshine ASR streaming chunks must use one sample rate and channel count");
    }
    streaming_audio_.samples.insert(streaming_audio_.samples.end(), chunk.samples.begin(), chunk.samples.end());

    runtime::StreamEvent event;
    event.is_final = false;
    if (stream_event_sink_ != nullptr) {
        stream_event_sink_(event);
    }
    return event;
}

runtime::TaskResult MoonshineSTTSession::finalize() {
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Moonshine ASR finalize() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Moonshine ASR finalize() requires start_stream()");
    }
    streaming_request_.options = normalize_request_options(streaming_request_.options, *contract_);
    auto result = transcribe_moonshine_asr_with_chunking(
        *assets_,
        *weights_,
        execution_context(),
        streaming_audio_,
        streaming_request_.options,
        runtime_config_,
        engine::audio::parse_audio_chunk_mode(streaming_request_.options) == engine::audio::AudioChunkMode::Vad
            ? &vad_session()
            : nullptr);
    reset();
    return result;
}

runtime::TaskResult MoonshineSTTSession::finish_stream() {
    return finalize();
}

runtime::IOfflineVoiceTaskSession & MoonshineSTTSession::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("Moonshine ASR VAD helper did not create an offline session");
        }
        vad_session_.reset(offline);
        session.release();
    }
    return *vad_session_;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<MoonshineAssets> config;
    config.family = kFamily;
    config.load_assets = load_moonshine_asr_assets;
    config.create_session = create_moonshine_asr_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::moonshine_asr
