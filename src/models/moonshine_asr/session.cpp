#include "engine/models/moonshine_asr/session.h"

#include "engine/models/moonshine_asr/runtime.h"
#include "engine/models/moonshine_asr/assets.h"
#include "engine/models/moonshine_asr/weights.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moonshine_asr {
namespace {

constexpr const char * kFamily = "moonshine_asr";

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

}  // namespace

MoonshineSTTSession::MoonshineSTTSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MoonshineAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, "Moonshine ASR");
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Moonshine ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Moonshine ASR supports offline and streaming sessions");
    }
    runtime_config_ = make_moonshine_runtime_config(assets_->config, execution_context().backend_type(), options.options);
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
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Moonshine ASR");
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
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Moonshine ASR");
    return transcribe_moonshine_asr(
        *assets_,
        *weights_,
        execution_context(),
        *request.audio_input,
        request.options,
        runtime_config_);
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
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Moonshine ASR");
    reset();
    streaming_request_ = request;
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
    runtime::validate_spec_backed_request_options(streaming_request_.options, *contract_, "Moonshine ASR");
    auto result = transcribe_moonshine_asr(
        *assets_,
        *weights_,
        execution_context(),
        streaming_audio_,
        streaming_request_.options,
        runtime_config_);
    reset();
    return result;
}

runtime::TaskResult MoonshineSTTSession::finish_stream() {
    return finalize();
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<MoonshineAssets> config;
    config.family = kFamily;
    config.load_assets = load_moonshine_asr_assets;
    config.create_session = create_moonshine_asr_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::moonshine_asr
