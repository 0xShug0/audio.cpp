#include "engine/models/meanvc2/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::models::meanvc2 {
namespace {

constexpr const char * kFamily = "meanvc2";
constexpr int64_t kInputChunkSamples = 2560;
constexpr size_t kWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kSpeakerDim = 256;
constexpr int64_t kMelDim = 80;
constexpr int64_t kVcConditionFrames = 16;
constexpr int64_t kVcChunkFrames = 12;
constexpr int64_t kVcBlockFrames = 4;

std::shared_ptr<const MeanVC2Assets> require_assets(std::shared_ptr<const MeanVC2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MeanVC2 session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MeanVC2 session requires a model contract");
    }
    return contract;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_meanvc2_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MeanVC2Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MeanVC2Session>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

core::BackendConfig vocoder_backend_config(core::BackendConfig config) {
    config.type = core::BackendType::Cpu;
    config.device = 0;
    return config;
}

}  // namespace

MeanVC2Session::MeanVC2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MeanVC2Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      execution_context_(RuntimeSessionBase::options().backend),
      vocoder_execution_context_(vocoder_backend_config(RuntimeSessionBase::options().backend)) {
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, "MeanVC2");
    if (task_.task != runtime::VoiceTaskKind::VoiceConversion) {
        throw std::runtime_error("MeanVC2 supports --task vc");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("MeanVC2 supports offline and streaming modes");
    }
    asr_encoder_ = std::make_unique<MeanVC2AsrEncoderRuntime>(
        assets_->asr_weights,
        execution_context_,
        kWeightContextBytes,
        kGraphContextBytes,
        assets::TensorStorageType::Native);
    speaker_encoder_ = std::make_unique<MeanVC2SpeakerEncoderRuntime>(
        assets_->speaker_wavlm_weights,
        assets_->speaker_ecapa_weights,
        execution_context_,
        assets::TensorStorageType::Native);
    flow_ = std::make_unique<MeanVC2FlowSamplerRuntime>(
        assets_->flow_weights,
        execution_context_,
        kWeightContextBytes,
        kGraphContextBytes,
        assets::TensorStorageType::Native);
    vocoder_ = std::make_unique<MeanVC2VocoderRuntime>(
        assets_->vocos_weights,
        vocoder_execution_context_,
        kWeightContextBytes,
        kGraphContextBytes,
        assets::TensorStorageType::Native,
        assets::TensorStorageType::Native);
}

MeanVC2Session::~MeanVC2Session() = default;

std::string MeanVC2Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MeanVC2Session::task_kind() const {
    return task_.task;
}

runtime::RunMode MeanVC2Session::run_mode() const {
    return task_.mode;
}

void MeanVC2Session::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MeanVC2");
    mark_prepared();
}

runtime::TaskResult MeanVC2Session::run(const runtime::TaskRequest & request) {
    require_prepared("MeanVC2 run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MeanVC2 run() requires offline mode");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MeanVC2");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("MeanVC2 requires --audio source audio");
    }
    return convert(request, *request.audio_input);
}

runtime::StreamingPolicy MeanVC2Session::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = kInputChunkSamples;
    policy.preferred_audio_chunk_seconds = static_cast<double>(kInputChunkSamples) / 16000.0;
    return policy;
}

void MeanVC2Session::start_stream(const runtime::TaskRequest & request) {
    require_prepared("MeanVC2 start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("MeanVC2 start_stream() requires a streaming session");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MeanVC2");
    if (!request.voice.has_value() ||
        !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("MeanVC2 streaming requires --voice-ref target speaker audio");
    }
    reset();
    streaming_request_ = request;
    streaming_request_.audio_input = std::nullopt;
    frontend_.reset();
    bn_adapter_.reset();
    asr_encoder_->reset();
    streaming_speaker_embedding_ = speaker_encoder_->embed(*request.voice->speaker->audio);
    streaming_speaker_memory_ = flow_->encode_speaker_memory(streaming_speaker_embedding_);
    streaming_seed_ = runtime::parse_u64_option(request.options, {"seed"}).value_or(42);
    flow_->start_streaming(streaming_speaker_embedding_, streaming_speaker_memory_, streaming_seed_);
    vocoder_->reset_streaming_state();
    stream_started_ = true;
}

void MeanVC2Session::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void MeanVC2Session::reset() {
    require_prepared("MeanVC2 reset()");
    streaming_request_ = runtime::TaskRequest{};
    streaming_output_ = runtime::AudioBuffer{};
    streaming_bn_buffer_.clear();
    streaming_speaker_embedding_.clear();
    streaming_speaker_memory_ = {};
    streaming_seed_ = 42;
    flow_->reset_streaming();
    vocoder_->reset_streaming_state();
    stream_started_ = false;
}

runtime::AudioBuffer MeanVC2Session::drain_streaming_conditions(bool finish) {
    runtime::AudioBuffer out;
    out.sample_rate = 16000;
    out.channels = 1;
    const auto run_condition = [&](const float * condition_ptr) {
        std::vector<float> condition(
            condition_ptr,
            condition_ptr + static_cast<std::ptrdiff_t>(kVcConditionFrames * kSpeakerDim));
        const auto mel_frames = flow_->synthesize_streaming_chunk(condition);
        auto audio = vocoder_->decode_streaming_chunk(
            mel_frames,
            static_cast<int64_t>(mel_frames.size() / static_cast<size_t>(kMelDim)));
        if (!audio.samples.empty()) {
            runtime::append_audio_buffer(out, audio);
        }
    };

    int steps = 0;
    while (static_cast<int64_t>(streaming_bn_buffer_.size() / static_cast<size_t>(kSpeakerDim)) >= kVcConditionFrames &&
           (finish || steps < 4)) {
        run_condition(streaming_bn_buffer_.data());
        streaming_bn_buffer_.erase(
            streaming_bn_buffer_.begin(),
            streaming_bn_buffer_.begin() + static_cast<std::ptrdiff_t>(kVcChunkFrames * kSpeakerDim));
        ++steps;
    }

    if (!finish) {
        return out;
    }

    while (static_cast<int64_t>(streaming_bn_buffer_.size() / static_cast<size_t>(kSpeakerDim)) >= kVcChunkFrames) {
        std::vector<float> condition(static_cast<size_t>(kVcConditionFrames * kSpeakerDim), 0.0F);
        const int64_t buffered = static_cast<int64_t>(streaming_bn_buffer_.size() / static_cast<size_t>(kSpeakerDim));
        const int64_t current = std::min<int64_t>(kVcChunkFrames, buffered);
        std::copy(
            streaming_bn_buffer_.begin(),
            streaming_bn_buffer_.begin() + static_cast<std::ptrdiff_t>(current * kSpeakerDim),
            condition.begin());
        const int64_t future = std::min<int64_t>(kVcBlockFrames, std::max<int64_t>(0, buffered - kVcChunkFrames));
        if (future > 0) {
            std::copy(
                streaming_bn_buffer_.begin() + static_cast<std::ptrdiff_t>(kVcChunkFrames * kSpeakerDim),
                streaming_bn_buffer_.begin() + static_cast<std::ptrdiff_t>((kVcChunkFrames + future) * kSpeakerDim),
                condition.begin() + static_cast<std::ptrdiff_t>(kVcChunkFrames * kSpeakerDim));
        }
        const float * repeat_frame = buffered > 0
            ? streaming_bn_buffer_.data() + static_cast<std::ptrdiff_t>((std::min<int64_t>(buffered, kVcChunkFrames) - 1) * kSpeakerDim)
            : condition.data();
        for (int64_t frame = current + future; frame < kVcConditionFrames; ++frame) {
            std::copy(
                repeat_frame,
                repeat_frame + static_cast<std::ptrdiff_t>(kSpeakerDim),
                condition.begin() + static_cast<std::ptrdiff_t>(frame * kSpeakerDim));
        }
        run_condition(condition.data());
        const int64_t erase_frames = std::min<int64_t>(kVcChunkFrames, buffered);
        streaming_bn_buffer_.erase(
            streaming_bn_buffer_.begin(),
            streaming_bn_buffer_.begin() + static_cast<std::ptrdiff_t>(erase_frames * kSpeakerDim));
    }
    if (!streaming_bn_buffer_.empty()) {
        std::vector<float> condition(static_cast<size_t>(kVcConditionFrames * kSpeakerDim), 0.0F);
        const int64_t buffered = static_cast<int64_t>(streaming_bn_buffer_.size() / static_cast<size_t>(kSpeakerDim));
        const float * repeat_frame =
            streaming_bn_buffer_.data() + static_cast<std::ptrdiff_t>((buffered - 1) * kSpeakerDim);
        for (int64_t frame = 0; frame < kVcConditionFrames; ++frame) {
            std::copy(
                repeat_frame,
                repeat_frame + static_cast<std::ptrdiff_t>(kSpeakerDim),
                condition.begin() + static_cast<std::ptrdiff_t>(frame * kSpeakerDim));
        }
        std::copy(streaming_bn_buffer_.begin(), streaming_bn_buffer_.end(), condition.begin());
        run_condition(condition.data());
        streaming_bn_buffer_.clear();
    }
    return out;
}

runtime::AudioBuffer MeanVC2Session::process_streaming_audio(const runtime::AudioBuffer & audio) {
    const auto source_audio = prepare_meanvc2_source_audio_16k(audio);
    for (size_t pos = 0; pos < source_audio.mono_16k.size();) {
        const size_t count = std::min<size_t>(
            static_cast<size_t>(kInputChunkSamples),
            source_audio.mono_16k.size() - pos);
        const auto windows = frontend_.encode_chunk(
            source_audio.mono_16k.data() + static_cast<std::ptrdiff_t>(pos),
            static_cast<int64_t>(count));
        const auto encoded = asr_encoder_->encode_windows(windows);
        const auto conditioned = bn_adapter_.append_encoded_frames(encoded);
        streaming_bn_buffer_.insert(streaming_bn_buffer_.end(), conditioned.begin(), conditioned.end());
        pos += count;
    }
    return drain_streaming_conditions(false);
}

runtime::StreamEvent MeanVC2Session::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("MeanVC2 process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("MeanVC2 process_audio_chunk() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("MeanVC2 process_audio_chunk() requires start_stream()");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;
    if (audio.channels <= 0 || audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MeanVC2 streaming audio chunk has invalid channel layout");
    }
    auto audio_output = process_streaming_audio(audio);
    if (!audio_output.samples.empty()) {
        runtime::append_audio_buffer(streaming_output_, audio_output);
        runtime::StreamEvent event;
        event.audio_output = std::move(audio_output);
        return event;
    }
    return {};
}

runtime::TaskResult MeanVC2Session::finish_stream() {
    return finalize();
}

runtime::TaskResult MeanVC2Session::finalize() {
    require_prepared("MeanVC2 finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("MeanVC2 finalize() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("MeanVC2 finalize() requires start_stream()");
    }
    auto final_audio = drain_streaming_conditions(true);
    auto final_overlap = vocoder_->finish_streaming();
    if (!final_overlap.samples.empty()) {
        runtime::append_audio_buffer(final_audio, final_overlap);
    }
    if (!final_audio.samples.empty()) {
        runtime::append_audio_buffer(streaming_output_, final_audio);
    }
    if (streaming_output_.samples.empty()) {
        throw std::runtime_error("MeanVC2 finalize() requires streamed audio");
    }
    runtime::TaskResult result;
    result.audio_output = std::move(streaming_output_);
    streaming_request_ = runtime::TaskRequest{};
    streaming_bn_buffer_.clear();
    streaming_speaker_embedding_.clear();
    streaming_speaker_memory_ = {};
    flow_->reset_streaming();
    stream_started_ = false;
    if (stream_event_sink_ != nullptr) {
        runtime::StreamEvent event;
        if (!final_audio.samples.empty()) {
            event.audio_output = std::move(final_audio);
        }
        event.is_final = true;
        stream_event_sink_(event);
    }
    return result;
}

runtime::TaskResult MeanVC2Session::convert(
    const runtime::TaskRequest & request,
    const runtime::AudioBuffer & source_input) {
    if (!request.voice.has_value() ||
        !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("MeanVC2 requires --voice-ref target speaker audio");
    }
    const auto source_audio = prepare_meanvc2_source_audio_16k(source_input);
    frontend_.reset();
    bn_adapter_.reset();
    asr_encoder_->reset();
    int64_t asr_windows = 0;
    std::vector<float> bn_features;
    const auto asr_start = std::chrono::steady_clock::now();
    for (size_t pos = 0; pos < source_audio.mono_16k.size();) {
        const size_t count = std::min<size_t>(
            static_cast<size_t>(kInputChunkSamples),
            source_audio.mono_16k.size() - pos);
        const auto windows = frontend_.encode_chunk(
            source_audio.mono_16k.data() + static_cast<std::ptrdiff_t>(pos),
            static_cast<int64_t>(count));
        asr_windows += static_cast<int64_t>(windows.size());
        const auto encoded = asr_encoder_->encode_windows(windows);
        const auto conditioned = bn_adapter_.append_encoded_frames(encoded);
        bn_features.insert(bn_features.end(), conditioned.begin(), conditioned.end());
        pos += count;
    }
    engine::debug::timing_log_scalar(
        "meanvc2.asr.encode_ms",
        engine::debug::elapsed_ms(asr_start, std::chrono::steady_clock::now()));
    engine::debug::timing_log_scalar("meanvc2.frontend.asr_windows", asr_windows);
    engine::debug::timing_log_scalar("meanvc2.asr.encoded_frames", static_cast<int64_t>(bn_features.size() / 256));
    engine::debug::timing_log_scalar("meanvc2.asr.bn_values", bn_features.size());
    engine::debug::timing_log_scalar("meanvc2.frontend.source_samples", source_audio.mono_16k.size());
    const auto speaker_embedding = speaker_encoder_->embed(*request.voice->speaker->audio);
    engine::debug::timing_log_scalar("meanvc2.speaker.embedding_dims", speaker_embedding.size());
    const auto speaker_memory = flow_->encode_speaker_memory(speaker_embedding);
    engine::debug::timing_log_scalar("meanvc2.vc.gtm_key_values", speaker_memory.keys.size());
    engine::debug::timing_log_scalar("meanvc2.vc.gtm_value_values", speaker_memory.values.size());
    const auto seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(42);
    const auto flow_start = std::chrono::steady_clock::now();
    const auto mel_frames = flow_->synthesize_mel(
        bn_features,
        static_cast<int64_t>(bn_features.size() / 256),
        speaker_embedding,
        speaker_memory,
        seed);
    engine::debug::timing_log_scalar(
        "meanvc2.vc.decode_ms",
        engine::debug::elapsed_ms(flow_start, std::chrono::steady_clock::now()));
    const int64_t mel_frame_count = static_cast<int64_t>(mel_frames.size() / static_cast<size_t>(kMelDim));
    const auto vocoder_start = std::chrono::steady_clock::now();
    auto output = vocoder_->decode_streaming(mel_frames, mel_frame_count);
    engine::debug::timing_log_scalar(
        "meanvc2.vocoder.decode_ms",
        engine::debug::elapsed_ms(vocoder_start, std::chrono::steady_clock::now()));
    runtime::TaskResult result;
    result.audio_output = std::move(output);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_meanvc2_loader() {
    runtime::SpecBackedVoiceModelConfig<MeanVC2Assets> config;
    config.family = kFamily;
    config.load_assets = load_meanvc2_assets;
    config.create_session = create_meanvc2_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::meanvc2
