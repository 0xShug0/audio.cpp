#include "engine/models/higgs_audio_stt/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::higgs_audio_stt {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const HiggsAudioSTTAssets> require_assets(std::shared_ptr<const HiggsAudioSTTAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Higgs Audio STT session requires assets");
    }
    return assets;
}

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("Higgs Audio STT audio chunking requires positive audio channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Higgs Audio STT audio samples must be divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, and f16");
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

size_t common_prefix_size(const std::string & lhs, const std::string & rhs) {
    const size_t limit = std::min(lhs.size(), rhs.size());
    size_t size = 0;
    while (size < limit && lhs[size] == rhs[size]) {
        ++size;
    }
    return size;
}

void emit_transcript_delta(
    const runtime::StreamEventCallback & sink,
    const runtime::Transcript & transcript,
    std::string & emitted_text) {
    if (!sink || transcript.text.empty()) {
        return;
    }
    const size_t prefix_size = common_prefix_size(emitted_text, transcript.text);
    if (prefix_size == transcript.text.size()) {
        emitted_text = transcript.text;
        return;
    }
    runtime::StreamEvent event;
    event.partial_text = runtime::Transcript{transcript.text.substr(prefix_size), transcript.language};
    sink(event);
    emitted_text = transcript.text;
}

std::string append_streaming_transcript(
    runtime::TaskResult & total,
    const runtime::Transcript & chunk) {
    if (!total.text_output.has_value()) {
        total.text_output = runtime::Transcript{"", chunk.language};
    }
    if (!chunk.language.empty()) {
        total.text_output->language = chunk.language;
    }
    if (chunk.text.empty()) {
        return "";
    }
    std::string delta;
    if (!total.text_output->text.empty()) {
        total.text_output->text.push_back(' ');
        delta.push_back(' ');
    }
    total.text_output->text += chunk.text;
    delta += chunk.text;
    return delta;
}

}  // namespace

HiggsAudioSTTSession::HiggsAudioSTTSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const HiggsAudioSTTAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"higgs_audio_stt.audio_encoder_graph_arena_mb"}, 512ull * 1024ull * 1024ull)),
      text_decoder_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"higgs_audio_stt.text_decoder_prefill_graph_arena_mb"}, 512ull * 1024ull * 1024ull)),
      text_decoder_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"higgs_audio_stt.text_decoder_decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      text_decoder_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"higgs_audio_stt.text_decoder_weight_context_mb"}, 4096ull * 1024ull * 1024ull)),
      audio_encoder_weight_storage_type_(option_weight_type(options, "higgs_audio_stt.audio_encoder_weight_type", engine::assets::TensorStorageType::Native)),
      text_decoder_weight_storage_type_(option_weight_type(
          options,
          "higgs_audio_stt.text_decoder_weight_type",
          option_weight_type(options, "higgs_audio_stt.weight_type", engine::assets::TensorStorageType::Native))),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_, execution_context(), audio_encoder_graph_arena_bytes_, audio_encoder_weight_storage_type_),
      text_decoder_(
          assets_,
          execution_context(),
          text_decoder_prefill_graph_arena_bytes_,
          text_decoder_decode_graph_arena_bytes_,
          text_decoder_weight_context_bytes_,
          text_decoder_weight_storage_type_),
      prompt_builder_(tokenizer_),
      postprocessor_(tokenizer_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Higgs Audio STT only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs Audio STT supports offline and streaming sessions");
    }
    validate_conv_weight_storage(audio_encoder_weight_storage_type_, "higgs_audio_stt.audio_encoder_weight_type");
    validate_matmul_weight_storage(text_decoder_weight_storage_type_, "higgs_audio_stt.text_decoder_weight_type");
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key.rfind("higgs_audio_stt.", 0) == 0 &&
            key != "higgs_audio_stt.audio_encoder_graph_arena_mb" &&
            key != "higgs_audio_stt.text_decoder_prefill_graph_arena_mb" &&
            key != "higgs_audio_stt.text_decoder_decode_graph_arena_mb" &&
            key != "higgs_audio_stt.text_decoder_weight_context_mb" &&
            key != "higgs_audio_stt.audio_encoder_weight_type" &&
            key != "higgs_audio_stt.text_decoder_weight_type" &&
            key != "higgs_audio_stt.weight_type") {
            throw std::runtime_error("unknown Higgs Audio STT session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

HiggsAudioSTTSession::~HiggsAudioSTTSession() = default;

std::string HiggsAudioSTTSession::family() const {
    return "higgs_audio_stt";
}

runtime::VoiceTaskKind HiggsAudioSTTSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HiggsAudioSTTSession::run_mode() const {
    return task_.mode;
}

void HiggsAudioSTTSession::prepare(const runtime::SessionPreparationRequest & request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Higgs Audio STT prepare() requires an audio contract");
    }
    mark_prepared();
    debug::timing_log_scalar("higgs_audio_stt.prepare_ms", engine::debug::elapsed_ms(prepare_start, Clock::now()));
    debug::trace_log_scalar("higgs_audio_stt.prepare.max_input_samples", request.audio->max_input_samples);
}

runtime::TaskResult HiggsAudioSTTSession::run(const runtime::TaskRequest & request) {
    require_prepared("Higgs Audio STT run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs Audio STT offline run called on non-offline session");
    }
    const auto chunks = audio_chunk_plan(request);
    if (chunks.empty()) {
        return run_single(make_request(request));
    }
    const auto & audio = *request.audio_input;
    if (chunks.size() == 1) {
        auto item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunks.front().source_span);
        return run_single(make_request(item_request));
    }
    runtime::TaskResult merged;
    std::ostringstream text;
    for (const auto & chunk : chunks) {
        auto item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunk.source_span);
        const auto item = run_single(make_request(item_request));
        if (item.text_output.has_value() && !item.text_output->text.empty()) {
            if (text.tellp() > 0) {
                text << ' ';
            }
            text << item.text_output->text;
            if (!merged.text_output.has_value()) {
                merged.text_output = runtime::Transcript{"", item.text_output->language};
            } else if (merged.text_output->language.empty()) {
                merged.text_output->language = item.text_output->language;
            }
        }
    }
    if (merged.text_output.has_value()) {
        merged.text_output->text = text.str();
    }
    return merged;
}

runtime::StreamingPolicy HiggsAudioSTTSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_seconds = 4.0;
    return policy;
}

void HiggsAudioSTTSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Higgs Audio STT start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs Audio STT start_stream called on non-streaming session");
    }
    reset();
    streaming_request_ = request;
    streaming_request_.audio_input = std::nullopt;
    streaming_result_ = runtime::TaskResult{};
    stream_started_ = true;
    streaming_chunks_processed_ = 0;
}

void HiggsAudioSTTSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void HiggsAudioSTTSession::reset() {
    require_prepared("Higgs Audio STT reset()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs Audio STT reset called on non-streaming session");
    }
    streaming_request_ = runtime::TaskRequest{};
    streaming_result_ = runtime::TaskResult{};
    stream_started_ = false;
    streaming_chunks_processed_ = 0;
}

runtime::StreamEvent HiggsAudioSTTSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Higgs Audio STT process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs Audio STT process_audio_chunk called on non-streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Higgs Audio STT process_audio_chunk requires start_stream");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;

    auto request = streaming_request_;
    request.audio_input = std::move(audio);
    runtime::StreamEventCallback saved_sink;
    saved_sink.swap(stream_event_sink_);
    runtime::TaskResult result;
    try {
        result = run_single(make_request(request));
        saved_sink.swap(stream_event_sink_);
    } catch (...) {
        saved_sink.swap(stream_event_sink_);
        throw;
    }
    ++streaming_chunks_processed_;

    runtime::StreamEvent event;
    if (result.text_output.has_value()) {
        const std::string delta = append_streaming_transcript(streaming_result_, *result.text_output);
        if (!delta.empty()) {
            event.partial_text = runtime::Transcript{delta, streaming_result_.text_output->language};
        }
    }
    event.is_final = false;
    return event;
}

runtime::TaskResult HiggsAudioSTTSession::finish_stream() {
    return finalize();
}

runtime::TaskResult HiggsAudioSTTSession::finalize() {
    require_prepared("Higgs Audio STT finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs Audio STT finalize called on non-streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Higgs Audio STT finalize requires start_stream");
    }
    if (streaming_chunks_processed_ == 0) {
        throw std::runtime_error("Higgs Audio STT finalize requires streamed audio");
    }
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output = runtime::Transcript{};
    }
    return streaming_result_;
}

runtime::TaskResult HiggsAudioSTTSession::run_single(const HiggsAudioSTTRequest & asr_request) {
    const auto wall_start = Clock::now();
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract(asr_request.audio);
    const auto frontend_end = Clock::now();
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(asr_request, features.encoder_tokens);
    const auto prompt_end = Clock::now();
    const auto encoder_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode(features);
    const auto encoder_end = Clock::now();
    const auto text_decoder_start = Clock::now();
    std::string emitted_text;
    HiggsAudioSTTTokenCallback token_callback;
    if (task_.mode == runtime::RunMode::Streaming && stream_event_sink_ != nullptr) {
        token_callback = [&](const HiggsAudioSTTGeneratedTokens & partial_tokens) {
            const auto partial = postprocessor_.decode(partial_tokens, asr_request);
            emit_transcript_delta(
                stream_event_sink_,
                runtime::Transcript{partial.text, partial.language},
                emitted_text);
        };
    }
    const auto tokens = text_decoder_.generate(prompt, audio_embeddings, asr_request.generation, token_callback);
    const auto text_decoder_end = Clock::now();
    const auto postprocess_start = Clock::now();
    const auto decoded = postprocessor_.decode(tokens, asr_request);
    if (task_.mode == runtime::RunMode::Streaming && stream_event_sink_ != nullptr) {
        emit_transcript_delta(
            stream_event_sink_,
            runtime::Transcript{decoded.text, decoded.language},
            emitted_text);
    }
    const auto postprocess_end = Clock::now();

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, decoded.language};

    debug::timing_log_scalar("higgs_audio_stt.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("higgs_audio_stt.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    debug::timing_log_scalar("higgs_audio_stt.audio_encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));
    debug::timing_log_scalar("higgs_audio_stt.text_decoder_ms", engine::debug::elapsed_ms(text_decoder_start, text_decoder_end));
    debug::timing_log_scalar("higgs_audio_stt.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("higgs_audio_stt.prompt_tokens", prompt.input_ids.size());
    debug::trace_log_scalar("higgs_audio_stt.audio_chunks", features.chunks);
    debug::trace_log_scalar("higgs_audio_stt.audio_frames", features.frames);
    debug::trace_log_scalar("higgs_audio_stt.audio_tokens", features.encoder_tokens);
    return result;
}

std::vector<HiggsAudioSTTSession::AudioChunkPlan> HiggsAudioSTTSession::audio_chunk_plan(
    const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        return {};
    }
    const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (mode == engine::audio::AudioChunkMode::None) {
        return {};
    }
    if (mode == engine::audio::AudioChunkMode::QuietEnergy ||
        mode == engine::audio::AudioChunkMode::Vad) {
        throw std::runtime_error("Higgs Audio STT supports audio_chunk_mode=auto, fixed, or none");
    }
    const auto & audio = *request.audio_input;
    const int64_t frames = audio_frame_count(audio);
    const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options).value_or(4.0F);
    if (!(seconds > 0.0F)) {
        throw std::runtime_error("Higgs Audio STT audio_chunk_seconds must be positive");
    }
    const int64_t samples = static_cast<int64_t>(
        std::llround(static_cast<double>(seconds) * static_cast<double>(audio.sample_rate)));
    if (samples <= 0) {
        throw std::runtime_error("Higgs Audio STT audio_chunk_seconds produced an empty chunk");
    }
    const auto chunks = engine::audio::plan_audio_chunks(
        frames,
        engine::audio::AudioChunkSpec{
            samples,
            samples,
            engine::audio::AudioChunkPadMode::Zero,
            engine::audio::AudioChunkTailAlignment::Start,
            0,
        });
    std::vector<AudioChunkPlan> plan;
    plan.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        plan.push_back(AudioChunkPlan{
            runtime::TimeSpan{
                chunk.output_start_sample,
                chunk.output_start_sample + chunk.valid_samples,
            },
        });
    }
    return plan;
}

HiggsAudioSTTRequest HiggsAudioSTTSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Higgs Audio STT run() requires audio_input");
    }
    HiggsAudioSTTRequest out;
    out.audio = *request.audio_input;
    out.generation.max_new_tokens = assets_->config.max_new_tokens;
    if (request.text_input.has_value()) {
        out.context = request.text_input->text;
        out.language = request.text_input->language;
    }
    if (const auto language = runtime::find_option(request.options, {"language"})) {
        out.language = *language;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
        if (out.generation.max_new_tokens <= 0) {
            throw std::runtime_error("Higgs Audio STT max_tokens must be positive");
        }
    }
    if (const auto value = runtime::find_option(request.options, {"enable_thinking"})) {
        out.generation.enable_thinking = runtime::parse_bool_option(*value, "enable_thinking");
    }
    return out;
}

}  // namespace engine::models::higgs_audio_stt
