#include "engine/models/voxtral_realtime/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>

namespace engine::models::voxtral_realtime {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultAudioEncoderGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultAudioEncoderWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultTextPrefillGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultTextDecodeGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultTextWeightContextBytes = 128ull * 1024ull * 1024ull;
// 1024 steps of 80 ms = ~82 s of decoder attention context, held in one graph for the whole
// session so a live stream never pays a cache-growth rebuild.
constexpr int64_t kDefaultStreamDecodeCacheSteps = 1024;
// Audio tokens per steady encoder forward. Batching amortizes the encoder's fixed per-dispatch
// cost across N tokens but delays every partial by N * 80 ms, so it stays opt-in.
constexpr int64_t kDefaultStreamBatchTokens = 1;

int64_t source_frames_for_target_samples(int64_t target_samples, int source_rate, int channels, int target_rate) {
    if (target_samples <= 0 || source_rate <= 0 || channels <= 0 || target_rate <= 0) {
        throw std::runtime_error("VoxTral streaming chunk sizing requires positive audio shape");
    }
    return std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(
            static_cast<double>(target_samples) * static_cast<double>(source_rate) /
            static_cast<double>(target_rate))));
}

VoxtralRealtimeGenerationOptions parse_generation_options(
    const std::unordered_map<std::string, std::string> & options) {
    VoxtralRealtimeGenerationOptions out;
    if (const auto max_new_tokens = runtime::parse_i64_option(options, {"max_new_tokens"})) {
        out.max_new_tokens = *max_new_tokens;
        out.max_new_tokens_set = true;
        if (out.max_new_tokens <= 0) {
            throw std::runtime_error("VoxTral realtime max_new_tokens must be positive");
        }
    }
    if (const auto do_sample = runtime::find_option_match(options, {"do_sample"})) {
        out.do_sample = runtime::parse_bool_option(do_sample->value, do_sample->key);
    }
    if (const auto temperature = runtime::parse_finite_float_option(options, {"temperature"})) {
        out.temperature = *temperature;
    }
    if (const auto top_p = runtime::parse_finite_float_option(options, {"top_p"})) {
        out.top_p = *top_p;
    }
    if (const auto top_k = runtime::parse_i64_option(options, {"top_k"})) {
        out.top_k = *top_k;
    }
    if (const auto seed = runtime::parse_u64_option(options, {"seed"})) {
        out.seed = *seed;
    }
    if (out.temperature <= 0.0F) {
        throw std::runtime_error("VoxTral realtime temperature must be positive");
    }
    if (out.top_p <= 0.0F || out.top_p > 1.0F) {
        throw std::runtime_error("VoxTral realtime top_p must be in (0, 1]");
    }
    if (out.top_k < 0) {
        throw std::runtime_error("VoxTral realtime top_k must be non-negative");
    }
    return out;
}

}  // namespace

VoxtralRealtimeSession::VoxtralRealtimeSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VoxtralRealtimeAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"voxtral_realtime.audio_encoder_graph_arena_mb"}, kDefaultAudioEncoderGraphArenaBytes)),
      audio_encoder_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"voxtral_realtime.audio_encoder_weight_context_mb"}, kDefaultAudioEncoderWeightContextBytes)),
      text_decoder_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"voxtral_realtime.text_decoder_prefill_graph_arena_mb"}, kDefaultTextPrefillGraphArenaBytes)),
      text_decoder_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"voxtral_realtime.text_decoder_decode_graph_arena_mb"}, kDefaultTextDecodeGraphArenaBytes)),
      text_decoder_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"voxtral_realtime.text_decoder_weight_context_mb"}, kDefaultTextWeightContextBytes)),
      audio_encoder_weight_storage_type_([&options] {
          auto weight_type = assets::TensorStorageType::Native;
          if (const auto shared = runtime::find_option_match(options.options, {"voxtral_realtime.weight_type"})) {
              weight_type = assets::parse_tensor_storage_type(shared->value);
          }
          if (const auto specific = runtime::find_option_match(options.options, {"voxtral_realtime.audio_encoder_weight_type"})) {
              weight_type = assets::parse_tensor_storage_type(specific->value);
          }
          return weight_type;
      }()),
      text_decoder_weight_storage_type_([&options] {
          auto weight_type = assets::TensorStorageType::Native;
          if (const auto shared = runtime::find_option_match(options.options, {"voxtral_realtime.weight_type"})) {
              weight_type = assets::parse_tensor_storage_type(shared->value);
          }
          if (const auto specific = runtime::find_option_match(options.options, {"voxtral_realtime.text_decoder_weight_type"})) {
              weight_type = assets::parse_tensor_storage_type(specific->value);
          }
          return weight_type;
      }()),
      stream_decode_cache_steps_(
          runtime::parse_i64_option(options.options, {"voxtral_realtime.stream_decode_cache_steps"})
              .value_or(kDefaultStreamDecodeCacheSteps)),
      stream_batch_tokens_(
          runtime::parse_i64_option(options.options, {"voxtral_realtime.stream_batch_tokens"})
              .value_or(kDefaultStreamBatchTokens)),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_, execution_context(), audio_encoder_graph_arena_bytes_, audio_encoder_weight_context_bytes_, audio_encoder_weight_storage_type_),
      text_decoder_(assets_, execution_context(), text_decoder_prefill_graph_arena_bytes_, text_decoder_decode_graph_arena_bytes_, text_decoder_weight_context_bytes_, text_decoder_weight_storage_type_, stream_decode_cache_steps_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("VoxTral realtime only supports ASR");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VoxTral realtime supports offline and streaming ASR");
    }
    if (stream_batch_tokens_ <= 0) {
        throw std::runtime_error("VoxTral realtime stream_batch_tokens must be positive");
    }
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("voxtral_realtime.", 0) == 0 &&
            key != "voxtral_realtime.audio_encoder_graph_arena_mb" &&
            key != "voxtral_realtime.audio_encoder_weight_context_mb" &&
            key != "voxtral_realtime.text_decoder_prefill_graph_arena_mb" &&
            key != "voxtral_realtime.text_decoder_decode_graph_arena_mb" &&
            key != "voxtral_realtime.text_decoder_weight_context_mb" &&
            key != "voxtral_realtime.audio_encoder_weight_type" &&
            key != "voxtral_realtime.text_decoder_weight_type" &&
            key != "voxtral_realtime.stream_decode_cache_steps" &&
            key != "voxtral_realtime.stream_batch_tokens" &&
            key != "voxtral_realtime.weight_type") {
            throw std::runtime_error("unknown VoxTral realtime session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

VoxtralRealtimeSession::~VoxtralRealtimeSession() = default;

std::string VoxtralRealtimeSession::family() const {
    return "voxtral_realtime";
}

runtime::VoiceTaskKind VoxtralRealtimeSession::task_kind() const {
    return task_.task;
}

runtime::RunMode VoxtralRealtimeSession::run_mode() const {
    return task_.mode;
}

void VoxtralRealtimeSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("VoxTral realtime prepare() requires an audio contract");
    }
    mark_prepared();
}

VoxtralRealtimeRequest VoxtralRealtimeSession::make_request(
    const runtime::TaskRequest & request,
    bool streaming) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("VoxTral realtime ASR request requires audio_input");
    }
    VoxtralRealtimeRequest out;
    out.audio = *request.audio_input;
    out.streaming = streaming;
    out.generation = parse_generation_options(request.options);
    return out;
}

runtime::TaskResult VoxtralRealtimeSession::run_single(
    const VoxtralRealtimeRequest & request,
    bool first_chunk) {
    const auto wall_start = Clock::now();
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract(request.audio, first_chunk);
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.frontend_ms",
        engine::debug::elapsed_ms(frontend_start));
    const auto prompt_start = Clock::now();
    auto prompt = tokenizer_.build_transcription_prompt(
        static_cast<int64_t>(request.audio.samples.size()) / std::max(1, request.audio.channels),
        request.streaming);
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.prompt_ms",
        engine::debug::elapsed_ms(prompt_start));
    const auto audio_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode(features);
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.audio_encoder_ms",
        engine::debug::elapsed_ms(audio_start));
    const auto config_start = Clock::now();
    auto generation = request.generation;
    const int64_t max_total_tokens = audio_embeddings.tokens;
    const int64_t model_new_tokens = std::max<int64_t>(0, max_total_tokens - static_cast<int64_t>(prompt.input_ids.size()));
    generation.max_new_tokens = generation.max_new_tokens_set
        ? std::min<int64_t>(generation.max_new_tokens, model_new_tokens)
        : model_new_tokens;
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.generation_config_ms",
        engine::debug::elapsed_ms(config_start));
    const auto decode_start = Clock::now();
    const auto generated = text_decoder_.generate(prompt, audio_embeddings, generation);
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.text_decoder_ms",
        engine::debug::elapsed_ms(decode_start));
    runtime::TaskResult result;
    const auto text_start = Clock::now();
    result.text_output = runtime::Transcript{tokenizer_.decode(generated.token_ids), ""};
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.text_decode_ms",
        engine::debug::elapsed_ms(text_start));
    engine::debug::timing_log_scalar("voxtral_realtime.session.audio_frames", features.frames);
    engine::debug::timing_log_scalar("voxtral_realtime.session.audio_tokens", audio_embeddings.tokens);
    engine::debug::timing_log_scalar("voxtral_realtime.session.generated_tokens", generated.token_ids.size());
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::TaskResult VoxtralRealtimeSession::run(const runtime::TaskRequest & request) {
    require_prepared("VoxTral realtime run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VoxTral realtime run() is only available for offline sessions");
    }
    return run_single(make_request(request, false), true);
}

runtime::StreamingPolicy VoxtralRealtimeSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = frontend_.steady_stream_chunk_samples(stream_batch_tokens_);
    policy.preferred_audio_chunk_seconds =
        static_cast<double>(frontend_.steady_stream_chunk_samples(stream_batch_tokens_)) /
        static_cast<double>(assets_->config.frontend.sample_rate);
    return policy;
}

void VoxtralRealtimeSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("VoxTral realtime start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VoxTral realtime start_stream() requires a streaming session");
    }
    reset();
    streaming_generation_ = parse_generation_options(request.options);
    stream_wall_start_ = Clock::now();
    stream_started_ = true;
}

void VoxtralRealtimeSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void VoxtralRealtimeSession::reset() {
    require_prepared("VoxTral realtime reset()");
    streaming_result_ = runtime::TaskResult{};
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_audio_offset_values_ = 0;
    streaming_steps_processed_ = 0;
    stream_frontend_ms_ = 0.0;
    stream_encoder_ms_ = 0.0;
    stream_decoder_ms_ = 0.0;
    streaming_generation_ = VoxtralRealtimeGenerationOptions{};
    frontend_stream_state_ = VoxtralRealtimeFrontendStreamState{};
    audio_stream_state_ = audio_encoder_.make_stream_state();
    streaming_text_.clear();
    streaming_published_bytes_ = 0;
    streaming_token_count_ = 0;
    previous_stream_token_ = 0;
    first_stream_chunk_ = true;
    have_previous_stream_token_ = false;
    stream_started_ = false;
    stream_wall_start_ = {};
}

runtime::StreamEvent VoxtralRealtimeSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("VoxTral realtime process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VoxTral realtime process_audio_chunk() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("VoxTral realtime process_audio_chunk() requires start_stream");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;
    if (audio.channels <= 0 || audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("VoxTral streaming audio chunk has invalid channel layout");
    }
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size() && streaming_audio_offset_values_ > 0) {
        streaming_audio_.samples.clear();
        streaming_audio_offset_values_ = 0;
    }
    runtime::append_audio_buffer(streaming_audio_, audio);
    return process_available_stream_chunks();
}

runtime::TaskResult VoxtralRealtimeSession::finish_stream() {
    return finalize();
}

runtime::TaskResult VoxtralRealtimeSession::finalize() {
    const auto finalize_start = Clock::now();
    require_prepared("VoxTral realtime finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VoxTral realtime finalize() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("VoxTral realtime finalize() requires start_stream");
    }
    if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
        throw std::runtime_error("VoxTral streaming pending audio offset is out of range");
    }
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size() && streaming_token_count_ == 0) {
        throw std::runtime_error("VoxTral realtime finalize() requires streamed audio");
    }
    // Any partial this last pass produces has already gone to the sink.
    (void) process_available_stream_chunks();
    streaming_result_ = runtime::TaskResult{};
    streaming_result_.text_output = runtime::Transcript{streaming_text_, ""};
    stream_started_ = false;
    if (stream_event_sink_ != nullptr) {
        // Every token has already been delivered as a partial, so the final event only marks the
        // end of the stream. The whole transcript remains available as result.text_output.
        runtime::StreamEvent event;
        event.is_final = true;
        stream_event_sink_(event);
    }
    engine::debug::timing_log_scalar("voxtral_realtime.session.stream.tokens", streaming_token_count_);
    engine::debug::timing_log_scalar("voxtral_realtime.session.stream.steps_processed", streaming_steps_processed_);
    engine::debug::timing_log_scalar("voxtral_realtime.session.stream.frontend_ms", stream_frontend_ms_);
    engine::debug::timing_log_scalar("voxtral_realtime.session.stream.encoder_ms", stream_encoder_ms_);
    engine::debug::timing_log_scalar("voxtral_realtime.session.stream.decoder_ms", stream_decoder_ms_);
    audio_encoder_.log_stream_timings();
    text_decoder_.log_stream_timings();
    engine::debug::timing_log_scalar(
        "voxtral_realtime.session.stream.finalize_ms",
        engine::debug::elapsed_ms(finalize_start));
    if (stream_wall_start_ != std::chrono::steady_clock::time_point{}) {
        engine::debug::timing_log_scalar(
            "voxtral_realtime.session.stream.wall_ms",
            engine::debug::elapsed_ms(stream_wall_start_));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(stream_wall_start_));
    }
    return streaming_result_;
}

runtime::StreamEvent VoxtralRealtimeSession::process_available_stream_chunks() {
    runtime::StreamEvent last_event;
    last_event.is_final = false;
    if (streaming_audio_.sample_rate <= 0 || streaming_audio_.channels <= 0) {
        return last_event;
    }
    if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
        throw std::runtime_error("VoxTral streaming pending audio offset is out of range");
    }
    if (streaming_audio_.samples.size() % static_cast<size_t>(streaming_audio_.channels) != 0 ||
        streaming_audio_offset_values_ % static_cast<size_t>(streaming_audio_.channels) != 0) {
        throw std::runtime_error("VoxTral streaming pending audio has invalid channel layout");
    }
    int64_t processed_chunks = 0;
    // Only the audio source running dry ends the loop: the decoder runs one step per 80 ms of
    // audio for the whole session, so a pause in live audio never ends it.
    while (true) {
        const int64_t target_samples = first_stream_chunk_
            ? frontend_.first_stream_chunk_samples()
            : frontend_.steady_stream_chunk_samples(stream_batch_tokens_);
        const int64_t chunk_frames = source_frames_for_target_samples(
            target_samples,
            streaming_audio_.sample_rate,
            streaming_audio_.channels,
            static_cast<int>(assets_->config.frontend.sample_rate));
        const int64_t advance_target_samples = first_stream_chunk_
            ? frontend_.first_stream_chunk_advance_samples()
            : frontend_.steady_stream_chunk_advance_samples(stream_batch_tokens_);
        const int64_t advance_frames = source_frames_for_target_samples(
            advance_target_samples,
            streaming_audio_.sample_rate,
            streaming_audio_.channels,
            static_cast<int>(assets_->config.frontend.sample_rate));
        const int64_t pending_frames = static_cast<int64_t>(
            (streaming_audio_.samples.size() - streaming_audio_offset_values_) /
            static_cast<size_t>(streaming_audio_.channels));
        if (pending_frames < chunk_frames) {
            break;
        }
        if (pending_frames <= 0) {
            break;
        }
        const int64_t take_frames = chunk_frames;
        const size_t take_values = static_cast<size_t>(take_frames * streaming_audio_.channels);
        const size_t advance_values = static_cast<size_t>(advance_frames * streaming_audio_.channels);
        runtime::AudioBuffer chunk;
        chunk.sample_rate = streaming_audio_.sample_rate;
        chunk.channels = streaming_audio_.channels;
        const auto begin = streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(streaming_audio_offset_values_);
        chunk.samples.assign(begin, begin + static_cast<std::ptrdiff_t>(take_values));
        streaming_audio_offset_values_ += advance_values;
        const int64_t chunk_steps = first_stream_chunk_ ? 1 : stream_batch_tokens_;
        last_event = process_one_stream_chunk(chunk);
        streaming_steps_processed_ += chunk_steps;
        ++processed_chunks;
        if (stream_event_sink_ != nullptr && last_event.partial_text.has_value()) {
            // Several chunks can be consumed per process_audio_chunk call, so the sink is what
            // delivers a partial per step rather than only the last one. The caller forwards
            // whatever this function returns, so drop what the sink already delivered instead of
            // emitting the same partial twice.
            stream_event_sink_(last_event);
            last_event.partial_text.reset();
        }
        if (pending_frames < chunk_frames) {
            break;
        }
    }
    if (processed_chunks > 0) {
        if (streaming_audio_offset_values_ == streaming_audio_.samples.size()) {
            streaming_audio_.samples.clear();
            streaming_audio_offset_values_ = 0;
        } else if (streaming_audio_offset_values_ > 1ull * 1024ull * 1024ull &&
                   streaming_audio_offset_values_ * 2 > streaming_audio_.samples.size()) {
            streaming_audio_.samples.erase(
                streaming_audio_.samples.begin(),
                streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(streaming_audio_offset_values_));
            streaming_audio_offset_values_ = 0;
        }
    }
    return last_event;
}

runtime::StreamEvent VoxtralRealtimeSession::process_one_stream_chunk(const runtime::AudioBuffer & audio) {
    runtime::StreamEvent event;
    event.is_final = false;
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract_stream_chunk(
        audio,
        first_stream_chunk_,
        frontend_stream_state_,
        stream_batch_tokens_);
    stream_frontend_ms_ += engine::debug::elapsed_ms(frontend_start);
    const auto encoder_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode_stream_chunk(features, audio_stream_state_);
    stream_encoder_ms_ += engine::debug::elapsed_ms(encoder_start);
    const auto decoder_start = Clock::now();
    if (first_stream_chunk_) {
        auto prompt = tokenizer_.build_transcription_prompt(frontend_.first_stream_chunk_samples(), true);
        record_stream_token(text_decoder_.begin_stream(prompt, audio_embeddings, streaming_generation_));
        first_stream_chunk_ = false;
    } else if (!have_previous_stream_token_) {
        throw std::runtime_error("VoxTral streaming decoder is missing previous token");
    } else {
        // One encoder forward covers `stream_batch_tokens_` audio tokens; the decoder still runs one
        // step per token, each consuming its own row of the batch.
        for (int64_t row = 0; row < audio_embeddings.tokens; ++row) {
            record_stream_token(text_decoder_.stream_step(
                previous_stream_token_,
                audio_embeddings,
                row,
                assets_->config.default_num_delay_tokens,
                streaming_generation_));
        }
    }
    stream_decoder_ms_ += engine::debug::elapsed_ms(decoder_start);
    // Once a chunk, not once a token: the chunk reports a single event however many it decoded.
    take_stream_delta(event);
    return event;
}

void VoxtralRealtimeSession::record_stream_token(int32_t token) {
    previous_stream_token_ = token;
    have_previous_stream_token_ = true;
    // The reference implementation gives EOS no special treatment: whatever token was sampled
    // feeds back as the next text-stream input, and special ids simply emit no text. Only the
    // audio source running dry ends the stream.
    if (!tokenizer_.is_stream_text_token(token)) {
        return;
    }
    ++streaming_token_count_;
    // Tekken decode concatenates each id's bytes, so decoding token by token builds the same
    // transcript as decoding the list, at an amortized O(1) append instead of a fresh decode.
    streaming_text_ += tokenizer_.decode({token});
}

void VoxtralRealtimeSession::take_stream_delta(runtime::StreamEvent & event) {
    // Partials carry only the text decoded since the last one, as the other streaming ASR sessions
    // already emit. Restating the transcript is quadratic in its length and hands a consumer of
    // transcript.text.delta text it was already given.
    if (streaming_published_bytes_ >= streaming_text_.size()) {
        return;
    }
    event.partial_text = runtime::Transcript{streaming_text_.substr(streaming_published_bytes_), ""};
    streaming_published_bytes_ = streaming_text_.size();
}

}  // namespace engine::models::voxtral_realtime
