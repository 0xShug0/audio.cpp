#include "engine/community_models/confucius4_r2t2/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/community_models/confucius4_r2t2/text_postprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::community_models::confucius4_r2t2 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr double kOfflineChunkSeconds = 30.0;

std::shared_ptr<const R2T2ASRAssets> require_assets(std::shared_ptr<const R2T2ASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("R2T2 ASR session requires assets");
    }
    return assets;
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_audio_encoder_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error("confucius4_r2t2.audio_encoder_weight_type currently supports only native, f32, and f16");
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

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("R2T2 ASR audio requires positive channel count");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("R2T2 ASR audio samples must be divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

bool language_is_supported(const R2T2ASRAssets & assets, const std::string & language) {
    const auto & supported = assets.config.supported_languages;
    return std::find(supported.begin(), supported.end(), language) != supported.end();
}

}  // namespace

R2T2ASRSession::R2T2ASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const R2T2ASRAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"confucius4_r2t2.audio_encoder_graph_arena_mb"}, 128ull * 1024ull * 1024ull)),
      thinker_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"confucius4_r2t2.thinker_prefill_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"confucius4_r2t2.thinker_decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"confucius4_r2t2.thinker_weight_context_mb"}, 64ull * 1024ull * 1024ull)),
      audio_encoder_weight_storage_type_(option_weight_type(options, "confucius4_r2t2.audio_encoder_weight_type", engine::assets::TensorStorageType::Native)),
      thinker_weight_storage_type_(option_weight_type(
          options,
          "confucius4_r2t2.thinker_weight_type",
          option_weight_type(options, "confucius4_r2t2.weight_type", engine::assets::TensorStorageType::Native))),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_, execution_context(), audio_encoder_graph_arena_bytes_, audio_encoder_weight_storage_type_),
      thinker_(
          assets_,
          execution_context(),
          thinker_prefill_graph_arena_bytes_,
          thinker_decode_graph_arena_bytes_,
          thinker_weight_context_bytes_,
          thinker_weight_storage_type_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("R2T2 ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("R2T2 ASR supports offline and streaming sessions");
    }
    validate_audio_encoder_weight_storage(audio_encoder_weight_storage_type_);
    validate_matmul_weight_storage(thinker_weight_storage_type_, "confucius4_r2t2.thinker_weight_type");

    if (const auto value = runtime::parse_float_option(options.options, {"confucius4_r2t2.chunk_size_ms"})) {
        stream_config_.chunk_seconds = static_cast<double>(*value) / 1000.0;
    }
    if (const auto value = runtime::parse_int_option(options.options, {"confucius4_r2t2.unfixed_chunk_num"})) {
        stream_config_.unfixed_chunk_num = *value;
    }
    if (const auto value = runtime::parse_int_option(options.options, {"confucius4_r2t2.unfixed_token_num"})) {
        stream_config_.unfixed_token_num = *value;
    }
    if (const auto value = runtime::find_option(options.options, {"confucius4_r2t2.rollback_punctuation"})) {
        stream_config_.rollback_punctuation = runtime::parse_bool_option(*value, "confucius4_r2t2.rollback_punctuation");
    }
    if (const auto value = runtime::parse_int_option(options.options, {"confucius4_r2t2.max_tokens"})) {
        stream_config_.max_new_tokens = *value;
    }
    if (stream_config_.chunk_seconds <= 0.0) {
        throw std::runtime_error("confucius4_r2t2.chunk_size_ms must be positive");
    }
    if (stream_config_.unfixed_chunk_num < 0 || stream_config_.unfixed_token_num < 0) {
        throw std::runtime_error("confucius4_r2t2.unfixed_chunk_num and confucius4_r2t2.unfixed_token_num must be non-negative");
    }
    if (stream_config_.max_new_tokens <= 0) {
        throw std::runtime_error("confucius4_r2t2.max_tokens must be positive");
    }
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("confucius4_r2t2.", 0) == 0 &&
            key != "confucius4_r2t2.audio_encoder_graph_arena_mb" &&
            key != "confucius4_r2t2.thinker_prefill_graph_arena_mb" &&
            key != "confucius4_r2t2.thinker_decode_graph_arena_mb" &&
            key != "confucius4_r2t2.thinker_weight_context_mb" &&
            key != "confucius4_r2t2.audio_encoder_weight_type" &&
            key != "confucius4_r2t2.thinker_weight_type" &&
            key != "confucius4_r2t2.weight_type" &&
            key != "confucius4_r2t2.chunk_size_ms" &&
            key != "confucius4_r2t2.unfixed_chunk_num" &&
            key != "confucius4_r2t2.unfixed_token_num" &&
            key != "confucius4_r2t2.rollback_punctuation" &&
            key != "confucius4_r2t2.max_tokens") {
            throw std::runtime_error("unknown R2T2 ASR session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

R2T2ASRSession::~R2T2ASRSession() = default;

std::string R2T2ASRSession::family() const {
    return "confucius4_r2t2";
}

runtime::VoiceTaskKind R2T2ASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode R2T2ASRSession::run_mode() const {
    return task_.mode;
}

void R2T2ASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

R2T2ASRRequest R2T2ASRSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("R2T2 ASR run() requires audio_input");
    }
    R2T2ASRRequest out;
    out.audio = *request.audio_input;
    out.generation.max_new_tokens = assets_->config.max_new_tokens;
    if (request.text_input.has_value()) {
        out.context = request.text_input->text;
        out.language = request.text_input->language;
    }
    if (const auto value = runtime::find_option(request.options, {"language"})) {
        out.language = *value == "Auto" ? std::string() : *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
        if (out.generation.max_new_tokens <= 0) {
            throw std::runtime_error("R2T2 ASR max_tokens must be positive");
        }
    }
    if (!out.language.empty()) {
        out.language = resolve_language(out.language);
        if (!language_is_supported(*assets_, out.language)) {
            throw std::runtime_error("R2T2 ASR language is not supported by this model: " + out.language);
        }
    }
    return out;
}

R2T2ASRResult R2T2ASRSession::run_single(const R2T2ASRRequest & request) {
    const auto wall_start = Clock::now();
    const auto features = frontend_.extract(request.audio);
    const auto prompt = tokenizer_.build_prompt(request.context, request.language, features.encoder_tokens);
    const auto audio_embeddings = audio_encoder_.encode(features);
    const auto tokens = thinker_.generate(prompt, audio_embeddings, request.generation);
    const std::string raw = tokenizer_.decode(tokens.token_ids);

    R2T2ASRResult result;
    const auto parsed = parse_asr_output(raw, request.language);
    result.language = parsed.language.empty() ? request.language : parsed.language;
    result.text = truncate_at_pipe(parsed.text);
    debug::timing_log_scalar("confucius4_r2t2.single_ms", engine::debug::elapsed_ms(wall_start));
    debug::trace_log_scalar("confucius4_r2t2.audio_frames", features.frames);
    return result;
}

runtime::TaskResult R2T2ASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("R2T2 ASR run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("R2T2 ASR run() requires an offline session");
    }
    const auto & audio = make_request(request).audio;
    const int64_t frames = audio_frame_count(audio);
    const int64_t frames_per_chunk = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(kOfflineChunkSeconds * static_cast<double>(audio.sample_rate))));
    // Use the framework chunk planner so padding and tail alignment stay
    // consistent with the other ASR families instead of hand-rolling slices.
    const auto chunk_spans = engine::audio::plan_audio_chunks(
        frames,
        engine::audio::AudioChunkSpec{
            frames_per_chunk,
            frames_per_chunk,
            engine::audio::AudioChunkPadMode::Zero,
            engine::audio::AudioChunkTailAlignment::Start,
            0,
        });
    runtime::TaskResult merged;
    std::ostringstream text;
    for (const auto & span : chunk_spans) {
        const auto valid_span = runtime::TimeSpan{span.output_start_sample, span.output_start_sample + span.valid_samples};
        runtime::TaskRequest item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, valid_span);
        auto item = run_single(make_request(item_request));
        if (!item.text.empty()) {
            if (text.tellp() > 0) {
                text << ' ';
            }
            text << item.text;
        }
        if (!item.language.empty()) {
            if (merged.text_output == std::nullopt) {
                merged.text_output = runtime::Transcript{"", item.language};
            } else if (merged.text_output->language.empty()) {
                merged.text_output->language = item.language;
            }
        }
    }
    if (merged.text_output == std::nullopt) {
        merged.text_output = runtime::Transcript{"", ""};
    }
    merged.text_output->text = text.str();
    return merged;
}

std::string R2T2ASRSession::generate_text(
    const R2T2ASRPrompt & prompt,
    const R2T2ASRAudioEmbeddings & embeddings) {
    R2T2ASRGenerationOptions options;
    options.max_new_tokens = stream_config_.max_new_tokens;
    options.reuse_graphs = true;
    const auto tokens = thinker_.generate(prompt, embeddings, options);
    return tokenizer_.decode(tokens.token_ids);
}

std::string R2T2ASRSession::decode_rollback_prefix(
    const std::vector<int32_t> & ids,
    int64_t rollback) const {
    // Mirrors the reference U+FFFD rollback loop: grow the rollback until the
    // decoded prefix contains no replacement character.
    int64_t k = rollback;
    while (true) {
        const int64_t end_index = std::max<int64_t>(0, static_cast<int64_t>(ids.size()) - k);
        std::string prefix;
        if (end_index > 0) {
            prefix = sanitize_utf8_lossy(tokenizer_.decode(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(end_index))));
        }
        if (prefix.find(kReplacementChar) == std::string::npos) {
            return prefix;
        }
        if (end_index == 0) {
            return {};
        }
        ++k;
    }
}

std::string R2T2ASRSession::build_stream_prefix(bool final_flush) const {
    if (chunk_id_ < stream_config_.unfixed_chunk_num) {
        return {};
    }
    const std::string raw_truncated = truncate_at_pipe(raw_decoded_);
    const auto ids = tokenizer_.encode(raw_truncated);
    if (final_flush) {
        // finish_streaming_transcribe uses a fixed rollback without the
        // replacement-character loop and never rolls back past the first token.
        const int64_t end_index = std::max<int64_t>(1, static_cast<int64_t>(ids.size()) - stream_config_.unfixed_token_num);
        return truncate_at_pipe(sanitize_utf8_lossy(tokenizer_.decode(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(end_index)))));
    }
    int64_t k = stream_config_.unfixed_token_num;
    if (stream_config_.rollback_punctuation && ends_with_rollback_punctuation(raw_truncated)) {
        k = 0;
    }
    return truncate_at_pipe(decode_rollback_prefix(ids, k));
}

R2T2ASRSession::StreamOutcome R2T2ASRSession::decode_stream_chunk(bool final_flush) {
    StreamOutcome outcome;
    const std::string prefix = build_stream_prefix(final_flush);

    runtime::AudioBuffer accum;
    accum.sample_rate = stream_sample_rate_ > 0 ? stream_sample_rate_ : assets_->config.sample_rate;
    accum.channels = stream_channels_;
    accum.samples = audio_accum_;
    const auto features = frontend_.extract(accum);
    const auto prompt = tokenizer_.build_raw_audio_prompt(prompt_raw_ + prefix, features.encoder_tokens);
    const auto embeddings = audio_encoder_.encode(features, /*reuse_graph=*/true);
    std::string generated = generate_text(prompt, embeddings);
    generated = normalize_punct_by_context(generated);
    generated = sanitize_utf8_lossy(generated);
    // Remove U+FFFD replacement characters, mirroring .replace('\ufffd', '').
    for (size_t pos = 0; (pos = generated.find(kReplacementChar, pos)) != std::string::npos;) {
        generated.erase(pos, std::char_traits<char>::length(kReplacementChar));
    }

    raw_decoded_ = prefix + generated;

    std::string detected;
    if (force_language_.empty()) {
        detected = parse_language_output(raw_decoded_, std::string()).language;
    }
    if (force_language_ == "Chinese" || detected == "Chinese") {
        raw_decoded_ = remove_spaces_between_chinese(raw_decoded_);
    }
    const auto parsed = parse_asr_output(raw_decoded_, force_language_);
    if (contains_asr_text_tag(raw_decoded_)) {
        raw_decoded_ = text_before_asr_tag(raw_decoded_) + kAsrTextTag + parsed.text;
    } else {
        raw_decoded_ = parsed.text;
    }
    raw_decoded_ = truncate_at_pipe(raw_decoded_);

    const auto current_ids = tokenizer_.encode(raw_decoded_);
    int64_t k = stream_config_.unfixed_token_num;
    if (stream_config_.rollback_punctuation && ends_with_rollback_punctuation(raw_decoded_)) {
        k = 0;
    }
    if (contains_asr_text_tag(raw_decoded_) && text_after_asr_tag(raw_decoded_).empty()) {
        k = 0;
    }
    std::string fixed_text = decode_rollback_prefix(current_ids, k);
    if (contains_asr_text_tag(fixed_text)) {
        fixed_text = text_after_asr_tag(fixed_text);
    } else if (force_language_.empty()) {
        // Rollback may remove the separator even when raw_decoded_ has it.
        // Until the stable prefix reaches <asr_text>, it is only metadata.
        fixed_text.clear();
    }
    fixed_text = truncate_at_pipe(fixed_text);

    if (!contains_asr_text_tag(raw_decoded_) && force_language_.empty()) {
        // The model has not emitted the language tag yet: nothing to commit.
        text_.clear();
        debug::trace_log_scalar("confucius4_r2t2.stream.final_flush", final_flush ? 1 : 0);
        debug::trace_log_scalar("confucius4_r2t2.stream.chunk_id", chunk_id_);
        debug::trace_log_scalar("confucius4_r2t2.stream.raw_decoded", raw_decoded_);
        debug::trace_log_scalar("confucius4_r2t2.stream.fixed_text", std::string_view{});
        debug::trace_log_scalar("confucius4_r2t2.stream.text", std::string_view{});
        return outcome;
    }

    language_ = parsed.language;
    text_ = truncate_at_pipe(parsed.text);
    ++chunk_id_;
    outcome.text = text_;
    outcome.fixed_text = fixed_text;
    debug::trace_log_scalar("confucius4_r2t2.stream.final_flush", final_flush ? 1 : 0);
    debug::trace_log_scalar("confucius4_r2t2.stream.chunk_id", chunk_id_);
    debug::trace_log_scalar("confucius4_r2t2.stream.raw_decoded", raw_decoded_);
    debug::trace_log_scalar("confucius4_r2t2.stream.fixed_text", fixed_text);
    debug::trace_log_scalar("confucius4_r2t2.stream.text", text_);
    return outcome;
}

void R2T2ASRSession::publish_stream_delta(const std::string & fixed_text, runtime::StreamEvent & event) {
    // fixed_text contains transcript text only; metadata must never advance
    // this code-point offset. Stable transcript prefixes may still shrink
    // between chunks, so only publish newly committed code points.
    const size_t length = utf8_codepoint_count(fixed_text);
    if (length <= published_codepoints_) {
        return;
    }
    runtime::Transcript transcript;
    transcript.text = utf8_slice_from_codepoint(fixed_text, published_codepoints_);
    if (transcript.text.empty()) {
        return;
    }
    transcript.language = language_;
    event.partial_text = std::move(transcript);
    published_codepoints_ = length;
}

runtime::StreamingPolicy R2T2ASRSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_seconds = stream_config_.chunk_seconds;
    return policy;
}

void R2T2ASRSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("R2T2 ASR start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("R2T2 ASR start_stream() requires a streaming session");
    }
    reset();
    streaming_request_ = request;
    if (streaming_request_.audio_input.has_value()) {
        streaming_request_.audio_input->samples.clear();
    }
    context_ = streaming_request_.text_input.has_value() ? streaming_request_.text_input->text : std::string();
    force_language_ = streaming_request_.text_input.has_value() ? streaming_request_.text_input->language : std::string();
    if (const auto value = runtime::find_option(streaming_request_.options, {"language"})) {
        force_language_ = *value == "Auto" ? std::string() : *value;
    }
    if (!force_language_.empty()) {
        force_language_ = resolve_language(force_language_);
        if (!language_is_supported(*assets_, force_language_)) {
            throw std::runtime_error("R2T2 ASR language is not supported by this model: " + force_language_);
        }
    }
    prompt_raw_ = tokenizer_.build_prompt_text(context_, force_language_);
    stream_started_ = true;
    stream_wall_start_ = Clock::now();
}

void R2T2ASRSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void R2T2ASRSession::reset() {
    require_prepared("R2T2 ASR reset()");
    streaming_request_ = runtime::TaskRequest{};
    streaming_result_ = runtime::TaskResult{};
    prompt_raw_.clear();
    force_language_.clear();
    context_.clear();
    language_.clear();
    text_.clear();
    raw_decoded_.clear();
    buffer_.clear();
    audio_accum_.clear();
    chunk_size_samples_ = 0;
    chunk_id_ = 0;
    published_codepoints_ = 0;
    stream_sample_rate_ = 0;
    stream_channels_ = 1;
    stream_started_ = false;
    stream_wall_start_ = {};
}

runtime::StreamEvent R2T2ASRSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("R2T2 ASR process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("R2T2 ASR process_audio_chunk() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("R2T2 ASR process_audio_chunk() requires start_stream");
    }
    if (chunk.sample_rate <= 0 || chunk.channels <= 0 ||
        chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::runtime_error("R2T2 ASR streaming audio chunk has invalid layout");
    }
    if (chunk_size_samples_ == 0) {
        stream_sample_rate_ = chunk.sample_rate;
        stream_channels_ = chunk.channels;
        chunk_size_samples_ = std::max<int64_t>(
            1,
            static_cast<int64_t>(std::llround(stream_config_.chunk_seconds * static_cast<double>(chunk.sample_rate))));
    } else if (chunk.sample_rate != stream_sample_rate_ || chunk.channels != stream_channels_) {
        // Chunk boundaries are counted in frames of the stream's first chunk;
        // a mid-stream format change would silently corrupt the slicing.
        throw std::runtime_error(
            "R2T2 ASR streaming audio format changed mid-stream (sample rate or channel count); start a new stream instead");
    }
    buffer_.insert(buffer_.end(), chunk.samples.begin(), chunk.samples.end());

    runtime::StreamEvent event;
    event.is_final = false;
    const size_t channel_stride = static_cast<size_t>(chunk.channels);
    while (buffer_.size() >= static_cast<size_t>(chunk_size_samples_) * channel_stride) {
        const size_t take_values = static_cast<size_t>(chunk_size_samples_) * channel_stride;
        audio_accum_.insert(audio_accum_.end(), buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(take_values));
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(take_values));
        const auto outcome = decode_stream_chunk(/*final_flush=*/false);
        if (!outcome.fixed_text.empty()) {
            publish_stream_delta(outcome.fixed_text, event);
        }
        if (!streaming_result_.text_output.has_value()) {
            streaming_result_.text_output = runtime::Transcript{outcome.text, language_};
        } else {
            streaming_result_.text_output->text = outcome.text;
            if (!language_.empty()) {
                streaming_result_.text_output->language = language_;
            }
        }
        if (stream_event_sink_ != nullptr && event.partial_text.has_value()) {
            stream_event_sink_(event);
            event.partial_text.reset();
        }
    }
    if (stream_event_sink_ != nullptr && event.partial_text.has_value()) {
        stream_event_sink_(event);
        event.partial_text.reset();
    }
    return event;
}

runtime::TaskResult R2T2ASRSession::finish_stream() {
    return finalize();
}

runtime::TaskResult R2T2ASRSession::finalize() {
    const auto finalize_start = Clock::now();
    require_prepared("R2T2 ASR finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("R2T2 ASR finalize() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("R2T2 ASR finalize() requires start_stream");
    }
    if (!buffer_.empty()) {
        audio_accum_.insert(audio_accum_.end(), buffer_.begin(), buffer_.end());
        buffer_.clear();
        const auto outcome = decode_stream_chunk(/*final_flush=*/true);
        if (!streaming_result_.text_output.has_value()) {
            streaming_result_.text_output = runtime::Transcript{outcome.text, language_};
        } else {
            streaming_result_.text_output->text = outcome.text;
            if (!language_.empty()) {
                streaming_result_.text_output->language = language_;
            }
        }
    }
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output = runtime::Transcript{text_, language_};
    }
    if (stream_event_sink_ != nullptr) {
        // The final transcript travels in the task result (the server emits it
        // as transcript.text.done); the reference integrator adds no final
        // delta here either.
        runtime::StreamEvent event;
        event.is_final = true;
        stream_event_sink_(event);
    }
    stream_started_ = false;
    debug::timing_log_scalar("confucius4_r2t2.session.stream.chunks", chunk_id_);
    debug::timing_log_scalar("confucius4_r2t2.session.stream.finalize_ms", engine::debug::elapsed_ms(finalize_start));
    if (stream_wall_start_ != std::chrono::steady_clock::time_point{}) {
        debug::timing_log_scalar("confucius4_r2t2.session.stream.wall_ms", engine::debug::elapsed_ms(stream_wall_start_));
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(stream_wall_start_));
    }
    return streaming_result_;
}

// Loading adapter: confucius4_r2t2 uses the schema-v1 spec-backed loader, so the loader
// wiring stays beside the session it constructs (no per-model loader.{h,cpp}).
std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_r2t2_loader() {
    runtime::SpecBackedVoiceModelConfig<R2T2ASRAssets> config;
    config.family = "confucius4_r2t2";
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_confucius4_r2t2_assets(model_path);
    };
    config.create_session = [](const runtime::TaskSpec & task,
                               const runtime::SessionOptions & options,
                               std::shared_ptr<const R2T2ASRAssets> assets,
                               std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        (void) contract;
        return std::make_unique<R2T2ASRSession>(task, options, std::move(assets));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::confucius4_r2t2
