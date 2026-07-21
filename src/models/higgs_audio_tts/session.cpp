#include "engine/models/higgs_audio_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kDefaultTextChunkSize = 1024;
constexpr int64_t kDefaultReferenceCacheSlots = 1;

void validate_matmul_weight_storage(assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

uint64_t fnv1a_mix(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_mix(hash, &bits, sizeof(bits));
    }
    return hash;
}

std::size_t resolve_reference_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"higgs_audio_tts.reference_cache_slots", "reference_cache_slots"})
        .value_or(kDefaultReferenceCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("higgs_audio_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("higgs_audio_tts.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

const runtime::AudioBuffer * find_reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value()
        && request.voice->speaker.has_value()
        && request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) {
        return &*request.audio_input;
    }
    return nullptr;
}

HiggsGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const HiggsConfig & config) {
    HiggsGenerationOptions options;
    options.max_tokens = 2048;
    options.temperature = 0.8F;
    options.top_p = 0.8F;
    options.top_k = 30;
    options.repetition_penalty = 1.1F;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value < 0) {
            throw std::runtime_error("Higgs TTS max_tokens must be non-negative");
        }
        if (*value > 0) {
            options.max_tokens = *value;
        }
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) {
        options.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        options.seed = *value;
    }
    if (options.max_tokens > config.text.max_position_embeddings) {
        throw std::runtime_error("Higgs TTS max_tokens exceeds model max_position_embeddings");
    }
    if (!(options.repetition_penalty > 0.0F) || !std::isfinite(options.repetition_penalty)) {
        throw std::runtime_error("Higgs TTS repetition_penalty must be finite and positive");
    }
    return options;
}

}  // namespace

HiggsTTSSession::HiggsTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const HiggsAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      reference_cache_(resolve_reference_cache_slots(this->options())) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs TTS session requires assets");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS currently supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS only supports the Tts task");
    }
    ar_weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.ar_weight_context_mb"}, ar_weight_context_bytes_);
    codec_weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.codec_weight_context_mb"}, codec_weight_context_bytes_);
    ar_decode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.ar_decode_graph_arena_mb"}, ar_decode_graph_arena_bytes_);
    codec_decode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.codec_decode_graph_arena_mb"}, codec_decode_graph_arena_bytes_);
    codec_encode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.codec_encode_graph_arena_mb"}, codec_encode_graph_arena_bytes_);

    if (const auto it = options.options.find("higgs_audio_tts.weight_type"); it != options.options.end()) {
        const auto storage_type = assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "higgs_audio_tts.weight_type");
        ar_weight_storage_type_ = storage_type;
        codec_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("higgs_audio_tts.ar_weight_type"); it != options.options.end()) {
        ar_weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(ar_weight_storage_type_, "higgs_audio_tts.ar_weight_type");
    }
    if (const auto it = options.options.find("higgs_audio_tts.codec_weight_type"); it != options.options.end()) {
        codec_weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(codec_weight_storage_type_, "higgs_audio_tts.codec_weight_type");
    }
    for (const auto & [key, _] : options.options) {
        if (key.rfind("higgs_audio_tts.", 0) == 0 &&
            key != "higgs_audio_tts.ar_weight_context_mb" &&
            key != "higgs_audio_tts.codec_weight_context_mb" &&
            key != "higgs_audio_tts.ar_decode_graph_arena_mb" &&
            key != "higgs_audio_tts.codec_decode_graph_arena_mb" &&
            key != "higgs_audio_tts.codec_encode_graph_arena_mb" &&
            key != "higgs_audio_tts.reference_cache_slots" &&
            key != "higgs_audio_tts.weight_type" &&
            key != "higgs_audio_tts.ar_weight_type" &&
            key != "higgs_audio_tts.codec_weight_type") {
            throw std::runtime_error("unknown Higgs TTS session option: " + key);
        }
    }

    ar_ = std::make_shared<HiggsARRuntime>(
        assets_,
        execution_context(),
        ar_weight_context_bytes_,
        ar_weight_storage_type_);
    codec_ = std::make_shared<HiggsCodecRuntime>(
        assets_,
        execution_context(),
        codec_weight_context_bytes_,
        codec_decode_graph_arena_bytes_,
        codec_encode_graph_arena_bytes_,
        codec_weight_storage_type_);
    generator_ = std::make_unique<HiggsGenerator>(
        assets_,
        ar_,
        codec_,
        ar_decode_graph_arena_bytes_);
}

std::string HiggsTTSSession::family() const {
    return "higgs_audio_tts";
}

runtime::VoiceTaskKind HiggsTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HiggsTTSSession::run_mode() const {
    return task_.mode;
}

void HiggsTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value()) {
        runtime::TaskRequest task_request;
        task_request.text_input = request.text;
        task_request.voice = request.voice;
        task_request.options = request.options;
        const auto generation_request = make_generation_request(task_request);
        generator_->prepare(generation_request);
    }
    mark_prepared();
}

runtime::TaskResult HiggsTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Higgs TTS run");
    const auto wall_start = Clock::now();
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
    const std::string reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
    const auto * reference_audio = find_reference_audio(request);
    const HiggsCodecEncodeOutput * reference_codes =
        reference_audio != nullptr ? &resolve_reference_codes(*reference_audio, reference_text) : nullptr;
    debug::trace_log_scalar("higgs_audio_tts.text_chunk_size", text_chunk_size);
    debug::trace_log_scalar("higgs_audio_tts.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    debug::trace_log_scalar("higgs_audio_tts.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));

    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const auto generation_request = make_generation_request(chunk_request, reference_codes);
        auto result = generator_->generate(generation_request);
        runtime::append_audio_buffer(merged_audio, runtime::AudioBuffer{
            result.audio.sample_rate,
            result.audio.channels,
            std::move(result.audio.values),
        });
    }

    runtime::TaskResult out;
    out.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return out;
}

const HiggsCodecEncodeOutput & HiggsTTSSession::resolve_reference_codes(
    const runtime::AudioBuffer & audio,
    const std::string & reference_text) {
    const uint64_t sample_count = static_cast<uint64_t>(audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(audio);
    ReferenceCacheKey key;
    key.reference_text = reference_text;
    key.sample_rate = audio.sample_rate;
    key.channels = audio.channels;
    key.sample_count = sample_count;
    key.sample_hash = sample_hash;
    debug::trace_log_scalar("higgs_audio_tts.reference_audio.sample_rate", audio.sample_rate);
    debug::trace_log_scalar("higgs_audio_tts.reference_audio.channels", audio.channels);
    debug::trace_log_f32("higgs_audio_tts.reference_audio.samples",
                         {static_cast<int64_t>(audio.samples.size())},
                         audio.samples);
    debug::trace_log_scalar("higgs_audio_tts.reference_cache.capacity", static_cast<int64_t>(reference_cache_.capacity()));
    debug::trace_log_scalar("higgs_audio_tts.reference_cache.size", static_cast<int64_t>(reference_cache_.size()));
    if (const auto * cached = reference_cache_.find(key)) {
        debug::trace_log_scalar("higgs_audio_tts.reference_cache.hit", 1);
        return cached->codes;
    }
    debug::trace_log_scalar("higgs_audio_tts.reference_cache.hit", 0);

    const auto encode_start = Clock::now();
    ReferenceCacheEntry entry;
    entry.codes = codec_->encode_reference(audio);
    codec_->release_encode_graph();
    debug::trace_log_scalar("higgs_audio_tts.reference_codes.frames", entry.codes.frames);
    debug::trace_log_scalar("higgs_audio_tts.reference_codes.codebooks", entry.codes.codebooks);
    debug::trace_log_i32("higgs_audio_tts.reference_codes.values",
                         {entry.codes.frames, entry.codes.codebooks},
                         entry.codes.codes);
    if (reference_cache_.capacity() == 0) {
        uncached_reference_ = std::move(entry);
        debug::timing_log_scalar("higgs_audio_tts.codec.encode_reference_ms", engine::debug::elapsed_ms(encode_start));
        return uncached_reference_->codes;
    }
    reference_cache_.put(key, std::move(entry));
    debug::timing_log_scalar("higgs_audio_tts.codec.encode_reference_ms", engine::debug::elapsed_ms(encode_start));
    return reference_cache_.find(key)->codes;
}

HiggsGenerationRequest HiggsTTSSession::make_generation_request(
    const runtime::TaskRequest & request,
    const HiggsCodecEncodeOutput * resolved_reference_codes) {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Higgs TTS requires text input");
    }
    const std::string reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");

    HiggsGenerationRequest out;
    out.text = request.text_input->text;
    out.reference_text = reference_text;
    out.options = generation_options_from_request(request, assets_->config);
    if (resolved_reference_codes != nullptr) {
        out.reference_codes = resolved_reference_codes->codes;
        out.reference_frames = resolved_reference_codes->frames;
        out.reference_codebooks = resolved_reference_codes->codebooks;
    } else if (const auto * reference_audio = find_reference_audio(request)) {
        const auto & reference_codes = resolve_reference_codes(*reference_audio, reference_text);
        out.reference_codes = reference_codes.codes;
        out.reference_frames = reference_codes.frames;
        out.reference_codebooks = reference_codes.codebooks;
    }
    return out;
}

}  // namespace engine::models::higgs_audio_tts
