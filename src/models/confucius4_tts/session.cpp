#include "engine/models/confucius4_tts/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::confucius4_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "confucius4_tts";

std::shared_ptr<const ConfuciusAssets> require_assets(std::shared_ptr<const ConfuciusAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Confucius4-TTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Confucius4-TTS session requires a model contract");
    }
    return contract;
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

ConfuciusReferenceIdentity reference_identity(const ConfuciusVoiceReference & reference) {
    if (!reference.cache_id.empty()) {
        return {"id:" + reference.cache_id};
    }
    if (!reference.audio.has_value()) {
        throw std::runtime_error("Confucius4-TTS cached_voice_id was not prepared");
    }
    const auto & audio = *reference.audio;
    return {
        "audio:" + std::to_string(audio.sample_rate) + ":" +
        std::to_string(audio.channels) + ":" +
        std::to_string(audio.samples.size()) + ":" +
        std::to_string(hash_audio_samples(audio)),
    };
}

std::size_t reference_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultReferenceCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"confucius4_tts.reference_cache_slots"})
        .value_or(kDefaultReferenceCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("confucius4_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<uint64_t>(slots) > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("confucius4_tts.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"confucius4_tts.mem_saver"})) {
        return runtime::parse_bool_option(*value, "confucius4_tts.mem_saver");
    }
    return false;
}

runtime::AudioBuffer merge_confucius_audio_chunks(
    const std::vector<runtime::AudioBuffer> & chunks,
    float cross_fade_duration_sec) {
    if (chunks.empty()) {
        return {};
    }
    if (chunks.size() == 1) {
        return chunks.front();
    }
    runtime::AudioBuffer merged = chunks.front();
    const int64_t total_frames = static_cast<int64_t>(
        static_cast<double>(std::max(0.0F, cross_fade_duration_sec)) *
        static_cast<double>(merged.sample_rate));
    const int64_t fade_frames = total_frames / 3;
    const int64_t silence_frames = fade_frames;
    for (size_t chunk_index = 1; chunk_index < chunks.size(); ++chunk_index) {
        runtime::AudioBuffer next = chunks[chunk_index];
        if (next.sample_rate != merged.sample_rate || next.channels != merged.channels) {
            throw std::runtime_error("Confucius4-TTS segment audio format mismatch");
        }
        const int64_t merged_frames = static_cast<int64_t>(merged.samples.size()) / merged.channels;
        const int64_t next_frames = static_cast<int64_t>(next.samples.size()) / next.channels;
        const int64_t fade_out = std::min(fade_frames, merged_frames);
        const int64_t fade_in = std::min(fade_frames, next_frames);
        for (int64_t i = 0; i < fade_out; ++i) {
            const float weight = fade_out == 1 ? 1.0F : 1.0F - static_cast<float>(i) / static_cast<float>(fade_out - 1);
            for (int channel = 0; channel < merged.channels; ++channel) {
                merged.samples[static_cast<size_t>((merged_frames - fade_out + i) * merged.channels + channel)] *= weight;
            }
        }
        for (int64_t i = 0; i < fade_in; ++i) {
            const float weight = fade_in == 1 ? 0.0F : static_cast<float>(i) / static_cast<float>(fade_in - 1);
            for (int channel = 0; channel < next.channels; ++channel) {
                next.samples[static_cast<size_t>(i * next.channels + channel)] *= weight;
            }
        }
        merged.samples.insert(
            merged.samples.end(),
            static_cast<size_t>(silence_frames * merged.channels),
            0.0F);
        runtime::append_audio_buffer(merged, next);
    }
    return merged;
}

}  // namespace

ConfuciusSession::ConfuciusSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ConfuciusAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(assets_),
      reference_cache_(reference_cache_slots_from_options(options)) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Confucius4-TTS");
    graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"confucius4_tts.graph_arena_mb"},
        graph_arena_bytes_);
    weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"confucius4_tts.weight_context_mb"},
        weight_context_bytes_);
    using T = engine::assets::TensorStorageType;
    matmul_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "confucius4_tts.weight_type",
        matmul_weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    conv_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "confucius4_tts.conv_weight_type",
        conv_weight_storage_type_,
        {T::Native, T::F32, T::F16});
    mem_saver_ = mem_saver_from_options(options);
    if (task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("Confucius4-TTS supports the VoiceCloning task");
    }

    semantic_encoder_ = std::make_unique<ConfuciusWav2Vec2BertRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    style_encoder_ = std::make_unique<ConfuciusStyleEncoder>(
        assets_,
        options.backend,
        conv_weight_storage_type_);
    t2s_ = std::make_unique<ConfuciusT2SRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    s2a_ = std::make_unique<ConfuciusS2ARuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    vocoder_ = std::make_unique<ConfuciusBigVganVocoder>(
        assets_,
        options.backend,
        conv_weight_storage_type_);
}

ConfuciusSession::~ConfuciusSession() = default;

bool ConfuciusSession::ReferenceIdentityEqual::operator()(
    const ConfuciusReferenceIdentity & lhs,
    const ConfuciusReferenceIdentity & rhs) const noexcept {
    return lhs.id == rhs.id;
}

std::string ConfuciusSession::family() const {
    return "confucius4_tts";
}

runtime::VoiceTaskKind ConfuciusSession::task_kind() const {
    return task_.task;
}

runtime::RunMode ConfuciusSession::run_mode() const {
    return task_.mode;
}

void ConfuciusSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Confucius4-TTS");
    const auto start = Clock::now();
    prepared_defaults_ = make_confucius_prepare_defaults(*assets_, request);
    debug::timing_log_scalar("confucius4_tts.prepare.request_parse_ms", engine::debug::elapsed_ms(start));
    if (prepared_defaults_.has_value() && prepared_defaults_->reference.audio.has_value()) {
        const auto voice_start = Clock::now();
        (void)resolve_voice(*prepared_defaults_);
        debug::timing_log_scalar("confucius4_tts.prepare.resolve_voice_ms", engine::debug::elapsed_ms(voice_start));
    }
    mark_prepared();
}

ConfuciusPreparedVoice ConfuciusSession::prepare_voice(const ConfuciusRequest & request) {
    if (!request.reference.audio.has_value()) {
        throw std::runtime_error("Confucius4-TTS cached voice reference was not prepared");
    }
    const auto start = Clock::now();
    const auto & audio = *request.reference.audio;
    ConfuciusPreparedVoice voice;
    auto stage_start = Clock::now();
    voice.audio = prepare_confucius_reference_audio(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        assets_->config.audio,
        static_cast<size_t>(std::max(1, options().backend.threads)));
    debug::timing_log_scalar("confucius4_tts.reference.features_ms", engine::debug::elapsed_ms(stage_start));
    stage_start = Clock::now();
    semantic_encoder_->prepare(voice.audio.semantic_features.frames);
    debug::timing_log_scalar("confucius4_tts.reference.semantic_prepare_ms", engine::debug::elapsed_ms(stage_start));
    stage_start = Clock::now();
    voice.semantic = semantic_encoder_->encode(voice.audio.semantic_features);
    debug::timing_log_scalar("confucius4_tts.reference.semantic_encode_ms", engine::debug::elapsed_ms(stage_start));
    stage_start = Clock::now();
    voice.style = style_encoder_->embed_fbank(
        voice.audio.campplus_fbank.values,
        voice.audio.campplus_fbank.frames,
        voice.audio.campplus_fbank.dims);
    debug::timing_log_scalar("confucius4_tts.reference.style_encode_ms", engine::debug::elapsed_ms(stage_start));
    if (mem_saver_) {
        stage_start = Clock::now();
        semantic_encoder_->release_graph();
        style_encoder_->release_runtime_graph();
        debug::timing_log_scalar("confucius4_tts.reference.release_graph_ms", engine::debug::elapsed_ms(stage_start));
    }
    debug::timing_log_scalar("confucius4_tts.reference_state_ms", engine::debug::elapsed_ms(start));
    return voice;
}

ConfuciusPreparedVoice & ConfuciusSession::resolve_voice(const ConfuciusRequest & request) {
    const auto identity = reference_identity(request.reference);
    if (auto * cached = reference_cache_.find(identity)) {
        debug::trace_log_scalar("confucius4_tts.reference_cache.hit", 1);
        debug::trace_log_scalar("confucius4_tts.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
        debug::trace_log_scalar("confucius4_tts.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
        return *cached;
    }
    if (!request.reference.audio.has_value()) {
        throw std::runtime_error("Confucius4-TTS cached_voice_id is not present in the session reference cache");
    }
    const bool will_evict = reference_cache_.capacity() > 0 && reference_cache_.size() >= reference_cache_.capacity();
    auto voice = prepare_voice(request);
    if (reference_cache_.capacity() == 0) {
        uncached_reference_ = std::move(voice);
        debug::trace_log_scalar("confucius4_tts.reference_cache.hit", 0);
        debug::trace_log_scalar("confucius4_tts.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
        debug::trace_log_scalar("confucius4_tts.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
        debug::trace_log_scalar("confucius4_tts.reference_cache.evicted", 0);
        return *uncached_reference_;
    }
    reference_cache_.put(identity, std::move(voice));
    debug::trace_log_scalar("confucius4_tts.reference_cache.hit", 0);
    debug::trace_log_scalar("confucius4_tts.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
    debug::trace_log_scalar("confucius4_tts.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
    debug::trace_log_scalar("confucius4_tts.reference_cache.evicted", will_evict ? 1 : 0);
    auto * cached = reference_cache_.find(identity);
    if (cached == nullptr) {
        throw std::runtime_error("Confucius4-TTS reference cache insert failed");
    }
    return *cached;
}

runtime::AudioBuffer ConfuciusSession::synthesize_segment(
    const ConfuciusTextSegment & segment,
    const ConfuciusPreparedVoice & voice,
    const ConfuciusGenerationOptions & options,
    uint64_t & rng_offset_blocks) {
    ConfuciusT2SGenerationRequest t2s_request;
    t2s_request.text_tokens = segment.token_ids;
    t2s_request.semantic_condition = voice.semantic;
    t2s_request.options = options;
    t2s_request.rng_offset_blocks = rng_offset_blocks;
    const auto segment_start = Clock::now();
    debug::trace_log_scalar("confucius4_tts.segment.text_tokens", static_cast<int64_t>(segment.token_ids.size()));
    const auto t2s_start = Clock::now();
    auto semantic = t2s_->generate(t2s_request);
    rng_offset_blocks = semantic.rng_offset_blocks;
    debug::timing_log_scalar("confucius4_tts.t2s.total_ms", engine::debug::elapsed_ms(t2s_start));
    if (mem_saver_) {
        t2s_->release_graphs();
    }
    if (semantic.semantic_codes.empty()) {
        throw std::runtime_error("Confucius4-TTS T2S generated no semantic codes");
    }
    const int64_t target_frames = static_cast<int64_t>(
        static_cast<float>(semantic.semantic_codes.size()) * 1.72F);
    debug::trace_log_scalar("confucius4_tts.segment.semantic_tokens", static_cast<int64_t>(semantic.semantic_codes.size()));
    debug::trace_log_scalar("confucius4_tts.segment.target_mel_frames", target_frames);
    const auto s2a_start = Clock::now();
    auto mel = s2a_->infer_mel(
        semantic,
        voice.audio.reference_mel,
        voice.style,
        target_frames,
        options.num_inference_steps,
        options.guidance_scale,
        options.seed,
        rng_offset_blocks);
    debug::timing_log_scalar("confucius4_tts.s2a.total_ms", engine::debug::elapsed_ms(s2a_start));
    if (mem_saver_) {
        s2a_->release_pre_cfm_graphs();
        s2a_->release_cfm_graph();
    }

    const auto vocoder_start = Clock::now();
    auto vocoded = vocoder_->synthesize(mel.values, mel.frames);
    debug::timing_log_scalar("confucius4_tts.vocoder_ms", engine::debug::elapsed_ms(vocoder_start));
    debug::trace_log_scalar("confucius4_tts.segment.mel_frames", mel.frames);
    debug::trace_log_scalar("confucius4_tts.segment.output_samples", static_cast<int64_t>(vocoded.waveform.size()));
    if (mem_saver_) {
        vocoder_->release_runtime_graph();
    }
    runtime::AudioBuffer out;
    out.sample_rate = vocoded.sample_rate;
    out.channels = 1;
    out.samples = std::move(vocoded.waveform);
    debug::timing_log_scalar("confucius4_tts.segment.total_ms", engine::debug::elapsed_ms(segment_start));
    return out;
}

runtime::AudioBuffer ConfuciusSession::synthesize(
    const ConfuciusRequest & request,
    const ConfuciusPreparedVoice & voice) {
    auto timing_start = Clock::now();
    const auto segments = tokenizer_.segment_request(request);
    debug::timing_log_scalar("confucius4_tts.text.segment_ms", engine::debug::elapsed_ms(timing_start));
    if (segments.empty()) {
        throw std::runtime_error("Confucius4-TTS text segmentation produced no segments");
    }
    std::vector<runtime::AudioBuffer> chunks;
    chunks.reserve(segments.size());
    uint64_t rng_offset_blocks = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        chunks.push_back(synthesize_segment(
            segments[i],
            voice,
            request.generation,
            rng_offset_blocks));
    }
    debug::trace_log_scalar("confucius4_tts.text.segment_count", static_cast<int64_t>(segments.size()));
    debug::trace_log_scalar("confucius4_tts.text.chunk_mode", engine::text::text_chunk_mode_name(request.generation.text_chunk_mode));
    debug::trace_log_scalar("confucius4_tts.text.max_tokens_per_segment", request.generation.max_text_tokens_per_segment);
    for (size_t i = 0; i < segments.size(); ++i) {
        const std::string prefix = "confucius4_tts.text.segment." + std::to_string(i);
        debug::trace_log_scalar(prefix + ".codepoints", engine::text::utf8_codepoint_count(segments[i].text, "Confucius4-TTS segment trace"));
        debug::trace_log_scalar(prefix + ".token_ids", static_cast<int64_t>(segments[i].token_ids.size()));
    }
    timing_start = Clock::now();
    auto merged = merge_confucius_audio_chunks(chunks, request.generation.cross_fade_duration_sec);
    debug::timing_log_scalar("confucius4_tts.audio.merge_chunks_ms", engine::debug::elapsed_ms(timing_start));
    return merged;
}

runtime::TaskResult ConfuciusSession::run(const runtime::TaskRequest & request) {
    require_prepared("Confucius4-TTS run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Confucius4-TTS");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Confucius4-TTS run requires an offline session");
    }
    const auto wall_start = Clock::now();
    auto timing_start = Clock::now();
    const auto parsed = make_confucius_request(*assets_, request, prepared_defaults_);
    debug::timing_log_scalar("confucius4_tts.request.parse_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    const auto & voice = resolve_voice(parsed);
    debug::timing_log_scalar("confucius4_tts.request.resolve_voice_ms", engine::debug::elapsed_ms(timing_start));
    runtime::TaskResult result;
    timing_start = Clock::now();
    result.audio_output = synthesize(parsed, voice);
    debug::timing_log_scalar("confucius4_tts.request.synthesize_ms", engine::debug::elapsed_ms(timing_start));
    debug::trace_log_scalar("confucius4_tts.path.language", parsed.language);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::StreamingPolicy ConfuciusSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void ConfuciusSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Confucius4-TTS streaming");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Confucius4-TTS");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Confucius4-TTS start_stream requires a streaming session");
    }
    reset();
    auto timing_start = Clock::now();
    streaming_request_ = make_confucius_request(*assets_, request, prepared_defaults_);
    debug::timing_log_scalar("confucius4_tts.streaming.request_parse_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    streaming_voice_ = &resolve_voice(*streaming_request_);
    debug::timing_log_scalar("confucius4_tts.streaming.resolve_voice_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    streaming_segments_ = tokenizer_.segment_request(*streaming_request_);
    debug::timing_log_scalar("confucius4_tts.streaming.segment_ms", engine::debug::elapsed_ms(timing_start));
    streaming_rng_offset_blocks_ = 0;
    if (streaming_segments_.empty()) {
        throw std::runtime_error("Confucius4-TTS streaming text segmentation produced no segments");
    }
    debug::trace_log_scalar("confucius4_tts.streaming.segment_count", static_cast<int64_t>(streaming_segments_.size()));
    debug::trace_log_scalar(
        "confucius4_tts.streaming.text.chunk_mode",
        engine::text::text_chunk_mode_name(streaming_request_->generation.text_chunk_mode));
    debug::trace_log_scalar(
        "confucius4_tts.streaming.text.max_tokens_per_segment",
        streaming_request_->generation.max_text_tokens_per_segment);
    for (size_t i = 0; i < streaming_segments_.size(); ++i) {
        const std::string prefix = "confucius4_tts.streaming.text.segment." + std::to_string(i);
        debug::trace_log_scalar(
            prefix + ".codepoints",
            engine::text::utf8_codepoint_count(streaming_segments_[i].text, "Confucius4-TTS streaming segment trace"));
        debug::trace_log_scalar(prefix + ".token_ids", static_cast<int64_t>(streaming_segments_[i].token_ids.size()));
    }
}

std::optional<runtime::StreamEvent> ConfuciusSession::next_stream_event() {
    if (!streaming_request_.has_value() || streaming_voice_ == nullptr) {
        throw std::runtime_error("Confucius4-TTS streaming has not been started");
    }
    if (streaming_segment_index_ >= streaming_segments_.size()) {
        return std::nullopt;
    }
    const size_t chunk_index = streaming_segment_index_++;
    const auto synthesize_start = Clock::now();
    auto chunk_audio = synthesize_segment(
        streaming_segments_[chunk_index],
        *streaming_voice_,
        streaming_request_->generation,
        streaming_rng_offset_blocks_);
    debug::timing_log_scalar("confucius4_tts.streaming.event.synthesize_ms", engine::debug::elapsed_ms(synthesize_start));
    streaming_chunks_.push_back(chunk_audio);
    runtime::StreamEvent event;
    event.named_audio_outputs.push_back({
        "segment_" + std::to_string(chunk_index),
        std::move(chunk_audio),
        {},
    });
    if (stream_sink_) {
        stream_sink_(event);
    }
    return event;
}

void ConfuciusSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}

runtime::TaskResult ConfuciusSession::finish_stream() {
    if (!streaming_request_.has_value()) {
        throw std::runtime_error("Confucius4-TTS streaming has not been started");
    }
    runtime::TaskResult result;
    result.audio_output = merge_confucius_audio_chunks(
        streaming_chunks_,
        streaming_request_->generation.cross_fade_duration_sec);
    reset();
    return result;
}

void ConfuciusSession::reset() {
    streaming_request_.reset();
    streaming_voice_ = nullptr;
    streaming_segments_.clear();
    streaming_chunks_.clear();
    streaming_segment_index_ = 0;
    streaming_rng_offset_blocks_ = 0;
}

runtime::StreamEvent ConfuciusSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("Confucius4-TTS streaming does not consume audio chunks");
}

runtime::TaskResult ConfuciusSession::finalize() {
    return finish_stream();
}

// Loading adapter: Confucius4-TTS uses the schema-v1 spec-backed loader, so the
// loader wiring stays beside the session it constructs.
std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<ConfuciusAssets> config;
    config.family = kFamily;
    config.load_assets = load_confucius_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const ConfuciusAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<ConfuciusSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::confucius4_tts
