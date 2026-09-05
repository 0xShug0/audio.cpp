#include "engine/community_models/mira_tts/session.h"

#include "engine/community_models/mira_tts/decoder.h"
#include "engine/community_models/mira_tts/generator.h"
#include "engine/community_models/mira_tts/processor.h"
#include "engine/community_models/mira_tts/prompt.h"
#include "engine/community_models/mira_tts/speaker_encoder.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::community_models::mira_tts {
namespace {

constexpr const char * kFamily = "mira_tts";
constexpr size_t kGraphBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kWeightBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultReferenceCacheSlots = 1;

using Clock = std::chrono::steady_clock;

std::shared_ptr<const MiraTTSAssets> require_assets(
    std::shared_ptr<const MiraTTSAssets> value) {
    if (value == nullptr) throw std::runtime_error("MiraTTS session requires assets");
    return value;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> value) {
    if (value == nullptr) throw std::runtime_error("MiraTTS session requires a model contract");
    return value;
}

MiraGenerationOptions generation_options(const runtime::TaskRequest & request) {
    MiraGenerationOptions out;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        out.max_new_tokens = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"min_p"})) {
        out.min_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(
            request.options, {"repetition_penalty"})) {
        out.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
        out.has_seed = true;
    }
    if (!out.has_seed) out.seed = runtime::random_u64_seed();
    if (out.max_new_tokens < 1 || out.top_k < 1 || out.temperature <= 0.0F ||
        out.top_p <= 0.0F || out.top_p > 1.0F || out.min_p < 0.0F ||
        out.min_p > 1.0F || out.repetition_penalty < 1.0F) {
        throw std::runtime_error("MiraTTS generation options are outside their valid ranges");
    }
    return out;
}

size_t reference_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"mira_tts.reference_cache_slots", "reference_cache_slots"})
        .value_or(static_cast<int64_t>(kDefaultReferenceCacheSlots));
    if (slots < 0) {
        throw std::runtime_error(
            "mira_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<uint64_t>(slots) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(
            "mira_tts.reference_cache_slots is too large");
    }
    return static_cast<size_t>(slots);
}

uint64_t mix_reference_hash(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MiraTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MiraTTSOfflineSession>(
        task, options, std::move(assets), std::move(contract));
}

}  // namespace

MiraTTSOfflineSession::MiraTTSOfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MiraTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      reference_cache_(reference_cache_slots(this->options())) {
    runtime::validate_spec_backed_session_options(
        options, *contract_, kFamily, "MiraTTS");
    if ((task.mode != runtime::RunMode::Offline &&
         task.mode != runtime::RunMode::Streaming) ||
        (task.task != runtime::VoiceTaskKind::Tts &&
         task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error(
            "MiraTTS supports offline and streaming TTS/voice cloning only");
    }
    const auto lm_type = runtime::parse_tensor_storage_option(
        options.options, "backbone_weight_type", assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto linear_type = runtime::parse_tensor_storage_option(
        options.options, "linear_weight_type", assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto conv_type = runtime::parse_tensor_storage_option(
        options.options, "conv_weight_type", assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16});
    auto & execution = execution_context();
    auto * generator_execution = &execution;
    if (execution.backend_type() == core::BackendType::Vulkan) {
        // Mira's Qwen language-model graph is not numerically reliable on the
        // current Vulkan backend. Keep the acoustic stages accelerated while
        // generating speech codes on CPU so Vulkan never returns corrupt audio.
        core::BackendConfig cpu_config;
        cpu_config.type = core::BackendType::Cpu;
        cpu_config.threads = std::max(1, execution.config().threads);
        generator_execution_ = std::make_unique<core::ExecutionContext>(cpu_config);
        generator_execution = generator_execution_.get();
    }
    prompt_ = std::make_unique<MiraPromptBuilder>(assets_);
    speaker_encoder_ = std::make_unique<MiraSpeakerEncoder>(
        *assets_, execution, kWeightBytes, kGraphBytes, linear_type, conv_type);
    generator_ = std::make_unique<MiraGenerator>(
        *assets_, *generator_execution, kGraphBytes, kGraphBytes, kWeightBytes, lm_type);
    processor_ = std::make_unique<MiraAcousticProcessor>(
        *assets_, execution, kWeightBytes, kGraphBytes, linear_type, conv_type);
    // ggml's current CUDA ConvTranspose1d kernel requires F32 weights.
    decoder_ = std::make_unique<MiraDecoder>(
        *assets_, execution, kWeightBytes, kGraphBytes,
        assets::TensorStorageType::F32);
}

MiraTTSOfflineSession::~MiraTTSOfflineSession() = default;

std::string MiraTTSOfflineSession::family() const { return kFamily; }

runtime::VoiceTaskKind MiraTTSOfflineSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MiraTTSOfflineSession::run_mode() const { return task_.mode; }

void MiraTTSOfflineSession::prepare(
    const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(
        request.options, *contract_, "MiraTTS");
    prepared_reference_.reset();
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        prepared_reference_ = *request.voice->speaker->audio;
        (void)context_codes(*prepared_reference_);
    }
    mark_prepared();
}

bool MiraTTSOfflineSession::ReferenceCacheKeyEqual::operator()(
    const ReferenceCacheKey & lhs,
    const ReferenceCacheKey & rhs) const noexcept {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

MiraTTSOfflineSession::ReferenceCacheKey
MiraTTSOfflineSession::make_reference_cache_key(
    const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = mix_reference_hash(hash, static_cast<uint64_t>(bits));
    }
    return ReferenceCacheKey{
        audio.sample_rate,
        audio.channels,
        static_cast<uint64_t>(audio.samples.size()),
        hash,
    };
}

const runtime::AudioBuffer & MiraTTSOfflineSession::reference_audio(
    const runtime::TaskRequest & request) const {
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return *request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) return *request.audio_input;
    if (prepared_reference_.has_value()) return *prepared_reference_;
    throw std::runtime_error(
        "MiraTTS requires a reference voice in voice.speaker.audio or audio_input");
}

const std::vector<int32_t> & MiraTTSOfflineSession::context_codes(
    const runtime::AudioBuffer & reference) {
    const auto key_start = Clock::now();
    auto key = make_reference_cache_key(reference);
    engine::debug::timing_log_scalar(
        "mira_tts.reference.hash_ms", engine::debug::elapsed_ms(key_start));
    if (const auto * cached = reference_cache_.find(key)) {
        engine::debug::trace_log_scalar("mira_tts.reference.cache_hit", true);
        return *cached;
    }

    engine::debug::trace_log_scalar("mira_tts.reference.cache_hit", false);
    const auto encode_start = Clock::now();
    auto encoded = speaker_encoder_->encode(reference);
    engine::debug::timing_log_scalar(
        "mira_tts.reference.encode_ms", engine::debug::elapsed_ms(encode_start));
    if (reference_cache_.capacity() == 0) {
        uncached_context_codes_ = std::move(encoded);
        return *uncached_context_codes_;
    }
    reference_cache_.put(key, std::move(encoded));
    return *reference_cache_.find(key);
}

runtime::TaskResult MiraTTSOfflineSession::run(
    const runtime::TaskRequest & request) {
    require_prepared("MiraTTS run");
    runtime::validate_spec_backed_request_options(
        request.options, *contract_, "MiraTTS");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MiraTTS requires non-empty text input");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MiraTTS run requires an offline session");
    }
    const auto & codes = context_codes(reference_audio(request));
    runtime::TaskResult result;
    result.audio_output = synthesize_text(
        request.text_input->text, codes, generation_options(request));
    return result;
}

runtime::AudioBuffer MiraTTSOfflineSession::synthesize_text(
    const std::string & text,
    const std::vector<int32_t> & context_codes,
    const MiraGenerationOptions & options) {
    const auto prompt_start = Clock::now();
    const auto prompt_ids = prompt_->build(text, context_codes);
    engine::debug::timing_log_scalar(
        "mira_tts.prompt_ms", engine::debug::elapsed_ms(prompt_start));
    if (generator_execution_ ||
        execution_context().backend_type() == core::BackendType::Cpu) {
        // The CPU decode graph may otherwise reuse a larger cache allocation
        // after a long request. Rebuilding its graph preserves deterministic
        // seeded output; model weights remain resident across this reset.
        generator_->release_runtime_graphs();
    }
    const auto generator_start = Clock::now();
    const auto speech_codes = generator_->generate(
        prompt_ids, options);
    engine::debug::timing_log_scalar(
        "mira_tts.generator_ms", engine::debug::elapsed_ms(generator_start));
    if (speech_codes.empty()) {
        throw std::runtime_error("MiraTTS generated no speech tokens");
    }
    const auto processor_start = Clock::now();
    const auto latents = processor_->process(speech_codes, context_codes);
    engine::debug::timing_log_scalar(
        "mira_tts.processor_ms", engine::debug::elapsed_ms(processor_start));
    const auto decoder_start = Clock::now();
    auto audio = decoder_->decode(
        latents, static_cast<int64_t>(speech_codes.size()));
    engine::debug::timing_log_scalar(
        "mira_tts.decoder_ms", engine::debug::elapsed_ms(decoder_start));
    return audio;
}

runtime::StreamingPolicy MiraTTSOfflineSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void MiraTTSOfflineSession::start_stream(
    const runtime::TaskRequest & request) {
    require_prepared("MiraTTS streaming");
    runtime::validate_spec_backed_request_options(
        request.options, *contract_, "MiraTTS");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("MiraTTS start_stream requires a streaming session");
    }
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MiraTTS streaming requires non-empty text input");
    }
    reset();
    const int64_t chunk_size = engine::text::parse_text_chunk_size_override(
        request.options).value_or(160);
    const auto chunk_mode = engine::text::parse_text_chunk_mode_override(
        request.options).value_or(engine::text::TextChunkMode::Default);
    streaming_text_chunks_ = engine::text::split_text_chunks(
        request.text_input->text, chunk_size, chunk_mode);
    if (streaming_text_chunks_.empty()) {
        throw std::runtime_error("MiraTTS streaming text chunking produced no segments");
    }
    streaming_context_codes_ = context_codes(reference_audio(request));
    streaming_generation_ = generation_options(request);
    streaming_started_ = true;
}

std::optional<runtime::StreamEvent>
MiraTTSOfflineSession::next_stream_event() {
    if (!streaming_started_ || !streaming_generation_.has_value()) {
        throw std::runtime_error("MiraTTS streaming has not been started");
    }
    if (streaming_chunk_index_ >= streaming_text_chunks_.size()) {
        return std::nullopt;
    }
    const size_t index = streaming_chunk_index_++;
    auto options = *streaming_generation_;
    options.seed += index;
    auto audio = synthesize_text(
        streaming_text_chunks_[index], streaming_context_codes_, options);
    streaming_audio_chunks_.push_back(audio);
    runtime::StreamEvent event;
    event.audio_output = std::move(audio);
    if (stream_sink_) {
        stream_sink_(event);
    }
    return event;
}

void MiraTTSOfflineSession::set_stream_event_sink(
    runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}

runtime::TaskResult MiraTTSOfflineSession::finish_stream() {
    if (!streaming_started_) {
        throw std::runtime_error("MiraTTS streaming has not been started");
    }
    runtime::TaskResult result;
    runtime::AudioBuffer merged;
    for (const auto & chunk : streaming_audio_chunks_) {
        runtime::append_audio_buffer(merged, chunk);
    }
    if (merged.sample_rate == 0) {
        throw std::runtime_error("MiraTTS streaming produced no audio chunks");
    }
    result.audio_output = std::move(merged);
    reset();
    return result;
}

void MiraTTSOfflineSession::reset() {
    streaming_context_codes_.clear();
    streaming_text_chunks_.clear();
    streaming_audio_chunks_.clear();
    streaming_generation_.reset();
    streaming_chunk_index_ = 0;
    streaming_started_ = false;
}

runtime::StreamEvent MiraTTSOfflineSession::process_audio_chunk(
    const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("MiraTTS streaming does not consume audio chunks");
}

runtime::TaskResult MiraTTSOfflineSession::finalize() {
    return finish_stream();
}

std::shared_ptr<runtime::IVoiceModelLoader> make_mira_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<MiraTTSAssets> config;
    config.family = kFamily;
    config.aliases = {"mira", "MiraTTS"};
    config.load_assets = load_mira_tts_assets;
    config.create_session = create_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::mira_tts
