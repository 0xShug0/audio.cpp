#include "engine/community_models/sopro_tts/session.h"

#include "engine/community_models/sopro_tts/acoustic.h"
#include "engine/community_models/sopro_tts/reference.h"
#include "engine/community_models/sopro_tts/semantic_encoder.h"
#include "engine/community_models/sopro_tts/semantic_lm.h"
#include "engine/community_models/sopro_tts/speaker_encoder.h"
#include "engine/community_models/sopro_tts/text_tokenizer.h"
#include "engine/community_models/sopro_tts/vocoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {
namespace {

constexpr const char * kFamily = "sopro_tts";
constexpr size_t kWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kGraphArenaBytes = 1024ull * 1024ull * 1024ull;
// SoproTTS.DECODE_CONTEXT_FRAMES: mel frames of prompt fed to the vocoder so
// its convolutions start warm, then dropped from the output.
constexpr int64_t kDecodeContextFrames = 32;

std::shared_ptr<const SoproTTSAssets> require_assets(std::shared_ptr<const SoproTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Sopro session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Sopro session requires a model contract");
    }
    return contract;
}

const runtime::AudioBuffer * reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    return request.audio_input.has_value() ? &*request.audio_input : nullptr;
}

std::vector<float> to_mono_24k(const runtime::AudioBuffer & audio, int target_rate) {
    if (audio.samples.empty()) {
        throw std::runtime_error("Sopro reference audio is empty");
    }
    const int channels = std::max(1, audio.channels);
    std::vector<float> mono = channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, channels);
    if (audio.sample_rate > 0 && audio.sample_rate != target_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, target_rate);
    }
    // sopro.audio.to_mono_resampled clamps before anything else touches it.
    for (auto & value : mono) {
        value = std::min(1.0F, std::max(-1.0F, value));
    }
    return mono;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const SoproTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<SoproTTSSession>(task, options, std::move(assets), std::move(contract));
}

}  // namespace

SoproTTSSession::SoproTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SoproTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Sopro");
    if (const auto value = runtime::find_option(options.options, {"language"})) {
        default_language_ = *value;
    }
    const auto matmul_storage = runtime::parse_tensor_storage_option(
        options.options,
        "matmul_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto conv_storage = runtime::parse_tensor_storage_option(
        options.options,
        "conv_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16});

    core::ExecutionContext & execution = execution_context();
    tokenizer_ = std::make_unique<SoproTextTokenizer>(
        assets_->tokenizer_path, assets_->config.model.max_text_len);
    speaker_encoder_ = std::make_unique<SoproSpeakerEncoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    semantic_encoder_ = std::make_unique<SoproSemanticEncoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    vocoder_ = std::make_unique<SoproVocoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    semantic_lm_ = std::make_unique<SoproSemanticLMRuntime>(
        *assets_, execution, kGraphArenaBytes, kGraphArenaBytes, kWeightContextBytes, matmul_storage);
    acoustic_ = std::make_unique<SoproAcousticRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    reference_builder_ = std::make_unique<SoproReferenceBuilder>(
        *assets_, *speaker_encoder_, *semantic_encoder_, *vocoder_);
}

SoproTTSSession::~SoproTTSSession() = default;

std::string SoproTTSSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind SoproTTSSession::task_kind() const {
    return runtime::VoiceTaskKind::Tts;
}

runtime::RunMode SoproTTSSession::run_mode() const {
    return task_.mode;
}

void SoproTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Sopro");
    mark_prepared();
}

SoproRequestOptions SoproTTSSession::parse_options(const runtime::TaskRequest & request) const {
    const auto & defaults = assets_->config.generation;
    SoproRequestOptions out;
    out.language = default_language_;
    out.temperature = defaults.temperature;
    out.top_p = defaults.top_p;
    out.top_k = defaults.top_k;
    out.steps = defaults.steps;
    out.max_seconds = defaults.max_seconds;
    out.min_seconds = defaults.min_seconds;
    out.max_segment_chars = defaults.max_segment_chars;
    out.ref_seconds = defaults.ref_seconds;

    if (const auto value = runtime::find_option(request.options, {"language"})) {
        out.language = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"num_inference_steps"})) {
        out.steps = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"max_seconds"})) {
        out.max_seconds = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"min_seconds"})) {
        out.min_seconds = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"text_chunk_size"})) {
        out.max_segment_chars = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"ref_seconds"})) {
        out.ref_seconds = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
        out.has_seed = true;
    }
    if (!out.has_seed) {
        out.seed = runtime::random_u64_seed();
    }
    if (out.steps < 1) {
        throw std::runtime_error("Sopro num_inference_steps must be positive");
    }
    if (out.max_segment_chars < 1) {
        throw std::runtime_error("Sopro text_chunk_size must be positive");
    }
    if (out.max_seconds <= 0.0F) {
        throw std::runtime_error("Sopro max_seconds must be positive");
    }
    if (out.ref_seconds <= 0.0F) {
        throw std::runtime_error("Sopro ref_seconds must be positive");
    }
    if (out.min_seconds < 0.0F) {
        throw std::runtime_error("Sopro min_seconds must not be negative");
    }
    // A min above max leaves min_steps > max_steps, which never lets the LM
    // emit EOS: every segment would run the full budget and be cut mid-word.
    if (out.min_seconds > out.max_seconds) {
        throw std::runtime_error("Sopro min_seconds must not exceed max_seconds");
    }
    // language_tag() rejects anything outside the four supported languages, so
    // fail before any weights are touched.
    (void) language_tag(out.language);
    return out;
}

// The per-run state that every text segment of one synthesis shares. Offline
// drains it in a loop; streaming keeps it alive between next_stream_event
// calls, so both paths draw from the seeded RNG in the same order and a given
// seed produces the same audio either way.
struct SoproSynthesisState {
    SoproRequestOptions options;
    SoproReference voice;
    SoproSemanticLMOptions lm_options;
    std::vector<int32_t> style_tokens;
    std::vector<int32_t> carry;
    std::vector<std::string> segments;
    std::mt19937_64 rng;
    size_t index = 0;    // next segment to synthesize
    size_t emitted = 0;  // segments that produced audio so far
    int sample_rate = 0;
    float gain = 0.0F;
    bool gain_ready = false;
};

std::unique_ptr<SoproSynthesisState> SoproTTSSession::begin_synthesis(
    const runtime::TaskRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Sopro");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Sopro requires non-empty text input");
    }
    const runtime::AudioBuffer * reference = reference_audio(request);
    if (reference == nullptr || reference->samples.empty()) {
        throw std::runtime_error(
            "Sopro requires reference voice audio (voice preset or voice_ref) for zero-shot cloning");
    }

    auto state = std::make_unique<SoproSynthesisState>();
    state->options = parse_options(request);
    const auto & config = assets_->config;
    state->sample_rate = static_cast<int>(config.sample_rate);
    state->rng.seed(state->options.seed);

    const auto reference_audio24 = to_mono_24k(*reference, state->sample_rate);
    const auto reference_start = std::chrono::steady_clock::now();
    state->voice = reference_builder_->build(
        reference_audio24, state->options.ref_seconds, state->rng);
    engine::debug::timing_log_scalar(
        "sopro_tts.reference.prepare_ms",
        engine::debug::elapsed_ms(reference_start, std::chrono::steady_clock::now()));
    if (state->voice.semantic_tokens.empty() || state->voice.mel_frames <= 0) {
        throw std::runtime_error("Sopro reference audio produced no semantic tokens");
    }

    // _steps(): one semantic token per token_samples output samples.
    const int64_t token_samples = config.semantic_encoder.token_samples_24k;
    const auto steps_for = [&](float seconds) {
        return std::max<int64_t>(
            1,
            static_cast<int64_t>(std::ceil(
                static_cast<double>(seconds) * static_cast<double>(state->sample_rate) /
                static_cast<double>(token_samples))));
    };

    const auto style_count = std::max<int64_t>(
        0,
        std::min<int64_t>(
            config.generation.style_tokens,
            static_cast<int64_t>(state->voice.semantic_tokens.size())));
    state->style_tokens.assign(
        state->voice.semantic_tokens.begin(),
        state->voice.semantic_tokens.begin() + static_cast<ptrdiff_t>(style_count));
    if (config.generation.prompt_tokens > 0) {
        const auto count = std::min<int64_t>(
            config.generation.prompt_tokens,
            static_cast<int64_t>(state->voice.semantic_tokens.size()));
        // Continue from the *end* of the reference. synthesize_segment places
        // this segment's tokens after the whole reference, and every later
        // segment carries the tail of its predecessor, so anchoring the first
        // one at the head would continue from the wrong point in the clip.
        state->carry.assign(
            state->voice.semantic_tokens.end() - static_cast<ptrdiff_t>(count),
            state->voice.semantic_tokens.end());
    }

    state->lm_options.max_steps = steps_for(state->options.max_seconds);
    state->lm_options.min_steps = steps_for(state->options.min_seconds);
    state->lm_options.temperature = state->options.temperature;
    state->lm_options.top_p = state->options.top_p;
    state->lm_options.top_k = state->options.top_k;

    state->segments = split_text(request.text_input->text, state->options.max_segment_chars);
    engine::debug::trace_log_scalar(
        "sopro_tts.text.segments", static_cast<int64_t>(state->segments.size()));
    return state;
}

std::vector<float> SoproTTSSession::synthesize_segment(SoproSynthesisState & state) {
    if (state.index >= state.segments.size()) {
        return {};
    }
    const std::string & segment = state.segments[state.index++];
    const auto & config = assets_->config;
    const int64_t token_samples = config.semantic_encoder.token_samples_24k;
    const int64_t hop_ratio = config.hop_ratio();
    const int64_t n_mels = config.model.acoustic_mel_n_mels;
    const int64_t vocoder_hop = vocoder_->hop_length();
    const auto prompt_budget = config.generation.prompt_tokens;

    const auto text_ids = tokenizer_->encode(segment, state.options.language);
    const auto lm_start = std::chrono::steady_clock::now();
    const auto tokens = semantic_lm_->generate(
        text_ids, state.style_tokens, state.carry, state.lm_options, state.rng);
    engine::debug::timing_log_scalar(
        "sopro_tts.semantic_lm.generate_ms",
        engine::debug::elapsed_ms(lm_start, std::chrono::steady_clock::now()));
    engine::debug::trace_log_scalar(
        "sopro_tts.semantic_lm.tokens", static_cast<int64_t>(tokens.size()));
    if (tokens.empty()) {
        return {};
    }
    if (prompt_budget > 0) {
        const auto count = std::min<int64_t>(prompt_budget, static_cast<int64_t>(tokens.size()));
        state.carry.assign(tokens.end() - static_cast<ptrdiff_t>(count), tokens.end());
    }

    SoproAcousticRequest acoustic;
    acoustic.semantic_tokens = state.voice.semantic_tokens;
    acoustic.semantic_tokens.insert(acoustic.semantic_tokens.end(), tokens.begin(), tokens.end());
    acoustic.cond_vec = state.voice.cond_vec;
    acoustic.prompt_mel = state.voice.mel;
    acoustic.prompt_frames = state.voice.mel_frames;
    acoustic.total_frames =
        state.voice.mel_frames + static_cast<int64_t>(tokens.size()) * hop_ratio;
    acoustic.steps = state.options.steps;
    acoustic.seed = state.rng();
    const auto acoustic_start = std::chrono::steady_clock::now();
    const auto mel = acoustic_->solve(acoustic);
    engine::debug::timing_log_scalar(
        "sopro_tts.acoustic.solve_ms",
        engine::debug::elapsed_ms(acoustic_start, std::chrono::steady_clock::now()));

    // Denormalise and hand the vocoder a short prompt run-up so its
    // convolution state matches the reference, then drop that run-up.
    const int64_t context = std::min(kDecodeContextFrames, state.voice.mel_frames);
    const int64_t begin = state.voice.mel_frames - context;
    const int64_t decode_frames = acoustic.total_frames - begin;
    std::vector<float> decode_mel(static_cast<size_t>(n_mels * decode_frames), 0.0F);
    for (int64_t c = 0; c < n_mels; ++c) {
        const float mean = config.model.acoustic_mel_mean[static_cast<size_t>(c)];
        const float scale = config.model.acoustic_mel_std[static_cast<size_t>(c)];
        const float * source = mel.data() + static_cast<size_t>(c * acoustic.total_frames + begin);
        float * target = decode_mel.data() + static_cast<size_t>(c * decode_frames);
        for (int64_t t = 0; t < decode_frames; ++t) {
            target[t] = source[t] * scale + mean;
        }
    }
    auto wav = vocoder_->decode(decode_mel, decode_frames);
    const int64_t skip = context * vocoder_hop;
    const int64_t target_length = static_cast<int64_t>(tokens.size()) * token_samples;
    if (static_cast<int64_t>(wav.size()) <= skip) {
        return {};
    }
    const int64_t end = std::min<int64_t>(static_cast<int64_t>(wav.size()), skip + target_length);
    return std::vector<float>(
        wav.begin() + static_cast<ptrdiff_t>(skip), wav.begin() + static_cast<ptrdiff_t>(end));
}

runtime::TaskResult SoproTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Sopro run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Sopro run requires an offline session");
    }
    auto state = begin_synthesis(request);

    std::vector<std::vector<float>> parts;
    while (state->index < state->segments.size()) {
        auto part = synthesize_segment(*state);
        if (!part.empty()) {
            parts.push_back(std::move(part));
        }
    }

    const int sample_rate = state->sample_rate;
    runtime::TaskResult result;
    runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    if (parts.empty()) {
        audio.samples.assign(
            static_cast<size_t>(assets_->config.semantic_encoder.token_samples_24k), 0.0F);
        result.audio_output = std::move(audio);
        return result;
    }

    // Level-match once over the whole utterance, then trim and cross-fade the
    // segment joins (SoproTTS.synthesize).
    std::vector<float> concatenated;
    for (const auto & part : parts) {
        concatenated.insert(concatenated.end(), part.begin(), part.end());
    }
    const float gain = audio_ops::match_gain(
        concatenated, sample_rate, audio_ops::kOutputLevelDb, state->voice.level_db);
    std::vector<std::vector<float>> trimmed;
    trimmed.reserve(parts.size());
    for (size_t index = 0; index < parts.size(); ++index) {
        auto part = parts[index];
        for (auto & value : part) {
            value *= gain;
        }
        part = index == 0
            ? audio_ops::trim_lead(part, sample_rate)
            : audio_ops::trim_lead(
                  part, sample_rate, audio_ops::kSegmentLeadSeconds, audio_ops::kSegmentSkipSeconds);
        trimmed.push_back(audio_ops::trim_trail(part, sample_rate));
    }
    auto out = audio_ops::join_segments(std::move(trimmed), sample_rate);
    audio_ops::soft_limit(out);
    audio_ops::fade_edges(out, sample_rate, false, true, audio_ops::kFinalFadeSeconds);
    audio.samples = std::move(out);
    result.audio_output = std::move(audio);
    return result;
}

// --------------------------------------------------------------------------- //
// Streaming interface
// --------------------------------------------------------------------------- //
runtime::StreamingPolicy SoproTTSSession::streaming_policy() const {
    // The acoustic head and the vocoder both look at a whole span at once, and
    // this checkpoint ships no causal vocoder, so one text segment is the
    // smallest unit that can leave without boundary artefacts. text_chunk_size
    // is what trades first-audio latency against segment length.
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void SoproTTSSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Sopro start_stream");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Sopro start_stream requires a streaming session");
    }
    reset();
    // The reference voice is encoded once here rather than per event, so every
    // event after the first costs only its own LM, solver and vocoder passes.
    stream_state_ = begin_synthesis(request);
    if (stream_state_->segments.empty()) {
        throw std::runtime_error("Sopro streaming text chunking produced no segments");
    }
}

std::optional<runtime::StreamEvent> SoproTTSSession::next_stream_event() {
    if (stream_state_ == nullptr) {
        throw std::runtime_error("Sopro streaming has not been started");
    }
    SoproSynthesisState & state = *stream_state_;
    const auto event_start = std::chrono::steady_clock::now();
    std::vector<float> part;
    while (part.empty() && state.index < state.segments.size()) {
        part = synthesize_segment(state);
    }
    if (part.empty()) {
        return std::nullopt;
    }
    engine::debug::timing_log_scalar(
        "sopro_tts.streaming.event.synthesize_ms",
        engine::debug::elapsed_ms(event_start, std::chrono::steady_clock::now()));

    // Offline levels the whole utterance at once. A stream cannot see the
    // segments it has not generated yet, so the first one fixes the gain for
    // all of them; that keeps their relative loudness instead of pushing every
    // segment onto the target level on its own.
    const int sample_rate = state.sample_rate;
    if (!state.gain_ready) {
        state.gain = audio_ops::match_gain(
            part, sample_rate, audio_ops::kOutputLevelDb, state.voice.level_db);
        state.gain_ready = true;
        engine::debug::trace_log_scalar(
            "sopro_tts.streaming.gain", static_cast<double>(state.gain));
    }
    for (auto & value : part) {
        value *= state.gain;
    }
    part = state.emitted == 0
        ? audio_ops::trim_lead(part, sample_rate)
        : audio_ops::trim_lead(
              part, sample_rate, audio_ops::kSegmentLeadSeconds, audio_ops::kSegmentSkipSeconds);
    part = audio_ops::trim_trail(part, sample_rate);
    // Same order as the offline tail: join fade, limiter, then the final fade
    // on whichever segment turns out to be the last one.
    const bool has_more = state.index < state.segments.size();
    audio_ops::fade_edges(part, sample_rate, state.emitted > 0, has_more);
    audio_ops::soft_limit(part);
    if (!has_more) {
        audio_ops::fade_edges(part, sample_rate, false, true, audio_ops::kFinalFadeSeconds);
    }

    runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    audio.samples = std::move(part);
    const size_t chunk_index = state.emitted++;
    stream_chunks_.push_back(audio);

    runtime::StreamEvent event;
    event.named_audio_outputs.push_back({
        "segment_" + std::to_string(chunk_index),
        std::move(audio),
        {},
    });
    return event;
}

void SoproTTSSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    // Every driver of a PullEvents session (app/streaming/streaming.cpp, and the
    // server through it) forwards whatever next_stream_event returns to its own
    // sink, so pushing here as well would deliver each segment twice.
    (void) sink;
}

runtime::TaskResult SoproTTSSession::finish_stream() {
    if (stream_state_ == nullptr) {
        throw std::runtime_error("Sopro streaming has not been started");
    }
    // Each event is already levelled, trimmed and faded, so the utterance is a
    // plain concatenation of what the consumer has already heard.
    runtime::AudioBuffer merged;
    merged.sample_rate = stream_state_->sample_rate;
    merged.channels = 1;
    if (stream_chunks_.empty()) {
        merged.samples.assign(
            static_cast<size_t>(assets_->config.semantic_encoder.token_samples_24k), 0.0F);
    }
    for (const auto & chunk : stream_chunks_) {
        runtime::append_audio_buffer(merged, chunk);
    }
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    reset();
    return result;
}

void SoproTTSSession::reset() {
    stream_state_.reset();
    stream_chunks_.clear();
}

runtime::StreamEvent SoproTTSSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void) chunk;
    throw std::runtime_error("Sopro is a TTS model and does not accept streamed audio input");
}

runtime::TaskResult SoproTTSSession::finalize() {
    return runtime::TaskResult{};
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sopro_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<SoproTTSAssets> config;
    config.family = kFamily;
    // The upstream repo and the model card both call the family "sopro"; keep
    // the short spelling working as a --family hint.
    config.aliases = {"sopro", "sopro_v2", "sopro_v2_turbo"};
    config.load_assets = load_sopro_tts_assets;
    config.create_session = create_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::sopro_tts
