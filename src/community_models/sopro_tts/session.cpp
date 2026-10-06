#include "engine/community_models/sopro_tts/session.h"

#include "engine/community_models/sopro_tts/acoustic.h"
#include "engine/community_models/sopro_tts/reference.h"
#include "engine/community_models/sopro_tts/semantic_encoder.h"
#include "engine/community_models/sopro_tts/semantic_lm.h"
#include "engine/community_models/sopro_tts/speaker_encoder.h"
#include "engine/community_models/sopro_tts/streaming.h"
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
#include <cstring>
#include <map>
#include <memory>
#include <optional>
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

// One FNV-1a step.
uint64_t mix(uint64_t hash, uint64_t value) {
    return (hash ^ value) * 1099511628211ull;
}

uint64_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// The first (last) `count` tokens, or all of them when there are fewer.
std::vector<int32_t> head_tokens(const std::vector<int32_t> & tokens, int64_t count) {
    const auto n = std::clamp<int64_t>(count, 0, static_cast<int64_t>(tokens.size()));
    return std::vector<int32_t>(tokens.begin(), tokens.begin() + static_cast<ptrdiff_t>(n));
}

std::vector<int32_t> tail_tokens(const std::vector<int32_t> & tokens, int64_t count) {
    const auto n = std::clamp<int64_t>(count, 0, static_cast<int64_t>(tokens.size()));
    return std::vector<int32_t>(tokens.end() - static_cast<ptrdiff_t>(n), tokens.end());
}

size_t voice_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots =
        runtime::parse_i64_option(options.options, {"sopro_tts.voice_cache_slots"}).value_or(0);
    if (slots < 0) {
        throw std::runtime_error("sopro_tts.voice_cache_slots must be non-negative");
    }
    return static_cast<size_t>(slots);
}

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
      contract_(require_contract(std::move(contract))),
      voice_cache_(voice_cache_slots(options)) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Sopro");
    if (const auto value = runtime::find_option(options.options, {"sopro_tts.language", "language"})) {
        default_language_ = *value;
    }
    const auto matmul_storage = runtime::parse_tensor_storage_option(
        options.options,
        "sopro_tts.matmul_weight_type",
        "matmul_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto conv_storage = runtime::parse_tensor_storage_option(
        options.options,
        "sopro_tts.conv_weight_type",
        "conv_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16});

    core::ExecutionContext & execution = execution_context();
    tokenizer_ = std::make_unique<SoproTextTokenizer>(assets_->tokenizer_path);
    speaker_encoder_ = std::make_unique<SoproSpeakerEncoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    semantic_encoder_ = std::make_unique<SoproSemanticEncoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    vocoder_ = std::make_unique<SoproVocosRuntime>(
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
    // SoproTTS.prepare_reference: with the voice cache on, a voice handed to
    // prepare is prepared here, and for a streaming session its prompt solved,
    // so a caller can warm a voice without generating anything.
    const auto & voice = request.voice;
    if (voice_cache_.capacity() > 0 && voice.has_value() && voice->speaker.has_value() &&
        voice->speaker->audio.has_value() && !voice->speaker->audio->samples.empty()) {
        runtime::TaskRequest options_only;
        options_only.options = request.options;
        const auto options = parse_options(options_only);
        auto prepared = prepare_voice(
            to_mono_24k(*voice->speaker->audio, static_cast<int>(assets_->config.sample_rate)), options.ref_seconds);
        if (task_.mode == runtime::RunMode::Streaming) {
            prompt_state(*prepared, options.steps);
        }
    }
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

    // The language option, else the transcript's (the server's top-level
    // "language"), else the session default.
    if (const auto value = runtime::find_option(request.options, {"language"})) {
        out.language = *value;
    } else if (request.text_input.has_value() && !request.text_input->language.empty()) {
        out.language = request.text_input->language;
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

// sopro Reference: the prepared reference plus the streaming prompt states
// solved from it, one per solver step count. Its own random draws (the room
// tone, the prompt noise) are seeded from the clip rather than the request, so
// a voice served from the cache yields exactly what a fresh one would.
struct SoproVoice : SoproReference {
    SoproVoice(SoproReference reference, uint64_t seed_in)
        : SoproReference(std::move(reference)), seed(seed_in) {}

    uint64_t seed = 0;
    std::map<int64_t, SoproPromptState> prompt_states;
};

// The per-run state that every text segment of one synthesis shares. Offline
// drains it in a loop; streaming keeps it alive between next_stream_event
// calls.
struct SoproSynthesisState {
    SoproRequestOptions options;
    std::shared_ptr<SoproVoice> voice;
    SoproSemanticLMOptions lm_options;
    std::vector<int32_t> style_tokens;
    std::vector<int32_t> carry;
    std::vector<std::string> segments;
    std::mt19937_64 rng;
    size_t index = 0;  // next segment to synthesize
    int sample_rate = 0;
    float gain = 0.0F;

    // Streaming (SoproTTS.stream): the voice's solved prompt, the segment being
    // streamed, and its lead-in gate.
    const SoproPromptState * prompt = nullptr;
    std::unique_ptr<SoproStreamSession> stream;
    bool first = true;
    bool last = true;
    bool gating = true;
    std::vector<float> pending;
    int64_t pending_offset = 0;
    int64_t emitted_samples = 0;
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
    state->voice = prepare_voice(to_mono_24k(*reference, state->sample_rate), state->options.ref_seconds);
    const SoproReference & voice = *state->voice;

    // _steps(): one semantic token per token_samples output samples.
    const int64_t token_samples = config.semantic_encoder.token_samples_24k;
    const auto steps_for = [&](float seconds) {
        return std::max<int64_t>(
            1,
            static_cast<int64_t>(std::ceil(
                static_cast<double>(seconds) * static_cast<double>(state->sample_rate) /
                static_cast<double>(token_samples))));
    };

    state->style_tokens = head_tokens(voice.semantic_tokens, config.generation.style_tokens);
    // SoproTTS._semantic_stream: the first segment is prompted with the first
    // prompt_tokens reference tokens; every later segment carries the tail of
    // its predecessor (tail_tokens).
    state->carry = head_tokens(voice.semantic_tokens, config.generation.prompt_tokens);
    // SoproTTS.synthesize / stream: one fixed gain from the prompt level, the
    // same for every segment and both modes.
    state->gain = audio_ops::output_gain(voice.level_db);

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

std::shared_ptr<SoproVoice> SoproTTSSession::prepare_voice(const std::vector<float> & audio24, float ref_seconds) {
    uint64_t hash = 1469598103934665603ull;  // FNV-1a offset basis
    for (const float sample : audio24) {
        hash = mix(hash, float_bits(sample));
    }
    hash = mix(hash, float_bits(ref_seconds));
    const VoiceKey key{static_cast<uint64_t>(audio24.size()), hash};
    if (const auto * cached = voice_cache_.find(key)) {
        engine::debug::trace_log_scalar("sopro_tts.voice_cache.hit", int64_t{1});
        return *cached;
    }
    const auto start = std::chrono::steady_clock::now();
    std::mt19937_64 rng(hash);
    auto voice = std::make_shared<SoproVoice>(reference_builder_->build(audio24, ref_seconds, rng), hash);
    engine::debug::timing_log_scalar(
        "sopro_tts.reference.prepare_ms", engine::debug::elapsed_ms(start, std::chrono::steady_clock::now()));
    if (voice->semantic_tokens.empty() || voice->mel_frames <= 0) {
        throw std::runtime_error("Sopro reference audio produced no semantic tokens");
    }
    voice_cache_.put(key, voice);  // a no-op without slots
    engine::debug::trace_log_scalar("sopro_tts.voice_cache.hit", int64_t{0});
    return voice;
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
    state.carry = tail_tokens(tokens, config.generation.prompt_tokens);

    const SoproReference & voice = *state.voice;
    SoproAcousticRequest acoustic;
    acoustic.semantic_tokens = voice.semantic_tokens;
    acoustic.semantic_tokens.insert(acoustic.semantic_tokens.end(), tokens.begin(), tokens.end());
    acoustic.cond_vec = voice.cond_vec;
    acoustic.prompt_mel = voice.mel;
    acoustic.prompt_frames = voice.mel_frames;
    acoustic.total_frames = voice.mel_frames + static_cast<int64_t>(tokens.size()) * hop_ratio;
    acoustic.steps = state.options.steps;
    acoustic.seed = state.rng();
    const auto acoustic_start = std::chrono::steady_clock::now();
    const auto mel = acoustic_->solve(acoustic);
    engine::debug::timing_log_scalar(
        "sopro_tts.acoustic.solve_ms",
        engine::debug::elapsed_ms(acoustic_start, std::chrono::steady_clock::now()));

    // Denormalise and hand the vocoder a short prompt run-up so its
    // convolution state matches the reference, then drop that run-up.
    const int64_t context = std::min(kDecodeContextFrames, voice.mel_frames);
    const int64_t begin = voice.mel_frames - context;
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

    // SoproTTS.synthesize: apply the output gain, trim each segment's lead and
    // trail, then cross-fade the joins.
    std::vector<std::vector<float>> trimmed;
    trimmed.reserve(parts.size());
    for (size_t index = 0; index < parts.size(); ++index) {
        auto part = parts[index];
        for (auto & value : part) {
            value *= state->gain;
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
    // SoproTTS.stream: audio leaves chunk by chunk as the semantic LM runs,
    // each chunk once its frames can no longer change.
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
    stream_state_ = begin_synthesis(request);
    if (stream_state_->segments.empty()) {
        throw std::runtime_error("Sopro streaming text chunking produced no segments");
    }
    // Every segment streams from a copy of the voice's prompt state.
    stream_state_->prompt = &prompt_state(*stream_state_->voice, stream_state_->options.steps);
}

const SoproPromptState & SoproTTSSession::prompt_state(SoproVoice & voice, int64_t steps) {
    const auto & config = assets_->config;
    const int64_t chunk_frames = config.generation.stream_chunk_frames;
    if (chunk_frames < 64 || chunk_frames % config.hop_ratio() != 0) {
        throw std::runtime_error("Sopro stream_chunk_frames must be a multiple of the hop ratio, at least 64");
    }
    auto prompt = voice.prompt_states.find(steps);
    if (prompt == voice.prompt_states.end()) {
        const auto prompt_start = std::chrono::steady_clock::now();
        std::mt19937_64 rng(mix(voice.seed, static_cast<uint64_t>(steps)));
        prompt = voice.prompt_states.emplace(steps, build_prompt_state(
            *acoustic_, voice, steps, chunk_frames, config.hop_ratio(),
            config.model.acoustic_pre_lookahead_frames, rng)).first;
        engine::debug::timing_log_scalar(
            "sopro_tts.streaming.prompt_state_ms",
            engine::debug::elapsed_ms(prompt_start, std::chrono::steady_clock::now()));
    }
    return prompt->second;
}

void SoproTTSSession::begin_stream_segment(SoproSynthesisState & state) {
    const auto & config = assets_->config;
    const size_t index = state.index++;
    state.first = index == 0;
    state.last = index + 1 == state.segments.size();
    semantic_lm_->begin(
        tokenizer_->encode(state.segments[index], state.options.language),
        state.style_tokens, state.carry, state.lm_options);
    state.stream = std::make_unique<SoproStreamSession>(
        *acoustic_, *vocoder_, *state.voice, *state.prompt,
        config.model.acoustic_mel_mean, config.model.acoustic_mel_std,
        state.options.steps, config.generation.stream_chunk_frames, config.hop_ratio(),
        config.model.acoustic_pre_lookahead_frames,
        std::min(kDecodeContextFrames, state.voice->mel_frames),
        state.rng);
    state.gating = true;
    state.pending.clear();
    state.pending_offset = 0;
    state.emitted_samples = 0;
}

// _stream_segment's gate: hold the opening audio until speech starts, then cut
// the lead-in the same way trim_lead does, backing off to an earlier energy
// onset when one sits right before the detected speech.
std::optional<std::vector<float>> SoproTTSSession::gate_stream_audio(
    SoproSynthesisState & state, std::vector<float> audio) const {
    if (!state.gating) {
        return audio;
    }
    const int sample_rate = state.sample_rate;
    const auto rate = static_cast<float>(sample_rate);
    const float lead = state.first ? audio_ops::kLeadInSeconds : audio_ops::kSegmentLeadSeconds;
    const float skip = state.first ? 0.0F : audio_ops::kSegmentSkipSeconds;
    state.pending.insert(state.pending.end(), audio.begin(), audio.end());
    const auto onset = audio_ops::speech_onset(state.pending, sample_rate);
    if (!onset.has_value()) {
        const auto keep = static_cast<int64_t>(std::max(lead, audio_ops::kGateHoldSeconds) * rate);
        const auto held = static_cast<int64_t>(state.pending.size());
        if (held > keep) {
            state.pending_offset += held - keep;
            state.pending.erase(state.pending.begin(), state.pending.begin() + static_cast<ptrdiff_t>(held - keep));
        }
        return std::nullopt;
    }
    int64_t cut = audio_ops::lead_cut(state.pending_offset + *onset, sample_rate, lead, skip);
    const auto guard = audio_ops::energy_onset(state.pending, sample_rate);
    if (guard.has_value() && *guard < *onset) {
        const bool near = *onset - *guard <= static_cast<int64_t>(0.05F * rate);
        if (near || audio_ops::energy_fraction(state.pending, sample_rate, *guard, *onset) >= 0.35F) {
            cut = std::min(cut, std::max(
                static_cast<int64_t>(skip * rate),
                state.pending_offset + *guard - static_cast<int64_t>(0.05F * rate)));
        }
    }
    const auto from = std::min<int64_t>(
        std::max<int64_t>(0, cut - state.pending_offset), static_cast<int64_t>(state.pending.size()));
    std::vector<float> out(state.pending.begin() + static_cast<ptrdiff_t>(from), state.pending.end());
    state.gating = false;
    state.pending.clear();
    audio_ops::fade_edges(out, sample_rate, !state.first, false);
    return out;
}

std::vector<float> SoproTTSSession::finish_stream_segment(SoproSynthesisState & state) {
    const auto & config = assets_->config;
    auto & session = *state.stream;
    const auto token_count = static_cast<int64_t>(session.tokens().size());
    std::vector<float> out;
    if (token_count > 0) {
        auto tail = session.finish();
        const int64_t target = std::max<int64_t>(
            0, token_count * config.semantic_encoder.token_samples_24k - state.emitted_samples);
        tail.resize(static_cast<size_t>(std::min<int64_t>(target, static_cast<int64_t>(tail.size()))));
        if (!tail.empty()) {
            for (auto & value : tail) {
                value *= state.gain;
            }
            auto gated = gate_stream_audio(state, std::move(tail));
            if (gated.has_value() && !gated->empty()) {
                out = audio_ops::trim_trail(*gated, state.sample_rate);
                audio_ops::fade_edges(
                    out, state.sample_rate, false, true,
                    state.last ? audio_ops::kFinalFadeSeconds : audio_ops::kJoinFadeSeconds);
                audio_ops::soft_limit(out);
            }
        } else if (state.gating && !state.pending.empty()) {
            out = state.pending;
            audio_ops::fade_edges(out, state.sample_rate, false, state.last, audio_ops::kFinalFadeSeconds);
            audio_ops::soft_limit(out);
        }
        state.carry = tail_tokens(session.tokens(), config.generation.prompt_tokens);
    }
    state.stream.reset();
    return out;
}

std::optional<runtime::StreamEvent> SoproTTSSession::next_stream_event() {
    if (stream_state_ == nullptr) {
        throw std::runtime_error("Sopro streaming has not been started");
    }
    SoproSynthesisState & state = *stream_state_;
    const auto & config = assets_->config;
    // SoproTTS.stream feeds the session one chunk's worth of tokens at a time.
    const int64_t tokens_per_push = config.generation.stream_chunk_frames / config.hop_ratio();
    const auto event_start = std::chrono::steady_clock::now();
    std::vector<float> out;
    while (out.empty()) {
        if (state.stream == nullptr) {
            if (state.index >= state.segments.size()) {
                return std::nullopt;
            }
            begin_stream_segment(state);
        }
        const auto tokens = semantic_lm_->next(tokens_per_push, state.rng);
        if (tokens.empty()) {
            out = finish_stream_segment(state);
            continue;
        }
        auto audio = state.stream->push(tokens);
        if (audio.empty()) {
            continue;
        }
        state.emitted_samples += static_cast<int64_t>(audio.size());
        for (auto & value : audio) {
            value *= state.gain;
        }
        auto gated = gate_stream_audio(state, std::move(audio));
        if (gated.has_value() && !gated->empty()) {
            out = std::move(*gated);
            audio_ops::soft_limit(out);
        }
    }

    engine::debug::timing_log_scalar(
        "sopro_tts.streaming.event_ms",
        engine::debug::elapsed_ms(event_start, std::chrono::steady_clock::now()));
    runtime::AudioBuffer audio;
    audio.sample_rate = state.sample_rate;
    audio.channels = 1;
    audio.samples = std::move(out);
    const size_t chunk_index = stream_chunks_.size();
    stream_chunks_.push_back(audio);
    runtime::StreamEvent event;
    event.named_audio_outputs.push_back({"chunk_" + std::to_string(chunk_index), std::move(audio), {}});
    return event;
}

void SoproTTSSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    // Every driver of a PullEvents session (app/streaming/streaming.cpp, and the
    // server through it) forwards whatever next_stream_event returns to its own
    // sink, so pushing here as well would deliver each chunk twice.
    (void) sink;
}

runtime::TaskResult SoproTTSSession::finish_stream() {
    if (stream_state_ == nullptr) {
        throw std::runtime_error("Sopro streaming has not been started");
    }
    // Each event is already levelled, gated and faded, so the utterance is the
    // concatenation of what the consumer has already heard.
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
