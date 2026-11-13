#include "engine/community_models/zonos2/session.h"

#include "engine/community_models/zonos2/codec_decoder.h"
#include "engine/community_models/zonos2/generator.h"
#include "engine/community_models/zonos2/npy.h"
#include "engine/community_models/zonos2/prompt.h"
#include "engine/community_models/zonos2/speaker_encoder.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::zonos2 {
namespace {

namespace json = engine::io::json;

constexpr int kSampleRate = 44100;
constexpr int kCodebooks = 9;
constexpr float kPi = 3.14159265358979323846f;
// Offline chunks decode in windows of this many frames plus the stream context on each side: the output equals one
// pass, and the decoder scratch stays bounded (about 1.8 MB per window frame on the CPU; one pass over 2000 frames
// would need about 3.6 GB).
constexpr int kOfflineDecodeFrames = 128;

// Parser exceptions are client errors.
template <typename Fn>
auto client(Fn && fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const runtime::InvalidRequestError &) {
        throw;
    } catch (const std::exception & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

using Options = std::unordered_map<std::string, std::string>;

bool bool_option(const Options & options, std::string_view key, bool fallback) {
    const auto value = runtime::find_option(options, {key});
    return value ? client([&] { return runtime::parse_bool_option(*value, key); }) : fallback;
}

float float_option(const Options & options, std::string_view key, float fallback, float min,
                   float max = std::numeric_limits<float>::infinity()) {
    const float value = client([&] { return runtime::parse_finite_float_option(options, {key}); }).value_or(fallback);
    if (value < min || value > max) {
        std::ostringstream bounds;
        bounds << "zonos2 " << key << " must be ";
        if (std::isinf(max)) bounds << ">= " << min;
        else bounds << "between " << min << " and " << max;
        throw runtime::InvalidRequestError(bounds.str());
    }
    return value;
}

int int_option(const Options & options, std::string_view key, int fallback, int min, int max = std::numeric_limits<int>::max()) {
    const int value = client([&] { return runtime::parse_int_option(options, {key}); }).value_or(fallback);
    if (value < min || value > max) {
        throw runtime::InvalidRequestError("zonos2 " + std::string(key) + " must be " +
                                           (max == std::numeric_limits<int>::max() ? ">= " + std::to_string(min)
                                               : "between " + std::to_string(min) + " and " + std::to_string(max)));
    }
    return value;
}

int64_t session_int(const Options & session, const char * key, int64_t fallback, int64_t min, int64_t max) {
    const auto value = runtime::parse_i64_option(session, {key}).value_or(fallback);
    if (value < min || value > max) {
        throw std::runtime_error(std::string(key) + " must be between " + std::to_string(min) + " and " + std::to_string(max));
    }
    return value;
}

bool session_bool(const Options & session, const char * key) {
    return runtime::parse_bool_option(runtime::find_option(session, {key}).value_or("false"), key);
}

uint64_t fnv1a(uint64_t hash, const void * data, size_t bytes) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// Mono reference audio at the speaker encoder's 24 kHz.
std::vector<float> reference_mono(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw runtime::InvalidRequestError("zonos2 reference audio is empty");
    }
    auto mono = audio.channels == 1 ? audio.samples : audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != kZonos2SpeakerSampleRate) {
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, kZonos2SpeakerSampleRate);
    }
    return mono;
}

// Cosine fade over the last n samples (zonos2-server apply_fade).
void apply_fade(std::vector<float> & pcm, size_t n) {
    n = std::min(n, pcm.size());
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        pcm[pcm.size() - n + i] *= 0.5f * (1.0f + std::cos(kPi * t));
    }
}

}  // namespace

Zonos2Session::Zonos2Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                             std::shared_ptr<const Zonos2Assets> assets,
                             std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
    if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("ZONOS2 supports TTS and voice cloning");
    }
    const auto & session = options.options;
    Zonos2BackboneOptions backbone;
    backbone.disk_experts = session_bool(session, "zonos2.disk_experts");
    backbone.pin_experts = session_bool(session, "zonos2.pin_experts");
    backbone.expert_cache_size = session_int(session, "zonos2.expert_cache_size", 8, 4, 16);
    backbone.disk_expert_read_threads = session_int(session, "zonos2.disk_expert_read_threads", 1, 1, 64);
    voices_.set_capacity(static_cast<size_t>(session_int(session, "zonos2.voice_cache_slots", 4, 0,
                                                         std::numeric_limits<int32_t>::max())));
    stream_chunk_frames_ = static_cast<int>(session_int(session, "zonos2.stream_chunk_frames", 40, 1, 256));
    stream_context_frames_ = static_cast<int>(session_int(session, "zonos2.stream_context_frames", 24, 16, 64));
    const auto encoder_backend = runtime::find_option(session, {"zonos2.encoder_backend"}).value_or("cpu");
    if (encoder_backend != "cpu" && encoder_backend != "same") {
        throw std::runtime_error("zonos2.encoder_backend must be cpu or same");
    }
    core::BackendConfig encoder_config = options.backend;
    if (encoder_backend == "cpu") {
        encoder_config = core::BackendConfig{};
        encoder_config.type = core::BackendType::Cpu;
        encoder_config.threads = options.backend.threads;
    }
    encoder_execution_ = std::make_unique<core::ExecutionContext>(encoder_config);
    backbone_ = std::make_unique<Zonos2Backbone>(assets_, execution_context(), backbone);
    decoder_ = std::make_unique<Zonos2CodecDecoder>(assets_->decoder, execution_context());
}

Zonos2Session::~Zonos2Session() = default;
std::string Zonos2Session::family() const { return "zonos2"; }
runtime::VoiceTaskKind Zonos2Session::task_kind() const { return task_.task; }
runtime::RunMode Zonos2Session::run_mode() const { return task_.mode; }

void Zonos2Session::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "ZONOS2");
    defaults_ = {};
    defaults_.text_input = request.text;
    defaults_.voice = request.voice;
    defaults_.options = request.options;
    mark_prepared();
}

runtime::TaskResult Zonos2Session::run(const runtime::TaskRequest & request) { return synthesize(request, false); }
runtime::TaskResult Zonos2Session::generate_stream(const runtime::TaskRequest & request) { return synthesize(request, true); }

std::vector<float> Zonos2Session::encode_voice(const runtime::AudioBuffer & audio) {
    const auto mono = reference_mono(audio);
    const uint64_t key = fnv1a(1469598103934665603ull, mono.data(), mono.size() * sizeof(float));
    if (const auto * cached = voices_.find(key)) return *cached;
    const auto started = std::chrono::steady_clock::now();
    if (!speaker_encoder_) {
        speaker_encoder_ = std::make_unique<Zonos2SpeakerEncoder>(assets_->speaker_encoder, *encoder_execution_);
    }
    auto embedding = speaker_encoder_->embed(mono);
    engine::debug::timing_log_scalar("zonos2.speaker_encode_ms",
                                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    voices_.put(key, embedding);
    return embedding;
}

runtime::TaskResult Zonos2Session::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("ZONOS2 generation");
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "ZONOS2");
    const auto started = std::chrono::steady_clock::now();
    auto effective = defaults_;
    if (request.text_input) effective.text_input = request.text_input;
    if (request.voice) effective.voice = request.voice;
    for (const auto & option : request.options) effective.options[option.first] = option.second;
    const auto & options = effective.options;
    const auto & hp = assets_->hparams;

    if (effective.voice && effective.voice->style) {
        throw runtime::InvalidRequestError("zonos2 has no style controls; use speaking_rate_bucket and quality_buckets");
    }
    Zonos2SamplingOptions sampling;
    sampling.temperature = float_option(options, "temperature", 1.15f, 0.0f);
    sampling.top_k = int_option(options, "top_k", 106, 1);
    sampling.top_p = float_option(options, "top_p", 0.0f, 0.0f, 1.0f);
    sampling.min_p = float_option(options, "min_p", 0.18f, 0.0f, 1.0f);
    sampling.repetition_penalty = float_option(options, "repetition_penalty", 1.2f, 1.0f);
    sampling.repetition_window = int_option(options, "repetition_window", 50, 0);
    sampling.repetition_codebooks = int_option(options, "repetition_codebooks", 8, -1, kCodebooks);
    sampling.seed = client([&] { return runtime::parse_u32_option(options, {"seed"}); }).value_or(runtime::random_u32_seed());
    const int max_tokens = int_option(options, "max_tokens", 2000, 1);
    const bool encode_only = bool_option(options, "encode_reference_only", false);
    const float fade_ms = float_option(options, "fade_out_ms", 0.0f, 0.0f, 2000.0f);
    const int chunk_size = int_option(options, "text_chunk_size", 300, 1);
    Zonos2PromptOptions prompt_options;
    prompt_options.clean_speaker_background = bool_option(options, "clean_speaker_background", false);
    prompt_options.accurate_mode = bool_option(options, "accurate_mode", true);
    if (runtime::find_option(options, {"speaking_rate_bucket"})) {
        prompt_options.speaking_rate_bucket = int_option(options, "speaking_rate_bucket", -1, 0, hp.speaking_rate_buckets - 1);
    }
    if (const auto quality = runtime::find_option(options, {"quality_buckets"})) {
        prompt_options.quality_buckets = parse_zonos2_quality_buckets(hp, *quality);
    }

    // Voice source: reference audio or an embedding file, at most one.
    const auto embedding_file = runtime::find_option(options, {"speaker_embedding_file"});
    const bool has_audio = effective.voice && effective.voice->speaker && effective.voice->speaker->audio;
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->cached_voice_id && !has_audio) {
        throw runtime::InvalidRequestError("zonos2 has no built-in voices; pass --voice-ref or speaker_embedding_file");
    }
    if (embedding_file && has_audio) {
        throw runtime::InvalidRequestError("zonos2: give one voice source: --voice-ref or speaker_embedding_file");
    }
    if (encode_only && !has_audio) throw runtime::InvalidRequestError("zonos2 encode_reference_only needs --voice-ref");
    if (task_.task == runtime::VoiceTaskKind::VoiceCloning && !embedding_file && !has_audio) {
        throw runtime::InvalidRequestError("zonos2 clone requires a voice: --voice-ref or speaker_embedding_file");
    }
    std::vector<float> speaker;
    std::optional<runtime::VoiceArtifact> voice_artifact;
    if (embedding_file) {
        speaker = read_speaker_embedding_npy(*embedding_file);
    } else if (has_audio) {
        speaker = encode_voice(*effective.voice->speaker->audio);
        voice_artifact = runtime::make_voice_artifact(
            runtime::ArtifactKind::SpeakerEmbedding, "zonos2_speaker_embedding", speaker_embedding_npy(speaker),
            {{"family", "zonos2"}, {"mime", "application/x-npy"}, {"extension", ".npy"}});
    }
    if (!speaker.empty() && speaker.size() != static_cast<size_t>(hp.spk_dim)) {
        throw runtime::InvalidRequestError("zonos2 speaker embedding must have " + std::to_string(hp.spk_dim) + " values");
    }
    if (encode_only) {
        runtime::TaskResult result;
        result.output_artifacts.push_back(std::move(*voice_artifact));
        return result;
    }
    prompt_options.speaker = !speaker.empty();

    if (!effective.text_input || effective.text_input->text.empty()) {
        throw runtime::InvalidRequestError("zonos2 requires input text");
    }
    const auto mode = engine::text::parse_text_chunk_mode_override(options).value_or(engine::text::TextChunkMode::Default);
    const auto chunks = runtime::chunk_text_request(effective, chunk_size, mode);
    size_t last_chunk = chunks.size();
    for (size_t index = 0; index < chunks.size(); ++index) {
        if (!chunks[index].text_input->text.empty()) last_chunk = index;
    }

    const size_t fade_samples = static_cast<size_t>(std::lround(fade_ms * kSampleRate / 1000.0));
    runtime::AudioBuffer merged{kSampleRate, 1, {}};
    int total_frames = 0;
    int produced_chunks = 0;
    bool reached_limit = false;
    for (size_t index = 0; index < chunks.size(); ++index) {
        const auto & text = chunks[index].text_input->text;
        if (text.empty()) continue;
        const auto prompt = build_zonos2_prompt(hp, text, prompt_options);
        if (prompt.n_rows + max_tokens > hp.n_ctx_train) {
            throw runtime::InvalidRequestError("zonos2: prompt (" + std::to_string(prompt.n_rows) + " rows) and " +
                                               std::to_string(max_tokens) + " generated frames exceed the " +
                                               std::to_string(hp.n_ctx_train) + "-position context");
        }
        auto chunk_sampling = sampling;
        chunk_sampling.seed += static_cast<uint32_t>(index);
        const size_t holdback = index == last_chunk ? fade_samples : 0;

        // Raw delayed rows; undelayed frame f is final once row f + 8 exists and before the end of speech.
        std::vector<int32_t> rows;
        int emitted = 0;
        std::vector<float> held;  // streaming: the faded tail, published at the end
        runtime::StreamingAudioController<int32_t> controller;
        if (streaming) {
            controller.begin({runtime::StreamingAudioChunkPolicy::Grow, static_cast<size_t>(stream_chunk_frames_),
                              static_cast<size_t>(kCodebooks), static_cast<size_t>(stream_context_frames_),
                              static_cast<size_t>(stream_context_frames_)},
                             [&](const runtime::StreamingAudioWindow<int32_t> & window) {
                                 auto pcm = decoder_->decode(window.data, static_cast<int>(window.frames()),
                                                             static_cast<int>(window.left_frames),
                                                             static_cast<int>(window.right_frames));
                                 if (holdback == 0) return runtime::AudioBuffer{kSampleRate, 1, std::move(pcm)};
                                 held.insert(held.end(), pcm.begin(), pcm.end());
                                 const size_t ready = held.size() > holdback ? held.size() - holdback : 0;
                                 std::vector<float> out(held.begin(), held.begin() + static_cast<std::ptrdiff_t>(ready));
                                 held.erase(held.begin(), held.begin() + static_cast<std::ptrdiff_t>(ready));
                                 return runtime::AudioBuffer{kSampleRate, 1, std::move(out)};
                             },
                             stream_event_sink(), {}, {},
                             [&] {
                                 apply_fade(held, holdback);
                                 return runtime::AudioBuffer{kSampleRate, 1, std::move(held)};
                             });
        }
        auto emit = [&](int end) {
            if (!streaming || end <= emitted) return;
            const int count = static_cast<int>(rows.size() / kCodebooks);
            controller.push(zonos2_undelay(rows.data(), count, emitted, end));
            emitted = end;
        };
        const auto generate_started = std::chrono::steady_clock::now();
        const auto codes = generate_zonos2_codes(hp, *backbone_, prompt, speaker.empty() ? nullptr : speaker.data(),
            max_tokens, chunk_sampling, [&](const int32_t * row, int eos_frame) {
                rows.insert(rows.end(), row, row + kCodebooks);
                int ready = static_cast<int>(rows.size() / kCodebooks) - (kCodebooks - 1);
                if (eos_frame >= 0) ready = std::min(ready, eos_frame);
                emit(std::max(ready, 0));
            });
        const int frames = codes.eos_frame >= 0 ? std::min(codes.eos_frame, codes.count) : codes.count;
        engine::debug::timing_log_scalar("zonos2.generate_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - generate_started).count());
        engine::debug::timing_log_scalar("zonos2.generated_frames", codes.count);
        reached_limit = reached_limit || codes.reached_limit;
        runtime::AudioBuffer chunk_audio{kSampleRate, 1, {}};
        if (streaming) {
            emit(frames);
            chunk_audio = controller.finish();
        } else if (frames > 0) {
            const auto decode_started = std::chrono::steady_clock::now();
            const auto undelayed = zonos2_undelay(codes.rows.data(), codes.count, 0, frames);
            chunk_audio.samples = decoder_->decode_tiled(undelayed.data(), frames, kOfflineDecodeFrames, stream_context_frames_);
            engine::debug::timing_log_scalar("zonos2.decode_ms", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_started).count());
            apply_fade(chunk_audio.samples, holdback);
        }
        if (frames <= 0 || chunk_audio.samples.empty()) continue;
        total_frames += frames;
        ++produced_chunks;
        runtime::append_audio_buffer(merged, chunk_audio);
    }
    if (produced_chunks == 0) throw std::runtime_error("zonos2 produced no audio");
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    if (voice_artifact) result.output_artifacts.push_back(std::move(*voice_artifact));
    const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    engine::debug::timing_log_scalar("session.wall_ms", wall_ms);
    result.custom_schema_output = runtime::CustomSchemaOutput{"zonos2_generation_v1", json::Value::make_object({
        {"frames", json::Value::make_number(total_frames)},
        {"chunks", json::Value::make_number(produced_chunks)},
        {"reached_limit", json::Value::make_bool(reached_limit)},
        {"wall_ms", json::Value::make_number(wall_ms)}})};
    return result;
}

}  // namespace engine::models::zonos2
