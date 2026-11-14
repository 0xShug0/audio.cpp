#include "engine/community_models/zonos3/session.h"

#include "engine/community_models/zonos3/codec_decoder.h"
#include "engine/community_models/zonos3/generator.h"
#include "engine/community_models/zonos3/npy.h"
#include "engine/community_models/zonos3/prompt.h"
#include "engine/community_models/zonos3/reference_encoder.h"
#include "engine/community_models/zonos3/speaker_encoder.h"
#include "engine/community_models/zonos3/textnorm.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::zonos3 {
namespace {

namespace json = engine::io::json;

constexpr int kSampleRate = 44100;
constexpr double kReferenceMaxSeconds = 60.0;

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
        bounds << "zonos3 " << key << " must be ";
        if (std::isinf(max)) bounds << ">= " << min;
        else bounds << "between " << min << " and " << max;
        throw runtime::InvalidRequestError(bounds.str());
    }
    return value;
}

int int_option(const Options & options, std::string_view key, int fallback, int min, int max = std::numeric_limits<int>::max()) {
    const int value = client([&] { return runtime::parse_int_option(options, {key}); }).value_or(fallback);
    if (value < min || value > max) {
        throw runtime::InvalidRequestError("zonos3 " + std::string(key) + " must be " +
                                           (max == std::numeric_limits<int>::max() ? ">= " + std::to_string(min)
                                               : "between " + std::to_string(min) + " and " + std::to_string(max)));
    }
    return value;
}

std::optional<double> number_option(const Options & options, std::string_view key) {
    return client([&] { return runtime::parse_finite_float_option(options, {key}); });
}

uint64_t fnv1a(uint64_t hash, const void * data, size_t bytes) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// -1 dBFS tanh soft limiter (ZONOS3 server output_limit_dbfs).
void soft_limit(std::vector<float> & pcm) {
    const float knee = static_cast<float>(std::pow(10.0, -1.0 / 20.0));
    const float width = 1.0f - knee;
    for (auto & x : pcm) {
        const float a = std::fabs(x);
        if (a > knee) x = std::copysign(knee + width * std::tanh((a - knee) / width), x);
    }
}

// Mono 44.1 kHz reference audio, at most 60 s.
std::vector<float> reference_mono(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw runtime::InvalidRequestError("zonos3 reference audio is empty");
    }
    auto mono = audio.channels == 1 ? audio.samples : audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != kSampleRate) {
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, kSampleRate);
    }
    const auto limit = static_cast<size_t>(kReferenceMaxSeconds * kSampleRate);
    if (mono.size() > limit) mono.resize(limit);
    return mono;
}

std::vector<int32_t> parse_condition_buckets(const std::string & text) {
    std::vector<int32_t> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            size_t used = 0;
            const long value = std::stol(item, &used);
            if (used != item.size() || value < 0 || value > std::numeric_limits<int32_t>::max()) throw std::invalid_argument("");
            out.push_back(static_cast<int32_t>(value));
        } catch (const std::exception &) {
            throw runtime::InvalidRequestError("zonos3 condition_buckets needs three comma-separated non-negative integers");
        }
    }
    if (out.size() != 3) {
        throw runtime::InvalidRequestError("zonos3 condition_buckets needs three comma-separated non-negative integers");
    }
    return out;
}

}  // namespace

Zonos3Session::Zonos3Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                             std::shared_ptr<const Zonos3Assets> assets,
                             std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      contract_(std::move(contract)),
      tokenizer_(assets_->tokenizer_json) {
    if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("ZONOS3 supports TTS and voice cloning");
    }
    const auto & session = options.options;
    Zonos3BackboneOptions backbone;
    backbone.moe = core::resolve_moe_execution_policy(session, "zonos3", backbone.moe,
                                                     static_cast<size_t>(assets_->hparams.n_layer - 1));
    if (backbone.moe.cache_slots > 64) {
        throw std::runtime_error("zonos3.expert_cache_size must be at most 64");
    }
    keep_reference_encoder_ = runtime::parse_bool_option(
        runtime::find_option(session, {"zonos3.keep_reference_encoder"}).value_or("false"), "zonos3.keep_reference_encoder");
    const auto slots = runtime::parse_i64_option(session, {"zonos3.voice_cache_slots"}).value_or(4);
    if (slots < 0) throw std::runtime_error("zonos3.voice_cache_slots must be >= 0");
    voices_.set_capacity(static_cast<size_t>(slots));
    stream_chunk_frames_ = static_cast<int>(runtime::parse_positive_i64_option(session, {"zonos3.stream_chunk_frames"}, 16));
    if (stream_chunk_frames_ > 64) throw std::runtime_error("zonos3.stream_chunk_frames must be 1..64");
    const auto encoder_backend = runtime::find_option(session, {"zonos3.encoder_backend"}).value_or("cpu");
    if (encoder_backend != "cpu" && encoder_backend != "same") {
        throw std::runtime_error("zonos3.encoder_backend must be cpu or same");
    }
    core::BackendConfig encoder_config = options.backend;
    if (encoder_backend == "cpu") {
        encoder_config = core::BackendConfig{};
        encoder_config.type = core::BackendType::Cpu;
        encoder_config.threads = options.backend.threads;
    }
    encoder_execution_ = std::make_unique<core::ExecutionContext>(encoder_config);
    backbone_ = std::make_unique<Zonos3Backbone>(assets_, execution_context(), backbone);
    decoder_ = std::make_unique<Zonos3CodecDecoder>(assets_->decoder, execution_context());
}

Zonos3Session::~Zonos3Session() = default;
std::string Zonos3Session::family() const { return "zonos3"; }
runtime::VoiceTaskKind Zonos3Session::task_kind() const { return task_.task; }
runtime::RunMode Zonos3Session::run_mode() const { return task_.mode; }

void Zonos3Session::prepare(const runtime::SessionPreparationRequest & request) {
    const auto execution = acquire_moe_execution();
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "ZONOS3");
    defaults_ = {};
    defaults_.text_input = request.text;
    defaults_.voice = request.voice;
    defaults_.options = request.options;
    mark_prepared();
}

runtime::TaskResult Zonos3Session::run(const runtime::TaskRequest & request) { return synthesize(request, false); }
runtime::TaskResult Zonos3Session::generate_stream(const runtime::TaskRequest & request) { return synthesize(request, true); }

Zonos3Voice Zonos3Session::encode_voice(const runtime::AudioBuffer & audio, bool reference_mode) {
    const auto mono = reference_mono(audio);
    uint64_t key = 1469598103934665603ull;
    const int mode = reference_mode ? 1 : 0;
    key = fnv1a(key, &mode, sizeof(mode));
    key = fnv1a(key, &kSampleRate, sizeof(kSampleRate));
    key = fnv1a(key, mono.data(), mono.size() * sizeof(float));
    if (const auto * cached = voices_.find(key)) return *cached;
    Zonos3Voice voice;
    if (reference_mode) {
        if (!assets_->resources.has_tensor_source("reference_encoder") || !assets_->resources.has_tensor_source("ssl_teacher")) {
            throw runtime::InvalidRequestError(
                "zonos3 speaker_mode=reference needs zac-encoder.gguf and ssl-teacher.gguf next to the backbone "
                "(package zonos3_mini_ptq_raw_reference)");
        }
        if (!reference_encoder_) {
            reference_encoder_ = std::make_unique<Zonos3ReferenceEncoder>(
                assets_->resources.open_tensor_source("reference_encoder"), assets_->resources.open_tensor_source("ssl_teacher"),
                *encoder_execution_);
        }
        voice.codes = reference_encoder_->encode(mono);
        if (voice.codes.size() > static_cast<size_t>(kZonos3ReferenceMaxFrames) * kZonos3Codebooks) {
            voice.codes.resize(static_cast<size_t>(kZonos3ReferenceMaxFrames) * kZonos3Codebooks);
        }
        if (!keep_reference_encoder_) reference_encoder_.reset();
    } else {
        if (!speaker_encoder_) {
            speaker_encoder_ = std::make_unique<Zonos3SpeakerEncoder>(assets_->speaker_encoder, *encoder_execution_);
        }
        voice.embedding = speaker_encoder_->embed(mono);
    }
    voices_.put(key, voice);
    return voice;
}

runtime::TaskResult Zonos3Session::synthesize(const runtime::TaskRequest & request, bool streaming) {
    const auto execution = acquire_moe_execution();
    require_prepared("ZONOS3 generation");
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "ZONOS3");
    const auto started = std::chrono::steady_clock::now();
    auto effective = defaults_;
    if (request.text_input) effective.text_input = request.text_input;
    if (request.voice) effective.voice = request.voice;
    for (const auto & option : request.options) effective.options[option.first] = option.second;
    const auto & options = effective.options;
    const auto & hp = assets_->hparams;

    if (effective.voice && effective.voice->style) {
        throw runtime::InvalidRequestError(
            "zonos3 has no style controls; use the loudness, fmax and speaking_rate_bps request options");
    }
    const std::string language = zonos3_language(runtime::find_option(options, {"language"}).value_or("en_us"));
    const bool normalize = bool_option(options, "text_normalization", true);
    Zonos3SamplingOptions sampling;
    sampling.temperature = float_option(options, "temperature", 0.75f, 0.0f);
    sampling.top_k = int_option(options, "top_k", 80, 1);
    sampling.top_p = float_option(options, "top_p", 0.69f, 0.0f, 1.0f);
    sampling.min_p = float_option(options, "min_p", 0.0f, 0.0f, 1.0f);
    sampling.repetition_penalty = float_option(options, "repetition_penalty", 2.4f, 1.0f);
    sampling.repetition_window = int_option(options, "repetition_window", 96, 1);
    sampling.repetition_codebooks = int_option(options, "repetition_codebooks", 6, -1, 10);
    sampling.seed = client([&] { return runtime::parse_u32_option(options, {"seed"}); }).value_or(runtime::random_u32_seed());
    const int max_tokens = int_option(options, "max_tokens", 2000, 1);
    const float guidance_scale = float_option(options, "guidance_scale", 1.0f, 0.0f);
    const auto speaker_mode = runtime::find_option(options, {"speaker_mode"}).value_or("embedding");
    if (speaker_mode != "embedding" && speaker_mode != "reference") {
        throw runtime::InvalidRequestError("zonos3 speaker_mode must be embedding or reference");
    }
    const bool encode_only = bool_option(options, "encode_reference_only", false);
    const int prefill = int_option(options, "silence_prefill_frames", 3, 0, kZonos3SilencePrefillMax);
    const float headroom_db = float_option(options, "headroom_db", decoder_->headroom_db(), 0.0f, 24.0f);
    const bool limit = bool_option(options, "soft_limit", true);
    const int chunk_size = int_option(options, "text_chunk_size", 300, 1);

    // Voice source: exactly one of an embedding file, a codes file, or reference audio.
    const auto embedding_file = runtime::find_option(options, {"speaker_embedding_file"});
    const auto codes_file = runtime::find_option(options, {"reference_codes_file"});
    const bool has_audio = effective.voice && effective.voice->speaker && effective.voice->speaker->audio;
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->cached_voice_id && !has_audio) {
        throw runtime::InvalidRequestError(
            "zonos3 has no built-in voices; pass --voice-ref, speaker_embedding_file or reference_codes_file");
    }
    const int sources = static_cast<int>(embedding_file.has_value()) + static_cast<int>(codes_file.has_value()) +
        static_cast<int>(has_audio);
    if (sources > 1) {
        throw runtime::InvalidRequestError(
            "zonos3: give one voice source: --voice-ref, speaker_embedding_file or reference_codes_file");
    }
    if (encode_only && !has_audio) throw runtime::InvalidRequestError("zonos3 encode_reference_only needs --voice-ref");
    if (task_.task == runtime::VoiceTaskKind::VoiceCloning && sources == 0) {
        throw runtime::InvalidRequestError(
            "zonos3 clone requires a voice: --voice-ref, speaker_embedding_file or reference_codes_file");
    }
    Zonos3Voice voice;
    std::optional<runtime::VoiceArtifact> voice_artifact;
    const std::unordered_map<std::string, std::string> npy_meta{
        {"family", "zonos3"}, {"mime", "application/x-npy"}, {"extension", ".npy"}};
    if (embedding_file) {
        voice.embedding = read_speaker_embedding_npy(*embedding_file);
    } else if (codes_file) {
        voice.codes = read_reference_codes_npy(*codes_file, kZonos3ReferenceMaxFrames);
    } else if (has_audio) {
        const bool reference = speaker_mode == "reference";
        voice = encode_voice(*effective.voice->speaker->audio, reference);
        voice_artifact = reference
            ? runtime::make_voice_artifact(runtime::ArtifactKind::AcousticTokens, "zonos3_reference_codes",
                                           reference_codes_npy(voice.codes), npy_meta)
            : runtime::make_voice_artifact(runtime::ArtifactKind::SpeakerEmbedding, "zonos3_speaker_embedding",
                                           speaker_embedding_npy(voice.embedding), npy_meta);
    }
    if (encode_only) {
        runtime::TaskResult result;
        result.output_artifacts.push_back(std::move(*voice_artifact));
        return result;
    }
    if (guidance_scale != 1.0f && sources == 0) throw runtime::InvalidRequestError("zonos3 guidance_scale needs a speaker source");

    // Condition buckets: an explicit id wins, then a value bucketed by the checkpoint's edges, then unknown.
    std::vector<int32_t> buckets = hp.cond_bucket_counts;
    std::vector<int32_t> explicit_buckets;
    if (const auto text = runtime::find_option(options, {"condition_buckets"})) explicit_buckets = parse_condition_buckets(*text);
    const std::array<const char *, 3> value_options{"loudness", "fmax", "speaking_rate_bps"};
    for (int i = 0; i < hp.n_cond; ++i) {
        if (!explicit_buckets.empty()) {
            buckets[static_cast<size_t>(i)] = explicit_buckets[static_cast<size_t>(i)];
        } else if (const auto value = number_option(options, value_options[static_cast<size_t>(i)])) {
            buckets[static_cast<size_t>(i)] = zonos3_condition_bucket(hp, i, *value);
        }
    }

    if (!effective.text_input || effective.text_input->text.empty()) {
        throw runtime::InvalidRequestError("zonos3 requires input text");
    }
    if (normalize && !zonos3_textnorm_linked()) {
        static const bool warned = [] {
            std::cerr << "zonos3: text normalization not built (AUDIOCPP_ZONOS3_TEXTNORM=OFF); line breaks collapsed only\n";
            return true;
        }();
        (void) warned;
    }
    auto normalized = effective;
    normalized.text_input->text = zonos3_normalize_text(effective.text_input->text, language, normalize);
    const auto mode = engine::text::parse_text_chunk_mode_override(options).value_or(engine::text::TextChunkMode::Default);
    const auto chunks = runtime::chunk_text_request(normalized, chunk_size, mode);

    Zonos3PromptOptions prompt_options;
    prompt_options.condition_buckets = buckets;
    prompt_options.silence_prefill_frames = prefill;
    if (!voice.embedding.empty()) prompt_options.speaker_embedding = voice.embedding.data();
    if (!voice.codes.empty()) {
        prompt_options.reference_codes = voice.codes.data();
        prompt_options.reference_frames = static_cast<int>(voice.codes.size() / kZonos3Codebooks);
    }
    const bool anchor_voice = chunks.size() > 1 && voice.embedding.empty() && voice.codes.empty();
    std::vector<int32_t> anchor_codes;
    if (anchor_voice) anchor_codes.reserve(static_cast<size_t>(kZonos3ReferenceMaxFrames) * kZonos3Codebooks);

    const int spf = decoder_->samples_per_frame();
    const float gain = static_cast<float>(std::pow(10.0, -headroom_db / 20.0));
    runtime::AudioBuffer merged{kSampleRate, 1, {}};
    int total_frames = 0;
    int produced_chunks = 0;
    bool reached_limit = false;
    for (size_t index = 0; index < chunks.size(); ++index) {
        const auto & text = chunks[index].text_input->text;
        if (text.empty()) continue;
        const auto prompt = build_zonos3_prompt(hp, tokenizer_, text, prompt_options);
        if (prompt.n_rows + max_tokens > hp.n_ctx_train) {
            throw runtime::InvalidRequestError("zonos3: prompt (" + std::to_string(prompt.n_rows) + " rows) and " +
                                               std::to_string(max_tokens) + " generated frames exceed the " +
                                               std::to_string(hp.n_ctx_train) + "-position context");
        }
        std::optional<Zonos3Prompt> twin;
        if (guidance_scale != 1.0f) twin = build_zonos3_cfg_twin(hp, prompt);
        auto chunk_sampling = sampling;
        chunk_sampling.seed += static_cast<uint32_t>(index);

        // The delayed stream (silence prefill rows + generated rows); frame f is complete once row f + 9 exists.
        std::vector<int32_t> stream = zonos3_prefill_codes(hp, prompt);
        stream.reserve(stream.size() + static_cast<size_t>(max_tokens) * kZonos3Codebooks);
        int emitted = 0;
        size_t skip = static_cast<size_t>(prompt.prefill_frames) * static_cast<size_t>(spf);
        auto post = [&](std::vector<float> pcm) {
            const size_t drop = std::min(skip, pcm.size());
            pcm.erase(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(drop));
            skip -= drop;
            for (auto & x : pcm) x *= gain;
            if (limit) soft_limit(pcm);
            return pcm;
        };
        Zonos3DecoderStream decoder(*decoder_);
        runtime::StreamingAudioController<int32_t> controller;
        std::vector<int32_t> pending;  // offline: complete frames
        if (!streaming) pending.reserve(stream.capacity());
        std::vector<int32_t> frame(kZonos3Codebooks);
        const bool capture_anchor = anchor_voice && index == 0;
        if (streaming) {
            controller.begin({runtime::StreamingAudioChunkPolicy::Grow, static_cast<size_t>(stream_chunk_frames_),
                              static_cast<size_t>(kZonos3Codebooks), 0, 0},
                             [&](const runtime::StreamingAudioWindow<int32_t> & window) {
                                 return runtime::AudioBuffer{kSampleRate, 1,
                                     post(decoder.decode(window.data, static_cast<int>(window.new_frames)))};
                             },
                             stream_event_sink(), {}, {},
                             [&] { return runtime::AudioBuffer{kSampleRate, 1, post(decoder.flush())}; });
        }
        auto emit = [&](int end) {
            for (; emitted < end; ++emitted) {
                for (int c = 0; c < kZonos3Codebooks; ++c) {
                    frame[static_cast<size_t>(c)] = std::min(
                        stream[static_cast<size_t>(emitted + c) * kZonos3Codebooks + static_cast<size_t>(c)],
                        static_cast<int32_t>(kZonos3CodebookSize - 1));
                }
                if (capture_anchor && emitted >= prompt.prefill_frames &&
                    anchor_codes.size() < static_cast<size_t>(kZonos3ReferenceMaxFrames) * kZonos3Codebooks) {
                    anchor_codes.insert(anchor_codes.end(), frame.begin(), frame.end());
                }
                if (streaming) controller.push(frame);
                else pending.insert(pending.end(), frame.begin(), frame.end());
            }
        };
        const auto codes = generate_codes(hp, *backbone_, prompt, twin ? &*twin : nullptr, guidance_scale, max_tokens,
            chunk_sampling, [&](const int32_t * row, int eos_frame) {
                stream.insert(stream.end(), row, row + kZonos3Codebooks);
                const int rows = static_cast<int>(stream.size() / kZonos3Codebooks);
                int ready = rows - (kZonos3Codebooks - 1);
                if (eos_frame >= 0) ready = std::min(ready, eos_frame);
                emit(std::max(ready, 0));
            });
        const int rows = static_cast<int>(stream.size() / kZonos3Codebooks);
        const int frames = std::max(0, std::min(codes.eos_frame >= 0 ? codes.eos_frame : rows, rows - (kZonos3Codebooks - 1)));
        emit(frames);
        reached_limit = reached_limit || codes.reached_limit;
        const auto codec_started = std::chrono::steady_clock::now();
        runtime::AudioBuffer chunk_audio{kSampleRate, 1, {}};
        if (streaming) {
            chunk_audio = controller.finish();
        } else if (frames > 0) {
            chunk_audio.samples = post(decoder.decode(pending.data(), frames));
            const auto tail = post(decoder.flush());
            chunk_audio.samples.insert(chunk_audio.samples.end(), tail.begin(), tail.end());
        }
        engine::debug::timing_log_scalar("zonos3.audio_decode_finish_ms",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - codec_started).count());
        if (frames <= prompt.prefill_frames || chunk_audio.samples.empty()) continue;
        if (capture_anchor) {
            prompt_options.reference_codes = anchor_codes.data();
            prompt_options.reference_frames = static_cast<int>(anchor_codes.size() / kZonos3Codebooks);
        }
        total_frames += frames - prompt.prefill_frames;
        ++produced_chunks;
        runtime::append_audio_buffer(merged, chunk_audio);
    }
    if (produced_chunks == 0) throw std::runtime_error("zonos3 produced no audio");
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    if (voice_artifact) result.output_artifacts.push_back(std::move(*voice_artifact));
    const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    engine::debug::timing_log_scalar("session.wall_ms", wall_ms);
    result.custom_schema_output = runtime::CustomSchemaOutput{"zonos3_generation_v1", json::Value::make_object({
        {"frames", json::Value::make_number(total_frames)},
        {"chunks", json::Value::make_number(produced_chunks)},
        {"reached_limit", json::Value::make_bool(reached_limit)},
        {"wall_ms", json::Value::make_number(wall_ms)}})};
    return result;
}

}  // namespace engine::models::zonos3
