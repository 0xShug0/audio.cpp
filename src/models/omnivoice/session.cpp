#include "engine/models/omnivoice/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

constexpr size_t kDefaultAudioTokenizerGraphArenaBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorPrefillGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorDecodeGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultAudioTokenizerWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultGeneratorWeightContextBytes = 256ull * 1024ull * 1024ull;

std::shared_ptr<const OmniVoiceAssets> require_assets(std::shared_ptr<const OmniVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("OmniVoice session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return fallback;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"omnivoice.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "omnivoice.mem_saver");
    }
    return false;
}

using Clock = std::chrono::steady_clock;

OmniVoiceGenerationOptions generation_options_from_options(const std::unordered_map<std::string, std::string> & options_map) {
    OmniVoiceGenerationOptions options;
    if (const auto seed = runtime::parse_u32_option(options_map, {"seed"})) {
        options.seed = *seed;
    }
    options.num_inference_steps = runtime::parse_int_option(options_map, {"num_inference_steps"})
        .value_or(options.num_inference_steps);
    options.guidance_scale = runtime::parse_float_option(options_map, {"guidance_scale"})
        .value_or(options.guidance_scale);
    options.speed = runtime::parse_float_option(options_map, {"speed"}).value_or(options.speed);
    options.duration_seconds = runtime::parse_float_option(options_map, {"duration"});
    options.t_shift = runtime::parse_float_option(options_map, {"t_shift"}).value_or(options.t_shift);
    if (const auto value = runtime::find_option(options_map, {"denoise"})) {
        options.denoise = runtime::parse_bool_option(*value, "denoise");
    }
    if (const auto value = runtime::find_option(options_map, {"preprocess_prompt"})) {
        options.preprocess_prompt = runtime::parse_bool_option(*value, "preprocess_prompt");
    }
    if (const auto value = runtime::find_option(options_map, {"postprocess_output"})) {
        options.postprocess_output = runtime::parse_bool_option(*value, "postprocess_output");
    }
    options.layer_penalty_factor = runtime::parse_float_option(
        options_map,
        {"layer_penalty_factor"})
        .value_or(options.layer_penalty_factor);
    options.position_temperature = runtime::parse_float_option(
        options_map,
        {"position_temperature"})
        .value_or(options.position_temperature);
    options.class_temperature = runtime::parse_float_option(
        options_map,
        {"class_temperature"})
        .value_or(options.class_temperature);
    options.audio_chunk_duration_seconds = runtime::parse_float_option(
        options_map,
        {"audio_chunk_duration"})
        .value_or(options.audio_chunk_duration_seconds);
    options.audio_chunk_threshold_seconds = runtime::parse_float_option(
        options_map,
        {"audio_chunk_threshold"})
        .value_or(options.audio_chunk_threshold_seconds);
    options.text_chunk_size = engine::text::parse_text_chunk_size_override(options_map);
    options.text_chunk_mode = engine::text::parse_text_chunk_mode_override(options_map)
        .value_or(engine::text::TextChunkMode::TagAware);
    return options;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(sample));
        std::memcpy(&bits, &sample, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>((bits >> shift) & 0xFFu);
            hash *= kPrime;
        }
    }
    return hash;
}

std::optional<std::string> request_option(
    const std::unordered_map<std::string, std::string> & options,
    const char * primary,
    const char * secondary = nullptr) {
    if (const auto it = options.find(primary); it != options.end()) {
        return it->second;
    }
    if (secondary != nullptr) {
        if (const auto it = options.find(secondary); it != options.end()) {
            return it->second;
        }
    }
    return std::nullopt;
}

void append_cross_faded_chunk(
    runtime::AudioBuffer & merged,
    const runtime::AudioBuffer & chunk,
    float silence_duration_seconds = 0.3F) {
    if (chunk.sample_rate <= 0 || chunk.channels != 1) {
        throw std::runtime_error("OmniVoice append_cross_faded_chunk requires mono chunk audio");
    }
    if (merged.sample_rate == 0) {
        merged = chunk;
        return;
    }
    if (merged.sample_rate != chunk.sample_rate || merged.channels != 1) {
        throw std::runtime_error("OmniVoice append_cross_faded_chunk requires matching mono chunk audio");
    }

    const int sample_rate = chunk.sample_rate;
    const size_t total_n = static_cast<size_t>(std::max(0.0F, silence_duration_seconds) * static_cast<float>(sample_rate));
    const size_t fade_n = total_n / 3;
    const size_t silence_n = fade_n;
    const auto fade_weight = [](size_t index, size_t count, float start, float end) {
        if (count <= 1) {
            return start;
        }
        const float t = static_cast<float>(index) / static_cast<float>(count - 1);
        return start + (end - start) * t;
    };
    const size_t fout_n = std::min(fade_n, merged.samples.size());
    if (fout_n > 0) {
        for (size_t i = 0; i < fout_n; ++i) {
            const float w_out = fade_weight(i, fout_n, 1.0F, 0.0F);
            merged.samples[merged.samples.size() - fout_n + i] *= w_out;
        }
    }
    merged.samples.insert(merged.samples.end(), silence_n, 0.0F);
    const size_t chunk_start = merged.samples.size();
    merged.samples.insert(merged.samples.end(), chunk.samples.begin(), chunk.samples.end());
    const size_t fin_n = std::min(fade_n, chunk.samples.size());
    if (fin_n > 0) {
        for (size_t i = 0; i < fin_n; ++i) {
            const float w_in = fade_weight(i, fin_n, 0.0F, 1.0F);
            merged.samples[chunk_start + i] *= w_in;
        }
    }
}

}  // namespace

OmniVoiceSession::OmniVoiceSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const OmniVoiceAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_tokenizer_graph_arena_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.audio_tokenizer_graph_arena_mb"}, kDefaultAudioTokenizerGraphArenaBytes)),
      generator_prefill_graph_arena_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.generator_prefill_graph_arena_mb"}, kDefaultGeneratorPrefillGraphArenaBytes)),
      generator_decode_graph_arena_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.generator_decode_graph_arena_mb"}, kDefaultGeneratorDecodeGraphArenaBytes)),
      audio_tokenizer_weight_context_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.audio_tokenizer_weight_context_mb"}, kDefaultAudioTokenizerWeightContextBytes)),
      generator_weight_context_bytes_(
          runtime::parse_size_mb_option(options.options, {"omnivoice.generator_weight_context_mb"}, kDefaultGeneratorWeightContextBytes)),
      audio_tokenizer_weight_storage_type_(
          option_weight_type(options, "omnivoice.audio_tokenizer_weight_type", engine::assets::TensorStorageType::Native)),
      generator_weight_storage_type_(
          option_weight_type(
              options,
              "omnivoice.generator_weight_type",
              option_weight_type(options, "omnivoice.weight_type", engine::assets::TensorStorageType::Native))),
      mem_saver_(mem_saver_from_options(options)),
      tokenizer_(assets_),
      audio_tokenizer_(
          assets_,
          execution_context(),
          audio_tokenizer_graph_arena_bytes_,
          audio_tokenizer_weight_context_bytes_,
          audio_tokenizer_weight_storage_type_),
      prompt_builder_(assets_, tokenizer_),
      generator_(
          assets_,
          execution_context(),
          generator_prefill_graph_arena_bytes_,
          generator_decode_graph_arena_bytes_,
          generator_weight_context_bytes_,
          generator_weight_storage_type_,
          mem_saver_) {
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("OmniVoice only supports VoiceTaskKind::Tts");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("OmniVoice currently supports offline sessions only");
    }
    assets_->model_weights->release_storage();
    assets_->audio_tokenizer_weights->release_storage();
}

std::string OmniVoiceSession::family() const {
    return "omnivoice";
}

runtime::VoiceTaskKind OmniVoiceSession::task_kind() const {
    return task_.task;
}

runtime::RunMode OmniVoiceSession::run_mode() const {
    return task_.mode;
}

void OmniVoiceSession::prepare(const runtime::SessionPreparationRequest & request) {
    session_defaults_ = {};
    reference_prompt_cache_.reset();
    if (request.text.has_value()) {
        session_defaults_.text = request.text;
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->audio.has_value()) {
        session_defaults_.reference_audio = *request.voice->speaker->audio;
    }
    session_defaults_.options = request.options;
    if (const auto reference_text = request_option(request.options, "reference_text"); reference_text.has_value()) {
        session_defaults_.reference_text = std::move(reference_text);
    }
    if (const auto instruct = request_option(request.options, "instruct"); instruct.has_value()) {
        session_defaults_.instruct = std::move(instruct);
    } else if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto it = request.voice->style->tags.find("instruct");
        if (it != request.voice->style->tags.end()) {
            session_defaults_.instruct = it->second;
        }
    }
    if (session_defaults_.reference_audio.has_value()) {
        const auto merged_options = session_defaults_.options;
        const auto generation = generation_options_from_options(merged_options);
        const bool reference_text_provided =
            session_defaults_.reference_text.has_value() && !session_defaults_.reference_text->empty();
        (void) resolve_reference_audio_tokens(
            *session_defaults_.reference_audio,
            generation.preprocess_prompt,
            reference_text_provided);
        if (mem_saver_) {
            audio_tokenizer_.release_runtime_graphs();
        }
    }
    mark_prepared();
}

runtime::TaskResult OmniVoiceSession::run(const runtime::TaskRequest & request) {
    require_prepared("OmniVoice run()");
    const auto wall_start = Clock::now();
    auto omni_request = make_request(request);
    if (omni_request.generation.seed.has_value()) {
        generator_.seed_rng(*omni_request.generation.seed);
    }
    runtime::TaskResult task_result;
    std::optional<double> reference_encode_ms = std::nullopt;
    bool encoder_graph_rebuilt = false;
    if (omni_request.reference_audio.has_value()) {
        const auto encode_start = Clock::now();
        const auto reference_tokens = resolve_reference_audio_tokens(
            *omni_request.reference_audio,
            omni_request.generation.preprocess_prompt,
            !omni_request.reference_text.empty());
        const auto encode_end = Clock::now();
        omni_request.reference_rms = reference_tokens.reference_rms;
        omni_request.reference_audio_tokens = std::move(reference_tokens);
        omni_request.reference_audio.reset();
        reference_encode_ms = engine::debug::elapsed_ms(encode_start, encode_end);
        const auto & tokenizer_stats = audio_tokenizer_.last_stats();
        encoder_graph_rebuilt = tokenizer_stats.encoder_graph_rebuilt;
        if (mem_saver_) {
            const auto release_start = Clock::now();
            audio_tokenizer_.release_runtime_graphs();
            engine::debug::timing_log_scalar(
                "omnivoice.reference_release.runtime_graphs_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
    }
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(omni_request);
    const auto prompt_end = Clock::now();
    const int64_t chunk_threshold = static_cast<int64_t>(std::llround(
        static_cast<double>(omni_request.generation.audio_chunk_threshold_seconds) *
        static_cast<double>(prompt_builder_.frame_rate())));
    OmniVoiceGeneratedAudioTokens generated_tokens;
    runtime::AudioBuffer audio = {};
    bool any_generator_graph_rebuilt = false;
    bool any_decoder_graph_rebuilt = false;
    double generator_rebuild_ms = 0.0;
    double decoder_rebuild_ms = 0.0;
    double chunk_generate_ms = 0.0;
    double chunk_decode_ms = 0.0;
    const auto generate_start = Clock::now();
    double measured_generate_ms = 0.0;
    double measured_decode_ms = 0.0;
    const bool use_explicit_text_chunking = omni_request.generation.text_chunk_size.has_value();
    if (!use_explicit_text_chunking && prompt.target_audio_tokens <= chunk_threshold) {
        generated_tokens = generator_.generate(prompt, omni_request.generation);
        {
            const auto & generator_stats = generator_.last_stats();
            any_generator_graph_rebuilt = any_generator_graph_rebuilt || generator_stats.graph_rebuilt;
            generator_rebuild_ms += generator_stats.rebuild_ms;
        }
        const auto generate_end = Clock::now();
        if (mem_saver_) {
            const auto release_start = Clock::now();
            generator_.release_runtime_graphs();
            engine::debug::timing_log_scalar(
                "omnivoice.generator_release.runtime_graphs_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
        measured_generate_ms = engine::debug::elapsed_ms(generate_start, generate_end);
        const auto decode_start = Clock::now();
        audio = audio_tokenizer_.decode_audio_tokens(generated_tokens);
        {
            const auto & tokenizer_stats = audio_tokenizer_.last_stats();
            any_decoder_graph_rebuilt = any_decoder_graph_rebuilt || tokenizer_stats.decoder_graph_rebuilt;
            decoder_rebuild_ms += tokenizer_stats.decoder_rebuild_ms;
        }
        const auto decode_end = Clock::now();
        if (mem_saver_) {
            const auto release_start = Clock::now();
            audio_tokenizer_.release_runtime_graphs();
            engine::debug::timing_log_scalar(
                "omnivoice.decoder_release.runtime_graphs_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
        measured_decode_ms = engine::debug::elapsed_ms(decode_start, decode_end);
    } else {
        std::vector<std::string> text_chunks;
        int64_t text_chunk_size = 0;
        if (use_explicit_text_chunking) {
            text_chunk_size = *omni_request.generation.text_chunk_size;
            text_chunks = engine::text::split_text_chunks(
                prompt.text,
                text_chunk_size,
                omni_request.generation.text_chunk_mode);
        } else {
            const double avg_tokens_per_char =
                static_cast<double>(prompt.target_audio_tokens) /
                static_cast<double>(std::max<size_t>(1, prompt.text.size()));
            text_chunk_size = std::max<int64_t>(
                1,
                static_cast<int64_t>(omni_request.generation.audio_chunk_duration_seconds *
                                     static_cast<float>(prompt_builder_.frame_rate()) / avg_tokens_per_char));
            text_chunks = prompt_builder_.chunk_text_punctuation(prompt.text, text_chunk_size, 3);
        }
        engine::debug::trace_log_scalar("omnivoice.text_chunk_size", text_chunk_size);
        if (use_explicit_text_chunking) {
            engine::debug::trace_log_scalar(
                "omnivoice.text_chunk_mode",
                engine::text::text_chunk_mode_name(omni_request.generation.text_chunk_mode));
        }
        engine::debug::trace_log_scalar("omnivoice.text_chunk_count", static_cast<int64_t>(text_chunks.size()));
        const bool has_reference_audio = omni_request.reference_audio_tokens.has_value();
        std::optional<OmniVoiceAudioTokens> first_chunk_reference = std::nullopt;
        std::string first_chunk_text;
        OmniVoiceRequest chunk_request = omni_request;
        int64_t chunk_codebooks = 0;
        chunk_request.generation.duration_seconds.reset();
        if (omni_request.generation.duration_seconds.has_value()) {
            const int64_t full_estimate = prompt_builder_.estimate_target_tokens(
                prompt.text,
                prompt.reference_text.empty() ? std::optional<std::string>() : std::make_optional(prompt.reference_text),
                prompt.reference_audio_tokens.has_value()
                    ? std::make_optional(prompt.reference_audio_tokens->frames)
                    : std::optional<int64_t>(),
                1.0F);
            const int64_t duration_target = std::max<int64_t>(
                1,
                static_cast<int64_t>(std::llround(
                static_cast<double>(*omni_request.generation.duration_seconds) *
                    static_cast<double>(prompt_builder_.frame_rate()))));
            chunk_request.generation.speed = static_cast<float>(full_estimate) / static_cast<float>(duration_target);
        }
        const int sample_rate = assets_->config.audio_tokenizer.sample_rate;
        const int64_t hop_length = assets_->config.audio_tokenizer.hop_length;
        if (sample_rate > 0 && hop_length > 0) {
            const size_t estimated_audio_samples =
                static_cast<size_t>(prompt.target_audio_tokens) * static_cast<size_t>(hop_length);
            const size_t chunk_gap_samples = text_chunks.size() > 1
                ? static_cast<size_t>(std::llround(0.3 * static_cast<double>(sample_rate))) *
                    static_cast<size_t>(text_chunks.size() - 1)
                : 0;
            audio.samples.reserve(estimated_audio_samples + chunk_gap_samples);
        }
        for (size_t chunk_index = 0; chunk_index < text_chunks.size(); ++chunk_index) {
            chunk_request.text = text_chunks[chunk_index];
            if (!has_reference_audio && chunk_index > 0) {
                chunk_request.reference_audio_tokens = *first_chunk_reference;
                chunk_request.reference_text = first_chunk_text;
            }
            const auto chunk_prompt = prompt_builder_.build(chunk_request);
            const auto chunk_generate_start = Clock::now();
            auto chunk_tokens = generator_.generate(chunk_prompt, chunk_request.generation);
            const auto chunk_generate_end = Clock::now();
            {
                const auto & generator_stats = generator_.last_stats();
                any_generator_graph_rebuilt = any_generator_graph_rebuilt || generator_stats.graph_rebuilt;
                generator_rebuild_ms += generator_stats.rebuild_ms;
            }
            if (mem_saver_) {
                const auto release_start = Clock::now();
                generator_.release_runtime_graphs();
                engine::debug::timing_log_scalar(
                    "omnivoice.generator_release.runtime_graphs_ms",
                    engine::debug::elapsed_ms(release_start, Clock::now()));
            }
            chunk_generate_ms += engine::debug::elapsed_ms(chunk_generate_start, chunk_generate_end);
            if (chunk_codebooks == 0) {
                chunk_codebooks = chunk_tokens.codebooks;
            } else if (chunk_codebooks != chunk_tokens.codebooks) {
                throw std::runtime_error("OmniVoice chunk token codebook mismatch");
            }
            if (!has_reference_audio && chunk_index == 0) {
                OmniVoiceAudioTokens chunk_ref = {};
                chunk_ref.frames = chunk_tokens.frames;
                chunk_ref.codebooks = chunk_tokens.codebooks;
                chunk_ref.token_ids = chunk_tokens.token_ids;
                first_chunk_reference = std::move(chunk_ref);
                first_chunk_text = text_chunks.front();
            }
            const auto chunk_decode_start = Clock::now();
            auto chunk_audio = audio_tokenizer_.decode_audio_tokens(chunk_tokens);
            const auto chunk_decode_end = Clock::now();
            {
                const auto & tokenizer_stats = audio_tokenizer_.last_stats();
                any_decoder_graph_rebuilt = any_decoder_graph_rebuilt || tokenizer_stats.decoder_graph_rebuilt;
                decoder_rebuild_ms += tokenizer_stats.decoder_rebuild_ms;
            }
            if (mem_saver_) {
                const auto release_start = Clock::now();
                audio_tokenizer_.release_runtime_graphs();
                engine::debug::timing_log_scalar(
                    "omnivoice.decoder_release.runtime_graphs_ms",
                    engine::debug::elapsed_ms(release_start, Clock::now()));
            }
            chunk_decode_ms += engine::debug::elapsed_ms(chunk_decode_start, chunk_decode_end);
            append_cross_faded_chunk(audio, chunk_audio);
        }
        measured_generate_ms = chunk_generate_ms;
        measured_decode_ms = chunk_decode_ms;
    }
    const auto postprocess_start = Clock::now();
    const auto result = postprocessor_.finalize(audio, omni_request);
    const auto postprocess_end = Clock::now();
    task_result.audio_output = result.audio;
    if (reference_encode_ms.has_value()) {
        engine::debug::timing_log_scalar("omnivoice.reference.encode_ms", *reference_encode_ms);
        engine::debug::timing_log_scalar("omnivoice.encoder.graph.rebuilt", encoder_graph_rebuilt);
    }
    engine::debug::timing_log_scalar("omnivoice.session.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    engine::debug::timing_log_scalar("omnivoice.session.generate_ms", measured_generate_ms);
    engine::debug::timing_log_scalar("omnivoice.generator.graph.rebuilt", any_generator_graph_rebuilt);
    engine::debug::timing_log_scalar("omnivoice.generator.graph.rebuild_ms", generator_rebuild_ms);
    engine::debug::timing_log_scalar("omnivoice.session.decode_ms", measured_decode_ms);
    engine::debug::timing_log_scalar("omnivoice.decoder.graph.rebuilt", any_decoder_graph_rebuilt);
    engine::debug::timing_log_scalar("omnivoice.decoder.graph.rebuild_ms", decoder_rebuild_ms);
    engine::debug::timing_log_scalar("omnivoice.session.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, postprocess_end));
    return task_result;
}

OmniVoiceRequest OmniVoiceSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() && !session_defaults_.text.has_value()) {
        throw std::runtime_error("OmniVoice run() requires text_input");
    }
    OmniVoiceRequest out;
    if (request.text_input.has_value()) {
        out.text = request.text_input->text;
        out.language = request.text_input->language;
    } else {
        out.text = session_defaults_.text->text;
        out.language = session_defaults_.text->language;
    }
    const auto merged_options = merged_request_options(request);
    out.generation = generation_options_from_options(merged_options);
    out.reference_audio = resolve_reference_audio(request);
    if (const auto reference_text = resolve_reference_text(request); reference_text.has_value()) {
        out.reference_text = *reference_text;
    }
    if (const auto instruct = resolve_instruct(request); instruct.has_value()) {
        out.instruct = *instruct;
    }
    return out;
}

std::optional<runtime::AudioBuffer> OmniVoiceSession::resolve_reference_audio(const runtime::TaskRequest & request) const {
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->audio.has_value()) {
        return *request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) {
        return *request.audio_input;
    }
    return session_defaults_.reference_audio;
}

std::optional<std::string> OmniVoiceSession::resolve_reference_text(const runtime::TaskRequest & request) const {
    if (const auto value = request_option(request.options, "reference_text"); value.has_value()) {
        return value;
    }
    return session_defaults_.reference_text;
}

std::optional<std::string> OmniVoiceSession::resolve_instruct(const runtime::TaskRequest & request) const {
    if (const auto value = request_option(request.options, "instruct"); value.has_value()) {
        return value;
    }
    if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto it = request.voice->style->tags.find("instruct");
        if (it != request.voice->style->tags.end()) {
            return it->second;
        }
    }
    return session_defaults_.instruct;
}

std::unordered_map<std::string, std::string> OmniVoiceSession::merged_request_options(const runtime::TaskRequest & request) const {
    auto merged = session_defaults_.options;
    for (const auto & [key, value] : request.options) {
        merged[key] = value;
    }
    return merged;
}

OmniVoiceAudioTokens OmniVoiceSession::resolve_reference_audio_tokens(
    const runtime::AudioBuffer & audio,
    bool preprocess_prompt,
    bool reference_text_provided) {
    const uint64_t sample_count = static_cast<uint64_t>(audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(audio);
    const bool cache_hit = reference_prompt_cache_.has_value() &&
        reference_prompt_cache_->preprocess_prompt == preprocess_prompt &&
        reference_prompt_cache_->reference_text_provided == reference_text_provided &&
        reference_prompt_cache_->sample_rate == audio.sample_rate &&
        reference_prompt_cache_->channels == audio.channels &&
        reference_prompt_cache_->sample_count == sample_count &&
        reference_prompt_cache_->sample_hash == sample_hash;
    if (!cache_hit) {
        const OmniVoiceReferenceAudioOptions reference_options = {
            preprocess_prompt,
            reference_text_provided,
        };
        ReferencePromptCacheEntry entry;
        entry.preprocess_prompt = preprocess_prompt;
        entry.reference_text_provided = reference_text_provided;
        entry.sample_rate = audio.sample_rate;
        entry.channels = audio.channels;
        entry.sample_count = sample_count;
        entry.sample_hash = sample_hash;
        entry.tokens = audio_tokenizer_.encode_reference_audio(audio, reference_options);
        reference_prompt_cache_ = std::move(entry);
    }
    return reference_prompt_cache_->tokens;
}

}  // namespace engine::models::omnivoice
