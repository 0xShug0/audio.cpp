#include "engine/models/higgs_audio_tts/generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/higgs_audio_tts/codebooks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_audio_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kInitialGeneratedCacheSteps = 128;
constexpr int64_t kMinimumCacheBucketSteps = 128;

int64_t bucketed_initial_cache_steps(int64_t prompt_steps, int64_t max_tokens) {
    const int64_t maximum = prompt_steps + max_tokens;
    const int64_t required = prompt_steps + std::min(max_tokens, kInitialGeneratedCacheSteps);
    int64_t bucket = kMinimumCacheBucketSteps;
    while (bucket < required && bucket <= maximum / 2) {
        bucket *= 2;
    }
    if (bucket < required) {
        bucket = required;
    }
    return std::min(bucket, maximum);
}

void validate_generation_options(const HiggsGenerationOptions & options) {
    if (options.max_tokens <= 0) {
        throw std::runtime_error("Higgs TTS max_tokens must be positive after default resolution");
    }
    if (!(options.temperature > 0.0F)) {
        throw std::runtime_error("Higgs TTS temperature must be positive");
    }
    if (options.top_p.has_value() && !(*options.top_p > 0.0F)) {
        throw std::runtime_error("Higgs TTS top_p must be positive");
    }
    if (options.top_k.has_value() && *options.top_k < 0) {
        throw std::runtime_error("Higgs TTS top_k must be non-negative");
    }
    if (!(options.repetition_penalty > 0.0F) || !std::isfinite(options.repetition_penalty)) {
        throw std::runtime_error("Higgs TTS repetition_penalty must be finite and positive");
    }
}

size_t flat_index(int64_t frame, int64_t codebook, int64_t codebooks) {
    return static_cast<size_t>(frame * codebooks + codebook);
}

struct HiggsPromptInput {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> reference_positions;
};

struct HiggsPreparedPrompt {
    HiggsPromptInput prompt;
    HiggsARPrefillInput ar_input;
    int64_t prefix_steps = 0;
};

HiggsPromptInput make_prompt_input(const HiggsPromptEncoding & prompt,
                                   int64_t delayed_reference_frames,
                                   const HiggsConfig & config) {
    HiggsPromptInput input;
    input.token_ids = prompt.token_ids;
    input.reference_positions.reserve(static_cast<size_t>(delayed_reference_frames));
    for (int64_t position = 0; position < static_cast<int64_t>(input.token_ids.size());
         ++position) {
        if (input.token_ids[static_cast<size_t>(position)] == config.audio_token_id) {
            input.reference_positions.push_back(static_cast<int32_t>(position));
            input.token_ids[static_cast<size_t>(position)] = 0;
        }
    }
    if (static_cast<int64_t>(input.reference_positions.size()) != delayed_reference_frames) {
        throw std::runtime_error("Higgs TTS prompt audio placeholder count does "
                                 "not match delayed reference codes");
    }
    return input;
}

HiggsPreparedPrompt make_prepared_prompt(const HiggsPromptEncoding & prompt,
                                         const std::vector<int32_t> & delayed_reference_codes,
                                         int64_t delayed_reference_frames,
                                         const HiggsConfig & config) {
    HiggsPreparedPrompt prepared;
    prepared.prompt = make_prompt_input(prompt, delayed_reference_frames, config);
    const int64_t prompt_steps = static_cast<int64_t>(prepared.prompt.token_ids.size());
    prepared.ar_input.steps = prompt_steps;
    prepared.ar_input.text_tokens = prepared.prompt.token_ids;
    prepared.ar_input.fused_code_ids.assign(
        static_cast<size_t>(prompt_steps * config.audio.num_codebooks), 0);
    prepared.ar_input.text_gate.assign(static_cast<size_t>(prompt_steps), 0.0F);
    prepared.ar_input.code_gate.assign(static_cast<size_t>(prompt_steps), 0.0F);
    size_t reference_row = 0;
    for (int64_t position = 0; position < prompt_steps; ++position) {
        if (reference_row < prepared.prompt.reference_positions.size() &&
            prepared.prompt.reference_positions[reference_row] == position) {
            for (int64_t codebook = 0; codebook < config.audio.num_codebooks; ++codebook) {
                const int32_t code = delayed_reference_codes[flat_index(
                    static_cast<int64_t>(reference_row), codebook, config.audio.num_codebooks)];
                if (code < 0 || code >= config.audio.vocab_size) {
                    throw std::runtime_error(
                        "Higgs TTS AR prefill codebook token is outside vocabulary");
                }
                prepared.ar_input
                    .fused_code_ids[flat_index(position, codebook, config.audio.num_codebooks)] =
                    static_cast<int32_t>(code + codebook * config.audio.vocab_size);
            }
            prepared.ar_input.code_gate[static_cast<size_t>(position)] = 1.0F;
            ++reference_row;
        } else {
            const int32_t text_token = prepared.prompt.token_ids[static_cast<size_t>(position)];
            if (text_token < 0 || text_token >= config.text.vocab_size) {
                throw std::runtime_error("Higgs TTS AR prefill text token is outside vocabulary");
            }
            prepared.ar_input.text_gate[static_cast<size_t>(position)] = 1.0F;
        }
    }
    if (reference_row != prepared.prompt.reference_positions.size()) {
        throw std::runtime_error(
            "Higgs TTS prompt prefill did not consume all reference code rows");
    }
    if (!prepared.prompt.reference_positions.empty()) {
        prepared.prefix_steps =
            static_cast<int64_t>(prepared.prompt.reference_positions.back()) + 1;
    }
    return prepared;
}

} // namespace

HiggsGenerator::HiggsGenerator(std::shared_ptr<const HiggsAssets> assets,
                               std::shared_ptr<HiggsARRuntime> ar,
                               std::shared_ptr<HiggsCodecRuntime> codec,
                               size_t ar_decode_graph_arena_bytes)
    : assets_([&]() {
          if (assets == nullptr) {
              throw std::runtime_error("Higgs TTS generator requires assets");
          }
          return std::move(assets);
      }()),
      ar_([&]() {
          if (ar == nullptr) {
              throw std::runtime_error("Higgs TTS generator requires AR runtime");
          }
          return std::move(ar);
      }()),
      codec_([&]() {
          if (codec == nullptr) {
              throw std::runtime_error("Higgs TTS generator requires codec runtime");
          }
          return std::move(codec);
      }()),
      tokenizer_(assets_),
      ar_decode_graph_arena_bytes_(ar_decode_graph_arena_bytes) {
    if (ar_decode_graph_arena_bytes_ == 0) {
        throw std::runtime_error("Higgs TTS generator graph arena bytes must be non-zero");
    }
}

void HiggsGenerator::prepare(const HiggsGenerationRequest & request) {
    const auto & config = assets_->config;
    const bool has_reference = request.reference_frames > 0 || !request.reference_codes.empty();
    if (has_reference) {
        if (request.reference_frames <= 0 ||
            request.reference_codebooks != config.audio.num_codebooks) {
            throw std::runtime_error("Higgs TTS generation requires reference codes "
                                     "shaped [frames, num_codebooks]");
        }
        if (static_cast<int64_t>(request.reference_codes.size()) !=
            request.reference_frames * request.reference_codebooks) {
            throw std::runtime_error("Higgs TTS generation reference code count mismatch");
        }
    } else if (request.reference_codebooks != 0) {
        throw std::runtime_error(
            "Higgs TTS generation got reference codebooks without reference codes");
    }
    validate_generation_options(request.options);
    const int64_t delayed_reference_frames =
        has_reference
            ? higgs_delayed_frame_count(request.reference_frames, request.reference_codebooks)
            : 0;
    std::vector<int32_t> delayed_reference_codes;
    if (has_reference) {
        delayed_reference_codes = apply_higgs_delay_pattern(
            request.reference_codes, request.reference_frames, request.reference_codebooks);
    }
    const HiggsPromptEncoding prompt = tokenizer_.encode_prompt({
        request.text,
        request.reference_text,
        delayed_reference_frames,
    });
    const auto prepared =
        make_prepared_prompt(prompt, delayed_reference_codes, delayed_reference_frames, config);
    const int64_t prompt_steps = prepared.ar_input.steps;
    if (prompt_steps + request.options.max_tokens > config.text.max_position_embeddings) {
        throw std::runtime_error("Higgs TTS generation exceeds text model max_position_embeddings");
    }
    if (has_reference && prepared.prefix_steps > 0) {
        ReferencePrefixCache cache;
        cache.reference_text = request.reference_text;
        cache.reference_codes = request.reference_codes;
        cache.reference_frames = request.reference_frames;
        cache.reference_codebooks = request.reference_codebooks;
        cache.delayed_reference_codes = std::move(delayed_reference_codes);
        cache.delayed_reference_frames = delayed_reference_frames;
        cache.prefix_steps = prepared.prefix_steps;
        cache.prefix_tokens.assign(prepared.prompt.token_ids.begin(),
                                   prepared.prompt.token_ids.begin() +
                                       static_cast<ptrdiff_t>(prepared.prefix_steps));
        const bool same_reference =
            reference_prefix_cache_.has_value() &&
            reference_prefix_cache_->reference_text == cache.reference_text &&
            reference_prefix_cache_->reference_codes == cache.reference_codes &&
            reference_prefix_cache_->reference_frames == cache.reference_frames &&
            reference_prefix_cache_->reference_codebooks == cache.reference_codebooks &&
            reference_prefix_cache_->prefix_steps == cache.prefix_steps &&
            reference_prefix_cache_->prefix_tokens == cache.prefix_tokens;
        reference_prefix_cache_ = std::move(cache);
        if (!same_reference) {
            reference_kv_ready_ = false;
        }
    } else {
        reference_prefix_cache_.reset();
        reference_kv_ready_ = false;
    }
}

HiggsGenerationResult HiggsGenerator::generate(const HiggsGenerationRequest & request) {
    const auto & config = assets_->config;
    const bool has_reference = request.reference_frames > 0 || !request.reference_codes.empty();
    if (has_reference) {
        if (request.reference_frames <= 0 ||
            request.reference_codebooks != config.audio.num_codebooks) {
            throw std::runtime_error("Higgs TTS generation requires reference codes "
                                     "shaped [frames, num_codebooks]");
        }
        if (static_cast<int64_t>(request.reference_codes.size()) !=
            request.reference_frames * request.reference_codebooks) {
            throw std::runtime_error("Higgs TTS generation reference code count mismatch");
        }
    } else if (request.reference_codebooks != 0) {
        throw std::runtime_error(
            "Higgs TTS generation got reference codebooks without reference codes");
    }
    validate_generation_options(request.options);
    engine::debug::trace_log_scalar("higgs_audio_tts.request.text", request.text);
    engine::debug::trace_log_scalar("higgs_audio_tts.request.reference_text", request.reference_text);
    engine::debug::trace_log_scalar("higgs_audio_tts.request.text_chars", request.text.size());
    engine::debug::trace_log_scalar("higgs_audio_tts.request.reference_text_chars",
                                    request.reference_text.size());
    engine::debug::trace_log_scalar("higgs_audio_tts.request.max_tokens",
                                    request.options.max_tokens);
    engine::debug::trace_log_scalar("higgs_audio_tts.request.temperature", request.options.temperature);
    engine::debug::trace_log_scalar("higgs_audio_tts.request.top_p",
                                    request.options.top_p.has_value() ?
                                        std::to_string(*request.options.top_p) : "none");
    engine::debug::trace_log_scalar("higgs_audio_tts.request.top_k",
                                    request.options.top_k.has_value() ?
                                        std::to_string(*request.options.top_k) : "none");
    engine::debug::trace_log_scalar("higgs_audio_tts.request.repetition_penalty",
                                    request.options.repetition_penalty);
    engine::debug::trace_log_scalar("higgs_audio_tts.request.has_seed", request.options.seed.has_value());
    engine::debug::trace_log_scalar("higgs_audio_tts.request.seed",
                                    request.options.seed.has_value() ?
                                        std::to_string(*request.options.seed) : "none");
    if (has_reference) {
        engine::debug::trace_log_i32("higgs_audio_tts.request.reference_codes",
                                     {request.reference_frames, request.reference_codebooks},
                                     request.reference_codes);
    }

    std::vector<int32_t> delayed_reference_codes_storage;
    const std::vector<int32_t> * delayed_reference_codes = &delayed_reference_codes_storage;
    int64_t delayed_reference_frames = 0;
    const ReferencePrefixCache * matching_reference_cache = nullptr;
    if (has_reference) {
        if (reference_prefix_cache_.has_value() &&
            reference_prefix_cache_->reference_text == request.reference_text &&
            reference_prefix_cache_->reference_frames == request.reference_frames &&
            reference_prefix_cache_->reference_codebooks == request.reference_codebooks &&
            reference_prefix_cache_->reference_codes == request.reference_codes) {
            matching_reference_cache = &*reference_prefix_cache_;
            delayed_reference_codes = &matching_reference_cache->delayed_reference_codes;
            delayed_reference_frames = matching_reference_cache->delayed_reference_frames;
        } else {
            delayed_reference_codes_storage = apply_higgs_delay_pattern(
                request.reference_codes, request.reference_frames, request.reference_codebooks);
            delayed_reference_frames =
                higgs_delayed_frame_count(request.reference_frames, request.reference_codebooks);
        }
    }
    const HiggsPromptEncoding prompt = tokenizer_.encode_prompt({
        request.text,
        request.reference_text,
        delayed_reference_frames,
    });
    engine::debug::trace_log_scalar("higgs_audio_tts.prompt.text_tokens", prompt.text_ids.size());
    engine::debug::trace_log_i32("higgs_audio_tts.prompt.text_ids",
                                 {static_cast<int64_t>(prompt.text_ids.size())},
                                 prompt.text_ids);
    engine::debug::trace_log_scalar("higgs_audio_tts.prompt.reference_text_tokens",
                                    prompt.reference_text_ids.size());
    engine::debug::trace_log_i32("higgs_audio_tts.prompt.reference_text_ids",
                                 {static_cast<int64_t>(prompt.reference_text_ids.size())},
                                 prompt.reference_text_ids);
    const auto prepared =
        make_prepared_prompt(prompt, *delayed_reference_codes, delayed_reference_frames, config);
    engine::debug::trace_log_scalar("higgs_audio_tts.prompt.tokens", prepared.prompt.token_ids.size());
    engine::debug::trace_log_i32("higgs_audio_tts.prompt.token_ids",
                                 {static_cast<int64_t>(prepared.prompt.token_ids.size())},
                                 prepared.prompt.token_ids);
    engine::debug::trace_log_scalar("higgs_audio_tts.prompt.delayed_reference_rows",
                                    delayed_reference_frames);
    engine::debug::trace_log_i32("higgs_audio_tts.prompt.delayed_reference_codes",
                                 {delayed_reference_frames, config.audio.num_codebooks},
                                 *delayed_reference_codes);
    engine::debug::trace_log_i32("higgs_audio_tts.ar.prefill.text_tokens",
                                 {prepared.ar_input.steps},
                                 prepared.ar_input.text_tokens);
    engine::debug::trace_log_i32("higgs_audio_tts.ar.prefill.fused_code_ids",
                                 {prepared.ar_input.steps, config.audio.num_codebooks},
                                 prepared.ar_input.fused_code_ids);
    engine::debug::trace_log_f32("higgs_audio_tts.ar.prefill.text_gate",
                                 {prepared.ar_input.steps},
                                 prepared.ar_input.text_gate);
    engine::debug::trace_log_f32("higgs_audio_tts.ar.prefill.code_gate",
                                 {prepared.ar_input.steps},
                                 prepared.ar_input.code_gate);
    const int64_t prompt_steps = prepared.ar_input.steps;
    if (prompt_steps + request.options.max_tokens > config.text.max_position_embeddings) {
        throw std::runtime_error("Higgs TTS generation exceeds text model max_position_embeddings");
    }

    const auto prefill_start = Clock::now();
    const bool reference_cache_hit =
        has_reference && prepared.prefix_steps > 0 && matching_reference_cache != nullptr &&
        matching_reference_cache->prefix_steps == prepared.prefix_steps &&
        static_cast<int64_t>(matching_reference_cache->prefix_tokens.size()) == prepared.prefix_steps &&
        std::equal(matching_reference_cache->prefix_tokens.begin(),
                   matching_reference_cache->prefix_tokens.end(),
                   prepared.prompt.token_ids.begin());
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.reference_prefix_cache_hit", reference_cache_hit);
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.reference_prefix_steps", prepared.prefix_steps);
    const int64_t max_cache_steps = prompt_steps + request.options.max_tokens;
    const int64_t initial_cache_steps = bucketed_initial_cache_steps(prompt_steps, request.options.max_tokens);
    const bool cache_rebuild =
        ar_kv_cache_ == nullptr || !ar_kv_cache_->can_run(*ar_, initial_cache_steps) ||
        ar_kv_cache_->cache_steps() != initial_cache_steps;
    if (cache_rebuild) {
        decode_graph_.reset();
        ar_kv_cache_ = std::make_unique<HiggsARKVCache>(ar_, initial_cache_steps);
        reference_kv_ready_ = false;
    }
    const bool reference_kv_cache_hit =
        reference_cache_hit && reference_kv_ready_ &&
        ar_kv_cache_->valid_steps() >= prepared.prefix_steps;
    const int64_t prefill_start_step = reference_kv_cache_hit ? prepared.prefix_steps : 0;
    if (reference_kv_cache_hit) {
        ar_kv_cache_->retain_prefix(prefill_start_step);
    } else {
        ar_kv_cache_->reset();
    }
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.reference_kv_cache_hit", reference_kv_cache_hit);
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.prefill_start_step", prefill_start_step);
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.prefill_run_steps", prompt_steps - prefill_start_step);
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.kv_cache_steps", initial_cache_steps);
    engine::debug::trace_log_scalar("higgs_audio_tts.generator.kv_cache_rebuild", cache_rebuild);
    if (prefill_graph_ == nullptr ||
        !prefill_graph_->matches(*ar_, prompt_steps, prefill_start_step)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<HiggsARPrefillGraph>(
            ar_, prompt_steps, prefill_start_step, ar_kv_cache_.get(), ar_decode_graph_arena_bytes_);
    }
    auto prefill_output = prefill_graph_->run(prepared.ar_input, prefill_start_step);
    prefill_graph_.reset();
    reference_kv_ready_ = reference_cache_hit;

    if (decode_graph_ == nullptr || !decode_graph_->can_run(*ar_, ar_kv_cache_->cache_steps())) {
        decode_graph_ = std::make_unique<HiggsARDecodeGraph>(
            ar_, ar_kv_cache_->cache_steps(), *ar_kv_cache_, ar_decode_graph_arena_bytes_);
    }
    if (!prefill_output.wrote_cache) {
        decode_graph_->import_prefill_state(prefill_output.kv_state);
    }
    HiggsARDecodeOutput prefill = std::move(prefill_output.output);
    engine::debug::timing_log_scalar("higgs_audio_tts.generator.prefill_ms",
                                     engine::debug::elapsed_ms(prefill_start, Clock::now()));
    engine::debug::trace_log_f32("higgs_audio_tts.sampler.prefill_logits",
                                 {config.audio.num_codebooks, config.audio.vocab_size},
                                 prefill.codebook_logits);

    HiggsCodebookSampler sampler(config.audio.num_codebooks, config.audio.vocab_size);
    HiggsSamplerState state = sampler.make_state();
    HiggsSamplingOptions sampling;
    sampling.temperature = request.options.temperature;
    sampling.top_p = request.options.top_p;
    sampling.top_k = request.options.top_k;
    // Python accepts repetition_penalty on the public speech request, but the
    // Higgs audio-codebook sampler does not consume it.
    sampling.has_seed = request.options.seed.has_value();
    sampling.seed = request.options.seed.value_or(runtime::random_u64_seed());
    std::mt19937 fallback_rng(static_cast<uint32_t>(sampling.seed));
    sampling.fallback_rng = &fallback_rng;
    if (!sampling.has_seed && !cuda_sampling_policy_.has_value()) {
        cuda_sampling_policy_ = engine::sampling::resolve_torch_cuda_sampling_policy(
            ar_->backend_type(),
            ar_->device(),
            "higgs_audio_tts.cuda_sampling_policy",
            "Higgs TTS",
            engine::sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault);
    }
    if (cuda_sampling_policy_.has_value()) {
        sampling.cuda_policy = *cuda_sampling_policy_;
    }
    engine::debug::trace_log_scalar("higgs_audio_tts.sampler.temperature", sampling.temperature);
    engine::debug::trace_log_scalar("higgs_audio_tts.sampler.has_seed", sampling.has_seed);
    engine::debug::trace_log_scalar("higgs_audio_tts.sampler.seed", sampling.seed);
    engine::debug::trace_log_scalar("higgs_audio_tts.sampler.top_p",
                                    sampling.top_p.has_value() ? std::to_string(*sampling.top_p)
                                                               : "none");
    engine::debug::trace_log_scalar("higgs_audio_tts.sampler.top_k",
                                    sampling.top_k.has_value() ? std::to_string(*sampling.top_k)
                                                               : "none");
    HiggsGenerationResult result;
    result.delayed_codes.reserve(
        static_cast<size_t>(request.options.max_tokens * config.audio.num_codebooks));
    const auto & first_sampled = sampler.step(prefill.codebook_logits.data(),
                                              static_cast<int64_t>(prefill.codebook_logits.size()),
                                              state,
                                              sampling);
    engine::debug::trace_log_i32("higgs_audio_tts.sampler.first_output_codes",
                                 {static_cast<int64_t>(first_sampled.size())},
                                 first_sampled);
    result.delayed_codes.insert(
        result.delayed_codes.end(), first_sampled.begin(), first_sampled.end());
    result.delayed_frames += 1;

    const auto decode_start = Clock::now();
    HiggsARDecodeOutput decoded;
    decoded.codebook_logits.reserve(
        static_cast<size_t>(config.audio.num_codebooks * config.audio.vocab_size));
    bool logged_decode_step_timing = false;
    double sampler_total_ms = 0.0;
    HiggsARDecodeTiming decode_timing_total;
    decode_graph_->begin_decode_run();
    while (!state.generation_done && result.delayed_frames < request.options.max_tokens) {
        if (ar_kv_cache_->valid_steps() >= ar_kv_cache_->cache_steps()) {
            decode_timing_total.add(decode_graph_->timing());
            const auto kv_state = ar_kv_cache_->export_state();
            const int64_t grown_cache_steps =
                std::min(max_cache_steps,
                         std::max(ar_kv_cache_->cache_steps() * 2, ar_kv_cache_->valid_steps() + 1));
            if (grown_cache_steps <= ar_kv_cache_->cache_steps()) {
                throw std::runtime_error("Higgs TTS AR cache cannot grow");
            }
            decode_graph_.reset();
            ar_kv_cache_ = std::make_unique<HiggsARKVCache>(ar_, grown_cache_steps);
            ar_kv_cache_->import_state(kv_state);
            engine::debug::trace_log_scalar("higgs_audio_tts.generator.kv_cache_grown_steps", grown_cache_steps);
            decode_graph_ = std::make_unique<HiggsARDecodeGraph>(
                ar_, ar_kv_cache_->cache_steps(), *ar_kv_cache_, ar_decode_graph_arena_bytes_);
            decode_graph_->begin_decode_run();
        }
        HiggsARDecodeInput input;
        input.use_last_codes = state.delay_count > 0;
        input.last_codes = state.last_codes;
        const auto step_start = Clock::now();
        decode_graph_->run_step_into(input, decoded, !logged_decode_step_timing);
        if (!logged_decode_step_timing) {
            engine::debug::timing_log_scalar("higgs_audio_tts.generator.decode.step0.ar_ms",
                                             engine::debug::elapsed_ms(step_start, Clock::now()));
        }
        const auto sample_start = Clock::now();
        const auto & sampled = sampler.step(decoded.codebook_logits.data(),
                                            static_cast<int64_t>(decoded.codebook_logits.size()),
                                            state,
                                            sampling);
        sampler_total_ms += engine::debug::elapsed_ms(sample_start, Clock::now());
        if (!logged_decode_step_timing) {
            engine::debug::timing_log_scalar("higgs_audio_tts.generator.decode.step0.sampler_ms",
                                             sampler_total_ms);
            logged_decode_step_timing = true;
        }
        if (!sampled.empty() && sampled.front() != kHiggsStopCode) {
            result.delayed_codes.insert(result.delayed_codes.end(), sampled.begin(), sampled.end());
            result.delayed_frames += 1;
        }
    }
    decode_timing_total.add(decode_graph_->timing());
    engine::debug::timing_log_scalar("higgs_audio_tts.ar.decode.steps", decode_timing_total.steps);
    engine::debug::timing_log_scalar("higgs_audio_tts.ar.decode.input_upload_ms", decode_timing_total.input_upload_ms);
    engine::debug::timing_log_scalar("higgs_audio_tts.ar.decode.mask_upload_ms", decode_timing_total.mask_upload_ms);
    engine::debug::timing_log_scalar("higgs_audio_tts.ar.decode.graph.compute_ms", decode_timing_total.graph_compute_ms);
    engine::debug::timing_log_scalar("higgs_audio_tts.ar.decode.output_read_ms", decode_timing_total.output_read_ms);
    engine::debug::timing_log_scalar("higgs_audio_tts.generator.decode.sampler_ms", sampler_total_ms);
    engine::debug::timing_log_scalar("higgs_audio_tts.generator.decode_ms",
                                     engine::debug::elapsed_ms(decode_start, Clock::now()));
    if (!state.generation_done) {
        throw std::runtime_error("Higgs TTS generation reached max_tokens before EOC");
    }

    result.raw_codes = reverse_higgs_delay_pattern(
        result.delayed_codes, result.delayed_frames, config.audio.num_codebooks);
    result.raw_frames = result.delayed_frames - (config.audio.num_codebooks - 1);
    engine::debug::trace_log_i32("higgs_audio_tts.generator.delayed_codes",
                                 {result.delayed_frames, config.audio.num_codebooks},
                                 result.delayed_codes);
    const int64_t delayed_head_rows = std::min<int64_t>(result.delayed_frames, 8);
    engine::debug::trace_log_i32("higgs_audio_tts.generator.delayed_codes_head8",
                                 {delayed_head_rows, config.audio.num_codebooks},
                                 std::vector<int32_t>(
                                     result.delayed_codes.begin(),
                                     result.delayed_codes.begin() +
                                         static_cast<std::ptrdiff_t>(
                                             delayed_head_rows * config.audio.num_codebooks)));
    const int32_t codec_vocab = static_cast<int32_t>(config.audio.vocab_size - 2);
#ifdef _OPENMP
#pragma omp parallel for if (static_cast<int64_t>(result.raw_codes.size()) > 1024)
#endif
    for (int64_t index = 0; index < static_cast<int64_t>(result.raw_codes.size()); ++index) {
        int32_t & code = result.raw_codes[static_cast<size_t>(index)];
        if (code >= codec_vocab) {
            code = 0;
        }
    }
    engine::debug::trace_log_i32("higgs_audio_tts.generator.raw_codes_for_codec",
                                 {result.raw_frames, config.audio.num_codebooks},
                                 result.raw_codes);
    const auto codec_start = Clock::now();
    result.audio =
        codec_->decode_codes(result.raw_codes, result.raw_frames, config.audio.num_codebooks);
    engine::debug::trace_log_f32("higgs_audio_tts.codec.decode.output_audio",
                                 {result.audio.samples},
                                 result.audio.values);
    engine::debug::timing_log_scalar("higgs_audio_tts.generator.codec_decode_ms",
                                     engine::debug::elapsed_ms(codec_start, Clock::now()));
    codec_->release_runtime_graphs();
    return result;
}

} // namespace engine::models::higgs_audio_tts
