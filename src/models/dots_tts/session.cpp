#include "engine/models/dots_tts/session.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"
#include "engine/framework/io/text.h"
#include "engine/models/dots_tts/audio_features.h"
#include "engine/framework/core/backend.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "dots_tts";
constexpr int64_t kDefaultMaxSequenceLength = 2048;
constexpr float kEosThreshold = 0.8F;
constexpr int64_t kInitialUnmergedVocoderPatches = 2;
constexpr size_t kDefaultPromptFeatureCacheSlots = 4;

std::size_t reference_cache_slots_from_options(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"dots_tts.reference_cache_slots"})
        .value_or(static_cast<int64_t>(kDefaultPromptFeatureCacheSlots));
    if (slots < 0) {
        throw std::runtime_error("dots_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<uint64_t>(slots) > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("dots_tts.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

uint64_t mix_prompt_feature_key(uint64_t key, uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
    return key;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key = mix_prompt_feature_key(key, bits);
    }
    return key;
}

uint32_t optional_float_bits(std::optional<float> value) {
    if (!value.has_value()) {
        return 0;
    }
    uint32_t bits = 0;
    const float v = *value;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

std::shared_ptr<const DotsAssets> require_assets(std::shared_ptr<const DotsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("DotTTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("DotTTS session requires a model contract");
    }
    return contract;
}

std::string normalized_language_tag(const std::string & language) {
    std::string stripped = engine::io::trim_ascii_whitespace(language);
    std::transform(stripped.begin(), stripped.end(), stripped.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    if (stripped.empty() || stripped == "NONE" || stripped == "UNKNOWN") {
        return {};
    }
    if (stripped == "AUTO_DETECT") {
        throw std::runtime_error("DotTTS language=auto_detect requires the Python language detector and is not supported by the native session");
    }
    return stripped;
}

std::string attach_language_tag(std::string text, const std::string & language) {
    const auto tag = normalized_language_tag(language);
    if (tag.empty() || text.empty()) {
        return text;
    }
    const std::string prefix = "[" + tag + "]";
    if (text.rfind(prefix, 0) == 0) {
        return text;
    }
    return prefix + text;
}

std::vector<int64_t> audio_span_positions(const DotsGenerationSchedule & schedule, int32_t span_id) {
    std::vector<int64_t> out;
    for (size_t i = 0; i < schedule.token_ids.size(); ++i) {
        if (schedule.token_ids[i] == span_id) {
            out.push_back(static_cast<int64_t>(i));
        }
    }
    return out;
}

bool next_token_is_audio_span(const DotsGenerationSchedule & schedule, int64_t position, int32_t span_id) {
    const int64_t next = position + 1;
    return next >= 0 &&
        next < static_cast<int64_t>(schedule.token_ids.size()) &&
        schedule.token_ids[static_cast<size_t>(next)] == span_id;
}

std::vector<float> slice_row(
    const std::vector<float> & values,
    int64_t row,
    int64_t width) {
    if (row < 0 || width <= 0 || static_cast<int64_t>(values.size()) < (row + 1) * width) {
        throw std::runtime_error("DotTTS row slice is out of range");
    }
    return std::vector<float>(
        values.begin() + static_cast<std::ptrdiff_t>(row * width),
        values.begin() + static_cast<std::ptrdiff_t>((row + 1) * width));
}

DotsLatentMatrix slice_patch(const DotsLatentMatrix & latents, int64_t patch_index, int64_t patch_size) {
    if (patch_index < 0 || patch_size <= 0 ||
        latents.dims <= 0 ||
        latents.frames < (patch_index + 1) * patch_size ||
        static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dims) {
        throw std::runtime_error("DotTTS latent patch slice is out of range");
    }
    DotsLatentMatrix out;
    out.frames = patch_size;
    out.dims = latents.dims;
    out.values.assign(
        latents.values.begin() + static_cast<std::ptrdiff_t>(patch_index * patch_size * latents.dims),
        latents.values.begin() + static_cast<std::ptrdiff_t>((patch_index + 1) * patch_size * latents.dims));
    return out;
}

void append_projected(std::vector<float> & dst, const DotsProjectedSequence & projected) {
    auto values = projected.values;
    engine::core::round_f32_to_bf16_in_place(values);
    dst.insert(dst.end(), values.begin(), values.end());
}

}  // namespace

struct DotsSession::PromptConditioning {
    std::vector<float> speaker_condition;
    DotsLatentMatrix prompt_latents;
    DotsLatentMatrix prompt_patches;
    std::string prompt_text;
    int64_t prompt_patch_count = 0;
    uint64_t rng_offset_blocks = 0;
};

struct DotsSession::SegmentState {
    DotsLlmState llm;
    DotsPatchEncoderState patch;
    DotsFlowDecodeState flow;
    std::vector<float> fm_sequence;
    std::vector<float> fm_cfg_sequence;
    std::vector<float> null_hidden_projected;
    DotsLlmHidden llm_hiddens;
    int64_t fm_seq_len = 0;
    uint64_t rng_offset_blocks = 0;
    bool end_flag = false;
};

bool DotsSession::PromptFeatureCacheKeyEqual::operator()(
    const PromptFeatureCacheKey & lhs,
    const PromptFeatureCacheKey & rhs) const noexcept {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash &&
        lhs.has_duration == rhs.has_duration &&
        lhs.duration_bits == rhs.duration_bits;
}

DotsSession::DotsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const DotsAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(assets_),
      latent_codec_(assets_),
      prompt_feature_cache_(reference_cache_slots_from_options(options)) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "DotTTS");
    using T = engine::assets::TensorStorageType;
    weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.weight_type",
        weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    speaker_encoder_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.speaker_encoder_weight_type",
        weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    vocoder_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.codec_weight_type",
        weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    patch_encoder_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.patch_encoder_weight_type",
        weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    llm_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.llm_weight_type",
        weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    flow_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.flow_weight_type",
        weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    conv_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "dots_tts.codec_conv_weight_type",
        conv_weight_storage_type_,
        {T::Native, T::F32, T::F16});
    if (task_.task != runtime::VoiceTaskKind::Tts && task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("DotTTS supports TTS and voice cloning tasks");
    }
    if (const auto value = runtime::find_option(options.options, {"dots_tts.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*value, "dots_tts.mem_saver");
    }
    if (!mem_saver_) {
        ensure_speaker_encoder_loaded();
        ensure_audio_vae_loaded();
        ensure_patch_encoder_loaded();
        ensure_llm_loaded();
        ensure_flow_loaded();
    }
}

DotsSession::~DotsSession() = default;

std::string DotsSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind DotsSession::task_kind() const {
    return task_.task;
}

runtime::RunMode DotsSession::run_mode() const {
    return task_.mode;
}

void DotsSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "DotTTS");
    prepared_defaults_ = make_dots_prepare_defaults(*assets_, request);
    mark_prepared();
}

DotsSession::PromptConditioning DotsSession::prepare_prompt_conditioning(const DotsRequest & request) {
    PromptConditioning out;
    if (!request.reference.audio.has_value()) {
        out.speaker_condition.assign(static_cast<size_t>(assets_->config.dit.hidden_size), 0.0F);
        return out;
    }
    PromptFeatureCacheKey key;
    key.sample_rate = request.reference.audio->sample_rate;
    key.channels = request.reference.audio->channels;
    key.sample_count = static_cast<uint64_t>(request.reference.audio->samples.size());
    key.sample_hash = hash_audio_samples(*request.reference.audio);
    key.has_duration = request.reference.duration_seconds.has_value();
    key.duration_bits = optional_float_bits(request.reference.duration_seconds);

    const PromptFeatureCacheEntry * features = prompt_feature_cache_.find(key);
    if (features == nullptr) {
        ensure_speaker_encoder_loaded();
        ensure_audio_vae_loaded();
        const auto prepared = prepare_dots_reference_audio(
            *request.reference.audio,
            assets_->config.vocoder.sample_rate,
            assets_->config.patch_size * audio_vae_.hop_size(),
            request.reference.duration_seconds);
        auto speaker = speaker_encoder_.embed_from_features(
            prepared.speaker_fbank.values,
            prepared.speaker_fbank.frames,
            prepared.speaker_fbank.dims);
        PromptFeatureCacheEntry entry;
        entry.speaker_embedding = std::move(speaker.embedding);
        if (!request.reference.reference_text.empty()) {
            entry.encoded_latents = audio_vae_.extract_latents(prepared.waveform_vocoder_rate);
        }
        prompt_feature_cache_.put(key, std::move(entry));
        features = prompt_feature_cache_.find(key);
        if (features == nullptr) {
            throw std::runtime_error("DotTTS prompt feature cache insert failed");
        }
        debug::trace_log_scalar("dots_tts.prompt_feature_cache.hit", 0);
    } else {
        debug::trace_log_scalar("dots_tts.prompt_feature_cache.hit", 1);
    }
    auto speaker_embedding = features->speaker_embedding;
    for (float & value : speaker_embedding) {
        value *= request.generation.speaker_scale;
    }
    ensure_flow_loaded();
    out.speaker_condition = flow_.project_speaker(speaker_embedding).values;
    engine::core::round_f32_to_bf16_in_place(out.speaker_condition);
    if (request.reference.reference_text.empty()) {
        return out;
    }
    const auto & encoded = features->encoded_latents;
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_context().backend_type(),
        execution_context().config().device,
        "dots_tts.prompt.rng",
        "DotTTS",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    auto sampled = latent_codec_.sample_from_encoder_latents(
        encoded.values,
        encoded.channels,
        encoded.frames,
        request.generation.seed,
        0,
        rng_policy);
    out.rng_offset_blocks = engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(assets_->config.latent_dim * encoded.frames),
        rng_policy);
    if (sampled.frames <= assets_->config.patch_size) {
        return out;
    }
    sampled.frames -= assets_->config.patch_size;
    sampled.values.resize(static_cast<size_t>(sampled.frames * sampled.dims));
    out.prompt_latents = std::move(sampled);
    out.prompt_patches = latent_codec_.normalize(out.prompt_latents);
    out.prompt_patch_count = out.prompt_patches.frames / assets_->config.patch_size;
    out.prompt_text = request.reference.reference_text;
    return out;
}

std::vector<DotsLatentMatrix> DotsSession::generate_latent_patches(
    const DotsRequest & request,
    const std::string & text,
    PromptConditioning & conditioning,
    const std::function<void(const DotsLatentMatrix &, int64_t)> & on_payload_patch) {
    ensure_llm_loaded();
    ensure_patch_encoder_loaded();
    ensure_flow_loaded();
    std::string normalized_text = engine::io::trim_ascii_whitespace(text);
    std::string prompt_text = engine::io::trim_ascii_whitespace(conditioning.prompt_text);
    if (!prompt_text.empty()) {
        prompt_text += "\n";
        prompt_text = attach_language_tag(prompt_text, request.generation.language);
    } else {
        normalized_text = attach_language_tag(normalized_text, request.generation.language);
    }
    auto schedule = tokenizer_.build_generation_schedule(
        prompt_text + normalized_text,
        request.generation.template_name,
        request.generation.max_tokens);
    debug::trace_log_scalar("dots_tts.schedule.length", static_cast<int64_t>(schedule.token_ids.size()));
    if (static_cast<int64_t>(schedule.token_ids.size()) > kDefaultMaxSequenceLength) {
        throw std::runtime_error("DotTTS generation schedule exceeds max_sequence_length");
    }
    const auto spans = audio_span_positions(schedule, tokenizer_.audio_gen_span_id());
    debug::trace_log_scalar("dots_tts.schedule.audio_spans", static_cast<int64_t>(spans.size()));
    debug::trace_log_scalar("dots_tts.prompt.patch_count", conditioning.prompt_patch_count);
    if (spans.empty()) {
        throw std::runtime_error("DotTTS generation schedule contains no audio spans");
    }
    if (conditioning.prompt_patch_count > 0 &&
        static_cast<int64_t>(spans.size()) <= conditioning.prompt_patch_count) {
        throw std::runtime_error("DotTTS generation schedule is too short for prompt prefill");
    }

    SegmentState state;
    state.llm = llm_.create_state(kDefaultMaxSequenceLength);
    state.patch = patch_encoder_.create_state(static_cast<int64_t>(spans.size()));
    state.rng_offset_blocks = conditioning.rng_offset_blocks;
    state.null_hidden_projected = flow_.project_llm_hidden(
        std::vector<float>(static_cast<size_t>(assets_->config.llm.hidden_size), 0.0F),
        1).values;
    engine::core::round_f32_to_bf16_in_place(state.null_hidden_projected);
    double flow_project_ms = 0.0;
    double flow_decode_ms = 0.0;
    DotsFlowRuntimeStats flow_runtime_stats;
    double patch_encoder_ms = 0.0;
    double llm_embed_ms = 0.0;
    double llm_prefill_ms = 0.0;
    double llm_decode_ms = 0.0;
    double eos_ms = 0.0;
    int64_t llm_state_version = 0;
    DotsPatchEmbeddings prompt_embeddings;
    if (conditioning.prompt_patch_count > 0) {
        const auto patch_prefill_start = Clock::now();
        prompt_embeddings = patch_encoder_.prefill(conditioning.prompt_latents.values, conditioning.prompt_latents.frames, state.patch);
        engine::core::round_f32_to_bf16_in_place(prompt_embeddings.values);
        patch_encoder_ms += engine::debug::elapsed_ms(patch_prefill_start, Clock::now());
    }

    const int64_t prefill_end = spans[static_cast<size_t>(conditioning.prompt_patch_count)];
    debug::trace_log_scalar("dots_tts.prefill.end", prefill_end);
    std::vector<int32_t> prompt_span_positions;
    prompt_span_positions.reserve(static_cast<size_t>(conditioning.prompt_patch_count));
    for (int64_t i = 0; i < conditioning.prompt_patch_count; ++i) {
        prompt_span_positions.push_back(static_cast<int32_t>(spans[static_cast<size_t>(i)]));
    }
    std::vector<int32_t> prefill_ids(
        schedule.token_ids.begin(),
        schedule.token_ids.begin() + static_cast<std::ptrdiff_t>(prefill_end));
    if (!prefill_ids.empty()) {
        const auto embed_start = Clock::now();
        auto embeddings = llm_.embed_tokens(prefill_ids);
        engine::core::round_f32_to_bf16_in_place(embeddings);
        llm_embed_ms += engine::debug::elapsed_ms(embed_start, Clock::now());
        for (int64_t prompt_index = 0; prompt_index < conditioning.prompt_patch_count; ++prompt_index) {
            const int64_t span_pos = spans[static_cast<size_t>(prompt_index)];
            const int64_t hidden = assets_->config.llm.hidden_size;
            std::copy(
                prompt_embeddings.values.begin() + static_cast<std::ptrdiff_t>(prompt_index * hidden),
                prompt_embeddings.values.begin() + static_cast<std::ptrdiff_t>((prompt_index + 1) * hidden),
                embeddings.begin() + static_cast<std::ptrdiff_t>(span_pos * hidden));
        }
        engine::core::round_f32_to_bf16_in_place(embeddings);
        const auto prefill_start = Clock::now();
        state.llm_hiddens = llm_.prefill_embeddings(embeddings, prefill_end, state.llm);
        llm_prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
        ++llm_state_version;
    }

    auto append_hidden = [&](const std::vector<float> & hidden) {
        const auto start = Clock::now();
        append_projected(state.fm_sequence, flow_.project_llm_hidden(hidden, 1));
        flow_project_ms += engine::debug::elapsed_ms(start, Clock::now());
        state.fm_cfg_sequence.insert(state.fm_cfg_sequence.end(), state.null_hidden_projected.begin(), state.null_hidden_projected.end());
        ++state.fm_seq_len;
    };
    auto append_history = [&](const DotsLatentMatrix & patch) {
        const auto start = Clock::now();
        auto projected = flow_.project_latents(patch.values, patch.frames);
        append_projected(state.fm_sequence, projected);
        append_projected(state.fm_cfg_sequence, projected);
        flow_project_ms += engine::debug::elapsed_ms(start, Clock::now());
        state.fm_seq_len += patch.frames;
    };

    int64_t cursor = 0;
    for (int64_t prompt_index = 0; prompt_index < conditioning.prompt_patch_count; ++prompt_index) {
        const int64_t span_pos = spans[static_cast<size_t>(prompt_index)];
        if (span_pos > cursor) {
            append_hidden(slice_row(state.llm_hiddens.values, span_pos - 1, state.llm_hiddens.hidden_size));
        }
        append_history(slice_patch(conditioning.prompt_patches, prompt_index, assets_->config.patch_size));
        if (next_token_is_audio_span(schedule, span_pos, tokenizer_.audio_gen_span_id())) {
            append_hidden(slice_row(state.llm_hiddens.values, span_pos, state.llm_hiddens.hidden_size));
        }
        cursor = span_pos + 1;
    }
    if (prefill_end > cursor) {
        append_hidden(slice_row(state.llm_hiddens.values, prefill_end - 1, state.llm_hiddens.hidden_size));
    }

    std::vector<DotsLatentMatrix> payload;
    int64_t payload_count = 0;
    float last_post_eos_probability = 0.0F;
    int64_t decoded_audio_count = 0;
    bool cached_eos_valid = false;
    int64_t cached_eos_version = -1;
    float cached_eos_probability = 0.0F;
    bool drop_regenerated_prompt_tail = conditioning.prompt_patch_count > 0;
    int64_t position = prefill_end;
    size_t span_cursor = static_cast<size_t>(std::lower_bound(spans.begin(), spans.end(), position) - spans.begin());
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_context().backend_type(),
        execution_context().config().device,
        "dots_tts.flow.rng",
        "DotTTS",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    while (position < static_cast<int64_t>(schedule.token_ids.size())) {
        const int32_t token_id = schedule.token_ids[static_cast<size_t>(position)];
        if (token_id == tokenizer_.audio_gen_span_id()) {
            float eos_probability = 0.0F;
            if (!drop_regenerated_prompt_tail && state.llm_hiddens.hidden_size > 0) {
                if (cached_eos_valid && cached_eos_version == llm_state_version) {
                    eos_probability = cached_eos_probability;
                } else {
                    const auto eos_start = Clock::now();
                    eos_probability = llm_.eos_probability(slice_row(
                        state.llm_hiddens.values,
                        state.llm_hiddens.steps - 1,
                        state.llm_hiddens.hidden_size));
                    eos_ms += engine::debug::elapsed_ms(eos_start, Clock::now());
                    cached_eos_valid = true;
                    cached_eos_version = llm_state_version;
                    cached_eos_probability = eos_probability;
                }
            }
            const bool stop_after_current = !drop_regenerated_prompt_tail &&
                state.llm_hiddens.hidden_size > 0 &&
                eos_probability > kEosThreshold;
            DotsFlowDecodeRequest flow_request;
            flow_request.sequence = &state.fm_sequence;
            flow_request.cfg_sequence = &state.fm_cfg_sequence;
            flow_request.speaker_condition = &conditioning.speaker_condition;
            flow_request.fm_seq_len = state.fm_seq_len;
            flow_request.num_inference_steps = request.generation.num_inference_steps;
            flow_request.ode_method = request.generation.ode_method;
            flow_request.guidance_scale = request.generation.guidance_scale;
            flow_request.seed = request.generation.seed;
            flow_request.rng_offset_blocks = state.rng_offset_blocks;
            flow_request.cache_capacity = static_cast<int64_t>(spans.size()) * (assets_->config.patch_size + 1);
            flow_request.runtime_stats = &flow_runtime_stats;
            flow_request.decode_state = &state.flow;
            flow_request.use_cached_dit = true;
            const auto flow_start = Clock::now();
            auto audio_patch = assets_->config.meanflow.has_value() && assets_->config.meanflow->enabled
                ? flow_.decode_next_meanflow(flow_request)
                : flow_.decode_next_soar(flow_request);
            flow_decode_ms += engine::debug::elapsed_ms(flow_start, Clock::now());
            ++decoded_audio_count;
            state.rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(audio_patch.values.size()),
                rng_policy);
            append_history(audio_patch);
            auto patch_encoder_input = latent_codec_.denormalize(audio_patch);
            const auto patch_start = Clock::now();
            auto llm_embedding = patch_encoder_.decode_patch(patch_encoder_input.values, state.patch);
            patch_encoder_ms += engine::debug::elapsed_ms(patch_start, Clock::now());
            engine::core::round_f32_to_bf16_in_place(llm_embedding.values);
            const auto llm_start = Clock::now();
            state.llm_hiddens = llm_.decode_embedding(llm_embedding.values, state.llm);
            llm_decode_ms += engine::debug::elapsed_ms(llm_start, Clock::now());
            ++llm_state_version;
            cached_eos_valid = false;
            float post_eos_probability = 0.0F;
            if (!drop_regenerated_prompt_tail && state.llm_hiddens.hidden_size > 0) {
                const auto eos_start = Clock::now();
                post_eos_probability = llm_.eos_probability(slice_row(
                    state.llm_hiddens.values,
                    state.llm_hiddens.steps - 1,
                    state.llm_hiddens.hidden_size));
                eos_ms += engine::debug::elapsed_ms(eos_start, Clock::now());
                last_post_eos_probability = post_eos_probability;
                cached_eos_valid = true;
                cached_eos_version = llm_state_version;
                cached_eos_probability = post_eos_probability;
            }
            if (next_token_is_audio_span(schedule, position, tokenizer_.audio_gen_span_id())) {
                append_hidden(slice_row(
                    state.llm_hiddens.values,
                    state.llm_hiddens.steps - 1,
                    state.llm_hiddens.hidden_size));
            }
            ++position;
            ++span_cursor;
            if (drop_regenerated_prompt_tail) {
                drop_regenerated_prompt_tail = false;
            } else {
                DotsLatentMatrix payload_patch = latent_codec_.denormalize(audio_patch);
                ++payload_count;
                if (on_payload_patch) {
                    on_payload_patch(payload_patch, payload_count);
                } else {
                    payload.push_back(std::move(payload_patch));
                }
            }
            if (stop_after_current) {
                break;
            }
            continue;
        }
        const int64_t next_audio_pos = span_cursor < spans.size()
            ? spans[span_cursor]
            : static_cast<int64_t>(schedule.token_ids.size());
        while (position < next_audio_pos) {
            const std::vector<int32_t> token = {schedule.token_ids[static_cast<size_t>(position)]};
            const auto embed_start = Clock::now();
            auto embedding = llm_.embed_tokens(token);
            llm_embed_ms += engine::debug::elapsed_ms(embed_start, Clock::now());
            const auto llm_start = Clock::now();
            state.llm_hiddens = llm_.decode_embedding(embedding, state.llm);
            llm_decode_ms += engine::debug::elapsed_ms(llm_start, Clock::now());
            ++llm_state_version;
            cached_eos_valid = false;
            ++position;
        }
        if (position > 0) {
            append_hidden(slice_row(
                state.llm_hiddens.values,
                state.llm_hiddens.steps - 1,
                state.llm_hiddens.hidden_size));
        }
    }
    if (payload_count == 0) {
        throw std::runtime_error("DotTTS generation produced no decodable latents");
    }
    debug::trace_log_scalar("dots_tts.decode.last_post_eos_probability", last_post_eos_probability);
    debug::trace_log_scalar("dots_tts.decode.payload_patches", payload_count);
    debug::timing_log_scalar("dots_tts.decode.flow_project_ms", flow_project_ms);
    debug::timing_log_scalar("dots_tts.decode.flow_ms", flow_decode_ms);
    debug::trace_log_scalar("dots_tts.decode.flow_velocity_calls", flow_runtime_stats.velocity_calls);
    debug::timing_log_scalar("dots_tts.decode.flow.modulation_graph_ms", flow_runtime_stats.modulation_graph_ms);
    debug::timing_log_scalar("dots_tts.decode.flow.modulation_upload_ms", flow_runtime_stats.modulation_input_upload_ms);
    debug::timing_log_scalar("dots_tts.decode.flow.modulation_compute_ms", flow_runtime_stats.modulation_compute_ms);
    debug::timing_log_scalar("dots_tts.decode.flow.velocity_graph_ms", flow_runtime_stats.velocity_graph_ms);
    debug::timing_log_scalar("dots_tts.decode.flow.velocity_upload_ms", flow_runtime_stats.velocity_input_upload_ms);
    debug::timing_log_scalar("dots_tts.decode.flow.velocity_compute_ms", flow_runtime_stats.velocity_compute_ms);
    debug::timing_log_scalar("dots_tts.decode.flow.velocity_read_ms", flow_runtime_stats.velocity_output_read_ms);
    debug::timing_log_scalar("dots_tts.decode.patch_encoder_ms", patch_encoder_ms);
    debug::timing_log_scalar("dots_tts.decode.llm_embed_ms", llm_embed_ms);
    debug::timing_log_scalar("dots_tts.decode.llm_prefill_ms", llm_prefill_ms);
    debug::timing_log_scalar("dots_tts.decode.llm_decode_ms", llm_decode_ms);
    debug::timing_log_scalar("dots_tts.decode.eos_ms", eos_ms);
    return payload;
}

runtime::AudioBuffer DotsSession::synthesize_segment(const DotsRequest & request, const std::string & text) {
    const auto conditioning_start = Clock::now();
    auto conditioning = prepare_prompt_conditioning(request);
    debug::timing_log_scalar("dots_tts.segment.conditioning_ms", engine::debug::elapsed_ms(conditioning_start, Clock::now()));
    release_conditioning_phase_components();
    auto patches = generate_latent_patches(request, text, conditioning);
    release_generation_phase_components();
    DotsLatentMatrix joined;
    joined.frames = 0;
    joined.dims = assets_->config.latent_dim;
    for (const auto & patch : patches) {
        joined.values.insert(joined.values.end(), patch.values.begin(), patch.values.end());
        joined.frames += patch.frames;
    }
    const auto vocoder_start = Clock::now();
    ensure_audio_vae_loaded();
    auto audio = audio_vae_.decode_latents(joined.values, joined.frames);
    debug::timing_log_scalar("dots_tts.segment.vocoder_ms", engine::debug::elapsed_ms(vocoder_start, Clock::now()));
    release_audio_phase_components();
    return {static_cast<int>(audio.sample_rate), 1, std::move(audio.samples)};
}

runtime::AudioBuffer DotsSession::synthesize_streaming_segment(
    const DotsRequest & request,
    const std::string & text,
    size_t segment_index,
    PromptConditioning & conditioning,
    DotsAudioVaeStreamState & stream_state,
    const runtime::StreamEventCallback & sink) {
    const int64_t merge_steps = request.generation.vocoder_merge_steps;
    runtime::AudioBuffer merged{static_cast<int>(audio_vae_.sample_rate()), 1, {}};
    std::vector<float> pending;
    int64_t pending_frames = 0;
    int64_t pending_start_patch = 0;
    size_t emitted_chunks = 0;
    double vocoder_ms = 0.0;
    auto emit_chunk = [&](std::vector<float> samples, int64_t first_patch_index, int64_t last_patch_index, bool flush) {
        if (samples.empty()) {
            return;
        }
        runtime::AudioBuffer audio{static_cast<int>(audio_vae_.sample_rate()), 1, std::move(samples)};
        runtime::append_audio_buffer(merged, audio);
        if (sink) {
            runtime::NamedAudioBuffer named;
            named.id = "segment_" + std::to_string(segment_index) + "_chunk_" + std::to_string(emitted_chunks);
            named.audio = std::move(audio);
            if (flush) {
                named.meta.insert_or_assign("flush", "1");
            } else {
                named.meta.insert_or_assign("first_patch", std::to_string(first_patch_index));
                named.meta.insert_or_assign("last_patch", std::to_string(last_patch_index));
            }
            runtime::StreamEvent event;
            event.named_audio_outputs.push_back(std::move(named));
            sink(event);
        }
        ++emitted_chunks;
    };
    auto on_patch = [&](const DotsLatentMatrix & patch, int64_t patch_index) {
        if (merge_steps == 1 || patch_index <= kInitialUnmergedVocoderPatches) {
            const auto start = Clock::now();
            auto audio = audio_vae_.stream_step(patch.values, patch.frames, stream_state);
            vocoder_ms += engine::debug::elapsed_ms(start, Clock::now());
            emit_chunk(std::move(audio.samples), patch_index, patch_index, false);
            return;
        }
        if (pending.empty()) {
            pending_start_patch = patch_index;
        }
        pending.insert(pending.end(), patch.values.begin(), patch.values.end());
        pending_frames += patch.frames;
        if (pending_frames < assets_->config.patch_size * merge_steps) {
            return;
        }
        const auto start = Clock::now();
        auto audio = audio_vae_.stream_step(pending, pending_frames, stream_state);
        vocoder_ms += engine::debug::elapsed_ms(start, Clock::now());
        emit_chunk(std::move(audio.samples), pending_start_patch, patch_index, false);
        pending.clear();
        pending_frames = 0;
        pending_start_patch = 0;
    };
    (void)generate_latent_patches(request, text, conditioning, on_patch);
    release_generation_phase_components();
    if (!pending.empty()) {
        const int64_t last_patch = pending_start_patch + pending_frames / assets_->config.patch_size - 1;
        const auto start = Clock::now();
        auto audio = audio_vae_.stream_step(pending, pending_frames, stream_state);
        vocoder_ms += engine::debug::elapsed_ms(start, Clock::now());
        emit_chunk(std::move(audio.samples), pending_start_patch, last_patch, false);
    }
    debug::trace_log_scalar("dots_tts.streaming.chunks", static_cast<int64_t>(emitted_chunks));
    debug::timing_log_scalar("dots_tts.streaming.vocoder_ms", vocoder_ms);
    return merged;
}

void DotsSession::ensure_speaker_encoder_loaded() {
    if (speaker_encoder_.weights() != nullptr) {
        return;
    }
    engine::modules::CampplusEncoderConfig speaker_config;
    speaker_config.feat_dim = 80;
    speaker_config.embedding_size = assets_->config.campplus_embedding_size;
    speaker_config.tensor_prefix = "model";
    speaker_config.weight_storage_type = speaker_encoder_weight_storage_type_;
    speaker_config.stats_variance_floor = 1.0e-2F;
    speaker_encoder_ = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        assets_->speaker_encoder_weights,
        options().backend,
        std::move(speaker_config));
}

void DotsSession::ensure_audio_vae_loaded() {
    if (audio_vae_.is_loaded()) {
        return;
    }
    audio_vae_ = DotsAudioVaeComponent::load_from_tensor_source(
        assets_->vocoder_weights,
        options().backend,
        assets_->config.vocoder,
        vocoder_weight_storage_type_,
        conv_weight_storage_type_);
}

void DotsSession::ensure_patch_encoder_loaded() {
    if (patch_encoder_.is_loaded()) {
        return;
    }
    patch_encoder_ = DotsPatchEncoderComponent::load_from_tensor_source(
        assets_->core_weights,
        options().backend,
        assets_->config,
        patch_encoder_weight_storage_type_);
}

void DotsSession::ensure_llm_loaded() {
    if (llm_.is_loaded()) {
        return;
    }
    llm_ = DotsLlmComponent::load_from_tensor_source(
        assets_->core_weights,
        options().backend,
        assets_->config.llm,
        llm_weight_storage_type_);
}

void DotsSession::ensure_flow_loaded() {
    if (flow_.is_loaded()) {
        return;
    }
    flow_ = DotsFlowComponent::load_from_tensor_source(
        assets_->core_weights,
        options().backend,
        assets_->config,
        flow_weight_storage_type_);
}

void DotsSession::release_conditioning_phase_components() {
    if (!mem_saver_) {
        return;
    }
    speaker_encoder_ = engine::modules::CampplusEncoderComponent();
    audio_vae_ = DotsAudioVaeComponent();
}

void DotsSession::release_generation_phase_components() {
    if (!mem_saver_) {
        return;
    }
    llm_ = DotsLlmComponent();
    patch_encoder_ = DotsPatchEncoderComponent();
    flow_ = DotsFlowComponent();
}

void DotsSession::release_audio_phase_components() {
    if (!mem_saver_) {
        return;
    }
    audio_vae_ = DotsAudioVaeComponent();
}

runtime::AudioBuffer DotsSession::synthesize_chunked(
    const runtime::TaskRequest & request,
    const DotsRequest & parsed) {
    const auto chunk_requests = runtime::chunk_text_request(
        request,
        parsed.generation.text_chunk_size,
        parsed.generation.text_chunk_mode);
    runtime::AudioBuffer merged;
    for (size_t i = 0; i < chunk_requests.size(); ++i) {
        auto chunk = make_dots_request(*assets_, chunk_requests[i], prepared_defaults_);
        if (i > 0) {
            chunk.generation.seed += static_cast<uint64_t>(i);
        }
        runtime::append_audio_buffer(merged, synthesize_segment(chunk, chunk.text));
    }
    debug::trace_log_scalar(
        "dots_tts.text_chunk_mode",
        engine::text::text_chunk_mode_name(parsed.generation.text_chunk_mode));
    debug::trace_log_scalar("dots_tts.text_chunk_size", parsed.generation.text_chunk_size);
    debug::trace_log_scalar("dots_tts.text.chunk_count", static_cast<int64_t>(chunk_requests.size()));
    return merged;
}

runtime::TaskResult DotsSession::run(const runtime::TaskRequest & request) {
    require_prepared("DotTTS run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("DotTTS run requires an offline session");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "DotTTS");
    const auto start = Clock::now();
    auto parsed = make_dots_request(*assets_, request, prepared_defaults_);
    runtime::TaskResult result;
    result.audio_output = synthesize_chunked(request, parsed);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    return result;
}

runtime::StreamingPolicy DotsSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    return policy;
}

void DotsSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("DotTTS streaming");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("DotTTS start_stream requires a streaming session");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "DotTTS");
    reset();
    const auto start = Clock::now();
    auto parsed = make_dots_request(*assets_, request, prepared_defaults_);
    const auto chunk_requests = runtime::chunk_text_request(
        request,
        parsed.generation.text_chunk_size,
        parsed.generation.text_chunk_mode);
    const auto conditioning_start = Clock::now();
    auto conditioning = prepare_prompt_conditioning(parsed);
    debug::timing_log_scalar("dots_tts.streaming.conditioning_ms", engine::debug::elapsed_ms(conditioning_start, Clock::now()));
    release_conditioning_phase_components();
    ensure_audio_vae_loaded();
    runtime::AudioBuffer merged{static_cast<int>(audio_vae_.sample_rate()), 1, {}};
    const int64_t merge_steps = parsed.generation.vocoder_merge_steps;
    auto stream_state = audio_vae_.create_stream_state(assets_->config.patch_size * merge_steps);
    for (size_t i = 0; i < chunk_requests.size(); ++i) {
        auto chunk = make_dots_request(*assets_, chunk_requests[i], prepared_defaults_);
        if (i > 0) {
            chunk.generation.seed += static_cast<uint64_t>(i);
        }
        runtime::append_audio_buffer(merged, synthesize_streaming_segment(
            chunk,
            chunk.text,
            i,
            conditioning,
            stream_state,
            stream_sink_));
    }
    debug::trace_log_scalar(
        "dots_tts.streaming.text_chunk_mode",
        engine::text::text_chunk_mode_name(parsed.generation.text_chunk_mode));
    debug::trace_log_scalar("dots_tts.streaming.text_chunk_size", parsed.generation.text_chunk_size);
    debug::trace_log_scalar("dots_tts.streaming.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));
    const auto flush_start = Clock::now();
    auto final_audio = audio_vae_.flush_stream(stream_state);
    debug::timing_log_scalar("dots_tts.streaming.flush_vocoder_ms", engine::debug::elapsed_ms(flush_start, Clock::now()));
    if (!final_audio.samples.empty()) {
        runtime::AudioBuffer audio{static_cast<int>(audio_vae_.sample_rate()), 1, std::move(final_audio.samples)};
        runtime::append_audio_buffer(merged, audio);
        if (stream_sink_) {
            runtime::NamedAudioBuffer named;
            named.id = "flush";
            named.audio = std::move(audio);
            named.meta.insert_or_assign("flush", "1");
            runtime::StreamEvent event;
            event.named_audio_outputs.push_back(std::move(named));
            stream_sink_(event);
        }
    }
    release_audio_phase_components();
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    streaming_result_ = std::move(result);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
}

std::optional<runtime::StreamEvent> DotsSession::next_stream_event() {
    if (!streaming_result_.has_value()) {
        throw std::runtime_error("DotTTS streaming has not been started");
    }
    return std::nullopt;
}

void DotsSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}

runtime::TaskResult DotsSession::finish_stream() {
    if (!streaming_result_.has_value()) {
        throw std::runtime_error("DotTTS streaming has not been started");
    }
    runtime::TaskResult result = std::move(*streaming_result_);
    reset();
    return result;
}

void DotsSession::reset() {
    streaming_result_.reset();
}

runtime::StreamEvent DotsSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("DotTTS streaming does not consume audio chunks");
}

runtime::TaskResult DotsSession::finalize() {
    return finish_stream();
}

std::shared_ptr<runtime::IVoiceModelLoader> make_dots_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<DotsAssets> config;
    config.family = kFamily;
    config.load_assets = load_dots_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const DotsAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<DotsSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::dots_tts
