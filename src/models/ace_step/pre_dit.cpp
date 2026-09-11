#include "engine/models/ace_step/pre_dit.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/ace_step/prompt_builder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::ace_step {
namespace {

constexpr int64_t kSamplesPerLatentFrame = 1920;

using Clock = std::chrono::steady_clock;

AceStepTokenizedText trim_to_valid_tokens(const AceStepTokenizedText & tokens) {
    const auto valid_end = std::find(tokens.attention_mask.begin(), tokens.attention_mask.end(), 0);
    const size_t valid = static_cast<size_t>(std::distance(tokens.attention_mask.begin(), valid_end));
    AceStepTokenizedText trimmed;
    trimmed.text = tokens.text;
    trimmed.input_ids.assign(tokens.input_ids.begin(), tokens.input_ids.begin() + static_cast<std::ptrdiff_t>(valid));
    trimmed.attention_mask.assign(tokens.attention_mask.begin(), tokens.attention_mask.begin() + static_cast<std::ptrdiff_t>(valid));
    return trimmed;
}

std::vector<float> load_silence_latent_transposed(
    const AceStepAssets & assets,
    int64_t & frames,
    int64_t & channels) {
    const auto raw = assets.dit_silence_latent->require_f32("silence_latent", {1, 64, 15000});
    channels = 64;
    frames = 15000;
    std::vector<float> transposed(static_cast<size_t>(frames * channels), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            transposed[static_cast<size_t>(frame * channels + channel)] =
                raw[static_cast<size_t>(channel * frames + frame)];
        }
    }
    return transposed;
}

AceStepLatents silence_latent_slice(
    const std::vector<float> & silence_latent,
    int64_t silence_frames,
    int64_t channels,
    int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("ACE-Step silence latent slice frame request must be positive");
    }
    AceStepLatents out;
    out.frames = frames;
    out.channels = channels;
    out.values.resize(static_cast<size_t>(frames * channels), 0.0F);
    if (frames <= silence_frames) {
        std::copy_n(silence_latent.begin(), static_cast<size_t>(frames * channels), out.values.begin());
        return out;
    }
    const int64_t repeats = (frames + silence_frames - 1) / silence_frames;
    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
        const int64_t dst_frame = repeat * silence_frames;
        const int64_t copy_frames = std::min<int64_t>(silence_frames, frames - dst_frame);
        std::copy_n(
            silence_latent.data(),
            static_cast<size_t>(copy_frames * channels),
            out.values.data() + static_cast<size_t>(dst_frame * channels));
    }
    return out;
}

AceStepLatents align_latents_to_length(
    const AceStepLatents & latents,
    const std::vector<float> & silence_latent,
    int64_t silence_frames,
    int64_t silence_channels,
    int64_t target_frames) {
    if (latents.channels != silence_channels) {
        throw std::runtime_error("ACE-Step latent alignment requires matching channel counts");
    }
    if (target_frames <= 0) {
        throw std::runtime_error("ACE-Step latent alignment target must be positive");
    }
    AceStepLatents out = silence_latent_slice(
        silence_latent,
        silence_frames,
        silence_channels,
        target_frames);
    const int64_t copy_frames = std::min(latents.frames, target_frames);
    std::copy_n(
        latents.values.begin(),
        static_cast<size_t>(copy_frames * latents.channels),
        out.values.begin());
    return out;
}

std::vector<int32_t> build_attention_mask(int64_t valid_frames, int64_t padded_frames) {
    if (valid_frames <= 0 || padded_frames < valid_frames) {
        throw std::runtime_error("ACE-Step latent attention mask shape is invalid");
    }
    std::vector<int32_t> mask(static_cast<size_t>(padded_frames), 0);
    std::fill_n(mask.begin(), static_cast<size_t>(valid_frames), 1);
    return mask;
}

bool is_silent_audio(const runtime::AudioBuffer & audio) {
    double sum = 0.0;
    for (const float sample : audio.samples) {
        sum += std::abs(static_cast<double>(sample));
    }
    return sum <= 1.0e-6;
}

runtime::AudioBuffer normalize_audio_to_stereo_48k(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("ACE-Step source audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("ACE-Step source audio channel count must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("ACE-Step source audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("ACE-Step source audio has invalid channel layout");
    }

    std::vector<float> left;
    std::vector<float> right;
    if (audio.channels == 1) {
        left = audio.samples;
        right = audio.samples;
    } else {
        left = engine::audio::extract_interleaved_channel(audio.samples, audio.channels, 0);
        right = engine::audio::extract_interleaved_channel(audio.samples, audio.channels, 1);
    }
    if (audio.sample_rate != kAceStepAudioSampleRate) {
        left = engine::audio::resample_mono_torchaudio_sinc_hann(left, audio.sample_rate, kAceStepAudioSampleRate);
        right = engine::audio::resample_mono_torchaudio_sinc_hann(right, audio.sample_rate, kAceStepAudioSampleRate);
    }

    std::vector<float> planar;
    planar.reserve(left.size() + right.size());
    for (float value : left) {
        planar.push_back(std::clamp(value, -1.0F, 1.0F));
    }
    for (float value : right) {
        planar.push_back(std::clamp(value, -1.0F, 1.0F));
    }

    runtime::AudioBuffer out;
    out.sample_rate = kAceStepAudioSampleRate;
    out.channels = kAceStepAudioChannels;
    out.samples = engine::audio::interleave_planar_channels(
        planar,
        kAceStepAudioChannels,
        static_cast<int64_t>(left.size()));
    return out;
}

int64_t latent_frames_for_audio(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("ACE-Step latent frame conversion requires non-empty audio");
    }
    const int64_t frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    const int64_t latent_frames = frames / kSamplesPerLatentFrame;
    if (latent_frames <= 0) {
        throw std::runtime_error("ACE-Step source audio is too short to encode into latents");
    }
    return latent_frames;
}

runtime::AudioBuffer make_silent_audio(float duration_seconds) {
    if (!(duration_seconds > 0.0F)) {
        throw std::runtime_error("ACE-Step duration_seconds must be positive");
    }
    const int64_t frames =
        static_cast<int64_t>(std::llround(static_cast<double>(duration_seconds) * kAceStepAudioSampleRate));
    if (frames <= 0) {
        throw std::runtime_error("ACE-Step silent audio duration produced no frames");
    }
    runtime::AudioBuffer out;
    out.sample_rate = kAceStepAudioSampleRate;
    out.channels = kAceStepAudioChannels;
    out.samples.assign(static_cast<size_t>(frames * out.channels), 0.0F);
    return out;
}

runtime::AudioBuffer make_silent_audio_for_latent_frames(int64_t latent_frames) {
    if (latent_frames <= 0) {
        throw std::runtime_error("ACE-Step silent audio latent length must be positive");
    }
    runtime::AudioBuffer out;
    out.sample_rate = kAceStepAudioSampleRate;
    out.channels = kAceStepAudioChannels;
    out.samples.assign(static_cast<size_t>(latent_frames * kSamplesPerLatentFrame * out.channels), 0.0F);
    return out;
}

int64_t latent_frames_for_audio_codes(const AceStepPlan & plan) {
    return static_cast<int64_t>(plan.audio_code_ids.size()) * 5;
}

struct TargetAudioPreparation {
    runtime::AudioBuffer target_audio;
    std::optional<float> repainting_start_seconds;
    std::optional<float> repainting_end_seconds;
};

TargetAudioPreparation prepare_target_audio(
    const AceStepRequest & request,
    const AceStepTaskRoute & route,
    const AceStepPlan & plan,
    const std::optional<runtime::AudioBuffer> & source_audio) {
    TargetAudioPreparation out = {};
    if (!source_audio.has_value()) {
        out.target_audio = plan.audio_code_ids.empty()
            ? make_silent_audio(request.generation.duration_seconds)
            : make_silent_audio_for_latent_frames(latent_frames_for_audio_codes(plan));
        return out;
    }

    if (!ace_step_route_uses_repaint_window(route)) {
        out.target_audio = *source_audio;
        return out;
    }

    const float source_duration_seconds =
        static_cast<float>(source_audio->samples.size() / static_cast<size_t>(source_audio->channels)) /
        static_cast<float>(kAceStepAudioSampleRate);
    const float start_seconds = request.repainting_start_seconds.value_or(0.0F);
    float end_seconds = request.repainting_end_seconds.value_or(source_duration_seconds);
    if (end_seconds < 0.0F) {
        end_seconds = source_duration_seconds;
    }
    const float left_padding_seconds = std::max(0.0F, -start_seconds);
    const float right_padding_seconds = std::max(0.0F, end_seconds - source_duration_seconds);
    const int64_t left_padding_frames =
        static_cast<int64_t>(std::llround(left_padding_seconds * kAceStepAudioSampleRate));
    const int64_t right_padding_frames =
        static_cast<int64_t>(std::llround(right_padding_seconds * kAceStepAudioSampleRate));

    out.target_audio = *source_audio;
    if (left_padding_frames > 0 || right_padding_frames > 0) {
        runtime::AudioBuffer padded;
        padded.sample_rate = source_audio->sample_rate;
        padded.channels = source_audio->channels;
        const int64_t source_frames =
            static_cast<int64_t>(source_audio->samples.size() / static_cast<size_t>(source_audio->channels));
        const int64_t total_frames = left_padding_frames + source_frames + right_padding_frames;
        padded.samples.assign(static_cast<size_t>(total_frames * padded.channels), 0.0F);
        std::copy(
            source_audio->samples.begin(),
            source_audio->samples.end(),
            padded.samples.begin() + static_cast<std::ptrdiff_t>(left_padding_frames * padded.channels));
        out.target_audio = std::move(padded);
    }
    out.repainting_start_seconds = start_seconds + left_padding_seconds;
    out.repainting_end_seconds = end_seconds + left_padding_seconds;
    return out;
}

AceStepLatents build_chunk_mask_latents(
    int64_t frames,
    int64_t channels,
    std::optional<std::pair<int64_t, int64_t>> repaint_span,
    std::string_view chunk_mask_mode) {
    if (frames <= 0 || channels <= 0) {
        throw std::runtime_error("ACE-Step chunk mask requires positive shape");
    }
    AceStepLatents out;
    out.frames = frames;
    out.channels = channels;
    out.values.resize(static_cast<size_t>(frames * channels), 0.0F);

    std::vector<float> per_frame(static_cast<size_t>(frames), 1.0F);
    if (repaint_span.has_value()) {
        std::fill(per_frame.begin(), per_frame.end(), 0.0F);
        const auto [start_frame, end_frame] = *repaint_span;
        if (start_frame < 0 || end_frame <= start_frame || end_frame > frames) {
            throw std::runtime_error("ACE-Step repaint span is invalid");
        }
        std::fill(
            per_frame.begin() + static_cast<std::ptrdiff_t>(start_frame),
            per_frame.begin() + static_cast<std::ptrdiff_t>(end_frame),
            1.0F);
    }
    if (chunk_mask_mode == "auto") {
        // Python writes 2.0 into a bool chunk mask, so DiT receives true -> 1.0.
        std::fill(per_frame.begin(), per_frame.end(), 1.0F);
    }
    for (int64_t frame = 0; frame < frames; ++frame) {
        float * dst = out.values.data() + static_cast<size_t>(frame * channels);
        std::fill_n(dst, static_cast<size_t>(channels), per_frame[static_cast<size_t>(frame)]);
    }
    return out;
}

std::vector<int32_t> build_repaint_mask(
    int64_t frames,
    std::optional<std::pair<int64_t, int64_t>> repaint_span) {
    if (!repaint_span.has_value()) {
        return {};
    }
    std::vector<int32_t> mask(static_cast<size_t>(frames), 0);
    const auto [start_frame, end_frame] = *repaint_span;
    if (start_frame < 0 || end_frame <= start_frame || end_frame > frames) {
        throw std::runtime_error("ACE-Step repaint mask span is invalid");
    }
    std::fill(
        mask.begin() + static_cast<std::ptrdiff_t>(start_frame),
        mask.begin() + static_cast<std::ptrdiff_t>(end_frame),
        1);
    return mask;
}

std::optional<std::pair<int64_t, int64_t>> repaint_span_from_seconds(
    int64_t frames,
    const std::optional<float> & start_seconds,
    const std::optional<float> & end_seconds) {
    if (!start_seconds.has_value() || !end_seconds.has_value()) {
        return std::nullopt;
    }
    int64_t start_latent =
        static_cast<int64_t>(std::floor(*start_seconds * kAceStepAudioSampleRate / kSamplesPerLatentFrame));
    int64_t end_latent =
        static_cast<int64_t>(std::floor(*end_seconds * kAceStepAudioSampleRate / kSamplesPerLatentFrame));
    start_latent = std::max<int64_t>(0, std::min<int64_t>(start_latent, frames - 1));
    end_latent = std::max<int64_t>(start_latent + 1, std::min<int64_t>(end_latent, frames));
    return std::pair<int64_t, int64_t>{start_latent, end_latent};
}

AceStepLatents build_src_latents(
    const AceStepLatents & target_latents,
    const AceStepLatents & silence_latents,
    std::optional<std::pair<int64_t, int64_t>> repaint_span,
    bool has_target_audio,
    const AceStepTaskRoute & route) {
    if (!has_target_audio) {
        return silence_latents;
    }
    AceStepLatents out = target_latents;
    if (repaint_span.has_value() && !route.preserve_repaint_source) {
        const auto [start_frame, end_frame] = *repaint_span;
        for (int64_t frame = start_frame; frame < end_frame; ++frame) {
            const float * src = silence_latents.values.data() + static_cast<size_t>(frame * silence_latents.channels);
            float * dst = out.values.data() + static_cast<size_t>(frame * out.channels);
            std::copy_n(src, static_cast<size_t>(out.channels), dst);
        }
    }
    return out;
}

AceStepLatents build_context_latents(const AceStepLatents & src_latents, const AceStepLatents & chunk_mask) {
    if (src_latents.frames <= 0 || src_latents.channels <= 0) {
        throw std::runtime_error("ACE-Step context latents require positive source latent shape");
    }
    if (src_latents.frames != chunk_mask.frames || src_latents.channels != chunk_mask.channels) {
        throw std::runtime_error("ACE-Step context latents require source and chunk mask shapes to match");
    }
    AceStepLatents out;
    out.frames = src_latents.frames;
    out.channels = src_latents.channels * 2;
    out.values.resize(static_cast<size_t>(out.frames * out.channels), 0.0F);
    for (int64_t frame = 0; frame < src_latents.frames; ++frame) {
        const float * src = src_latents.values.data() + static_cast<size_t>(frame * src_latents.channels);
        const float * mask = chunk_mask.values.data() + static_cast<size_t>(frame * chunk_mask.channels);
        float * dst = out.values.data() + static_cast<size_t>(frame * out.channels);
        std::copy_n(src, static_cast<size_t>(src_latents.channels), dst);
        std::copy_n(mask, static_cast<size_t>(chunk_mask.channels), dst + src_latents.channels);
    }
    return out;
}

const AceStepTaskRoute & effective_dit_route_for_plan(const AceStepTaskRoute & route, const AceStepPlan & plan) {
    if (route.task == AceStepTaskType::TextToMusic && !plan.audio_code_ids.empty()) {
        return ace_step_task_route(AceStepTaskType::Cover);
    }
    return route;
}

}  // namespace

AceStepPreDitRuntime::AceStepPreDitRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const AceStepAssets> assets,
    std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime,
    assets::TensorStorageType dit_weight_storage_type,
    assets::TensorStorageType vae_encoder_weight_storage_type,
    assets::TensorStorageType text_encoder_weight_storage_type)
    : assets_(std::move(assets)),
      dit_weights_runtime_(std::move(dit_weights_runtime)),
      dit_weight_storage_type_(dit_weight_storage_type),
      vae_encoder_weight_storage_type_(vae_encoder_weight_storage_type),
      text_encoder_weight_storage_type_(text_encoder_weight_storage_type),
      tokenizer_(assets_),
      execution_(&execution) {
    if (assets_ == nullptr) {
        throw std::runtime_error("ACE-Step pre-DiT runtime requires assets");
    }
    if (execution_ == nullptr) {
        throw std::runtime_error("ACE-Step pre-DiT runtime requires execution context");
    }
    silence_latent_ = load_silence_latent_transposed(*assets_, silence_latent_frames_, silence_latent_channels_);
}

void AceStepPreDitRuntime::prepare_runtime() const {
    ensure_text_encoder();
    text_encoder_->prepare_runtime();
    ensure_condition_encoder();
    ensure_detokenizer();
}

void AceStepPreDitRuntime::release_runtime_graphs() const {
    if (text_encoder_) {
        text_encoder_->release_runtime_graphs();
    }
    if (condition_encoder_) {
        condition_encoder_->release_runtime_graphs();
    }
    if (cover_tokenizer_) {
        cover_tokenizer_->release_runtime_graphs();
    }
    if (detokenizer_) {
        detokenizer_->release_runtime_graphs();
    }
    if (vae_encoder_) {
        vae_encoder_->release_runtime_graphs();
    }
}

AceStepPreDitInputs AceStepPreDitRuntime::prepare(
    const AceStepRequest & request,
    const AceStepTaskRoute & route,
    const AceStepPlan & plan) const {
    const auto total_start = Clock::now();

    const auto source_audio_start = Clock::now();
    std::optional<runtime::AudioBuffer> source_audio;
    if (ace_step_route_uses_request_source_audio(route, request, plan)) {
        source_audio = normalize_audio_to_stereo_48k(*request.source_audio);
    } else if (!ace_step_route_has_required_source_audio_input(route, request, plan)) {
        throw std::runtime_error("ACE-Step task requires source audio");
    }
    engine::debug::timing_log_scalar("ace_step.pre_dit.source_audio_ms", engine::debug::elapsed_ms(source_audio_start, Clock::now()));

    const auto target_audio_start = Clock::now();
    TargetAudioPreparation target_audio = prepare_target_audio(request, route, plan, source_audio);
    const bool target_is_silent = is_silent_audio(target_audio.target_audio);
    engine::debug::timing_log_scalar("ace_step.pre_dit.target_audio_ms", engine::debug::elapsed_ms(target_audio_start, Clock::now()));
    const auto target_latents_start = Clock::now();
    AceStepLatents raw_target_latents;
    if (target_is_silent) {
        raw_target_latents = silence_latent_slice(
            silence_latent_,
            silence_latent_frames_,
            silence_latent_channels_,
            latent_frames_for_audio(target_audio.target_audio));
        engine::debug::timing_log_scalar("ace_step.pre_dit.ensure_vae_encoder_ms", 0.0);
    } else {
        const auto vae_encoder_ensure_start = Clock::now();
        ensure_vae_encoder();
        engine::debug::timing_log_scalar(
            "ace_step.pre_dit.ensure_vae_encoder_ms",
            engine::debug::elapsed_ms(vae_encoder_ensure_start, Clock::now()));
        raw_target_latents = vae_encoder_->encode(
            target_audio.target_audio,
            request.generation.seed,
            request.generation.noise_file);
    }
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.target_latents_ms",
        engine::debug::elapsed_ms(target_latents_start, Clock::now()));
    if (execution_->backend_type() == core::BackendType::Metal) {
        release_vae_encoder();
    }
    engine::debug::trace_log_scalar("ace_step.pre_dit.target_is_silent", target_is_silent);
    engine::debug::trace_log_scalar("ace_step.pre_dit.raw_target_latent_frames", raw_target_latents.frames);

    const auto latent_assembly_start = Clock::now();
    const int64_t max_latent_length = std::max<int64_t>(128, raw_target_latents.frames);
    const AceStepLatents silence_latents = silence_latent_slice(
        silence_latent_,
        silence_latent_frames_,
        silence_latent_channels_,
        max_latent_length);
    const auto repaint_span = repaint_span_from_seconds(
        max_latent_length,
        target_audio.repainting_start_seconds,
        target_audio.repainting_end_seconds);

    AceStepPreDitInputs out;
    out.plan = plan;
    out.target_latents = align_latents_to_length(
        raw_target_latents,
        silence_latent_,
        silence_latent_frames_,
        silence_latent_channels_,
        max_latent_length);
    out.latent_attention_mask = build_attention_mask(raw_target_latents.frames, max_latent_length);
    out.chunk_mask = build_chunk_mask_latents(
        max_latent_length,
        silence_latent_channels_,
        repaint_span,
        request.chunk_mask_mode);
    out.repaint_mask = build_repaint_mask(max_latent_length, repaint_span);
    const bool use_lm_hints_as_src = route.uses_cover_conditioning || !plan.audio_code_ids.empty();
    out.is_cover = use_lm_hints_as_src;
    out.src_latents = build_src_latents(
        out.target_latents,
        silence_latents,
        repaint_span,
        source_audio.has_value(),
        route);
    if (route.task == AceStepTaskType::Repaint) {
        out.repaint_splice_start_seconds = target_audio.repainting_start_seconds.value_or(0.0F);
        out.repaint_splice_end_seconds = target_audio.repainting_end_seconds.value_or(
            static_cast<float>(target_audio.target_audio.samples.size() /
                               static_cast<size_t>(target_audio.target_audio.channels)) /
            static_cast<float>(kAceStepAudioSampleRate));
        out.repaint_splice_audio = std::move(target_audio.target_audio);
    }
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.latent_assembly_ms",
        engine::debug::elapsed_ms(latent_assembly_start, Clock::now()));
    engine::debug::trace_log_scalar("ace_step.pre_dit.max_latent_length", max_latent_length);

    const auto tokenize_start = Clock::now();
    const AceStepMetadata metadata = {
        plan.metadata.bpm,
        plan.metadata.duration,
        plan.metadata.keyscale,
        plan.metadata.timesignature,
        std::nullopt,
        std::nullopt,
    };
    const std::string caption =
        request.generation.use_cot_caption && !plan.caption.empty() ? plan.caption : request.prompt;
    const std::string lyric_language =
        request.generation.use_cot_language && plan.metadata.language.has_value() && !plan.metadata.language->empty()
            ? *plan.metadata.language
            : request.vocal_language;
    const AceStepTaskRoute & dit_route = effective_dit_route_for_plan(route, plan);
    const std::string dit_instruction = ace_step_task_instruction(dit_route, request);
    out.text_prompt = trim_to_valid_tokens(tokenizer_.tokenize_text(
        ace_step_build_dit_caption_prompt(
            dit_instruction,
            caption,
            metadata,
            request.generation.duration_seconds),
        256));
    out.lyrics_prompt = trim_to_valid_tokens(tokenizer_.tokenize_text(
        ace_step_format_lyrics(
            request.lyrics,
            lyric_language),
        2048));
    if (request.generation.audio_cover_strength < 1.0F) {
        out.non_cover_text_prompt = trim_to_valid_tokens(tokenizer_.tokenize_text(
            ace_step_build_dit_caption_prompt(
                dit_instruction,
                caption,
                metadata,
                request.generation.duration_seconds),
            256));
    }
    engine::debug::timing_log_scalar("ace_step.pre_dit.tokenize_ms", engine::debug::elapsed_ms(tokenize_start, Clock::now()));
    engine::debug::trace_log_scalar("ace_step.pre_dit.text_tokens", out.text_prompt.input_ids.size());
    engine::debug::trace_log_scalar("ace_step.pre_dit.lyric_tokens", out.lyrics_prompt.input_ids.size());

    const auto text_encoder_ensure_start = Clock::now();
    ensure_text_encoder();
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.ensure_text_encoder_ms",
        engine::debug::elapsed_ms(text_encoder_ensure_start, Clock::now()));
    const auto text_encode_start = Clock::now();
    out.text_hidden_states = text_encoder_->encode(out.text_prompt);
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.text_encode_ms",
        engine::debug::elapsed_ms(text_encode_start, Clock::now()));
    if (!out.non_cover_text_prompt.input_ids.empty()) {
        const auto non_cover_text_encode_start = Clock::now();
        out.non_cover_text_hidden_states = text_encoder_->encode(out.non_cover_text_prompt);
        engine::debug::timing_log_scalar(
            "ace_step.pre_dit.non_cover_text_encode_ms",
            engine::debug::elapsed_ms(non_cover_text_encode_start, Clock::now()));
    }
    const auto lyric_embed_start = Clock::now();
    out.lyric_token_embeddings = text_encoder_->embed_tokens(out.lyrics_prompt);
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.lyric_embed_ms",
        engine::debug::elapsed_ms(lyric_embed_start, Clock::now()));

    const auto detokenizer_ensure_start = Clock::now();
    ensure_detokenizer();
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.ensure_detokenizer_ms",
        engine::debug::elapsed_ms(detokenizer_ensure_start, Clock::now()));
    const auto detokenizer_start = Clock::now();
    if (plan.audio_code_ids.empty()) {
        if (route.uses_planner && request.generation.thinking) {
            throw std::runtime_error("ACE-Step planner route produced no audio codes");
        }
        if (route.uses_cover_conditioning) {
            const auto cover_tokenizer_ensure_start = Clock::now();
            ensure_cover_tokenizer();
            engine::debug::timing_log_scalar(
                "ace_step.pre_dit.ensure_cover_tokenizer_ms",
                engine::debug::elapsed_ms(cover_tokenizer_ensure_start, Clock::now()));
            const std::vector<int32_t> cover_audio_codes = cover_tokenizer_->encode_audio_codes(
                out.src_latents,
                silence_latent_,
                silence_latent_frames_,
                silence_latent_channels_);
            out.lm_hints_25hz = detokenizer_->decode_audio_codes(cover_audio_codes);
        } else {
            engine::debug::timing_log_scalar("ace_step.pre_dit.ensure_cover_tokenizer_ms", 0.0);
            out.lm_hints_25hz = silence_latents;
        }
    } else {
        engine::debug::timing_log_scalar("ace_step.pre_dit.ensure_cover_tokenizer_ms", 0.0);
        out.lm_hints_25hz = detokenizer_->decode_audio_codes(plan.audio_code_ids);
    }
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.detokenizer_decode_ms",
        engine::debug::elapsed_ms(detokenizer_start, Clock::now()));
    const auto lm_hints_align_start = Clock::now();
    out.lm_hints_25hz = align_latents_to_length(
        out.lm_hints_25hz,
        silence_latent_,
        silence_latent_frames_,
        silence_latent_channels_,
        max_latent_length);
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.lm_hints_align_ms",
        engine::debug::elapsed_ms(lm_hints_align_start, Clock::now()));

    const auto timbre_slice_start = Clock::now();
    const std::vector<float> refer_audio_acoustic_hidden_states_packed =
        silence_latent_slice(
            silence_latent_,
            silence_latent_frames_,
            silence_latent_channels_,
            assets_->config.diffusion.timbre_fix_frame)
            .values;
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.timbre_slice_ms",
        engine::debug::elapsed_ms(timbre_slice_start, Clock::now()));

    const auto condition_encoder_ensure_start = Clock::now();
    ensure_condition_encoder();
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.ensure_condition_encoder_ms",
        engine::debug::elapsed_ms(condition_encoder_ensure_start, Clock::now()));
    const auto condition_encoder_start = Clock::now();
    out.encoder_hidden_states = condition_encoder_->encode(
        out.text_hidden_states,
        out.lyric_token_embeddings,
        refer_audio_acoustic_hidden_states_packed,
        1,
        assets_->config.diffusion.timbre_fix_frame,
        {0});
    engine::debug::timing_log_scalar(
        "ace_step.pre_dit.condition_encoder_ms",
        engine::debug::elapsed_ms(condition_encoder_start, Clock::now()));
    if (!out.non_cover_text_hidden_states.values.empty()) {
        const auto non_cover_condition_encoder_start = Clock::now();
        out.encoder_hidden_states_non_cover = condition_encoder_->encode(
            out.non_cover_text_hidden_states,
            out.lyric_token_embeddings,
            refer_audio_acoustic_hidden_states_packed,
            1,
            assets_->config.diffusion.timbre_fix_frame,
            {0});
        engine::debug::timing_log_scalar(
            "ace_step.pre_dit.non_cover_condition_encoder_ms",
            engine::debug::elapsed_ms(non_cover_condition_encoder_start, Clock::now()));
    }
    const auto context_start = Clock::now();
    const AceStepLatents main_src_latents = use_lm_hints_as_src ? out.lm_hints_25hz : out.src_latents;
    out.context_latents = build_context_latents(main_src_latents, out.chunk_mask);
    if (!out.encoder_hidden_states_non_cover.values.empty()) {
        out.context_latents_non_cover = build_context_latents(silence_latents, out.chunk_mask);
    }
    engine::debug::timing_log_scalar("ace_step.pre_dit.context_latents_ms", engine::debug::elapsed_ms(context_start, Clock::now()));
    engine::debug::timing_log_scalar("ace_step.pre_dit.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return out;
}

void AceStepPreDitRuntime::ensure_text_encoder() const {
    if (!text_encoder_) {
        text_encoder_ = std::make_unique<AceStepQwenTextEncoderRuntime>(
            *execution_,
            assets_,
            text_encoder_weight_storage_type_);
    }
}

void AceStepPreDitRuntime::ensure_condition_encoder() const {
    if (!condition_encoder_) {
        condition_encoder_ = std::make_unique<AceStepConditionEncoderRuntime>(
            *execution_,
            assets_,
            dit_weights_runtime_);
    }
}

void AceStepPreDitRuntime::ensure_cover_tokenizer() const {
    if (!cover_tokenizer_) {
        cover_tokenizer_ = std::make_unique<AceStepCoverTokenizerRuntime>(
            *execution_,
            assets_,
            dit_weight_storage_type_);
    }
}

void AceStepPreDitRuntime::ensure_detokenizer() const {
    if (!detokenizer_) {
        detokenizer_ = std::make_unique<AceStepAudioDetokenizerRuntime>(*execution_, assets_, dit_weights_runtime_);
    }
}

void AceStepPreDitRuntime::ensure_vae_encoder() const {
    if (!vae_encoder_) {
        vae_encoder_ = std::make_unique<AceStepVAEEncoderRuntime>(
            assets_,
            *execution_,
            vae_encoder_weight_storage_type_);
    }
}

void AceStepPreDitRuntime::release_vae_encoder() const {
    vae_encoder_.reset();
}

}  // namespace engine::models::ace_step
