#include "request.h"

#include "args.h"

#include "engine/framework/audio/wav_reader.h"

#include <filesystem>
#include <string>
#include <utility>

namespace minitts::cli {
namespace {

std::string path_arg_string(const std::filesystem::path & path) {
    return path.string();
}

std::filesystem::path resolve_case_path(
    const std::filesystem::path & base_dir,
    const std::string & value) {
    std::filesystem::path path(value);
    return path.is_absolute() ? path : base_dir / path;
}

void set_option_from_json_field(
    std::unordered_map<std::string, std::string> & options,
    const engine::io::json::Value & object,
    const std::string & field,
    const std::string & option_key) {
    const auto * value = object.find(field);
    if (value != nullptr && !value->is_null()) {
        set_option(options, option_key, json_option_string(*value));
    }
}

}  // namespace

engine::runtime::AudioBuffer read_audio_buffer(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    return engine::runtime::AudioBuffer{
        wav.sample_rate,
        wav.channels,
        wav.samples,
    };
}

std::string json_option_string(const engine::io::json::Value & value) {
    if (value.is_string()) {
        return value.as_string();
    }
    if (value.is_bool()) {
        return value.as_bool() ? "true" : "false";
    }
    if (value.is_number()) {
        return engine::io::json::stringify_number(value.as_number());
    }
    return engine::io::json::stringify(value);
}

std::unordered_map<std::string, std::string> json_options_map(const engine::io::json::Value * value) {
    std::unordered_map<std::string, std::string> options;
    if (value == nullptr || value->is_null()) {
        return options;
    }
    for (const auto & [key, child] : value->as_object()) {
        options[key] = json_option_string(child);
    }
    return options;
}

std::optional<std::string> json_optional_string(
    const engine::io::json::Value & object,
    const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return std::nullopt;
    }
    return value->as_string();
}

std::optional<float> json_optional_float(
    const engine::io::json::Value & object,
    const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return std::nullopt;
    }
    return value->as_f32();
}

engine::runtime::TaskRequest build_request_from_json(
    const engine::io::json::Value & value,
    const std::filesystem::path & base_dir) {
    engine::runtime::TaskRequest request;
    const std::string language = json_optional_string(value, "language").value_or("");
    if (const auto text = json_optional_string(value, "text")) {
        request.text_input = engine::runtime::Transcript{*text, language};
    }
    if (const auto audio = json_optional_string(value, "audio")) {
        request.audio_input = read_audio_buffer(resolve_case_path(base_dir, *audio));
    }

    engine::runtime::VoiceCondition voice;
    bool has_voice = false;
    if (const auto voice_id = json_optional_string(value, "voice_id")) {
        engine::runtime::VoiceReference reference;
        reference.cached_voice_id = *voice_id;
        voice.speaker = std::move(reference);
        has_voice = true;
    }
    if (const auto voice_ref = json_optional_string(value, "voice_ref")) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        voice.speaker->audio = read_audio_buffer(resolve_case_path(base_dir, *voice_ref));
        has_voice = true;
    }

    engine::runtime::StyleCondition style;
    if (const auto style_language = json_optional_string(value, "style_language")) {
        style.language = *style_language;
        has_voice = true;
    }
    if (const auto emotion = json_optional_string(value, "emotion")) {
        style.emotion = *emotion;
        has_voice = true;
    }
    if (const auto speaking_rate = json_optional_float(value, "speaking_rate")) {
        style.speaking_rate = *speaking_rate;
        has_voice = true;
    }
    if (const auto pitch_shift = json_optional_float(value, "pitch_shift")) {
        style.pitch_shift = *pitch_shift;
        has_voice = true;
    }
    if (const auto energy_scale = json_optional_float(value, "energy_scale")) {
        style.energy_scale = *energy_scale;
        has_voice = true;
    }
    style.tags = json_options_map(value.find("style_tags"));
    if (!style.tags.empty()) {
        has_voice = true;
    }
    if (style.language.has_value() || style.emotion.has_value() || style.speaking_rate.has_value() ||
        style.pitch_shift.has_value() || style.energy_scale.has_value() || !style.tags.empty()) {
        voice.style = std::move(style);
    }
    if (has_voice) {
        request.voice = std::move(voice);
    }

    request.options = json_options_map(value.find("options"));
    if (!language.empty()) {
        set_option(request.options, "language", language);
    }
    if (const auto route = json_optional_string(value, "task_route")) {
        set_option(request.options, "route", *route);
    }
    if (const auto route = json_optional_string(value, "route")) {
        set_option(request.options, "route", *route);
    }
    if (const auto source_audio = json_optional_string(value, "source_audio")) {
        set_option(request.options, "source_audio", path_arg_string(resolve_case_path(base_dir, *source_audio)));
    }
    if (const auto target_voice = json_optional_string(value, "target_voice")) {
        set_option(request.options, "target_voice", path_arg_string(resolve_case_path(base_dir, *target_voice)));
    }
    if (const auto prosody_ref = json_optional_string(value, "prosody_ref")) {
        set_option(request.options, "prosody_ref", path_arg_string(resolve_case_path(base_dir, *prosody_ref)));
    }
    if (const auto style_ref = json_optional_string(value, "style_ref")) {
        set_option(request.options, "style_ref", path_arg_string(resolve_case_path(base_dir, *style_ref)));
    }
    if (const auto target_text = json_optional_string(value, "target_text")) {
        set_option(request.options, "target_text", *target_text);
    }
    if (const auto style_ref_text = json_optional_string(value, "style_ref_text")) {
        set_option(request.options, "style_ref_text", *style_ref_text);
    }
    if (const auto lyrics = json_optional_string(value, "lyrics")) {
        set_option(request.options, "lyrics", *lyrics);
    }
    if (const auto track_name = json_optional_string(value, "track_name")) {
        set_option(request.options, "track_name", *track_name);
    }
    if (const auto speaker = json_optional_string(value, "speaker")) {
        set_option(request.options, "speaker", *speaker);
    }
    if (const auto duration_seconds = json_optional_float(value, "duration_seconds")) {
        set_option(request.options, "duration_seconds", std::to_string(*duration_seconds));
    }
    if (const auto repaint_start = json_optional_float(value, "repaint_start")) {
        set_option(request.options, "repainting_start", std::to_string(*repaint_start));
    }
    if (const auto repaint_end = json_optional_float(value, "repaint_end")) {
        set_option(request.options, "repainting_end", std::to_string(*repaint_end));
    }
    if (const auto repaint_mode = json_optional_string(value, "repaint_mode")) {
        set_option(request.options, "repaint_mode", *repaint_mode);
    }
    if (const auto repaint_strength = json_optional_float(value, "repaint_strength")) {
        set_option(request.options, "repaint_strength", std::to_string(*repaint_strength));
    }
    set_option_from_json_field(request.options, value, "seed", "seed");
    set_option_from_json_field(request.options, value, "max_tokens", "max_tokens");
    set_option_from_json_field(request.options, value, "max_steps", "max_steps");
    set_option_from_json_field(request.options, value, "temperature", "temperature");
    set_option_from_json_field(request.options, value, "top_k", "top_k");
    set_option_from_json_field(request.options, value, "top_p", "top_p");
    set_option_from_json_field(request.options, value, "repetition_penalty", "repetition_penalty");
    set_option_from_json_field(request.options, value, "do_sample", "do_sample");
    set_option_from_json_field(request.options, value, "num_beams", "num_beams");
    set_option_from_json_field(request.options, value, "guidance_scale", "guidance_scale");
    set_option_from_json_field(request.options, value, "num_inference_steps", "num_inference_steps");
    set_option_from_json_field(request.options, value, "text_chunk_size", "text_chunk_size");
    set_option_from_json_field(request.options, value, "audio_chunk_seconds", "audio_chunk_seconds");
    set_option_from_json_field(request.options, value, "audio_chunk_mode", "audio_chunk_mode");
    set_option_from_json_field(request.options, value, "return_timestamps", "return_timestamps");
    set_option_from_json_field(request.options, value, "use_prosody_code", "use_prosody_code");
    set_option_from_json_field(request.options, value, "predict_target_prosody", "predict_target_prosody");
    set_option_from_json_field(request.options, value, "use_pitch_shift", "use_pitch_shift");
    set_option_from_json_field(request.options, value, "source_shift_steps", "source_shift_steps");
    set_option_from_json_field(request.options, value, "prosody_shift_steps", "prosody_shift_steps");
    set_option_from_json_field(request.options, value, "style_shift_steps", "style_shift_steps");
    set_option_from_json_field(request.options, value, "target_duration_seconds", "target_duration_seconds");
    set_option_from_json_field(request.options, value, "reference_duration_seconds", "reference_duration_seconds");
    if (const auto reference_text = json_optional_string(value, "reference_text")) {
        set_option(request.options, "reference_text", *reference_text);
    }
    if (const auto instruct = json_optional_string(value, "instruct")) {
        set_option(request.options, "instruct", *instruct);
    }
    return request;
}

engine::runtime::TaskRequest build_request_from_cli(int argc, char ** argv) {
    engine::runtime::TaskRequest request;
    const auto language = find_arg(argc, argv, "--language").value_or("");
    if (const auto text = find_arg(argc, argv, "--text")) {
        request.text_input = engine::runtime::Transcript{*text, language};
    }
    if (const auto audio_path = find_arg(argc, argv, "--audio")) {
        request.audio_input = read_audio_buffer(std::filesystem::path(*audio_path));
    }
    engine::runtime::VoiceCondition voice;
    bool has_voice = false;
    if (const auto voice_id = find_arg(argc, argv, "--voice-id")) {
        engine::runtime::VoiceReference reference;
        reference.cached_voice_id = *voice_id;
        voice.speaker = std::move(reference);
        has_voice = true;
    }
    if (const auto voice_ref = find_arg(argc, argv, "--voice-ref")) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        voice.speaker->audio = read_audio_buffer(std::filesystem::path(*voice_ref));
        has_voice = true;
    }
    engine::runtime::StyleCondition style;
    if (const auto style_language = find_arg(argc, argv, "--style-language")) {
        style.language = *style_language;
        has_voice = true;
    }
    if (const auto emotion = find_arg(argc, argv, "--emotion")) {
        style.emotion = *emotion;
        has_voice = true;
    }
    if (const auto speaking_rate = parse_optional_float_arg(argc, argv, "--speaking-rate")) {
        style.speaking_rate = *speaking_rate;
        has_voice = true;
    }
    if (const auto pitch_shift = parse_optional_float_arg(argc, argv, "--pitch-shift")) {
        style.pitch_shift = *pitch_shift;
        has_voice = true;
    }
    if (const auto energy_scale = parse_optional_float_arg(argc, argv, "--energy-scale")) {
        style.energy_scale = *energy_scale;
        has_voice = true;
    }
    style.tags = collect_key_value_args(argc, argv, "--style-tag");
    if (!style.tags.empty()) {
        has_voice = true;
    }
    if (style.language.has_value() || style.emotion.has_value() || style.speaking_rate.has_value() ||
        style.pitch_shift.has_value() || style.energy_scale.has_value() || !style.tags.empty()) {
        voice.style = std::move(style);
    }
    if (has_voice) {
        request.voice = std::move(voice);
    }
    request.options = collect_key_value_args(argc, argv, "--request-option");
    if (!language.empty()) {
        set_option(request.options, "language", language);
    }
    if (const auto route = find_arg(argc, argv, "--task-route")) {
        set_option(request.options, "route", *route);
    }
    if (const auto source_audio = find_arg(argc, argv, "--source-audio")) {
        set_option(request.options, "source_audio", *source_audio);
    }
    if (const auto target_voice = find_arg(argc, argv, "--target-voice")) {
        set_option(request.options, "target_voice", *target_voice);
    }
    if (const auto prosody_ref = find_arg(argc, argv, "--prosody-ref")) {
        set_option(request.options, "prosody_ref", *prosody_ref);
    }
    if (const auto style_ref = find_arg(argc, argv, "--style-ref")) {
        set_option(request.options, "style_ref", *style_ref);
    }
    set_option_from_arg(argc, argv, "--target-text", "target_text", request.options);
    set_option_from_arg(argc, argv, "--style-ref-text", "style_ref_text", request.options);
    set_option_from_arg(argc, argv, "--lyrics", "lyrics", request.options);
    set_option_from_arg(argc, argv, "--track-name", "track_name", request.options);
    set_option_from_arg(argc, argv, "--speaker", "speaker", request.options);
    set_option_from_arg(argc, argv, "--duration-seconds", "duration_seconds", request.options);
    set_option_from_arg(argc, argv, "--repaint-start", "repainting_start", request.options);
    set_option_from_arg(argc, argv, "--repaint-end", "repainting_end", request.options);
    set_option_from_arg(argc, argv, "--repaint-mode", "repaint_mode", request.options);
    set_option_from_arg(argc, argv, "--repaint-strength", "repaint_strength", request.options);
    set_option_from_arg(argc, argv, "--seed", "seed", request.options);
    set_option_from_arg(argc, argv, "--max-tokens", "max_tokens", request.options);
    set_option_from_arg(argc, argv, "--max-steps", "max_steps", request.options);
    set_option_from_arg(argc, argv, "--temperature", "temperature", request.options);
    set_option_from_arg(argc, argv, "--top-k", "top_k", request.options);
    set_option_from_arg(argc, argv, "--top-p", "top_p", request.options);
    set_option_from_arg(argc, argv, "--repetition-penalty", "repetition_penalty", request.options);
    set_option_from_arg(argc, argv, "--do-sample", "do_sample", request.options);
    set_option_from_arg(argc, argv, "--num-beams", "num_beams", request.options);
    set_option_from_arg(argc, argv, "--guidance-scale", "guidance_scale", request.options);
    set_option_from_arg(argc, argv, "--num-inference-steps", "num_inference_steps", request.options);
    set_option_from_arg(argc, argv, "--text-chunk-size", "text_chunk_size", request.options);
    set_option_from_arg(argc, argv, "--audio-chunk-seconds", "audio_chunk_seconds", request.options);
    set_option_from_arg(argc, argv, "--audio-chunk-mode", "audio_chunk_mode", request.options);
    set_option_from_arg(argc, argv, "--use-prosody-code", "use_prosody_code", request.options);
    set_option_from_arg(argc, argv, "--predict-target-prosody", "predict_target_prosody", request.options);
    set_option_from_arg(argc, argv, "--use-pitch-shift", "use_pitch_shift", request.options);
    set_option_from_arg(argc, argv, "--source-shift-steps", "source_shift_steps", request.options);
    set_option_from_arg(argc, argv, "--prosody-shift-steps", "prosody_shift_steps", request.options);
    set_option_from_arg(argc, argv, "--style-shift-steps", "style_shift_steps", request.options);
    set_option_from_arg(argc, argv, "--target-duration-seconds", "target_duration_seconds", request.options);
    set_option_from_arg(argc, argv, "--reference-duration-seconds", "reference_duration_seconds", request.options);
    if (const auto reference_text = find_arg(argc, argv, "--reference-text")) {
        set_option(request.options, "reference_text", *reference_text);
    }
    if (const auto instruct = find_arg(argc, argv, "--instruct")) {
        set_option(request.options, "instruct", *instruct);
    }
    return request;
}

}  // namespace minitts::cli
