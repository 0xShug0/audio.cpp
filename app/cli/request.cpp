#include "request.h"

#include "args.h"

#include "../common/base64.h"

#include "engine/framework/audio/wav_reader.h"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace minitts::cli {

using engine::runtime::InvalidRequestError;

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

struct ArtifactKindName {
    const char * name;
    engine::runtime::ArtifactKind kind;
};

// The names the server writes a result artifact's kind with.
constexpr ArtifactKindName kArtifactKinds[] = {
    {"speaker_embedding", engine::runtime::ArtifactKind::SpeakerEmbedding},
    {"style_embedding", engine::runtime::ArtifactKind::StyleEmbedding},
    {"prompt_embedding", engine::runtime::ArtifactKind::PromptEmbedding},
    {"acoustic_tokens", engine::runtime::ArtifactKind::AcousticTokens},
    {"midi", engine::runtime::ArtifactKind::Midi},
    {"transcript_alignment", engine::runtime::ArtifactKind::TranscriptAlignment},
    {"diarization_state", engine::runtime::ArtifactKind::DiarizationState},
    {"vad_state", engine::runtime::ArtifactKind::VadState},
    {"custom", engine::runtime::ArtifactKind::Custom},
};

constexpr bool artifact_kinds_in_enum_order() {
    for (size_t index = 0; index < std::size(kArtifactKinds); ++index) {
        if (kArtifactKinds[index].kind != static_cast<engine::runtime::ArtifactKind>(index)) {
            return false;
        }
    }
    return true;
}

// Custom is the last ArtifactKind, so a kind added to the enum without a name
// here fails to build instead of being turned away as unknown.
static_assert(
    std::size(kArtifactKinds) == static_cast<size_t>(engine::runtime::ArtifactKind::Custom) + 1 &&
        artifact_kinds_in_enum_order(),
    "kArtifactKinds must name every ArtifactKind, in enum order");

bool is_artifact_field(const std::string & key) {
    for (const char * field : {"id", "kind", "payload", "path", "meta"}) {
        if (key == field) {
            return true;
        }
    }
    return false;
}

engine::runtime::ArtifactKind parse_artifact_kind(const std::string & name, const std::string & where) {
    for (const auto & entry : kArtifactKinds) {
        if (name == entry.name) {
            return entry.kind;
        }
    }
    std::string names;
    for (const auto & entry : kArtifactKinds) {
        names += (names.empty() ? "" : ", ") + std::string(entry.name);
    }
    throw InvalidRequestError(where + ": unknown kind '" + name + "', expected one of " + names);
}

// A field that is absent or null reads as not given.
const engine::io::json::Value * artifact_field(
    const engine::io::json::Value & entry,
    const std::string & key) {
    const auto * value = entry.find(key);
    return value == nullptr || value->is_null() ? nullptr : value;
}

std::string artifact_string(
    const engine::io::json::Value & entry,
    const std::string & key,
    const std::string & where) {
    const auto * value = artifact_field(entry, key);
    if (value == nullptr || !value->is_string() || value->as_string().empty()) {
        throw InvalidRequestError(where + ": " + key + " must be a non-empty string");
    }
    return value->as_string();
}

[[noreturn]] void throw_artifact_limit(const std::string & where) {
    throw InvalidRequestError(
        where + ": the request's artifact payloads total more than " +
        std::to_string(kMaxRequestArtifactBytes) + " bytes");
}

std::vector<std::byte> decode_artifact_payload(
    const engine::io::json::Value & payload,
    uint64_t budget,
    const std::string & where) {
    if (!payload.is_string()) {
        throw InvalidRequestError(where + ": payload must be a base64 string");
    }
    // An inline payload is already in memory as part of the request, so the
    // server's request body limit bounds this decode.
    std::vector<std::byte> bytes;
    try {
        bytes = minitts::app::base64_decode_bytes(payload.as_string());
    } catch (const std::runtime_error & error) {
        throw InvalidRequestError(where + ": payload is not valid base64 (" + error.what() + ")");
    }
    if (bytes.size() > budget) {
        throw_artifact_limit(where);
    }
    return bytes;
}

std::vector<std::byte> read_artifact_file(
    const std::filesystem::path & path,
    uint64_t budget,
    const std::string & where) {
    // Regular files only: a FIFO or a device such as /dev/zero has no size to
    // check and may never end.
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        throw InvalidRequestError(where + ": path is not a regular file: " + path.string());
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        throw InvalidRequestError(where + ": cannot read the size of " + path.string() + ": " + error.message());
    }
    if (static_cast<uint64_t>(size) > budget) {
        throw_artifact_limit(where);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw InvalidRequestError(where + ": failed to open " + path.string());
    }
    std::vector<std::byte> bytes(static_cast<size_t>(size));
    if (!bytes.empty() &&
        !input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw InvalidRequestError(where + ": failed to read " + path.string());
    }
    return bytes;
}

std::unordered_map<std::string, std::string> artifact_meta(
    const engine::io::json::Value & entry,
    const std::string & where) {
    std::unordered_map<std::string, std::string> meta;
    const auto * value = artifact_field(entry, "meta");
    if (value == nullptr) {
        return meta;
    }
    if (!value->is_object()) {
        throw InvalidRequestError(where + ": meta must be an object");
    }
    for (const auto & [key, child] : value->as_object()) {
        if (!child.is_string() && !child.is_number() && !child.is_bool()) {
            throw InvalidRequestError(where + ": meta." + key + " must be a string, number or boolean");
        }
        meta[key] = json_option_string(child);
    }
    return meta;
}

}  // namespace

bool is_stdin_audio_source(std::string_view audio_arg) {
    return audio_arg == "-";
}

engine::runtime::AudioBuffer read_audio_buffer(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    return engine::runtime::AudioBuffer{
        wav.sample_rate,
        wav.channels,
        wav.samples,
    };
}

engine::runtime::AudioBuffer read_audio_buffer(std::istream & input) {
    const auto wav = engine::audio::read_wav_f32(input);
    return engine::runtime::AudioBuffer{
        wav.sample_rate,
        wav.channels,
        wav.samples,
    };
}

engine::runtime::AudioBuffer read_audio_buffer(std::string_view input) {
    const auto wav = engine::audio::read_wav_f32(input);
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
        const double number = value.as_number();
        constexpr double kMaxExactJsonInteger = 9007199254740992.0;  // 2^53
        if (std::isfinite(number) && std::trunc(number) == number && std::fabs(number) < kMaxExactJsonInteger) {
            std::ostringstream out;
            out << std::fixed << std::setprecision(0) << number;
            return out.str();
        }
        return engine::io::json::stringify_number(number);
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

std::vector<engine::runtime::VoiceArtifact> json_request_artifacts(
    const engine::io::json::Value * value,
    const std::filesystem::path & base_dir) {
    std::vector<engine::runtime::VoiceArtifact> artifacts;
    if (value == nullptr || value->is_null()) {
        return artifacts;
    }
    if (!value->is_array()) {
        throw InvalidRequestError("artifacts must be an array of artifact objects");
    }
    uint64_t total_bytes = 0;
    const auto & entries = value->as_array();
    for (size_t index = 0; index < entries.size(); ++index) {
        const auto & entry = entries[index];
        std::string where = "artifacts[" + std::to_string(index) + "]";
        if (!entry.is_object()) {
            throw InvalidRequestError(where + " must be an object with id, kind, and payload or path");
        }
        for (const auto & field : entry.as_object()) {
            if (!is_artifact_field(field.first)) {
                throw InvalidRequestError(
                    where + ": unknown field '" + field.first + "', an artifact has id, kind, payload or path, and meta");
            }
        }
        engine::runtime::VoiceArtifact artifact;
        artifact.id = artifact_string(entry, "id", where);
        where += " (" + artifact.id + ")";
        const auto * kind = artifact_field(entry, "kind");
        if (kind == nullptr || !kind->is_string()) {
            throw InvalidRequestError(where + ": kind must be a string");
        }
        artifact.kind = parse_artifact_kind(kind->as_string(), where);
        const auto * payload = artifact_field(entry, "payload");
        const bool has_path = artifact_field(entry, "path") != nullptr;
        if ((payload != nullptr) == has_path) {
            throw InvalidRequestError(
                where + ": give exactly one of payload (base64) and path (a file holding the payload)");
        }
        const uint64_t budget = kMaxRequestArtifactBytes - total_bytes;
        artifact.payload = payload != nullptr
            ? decode_artifact_payload(*payload, budget, where)
            : read_artifact_file(resolve_case_path(base_dir, artifact_string(entry, "path", where)), budget, where);
        total_bytes += artifact.payload.size();
        artifact.meta = artifact_meta(entry, where);
        artifacts.push_back(std::move(artifact));
    }
    return artifacts;
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
    request.input_artifacts = json_request_artifacts(value.find("artifacts"), base_dir);

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
    set_option_from_json_field(request.options, value, "text_chunk_mode", "text_chunk_mode");
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
    // Read outside the branch below: only a live stdin source uses them, but an option the CLI
    // never looks up cannot be told apart from a misspelling.
    const int input_rate = parse_int_arg(argc, argv, "--input-rate", 16000);
    const int input_channels = parse_int_arg(argc, argv, "--input-channels", 1);
    if (const auto audio_path = find_arg(argc, argv, "--audio")) {
        if (is_stdin_audio_source(*audio_path)) {
            // Live PCM arrives chunk by chunk, so only the format contract is known up front.
            // The samples stay empty; the streaming driver pulls them from stdin instead.
            request.audio_input = engine::runtime::AudioBuffer{input_rate, input_channels, {}};
        } else {
            request.audio_input = read_audio_buffer(std::filesystem::path(*audio_path));
        }
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
    set_option_from_arg(argc, argv, "--text-chunk-mode", "text_chunk_mode", request.options);
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
