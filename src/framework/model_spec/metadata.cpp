#include "engine/framework/model_spec/metadata.h"

#include "engine/framework/model_spec/options.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::model_spec {
namespace {

namespace json = engine::io::json;

runtime::VoiceTaskKind parse_task_kind(const std::string & value) {
    if (value == "vad") {
        return runtime::VoiceTaskKind::Vad;
    }
    if (value == "asr") {
        return runtime::VoiceTaskKind::Asr;
    }
    if (value == "diar") {
        return runtime::VoiceTaskKind::Diarization;
    }
    if (value == "sep") {
        return runtime::VoiceTaskKind::SourceSeparation;
    }
    if (value == "audio_generation" || value == "music" || value == "sfx" || value == "edit") {
        return runtime::VoiceTaskKind::AudioGeneration;
    }
    if (value == "tts") {
        return runtime::VoiceTaskKind::Tts;
    }
    if (value == "clone") {
        return runtime::VoiceTaskKind::VoiceCloning;
    }
    if (value == "vc") {
        return runtime::VoiceTaskKind::VoiceConversion;
    }
    if (value == "s2s") {
        return runtime::VoiceTaskKind::SpeechToSpeech;
    }
    if (value == "align") {
        return runtime::VoiceTaskKind::Alignment;
    }
    if (value == "design") {
        return runtime::VoiceTaskKind::VoiceDesign;
    }
    if (value == "speaker") {
        return runtime::VoiceTaskKind::SpeakerRecognition;
    }
    if (value == "svc") {
        return runtime::VoiceTaskKind::Svc;
    }
    throw std::runtime_error("unknown model spec task: " + value);
}

runtime::RunMode parse_run_mode(const std::string & value) {
    if (value == "offline") {
        return runtime::RunMode::Offline;
    }
    if (value == "streaming") {
        return runtime::RunMode::Streaming;
    }
    throw std::runtime_error("unknown model spec run mode: " + value);
}

std::vector<runtime::RunMode> parse_run_modes(const json::Value & value) {
    std::vector<runtime::RunMode> modes;
    for (const auto & item : value.as_array()) {
        modes.push_back(parse_run_mode(item.as_string()));
    }
    return modes;
}

std::vector<runtime::TaskCapability> parse_tasks(const json::Value & tasks_value, const json::Value & modes_value) {
    const auto modes = parse_run_modes(modes_value);
    std::vector<runtime::TaskCapability> tasks;
    for (const auto & item : tasks_value.as_array()) {
        runtime::TaskCapability task;
        task.task = parse_task_kind(item.as_string());
        task.modes = modes;
        tasks.push_back(std::move(task));
    }
    return tasks;
}

json::Value load_spec_for_family(std::string_view family) {
    return engine::model_spec::load_spec(engine::model_spec::default_spec_path(family));
}

std::string join_values(const std::vector<std::string> & values) {
    std::string out;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out += "|";
        }
        out += values[index];
    }
    return out;
}

std::vector<runtime::CliOptionInfo> parse_cli_options(const json::Value * value, std::string_view family_prefix = {}) {
    std::vector<runtime::CliOptionInfo> options;
    if (value == nullptr || value->is_null()) {
        return options;
    }
    for (const auto & item : value->as_array()) {
        runtime::CliOptionInfo option;
        option.name = json::require_string(item, "name");
        if (!family_prefix.empty()) {
            option.name = std::string(family_prefix) + "." + option.name;
        }
        const auto option_type = json::require_string(item, "type");
        if (option_type == "enum") {
            if (const auto * preset = item.find("preset")) {
                option.value_name = join_values(require_option_preset(preset->as_string()));
            } else {
                option.value_name = join_values(json::require_string_array(item, "values"));
            }
        } else {
            option.value_name = option_type;
        }
        option.description = json::require_string(item, "description");
        option.required = json::require_bool(item, "required");
        if (const auto * default_value = item.find("default")) {
            option.default_value = json::stringify(*default_value);
        }
        if (const auto * min_value = item.find("min")) {
            option.min_value = json::stringify(*min_value);
        }
        if (const auto * max_value = item.find("max")) {
            option.max_value = json::stringify(*max_value);
        }
        options.push_back(std::move(option));
    }
    return options;
}

std::vector<ModelDependencyCondition> parse_dependency_conditions(const json::Value * value) {
    std::vector<ModelDependencyCondition> out;
    if (value == nullptr || value->is_null()) {
        return out;
    }
    for (const auto & item : value->as_array()) {
        ModelDependencyCondition condition;
        condition.scope = json::require_string(item, "scope");
        condition.option_key = json::require_string(item, "option_key");
        const auto & equals = item.require("equals");
        if (equals.is_bool()) {
            condition.equals_type = ModelSpecValueType::Bool;
            condition.equals_bool = equals.as_bool();
        } else if (equals.is_number()) {
            condition.equals_type = ModelSpecValueType::Number;
            condition.equals_number = equals.as_number();
        } else {
            condition.equals_type = ModelSpecValueType::String;
            condition.equals_string = equals.as_string();
        }
        out.push_back(std::move(condition));
    }
    return out;
}

}  // namespace

std::optional<runtime::CapabilitySet> advertised_capabilities(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    const auto * capabilities = spec.find("capabilities");
    if (capabilities == nullptr || capabilities->is_null()) {
        return std::nullopt;
    }
    runtime::CapabilitySet out;
    out.supported_tasks = parse_tasks(spec.require("tasks"), spec.require("modes"));
    out.languages = json::optional_string_array(*capabilities, "languages");
    out.supports_speaker_reference = json::optional_bool(*capabilities, "speaker_reference", false);
    out.supports_style_condition = json::optional_bool(*capabilities, "style_condition", false);
    out.supports_timestamps = json::optional_bool(*capabilities, "timestamps", false);
    return out;
}

std::optional<runtime::ModelCliInterface> cli_interface(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    const auto * options = spec.find("options");
    if (options == nullptr || options->is_null()) {
        return std::nullopt;
    }
    runtime::ModelCliInterface out;
    const auto family_string = std::string(family);
    out.request_options = parse_cli_options(options->find("request"));
    out.session_options = parse_cli_options(options->find("session"), family_string);
    out.load_options = parse_cli_options(options->find("load"), family_string);
    return out;
}

std::vector<ModelDependency> dependencies(std::string_view family) {
    const auto family_string = std::string(family);
    const auto spec = load_spec_for_family(family);
    const auto * rows = spec.find("dependencies");
    if (rows == nullptr || rows->is_null()) {
        return {};
    }
    std::vector<ModelDependency> out;
    for (const auto & item : rows->as_array()) {
        ModelDependency dependency;
        dependency.kind = json::require_string(item, "kind");
        dependency.family = json::require_string(item, "family");
        dependency.scope = json::require_string(item, "scope");
        dependency.option = json::require_string(item, "option");
        dependency.option_key = family_string + "." + dependency.option;
        dependency.required = json::require_bool(item, "required");
        dependency.required_when = parse_dependency_conditions(item.find("required_when"));
        if (const auto * path = item.find("path")) {
            dependency.path = path->as_string();
        }
        out.push_back(std::move(dependency));
    }
    return out;
}

}  // namespace engine::model_spec
