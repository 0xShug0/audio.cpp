#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/task_vocabulary.h"

#include "engine/framework/model_spec/options.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/model_spec/schema.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::model_spec {
namespace {

namespace json = engine::io::json;

runtime::VoiceTaskKind parse_task_kind(const std::string & value) {
    // Spec names, not ABI tokens: "music" here is "gen" there. The mapping is
    // in runtime::task_vocabulary, which the schema's allowed set reads too, so
    // a name the schema accepts cannot be one this rejects.
    const auto token = runtime::task_token_for_spec_name(value);
    if (!token.empty()) {
        return runtime::parse_voice_task_kind(std::string(token));
    }
    std::string expected;
    std::size_t count = 0;
    const auto * entries = runtime::task_vocabulary(count);
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t alias = 0; alias < entries[i].alias_count; ++alias) {
            if (!expected.empty()) {
                expected += ", ";
            }
            expected.append(entries[i].aliases[alias]);
        }
    }
    throw std::runtime_error("unknown model spec task: " + value + " (expected one of " + expected + ")");
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
    return engine::model_spec::load_spec(engine::model_spec::default_contract_spec_path(family));
}

json::Value load_contract_spec_for_family(std::string_view family) {
    return engine::model_spec::load_contract_spec(engine::model_spec::default_contract_spec_path(family));
}

std::string ref_candidate_path(const std::string & ref) {
    const auto split = ref.find(':');
    if (split == std::string::npos) {
        return ref;
    }
    return ref.substr(split + 1);
}

std::string resource_candidate_path(const json::Value & ref) {
    if (ref.is_string()) {
        return ref_candidate_path(ref.as_string());
    }
    if (const auto * source = ref.find("source")) {
        return ref_candidate_path(source->as_string());
    }
    return {};
}

void push_unique_candidate(std::vector<std::string> & out, const std::string & candidate) {
    if (candidate.empty()) {
        return;
    }
    if (std::find(out.begin(), out.end(), candidate) == out.end()) {
        out.push_back(candidate);
    }
}

void append_resource_candidates(
    std::vector<std::string> & out,
    const json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    std::vector<std::string> candidates;
    candidates.reserve(map_value->as_object().size());
    for (const auto & [_, ref] : map_value->as_object()) {
        candidates.push_back(resource_candidate_path(ref));
    }
    std::sort(candidates.begin(), candidates.end());
    for (const auto & candidate : candidates) {
        push_unique_candidate(out, candidate);
    }
}

void append_source_candidates(
    std::vector<std::string> & config_candidates,
    std::vector<std::string> & weight_candidates,
    const json::Value & source) {
    append_resource_candidates(config_candidates, source.find("files"));
    append_resource_candidates(config_candidates, source.find("optional_files"));
    append_resource_candidates(weight_candidates, source.find("tensors"));
}

void append_package_candidates(std::vector<std::string> & weight_candidates, const json::Value * packages) {
    if (packages == nullptr || packages->is_null()) {
        return;
    }
    for (const auto & package : packages->as_array()) {
        for (const auto & file : package.require("files").as_array()) {
            push_unique_candidate(weight_candidates, file.as_string());
        }
    }
}

bool has_capability(const json::Value & capabilities, std::string_view capability) {
    const std::string capability_string(capability);
    for (const auto & [_, task_capabilities] : capabilities.as_object()) {
        for (const auto & item : task_capabilities.as_array()) {
            if (item.as_string() == capability_string) {
                return true;
            }
        }
    }
    return false;
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

runtime::CapabilitySet capabilities_from_spec(const json::Value & spec) {
    runtime::CapabilitySet out;
    const auto * capabilities = spec.find("capabilities");
    if (capabilities == nullptr || capabilities->is_null()) {
        throw std::runtime_error("model spec has no capabilities contract");
    }
    out.supported_tasks = parse_tasks(spec.require("tasks"), spec.require("modes"));
    out.languages = json::optional_string_array(spec, "languages");
    out.supports_speaker_reference = has_capability(*capabilities, "speaker_reference");
    out.supports_style_condition =
        has_capability(*capabilities, "style_control") || has_capability(*capabilities, "emotion_control");
    out.supports_timestamps =
        has_capability(*capabilities, "word_timestamps") || has_capability(*capabilities, "segments") ||
        has_capability(*capabilities, "speech_segments");
    return out;
}

runtime::ModelMetadata metadata_from_spec(const json::Value & spec) {
    runtime::ModelMetadata out;
    out.family = json::require_string(spec, "family");
    out.variant = json::require_string(spec, "display_name");
    out.description = json::require_string(spec, "description");
    append_package_candidates(out.weight_candidates, spec.find("packages"));
    if (const auto * sources = spec.find("sources")) {
        for (const auto & source : sources->as_array()) {
            append_source_candidates(out.config_candidates, out.weight_candidates, source);
        }
    }
    return out;
}

runtime::ModelCliInterface cli_from_spec(const json::Value & spec) {
    const auto * options = spec.find("options");
    if (options == nullptr || options->is_null()) {
        throw std::runtime_error("model spec has no options contract");
    }
    runtime::ModelCliInterface out;
    const auto family = json::require_string(spec, "family");
    out.request_options = parse_cli_options(options->find("request"));
    out.session_options = parse_cli_options(options->find("session"), family);
    out.load_options = parse_cli_options(options->find("load"), family);
    return out;
}

std::unordered_set<std::string> option_keys(const std::vector<runtime::CliOptionInfo> & options) {
    std::unordered_set<std::string> keys;
    keys.reserve(options.size());
    for (const auto & option : options) {
        keys.insert(option.name);
    }
    return keys;
}

void add_alias_keys(ModelContract & contract, const json::Value & spec);

ModelContract contract_from_spec(const json::Value & spec) {
    ModelContract out;
    out.metadata.family = json::require_string(spec, "family");
    out.metadata.variant = json::require_string(spec, "display_name");
    out.metadata.description = json::require_string(spec, "description");
    out.capabilities = capabilities_from_spec(spec);
    out.cli = cli_from_spec(spec);
    out.request_option_keys = option_keys(out.cli.request_options);
    out.session_option_keys = option_keys(out.cli.session_options);
    out.load_option_keys = option_keys(out.cli.load_options);
    add_alias_keys(out, spec);
    return out;
}

json::Value expand_preset(const json::Value & row);

json::Value wire_form_definition(const std::string & form) {
    if (form == "path_string") {
        return json::parse(R"({"kind":"string","meaning":"filesystem path","media_type":"audio/wav"})");
    }
    if (form == "voice_ref_object") {
        return json::parse(R"({
          "kind": "object",
          "type_field": "type",
          "type_values": ["path", "base64"],
          "path_field": "path",
          "data_field": "data",
          "max_decoded_bytes": 5242880,
          "media_type": "audio/wav"
        })");
    }
    return json::Value::make_null();
}

json::Value expand_input_schema(json::Value schema) {
    schema = expand_preset(schema);
    if (!schema.is_object()) {
        return schema;
    }
    auto object = schema.as_object();
    const auto wire_it = object.find("wire");
    if (wire_it == object.end() || !wire_it->second.is_array() || object.find("wire_forms") != object.end()) {
        return json::Value::make_object(std::move(object));
    }
    json::Value::Object forms;
    for (const auto & form : wire_it->second.as_array()) {
        if (!form.is_string()) {
            continue;
        }
        const auto name = form.as_string();
        auto definition = wire_form_definition(name);
        if (!definition.is_null()) {
            forms.emplace(name, std::move(definition));
        }
    }
    if (!forms.empty()) {
        object.emplace("wire_forms", json::Value::make_object(std::move(forms)));
    }
    return json::Value::make_object(std::move(object));
}

json::Value::Array instruction_fields(const json::Value::Array & inputs) {
    json::Value::Array fields;
    for (const auto & input : inputs) {
        if (!input.is_object()) {
            continue;
        }
        const auto id_it = input.as_object().find("id");
        if (id_it == input.as_object().end() || !id_it->second.is_string() || id_it->second.as_string() != "instructions") {
            continue;
        }
        json::Value::Object field;
        field["id"] = id_it->second;
        if (const auto tasks_it = input.as_object().find("tasks");
            tasks_it != input.as_object().end()) {
            field["tasks"] = tasks_it->second;
        }
        const auto schema_it = input.as_object().find("schema");
        if (schema_it != input.as_object().end() && schema_it->second.is_object()) {
            const auto & schema = schema_it->second.as_object();
            if (const auto text_it = schema.find("text"); text_it != schema.end() && text_it->second.is_object()) {
                const auto & text = text_it->second.as_object();
                for (const std::string key : {"caller", "engine_prefix", "engine_suffix", "engine_boundary"}) {
                    if (const auto item = text.find(key); item != text.end()) {
                        field.emplace(key, item->second);
                    }
                }
            } else if (const auto type_it = schema.find("type");
                       type_it != schema.end() && type_it->second.is_string() && type_it->second.as_string() == "enum") {
                field["caller"] = json::Value::make_string("enum");
                if (const auto values_it = schema.find("values"); values_it != schema.end()) {
                    field["values"] = values_it->second;
                }
            }
        }
        fields.push_back(json::Value::make_object(std::move(field)));
    }
    return fields;
}

void add_alias_keys(ModelContract & contract, const json::Value & spec) {
    const auto * options = spec.find("options");
    if (options == nullptr || !options->is_object()) {
        return;
    }
    for (const std::string scope : {"request", "session", "load"}) {
        const auto * rows = options->find(scope);
        if (rows == nullptr || !rows->is_array()) {
            continue;
        }
        for (const auto & row : rows->as_array()) {
            if (!row.is_object()) {
                continue;
            }
            const auto * aliases = row.find("aliases");
            if (aliases == nullptr || !aliases->is_array()) {
                continue;
            }
            for (const auto & alias : aliases->as_array()) {
                if (!alias.is_string()) {
                    continue;
                }
                const auto key = scope == "request" ? alias.as_string()
                                                    : contract.metadata.family + "." + alias.as_string();
                if (scope == "request") {
                    contract.request_option_keys.insert(key);
                } else if (scope == "session") {
                    contract.session_option_keys.insert(key);
                } else {
                    contract.load_option_keys.insert(key);
                }
            }
        }
    }
}

json::Value expand_preset(const json::Value & row) {
    if (!row.is_object()) {
        return row;
    }
    auto object = row.as_object();
    const auto preset_it = object.find("preset");
    // Unknown or non-string presets stay as written. Typed specs already reject
    // them in validate_spec; legacy specs must keep the previous pass-through.
    if (preset_it != object.end() && preset_it->second.is_string() && object.find("values") == object.end()) {
        const auto known = option_presets().find(preset_it->second.as_string());
        if (known != option_presets().end()) {
            json::Value::Array values;
            for (const auto & value : known->second) {
                values.push_back(json::Value::make_string(value));
            }
            object.emplace("values", json::Value::make_array(std::move(values)));
        }
    }
    return json::Value::make_object(std::move(object));
}

json::Value::Object startup_surface(const json::Value & default_task, const json::Value & default_mode) {
    json::Value::Object startup;
    startup["default_task"] = default_task;
    const auto token = runtime::task_token_for_spec_name(default_task.as_string());
    startup["default_task_token"] = json::Value::make_string(std::string(token));
    startup["default_mode"] = default_mode;
    // CLI --task and server config "task" take task_tokens[spec task], not the spec name.
    startup["task_value"] = json::Value::make_string("task_tokens[<spec task>]");
    startup["option_assignment"] = json::Value::make_string("public_key=value");
    startup["server_option_key"] = json::Value::make_string("public_key");
    startup["package_tasks"] = json::Value::make_string("packages[].tasks");
    startup["package_default_task"] = json::Value::make_string("packages[].default_task");
    // A directory is the model root. A file path uses that file's parent. A directory
    // with more than one weight file must be passed as the file, not the directory.
    startup["model_path"] = json::Value::make_string(
        "directory, or a file in that directory when it contains more than one weight");
    startup["server_id"] = json::Value::make_string("caller-chosen");
    json::Value::Object cli;
    cli["family"] = json::Value::make_string("--family");
    cli["model"] = json::Value::make_string("--model");
    cli["task"] = json::Value::make_string("--task");
    cli["mode"] = json::Value::make_string("--mode");
    cli["load_option"] = json::Value::make_string("--load-option");
    cli["session_option"] = json::Value::make_string("--session-option");
    cli["request_option"] = json::Value::make_string("--request-option");
    startup["cli"] = json::Value::make_object(std::move(cli));
    json::Value::Object server_config;
    server_config["id"] = json::Value::make_string("id");
    server_config["family"] = json::Value::make_string("family");
    server_config["path"] = json::Value::make_string("path");
    server_config["task"] = json::Value::make_string("task");
    server_config["mode"] = json::Value::make_string("mode");
    server_config["load_options"] = json::Value::make_string("load_options");
    server_config["session_options"] = json::Value::make_string("session_options");
    startup["server_config"] = json::Value::make_object(std::move(server_config));
    return startup;
}

json::Value resolve_spec(json::Value spec) {
    auto root = spec.as_object();
    const auto family_it = root.find("family");
    const std::string family =
        family_it != root.end() && family_it->second.is_string() ? family_it->second.as_string() : std::string();
    if (const auto options_it = root.find("options");
        options_it != root.end() && options_it->second.is_object()) {
        auto options = options_it->second.as_object();
        for (const std::string scope : {"request", "session", "load"}) {
            const auto rows_it = options.find(scope);
            if (rows_it == options.end() || !rows_it->second.is_array()) {
                continue;
            }
            json::Value::Array rows;
            rows.reserve(rows_it->second.as_array().size());
            for (const auto & row : rows_it->second.as_array()) {
                auto expanded = expand_preset(row);
                if (expanded.is_object()) {
                    auto object = expanded.as_object();
                    if (const auto name_it = object.find("name");
                        name_it != object.end() && name_it->second.is_string()) {
                        const auto name = name_it->second.as_string();
                        object["public_key"] = json::Value::make_string(
                            scope == "request" || family.empty() ? name : family + "." + name);
                    }
                    rows.push_back(json::Value::make_object(std::move(object)));
                } else {
                    rows.push_back(std::move(expanded));
                }
            }
            options[scope] = json::Value::make_array(std::move(rows));
        }
        root["options"] = json::Value::make_object(std::move(options));
    }
    if (const auto inputs_it = root.find("inputs");
        inputs_it != root.end() && inputs_it->second.is_array()) {
        json::Value::Array inputs;
        inputs.reserve(inputs_it->second.as_array().size());
        for (const auto & input : inputs_it->second.as_array()) {
            if (!input.is_object()) {
                inputs.push_back(input);
                continue;
            }
            auto input_object = input.as_object();
            const auto schema_it = input_object.find("schema");
            if (schema_it != input_object.end()) {
                input_object["schema"] = expand_input_schema(schema_it->second);
            }
            inputs.push_back(json::Value::make_object(std::move(input_object)));
        }
        root["inputs"] = json::Value::make_array(std::move(inputs));
    }
    const auto version_it = root.find("schema_version");
    if (version_it != root.end() && version_it->second.is_number() &&
        version_it->second.as_number() == kModelSpecSchemaVersion) {
        json::Value::Array inputs;
        if (const auto inputs_it = root.find("inputs");
            inputs_it != root.end() && inputs_it->second.is_array()) {
            inputs = inputs_it->second.as_array();
        }
        root["instruction_fields"] = json::Value::make_array(instruction_fields(inputs));
    }
    json::Value::Object default_download;
    if (const auto defaults_it = root.find("package_defaults");
        defaults_it != root.end() && defaults_it->second.is_object()) {
        if (const auto * download = defaults_it->second.find("download");
            download != nullptr && download->is_object()) {
            default_download = download->as_object();
        }
    }
    if (const auto packages_it = root.find("packages");
        packages_it != root.end() && packages_it->second.is_array()) {
        json::Value::Array packages;
        packages.reserve(packages_it->second.as_array().size());
        for (const auto & package : packages_it->second.as_array()) {
            if (!package.is_object()) {
                packages.push_back(package);
                continue;
            }
            auto package_object = package.as_object();
            auto download = default_download;
            if (const auto package_download_it = package_object.find("download");
                package_download_it != package_object.end() && package_download_it->second.is_object()) {
                for (const auto & [key, value] : package_download_it->second.as_object()) {
                    download[key] = value;
                }
            }
            if (!download.empty()) {
                package_object["download"] = json::Value::make_object(std::move(download));
            }
            packages.push_back(json::Value::make_object(std::move(package_object)));
        }
        root["packages"] = json::Value::make_array(std::move(packages));
    }
    if (const auto tasks_it = root.find("tasks");
        tasks_it != root.end() && tasks_it->second.is_array()) {
        json::Value::Object tokens;
        for (const auto & task : tasks_it->second.as_array()) {
            if (!task.is_string()) {
                continue;
            }
            const auto name = task.as_string();
            const auto token = runtime::task_token_for_spec_name(name);
            if (!token.empty()) {
                tokens.emplace(name, json::Value::make_string(std::string(token)));
            }
        }
        root["task_tokens"] = json::Value::make_object(std::move(tokens));
    }
    if (const auto operations_it = root.find("task_operations");
        operations_it != root.end() && operations_it->second.is_object()) {
        json::Value::Object filled;
        for (const auto & [task, row] : operations_it->second.as_object()) {
            if (!row.is_object()) {
                filled.emplace(task, row);
                continue;
            }
            auto object = row.as_object();
            const auto preferred_it = object.find("preferred_operation");
            if (preferred_it != object.end() && preferred_it->second.is_string()) {
                const auto operation = preferred_it->second.as_string();
                if (const auto surface = operation_surface(operation)) {
                    for (const auto & [key, value] : surface->as_object()) {
                        if (object.find(key) == object.end()) {
                            object.emplace(key, value);
                        }
                    }
                }
                if (const auto stream_it = object.find("stream");
                    stream_it != object.end() && stream_it->second.is_bool() && stream_it->second.as_bool()) {
                    if (const auto stream = stream_response_surface(operation)) {
                        object["stream_response"] = *stream;
                    }
                }
            }
            filled.emplace(task, json::Value::make_object(std::move(object)));
        }
        root["task_operations"] = json::Value::make_object(std::move(filled));
    }
    const auto default_task_it = root.find("default_task");
    const auto default_mode_it = root.find("default_mode");
    if (default_task_it != root.end() && default_task_it->second.is_string() &&
        default_mode_it != root.end() && default_mode_it->second.is_string()) {
        root["startup"] = json::Value::make_object(
            startup_surface(default_task_it->second, default_mode_it->second));
    }
    return json::Value::make_object(std::move(root));
}

}  // namespace

std::optional<ModelContract> model_contract(std::string_view family) {
    const auto spec = load_contract_spec_for_family(family);
    const auto * version = spec.find("schema_version");
    if (version == nullptr) {
        return std::nullopt;
    }
    const bool version_1 =
        version->is_number() && version->as_number() == kModelSpecSchemaVersionV1;
    const bool version_2 =
        version->is_number() && version->as_number() == kModelSpecSchemaVersion;
    if (!version_1 && !version_2) {
        throw std::runtime_error("model spec schema_version: expected numeric 1 or 2");
    }
    return contract_from_spec(spec);
}

std::optional<ModelContract> find_model_contract(std::string_view family) {
    if (!engine::model_spec::find_contract_spec_path(family).has_value()) {
        return std::nullopt;
    }
    return model_contract(family);
}

std::optional<runtime::CapabilitySet> advertised_capabilities(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    if (spec.find("schema_version") == nullptr || spec.find("capabilities") == nullptr) {
        return std::nullopt;
    }
    return capabilities_from_spec(spec);
}

std::optional<runtime::ModelMetadata> model_metadata(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    if (spec.find("schema_version") == nullptr) {
        return std::nullopt;
    }
    return metadata_from_spec(spec);
}

std::optional<runtime::ModelCliInterface> cli_interface(std::string_view family) {
    const auto spec = load_spec_for_family(family);
    if (spec.find("schema_version") == nullptr || spec.find("options") == nullptr) {
        return std::nullopt;
    }
    return cli_from_spec(spec);
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

engine::io::json::Value resolved_spec(std::string_view family) {
    return resolve_spec(
        engine::model_spec::load_spec(engine::model_spec::default_contract_spec_path(family)));
}

}  // namespace engine::model_spec
