#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

namespace json = engine::io::json;

struct PackageSpecCandidate {
    engine::assets::GgufEmbeddedModelSpec spec;
    std::string origin;
    int priority = 0;
};

const std::unordered_map<std::string, std::string_view> & converter_model_specs() {
    static const std::unordered_map<std::string, std::string_view> specs = {
#include "gguf_converter_model_specs.inc"
    };
    return specs;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool excluded_sidecar(const std::filesystem::path & path, const std::filesystem::path & output) {
    const std::string extension = lower_ascii(path.extension().string());
    return extension == ".safetensors" || extension == ".gguf" || extension == ".bin" || extension == ".pt" ||
           extension == ".pth" ||
           std::filesystem::weakly_canonical(path) == std::filesystem::weakly_canonical(output) ||
           std::filesystem::file_size(path) > 64u * 1024u * 1024u;
}

std::set<std::string>
planned_sidecar_destinations(const std::filesystem::path & root, const std::filesystem::path & output,
                             const std::vector<engine::assets::GgufEmbeddedFile> & explicit_sidecars,
                             bool embed_sidecars) {
    std::set<std::string> destinations;
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator it(root, error), end; !error && it != end; it.increment(error)) {
        if (it->is_directory() && (it->path().filename() == ".cache" || it->path().filename() == ".git")) {
            it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file() ||
            (embed_sidecars && excluded_sidecar(it->path(), output)) ||
            (!embed_sidecars && std::filesystem::weakly_canonical(it->path()) ==
                                   std::filesystem::weakly_canonical(output)))
            continue;
        destinations.insert(std::filesystem::relative(it->path(), root).lexically_normal().generic_string());
    }
    if (error)
        throw std::runtime_error("failed to enumerate model sidecars: " + error.message());
    std::set<std::string> explicit_destinations;
    for (const auto & sidecar : explicit_sidecars) {
        const auto destination = sidecar.destination.lexically_normal();
        if (!engine::io::is_existing_file(sidecar.source_path)) {
            throw std::runtime_error("embedded sidecar source does not exist: " + sidecar.source_path.string());
        }
        if (destination.empty() || destination.is_absolute() || *destination.begin() == "..") {
            throw std::runtime_error("invalid embedded sidecar destination: " + sidecar.destination.string());
        }
        const auto normalized = destination.generic_string();
        if (!explicit_destinations.insert(normalized).second) {
            throw std::runtime_error("duplicate embedded sidecar destination: " + destination.generic_string());
        }
        destinations.insert(normalized);
    }
    return destinations;
}

PackageSpecCandidate parse_package_spec(const std::string & text, std::string origin, int priority) {
    const auto root = json::parse(text);
    PackageSpecCandidate candidate;
    candidate.spec.family = json::require_string(root, "family");
    if (candidate.spec.family.empty())
        throw std::runtime_error("model package spec family is empty: " + origin);
    const auto & sources = root.require("sources").as_array();
    if (sources.empty())
        throw std::runtime_error("model package spec has no sources: " + origin);
    bool has_gguf = false;
    for (const auto & source : sources) {
        if (json::require_string(source, "format") == "gguf")
            has_gguf = true;
    }
    if (!has_gguf)
        throw std::runtime_error("model package spec has no GGUF source: " + origin);
    candidate.spec.json = json::stringify(root);
    candidate.origin = std::move(origin);
    candidate.priority = priority;
    return candidate;
}

void add_candidate(std::vector<PackageSpecCandidate> & candidates, PackageSpecCandidate candidate) {
    for (auto & existing : candidates) {
        if (existing.spec.family == candidate.spec.family && existing.spec.json == candidate.spec.json) {
            if (candidate.priority < existing.priority)
                existing = std::move(candidate);
            return;
        }
    }
    candidates.push_back(std::move(candidate));
}

void add_spec_file(std::vector<PackageSpecCandidate> & candidates, const std::filesystem::path & path, int priority) {
    if (!engine::io::is_existing_file(path))
        return;
    add_candidate(candidates, parse_package_spec(engine::io::read_text_file(path),
                                                 std::filesystem::weakly_canonical(path).string(), priority));
}

void add_spec_path(std::vector<PackageSpecCandidate> & candidates, const std::filesystem::path & path,
                   const std::optional<std::string> & family, int priority) {
    if (engine::io::is_existing_file(path)) {
        add_spec_file(candidates, path, priority);
        return;
    }
    if (!engine::io::is_existing_directory(path)) {
        throw std::runtime_error("model package spec path does not exist: " + path.string());
    }
    if (family.has_value()) {
        const auto file = path / (*family + ".json");
        if (!engine::io::is_existing_file(file)) {
            throw std::runtime_error("model package spec not found: " + file.string());
        }
        add_spec_file(candidates, file, priority);
        return;
    }
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && lower_ascii(entry.path().extension().string()) == ".json") {
            try {
                add_spec_file(candidates, entry.path(), priority);
            } catch (...) {
                // A model_specs directory may contain unrelated JSON files. Only valid
                // package specs participate in automatic matching.
            }
        }
    }
}

void add_discovered_spec_directories(std::vector<PackageSpecCandidate> & candidates,
                                     const std::filesystem::path & start,
                                     const std::optional<std::string> & family,
                                     int priority) {
    auto cursor = std::filesystem::weakly_canonical(start);
    while (!cursor.empty()) {
        const auto directory = cursor / "model_specs";
        if (engine::io::is_existing_directory(directory)) {
            if (!family.has_value() || engine::io::is_existing_file(directory / (*family + ".json"))) {
                add_spec_path(candidates, directory, family, priority);
            }
        }
        const auto parent = cursor.parent_path();
        if (parent == cursor || parent.empty())
            break;
        cursor = parent;
    }
}

void add_converter_catalog(std::vector<PackageSpecCandidate> & candidates,
                           const std::optional<std::string> & family,
                           int priority) {
    for (const auto & [catalog_family, text] : converter_model_specs()) {
        if (family.has_value() && *family != catalog_family)
            continue;
        add_candidate(candidates,
                      parse_package_spec(std::string(text), "converter-catalog:" + catalog_family, priority));
    }
}

const json::Value * embedded_spec_value(const json::Value & config) {
    if (const auto * value = config.find("audiocpp_model_spec"))
        return value;
    if (const auto * value = config.find("model_spec"))
        return value;
    if (const auto * audiocpp = config.find("audiocpp"); audiocpp != nullptr && audiocpp->is_object()) {
        if (const auto * value = audiocpp->find("model_spec"))
            return value;
        if (const auto * value = audiocpp->find("package_spec"))
            return value;
    }
    return nullptr;
}

std::optional<std::string> infer_family_from_config(const json::Value & config) {
    if (const auto * value = config.find("audiocpp_family"); value != nullptr && value->is_string()) {
        return value->as_string();
    }
    if (const auto * audiocpp = config.find("audiocpp"); audiocpp != nullptr && audiocpp->is_object()) {
        if (const auto * value = audiocpp->find("family"); value != nullptr && value->is_string()) {
            return value->as_string();
        }
    }
    const auto * model_type_value = config.find("model_type");
    if (model_type_value == nullptr || !model_type_value->is_string())
        return std::nullopt;
    const std::string model_type = model_type_value->as_string();
    if (model_type == "qwen3_asr") {
        const auto * thinker = config.find("thinker_config");
        if (thinker != nullptr && thinker->is_object()) {
            const auto * thinker_type = thinker->find("model_type");
            if (thinker_type != nullptr && thinker_type->is_string() &&
                thinker_type->as_string() == "qwen3_forced_aligner") {
                return "qwen3_forced_aligner";
            }
        }
        return "qwen3_asr";
    }
    if (model_type == "nemotron3_5_asr")
        return "nemotron_asr";
    if (model_type == "vibevoice")
        return "vibevoice_asr";
    if (model_type == "higgs_audio_3")
        return "higgs_audio_stt";
    if (model_type == "cohere_asr")
        return "hviske_asr";
    if (model_type == "qwen3_tts")
        return "qwen3_tts";
    return std::nullopt;
}

std::optional<std::string> add_model_config_spec(std::vector<PackageSpecCandidate> & candidates,
                                                 const std::filesystem::path & model_root) {
    const auto config_path = model_root / "config.json";
    if (!engine::io::is_existing_file(config_path))
        return std::nullopt;
    const auto config = json::parse_file(config_path);
    if (const auto * value = embedded_spec_value(config)) {
        if (value->is_object()) {
            add_candidate(candidates,
                          parse_package_spec(json::stringify(*value), config_path.string() + "#model_spec", 1));
        } else if (value->is_string()) {
            const std::string stored = value->as_string();
            const size_t first_content = stored.find_first_not_of(" \t\r\n");
            if (first_content == std::string::npos) {
                throw std::runtime_error("config.json model_spec string is empty");
            }
            if (stored[first_content] == '{') {
                add_candidate(candidates, parse_package_spec(stored, config_path.string() + "#model_spec", 1));
            } else {
                const auto stored_path = model_root / stored;
                if (!engine::io::is_existing_file(stored_path)) {
                    throw std::runtime_error("config.json model_spec file does not exist: " + stored_path.string());
                }
                add_spec_file(candidates, stored_path, 1);
            }
        } else {
            throw std::runtime_error("config.json model_spec must be an object, JSON "
                                     "string, or relative path");
        }
    }
    return infer_family_from_config(config);
}

const json::Value & require_gguf_source(const json::Value & spec) {
    for (const auto & source : spec.require("sources").as_array()) {
        if (json::require_string(source, "format") == "gguf")
            return source;
    }
    throw std::runtime_error("model package spec has no GGUF source");
}

std::string tensor_prefix(const json::Value & value) {
    if (value.is_string())
        return {};
    const auto * prefix = value.find("prefix");
    return prefix == nullptr ? std::string() : prefix->as_string();
}

std::optional<std::string> required_destination(const json::Value & source, const std::string & reference) {
    const size_t separator = reference.find(':');
    if (separator == std::string::npos || separator == 0) {
        throw std::runtime_error("invalid package resource reference: " + reference);
    }
    const std::string root_id = reference.substr(0, separator);
    const std::string relative = reference.substr(separator + 1);
    const auto & roots = source.require("roots").as_object();
    const auto root = roots.find(root_id);
    if (root == roots.end())
        throw std::runtime_error("unknown package root: " + root_id);
    if (root->second.as_string() == "$gguf")
        return std::nullopt;
    const auto destination = (std::filesystem::path(root->second.as_string()) / relative).lexically_normal();
    if (destination.is_absolute() || destination.empty() || *destination.begin() == "..") {
        throw std::runtime_error("unsafe GGUF package resource destination: " + destination.string());
    }
    return destination.generic_string();
}

std::vector<std::string> validate_candidate(const PackageSpecCandidate & candidate,
                                            const std::set<std::string> & actual_prefixes,
                                            const std::set<std::string> & sidecars) {
    std::vector<std::string> errors;
    try {
        const auto spec = json::parse(candidate.spec.json);
        const auto & source = require_gguf_source(spec);
        std::set<std::string> expected_prefixes;
        for (const auto & [unused, value] : source.require("tensors").as_object()) {
            (void)unused;
            expected_prefixes.insert(tensor_prefix(value));
        }
        for (const auto & prefix : expected_prefixes) {
            if (actual_prefixes.find(prefix) == actual_prefixes.end()) {
                errors.push_back("missing tensor namespace '" + (prefix.empty() ? std::string("<root>") : prefix) +
                                 "'");
            }
        }
        for (const auto & prefix : actual_prefixes) {
            if (expected_prefixes.find(prefix) == expected_prefixes.end()) {
                errors.push_back("unexpected tensor namespace '" + (prefix.empty() ? std::string("<root>") : prefix) +
                                 "'");
            }
        }
        if (const auto * files = source.find("files")) {
            for (const auto & [id, value] : files->as_object()) {
                const auto destination = required_destination(source, value.as_string());
                if (destination.has_value() && sidecars.find(*destination) == sidecars.end()) {
                    errors.push_back("missing required sidecar '" + id + "' at " + *destination);
                }
            }
        }
    } catch (const std::exception & error) {
        errors.push_back(error.what());
    }
    return errors;
}

PackageSpecCandidate select_package_spec(const std::vector<engine::assets::TensorSourceInput> & inputs,
                                         const std::filesystem::path & model_root, const std::filesystem::path & output,
                                         const std::vector<engine::assets::GgufEmbeddedFile> & explicit_sidecars,
                                         const std::optional<std::filesystem::path> & requested_spec,
                                         std::optional<std::string> family,
                                         bool embed_sidecars) {
    std::vector<PackageSpecCandidate> candidates;
    if (requested_spec.has_value()) {
        add_spec_path(candidates, *requested_spec, family, 0);
    } else {
        const size_t config_candidate_count = candidates.size();
        const auto config_family = add_model_config_spec(candidates, model_root);
        if (!family.has_value())
            family = config_family;

        if (candidates.size() == config_candidate_count) {
            if (engine::io::is_existing_file(model_root / "model_spec.json")) {
                add_spec_file(candidates, model_root / "model_spec.json", 2);
            }
            if (engine::io::is_existing_directory(model_root / "model_specs")) {
                add_spec_path(candidates, model_root / "model_specs", family, 2);
            }
            add_discovered_spec_directories(candidates, std::filesystem::current_path(), family, 3);
            add_converter_catalog(candidates, family, 4);
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("no model package spec was found; pass --family/--model-spec, add "
                                 "model_spec to config.json, "
                                 "install model_specs/*.json, or use --allow-missing-model-spec for a "
                                 "non-runtime tensor archive");
    }

    std::set<std::string> prefixes;
    for (const auto & input : inputs) {
        if (!prefixes.insert(input.tensor_prefix).second) {
            throw std::runtime_error("duplicate tensor namespace in conversion inputs: '" +
                                     (input.tensor_prefix.empty() ? std::string("<root>") : input.tensor_prefix) + "'");
        }
    }
    const auto sidecars = planned_sidecar_destinations(model_root, output, explicit_sidecars, embed_sidecars);
    std::vector<std::pair<PackageSpecCandidate, std::vector<std::string>>> matches;
    std::vector<std::pair<std::string, std::vector<std::string>>> rejected;

    std::optional<int> selected_priority;
    for (const auto & candidate : candidates) {
        if (family.has_value() && candidate.spec.family != *family)
            continue;
        if (!selected_priority.has_value() || candidate.priority < *selected_priority)
            selected_priority = candidate.priority;
    }
    if (!selected_priority.has_value()) {
        throw std::runtime_error("no model package spec was found for family '" +
                                 (family.has_value() ? *family : std::string("<auto>")) + "'");
    }
    for (const auto & candidate : candidates) {
        if ((family.has_value() && candidate.spec.family != *family) || candidate.priority != *selected_priority)
            continue;
        auto errors = validate_candidate(candidate, prefixes, sidecars);
        if (errors.empty())
            matches.push_back({candidate, {}});
        else
            rejected.push_back({candidate.spec.family + " (" + candidate.origin + ")", std::move(errors)});
    }
    if (matches.empty()) {
        std::ostringstream message;
        message << "no model package spec matches the conversion inputs";
        if (family.has_value())
            message << " for family '" << *family << "'";
        for (const auto & [name, errors] : rejected) {
            message << "\n  " << name << ':';
            for (const auto & error : errors)
                message << "\n    - " << error;
        }
        throw std::runtime_error(message.str());
    }
    if (matches.size() != 1) {
        std::ostringstream message;
        message << "multiple model package specs match:";
        for (const auto & match : matches)
            message << ' ' << match.first.spec.family;
        message << "; pass --family or --model-spec explicitly";
        throw std::runtime_error(message.str());
    }
    return matches.front().first;
}

void print_usage() {
    std::cout << "Usage: audiocpp_gguf --input [namespace=]<weights> [--input "
                 "namespace=<weights> ...] "
                 "--output <weights.gguf> --type "
                 "<orig|f16|bf16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k> "
                 "[--family <family>] [--model-spec <json-or-directory>] "
                 "[--root <model-dir>] [--sidecar <source>=<destination>] "
                 "[--overwrite] [--no-sidecars] "
                 "[--allow-missing-model-spec]\n"
        << "       audiocpp_gguf --inspect <model.gguf>\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::vector<engine::assets::TensorSourceInput> inputs;
        std::vector<engine::assets::GgufEmbeddedFile> sidecars;
        std::filesystem::path output;
        std::filesystem::path inspect_path;
        std::filesystem::path sidecar_root;
        std::optional<std::filesystem::path> model_spec_path;
        std::optional<std::string> family;
        std::string type;
        bool overwrite = false;
        bool embed_sidecars = true;
        bool allow_missing_model_spec = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--overwrite") {
                overwrite = true;
                continue;
            }
            if (arg == "--no-sidecars") {
                embed_sidecars = false;
                continue;
            }
            if (arg == "--allow-missing-model-spec") {
                allow_missing_model_spec = true;
                continue;
            }
            if ((arg == "--input" || arg == "--output" || arg == "--type" || arg == "--inspect" || arg == "--root" ||
                 arg == "--sidecar" || arg == "--family" || arg == "--model-spec" || arg == "--model-spec-override") &&
                i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--input") {
                    const auto separator = value.find('=');
                    if (separator == std::string::npos)
                        inputs.push_back({value, ""});
                    else
                        inputs.push_back({value.substr(separator + 1), value.substr(0, separator)});
                } else if (arg == "--output")
                    output = value;
                else if (arg == "--type")
                    type = value;
                else if (arg == "--inspect")
                    inspect_path = value;
                else if (arg == "--root")
                    sidecar_root = value;
                else if (arg == "--family")
                    family = value;
                else if (arg == "--model-spec" || arg == "--model-spec-override") {
                    model_spec_path = std::filesystem::path(value);
                } else {
                    const auto separator = value.find('=');
                    if (separator == std::string::npos || separator == 0 || separator + 1 == value.size()) {
                        throw std::runtime_error("--sidecar requires <source>=<destination>");
                    }
                    sidecars.push_back({value.substr(0, separator), value.substr(separator + 1)});
                }
                continue;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
        if (!inspect_path.empty()) {
            const auto source = engine::assets::open_tensor_source(inspect_path);
            const auto tensors = source->tensors();
            size_t scalar_count = 0;
            std::set<std::string> namespaces;
            for (const auto & tensor : tensors) {
                if (tensor.shape.empty())
                    ++scalar_count;
                const auto separator = tensor.name.find('/');
                if (separator != std::string::npos)
                    namespaces.insert(tensor.name.substr(0, separator));
            }
            std::cout << "gguf=" << std::filesystem::weakly_canonical(inspect_path).string() << "\n";
            std::cout << "tensors=" << tensors.size() << "\n";
            std::cout << "rank0_scalars=" << scalar_count << "\n";
            std::cout << "embedded_sidecars="
                      << (engine::assets::gguf_has_embedded_sidecars(inspect_path) ? "true" : "false") << "\n";
            const auto model_spec = engine::assets::read_gguf_embedded_model_spec(inspect_path);
            std::cout << "embedded_model_spec=" << (model_spec.has_value() ? "true" : "false") << "\n";
            if (model_spec.has_value())
                std::cout << "model_spec_family=" << model_spec->family << "\n";
            for (const auto & value : namespaces)
                std::cout << "namespace=" << value << "\n";
            return 0;
        }
        if (inputs.empty() || output.empty() || type.empty()) {
            print_usage();
            return 2;
        }
        if (!embed_sidecars && !sidecars.empty()) {
            throw std::runtime_error("--sidecar cannot be combined with --no-sidecars");
        }
        for (const auto & input : inputs) {
            if (!std::filesystem::is_regular_file(input.path)) {
                throw std::runtime_error("input tensor file does not exist: " + input.path.string());
            }
        }
        if (std::filesystem::exists(output)) {
            if (!overwrite) {
                throw std::runtime_error("output already exists; pass --overwrite to replace it: " + output.string());
            }
        }
        const auto storage_type = engine::assets::parse_tensor_storage_type(type);
        const auto resolved_sidecar_root =
            std::filesystem::weakly_canonical(sidecar_root.empty() ? inputs.front().path.parent_path() : sidecar_root);
        std::optional<engine::assets::GgufEmbeddedModelSpec> embedded_model_spec;
        if (!allow_missing_model_spec) {
            embedded_model_spec =
                select_package_spec(inputs, resolved_sidecar_root, output, sidecars, model_spec_path, family,
                                    embed_sidecars).spec;
        } else {
            try {
                embedded_model_spec =
                    select_package_spec(inputs, resolved_sidecar_root, output, sidecars, model_spec_path, family,
                                        embed_sidecars).spec;
            } catch (const std::exception & warning) {
                std::cerr << "warning: creating GGUF without model package spec: " << warning.what() << "\n";
            }
        }
        engine::assets::convert_tensor_sources_to_gguf(inputs, output, storage_type, overwrite, embed_sidecars,
                                                       resolved_sidecar_root, sidecars, embedded_model_spec);
        std::cout << "gguf=" << std::filesystem::weakly_canonical(output).string() << "\n";
        std::cout << "weight_type=" << type << "\n";
        std::cout << "tensor_sources=" << inputs.size() << "\n";
        std::cout << "embedded_sidecars=" << (engine::assets::gguf_has_embedded_sidecars(output) ? "true" : "false")
                  << "\n";
        std::cout << "embedded_model_spec=" << (embedded_model_spec.has_value() ? "true" : "false") << "\n";
        if (embedded_model_spec.has_value()) {
            std::cout << "model_spec_family=" << embedded_model_spec->family << "\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
