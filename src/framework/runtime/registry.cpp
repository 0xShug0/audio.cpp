#include "engine/framework/runtime/registry.h"

#include "engine/framework/core/load_progress.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/audio/utilities/utility_api.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/models/marblenet_vad/session.h"
#include "engine/models/silero_vad/session.h"

#include "model_registry_includes.inc"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <system_error>

namespace engine::runtime {

namespace {

std::vector<std::string> split_csv(std::string value) {
    std::vector<std::string> items;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(',', start);
        std::string item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) { return std::isspace(ch) == 0; }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) { return std::isspace(ch) == 0; }).base(), item.end());
        if (!item.empty()) {
            items.push_back(std::move(item));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return items;
}

void log_model_load_trace(const ModelInspection & inspection, const ILoadedVoiceModel & model) {
    if (!engine::debug::trace_log_enabled()) {
        return;
    }
    const auto & metadata = model.metadata();
    engine::debug::trace_log_scalar("runtime.model.family", metadata.family);
    engine::debug::trace_log_scalar("runtime.model.variant", metadata.variant);
    engine::debug::trace_log_scalar("runtime.model.root", inspection.model_root.string());
    engine::debug::trace_log_scalar("runtime.model.discovered_config_count", inspection.discovered_configs.size());
    engine::debug::trace_log_scalar("runtime.model.discovered_weight_count", inspection.discovered_weights.size());
    engine::debug::trace_log_scalar("runtime.model.task_count", model.capabilities().supported_tasks.size());
    engine::debug::trace_log_scalar("runtime.model.language_count", model.capabilities().languages.size());
    engine::debug::trace_log_scalar("runtime.model.supports_speaker_reference", model.capabilities().supports_speaker_reference);
    engine::debug::trace_log_scalar("runtime.model.supports_style_condition", model.capabilities().supports_style_condition);
    engine::debug::trace_log_scalar("runtime.model.supports_timestamps", model.capabilities().supports_timestamps);
}

uint64_t discovered_weight_bytes(const ModelInspection & inspection) {
    std::error_code ec;
    uint64_t bytes = 0;
    for (const auto & weight : inspection.discovered_weights) {
        const uintmax_t size = std::filesystem::file_size(weight.path, ec);
        if (!ec) {
            bytes += static_cast<uint64_t>(size);
        }
        ec.clear();
    }
    return bytes;
}

}  // namespace

void finish_model_load_trace() {
    // Opt-in additions only (--load-progress 1): without the flag this is a
    // no-op and the log matches the pre-load-progress behavior exactly.
    if (!engine::core::load_progress_enabled() || !engine::debug::trace_log_enabled()) {
        return;
    }
    engine::debug::trace_log_scalar("runtime.load.phase", std::string_view{"loaded"});
    engine::core::end_model_load();
}

void ModelRegistry::register_loader(std::shared_ptr<IVoiceModelLoader> loader) {
    if (loader == nullptr) {
        throw std::invalid_argument("model loader must not be null");
    }
    loaders_.push_back(std::move(loader));
}

bool ModelRegistry::empty() const noexcept {
    return loaders_.empty();
}

size_t ModelRegistry::size() const noexcept {
    return loaders_.size();
}

std::vector<std::string> ModelRegistry::families() const {
    std::vector<std::string> names;
    names.reserve(loaders_.size());
    for (const auto & loader : loaders_) {
        names.push_back(loader->family());
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

bool ModelRegistry::supports_family(const std::string & family) const noexcept {
    for (const auto & loader : loaders_) {
        if (loader->family() == family) {
            return true;
        }
        const auto aliases = loader->family_aliases();
        if (std::find(aliases.begin(), aliases.end(), family) != aliases.end()) {
            return true;
        }
    }
    return false;
}

std::vector<LoaderAdvertisement> ModelRegistry::advertise_loaders() const {
    std::vector<LoaderAdvertisement> out;
    out.reserve(loaders_.size());
    for (const auto & loader : loaders_) {
        if (loader == nullptr) {
            continue;
        }
        out.push_back(loader->advertise());
    }
    std::sort(out.begin(), out.end(), [](const LoaderAdvertisement & a, const LoaderAdvertisement & b) {
        return a.family < b.family;
    });
    return out;
}

ModelInspection ModelRegistry::inspect(const ModelLoadRequest & request) const {
    engine::model_spec::ScopedSpecOverride spec_override(request.model_spec_override, request.model_path);
    validate_request(request);
    const auto * loader = find_loader(request);
    if (loader == nullptr) {
        throw std::runtime_error("no registered model loader can inspect: " + request.model_path.string());
    }
    return loader->inspect(request);
}

ModelInspection ModelRegistry::inspect(const std::filesystem::path & model_path) const {
    ModelLoadRequest request;
    request.model_path = model_path;
    return inspect(request);
}

std::unique_ptr<ILoadedVoiceModel> ModelRegistry::load(const ModelLoadRequest & request) const {
    engine::model_spec::ScopedSpecOverride spec_override(request.model_spec_override, request.model_path);
    validate_request(request);
    const auto * loader = find_loader(request);
    if (loader == nullptr) {
        throw std::runtime_error("no registered model loader can load: " + request.model_path.string());
    }
    // Phase brackets + the uploaded-bytes curve give trace-log consumers a
    // continuous load progress (llama.cpp-style). Opt-in via --load-progress
    // 1 so the log for existing downstream pipelines stays byte-identical:
    // the runtime.model.* inspection trace below predates it and keeps its
    // original trace-log gating. The curve STARTS here and is closed by
    // finish_model_load_trace() once the host has built its task session:
    // the actual weight stores upload during session construction, not
    // inside the registry load.
    const bool log_traced = engine::debug::trace_log_enabled();
    const bool progress = engine::core::load_progress_enabled();
    std::optional<ModelInspection> inspection;
    if (log_traced) {
        if (progress) {
            engine::debug::trace_log_scalar("runtime.load.phase", std::string_view{"inspect"});
        }
        inspection = loader->inspect(request);
        if (progress) {
            engine::core::begin_model_load(discovered_weight_bytes(*inspection));
            engine::debug::trace_log_scalar("runtime.load.phase", std::string_view{"load"});
        }
    }
    auto model = loader->load(request);
    if (inspection.has_value()) {
        log_model_load_trace(*inspection, *model);
    }
    return model;
}

std::unique_ptr<ILoadedVoiceModel> ModelRegistry::load(const std::filesystem::path & model_path) const {
    ModelLoadRequest request;
    request.model_path = model_path;
    return load(request);
}

ModelResourcePlan ModelRegistry::model_resources(
    const std::string & family,
    const std::filesystem::path & model_path,
    const std::unordered_map<std::string, std::string> & session_options) const {
    for (const auto & loader : loaders_) {
        const auto aliases = loader->family_aliases();
        if (loader->family() == family ||
            std::find(aliases.begin(), aliases.end(), family) != aliases.end()) {
            return loader->model_resources(model_path, session_options);
        }
    }
    return {};
}

void ModelRegistry::validate_request(const ModelLoadRequest & request) const {
    const bool builtin_audio_utility_id =
        request.family_hint.has_value() &&
        *request.family_hint == "builtin_audio_utils" &&
        engine::audio::find_builtin_audio_utility(request.model_path.generic_string()).has_value();
    if (!builtin_audio_utility_id &&
        !engine::io::is_existing_file(request.model_path) &&
        !engine::io::is_existing_directory(request.model_path)) {
        throw std::runtime_error("model path does not exist: " + request.model_path.string());
    }
    if (request.family_hint.has_value() && !supports_family(*request.family_hint)) {
        throw std::runtime_error("unsupported model family hint: " + *request.family_hint);
    }
    if (request.model_spec_override.has_value() &&
        *request.model_spec_override != "embedded" &&
        !engine::io::is_existing_file(*request.model_spec_override) &&
        !engine::io::is_existing_directory(*request.model_spec_override)) {
        throw std::runtime_error("model spec override path does not exist: " + request.model_spec_override->string());
    }
}

const IVoiceModelLoader * ModelRegistry::find_loader(const ModelLoadRequest & request) const {
    if (!request.family_hint.has_value()) {
        for (const auto & loader : loaders_) {
            if (loader->identifies(request)) {
                return loader.get();
            }
        }
    }
    for (const auto & loader : loaders_) {
        if (request.family_hint.has_value()) {
            if (loader->family() == *request.family_hint) {
                return loader.get();
            }
            const auto aliases = loader->family_aliases();
            if (std::find(aliases.begin(), aliases.end(), *request.family_hint) != aliases.end()) {
                return loader.get();
            }
            continue;
        }
        if (loader->can_load(request)) {
            return loader.get();
        }
    }
    return nullptr;
}

RegistryConfig load_registry_config(const std::filesystem::path & path) {
    const auto config = engine::io::load_config_map(path);
    RegistryConfig registry_config;
    if (const auto it = config.find("families"); it != config.end()) {
        registry_config.enabled_families = split_csv(it->second);
    } else if (const auto it = config.find("loaders"); it != config.end()) {
        registry_config.enabled_families = split_csv(it->second);
    }
    return registry_config;
}

ModelRegistry make_registry_from_config(
    const RegistryConfig & config,
    const std::vector<std::shared_ptr<IVoiceModelLoader>> & available_loaders) {
    ModelRegistry registry;
    if (config.enabled_families.empty()) {
        return registry;
    }

    for (const auto & family : config.enabled_families) {
        auto it = std::find_if(
            available_loaders.begin(),
            available_loaders.end(),
            [&](const std::shared_ptr<IVoiceModelLoader> & loader) {
                return loader != nullptr && loader->family() == family;
            });
        if (it == available_loaders.end()) {
            throw std::runtime_error("registry config requested unknown loader family: " + family);
        }
        registry.register_loader(*it);
    }
    return registry;
}

namespace {

std::vector<std::shared_ptr<IVoiceModelLoader>> default_loaders() {
    return {
        engine::models::silero_vad::make_silero_vad_loader(),
        engine::models::marblenet_vad::make_marblenet_vad_loader(),
#include "model_registry_loaders.inc"
    };
}

}  // namespace

ModelRegistry make_default_registry(const std::optional<std::filesystem::path> & config_path) {
    const std::vector<std::shared_ptr<IVoiceModelLoader>> available_loaders = default_loaders();
    if (!config_path.has_value()) {
        ModelRegistry registry;
        for (const auto & loader : available_loaders) {
            registry.register_loader(loader);
        }
        return registry;
    }
    if (!engine::io::is_existing_file(*config_path)) {
        throw std::runtime_error("registry config path does not exist: " + config_path->string());
    }
    const auto config = load_registry_config(*config_path);
    return make_registry_from_config(config, available_loaders);
}

namespace {

bool has_offline_vad(const CapabilitySet & capabilities) {
    return std::any_of(
        capabilities.supported_tasks.begin(), capabilities.supported_tasks.end(), [](const TaskCapability & task) {
            return task.task == VoiceTaskKind::Vad &&
                   std::find(task.modes.begin(), task.modes.end(), RunMode::Offline) != task.modes.end();
        });
}

bool same_file(const std::filesystem::path & a, const std::filesystem::path & b) {
    std::error_code ec_a;
    std::error_code ec_b;
    const auto ca = std::filesystem::weakly_canonical(a, ec_a);
    const auto cb = std::filesystem::weakly_canonical(b, ec_b);
    return !ec_a && !ec_b && ca == cb;
}

// True when the loader would read exactly this weight file, not just a package in its folder.
bool claims_exact_file(const IVoiceModelLoader & loader, const ModelLoadRequest & request) {
    try {
        if (!loader.can_load(request)) {
            return false;
        }
        const auto inspection = loader.inspect(request);
        return std::any_of(
            inspection.discovered_weights.begin(), inspection.discovered_weights.end(),
            [&](const NamedAsset & asset) { return same_file(asset.path, request.model_path); });
    } catch (const std::exception &) {
        return false;
    }
}

}  // namespace

std::unique_ptr<ILoadedVoiceModel> load_vad_model(const ModelLoadRequest & request) {
    engine::model_spec::ScopedSpecOverride spec_override(request.model_spec_override, request.model_path);
    // Internal load for a host session (e.g. ASR chunking): call the loader directly so the
    // host's load trace and progress bracket are left alone. Only VAD loaders are considered,
    // so a wrong path never loads a large non-VAD model first.
    const auto loaders = default_loaders();
    std::vector<const IVoiceModelLoader *> vad_loaders;
    const IVoiceModelLoader * silero = nullptr;
    std::string built;
    for (const auto & loader : loaders) {
        bool vad = false;
        try {
            vad = has_offline_vad(loader->advertised_capabilities());
        } catch (const std::exception &) {
        }
        if (!vad) {
            continue;
        }
        vad_loaders.push_back(loader.get());
        built += (built.empty() ? "" : ", ") + loader->family();
        if (loader->family() == "silero_vad") {
            silero = loader.get();
        }
    }

    const IVoiceModelLoader * selected = nullptr;
    if (request.family_hint.has_value()) {
        for (const auto * loader : vad_loaders) {
            const auto aliases = loader->family_aliases();
            if (loader->family() == *request.family_hint ||
                std::find(aliases.begin(), aliases.end(), *request.family_hint) != aliases.end()) {
                selected = loader;
                break;
            }
        }
        if (selected == nullptr) {
            throw std::runtime_error(
                "VAD family '" + *request.family_hint + "' is not built into this binary (built VAD families: " +
                built + ")");
        }
        return selected->load(request);
    }

    const bool is_file = engine::io::is_existing_file(request.model_path);
    if (is_file) {
        // A weight file: prefer the loader that uses exactly this file. Package loaders resolve a
        // file to its folder, so a different package next to the file must not claim it.
        for (const auto * loader : vad_loaders) {
            if (claims_exact_file(*loader, request)) {
                return loader->load(request);
            }
        }
    } else {
        for (const auto * loader : vad_loaders) {
            if (loader->can_load(request)) {
                return loader->load(request);
            }
        }
    }

    // Unclaimed path: load it as Silero, as the Silero-only loader did before (e.g. Silero
    // weights under another file name). Then try any folder-level VAD package for a file.
    std::string silero_error = "silero_vad is not built into this binary";
    if (silero != nullptr) {
        try {
            return silero->load(request);
        } catch (const std::exception & error) {
            silero_error = error.what();
        }
    }
    if (is_file) {
        for (const auto * loader : vad_loaders) {
            if (loader != silero && loader->can_load(request)) {
                return loader->load(request);
            }
        }
    }
    throw std::runtime_error(
        "no VAD model built into this binary can load " + request.model_path.string() +
        " (built VAD families: " + built + "; loading it as Silero VAD failed: " + silero_error + ")");
}

}  // namespace engine::runtime
