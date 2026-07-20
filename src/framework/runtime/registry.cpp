#include "engine/framework/runtime/registry.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
// Development registry entries from Share/AudioCPP that are not present in this release tree yet:
// #include "engine/models/higgs_tts/loader.h"
// #include "engine/models/kokoro_tts/loader.h"
// #include "engine/models/parakeet_tdt/loader.h"
#include "engine/models/ace_step/loader.h"
#include "engine/models/chatterbox/loader.h"
#include "engine/models/citrinet_asr/session.h"
#include "engine/models/demucs/loader.h"
#include "engine/models/heartmula/loader.h"
#include "engine/models/higgs_audio_stt/loader.h"
#include "engine/models/hviske_asr/loader.h"
#include "engine/models/index_tts2/loader.h"
#include "engine/models/irodori_tts/loader.h"
#include "engine/models/marblenet_vad/session.h"
#include "engine/models/miocodec/loader.h"
#include "engine/models/miotts/loader.h"
#include "engine/models/moss/moss_tts_local/loader.h"
#include "engine/models/moss/moss_tts_nano/loader.h"
#include "engine/models/nemotron_asr/loader.h"
#include "engine/models/omnivoice/loader.h"
#include "engine/community_models/outetts/loader.h"
#include "engine/models/pocket_tts/loader.h"
#include "engine/models/qwen3_asr/loader.h"
#include "engine/models/qwen3_forced_aligner/loader.h"
#include "engine/models/qwen3_tts/loader.h"
#include "engine/community_models/vietneu_tts/loader.h"
#include "engine/models/roformer/loader.h"
#include "engine/models/silero_vad/session.h"
#include "engine/models/seed_vc/loader.h"
#include "engine/models/sortformer_diar/loader.h"
#include "engine/models/stable_audio/loader.h"
#include "engine/models/supertonic/loader.h"
#include "engine/models/vevo2/loader.h"
#include "engine/models/vibevoice/loader.h"
#include "engine/models/vibevoice_asr/loader.h"
#include "engine/models/voxtral_realtime/loader.h"
#include "engine/models/voxcpm2/loader.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

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

}  // namespace

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
    engine::assets::ScopedModelPackageSpecOverride spec_override(request.model_spec_override, request.model_path);
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
    engine::assets::ScopedModelPackageSpecOverride spec_override(request.model_spec_override, request.model_path);
    validate_request(request);
    const auto * loader = find_loader(request);
    if (loader == nullptr) {
        throw std::runtime_error("no registered model loader can load: " + request.model_path.string());
    }
    const auto inspection = engine::debug::trace_log_enabled()
        ? std::optional<ModelInspection>(loader->inspect(request))
        : std::nullopt;
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

void ModelRegistry::validate_request(const ModelLoadRequest & request) const {
    if (!engine::io::is_existing_file(request.model_path) && !engine::io::is_existing_directory(request.model_path)) {
        throw std::runtime_error("model path does not exist: " + request.model_path.string());
    }
    if (request.family_hint.has_value() && !supports_family(*request.family_hint)) {
        throw std::runtime_error("unsupported model family hint: " + *request.family_hint);
    }
    if (request.model_spec_override.has_value() &&
        !engine::io::is_existing_file(*request.model_spec_override) &&
        !engine::io::is_existing_directory(*request.model_spec_override)) {
        throw std::runtime_error("model package spec override path does not exist: " + request.model_spec_override->string());
    }
}

const IVoiceModelLoader * ModelRegistry::find_loader(const ModelLoadRequest & request) const {
    for (const auto & loader : loaders_) {
        if (request.family_hint.has_value() && loader->family() != *request.family_hint) {
            continue;
        }
        if (request.family_hint.has_value()) {
            return loader.get();
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

ModelRegistry make_default_registry(const std::optional<std::filesystem::path> & config_path) {
    const std::vector<std::shared_ptr<IVoiceModelLoader>> available_loaders = {
        // Development registry entries from Share/AudioCPP that are not present in this release tree yet:
        // engine::models::kokoro_tts::make_kokoro_tts_loader(),
        // engine::models::higgs_tts::make_higgs_tts_loader(),
        // engine::models::parakeet_tdt::make_parakeet_tdt_loader(),
        engine::models::ace_step::make_ace_step_loader(),
        engine::models::demucs::make_htdemucs_loader(),
        engine::models::roformer::make_mel_band_roformer_loader(),
        engine::models::omnivoice::make_omnivoice_loader(),
        engine::models::outetts::make_outetts_loader(),
        engine::models::miocodec::make_miocodec_loader(),
        engine::models::miotts::make_miotts_loader(),
        engine::models::moss_tts_local::make_moss_tts_local_loader(),
        engine::models::moss_tts_nano::make_moss_tts_nano_loader(),
        engine::models::voxcpm2::make_voxcpm2_loader(),
        engine::models::vibevoice::make_vibevoice_loader(),
        engine::models::vibevoice_asr::make_vibevoice_asr_loader(),
        engine::models::voxtral_realtime::make_voxtral_realtime_loader(),
        engine::models::heartmula::make_heartmula_loader(),
        engine::models::higgs_audio_stt::make_higgs_audio_stt_loader(),
        engine::models::hviske_asr::make_hviske_asr_loader(),
        engine::models::irodori_tts::make_irodori_tts_loader(),
        engine::models::nemotron_asr::make_nemotron_asr_loader(),
        engine::models::pocket_tts::make_pocket_tts_loader(),
        engine::models::qwen3_forced_aligner::make_qwen3_forced_aligner_loader(),
        engine::models::qwen3_asr::make_qwen3_asr_loader(),
        engine::models::index_tts2::make_index_tts2_loader(),
        engine::models::qwen3_tts::make_qwen3_tts_loader(),
        engine::models::vietneu_tts::make_vietneu_tts_loader(),
        engine::models::sortformer_diar::make_sortformer_diar_loader(),
        engine::models::stable_audio::make_stable_audio_loader(),
        engine::models::supertonic::make_supertonic_loader(),
        engine::models::silero_vad::make_silero_vad_loader(),
        engine::models::citrinet_asr::make_citrinet_asr_loader(),
        engine::models::marblenet_vad::make_marblenet_vad_loader(),
        engine::models::vevo2::make_vevo2_loader(),
        engine::models::seed_vc::make_seed_vc_loader(),
        engine::models::chatterbox::make_chatterbox_loader(),
    };
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

}  // namespace engine::runtime
