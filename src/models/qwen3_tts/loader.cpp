#include "engine/models/qwen3_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/qwen3_tts/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

std::vector<std::string> supported_languages(const Qwen3TTSConfig & config) {
    std::vector<std::string> languages;
    languages.reserve(config.talker.codec_language_id.size() + 1);
    languages.push_back("Auto");
    for (const auto & [language, id] : config.talker.codec_language_id) {
        (void) id;
        languages.push_back(language);
    }
    std::sort(languages.begin() + 1, languages.end());
    return languages;
}

runtime::ModelMetadata metadata(const Qwen3TTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "qwen3_tts";
    out.variant = assets.config.tts_model_size + "-" + assets.config.tts_model_type;
    out.description = "Qwen3 TTS loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const Qwen3TTSAssets & assets) {
    runtime::CapabilitySet out;
    if (assets.config.variant == Qwen3TTSVariant::Base) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
    } else if (assets.config.variant == Qwen3TTSVariant::VoiceDesign) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
        };
        out.supports_style_condition = true;
    } else if (assets.config.variant == Qwen3TTSVariant::CustomVoice) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.supports_style_condition = true;
    }
    out.languages = supported_languages(assets.config);
    return out;
}

runtime::ModelCliInterface cli(const Qwen3TTSAssets &) {
    runtime::ModelCliInterface out;
    out.session_options = {
        {"qwen3_tts.mem_saver", "true|false", "Release the talker cached-step graph after each request; default false."},
        {"qwen3_tts.perf_mode", "off|flash_attention", "Q8_0-only attention performance mode; default off keeps the exact-safe path."},
        {"qwen3_tts.voice_prompt_cache_slots", "n", "Voice prompt cache slots; default 1."},
    };
    return out;
}

class Qwen3TTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_tts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_qwen3_tts_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_qwen3_tts_model(request.model_path);
    }
};

}  // namespace

Qwen3TTSLoadedModel::Qwen3TTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Qwen3TTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Qwen3TTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Qwen3TTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Qwen3TTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 TTS only supports offline sessions");
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base && task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 base TTS model only supports the Tts task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign && task.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("Qwen3 voice design model only supports the VoiceDesign task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 custom voice model only supports the Tts task");
    }
    return std::make_unique<Qwen3TTSSession>(task, options, assets_);
}

std::unique_ptr<Qwen3TTSLoadedModel> load_qwen3_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_qwen3_tts_assets(model_path);
    return std::make_unique<Qwen3TTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_tts_loader() {
    return std::make_shared<Qwen3TTSLoader>();
}

}  // namespace engine::models::qwen3_tts
