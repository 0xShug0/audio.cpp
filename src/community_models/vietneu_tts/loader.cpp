#include "engine/community_models/vietneu_tts/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/community_models/vietneu_tts/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::vietneu_tts {
namespace {

std::vector<std::string> supported_languages(const VietneuTTSConfig & config) {
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

runtime::ModelMetadata metadata(const VietneuTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "vietneu_tts";
    out.variant = assets.config.tts_model_size + "-" + assets.config.tts_model_type;
    out.description = "VieNeu-TTS loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const VietneuTTSAssets & assets) {
    runtime::CapabilitySet out;
    if (assets.config.variant == VietneuTTSVariant::Base) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
    }
    out.languages = supported_languages(assets.config);
    return out;
}

runtime::ModelCliInterface cli(const VietneuTTSAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"reference_text", "<text>", "Transcript of the reference speaker WAV."},
        {"speaker_embedding_file", "<path>", "Path to the speaker embedding .emb.txt file."},
        {"speaker_embedding", "<csv>", "Comma-separated list of 192 speaker embedding float values."},
        {"subtalker_temperature", "<float>", "Acoustic decoder sampling temperature (default 0.8)."},
        {"subtalker_do_sample", "true|false", "Enable sampling in the acoustic decoder (default true)."},
        {"text_chunk_size", "<int>", "Maximum character budget per chunk (default 200)."},
        {"text_chunk_mode", "default|endline|tag_aware", "Text chunking mode (default 'default')."},
    };
    out.session_options = {
        {"vietneu_tts.mem_saver", "true|false", "Release the talker cached-step graph after each request; default false."},
        {"vietneu_tts.voice_prompt_cache_slots", "n", "Voice prompt cache slots; default 1."},
    };
    return out;
}

class VietneuTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "vietneu_tts";
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
            const auto package_spec = engine::assets::default_model_package_spec_path(family());
            (void) engine::assets::load_resource_bundle_from_package_spec(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_vietneu_tts_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::assets::ModelPackageResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_vietneu_tts_model(request.model_path);
    }
};

}  // namespace

VietneuTTSLoadedModel::VietneuTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VietneuTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VietneuTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VietneuTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> VietneuTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VieNeu-TTS TTS only supports offline sessions");
    }
    if (assets_->config.variant == VietneuTTSVariant::Base && task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VieNeu-TTS base TTS model only supports the Tts task");
    }
    return std::make_unique<VietneuTTSSession>(task, options, assets_);
}

std::unique_ptr<VietneuTTSLoadedModel> load_vietneu_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_vietneu_tts_assets(model_path);
    return std::make_unique<VietneuTTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vietneu_tts_loader() {
    return std::make_shared<VietneuTTSLoader>();
}

}  // namespace engine::models::vietneu_tts
