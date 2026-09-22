#include "engine/community_models/vieneu_v3_turbo/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/community_models/vieneu_v3_turbo/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::vieneu_v3_turbo {
namespace {

std::vector<std::string> supported_languages(const VieNeuTTSConfig & config) {
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

runtime::ModelMetadata metadata(const VieNeuTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = kFamily;
    out.variant = assets.config.tts_model_size + "-" + assets.config.tts_model_type;
    out.description = "VieNeu-TTS loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const VieNeuTTSAssets & assets) {
    runtime::CapabilitySet out;
    if (assets.config.variant == VieNeuTTSVariant::Base) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
    }
    out.languages = supported_languages(assets.config);
    return out;
}

runtime::ModelCliInterface cli(const VieNeuTTSAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"reference_text", "<text>", "Transcript of the reference speaker WAV."},
        {"speaker_embedding_file", "<path>", "Path to the speaker embedding .emb.txt file."},
        {"speaker_embedding", "<csv>", "Comma-separated list of 192 speaker embedding float values."},
        {"subtalker_temperature", "<float>", "Acoustic decoder sampling temperature (default 0.8)."},
        {"subtalker_do_sample", "true|false", "Enable sampling in the acoustic decoder (default true)."},
        {"text_chunk_size", "<int>", "Maximum character budget per chunk (default 200)."},
        {"text_chunk_mode", "default|endline|tag_aware", "Text chunking mode (default 'default')."},
        {"reference_codes_file", "<path>", "Pre-encoded reference codes (one frame per line, 16 ints); replaces the codec encoder pass and makes --voice-ref optional."},
        {"codes_dump_file", "<path>", "Append prompt ids, reference codes and generated codes as text (parity debugging)."},
    };
    out.session_options = {
        {"vieneu_v3_turbo.mem_saver", "true|false", "Release the talker cached-step graph after each request; default false."},
        {"vieneu_v3_turbo.voice_prompt_cache_slots", "n", "Voice prompt cache slots; default 1."},
    };
    return out;
}

class VieNeuTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return kFamily;
    }

    std::vector<std::string> family_aliases() const override {
        return {kLegacyFamily};
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
            const auto package_spec = resolve_package_spec_path();
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            if (!request.family_hint.has_value()) {
                return true;
            }
            const auto aliases = family_aliases();
            return *request.family_hint == family()
                || std::find(aliases.begin(), aliases.end(), *request.family_hint) != aliases.end();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_vieneu_v3_turbo_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = resolve_package_spec_path();
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
        return load_vieneu_v3_turbo_model(request.model_path);
    }
};

}  // namespace

VieNeuTTSLoadedModel::VieNeuTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VieNeuTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VieNeuTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VieNeuTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> VieNeuTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VieNeu-TTS TTS only supports offline sessions");
    }
    if (assets_->config.variant == VieNeuTTSVariant::Base && task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VieNeu-TTS base TTS model only supports the Tts task");
    }
    return std::make_unique<VieNeuTTSSession>(task, options, assets_);
}

std::unique_ptr<VieNeuTTSLoadedModel> load_vieneu_v3_turbo_model(const std::filesystem::path & model_path) {
    auto assets = load_vieneu_v3_turbo_assets(model_path);
    return std::make_unique<VieNeuTTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vieneu_v3_turbo_loader() {
    return std::make_shared<VieNeuTTSLoader>();
}

}  // namespace engine::models::vieneu_v3_turbo
