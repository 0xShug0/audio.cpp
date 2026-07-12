#include "engine/models/qwen3_tts/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/qwen3_tts/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Qwen3 TTS model path does not exist: " + model_path.string());
}

bool has_qwen3_tts_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json")
        && (engine::io::is_existing_file(root / "model.gguf")
            || engine::io::is_existing_file(root / "model.safetensors"))
        && engine::io::is_existing_file(root / "tokenizer_config.json")
        && engine::io::is_existing_file(root / "vocab.json")
        && engine::io::is_existing_file(root / "merges.txt")
        && engine::io::is_existing_file(root / "speech_tokenizer" / "config.json")
        && (engine::io::is_existing_file(root / "speech_tokenizer" / "model.gguf")
            || engine::io::is_existing_file(root / "speech_tokenizer" / "model.safetensors"));
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"config.json", "generation_config.json", "tokenizer_config.json", "speech_tokenizer/config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"model.gguf", "model.safetensors", "speech_tokenizer/model.gguf", "speech_tokenizer/model.safetensors"});
}

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

class Qwen3TTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_tts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_qwen3_tts_assets(root)
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_qwen3_tts_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.tts_model_size + "-" + assets->config.tts_model_type;
        inspection.metadata.description = "Qwen3 TTS loaded from local extracted assets.";
        inspection.metadata.config_candidates = {
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "speech_tokenizer/config.json",
        };
        inspection.metadata.weight_candidates = {
            "model.gguf", "model.safetensors", "speech_tokenizer/model.gguf", "speech_tokenizer/model.safetensors"};
        inspection.cli.session_options = {
            {"qwen3_tts.mem_saver", "true|false", "Release the talker cached-step graph after each request; default false."},
        };
        if (assets->config.variant == Qwen3TTSVariant::Base) {
            inspection.capabilities.supported_tasks = {
                {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            };
            inspection.capabilities.supports_speaker_reference = true;
        } else if (assets->config.variant == Qwen3TTSVariant::VoiceDesign) {
            inspection.capabilities.supported_tasks = {
                {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
            };
            inspection.capabilities.supports_style_condition = true;
        } else if (assets->config.variant == Qwen3TTSVariant::CustomVoice) {
            inspection.capabilities.supported_tasks = {
                {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            };
            inspection.capabilities.supports_style_condition = true;
        }
        inspection.capabilities.languages = supported_languages(assets->config);
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_qwen3_tts_model(resolve_model_root(request.model_path));
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

    runtime::ModelMetadata metadata;
    metadata.family = "qwen3_tts";
    metadata.variant = assets->config.tts_model_size + "-" + assets->config.tts_model_type;
    metadata.description = "Qwen3 TTS loaded from local extracted assets.";
    metadata.config_candidates = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "speech_tokenizer/config.json",
    };
    metadata.weight_candidates = {
        "model.gguf", "model.safetensors", "speech_tokenizer/model.gguf", "speech_tokenizer/model.safetensors"};

    runtime::CapabilitySet capabilities;
    if (assets->config.variant == Qwen3TTSVariant::Base) {
        capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        capabilities.supports_speaker_reference = true;
    } else if (assets->config.variant == Qwen3TTSVariant::VoiceDesign) {
        capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
        };
        capabilities.supports_style_condition = true;
    } else if (assets->config.variant == Qwen3TTSVariant::CustomVoice) {
        capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        capabilities.supports_style_condition = true;
    }
    capabilities.languages = supported_languages(assets->config);

    return std::make_unique<Qwen3TTSLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_tts_loader() {
    return std::make_shared<Qwen3TTSLoader>();
}

}  // namespace engine::models::qwen3_tts
