#include "engine/community_models/xtts_v2/loader.h"
#include "engine/community_models/xtts_v2/assets.h"
#include "engine/community_models/xtts_v2/session.h"

namespace engine::models::xtts_v2 {
namespace {

runtime::ModelMetadata make_metadata() {
    runtime::ModelMetadata out;
    out.family = "xtts_v2";
    out.description = "xTTS-v2 (Coqui) cross-lingual TTS engine.";
    return out;
}

runtime::CapabilitySet make_capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {{runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}}};
    out.supports_speaker_reference = true;
    out.languages = {"English", "German", "French", "Spanish", "Italian",
        "Portuguese", "Polish", "Turkish", "Russian", "Dutch",
        "Czech", "Arabic", "Hungarian", "Chinese", "Japanese",
        "Korean", "Hindi"};
    return out;
}

}  // namespace

class XttSv2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "xtts_v2"; }

    bool can_load(const runtime::ModelLoadRequest& request) const override {
        std::filesystem::path g, v;
        return discover_xtts_v2_files(request.model_path, g, v);
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest& request) const override {
        runtime::ModelInspection out;
        out.metadata = make_metadata();
        out.capabilities = make_capabilities();
        out.cli.session_options = {
            {"temperature", "float", "Sampling temperature (default 0.75)"},
            {"top_k", "n", "Top-k sampling (default 50)"},
            {"top_p", "float", "Top-p sampling (default 0.85)"},
            {"repetition_penalty", "float", "Repetition penalty (default 5.0)"},
            {"max_mel_tokens", "n", "Max mel tokens (default 1500)"},
            {"language", "text", "Source language code"},
            {"n_threads", "n", "Number of computation threads"},
        };
        return out;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest& request) const override {
        auto assets = load_xtts_v2_assets(request.model_path);
        if (!assets) return nullptr;
        auto ptr = std::make_shared<const XttSv2Assets>(std::move(*assets));
        return std::make_unique<XttSv2LoadedModel>(std::move(ptr));
    }

    runtime::CapabilitySet advertised_capabilities() const override { return make_capabilities(); }
};

XttSv2LoadedModel::XttSv2LoadedModel(std::shared_ptr<const XttSv2Assets> assets)
    : assets_(std::move(assets)) {
    metadata_ = make_metadata();
    caps_ = make_capabilities();
}

const runtime::ModelMetadata& XttSv2LoadedModel::metadata() const noexcept { return metadata_; }
const runtime::CapabilitySet& XttSv2LoadedModel::capabilities() const noexcept { return caps_; }

std::unique_ptr<runtime::IVoiceTaskSession> XttSv2LoadedModel::create_task_session(
    const runtime::TaskSpec& task, const runtime::SessionOptions& options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) return nullptr;
    return std::make_unique<XttSv2TtsSession>(assets_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_xtts_v2_loader() { return std::make_shared<XttSv2Loader>(); }

}  // namespace engine::models::xtts_v2
