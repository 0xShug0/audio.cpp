#include "engine/community_models/moss_tts_v15/loader.h"

#include "engine/community_models/moss_tts_v15/session.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_v15 {
namespace {

runtime::ModelMetadata metadata(const Assets & assets) {
    runtime::ModelMetadata out;
    out.family = "moss_tts_v15";
    out.variant = std::to_string(assets.config.num_codebooks) + "vq";
    out.description = "MOSS-TTS-v1.5: 8B delay-pattern TTS with zero-shot voice cloning.";
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    // Cloning from a reference recording is what this checkpoint adds over voice design,
    // and on the reference it is the capability that works most reliably. The style tag
    // is still accepted, but see the docs: instructions describing a speaker are followed
    // only loosely.
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    out.languages = {"en", "zh"};
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"min_frames", "<int>", "Minimum generated audio frames per text chunk; nonpositive values derive a text-length-based floor. Separate from duration_frames conditioning.", false, "0"},
        {"max_frames", "<int>", "Generated audio-frame ceiling per text chunk; nonpositive values derive a text-length-based ceiling. Runtime raises it to at least the effective min_frames plus the number of codebooks.", false, "0"},
        {"instruct", "<text>", "Voice description. Followed only loosely on this checkpoint; a reference recording is far more reliable."},
        {"duration_frames", "int", "Duration budget in codec frames at 12.5 a second, the model's own '- Tokens:' field. Nonpositive values disable duration conditioning.", false, "0"},
        {"text_temperature", "float", "Text-token sampling temperature. Must be finite and positive when sampling. Default: 1.5.", false, "1.5"},
        {"language", "English|Chinese", "Full language name; the model does not understand codes like 'en'."},
        {"seed", "int", "Reproduces a take exactly. Parsed as signed 32-bit integer then converted to uint32; negative values wrap.", false, "0"},
        {"temperature", "float", "Audio sampling temperature. Must be finite and positive when sampling.", false, "1.5"},
        {"top_p", "<float>", "Audio nucleus sampling cutoff (default 0.6).", false, "0.6"},
        {"top_k", "<int>", "Audio top-k (default 50).", false, "50"},
        {"repetition_penalty", "float", "Audio repetition penalty. Must be finite and positive when history is nonempty.", false, "1.1"},
        {"text_chunk_size", "<int>", "Maximum Unicode codepoints per long-form text chunk (default 200).", false, "200", "1"},
        {"text_chunk_mode", "word_budget|tag_aware|japanese|endline", "Text chunking policy; word_budget is the default and also accepts the legacy name default.", false, "word_budget"},
    };
    out.session_options = {
        {"moss_tts_v15.weight_type", "native|f32|bf16|q8_0", "Weight storage; default native preserves packaged quantization. f16 is rejected because this backbone produces NaN in it.", false, "native"},
    };
    return out;
}

class Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moss_tts_v15";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
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
        const auto assets = load_moss_tts_v15_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities();
        inspection.cli = cli();
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
        return load_moss_tts_v15_model(request.model_path);
    }
};

}  // namespace

LoadedModel::LoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Assets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & LoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & LoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> LoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MOSS-TTS-v1.5 only supports offline sessions");
    }
    // One path serves both: cloning is tts with a reference recording attached.
    if (task.task != runtime::VoiceTaskKind::Tts
        && task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("MOSS-TTS-v1.5 supports the tts and clone tasks");
    }
    return std::make_unique<MossTtsV15Session>(task, options, assets_);
}

std::unique_ptr<LoadedModel> load_moss_tts_v15_model(const std::filesystem::path & model_path) {
    auto assets = load_moss_tts_v15_assets(model_path);
    return std::make_unique<LoadedModel>(metadata(*assets), capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_v15_loader() {
    return std::make_shared<Loader>();
}

}  // namespace engine::models::moss_tts_v15
