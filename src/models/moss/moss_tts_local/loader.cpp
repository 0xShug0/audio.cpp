#include "engine/models/moss/moss_tts_local/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/moss/moss_tts_local/assets.h"
#include "engine/models/moss/moss_tts_local/session.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_tts_local {
namespace {

runtime::CapabilitySet capabilities(const MossTTSLocalAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    out.languages = {"Auto"};
    out.supports_speaker_reference = true;
    return out;
}

runtime::ModelMetadata metadata(const MossTTSLocalAssets &) {
    runtime::ModelMetadata out;
    out.family = "moss_tts_local";
    out.variant = "MOSS-TTS-Local-Transformer-v1.5";
    out.description = "MOSS-TTS Local Transformer with the MOSS-Audio-Tokenizer-v2 codec.";
    return out;
}

runtime::ModelCliInterface cli(const MossTTSLocalAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"seed", "int", "Sampling seed; omitted chooses a random seed.", false, std::nullopt, "0", "4294967295"},
        {"max_tokens", "n", "Maximum generated audio frames.", false, "4096", "1"},
        {"do_sample", "true|false", "Enable stochastic audio-token sampling.", false, "true"},
        {"temperature", "float", "Audio sampling temperature. Must be finite and positive when sampling. Default: 1.7.", false, "1.7"},
        {"top_p", "float", "Audio-token nucleus sampling limit.", false, "0.8"},
        {"top_k", "n", "Audio-token top-k sampling limit.", false, "25"},
        {"repetition_penalty", "float", "Audio repetition penalty. Must be finite and positive when token history is nonempty. Default: 1.", false, "1"},
        {"text_temperature", "float", "Text sampling temperature. Must be finite and positive when sampling. Default: 1.", false, "1"},
        {"text_top_p", "float", "Text-gate nucleus sampling limit.", false, "1"},
        {"text_top_k", "n", "Text-gate top-k sampling limit.", false, "50"},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Long-form text chunking mode; default splitter.", false, "default"},
        {"text_chunk_size", "int", "Framework long-form text chunk size. Default: 2048.", false, "2048", "1"},
    };
    out.session_options = {
        {"moss_tts_local.weight_type", "auto|native|f32|f16|bf16|q4_0|q4_1|q5_0|q5_1|q2_k|q3_k|q4_k|q5_k|q6_k|q8_0|nvfp4", "Backbone weight storage; auto uses F32 on CPU, BF16 on CUDA, native on other backends. Default: auto.", false, "auto"},
        {"moss_tts_local.reference_cache_slots", "n", "Prepared reference-voice cache slots; default 1, set 0 to disable.", false, "1", "0"},
    };
    return out;
}

class MossTTSLocalLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    MossTTSLocalLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const MossTTSLocalAssets> assets)
        : metadata_(std::move(metadata)),
          capabilities_(std::move(capabilities)),
          assets_(std::move(assets)) {}

    const runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        if (task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("MOSS-TTS-Local only supports offline sessions");
        }
        if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) {
            throw std::runtime_error("MOSS-TTS-Local only supports the Tts and VoiceCloning tasks");
        }
        return std::make_unique<MossTTSLocalSession>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const MossTTSLocalAssets> assets_;
};

class MossTTSLocalLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moss_tts_local";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_moss_tts_local_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto spec_path = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        auto assets = load_moss_tts_local_assets(request.model_path);
        return std::make_unique<MossTTSLocalLoadedModel>(
            metadata(*assets), capabilities(*assets), std::move(assets));
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_local_loader() {
    return std::make_shared<MossTTSLocalLoader>();
}

}  // namespace engine::models::moss_tts_local
