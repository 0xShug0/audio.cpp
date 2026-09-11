#include "engine/models/moonshine_stt/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/moonshine_stt/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moonshine_stt {
namespace {

runtime::ModelMetadata metadata(const MoonshineAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "moonshine_stt";
    out.variant = assets.config.variant.empty() ? assets.config.model_type : assets.config.variant;
    out.description = "Moonshine streaming ASR loaded from local GGUF or safetensors assets.";
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = {"en"};
    return out;
}

class MoonshineSTTLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moonshine_stt";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
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
        const auto assets = load_moonshine_stt_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities();
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        inspection.cli.request_options = {
            {"max_tokens", "n", "Maximum generated transcript tokens; 0 uses audio-duration derived limit."},
        };
        inspection.cli.session_options = {
            {"moonshine_stt.weight_type", "native|f32|f16", "Encoder/general matmul weight storage type."},
            {"moonshine_stt.decoder_weight_type", "native|f32|f16", "Decoder matmul weight storage type."},
            {"moonshine_stt.conv_weight_type", "native|f32|f16", "Frontend convolution weight storage type."},
            {"moonshine_stt.encoder_gelu", "erf|exact|tanh|quick", "Encoder GELU lowering."},
            {"moonshine_stt.cpu_blas_scheduler", "true|false", "Use BLAS/Accelerate for supported CPU encoder matmuls."},
            {"moonshine_stt.weight_context_mb", "mb", "Weight context arena size."},
            {"moonshine_stt.graph_arena_mb", "mb", "Graph arena size."},
        };
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_moonshine_stt_model(request.model_path);
    }
};

}  // namespace

MoonshineSTTLoadedModel::MoonshineSTTLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MoonshineAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MoonshineSTTLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MoonshineSTTLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MoonshineSTTLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Moonshine STT only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Moonshine STT supports offline and streaming sessions");
    }
    return std::make_unique<MoonshineSTTSession>(task, options, assets_);
}

std::unique_ptr<MoonshineSTTLoadedModel> load_moonshine_stt_model(const std::filesystem::path & model_path) {
    auto assets = load_moonshine_stt_assets(model_path);
    return std::make_unique<MoonshineSTTLoadedModel>(
        metadata(*assets),
        capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_stt_loader() {
    return std::make_shared<MoonshineSTTLoader>();
}

}  // namespace engine::models::moonshine_stt
