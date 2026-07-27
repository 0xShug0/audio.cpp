#include "engine/community_models/parakeet_tdt/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/community_models/parakeet_tdt/session.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

runtime::ModelMetadata metadata(const ParakeetTDTAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "parakeet_tdt";
    out.variant = assets.config.model_type;
    out.description = "NVIDIA Parakeet-TDT 0.6B v3 ASR loaded from local assets.";
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    out.supports_timestamps = true;
    return out;
}

class ParakeetTDTLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "parakeet_tdt"; }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
        out.supports_timestamps = true;
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
        const auto assets = load_parakeet_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities();
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec, engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec, engine::model_spec::ResourceKind::Tensors);
        inspection.cli.request_options = {
            {"max_tokens", "n", "Maximum generated tokens; 0 uses the model-derived limit."},
            {"keep_language_tags", "bool", "Keep language tag tokens in decoded text."},
        };
        inspection.cli.session_options = {
            {"parakeet_tdt.weight_type", "native|f32|f16|bf16|q8_0", "Shared matmul weight storage type."},
            {"parakeet_tdt.matmul_weight_type", "native|f32|f16|bf16|q8_0", "Encoder and decoder matmul weight storage type."},
            {"parakeet_tdt.conv_weight_type", "native|f32|f16", "Convolution weight storage type."},
            {"parakeet_tdt.encoder_flash_attention", "bool", "Run encoder self-attention through the fused flash-attention op; default false (measured slower on the hardware tested)."},
            {"parakeet_tdt.weight_context_mb", "mb", "Weight context arena size."},
            {"parakeet_tdt.encoder_graph_arena_mb", "mb", "Encoder graph arena size."},
            {"parakeet_tdt.decoder_graph_arena_mb", "mb", "Decoder graph arena size."},
        };
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_parakeet_model(request.model_path);
    }
};

}  // namespace

ParakeetTDTLoadedModel::ParakeetTDTLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ParakeetTDTAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & ParakeetTDTLoadedModel::metadata() const noexcept { return metadata_; }
const runtime::CapabilitySet & ParakeetTDTLoadedModel::capabilities() const noexcept { return capabilities_; }

std::unique_ptr<runtime::IVoiceTaskSession> ParakeetTDTLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Parakeet TDT only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Parakeet TDT currently only supports offline sessions");
    }
    return std::make_unique<ParakeetTDTOfflineSession>(task, options, assets_);
}

std::unique_ptr<ParakeetTDTLoadedModel> load_parakeet_model(const std::filesystem::path & model_path) {
    auto assets = load_parakeet_assets(model_path);
    return std::make_unique<ParakeetTDTLoadedModel>(
        metadata(*assets),
        capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_parakeet_tdt_loader() {
    return std::make_shared<ParakeetTDTLoader>();
}

}  // namespace engine::community_models::parakeet_tdt
