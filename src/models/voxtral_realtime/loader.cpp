#include "engine/models/voxtral_realtime/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/voxtral_realtime/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::voxtral_realtime {
namespace {

runtime::ModelMetadata metadata(const VoxtralRealtimeAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "voxtral_realtime";
    out.variant = assets.config.model_type + "-" + assets.config.dtype;
    out.description = "Mistral VoxTral Mini Realtime ASR loaded from local assets.";
    return out;
}

runtime::CapabilitySet capabilities(const VoxtralRealtimeAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = assets.config.supported_languages;
    out.supports_timestamps = false;
    return out;
}

runtime::ModelCliInterface cli(const VoxtralRealtimeAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"max_new_tokens", "n", "Maximum generated transcript tokens."},
        {"do_sample", "bool", "Enable sampling instead of greedy decode."},
        {"temperature", "float", "Sampling temperature."},
        {"top_p", "float", "Nucleus sampling limit."},
        {"top_k", "n", "Top-k sampling limit; 0 disables top-k."},
        {"seed", "n", "Sampling seed."},
    };
    out.session_options = {
        {"voxtral_realtime.weight_type", "native|f32|f16|bf16|q8_0", "Shared matmul weight storage type."},
        {"voxtral_realtime.audio_encoder_weight_type", "native|f32|f16|bf16|q8_0",
         "Audio encoder matmul weight storage type."},
        {"voxtral_realtime.text_decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder matmul weight storage type."},
        {"voxtral_realtime.audio_encoder_graph_arena_mb", "mb", "Audio encoder graph arena size."},
        {"voxtral_realtime.audio_encoder_weight_context_mb", "mb", "Audio encoder weight context arena size."},
        {"voxtral_realtime.text_decoder_prefill_graph_arena_mb", "mb", "Text decoder prefill graph arena size."},
        {"voxtral_realtime.text_decoder_decode_graph_arena_mb", "mb", "Text decoder cached-step graph arena size."},
        {"voxtral_realtime.text_decoder_weight_context_mb", "mb", "Text decoder weight context arena size."},
    };
    return out;
}

class VoxtralRealtimeLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "voxtral_realtime";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_voxtral_realtime_assets(request.model_path);
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
        return load_voxtral_realtime_model(request.model_path);
    }
};

}  // namespace

VoxtralRealtimeLoadedModel::VoxtralRealtimeLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VoxtralRealtimeAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VoxtralRealtimeLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VoxtralRealtimeLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> VoxtralRealtimeLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("VoxTral realtime only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VoxTral realtime supports offline and streaming sessions");
    }
    return std::make_unique<VoxtralRealtimeSession>(task, options, assets_);
}

std::unique_ptr<VoxtralRealtimeLoadedModel> load_voxtral_realtime_model(const std::filesystem::path & model_path) {
    auto assets = load_voxtral_realtime_assets(model_path);
    return std::make_unique<VoxtralRealtimeLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_voxtral_realtime_loader() {
    return std::make_shared<VoxtralRealtimeLoader>();
}

}  // namespace engine::models::voxtral_realtime
