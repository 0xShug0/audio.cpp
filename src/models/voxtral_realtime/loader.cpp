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
        {"max_tokens", "int", "Maximum generated transcript tokens; omitted uses the audio-derived limit. Offline decoding only; ignored by live streaming.", false, std::nullopt, "1"},
        {"do_sample", "bool", "Enable sampling instead of greedy decode.", false, "false"},
        {"temperature", "float", "Sampling temperature; must be > 0.", false, "1", "0"},
        {"top_p", "float", "Nucleus sampling probability in (0, 1].", false, "1", "0", "1"},
        {"top_k", "n", "Top-k sampling limit; 0 disables top-k.", false, "50", "0"},
        {"seed", "n", "Sampling seed.", false, "1234", "0"},
    };
    out.session_options = {
        {"voxtral_realtime.weight_type", "native|f32|f16|bf16|q4_0|q4_1|q5_0|q5_1|q2_k|q3_k|q4_k|q5_k|q6_k|q8_0|nvfp4", "Shared weight storage type. Default: native.", false, "native"},
        {"voxtral_realtime.audio_encoder_weight_type", "native|f32|f16|bf16|q4_0|q4_1|q5_0|q5_1|q2_k|q3_k|q4_k|q5_k|q6_k|q8_0|nvfp4", "Audio encoder weight storage; inherits weight_type."},
        {"voxtral_realtime.text_decoder_weight_type", "native|f32|f16|bf16|q4_0|q4_1|q5_0|q5_1|q2_k|q3_k|q4_k|q5_k|q6_k|q8_0|nvfp4", "Text decoder weight storage; inherits weight_type."},
        {"voxtral_realtime.audio_encoder_graph_arena_mb", "mb", "Audio encoder graph arena size.", false, "512", "1"},
        {"voxtral_realtime.audio_encoder_weight_context_mb", "mb", "Audio encoder weight context arena size.", false, "128", "1"},
        {"voxtral_realtime.text_decoder_prefill_graph_arena_mb", "int", "Decoder prefill graph arena in MiB. Decoder uses the maximum of prefill and decode arena settings. Default: 512.", false, "512", "1"},
        {"voxtral_realtime.text_decoder_decode_graph_arena_mb", "int", "Decoder step graph arena in MiB. Decoder uses the maximum of prefill and decode arena settings. Default: 512.", false, "512", "1"},
        {"voxtral_realtime.text_decoder_weight_context_mb", "mb", "Text decoder weight context arena size.", false, "128", "1"},
        {"voxtral_realtime.stream_decode_cache_steps", "int", "Streaming decoder cache capacity in 80 ms steps. Must be positive and no greater than checkpoint sliding_window. Default: 1024.", false, "1024"},
        {"voxtral_realtime.stream_batch_tokens", "int", "Audio tokens per streaming encoder forward. Default: 1.", false, "1", "1"},
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
