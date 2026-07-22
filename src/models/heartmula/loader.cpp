#include "engine/models/heartmula/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/heartmula/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::heartmula {
namespace {

runtime::CapabilitySet capabilities(const HeartMuLaAssets &) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"Auto"};
    capabilities.supports_style_condition = true;
    return capabilities;
}

runtime::ModelMetadata metadata(const HeartMuLaAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "heartmula";
    metadata.variant = assets.mula_config.backbone_flavor + "-heartcodec";
    metadata.description = "HeartMuLa text-to-music model loaded from local assets.";
    return metadata;
}

runtime::ModelCliInterface cli(const HeartMuLaAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"lyrics", "text", "Lyrics text."},
        {"tags", "text", "Comma-separated music tags."},
        {"duration_seconds", "seconds", "Maximum generated audio duration."},
        {"temperature", "float", "Audio token sampling temperature."},
        {"top_k", "n", "Audio token top-k sampling limit."},
        {"guidance_scale", "float", "MuLa classifier-free guidance scale."},
        {"codec_duration", "seconds", "Codec detokenization chunk duration."},
        {"num_inference_steps", "n", "Codec flow solver steps."},
        {"codec_guidance_scale", "float", "Codec classifier-free guidance scale."},
        {"infinite_mode", "bool", "Generate long outputs by splitting lyrics into bounded HeartMuLa requests."},
        {"text_chunk_size", "n", "Text chunk size for infinite mode."},
        {"infinite_chunk_audio_length_ms", "n", "Per-chunk audio cap for infinite mode."},
        {"seed", "n", "Torch RNG seed."},
    };
    out.session_options = {
        {"heartmula.weight_type", "native|f32|f16|bf16|q8_0", "MuLa and codec weight storage type."},
        {"heartmula.mula_weight_type", "native|f32|f16|bf16|q8_0", "MuLa weight storage type."},
        {"heartmula.codec_weight_type", "native|f32|f16|bf16|q8_0", "Codec weight storage type."},
        {"heartmula.mula_weight_context_mb", "n", "MuLa weight context size."},
        {"heartmula.codec_weight_context_mb", "n", "Codec weight context size."},
        {"heartmula.mula_constant_context_mb", "n", "MuLa reusable constant context size."},
        {"heartmula.mula_backbone_prefill_graph_arena_mb", "n", "MuLa backbone prefill graph arena size."},
        {"heartmula.mula_backbone_step_graph_arena_mb", "n", "MuLa backbone cached-step graph arena size."},
        {"heartmula.mula_decoder_prefill_graph_arena_mb", "n", "MuLa decoder prefill graph arena size."},
        {"heartmula.mula_decoder_step_graph_arena_mb", "n", "MuLa decoder cached-step graph arena size."},
        {"heartmula.mula_frame_embedding_graph_arena_mb", "n", "MuLa frame-embedding graph arena size."},
        {"heartmula.codec_flow_estimator_graph_arena_mb", "n", "Codec flow-estimator graph arena size."},
        {"heartmula.codec_conditioning_graph_arena_mb", "n", "Codec conditioning graph arena size."},
        {"heartmula.codec_scalar_decoder_graph_arena_mb", "n", "Codec scalar-decoder graph arena size."},
        {"heartmula.mem_saver", "true|false", "Release staged runtime graphs after each request; default false."},
    };
    return out;
}

class HeartMuLaLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "heartmula";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
        };
        out.supports_style_condition = true;
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
        const auto assets = load_heartmula_assets(request.model_path);
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
        return load_heartmula_model(request.model_path);
    }
};

}  // namespace

HeartMuLaLoadedModel::HeartMuLaLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HeartMuLaAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & HeartMuLaLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HeartMuLaLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HeartMuLaLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("HeartMuLa only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("HeartMuLa only supports the gen task");
    }
    return std::make_unique<HeartMuLaSession>(task, options, assets_);
}

std::unique_ptr<HeartMuLaLoadedModel> load_heartmula_model(const std::filesystem::path & model_path) {
    auto assets = load_heartmula_assets(model_path);
    return std::make_unique<HeartMuLaLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_heartmula_loader() {
    return std::make_shared<HeartMuLaLoader>();
}

}  // namespace engine::models::heartmula
