#include "engine/models/confucius4_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/confucius4_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::confucius4_tts {
namespace {

runtime::ModelMetadata metadata(const ConfuciusAssets &) {
    runtime::ModelMetadata out;
    out.family = "confucius4_tts";
    out.variant = "Confucius4-TTS";
    out.description = "Confucius4-TTS loaded from prepared local assets.";
    out.config_candidates = {
        "inference_config.yaml",
        "tokenizer_config.json",
        "tokenizer.json",
        "w2v_preprocessor_config.json",
        "bigvgan_config.json",
    };
    out.weight_candidates = {
        "t2s.safetensors",
        "s2a.safetensors",
        "semantic_encoder.safetensors",
        "semantic_stats.safetensors",
        "style_encoder.safetensors",
        "vocoder.safetensors",
        "model.gguf",
    };
    return out;
}

runtime::CapabilitySet capabilities(const ConfuciusAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = false;
    out.languages = {"zh", "en", "ja", "ko", "de", "fr", "es", "id", "it", "th", "pt", "ru", "ms", "vi"};
    return out;
}

runtime::ModelCliInterface cli(const ConfuciusAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"language", "CODE", "Target language code used by the Confucius text frontend."},
        {"temperature", "FLOAT", "T2S sampling temperature."},
        {"top_p", "FLOAT", "T2S nucleus sampling value."},
        {"top_k", "N", "T2S top-k sampling value."},
        {"num_beams", "N", "T2S beam count."},
        {"repetition_penalty", "FLOAT", "T2S repetition penalty."},
        {"max_tokens", "N", "Maximum T2S sequence length."},
        {"num_inference_steps", "N", "S2A flow-matching step count."},
        {"guidance_scale", "FLOAT", "S2A classifier-free guidance rate."},
        {"text_chunk_size", "N", "Maximum tokenizer tokens per text segment."},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode."},
        {"cross_fade_duration", "SECONDS", "Cross-fade duration between generated segments."},
        {"edge_fade_duration", "SECONDS", "Fade duration at segment edges."},
        {"edge_pad_duration", "SECONDS", "Padding duration at segment edges."},
        {"seed", "N", "Random seed for T2S sampling and S2A initialization."},
    };
    out.session_options = {
        {"confucius4_tts.graph_arena_mb", "MiB", "Reusable ggml graph arena size for Confucius stages."},
        {"confucius4_tts.weight_context_mb", "MiB", "Weight loading context size."},
        {"confucius4_tts.weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type; default native."},
        {"confucius4_tts.conv_weight_type", "native|f32|f16", "Convolution weight storage type; default native."},
        {"confucius4_tts.reference_cache_slots", "n", "Prepared reference-audio cache slots; default 1."},
        {"confucius4_tts.mem_saver", "true|false", "Release staged graphs after request phases; default false."},
    };
    return out;
}

class ConfuciusLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "confucius4_tts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = false;
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
        const auto assets = load_confucius_assets(request.model_path);
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
        return load_confucius_model(request.model_path);
    }
};

}  // namespace

ConfuciusLoadedModel::ConfuciusLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ConfuciusAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & ConfuciusLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ConfuciusLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ConfuciusLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<ConfuciusSession>(task, options, assets_);
}

std::unique_ptr<ConfuciusLoadedModel> load_confucius_model(const std::filesystem::path & model_path) {
    auto assets = load_confucius_assets(model_path);
    return std::make_unique<ConfuciusLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader() {
    return std::make_shared<ConfuciusLoader>();
}

}  // namespace engine::models::confucius4_tts
