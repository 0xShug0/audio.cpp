#include "engine/models/hviske_asr/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/hviske_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

runtime::ModelMetadata metadata(const HviskeASRAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "hviske_asr";
    out.variant = assets.config.variant.empty() ? assets.config.model_type : assets.config.variant;
    out.description = "Hviske/Cohere ASR loaded from local tensor assets.";
    return out;
}

runtime::CapabilitySet capabilities(const HviskeASRAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    out.languages = assets.config.supported_languages;
    out.supports_timestamps = false;
    return out;
}

runtime::ModelCliInterface cli(const HviskeASRAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"language", "code", "ASR language code; defaults to da."},
        {"punctuation", "bool", "Enable or disable punctuation tokens in the decoder prompt."},
        {"max_tokens", "n", "Maximum generated transcript tokens."},
        {"num_beams", "n", "Beam-search beam count; 1 uses greedy or sampling decode."},
        {"length_penalty", "float", "Beam-search length penalty."},
        {"do_sample", "bool", "Enable sampling instead of greedy decode when num_beams is 1."},
        {"temperature", "float", "Sampling temperature."},
        {"top_k", "n", "Top-k sampling limit; 0 disables top-k."},
        {"top_p", "float", "Nucleus sampling limit."},
        {"seed", "n", "Sampling seed."},
    };
    out.session_options = {
        {"hviske_asr.weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type."},
        {"hviske_asr.conv_weight_type", "native|f32|f16", "Convolution weight storage type."},
        {"hviske_asr.weight_context_mb", "mb", "Weight context arena size."},
        {"hviske_asr.encoder_graph_arena_mb", "mb", "Encoder graph arena size."},
        {"hviske_asr.decoder_prefill_graph_arena_mb", "mb", "Decoder prefill graph arena size."},
        {"hviske_asr.decoder_decode_graph_arena_mb", "mb", "Decoder cached-step graph arena size."},
    };
    return out;
}

class HviskeASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "hviske_asr";
    }

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
        const auto assets = load_hviske_asr_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        inspection.cli = cli(*assets);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_hviske_asr_model(request.model_path);
    }
};

}  // namespace

HviskeASRLoadedModel::HviskeASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HviskeASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & HviskeASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HviskeASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HviskeASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Hviske ASR only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Hviske ASR currently supports offline sessions");
    }
    return std::make_unique<HviskeASRSession>(task, options, assets_);
}

std::unique_ptr<HviskeASRLoadedModel> load_hviske_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_hviske_asr_assets(model_path);
    return std::make_unique<HviskeASRLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_hviske_asr_loader() {
    return std::make_shared<HviskeASRLoader>();
}

}  // namespace engine::models::hviske_asr
