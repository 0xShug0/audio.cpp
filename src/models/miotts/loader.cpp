#include "engine/models/miotts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/miotts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::miotts {
namespace {

runtime::ModelMetadata metadata(const MioTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "miotts";
    out.variant = assets.config.model_type;
    out.description = "MioTTS loaded from local Qwen3 assets with MioCodec decoding.";
    return out;
}

runtime::CapabilitySet capabilities(const MioTTSAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_timestamps = false;
    return out;
}

runtime::ModelCliInterface cli(const MioTTSAssets & assets) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"max_tokens", "n", "Maximum generated tokens; defaults to the checkpoint config.", false, std::to_string(assets.config.max_tokens), "1"},
        {"top_k", "int", "Top-k sampling limit; defaults to the checkpoint config.", false, std::to_string(assets.config.top_k)},
        {"top_p", "float", "Nucleus sampling cutoff; defaults to the checkpoint config.", false, std::to_string(assets.config.top_p), "0", "1"},
        {"temperature", "float", "Sampling temperature; defaults to the checkpoint config.", false, std::to_string(assets.config.temperature), "0", "2"},
        {"repetition_penalty", "float", "Token repetition penalty; defaults to the checkpoint config.", false, std::to_string(assets.config.repetition_penalty), "1", "1.5"},
        {"do_sample", "true|false", "Enable sampling; defaults to the checkpoint config.", false, assets.config.do_sample ? "true" : "false"},
        {"seed", "int", "Sampling seed; omitted chooses a random seed.", false, std::nullopt, "0", "4294967295"},
        {"best_of_n", "n", "Generate n candidates for this request and select by ASR scoring.", false, std::nullopt, "1"},
        {"best_of_n_enabled", "true|false", "Enable best-of-N candidate selection for this request."},
        {"best_of_n_language", "auto|en|ja", "Language used when scoring best-of-N candidates."},
        {"presence_penalty", "float", "Generation control; defaults to the checkpoint config.", false, std::to_string(assets.config.presence_penalty), "0", "1"},
        {"frequency_penalty", "float", "Generation control; defaults to the checkpoint config.", false, std::to_string(assets.config.frequency_penalty), "0", "1"},
        {"text_chunk_size", "int", "Framework long-form text chunk size; inherits the session setting.", false, std::nullopt, "1"},
    };
    out.session_options = {
        {"miotts.codec_model_path", "dir", "MioCodec model directory used by MioTTS acoustic decoding."},
        {"miotts.best_of_n_enabled", "true|false", "Enable best-of-N candidate selection by default.", false, "false"},
        {"miotts.best_of_n_default", "n", "Default best-of-N candidate count.", false, "1", "1"},
        {"miotts.best_of_n_max", "n", "Maximum best-of-N candidate count.", false, "8", "1"},
        {"miotts.best_of_n_language", "auto|en|ja", "Default language used when scoring best-of-N candidates.", false, "auto"},
        {"miotts.best_of_n_asr_model_path", "dir", "Qwen3-ASR model directory used by best-of-N scoring."},
        {"miotts.weight_type", "native|f32|f16|bf16|q8_0", "MioTTS language-model matmul weight storage type."},
        {"miotts.text_chunk_size", "int", "Default framework text chunk size.", false, "180"},
        {"miotts.prefill_graph_arena_mb", "int", "Context or graph arena size in MiB. Default: 1024.", false, "1024", "1"},
        {"miotts.decode_graph_arena_mb", "int", "Context or graph arena size in MiB. Default: 1024.", false, "1024", "1"},
        {"miotts.weight_context_mb", "int", "Context or graph arena size in MiB. Default: 512.", false, "512", "1"},
        {"miotts.codec_weight_context_mb", "int", "Context or graph arena size in MiB. Default: 256.", false, "256", "1"},
        {"miotts.codec_constant_context_mb", "int", "Context or graph arena size in MiB. Default: 256.", false, "256", "1"},
        {"miotts.global_graph_arena_mb", "int", "Context or graph arena size in MiB. Default: 256.", false, "256", "1"},
        {"miotts.wave_graph_arena_mb", "int", "Context or graph arena size in MiB. Default: 512.", false, "512", "1"},
    };
    return out;
}

class MioTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "miotts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_miotts_assets(request.model_path);
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
        return load_miotts_model(request.model_path);
    }
};

}  // namespace

MioTTSLoadedModel::MioTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MioTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MioTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MioTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MioTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("MioTTS only supports the Tts task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioTTS currently supports offline sessions");
    }
    return std::make_unique<MioTTSSession>(task, options, assets_);
}

std::unique_ptr<MioTTSLoadedModel> load_miotts_model(const std::filesystem::path & model_path) {
    auto assets = load_miotts_assets(model_path);
    return std::make_unique<MioTTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_miotts_loader() {
    return std::make_shared<MioTTSLoader>();
}

}  // namespace engine::models::miotts
