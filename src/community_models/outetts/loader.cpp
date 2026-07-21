#include "engine/community_models/outetts/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/community_models/outetts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::outetts {
namespace {

runtime::ModelMetadata metadata(const OuteTTSAssets &) {
  runtime::ModelMetadata out;
  out.family = "outetts";
  out.variant = "1.0-1B";
  out.description =
      "OuteTTS 1.0 1B with native IBM DAC speech synthesis and voice cloning.";
  return out;
}

runtime::CapabilitySet capabilities(const OuteTTSAssets &) {
  runtime::CapabilitySet out;
  out.supported_tasks = {
      {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
      {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
  };
  out.supports_speaker_reference = true;
  out.languages = {
      "Auto",       "Arabic",  "Belarusian", "Bengali",    "Chinese",
      "Dutch",      "English", "French",     "Georgian",   "German",
      "Hungarian",  "Italian", "Japanese",   "Korean",     "Latvian",
      "Lithuanian", "Persian", "Polish",     "Portuguese", "Russian",
      "Spanish",    "Swahili", "Tamil",      "Ukrainian",
  };
  return out;
}

runtime::ModelCliInterface cli(const OuteTTSAssets &) {
  runtime::ModelCliInterface out;
  out.request_options = {
      {"max_tokens", "n",
       "Maximum generated audio tokens per chunk. When omitted, OuteTTS "
       "estimates a safe value from each chunk."},
      {"temperature", "float",
       "Sampling temperature; official cloning default 0.4."},
      {"top_k", "n", "Top-k sampling; official default 40."},
      {"top_p", "float", "Nucleus sampling; official default 0.9."},
      {"min_p", "float",
       "Minimum probability relative to the best token; official default "
       "0.05."},
      {"repetition_penalty", "float",
       "Windowed repetition penalty; official default 1.1."},
      {"repetition_window", "n",
       "Recent-token penalty window; official value 64."},
      {"seed", "n",
       "Sampling seed; cloning defaults to 4099 for native weights and "
       "42 for quantized weights."},
      {"reference_text", "text",
       "Transcript matching the --voice-ref audio for voice cloning."},
      {"reference_language", "code",
       "Language code used to align the reference transcript; default en."},
      {"text_chunk_size", "n",
       "Framework long-form text chunk size; default 256 characters. Chunks "
       "are split further when required by max_tokens."},
      {"text_chunk_mode", "default|tag_aware|japanese|endline",
       "Framework long-form text chunking mode."},
  };
  out.session_options = {
      {"outetts.weight_type", "native|f32|f16|bf16|q8_0",
       "Language-model weight storage type. Quantized CUDA voice cloning "
       "is expanded to F32 in memory for generation correctness."},
      {"outetts.llama_weight_context_mb", "n",
       "Language-model weight context size in MiB."},
      {"outetts.constant_context_mb", "n",
       "Language-model constant tensor context size in MiB."},
      {"outetts.dac_weight_context_mb", "n",
       "DAC decoder weight context size in MiB."},
      {"outetts.dac_graph_context_mb", "n",
       "DAC decoder graph context size in MiB."},
      {"outetts.aligner_model_path", "path",
       "Optional Qwen3 Forced Aligner override. Cloning automatically uses "
       "the aligner embedded in a standalone OuteTTS GGUF when present."},
      {"outetts.reference_cache_slots", "n",
       "Prepared reference-profile cache slots; default 1, set 0 to "
       "disable."},
      {"outetts.mem_saver", "true|false",
       "Release cached-step and aligner runtime state after use; default "
       "false."},
  };
  return out;
}

class OuteTTSLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "outetts"; }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    return out;
  }

  std::vector<runtime::LoaderCompanion> advertised_companions() const override {
    return {
        {
            "forced_aligner",
            "outetts.aligner_model_path",
            "session",
            "qwen3_forced_aligner",
            true,
            {"speaker_reference"},
            "Forced aligner (voice cloning)",
            "",
        },
    };
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      const auto package_spec =
          assets::default_model_package_spec_path(family());
      (void)assets::load_resource_bundle_from_package_spec(request.model_path,
                                                           package_spec);
      return !request.family_hint.has_value() ||
             *request.family_hint == family();
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto model_assets = load_outetts_assets(request.model_path);
    runtime::ModelInspection inspection;
    inspection.model_root = model_assets->resources.model_root();
    inspection.metadata = metadata(*model_assets);
    inspection.capabilities = capabilities(*model_assets);
    inspection.cli = cli(*model_assets);
    const auto package_spec = assets::default_model_package_spec_path(family());
    inspection.discovered_configs =
        runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec,
            assets::ModelPackageResourceKind::Files);
    inspection.discovered_weights =
        runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec,
            assets::ModelPackageResourceKind::Tensors);
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    return load_outetts_model(request.model_path);
  }
};

} // namespace

OuteTTSLoadedModel::OuteTTSLoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const OuteTTSAssets> assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
  if (assets_ == nullptr) {
    throw std::runtime_error("OuteTTS loaded model requires assets");
  }
}

const runtime::ModelMetadata &OuteTTSLoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
OuteTTSLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
OuteTTSLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  if ((task.task != runtime::VoiceTaskKind::Tts &&
       task.task != runtime::VoiceTaskKind::VoiceCloning) ||
      task.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "OuteTTS supports offline TTS and voice cloning only");
  }
  return std::make_unique<OuteTTSSession>(task, options, assets_);
}

std::unique_ptr<OuteTTSLoadedModel>
load_outetts_model(const std::filesystem::path &model_path) {
  auto model_assets = load_outetts_assets(model_path);
  return std::make_unique<OuteTTSLoadedModel>(metadata(*model_assets),
                                              capabilities(*model_assets),
                                              std::move(model_assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_outetts_loader() {
  return std::make_shared<OuteTTSLoader>();
}

} // namespace engine::models::outetts
