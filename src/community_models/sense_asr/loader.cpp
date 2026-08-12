#include "engine/community_models/sense_asr/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/community_models/sense_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::sense_asr {
namespace {

runtime::ModelMetadata metadata(const SenseAsrAssets &assets) {
  runtime::ModelMetadata result;
  result.family = "sense_asr";
  result.variant = assets.config.model_type;
  result.description =
      "SenseVoice-Small multilingual speech recognition with event, emotion, "
      "and ITN tags via a SAN-M encoder and CTC head.";
  return result;
}

runtime::CapabilitySet capabilities() {
  runtime::CapabilitySet result;
  result.supported_tasks = {
      {runtime::VoiceTaskKind::Asr,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}}};
  result.languages = {"auto", "zh", "en", "yue", "ja", "ko",
                      "pt",  "ru", "es", "it",   "fr", "de",
                      "nl",  "pl", "tr", "ar",   "hi", "vi",
                      "th",  "id", "ms", "fa",   "nospeech"};
  result.supports_timestamps = false;
  return result;
}

runtime::ModelCliInterface cli() {
  runtime::ModelCliInterface result;
  result.request_options = {
      {"language", "auto|zh|en|yue|ja|ko|...",
       "Recognition language, or auto for model inference.", false, "auto"},
      {"enable_itn", "true|false",
       "Enable inverse text normalization (selects the withitn query token).",
       false, "true"},
      {"keep_tags", "true|false",
       "Keep <|event|>/<|emotion|>/<|language|> tags in the text.", false,
       "false"},
      {"audio_chunk_mode", "auto|fixed|none",
       "Audio chunking mode; auto uses bundled silero VAD segmentation.",
       false, "auto"},
{"audio_chunk_duration_sec", "seconds",
        "Max audio chunk duration in seconds.", false, "30", "0"},
  };
  result.session_options = {
      {"sense_asr.weight_type", "native|f32|f16|bf16|q8_0",
       "Shared model weight storage preference."},
      {"sense_asr.encoder_graph_arena_mb", "mb",
       "Encoder graph arena size."},
      {"sense_asr.vad_model_path", "path",
       "Path to the bundled silero VAD model directory."},
  };
  return result;
}

class SenseAsrLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "sense_asr"; }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {
        {runtime::VoiceTaskKind::Asr,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}}};
    result.languages = {"auto", "zh", "en", "yue", "ja", "ko",
                        "pt",  "ru", "es", "it",   "fr", "de",
                        "nl",  "pl", "tr", "ar",   "hi", "vi",
                        "th",  "id", "ms", "fa",   "nospeech"};
    result.supports_timestamps = false;
    return result;
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      if (request.family_hint.has_value() && *request.family_hint != family()) {
        return false;
      }
      (void)engine::model_spec::load_resource_bundle(
          request.model_path, engine::model_spec::default_spec_path(family()));
      return true;
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto assets = load_sense_asr_assets(request.model_path);
    const auto package_spec = engine::model_spec::default_spec_path(family());
    runtime::ModelInspection result;
    result.model_root = assets->resources.model_root();
    result.metadata = metadata(*assets);
    result.capabilities = capabilities();
    result.discovered_configs =
        runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec,
            engine::model_spec::ResourceKind::Files);
    result.discovered_weights =
        runtime::discover_named_assets_from_package_spec(
            request.model_path, package_spec,
            engine::model_spec::ResourceKind::Tensors);
    result.cli = cli();
    return result;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    return load_sense_asr_model(request.model_path);
  }
};

} // namespace

SenseAsrLoadedModel::SenseAsrLoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const SenseAsrAssets> assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata &
SenseAsrLoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
SenseAsrLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
SenseAsrLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  if (task.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("SenseVoice only supports the Asr task");
  }
  if (task.mode != runtime::RunMode::Offline &&
      task.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "SenseVoice supports offline and streaming sessions");
  }
  return std::make_unique<SenseAsrSession>(task, options, assets_);
}

std::unique_ptr<SenseAsrLoadedModel>
load_sense_asr_model(const std::filesystem::path &model_path) {
  auto assets = load_sense_asr_assets(model_path);
  return std::make_unique<SenseAsrLoadedModel>(
      metadata(*assets), capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sense_asr_loader() {
  return std::make_shared<SenseAsrLoader>();
}

} // namespace engine::community_models::sense_asr
