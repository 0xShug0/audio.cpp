#include "engine/models/voxcpm2/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/voxcpm2/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::voxcpm2 {
namespace {

runtime::CapabilitySet capabilities(const VoxCPM2Assets &) {
  runtime::CapabilitySet out;
  out.supported_tasks = {
      {runtime::VoiceTaskKind::Tts,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
  };
  out.languages = {"Auto"};
  out.supports_speaker_reference = true;
  return out;
}

runtime::ModelMetadata metadata(const VoxCPM2Assets &assets) {
  runtime::ModelMetadata out;
  out.family = "voxcpm2";
  out.variant = assets.config.architecture;
  out.description = "VoxCPM2 loaded from local safetensors assets.";
  return out;
}

runtime::ModelCliInterface cli(const VoxCPM2Assets &) {
  runtime::ModelCliInterface out;
  out.request_options = {
      {"text_chunk_mode", "default|tag_aware|japanese|endline",
       "Text chunking mode; default tag_aware."},
  };
  out.session_options = {
      {"voxcpm2.mem_saver", "true|false",
       "Use tighter graph workspaces and release request runtime graphs; default false."},
      {"voxcpm2.prompt_cache_slots", "n",
       "Prompt and prompt-audio embedding cache slots; default 1."},
  };
  return out;
}

class VoxCPM2Loader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "voxcpm2"; }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_speaker_reference = true;
    return out;
  }

  std::string advertised_instructions_policy() const override {
    return "text_prefix";
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      (void)engine::model_spec::load_resource_bundle(
          request.model_path,
          engine::model_spec::default_spec_path(family()));
      return !request.family_hint.has_value() || *request.family_hint == family();
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto assets =
        load_voxcpm2_assets(request.model_path);
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

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    return load_voxcpm2_model(request.model_path);
  }
};

} // namespace

VoxCPM2LoadedModel::VoxCPM2LoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const VoxCPM2Assets> assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata &VoxCPM2LoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
VoxCPM2LoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
VoxCPM2LoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  if (task.mode != runtime::RunMode::Offline &&
      task.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM2 only supports offline and streaming sessions");
  }
  if (task.task != runtime::VoiceTaskKind::Tts) {
    throw std::runtime_error("VoxCPM2 only supports the Tts task");
  }
  if (task.mode == runtime::RunMode::Streaming) {
    return std::make_unique<VoxCPM2StreamingSession>(task, options, assets_);
  }
  return std::make_unique<VoxCPM2OfflineSession>(task, options, assets_);
}

std::unique_ptr<VoxCPM2LoadedModel>
load_voxcpm2_model(const std::filesystem::path &model_path) {
  auto assets = load_voxcpm2_assets(model_path);
  return std::make_unique<VoxCPM2LoadedModel>(
      metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_voxcpm2_loader() {
  return std::make_shared<VoxCPM2Loader>();
}

} // namespace engine::models::voxcpm2
