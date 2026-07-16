#include "engine/models/voxcpm2/loader.h"

#include "engine/framework/assets/model_package.h"
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
  };
  return out;
}

class VoxCPM2Loader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "voxcpm2"; }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      (void)engine::assets::load_resource_bundle_from_package_spec(
          request.model_path,
          engine::assets::default_model_package_spec_path(family()));
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
    const auto spec_path = engine::assets::default_model_package_spec_path(family());
    inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
        request.model_path,
        spec_path,
        engine::assets::ModelPackageResourceKind::Files);
    inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
        request.model_path,
        spec_path,
        engine::assets::ModelPackageResourceKind::Tensors);
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
