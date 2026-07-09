#include "engine/models/voxcpm2/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/voxcpm2/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::voxcpm2 {
namespace {

std::filesystem::path
resolve_model_root(const std::filesystem::path &model_path) {
  if (engine::io::is_existing_directory(model_path)) {
    return std::filesystem::weakly_canonical(model_path);
  }
  if (engine::io::is_existing_file(model_path)) {
    return std::filesystem::weakly_canonical(model_path.parent_path());
  }
  throw std::runtime_error("VoxCPM2 model path does not exist: " +
                           model_path.string());
}

bool has_voxcpm2_assets(const std::filesystem::path &root) {
  return engine::io::is_existing_file(root / "config.json") &&
         engine::io::is_existing_file(root / "model.safetensors") &&
         engine::io::is_existing_file(root / "audiovae.safetensors") &&
         engine::io::is_existing_file(root / "tokenizer_config.json") &&
         engine::io::is_existing_file(root / "tokenizer.json") &&
         engine::io::is_existing_file(root / "special_tokens_map.json");
}

std::vector<runtime::NamedAsset>
discover_config_assets(const runtime::ModelLoadRequest &request) {
  const auto root = resolve_model_root(request.model_path);
  return runtime::discover_named_assets(
      root, {"config.json", "tokenizer_config.json", "tokenizer.json",
             "special_tokens_map.json"});
}

std::vector<runtime::NamedAsset>
discover_weight_assets(const runtime::ModelLoadRequest &request) {
  const auto root = resolve_model_root(request.model_path);
  return runtime::discover_named_assets(
      root, {"model.safetensors", "audiovae.safetensors"});
}

runtime::CapabilitySet voxcpm2_capabilities() {
  runtime::CapabilitySet capabilities;
  capabilities.supported_tasks = {
      {runtime::VoiceTaskKind::Tts,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
  };
  capabilities.languages = {"Auto"};
  capabilities.supports_speaker_reference = true;
  return capabilities;
}

runtime::ModelMetadata voxcpm2_metadata(const VoxCPM2Assets &assets) {
  runtime::ModelMetadata metadata;
  metadata.family = "voxcpm2";
  metadata.variant = assets.config.architecture;
  metadata.description = "VoxCPM2 loaded from local safetensors assets.";
  metadata.config_candidates = {
      "config.json",
      "tokenizer_config.json",
      "tokenizer.json",
      "special_tokens_map.json",
  };
  metadata.weight_candidates = {"model.safetensors", "audiovae.safetensors"};
  return metadata;
}

class VoxCPM2Loader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "voxcpm2"; }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      const auto root = resolve_model_root(request.model_path);
      return has_voxcpm2_assets(root) && (!request.family_hint.has_value() ||
                                          *request.family_hint == family());
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto assets =
        load_voxcpm2_assets(resolve_model_root(request.model_path));
    runtime::ModelInspection inspection;
    inspection.model_root = assets->paths.model_root;
    inspection.metadata = voxcpm2_metadata(*assets);
    inspection.capabilities = voxcpm2_capabilities();
    inspection.cli.session_options = {
        {"voxcpm2.mem_saver", "true|false",
         "Use tighter graph workspaces and release request runtime graphs; default false."},
    };
    inspection.discovered_configs = discover_config_assets(request);
    inspection.discovered_weights = discover_weight_assets(request);
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    return load_voxcpm2_model(resolve_model_root(request.model_path));
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
      voxcpm2_metadata(*assets), voxcpm2_capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_voxcpm2_loader() {
  return std::make_shared<VoxCPM2Loader>();
}

} // namespace engine::models::voxcpm2
