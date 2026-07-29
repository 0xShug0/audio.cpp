#include "engine/models/fun_asr_nano/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/fun_asr_nano/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fun_asr_nano {
namespace {

runtime::ModelMetadata metadata(const FunAsrNanoAssets &assets) {
  runtime::ModelMetadata result;
  result.family = "fun_asr_nano";
  result.variant = assets.config.model_type;
  result.description =
      "Fun-ASR-Nano-2512 offline multilingual speech recognition.";
  return result;
}

runtime::CapabilitySet capabilities(const FunAsrNanoAssets &assets) {
  runtime::CapabilitySet result;
  result.supported_tasks = {
      {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
  result.languages = assets.config.supported_languages;
  result.languages.insert(result.languages.begin(), "auto");
  result.supports_timestamps = false;
  return result;
}

runtime::ModelCliInterface cli() {
  runtime::ModelCliInterface result;
  result.request_options = {
      {"language", "auto|zh|en|ja", "Recognition language.", false, "auto"},
      {"enable_itn", "true|false", "Enable inverse text normalization.", false,
       "true"},
      {"max_tokens", "n", "Maximum generated transcript tokens.", false, "512",
       "1"},
      {"audio_chunk_mode", "auto|fixed|none", "Audio chunking mode.", false,
       "auto"},
      {"audio_chunk_seconds", "seconds", "Fixed audio chunk duration.", false,
       "30", "0"},
  };
  result.session_options = {
      {"fun_asr_nano.weight_type", "native|f32|f16|bf16|q8_0",
       "Shared model weight storage preference."},
      {"fun_asr_nano.encoder_weight_type", "native|f32|f16|bf16|q8_0",
       "Encoder matmul weight storage type."},
      {"fun_asr_nano.adaptor_weight_type", "native|f32|f16|bf16|q8_0",
       "Adaptor matmul weight storage type."},
      {"fun_asr_nano.decoder_weight_type", "native|f32|f16|bf16|q8_0",
       "Decoder matmul weight storage type; CUDA promotes unsafe shared "
       "native/F16/Q8 requests to BF16 unless this option is explicit."},
      {"fun_asr_nano.encoder_graph_arena_mb", "mb",
       "Encoder graph arena size."},
      {"fun_asr_nano.adaptor_graph_arena_mb", "mb",
       "Adaptor graph arena size."},
      {"fun_asr_nano.decoder_prefill_graph_arena_mb", "mb",
       "Decoder prefill graph arena size."},
      {"fun_asr_nano.decoder_decode_graph_arena_mb", "mb",
       "Decoder cached-step graph arena size."},
      {"fun_asr_nano.decoder_weight_context_mb", "mb",
       "Decoder weight context arena size."},
  };
  return result;
}

class FunAsrNanoLoader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "fun_asr_nano"; }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet result;
    result.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}};
    result.languages = {"auto", "zh", "en", "ja"};
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
    const auto assets = load_fun_asr_nano_assets(request.model_path);
    const auto package_spec = engine::model_spec::default_spec_path(family());
    runtime::ModelInspection result;
    result.model_root = assets->resources.model_root();
    result.metadata = metadata(*assets);
    result.capabilities = capabilities(*assets);
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
    return load_fun_asr_nano_model(request.model_path);
  }
};

} // namespace

FunAsrNanoLoadedModel::FunAsrNanoLoadedModel(
    runtime::ModelMetadata metadata, runtime::CapabilitySet capabilities,
    std::shared_ptr<const FunAsrNanoAssets> assets)
    : metadata_(std::move(metadata)), capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata &FunAsrNanoLoadedModel::metadata() const noexcept {
  return metadata_;
}

const runtime::CapabilitySet &
FunAsrNanoLoadedModel::capabilities() const noexcept {
  return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
FunAsrNanoLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  if (task.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("Fun-ASR-Nano only supports the Asr task");
  }
  if (task.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "Fun-ASR-Nano currently supports offline sessions");
  }
  return std::make_unique<FunAsrNanoSession>(task, options, assets_);
}

std::unique_ptr<FunAsrNanoLoadedModel>
load_fun_asr_nano_model(const std::filesystem::path &model_path) {
  auto assets = load_fun_asr_nano_assets(model_path);
  return std::make_unique<FunAsrNanoLoadedModel>(
      metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_fun_asr_nano_loader() {
  return std::make_shared<FunAsrNanoLoader>();
}

} // namespace engine::models::fun_asr_nano
