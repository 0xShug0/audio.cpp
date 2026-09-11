#include "engine/models/marblenet_vad/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::marblenet_vad {
namespace {

std::filesystem::path resolve_weight_path(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("MarbleNet VAD model path does not exist: " + model_path.string());
    }
    const auto candidate = model_path / "marblenet_vad.safetensors";
    if (!engine::io::is_existing_file(candidate)) {
        throw std::runtime_error("MarbleNet VAD weights not found: " + candidate.string());
    }
    return std::filesystem::weakly_canonical(candidate);
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    if (engine::io::is_existing_file(request.model_path)) {
        return {{"default", std::filesystem::weakly_canonical(request.model_path)}};
    }
    return runtime::discover_named_assets(
        std::filesystem::weakly_canonical(request.model_path),
        {"marblenet_vad.safetensors"});
}

engine::assets::TensorStorageType marblenet_weight_type_from_options(const runtime::SessionOptions & options) {
    const auto it = options.options.find("marblenet_vad.weight_type");
    if (it == options.options.end()) {
        return engine::assets::TensorStorageType::Native;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("marblenet_vad.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

class MarbleNetVADLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "marblenet_vad";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline}},
        };
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) resolve_marblenet_assets(resolve_weight_path(request.model_path));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("MarbleNet VAD does not expose selectable config assets");
        }
        const auto weight_path = resolve_weight_path(request.model_path);
        const auto assets = resolve_marblenet_assets(weight_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = weight_path.stem().string();
        inspection.metadata.description = "MarbleNet VAD loaded from safetensors weights.";
        inspection.metadata.weight_candidates = {"marblenet_vad.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_timestamps = true;
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_marblenet_vad_model(request);
    }
};

}  // namespace

MarbleNetVADSession::MarbleNetVADSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MarbleNetWeights> weights)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      weight_storage_type_(marblenet_weight_type_from_options(RuntimeSessionBase::options())),
      runtime_(std::move(weights), execution_context(), weight_storage_type_) {
    if (task_.task != runtime::VoiceTaskKind::Vad) {
        throw std::runtime_error("MarbleNet VAD only supports VoiceTaskKind::Vad");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MarbleNet VAD only supports offline mode");
    }
}

std::string MarbleNetVADSession::family() const {
    return "marblenet_vad";
}

runtime::VoiceTaskKind MarbleNetVADSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MarbleNetVADSession::run_mode() const {
    return task_.mode;
}

void MarbleNetVADSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("MarbleNet VAD prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult MarbleNetVADSession::run(const runtime::TaskRequest & request) {
    require_prepared("MarbleNet VAD run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("MarbleNet VAD run() requires audio_input");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    const float threshold = runtime::parse_float_option(
        request.options,
        {"marblenet_vad.threshold", "threshold"}).value_or(0.5f);
    runtime::TaskResult result;
    result.speech_segments = runtime_.detect_speech(*request.audio_input, threshold);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

MarbleNetVADLoadedModel::MarbleNetVADLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MarbleNetWeights> weights)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      weights_(std::move(weights)) {}

const runtime::ModelMetadata & MarbleNetVADLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MarbleNetVADLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MarbleNetVADLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<MarbleNetVADSession>(task, options, weights_);
}

std::unique_ptr<MarbleNetVADLoadedModel> load_marblenet_vad_model(const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::runtime_error("MarbleNet VAD does not expose selectable config assets");
    }
    const auto weights = discover_weight_assets(request);
    const auto * selected_weight = runtime::select_named_asset(weights, request.weight_id, "weight");
    const auto weight_path = selected_weight != nullptr ? selected_weight->path : resolve_weight_path(request.model_path);
    const auto assets = resolve_marblenet_assets(weight_path);
    runtime::ModelMetadata metadata;
    metadata.family = "marblenet_vad";
    metadata.variant = weight_path.stem().string();
    metadata.description = "MarbleNet VAD loaded from safetensors weights.";
    metadata.weight_candidates = {"marblenet_vad.safetensors"};
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline}},
    };
    capabilities.supports_timestamps = true;
    return std::make_unique<MarbleNetVADLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        load_marblenet_weights_cached(assets.checkpoint_path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_marblenet_vad_loader() {
    return std::make_shared<MarbleNetVADLoader>();
}

}  // namespace engine::models::marblenet_vad
