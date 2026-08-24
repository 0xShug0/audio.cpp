#include "engine/models/controlfoley/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/models/controlfoley/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::controlfoley {
namespace {

constexpr const char * kFamily = "controlfoley";

std::shared_ptr<const ControlFoleyAssets> require_assets(
    std::shared_ptr<const ControlFoleyAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("ControlFoley session requires assets");
    }
    return assets;
}

engine::runtime::ModelMetadata make_metadata() {
    engine::runtime::ModelMetadata out;
    out.family = kFamily;
    out.variant = "large_44k";
    out.description = "ControlFoley multimodal Foley generation from GGUF.";
    return out;
}

engine::runtime::CapabilitySet make_capabilities() {
    engine::runtime::CapabilitySet out;
    out.supported_tasks = {
        {engine::runtime::VoiceTaskKind::AudioGeneration, {engine::runtime::RunMode::Offline}},
    };
    return out;
}

engine::runtime::ModelCliInterface make_cli() {
    engine::runtime::ModelCliInterface out;
    out.request_options = {
        {"duration_sec", "seconds", "Target temporal budget."},
        {"num_inference_steps", "n", "Euler flow inference steps."},
        {"cfg_strength", "scale", "Classifier-free guidance strength."},
        {"negative_text", "text", "Negative text prompt for CFG."},
        {"video", "path", "Optional video path for video-conditioned Foley generation."},
        {"mask_away_clip", "true|false", "Disable CLIP video conditioning."},
        {"seed", "n", "Generation seed."},
    };
    return out;
}

engine::assets::TensorStorageType weight_type(const engine::runtime::SessionOptions & options) {
    return engine::runtime::parse_tensor_storage_option(
        options.options,
        "controlfoley.weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
            engine::assets::TensorStorageType::BF16,
            engine::assets::TensorStorageType::Q8_0,
        });
}

class ControlFoleyLoadedModel final : public engine::runtime::ILoadedVoiceModel {
public:
    explicit ControlFoleyLoadedModel(std::shared_ptr<const ControlFoleyAssets> assets)
        : assets_(require_assets(std::move(assets))),
          metadata_(make_metadata()),
          capabilities_(make_capabilities()) {}

    const engine::runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const engine::runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<engine::runtime::IVoiceTaskSession> create_task_session(
        const engine::runtime::TaskSpec & task,
        const engine::runtime::SessionOptions & options) const override {
        return std::make_unique<ControlFoleySession>(task, options, assets_);
    }

private:
    std::shared_ptr<const ControlFoleyAssets> assets_;
    engine::runtime::ModelMetadata metadata_;
    engine::runtime::CapabilitySet capabilities_;
};

class ControlFoleyLoader final : public engine::runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return kFamily;
    }

    engine::runtime::CapabilitySet advertised_capabilities() const override {
        return make_capabilities();
    }

    bool can_load(const engine::runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) load_controlfoley_assets(request.model_path);
            return true;
        } catch (...) {
            if (request.family_hint.has_value() && *request.family_hint == family()) {
                throw;
            }
            return false;
        }
    }

    engine::runtime::ModelInspection inspect(const engine::runtime::ModelLoadRequest & request) const override {
        const auto assets = load_controlfoley_assets(request.model_path);
        engine::runtime::ModelInspection out;
        out.model_root = assets->model_root;
        out.metadata = make_metadata();
        out.capabilities = make_capabilities();
        out.cli = make_cli();
        out.discovered_weights = {
            {"weights", assets->gguf_path},
        };
        return out;
    }

    std::unique_ptr<engine::runtime::ILoadedVoiceModel> load(
        const engine::runtime::ModelLoadRequest & request) const override {
        return std::make_unique<ControlFoleyLoadedModel>(load_controlfoley_assets(request.model_path));
    }
};

}  // namespace

ControlFoleySession::ControlFoleySession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const ControlFoleyAssets> assets)
    : engine::runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))) {
    if (task_.task != engine::runtime::VoiceTaskKind::AudioGeneration ||
        task_.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("ControlFoley supports only offline gen");
    }
    execution_ = std::make_unique<engine::core::ExecutionContext>(options_.backend);
    pipeline_ = std::make_unique<ControlFoleyPipelineRuntime>(
        assets_,
        *execution_,
        weight_type(options_));
}

ControlFoleySession::~ControlFoleySession() = default;

std::string ControlFoleySession::family() const {
    return kFamily;
}

engine::runtime::VoiceTaskKind ControlFoleySession::task_kind() const {
    return task_.task;
}

engine::runtime::RunMode ControlFoleySession::run_mode() const {
    return task_.mode;
}

void ControlFoleySession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

engine::runtime::TaskResult ControlFoleySession::run(const engine::runtime::TaskRequest & request) {
    require_prepared("ControlFoley run");
    engine::runtime::TaskResult result;
    result.audio_output = pipeline_->run(request);
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_controlfoley_loader() {
    return std::make_shared<ControlFoleyLoader>();
}

}  // namespace engine::models::controlfoley
