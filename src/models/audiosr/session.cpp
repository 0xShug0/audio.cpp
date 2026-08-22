#include "engine/models/audiosr/session.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/audiosr/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::audiosr {
namespace {

constexpr const char * kFamily = "audiosr";

std::shared_ptr<const AudioSRAssets> require_assets(std::shared_ptr<const AudioSRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("AudioSR session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType weight_type(const engine::runtime::SessionOptions & options) {
    return engine::runtime::parse_tensor_storage_option(
        options.options,
        "audiosr.weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
            engine::assets::TensorStorageType::BF16,
            engine::assets::TensorStorageType::Q8_0,
        });
}

engine::runtime::ModelMetadata make_metadata(const AudioSRAssets & assets) {
    engine::runtime::ModelMetadata out;
    out.family = kFamily;
    out.variant = assets.variant;
    out.description = "AudioSR latent diffusion audio super-resolution from GGUF.";
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
        {"num_inference_steps", "n", "DDIM inference steps."},
        {"guidance_scale", "float", "Classifier-free guidance scale."},
        {"ddim_eta", "float", "DDIM eta; 1.0 matches the official AudioSR default."},
        {"audio_chunk_duration_sec", "seconds", "Long-audio chunk duration."},
        {"audio_chunk_overlap_sec", "seconds", "Long-audio chunk overlap."},
        {"seed", "n", "Generation seed."},
    };
    out.session_options = {
        {"audiosr.weight_type", "native|f32|f16|bf16|q8_0", "AudioSR weight storage type."},
    };
    return out;
}

class AudioSRLoadedModel final : public engine::runtime::ILoadedVoiceModel {
public:
    explicit AudioSRLoadedModel(std::shared_ptr<const AudioSRAssets> assets)
        : assets_(require_assets(std::move(assets))),
          metadata_(make_metadata(*assets_)),
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
        return std::make_unique<AudioSRSession>(task, options, assets_);
    }

private:
    std::shared_ptr<const AudioSRAssets> assets_;
    engine::runtime::ModelMetadata metadata_;
    engine::runtime::CapabilitySet capabilities_;
};

class AudioSRLoader final : public engine::runtime::IVoiceModelLoader {
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
            (void) load_audiosr_assets(request.model_path);
            return true;
        } catch (...) {
            if (request.family_hint.has_value() && *request.family_hint == family()) {
                throw;
            }
            return false;
        }
    }

    engine::runtime::ModelInspection inspect(const engine::runtime::ModelLoadRequest & request) const override {
        const auto assets = load_audiosr_assets(request.model_path);
        engine::runtime::ModelInspection out;
        out.model_root = assets->model_root;
        out.metadata = make_metadata(*assets);
        out.capabilities = make_capabilities();
        out.cli = make_cli();
        out.discovered_weights = {
            {"weights", assets->gguf_path},
        };
        return out;
    }

    std::unique_ptr<engine::runtime::ILoadedVoiceModel> load(
        const engine::runtime::ModelLoadRequest & request) const override {
        return std::make_unique<AudioSRLoadedModel>(load_audiosr_assets(request.model_path));
    }
};

}  // namespace

AudioSRSession::AudioSRSession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const AudioSRAssets> assets)
    : engine::runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))) {
    if (task_.task != engine::runtime::VoiceTaskKind::AudioGeneration ||
        task_.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("AudioSR supports only offline gen");
    }
    execution_ = std::make_unique<engine::core::ExecutionContext>(options_.backend);
    pipeline_ = std::make_unique<AudioSRPipelineRuntime>(
        assets_,
        *execution_,
        weight_type(options_));
}

AudioSRSession::~AudioSRSession() = default;

std::string AudioSRSession::family() const {
    return kFamily;
}

engine::runtime::VoiceTaskKind AudioSRSession::task_kind() const {
    return task_.task;
}

engine::runtime::RunMode AudioSRSession::run_mode() const {
    return task_.mode;
}

void AudioSRSession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

engine::runtime::TaskResult AudioSRSession::run(const engine::runtime::TaskRequest & request) {
    require_prepared("AudioSR run");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("AudioSR requires --audio input");
    }
    engine::runtime::TaskResult result;
    result.audio_output = pipeline_->run(*request.audio_input, parse_audiosr_options(request.options));
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_audiosr_loader() {
    return std::make_shared<AudioSRLoader>();
}

}  // namespace engine::models::audiosr
