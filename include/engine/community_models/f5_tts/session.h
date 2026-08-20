#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::f5_tts {

// F5-TTS community model assets: resource bundle + resolved checkpoint path.
// Weights are loaded lazily by the runtime on first synthesis (graph cache).
struct F5TTSAssets {
    assets::ResourceBundle resources;
    std::filesystem::path checkpoint;  // DiT *.safetensors (ema weights)
};

class F5TTSSession final : public runtime::IOfflineVoiceTaskSession {
public:
    F5TTSSession(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const F5TTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const noexcept override;
    runtime::VoiceTaskKind task_kind() const noexcept override;
    runtime::RunMode run_mode() const noexcept override;
    void prepare(const runtime::SessionPreparationRequest & request) override;

    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::VoiceTaskKind task_kind_;
    runtime::RunMode run_mode_;
    std::shared_ptr<const F5TTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::string vocos_path_;
    std::string dialect_ = "UNK";
    bool use_cuda_ = false;
    int cuda_device_ = 0;
    int threads_ = 0;
};

std::shared_ptr<const F5TTSAssets> load_f5_tts_assets(
    const std::filesystem::path & model_path);

std::shared_ptr<runtime::IVoiceModelLoader> make_f5_tts_loader();

}  // namespace engine::models::f5_tts
