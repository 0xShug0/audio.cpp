#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/roformer/assets.h"
#include "engine/models/roformer/runtime.h"

#include <memory>

namespace engine::models::roformer {

std::shared_ptr<runtime::IVoiceModelLoader> make_mel_band_roformer_loader();
std::shared_ptr<runtime::IVoiceModelLoader> make_bs_roformer_loader();

class RoformerSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    RoformerSession(
        const runtime::TaskSpec & task,
        runtime::SessionOptions options,
        std::shared_ptr<const RoformerAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~RoformerSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const RoformerAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    std::unique_ptr<RoformerRuntime> runtime_;
    int64_t chunk_size_ = 0;
    int64_t step_ = 0;
    int64_t fade_size_ = 0;
    int64_t border_ = 0;
    std::vector<float> chunk_window_;
    std::vector<float> first_chunk_window_;
    std::vector<float> last_chunk_window_;
    std::vector<float> only_chunk_window_;
    std::vector<float> chunk_planar_work_;
    std::vector<float> result_work_;
    std::vector<float> counter_work_;
    std::vector<float> vocals_planar_work_;
};

}  // namespace engine::models::roformer
