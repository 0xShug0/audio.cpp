#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/demucs/assets.h"
#include "engine/models/demucs/pipeline.h"

#include <memory>
#include <vector>

namespace engine::models::demucs {

std::shared_ptr<runtime::IVoiceModelLoader> make_htdemucs_loader();

class HTDemucsSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    HTDemucsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const HTDemucsAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~HTDemucsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const HTDemucsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    std::unique_ptr<HTDemucsPipeline> pipeline_;
    int64_t chunk_size_ = 0;
    int64_t step_ = 0;
    std::vector<float> chunk_window_;
    std::vector<float> chunk_planar_work_;
    std::vector<float> result_work_;
    std::vector<float> counter_work_;
};

}  // namespace engine::models::demucs
