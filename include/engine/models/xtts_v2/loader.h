#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/xtts_v2/assets.h"

#include <memory>

namespace engine::models::xtts_v2 {

// Loader factory function (registered via CMakeLists.txt).
extern "C" std::shared_ptr<runtime::IVoiceModelLoader> make_xtts_v2_loader();

class XttSv2LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    explicit XttSv2LoadedModel(std::shared_ptr<const XttSv2Assets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    std::shared_ptr<const XttSv2Assets> assets_;
    mutable runtime::ModelMetadata metadata_;
    mutable runtime::CapabilitySet caps_;
};

}  // namespace engine::models::xtts_v2

