#pragma once

#include "engine/community_models/inflect_v2/assets.h"
#include "engine/community_models/inflect_v2/frontend.h"
#include "engine/community_models/inflect_v2/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::models::inflect_v2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_inflect_v2_loader();

class InflectV2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    InflectV2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const InflectV2Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~InflectV2Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    InflectV2GenerationOptions generation_options(
        const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const InflectV2Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<InflectV2Frontend> frontend_;
    std::unique_ptr<InflectV2NativeRuntime> runtime_;
};

}  // namespace engine::models::inflect_v2
