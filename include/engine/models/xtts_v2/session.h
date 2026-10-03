#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/xtts_v2/assets.h"
#include "engine/models/xtts_v2/conditioning.h"
#include "engine/models/xtts_v2/decoder.h"
#include "engine/models/xtts_v2/gpt.h"
#include "engine/models/xtts_v2/speaker_encoder.h"
#include "engine/models/xtts_v2/tokenizer.h"

#include <memory>

namespace engine::models::xtts_v2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_xtts_v2_loader();

class XttsV2Session final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession {
public:
    XttsV2Session(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const XttsV2Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~XttsV2Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const XttsV2Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    XttsV2Tokenizer tokenizer_;
    std::unique_ptr<XttsV2ConditioningRuntime> conditioning_;
    std::unique_ptr<XttsV2SpeakerEncoderRuntime> speaker_;
    std::unique_ptr<XttsV2GptRuntime> gpt_;
    std::unique_ptr<XttsV2DecoderRuntime> decoder_;
};

}  // namespace engine::models::xtts_v2
