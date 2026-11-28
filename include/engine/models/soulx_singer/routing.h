#pragma once

#include "engine/models/soulx_singer/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/modules/pitch_extractors/rmvpe_pitch_extractor.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"

namespace engine::models::soulx_singer {

struct ConditioningComponents {
    std::unique_ptr<modules::WhisperFrontendComponent> whisper;
    std::unique_ptr<modules::RmvpePitchExtractorComponent> pitch;
    std::unique_ptr<ScoreConditioner> score;
};

struct SingerImplementation {
    runtime::VoiceTaskKind task;
    void (*load_conditioning_assets)(ModelAssets &);
    ConditioningComponents (*create_conditioning)(const ModelAssets &, core::ExecutionContext &);
    std::unique_ptr<runtime::IVoiceTaskSession> (*create_session)(
        const runtime::TaskSpec &, const runtime::SessionOptions &,
        std::shared_ptr<const ModelAssets>, std::shared_ptr<const model_spec::ModelContract>);
};

const SingerImplementation & implementation_for_task(runtime::VoiceTaskKind task);

} // namespace engine::models::soulx_singer
