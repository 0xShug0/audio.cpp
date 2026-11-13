#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::model_spec { struct ModelContract; }

namespace engine::models::mossformer2 {

struct MossFormer2Assets;

std::unique_ptr<runtime::IVoiceTaskSession> create_separation_session(
    const runtime::TaskSpec &, const runtime::SessionOptions &,
    std::shared_ptr<const MossFormer2Assets>, std::shared_ptr<const model_spec::ModelContract>);
std::unique_ptr<runtime::IVoiceTaskSession> create_restoration_session(
    const runtime::TaskSpec &, const runtime::SessionOptions &,
    std::shared_ptr<const MossFormer2Assets>, std::shared_ptr<const model_spec::ModelContract>);

std::shared_ptr<runtime::IVoiceModelLoader> make_mossformer2_loader();

}  // namespace engine::models::mossformer2
