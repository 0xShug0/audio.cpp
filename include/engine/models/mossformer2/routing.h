#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>
#include <string_view>

namespace engine::model_spec { struct ModelContract; }

namespace engine::models::mossformer2 {

struct MossFormer2Assets;

struct MossFormer2Implementation {
    runtime::VoiceTaskKind task = runtime::VoiceTaskKind::SpeechToSpeech;
    bool dilated_fsmn = false;
    bool use_vocoder = false;
    std::unique_ptr<runtime::IVoiceTaskSession> (*create_session)(
        const runtime::TaskSpec &, const runtime::SessionOptions &,
        std::shared_ptr<const MossFormer2Assets>, std::shared_ptr<const model_spec::ModelContract>) = nullptr;
};

const MossFormer2Implementation & implementation_for_variant(std::string_view variant);

}  // namespace engine::models::mossformer2
