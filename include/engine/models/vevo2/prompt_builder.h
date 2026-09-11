#pragma once

#include "engine/models/vevo2/types.h"

namespace engine::models::vevo2 {

Vevo2PromptParts build_vevo2_prompt_parts(
    const Vevo2Request & request,
    const Vevo2TokenSequence & prosody_tokens,
    const Vevo2TokenSequence & style_content_tokens);

}  // namespace engine::models::vevo2
