#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/magpie_tts/assets.h"
#include "engine/models/magpie_tts/types.h"

#include <optional>

namespace engine::models::magpie_tts {

std::optional<MagpieTTSRequest> make_magpie_tts_prepare_defaults(
    const MagpieTTSAssets & assets,
    const runtime::SessionPreparationRequest & request);

MagpieTTSRequest make_magpie_tts_request(
    const MagpieTTSAssets & assets,
    const runtime::TaskRequest & request,
    const std::optional<MagpieTTSRequest> & defaults);

}  // namespace engine::models::magpie_tts
