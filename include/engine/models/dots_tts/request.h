#pragma once

#include "engine/models/dots_tts/assets.h"

#include <optional>

namespace engine::models::dots_tts {

DotsRequest make_dots_request(
    const DotsAssets & assets,
    const runtime::TaskRequest & request,
    const std::optional<DotsRequest> & defaults);

std::optional<DotsRequest> make_dots_prepare_defaults(
    const DotsAssets & assets,
    const runtime::SessionPreparationRequest & request);

}  // namespace engine::models::dots_tts
