#pragma once

#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/types.h"

namespace engine::models::confucius4_tts {

ConfuciusRequest make_confucius_request(
    const ConfuciusAssets & assets,
    const runtime::TaskRequest & request,
    const std::optional<ConfuciusRequest> & defaults);

std::optional<ConfuciusRequest> make_confucius_prepare_defaults(
    const ConfuciusAssets & assets,
    const runtime::SessionPreparationRequest & request);

}  // namespace engine::models::confucius4_tts
