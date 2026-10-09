#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/ast_audioset/assets.h"

#include <vector>

namespace engine::models::ast_audioset {

std::vector<float> extract_features(
    const runtime::AudioBuffer & audio,
    const Config & config);

}  // namespace engine::models::ast_audioset
