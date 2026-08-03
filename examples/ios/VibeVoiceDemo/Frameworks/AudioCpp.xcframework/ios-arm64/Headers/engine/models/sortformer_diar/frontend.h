#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/types.h"

namespace engine::models::sortformer_diar {

SortformerFeatureBatch compute_sortformer_features(
    const runtime::AudioBuffer & audio,
    const SortformerAssets & assets,
    int64_t threads,
    SortformerRunTimings * timings = nullptr);

}  // namespace engine::models::sortformer_diar
