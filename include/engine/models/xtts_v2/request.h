#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/xtts_v2/types.h"

namespace engine::models::xtts_v2 {

XttsV2Request parse_xtts_v2_request(const runtime::TaskRequest & request);

}  // namespace engine::models::xtts_v2
