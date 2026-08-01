#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/ace_step/types.h"

namespace engine::models::ace_step {

AceStepRequest ace_step_parse_request(const runtime::TaskRequest & request);

}  // namespace engine::models::ace_step
