#pragma once

// The artifact-list reader of the RKNN module: a family's execution plan travels as a session
// option, and answering "which compiled engines will this load hold resident" needs the RKNN
// vocabulary -- artifact strings that name engine images, ladders that travel comma-separated,
// paths resolved the way a sidecar resolves them. That reading lives here, beside the engine
// reader it calls into, not in the generic runtime; a host consumer prices and de-duplicates the
// paths through IVoiceModelLoader::model_resources.

#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <string>
#include <vector>

namespace engine::rknn {

// The compiled engines one `exec_plan` value declares: every artifact string, comma-separated
// ladders included, resolved against `model_root` the way a sidecar resolves it
// (`rknn::resolve_engine_path`) and priced by the target runtime, which reports 0 when it cannot
// say. A family uses this to answer IVoiceModelLoader::model_resources without re-deriving its
// component vocabulary, so plan parsing lives in one place: an absent option, a malformed plan and
// a non-string artifact all contribute nothing.
std::vector<runtime::ModelEngineRequirement> exec_plan_engine_requirements(
    const std::string & exec_plan_value,
    const std::filesystem::path & model_root);

}  // namespace engine::rknn
