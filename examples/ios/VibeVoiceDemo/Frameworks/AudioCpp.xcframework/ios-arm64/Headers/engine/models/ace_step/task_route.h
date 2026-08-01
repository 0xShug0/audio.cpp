#pragma once

#include "engine/models/ace_step/types.h"

#include <string>
#include <string_view>

namespace engine::models::ace_step {

enum class AceStepSourceAudioPolicy {
    Ignored,
    Optional,
    Required,
};

struct AceStepTaskRoute {
    AceStepTaskType task = AceStepTaskType::TextToMusic;
    std::string_view name;
    bool uses_planner = true;
    AceStepSourceAudioPolicy source_audio_policy = AceStepSourceAudioPolicy::Ignored;
    bool locks_duration_to_source = false;
    bool uses_repaint_window = false;
    bool preserve_repaint_source = false;
    bool uses_cover_conditioning = false;
    std::string_view dit_instruction;
    std::string_view missing_component;
};

AceStepTaskType ace_step_parse_task_type(std::string_view task_type);
std::string_view ace_step_task_type_name(AceStepTaskType task);
const AceStepTaskRoute & ace_step_task_route(AceStepTaskType task);
const AceStepTaskRoute & ace_step_task_route(const AceStepRequest & request);
std::string ace_step_task_instruction(const AceStepTaskRoute & route, const AceStepRequest & request);

bool ace_step_route_uses_repaint_window(const AceStepTaskRoute & route);
bool ace_step_route_accepts_source_audio(const AceStepTaskRoute & route);
bool ace_step_route_requires_source_audio(const AceStepTaskRoute & route);
bool ace_step_route_uses_request_source_audio(
    const AceStepTaskRoute & route,
    const AceStepRequest & request,
    const AceStepPlan & plan);
bool ace_step_route_has_required_source_audio_input(
    const AceStepTaskRoute & route,
    const AceStepRequest & request,
    const AceStepPlan & plan);
bool ace_step_route_has_missing_component(const AceStepTaskRoute & route);
bool ace_step_request_uses_flow_edit_morph(const AceStepRequest & request);

}  // namespace engine::models::ace_step
