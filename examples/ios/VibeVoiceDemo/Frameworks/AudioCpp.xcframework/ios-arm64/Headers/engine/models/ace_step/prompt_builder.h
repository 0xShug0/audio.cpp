#pragma once

#include "engine/models/ace_step/types.h"

#include <string>
#include <string_view>

namespace engine::models::ace_step {

constexpr const char * kDefaultDitInstruction = "Fill the audio semantic mask based on the given conditions:";
constexpr const char * kDefaultLmInstruction = "Generate audio semantic tokens based on the given conditions:";

bool ace_step_metadata_text_is_provided(std::string_view value);
std::string ace_step_metadata_string(const AceStepMetadata & metadata, float duration_seconds);
std::string ace_step_format_instruction(const std::string & instruction);
std::string ace_step_format_lyrics(const std::string & lyrics, const std::string & language);
std::string ace_step_build_dit_caption_prompt(
    const std::string & instruction,
    const std::string & caption,
    const AceStepMetadata & metadata,
    float duration_seconds);
std::string ace_step_build_lm_prompt(
    const std::string & caption,
    const std::string & lyrics);
AceStepPlan ace_step_parse_lm_output(const std::string & output_text);

}  // namespace engine::models::ace_step
