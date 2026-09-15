#include "engine/models/yue2/types.h"

#include <stdexcept>

namespace engine::models::yue2 {

const char * cot_mode_name(Yue2CotMode mode) noexcept {
    switch (mode) {
        case Yue2CotMode::Off:
            return "off";
        case Yue2CotMode::Melody:
            return "melody";
        case Yue2CotMode::Full:
            return "full";
    }
    return "full";
}

Yue2CotMode parse_cot_mode(const std::string & value) {
    if (value == "off") {
        return Yue2CotMode::Off;
    }
    if (value == "melody") {
        return Yue2CotMode::Melody;
    }
    if (value == "full") {
        return Yue2CotMode::Full;
    }
    throw std::runtime_error("yue2.cot must be one of off, melody, or full");
}

const char * cot_instruction(Yue2CotMode mode) noexcept {
    switch (mode) {
        case Yue2CotMode::Off:
            return "Generate music with codec tokens from the given conditions.";
        case Yue2CotMode::Melody:
            return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.";
        case Yue2CotMode::Full:
            return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";
    }
    return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";
}

float request_guidance_scale(const Yue2Request & request) noexcept {
    if (request.cfg_scale >= 0.0F) {
        return request.cfg_scale;
    }
    return request.cot == Yue2CotMode::Off ? 1.01F : 1.0F;
}

}  // namespace engine::models::yue2
