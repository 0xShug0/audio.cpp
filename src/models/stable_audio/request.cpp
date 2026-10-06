#include "engine/models/stable_audio/request.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine::models::stable_audio {
namespace {

std::string trim(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }).base(), value.end());
    return value;
}

std::vector<std::string> split_pipe(const std::string & value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, '|')) {
        out.push_back(trim(item));
    }
    return out;
}

std::vector<float> split_float_list(const std::string & value, const char * option_name) {
    std::vector<float> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            throw std::runtime_error(std::string(option_name) + " contains an empty value");
        }
        size_t parsed = 0;
        const float parsed_value = std::stof(item, &parsed);
        if (parsed != item.size() || !std::isfinite(parsed_value)) {
            throw std::runtime_error(std::string(option_name) + " must contain finite floats");
        }
        out.push_back(parsed_value);
    }
    return out;
}

void require_positive(float value, const char * name) {
    if (!(value > 0.0F) || !std::isfinite(value)) {
        throw std::runtime_error(std::string("Stable Audio ") + name + " must be a positive finite value");
    }
}

std::vector<float> durations_from_request(const std::unordered_map<std::string, std::string> & options, int batch_size) {
    std::vector<float> durations;
    if (const auto value = runtime::find_option(options, {"duration_sec"})) {
        durations = split_float_list(*value, "duration_sec");
    } else {
        durations.push_back(120.0F);
    }
    if (durations.size() == 1 && batch_size > 1) {
        durations.resize(static_cast<size_t>(batch_size), durations.front());
    }
    if (durations.size() != static_cast<size_t>(batch_size)) {
        throw std::runtime_error("Stable Audio duration count must be one or match batch_size");
    }
    for (const float duration : durations) {
        require_positive(duration, "duration");
    }
    return durations;
}

std::vector<std::string> prompts_from_request(const runtime::TaskRequest & request, int batch_size) {
    std::string prompt;
    if (request.text_input.has_value()) {
        prompt = request.text_input->text;
    } else if (const auto value = runtime::find_option(request.options, {"prompt", "text"})) {
        prompt = *value;
    } else {
        throw std::runtime_error("Stable Audio request requires text_input or prompt option");
    }
    auto prompts = split_pipe(prompt);
    if (prompts.size() == 1 && batch_size > 1) {
        prompts.resize(static_cast<size_t>(batch_size), prompts.front());
    }
    if (prompts.size() != static_cast<size_t>(batch_size)) {
        throw std::runtime_error("Stable Audio prompt count must be one or match batch_size");
    }
    return prompts;
}

std::vector<std::string> negative_prompts_from_request(const runtime::TaskRequest & request, int batch_size) {
    const auto value = runtime::find_option(request.options, {"negative_prompt", "negative"});
    if (!value.has_value()) {
        return {};
    }
    auto prompts = split_pipe(*value);
    if (prompts.size() == 1 && batch_size > 1) {
        prompts.resize(static_cast<size_t>(batch_size), prompts.front());
    }
    if (prompts.size() != static_cast<size_t>(batch_size)) {
        throw std::runtime_error("Stable Audio negative_prompt count must be one or match batch_size");
    }
    return prompts;
}

std::vector<StableAudioInpaintRegion> inpaint_regions_from_request(const std::unordered_map<std::string, std::string> & options) {
    const auto starts_value = runtime::find_option(options, {"inpaint_mask_start_sec", "inpaint_start"});
    const auto ends_value = runtime::find_option(options, {"inpaint_mask_end_sec", "inpaint_end"});
    if (starts_value.has_value() != ends_value.has_value()) {
        throw std::runtime_error("Stable Audio inpaint start/end options must be provided together");
    }
    if (!starts_value.has_value()) {
        return {};
    }
    const auto starts = split_float_list(*starts_value, "inpaint_mask_start_sec");
    const auto ends = split_float_list(*ends_value, "inpaint_mask_end_sec");
    if (starts.size() != ends.size()) {
        throw std::runtime_error("Stable Audio inpaint start/end counts must match");
    }
    std::vector<StableAudioInpaintRegion> regions;
    regions.reserve(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) {
        if (!std::isfinite(starts[i]) || !std::isfinite(ends[i]) || starts[i] < 0.0F || ends[i] <= starts[i]) {
            throw std::runtime_error("Stable Audio inpaint regions must be finite and ordered");
        }
        regions.push_back(StableAudioInpaintRegion{starts[i], ends[i]});
    }
    return regions;
}

}  // namespace

StableAudioRequest parse_stable_audio_request(const runtime::TaskRequest & request) {
    const auto options = runtime::apply_option_v1_compatibility(
        request.options,
        {{"duration_seconds", "duration_sec"},
         {"duration_padding_seconds", "duration_padding_sec"},
         {"inpaint_mask_start_seconds", "inpaint_mask_start_sec"},
         {"inpaint_mask_end_seconds", "inpaint_mask_end_sec"}},
        "Stable Audio", "request");
    StableAudioRequest out;
    if (const auto value = runtime::parse_int_option(request.options, {"batch_size"})) {
        if (*value <= 0) {
            throw std::runtime_error("Stable Audio batch_size must be positive");
        }
        out.batch_size = *value;
    }
    out.prompts = prompts_from_request(request, out.batch_size);
    out.negative_prompts = negative_prompts_from_request(request, out.batch_size);
    out.durations_seconds = durations_from_request(options, out.batch_size);
    if (const auto value = runtime::parse_int_option(request.options, {"num_inference_steps"})) {
        if (*value <= 0) {
            throw std::runtime_error("Stable Audio num_inference_steps must be positive");
        }
        out.num_inference_steps = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"sampler"})) {
        if (*value != "pingpong" && *value != "euler") {
            throw std::runtime_error("Stable Audio sampler must be pingpong or euler");
        }
        out.sampler = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"guidance_scale"})) {
        out.guidance_scale = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"apg_scale"})) {
        out.apg_scale = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
        out.seed_specified = true;
    } else {
        out.seed = runtime::random_u64_seed();
    }
    if (const auto value = runtime::find_option(request.options, {"truncate_output_to_duration"})) {
        out.truncate_output_to_duration = runtime::parse_bool_option(*value, "truncate_output_to_duration");
    }
    if (const auto value = runtime::find_option(request.options, {"chunked_decode"})) {
        out.chunked_decode = runtime::parse_bool_option(*value, "chunked_decode");
    }
    if (const auto value = runtime::parse_finite_float_option(options, {"duration_padding_sec"})) {
        if (*value < 0.0F) {
            throw std::runtime_error("Stable Audio duration_padding_sec must be non-negative");
        }
        out.duration_padding_seconds = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"init_noise_level"})) {
        if (*value < 0.0F || *value > 1.0F) {
            throw std::runtime_error("Stable Audio init_noise_level must be in [0, 1]");
        }
        out.init_noise_level = *value;
    }
    const auto audio_input_kind = runtime::find_option(request.options, {"audio_input_kind"});
    if (audio_input_kind.has_value() && *audio_input_kind != "init_audio" && *audio_input_kind != "inpaint_audio") {
        throw std::runtime_error("Stable Audio audio_input_kind must be init_audio or inpaint_audio");
    }
    if (request.audio_input.has_value() && (!audio_input_kind.has_value() || *audio_input_kind == "init_audio")) {
        out.init_audio = request.audio_input;
    } else if (request.audio_input.has_value() && *audio_input_kind == "inpaint_audio") {
        out.inpaint_audio = request.audio_input;
    }
    out.inpaint_regions = inpaint_regions_from_request(options);
    if (!out.inpaint_regions.empty() && request.audio_input.has_value() && !out.inpaint_audio.has_value()) {
        out.inpaint_audio = request.audio_input;
    }
    return out;
}

}  // namespace engine::models::stable_audio
