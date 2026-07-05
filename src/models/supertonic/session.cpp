#include "engine/models/supertonic/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/models/supertonic/runtime.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::supertonic {
namespace {

std::shared_ptr<const SupertonicAssets> require_assets(std::shared_ptr<const SupertonicAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Supertonic session requires assets");
    }
    return assets;
}

void validate_session_options(const runtime::SessionOptions & options) {
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key == "max_text_length") {
            continue;
        }
        if (key.rfind("supertonic.", 0) == 0) {
            throw std::runtime_error("unknown Supertonic session option: " + key);
        }
    }
}

}  // namespace

SupertonicSession::SupertonicSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SupertonicAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(assets_),
      runtime_(std::make_unique<SupertonicNativeRuntime>(assets_, options.backend)) {
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Supertonic only supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Supertonic only supports the Tts task");
    }
    validate_session_options(options);
}

SupertonicSession::~SupertonicSession() = default;

std::string SupertonicSession::family() const {
    return "supertonic";
}

runtime::VoiceTaskKind SupertonicSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SupertonicSession::run_mode() const {
    return task_.mode;
}

void SupertonicSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

runtime::TaskResult SupertonicSession::run(const runtime::TaskRequest & request) {
    require_prepared("Supertonic run");
    validate_request(request);
    const auto options = generation_options_from_request(request);
    runtime::TaskResult result;
    result.audio_output = runtime_->synthesize(request.text_input->text, options, tokenizer_);
    return result;
}

SupertonicGenerationOptions SupertonicSession::generation_options_from_request(const runtime::TaskRequest & request) const {
    SupertonicGenerationOptions options;
    if (request.text_input.has_value() && !request.text_input->language.empty()) {
        options.language = request.text_input->language;
    }
    if (request.voice.has_value() && request.voice->style.has_value() && request.voice->style->language.has_value()) {
        options.language = *request.voice->style->language;
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->cached_voice_id.has_value()) {
        options.voice = *request.voice->speaker->cached_voice_id;
    }
    if (const auto value = runtime::find_option(request.options, {"voice"})) {
        options.voice = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"num_inference_steps"})) {
        if (*value <= 0) {
            throw std::runtime_error("Supertonic num_inference_steps must be positive");
        }
        options.num_inference_steps = static_cast<int>(*value);
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"speaking_rate"})) {
        if (*value <= 0.0F) {
            throw std::runtime_error("Supertonic speaking_rate must be positive");
        }
        options.speaking_rate = *value;
    }
    if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
        options.seed = *value;
    }
    return options;
}

void SupertonicSession::validate_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Supertonic requires --text input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Supertonic does not accept audio input");
    }
}

}  // namespace engine::models::supertonic
