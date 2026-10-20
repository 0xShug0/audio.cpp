#include "../recipe.h"

#include "engine/framework/runtime/errors.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace minitts::app {
namespace {

using namespace engine::runtime;

class CrossLanguageDubbingRecipe final : public IRecipe {
public:
    std::string id() const override { return "cross-language-dubbing"; }
    std::string description() const override {
        return "Translate speech with an ASR translation model and dub it with a target-language TTS model";
    }
    std::vector<RecipeModelRole> roles() const override {
        return {
            {"translator", {VoiceTaskKind::Asr, RunMode::Offline}},
            {"tts", {VoiceTaskKind::Tts, RunMode::Offline}},
        };
    }

    TaskResult run(
        IRecipeModelExecutor & executor,
        const TaskRequest & request,
        const RecipeEventSink & event_sink) const override {
        if (!request.audio_input) {
            throw InvalidRequestError("cross-language-dubbing recipe requires input audio");
        }
        if (request.options.find("translator.target_language") == request.options.end()) {
            throw InvalidRequestError(
                "cross-language-dubbing recipe requires translator.target_language");
        }

        TaskRequest translation_request;
        translation_request.audio_input = request.audio_input;
        translation_request.text_input = request.text_input;
        translation_request.options = options_for_recipe_role(
            request.options, "translator", {"translator", "tts"}, false);
        auto translated = executor.run(
            "translator", {VoiceTaskKind::Asr, RunMode::Offline}, translation_request);
        if (!translated.text_output || translated.text_output->text.empty()) {
            throw std::runtime_error("cross-language-dubbing translator returned no text");
        }

        TaskRequest synthesis;
        synthesis.text_input = Transcript{translated.text_output->text, {}};
        synthesis.voice = request.voice;
        synthesis.options = options_for_recipe_role(
            request.options, "tts", {"translator", "tts"}, true);
        auto result = executor.run(
            "tts", {VoiceTaskKind::Tts, RunMode::Offline}, synthesis);
        if (!result.audio_output) {
            throw std::runtime_error("cross-language-dubbing TTS model returned no audio");
        }
        result.text_output = std::move(translated.text_output);
        if (event_sink) {
            StreamEvent event;
            event.audio_output = result.audio_output;
            event.partial_text = result.text_output;
            event.is_final = true;
            event_sink(event);
        }
        return result;
    }
};

const RecipePluginRegistration registration([] {
    return std::make_unique<CrossLanguageDubbingRecipe>();
});

}  // namespace
}  // namespace minitts::app
