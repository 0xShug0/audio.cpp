#include "../recipe.h"

#include "engine/framework/runtime/errors.h"
#include "engine/framework/text/chunking.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace minitts::app {
namespace {

using namespace engine::runtime;

class AnyTtsRecipe final : public IRecipe {
public:
    std::string id() const override { return "any-tts"; }
    std::string description() const override {
        return "Generate speech with any offline TTS model, then clone a reference voice with Tone Color VC";
    }
    std::vector<RecipeModelRole> roles() const override {
        return {
            {"tts", {VoiceTaskKind::Tts, RunMode::Offline}},
            {"converter", {VoiceTaskKind::VoiceConversion, RunMode::Offline}},
        };
    }

    TaskResult run(
        IRecipeModelExecutor & executor,
        const TaskRequest & request,
        const RecipeEventSink & event_sink) const override {
        if (!request.text_input || request.text_input->text.empty()) {
            throw InvalidRequestError("any-tts recipe requires text");
        }
        if (!request.voice || !request.voice->speaker || !request.voice->speaker->audio) {
            throw InvalidRequestError("any-tts recipe requires reference speaker audio");
        }

        const auto chunk_size = engine::text::parse_text_chunk_size_override(request.options).value_or(240);
        const auto chunk_mode = engine::text::parse_text_chunk_mode_override(request.options)
                                    .value_or(engine::text::TextChunkMode::Default);
        auto chunks = engine::text::split_text_chunks(request.text_input->text, chunk_size, chunk_mode);

        TaskResult result;
        for (size_t i = 0; i < chunks.size(); ++i) {
            TaskRequest synthesis = request;
            synthesis.text_input->text = std::move(chunks[i]);
            synthesis.options = options_for_recipe_role(
                request.options, "tts", {"tts", "converter"}, true);
            synthesis.options.erase("text_chunk_size");
            synthesis.options.erase("text_chunk_mode");
            if (synthesis.voice->speaker->cached_voice_id) {
                synthesis.voice->speaker->audio.reset();
            } else {
                synthesis.voice->speaker.reset();
            }

            auto generated = executor.run(
                "tts", {VoiceTaskKind::Tts, RunMode::Offline}, synthesis);
            if (!generated.audio_output) {
                throw std::runtime_error("any-tts TTS model returned no audio");
            }

            TaskRequest conversion;
            conversion.audio_input = std::move(generated.audio_output);
            conversion.voice = request.voice;
            conversion.options = options_for_recipe_role(
                request.options, "converter", {"tts", "converter"}, false);
            auto converted = executor.run(
                "converter", {VoiceTaskKind::VoiceConversion, RunMode::Offline}, conversion);
            if (!converted.audio_output) {
                throw std::runtime_error("any-tts converter returned no audio");
            }

            if (!result.audio_output) {
                result.audio_output.emplace();
            }
            append_audio_buffer(*result.audio_output, *converted.audio_output);
            if (event_sink) {
                StreamEvent event;
                event.audio_output = std::move(converted.audio_output);
                event.is_final = i + 1 == chunks.size();
                event_sink(event);
            }
        }
        return result;
    }
};

const RecipePluginRegistration registration([] {
    return std::make_unique<AnyTtsRecipe>();
});

}  // namespace
}  // namespace minitts::app
