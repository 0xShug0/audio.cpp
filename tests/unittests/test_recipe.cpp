#include "recipe.h"

#include "test_assert.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace engine::runtime;

struct Invocation {
    std::string role;
    TaskSpec task;
    TaskRequest request;
};

class RecordingExecutor final : public minitts::app::IRecipeModelExecutor {
public:
    TaskResult run(
        std::string_view role,
        const TaskSpec & task,
        const TaskRequest & request) override {
        invocations.push_back({std::string(role), task, request});

        TaskResult result;
        if (role == "translator") {
            result.text_output = Transcript{"Translated words.", "English"};
        } else if (role == "tts") {
            result.audio_output = AudioBuffer{24000, 1, {1.0F, 2.0F}};
        } else if (role == "converter") {
            engine::test::require(request.audio_input.has_value(), "converter audio input");
            result.audio_output = request.audio_input;
        }
        return result;
    }

    std::vector<Invocation> invocations;
};

void test_any_tts() {
    auto recipes = minitts::app::make_recipe_registry();
    RecordingExecutor executor;
    TaskRequest request;
    request.text_input = Transcript{"Alpha. Beta.", {}};
    request.voice = VoiceCondition{VoiceReference{AudioBuffer{24000, 1, {0.25F}}, std::nullopt}, std::nullopt};
    request.options = {
        {"text_chunk_size", "6"},
        {"qwen3_tts.temperature", "0.7"},
        {"tts.speed", "1.1"},
        {"converter.normalize_output", "true"},
    };

    size_t events = 0;
    const auto result = recipes.require("any-tts").run(
        executor,
        request,
        [&](const StreamEvent & event) {
            engine::test::require(event.audio_output.has_value(), "any-tts event audio");
            ++events;
        });

    engine::test::require_eq(executor.invocations.size(), static_cast<size_t>(4), "any-tts stage calls");
    engine::test::require_eq(events, static_cast<size_t>(2), "any-tts stream events");
    engine::test::require_eq(result.audio_output->samples.size(), static_cast<size_t>(4), "any-tts merged samples");
    for (size_t i = 0; i < executor.invocations.size(); i += 2) {
        const auto & tts = executor.invocations[i];
        const auto & converter = executor.invocations[i + 1];
        engine::test::require_eq(tts.role, std::string("tts"), "any-tts TTS role");
        engine::test::require_eq(tts.request.options.at("speed"), std::string("1.1"), "TTS role option");
        engine::test::require_eq(
            tts.request.options.at("qwen3_tts.temperature"),
            std::string("0.7"),
            "family-prefixed TTS option");
        engine::test::require(
            tts.request.options.find("text_chunk_size") == tts.request.options.end() &&
                tts.request.options.find("converter.normalize_output") == tts.request.options.end(),
            "recipe-owned and converter options excluded from TTS");
        engine::test::require_eq(converter.role, std::string("converter"), "any-tts converter role");
        engine::test::require_eq(
            converter.request.options.at("normalize_output"),
            std::string("true"),
            "converter role option");
        engine::test::require_eq(
            converter.request.voice->speaker->audio->samples.size(),
            static_cast<size_t>(1),
            "converter reference audio");
    }
}

void test_cross_language_dubbing() {
    auto recipes = minitts::app::make_recipe_registry();
    RecordingExecutor executor;
    TaskRequest request;
    request.audio_input = AudioBuffer{16000, 1, {0.5F}};
    request.options = {
        {"translator.language", "zho"},
        {"translator.target_language", "eng"},
        {"tts.speed", "0.9"},
        {"qwen3_tts.temperature", "0.6"},
        {"text_chunk_size", "120"},
    };

    const auto result = recipes.require("cross-language-dubbing").run(executor, request);
    engine::test::require_eq(executor.invocations.size(), static_cast<size_t>(2), "dubbing stage calls");
    engine::test::require_eq(executor.invocations[0].role, std::string("translator"), "translator role");
    const auto & translator = executor.invocations[0].request;
    const auto & tts = executor.invocations[1].request;
    engine::test::require_eq(translator.options.at("language"), std::string("zho"), "source language");
    engine::test::require_eq(translator.options.at("target_language"), std::string("eng"), "target language");
    engine::test::require_eq(tts.options.at("speed"), std::string("0.9"), "TTS role option");
    engine::test::require_eq(
        tts.options.at("qwen3_tts.temperature"), std::string("0.6"), "TTS family option");
    engine::test::require_eq(tts.options.at("text_chunk_size"), std::string("120"), "TTS chunk option");
    engine::test::require_eq(tts.text_input->text, std::string("Translated words."), "translation handoff");
    engine::test::require(tts.text_input->language.empty(), "translator language is not imposed on TTS");
    engine::test::require_eq(result.text_output->text, std::string("Translated words."), "translated result");
    engine::test::require(result.audio_output.has_value(), "dubbing audio result");
}

void test_role_option_precedence() {
    for (const auto & options : {
             std::unordered_map<std::string, std::string>{{"speed", "0.8"}, {"tts.speed", "1.1"}},
             std::unordered_map<std::string, std::string>{{"tts.speed", "1.1"}, {"speed", "0.8"}}}) {
        const auto selected = minitts::app::options_for_recipe_role(options, "tts", {"tts", "converter"}, true);
        engine::test::require_eq(selected.at("speed"), std::string("1.1"), "role option overrides shared option");
    }
}

}  // namespace

int main() {
    test_any_tts();
    test_cross_language_dubbing();
    test_role_option_precedence();
    std::cout << "recipe_test: all passed\n";
    return 0;
}
