#include "recipe.h"

#include "test_assert.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/streaming_resampler.h"
#include "streaming_tempo.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <stdexcept>

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

class StreamingExecutor final : public minitts::app::IRecipeModelExecutor {
public:
    std::atomic<bool> complete{false};
    bool fail_source = false, fail_avatar = false, early = false;
    bool chat = false, fail_conversation = false, conversation_early = false;
    std::atomic<bool> conversation_done{false};
    std::atomic<size_t> tts_calls{0};
    float speed = 1.0F;
    std::vector<float> consumed;
    std::mutex mutex;
    std::condition_variable ready;
    bool video_started = false;
    std::vector<std::string> phrases;
    const std::string first = "Here is a practical suggestion for your afternoon: take a gentle walk outside and notice the trees around you. ";
    const std::string tail = "Then take a slow breath. Keep your shoulders relaxed. 最後に、少し休んでください。";
    TaskResult run(std::string_view, const TaskSpec &, const TaskRequest &) override {
        throw std::runtime_error("offline execution is forbidden in the streaming recipe");
    }
    TaskResult stream(std::string_view role, const TaskSpec &, const TaskRequest & request,
        const minitts::app::RecipeEventSink & sink, const minitts::app::AudioChunkStream * source = nullptr) override {
        TaskResult result;
        if (role == "conversation") {
            if (fail_conversation) throw std::runtime_error("conversation failed");
            engine::test::require_eq(request.input_artifacts.size(), size_t{1}, "conversation history forwarded");
            engine::test::require(request.text_input && request.text_input->text == "A warm, concise assistant.", "conversation system prompt forwarded");
            StreamEvent event; event.partial_text = Transcript{first, {}}; sink(event);
            {
                std::unique_lock<std::mutex> lock(mutex);
                engine::test::require(ready.wait_for(lock, std::chrono::seconds(5), [&] { return video_started; }),
                    "animation must start before conversation can finish");
            }
            event.partial_text->text = tail; sink(event);
            conversation_done = true;
            result.text_output = Transcript{first + tail, {}};
            result.output_artifacts = request.input_artifacts;
        } else if (role == "tts") {
            ++tts_calls;
            engine::test::require(!request.options.count("speaking_speed"), "recipe tempo is not forwarded to a model");
            phrases.push_back(request.text_input->text);
            engine::test::require(request.options.at("temperature") == "0.5", "TTS options forwarded");
            engine::test::require(request.options.at("seed") == "42", "shared seed forwarded");
            for (int i = 0; i < 50; ++i) {
                if (fail_source && i == 2) throw std::runtime_error("source failed");
                StreamEvent event;
                event.audio_output = AudioBuffer{24000, 1, std::vector<float>(1920, 0.25F)};
                sink(event);
                if (!result.audio_output) result.audio_output = AudioBuffer{24000, 1, {}};
                append_audio_buffer(*result.audio_output, *event.audio_output);
            }
            complete = true;
        } else {
            engine::test::require(source != nullptr, "avatar receives a live source");
            engine::test::require(request.options.at("reference_image_file") == "portrait.png", "avatar options forwarded");
            std::vector<float> chunk;
            size_t frames = 0;
            while (source->read(23040, chunk)) {
                if (fail_avatar) throw std::runtime_error("avatar failed");
                if (speed == 1.0F) for (auto value : chunk) engine::test::require(value == 0.25F, "PCM preserved");
                consumed.insert(consumed.end(), chunk.begin(), chunk.end());
                frames += chunk.size();
                if (!early && frames >= 23040) {
                    early = !complete.load();
                    StreamEvent event;
                    event.output_artifacts.push_back(make_voice_artifact(ArtifactKind::Custom, "video", {std::byte{1}}, {}));
                    sink(event);
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        conversation_early = !conversation_done.load(); video_started = true; ready.notify_all();
                    }
                }
            }
            if (speed == 1.0F) engine::test::require_eq(frames, size_t{96000} * tts_calls.load(), "all source frames consumed");
            else engine::test::require(std::abs(double(frames) - 96000.0 * tts_calls.load() / speed) < 48, "avatar consumes the shortened speech duration");
        }
        return result;
    }
};

void test_streaming_avatar() {
    auto recipes = minitts::app::make_recipe_registry();
    TaskRequest request;
    request.text_input = Transcript{"A complete spoken response.", {}};
    request.options = {{"avatar_sample_rate", "24000"}, {"seed", "42"}, {"tts.temperature", "0.5"}, {"avatar.reference_image_file", "portrait.png"}};
    StreamingExecutor executor;
    const auto result = recipes.require("speaking-avatar").run(executor, request);
    engine::test::require(executor.early, "video must begin before TTS completes");
    engine::test::require_eq(result.audio_output->samples.size(), size_t{96000}, "full replay audio retained");
    for (bool source_failure : {true, false}) {
        StreamingExecutor failed;
        failed.fail_source = source_failure;
        failed.fail_avatar = !source_failure;
        bool rejected = false;
        try { recipes.require("speaking-avatar").run(failed, request); }
        catch (const std::runtime_error &) { rejected = true; }
        engine::test::require(rejected, "producer and consumer failures propagate and join");
    }
}

void test_streaming_conversation() {
    auto recipes = minitts::app::make_recipe_registry();
    TaskRequest request;
    request.audio_input = AudioBuffer{16000, 1, {0.1F}};
    request.text_input = Transcript{"A warm, concise assistant.", {}};
    request.options = {{"avatar_sample_rate", "24000"}, {"seed", "42"}, {"tts.temperature", "0.5"}, {"avatar.reference_image_file", "portrait.png"}};
    request.input_artifacts.push_back(make_voice_artifact(ArtifactKind::Custom, "history", {std::byte{1}}, {}));
    StreamingExecutor executor;
    const auto result = recipes.require("voice-chat-avatar").run(executor, request);
    engine::test::require(executor.conversation_early, "text, synthesis and animation overlap before reply completion");
    engine::test::require(executor.tts_calls > 1, "reply synthesized incrementally");
    engine::test::require_eq(result.text_output->text, executor.first + executor.tail, "complete streamed reply retained");
    engine::test::require_eq(result.output_artifacts.size(), size_t{1}, "conversation artifacts retained");
    engine::test::require_eq(result.audio_output->samples.size(), size_t{96000} * executor.tts_calls.load(), "all phrases in replay");
    std::string reconstructed;
    for (const auto & phrase : executor.phrases) { if (!reconstructed.empty()) reconstructed += ' '; reconstructed += phrase; }
    engine::test::require_eq(reconstructed, executor.first + executor.tail, "no text lost or repeated across phrases, including UTF-8");
    StreamingExecutor failed; failed.fail_conversation = true;
    bool rejected = false;
    try { recipes.require("voice-chat-avatar").run(failed, request); }
    catch (const std::runtime_error &) { rejected = true; }
    engine::test::require(rejected, "conversation failure releases waiting TTS and avatar");
}

void test_streaming_tempo() {
    constexpr int rate = 24000;
    std::vector<float> input(rate * 3);
    for (size_t i = 0; i < input.size(); ++i) input[i] = .5F * std::sin(float(i) * 2.0F * 3.14159265358979323846F * 220 / rate);
    minitts::app::StreamingSpeechTempo identity(rate, 1.0F);
    engine::test::require(identity.process(input, true) == input, "normal tempo is byte-exact passthrough");
    std::unique_ptr<minitts::app::StreamingSpeechTempo> tempo;
    try { tempo = std::make_unique<minitts::app::StreamingSpeechTempo>(rate, 1.3F); }
    catch (const std::runtime_error & error) {
        if (std::string(error.what()) != "speaking_speed other than 1 requires libsonic") throw;
        std::cout << "SKIP: libsonic unavailable for optional speech-tempo tests\n";
        return;
    }
    std::vector<float> output;
    bool incremental = false;
    for (size_t offset = 0; offset < input.size(); offset += 317) {
        const size_t end = std::min(input.size(), offset + 317);
        auto block = tempo->process(std::vector<float>(input.begin() + offset, input.begin() + end));
        incremental = incremental || !block.empty();
        output.insert(output.end(), block.begin(), block.end());
    }
    const auto tail = tempo->process({}, true);
    output.insert(output.end(), tail.begin(), tail.end());
    engine::test::require(incremental, "tempo produces samples before end of input");
    engine::test::require(std::abs(double(output.size()) - double(input.size()) / 1.3) < 48, "tempo duration matches speed");
    const size_t first = rate / 2, last = output.size() - rate / 2;
    size_t crossings = 0;
    for (size_t i = first; i + 1 < last; ++i) if (output[i] <= 0 && output[i + 1] > 0) ++crossings;
    const double frequency = double(crossings) * rate / (last - first);
    engine::test::require(std::abs(frequency - 220.0) < 2.0, "tempo preserves pitch instead of resampling it");

    TaskRequest request;
    request.text_input = Transcript{"A complete spoken response.", {}};
    request.options = {{"avatar_sample_rate", "24000"}, {"speaking_speed", "1.3"}, {"seed", "42"}, {"tts.temperature", "0.5"}, {"avatar.reference_image_file", "portrait.png"}};
    StreamingExecutor executor; executor.speed = 1.3F;
    std::vector<float> streamed;
    const auto result = minitts::app::make_recipe_registry().require("speaking-avatar").run(executor, request,
        [&](const StreamEvent & event) { if (event.audio_output) streamed.insert(streamed.end(), event.audio_output->samples.begin(), event.audio_output->samples.end()); });
    engine::test::require(executor.early, "fast audio still animates before TTS completion");
    engine::test::require(streamed == executor.consumed, "browser and avatar receive identical sped-up PCM");
    engine::test::require(streamed == result.audio_output->samples, "replay and live audio have identical tempo and samples");
}

void test_streaming_resampling() {
    std::vector<float> input(24000);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * .037F);
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::ExplicitFloat32Runtime;
    const auto reference = engine::audio::try_resample_mono_soxr(input, 24000, 16000, options);
    if (!reference) { std::cout << "SKIP: libsoxr is unavailable for resampling comparison\n"; return; }
    engine::audio::StreamingSoxrResampler resampler(24000, 16000);
    std::vector<float> actual;
    for (size_t offset = 0; offset < input.size();) {
        const size_t count = std::min(size_t{317}, input.size() - offset);
        auto chunk = resampler.process(std::vector<float>(input.begin() + offset, input.begin() + offset + count));
        actual.insert(actual.end(), chunk.begin(), chunk.end());
        offset += count;
    }
    auto tail = resampler.process({}, true);
    actual.insert(actual.end(), tail.begin(), tail.end());
    engine::test::require_eq(actual.size(), reference->size(), "streaming resampler length");
    for (size_t i = 0; i < actual.size(); ++i)
        engine::test::require(std::abs(actual[i] - (*reference)[i]) < 1e-6F, "streaming resampler matches one-shot");
}

}  // namespace

int main() {
    test_any_tts();
    test_cross_language_dubbing();
    test_role_option_precedence();
    test_streaming_avatar();
    test_streaming_conversation();
    test_streaming_resampling();
    test_streaming_tempo();
    std::cout << "recipe_test: all passed\n";
    return 0;
}
