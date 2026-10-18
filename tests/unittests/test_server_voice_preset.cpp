// A voice preset's options reach the model as request options: they fill what the
// request leaves unset, for named and default presets alike, and the server never
// interprets them.
#include "server_protocol.h"

#include "engine/framework/io/json.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

namespace fs = std::filesystem;
using minitts::server::RuntimeVoicePreset;
using minitts::server::SpeechRequestContext;

void expect(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

engine::runtime::TaskRequest speech(
    const std::unordered_map<std::string, RuntimeVoicePreset> & presets,
    const std::optional<RuntimeVoicePreset> & default_preset,
    const std::unordered_map<std::string, std::string> & default_options,
    const std::string & body_text) {
    const fs::path base = fs::current_path();
    const std::optional<fs::path> voice_dir;
    return minitts::server::build_speech_request(
        engine::io::json::parse(body_text),
        SpeechRequestContext{base, voice_dir, presets, default_preset, true, true, true, default_options});
}

}  // namespace

int main() {
    try {
        RuntimeVoicePreset narrator;
        narrator.options = {
            {"speaker_embedding_file", "/voices/narrator.speaker.safetensors"},
            {"num_steps", "8"},
        };
        RuntimeVoicePreset with_text;
        with_text.reference_text = "preset transcript";
        with_text.options = {{"reference_text", "option transcript"}};
        const std::unordered_map<std::string, RuntimeVoicePreset> presets{
            {"narrator", narrator},
            {"with_text", with_text},
        };
        const std::optional<RuntimeVoicePreset> no_default;
        const std::unordered_map<std::string, std::string> no_defaults;

        auto request = speech(presets, no_default, no_defaults, R"({"input":"x","voice":"narrator"})");
        expect(request.options.at("speaker_embedding_file") == "/voices/narrator.speaker.safetensors",
               "preset option did not reach the request");
        expect(request.options.at("num_steps") == "8", "second preset option did not reach the request");
        expect(!request.voice.has_value(), "an options-only preset should not add a voice");

        // The request's own options win over the preset's.
        request = speech(presets, no_default, no_defaults,
            R"({"input":"x","voice":"narrator","options":{"num_steps":4}})");
        expect(request.options.at("num_steps") == "4", "request option did not override the preset");
        expect(request.options.at("speaker_embedding_file") == "/voices/narrator.speaker.safetensors",
               "unrelated preset option was dropped");

        // The preset wins over the model's default_request_options.
        const std::unordered_map<std::string, std::string> defaults{{"num_steps", "32"}, {"cfg_scale", "2"}};
        request = speech(presets, no_default, defaults, R"({"input":"x","voice":"narrator"})");
        expect(request.options.at("num_steps") == "8", "default_request_options overrode the preset");
        expect(request.options.at("cfg_scale") == "2", "default_request_options were dropped");

        // The dedicated reference_text field keeps its meaning.
        request = speech(presets, no_default, no_defaults, R"({"input":"x","voice":"with_text"})");
        expect(request.options.at("reference_text") == "preset transcript",
               "preset reference_text should win over the same key in its options");

        // A default preset carries its options too, until the request names a voice.
        const std::optional<RuntimeVoicePreset> default_preset = narrator;
        request = speech(presets, default_preset, no_defaults, R"({"input":"x"})");
        expect(request.options.at("speaker_embedding_file") == "/voices/narrator.speaker.safetensors",
               "default preset options did not reach the request");
        request = speech(presets, default_preset, no_defaults, R"({"input":"x","voice":"alba"})");
        expect(request.options.count("speaker_embedding_file") == 0,
               "default preset options leaked into a request that named another voice");
        expect(request.voice.has_value() && request.voice->speaker.has_value() &&
                   request.voice->speaker->cached_voice_id == "alba",
               "a voice that is not a preset should stay a model-native voice id");

        // Instruction aliases follow request > preset > model precedence.
        for (const std::string key : {"instruct", "instruction"}) {
            const std::string alias = key == "instruct" ? "instruction" : "instruct";
            RuntimeVoicePreset styled = narrator;
            styled.options[key] = "calm";
            const std::unordered_map<std::string, RuntimeVoicePreset> styled_presets{{"styled", styled}};
            const std::unordered_map<std::string, std::string> styled_defaults{{alias, "serious"}};
            auto check = [&](const std::string & body, const std::optional<RuntimeVoicePreset> & selected,
                             const std::string & expected_key, const std::string & expected_value) {
                const auto result = speech(styled_presets, selected, styled_defaults, body);
                expect(result.options.at(expected_key) == expected_value, "instruction precedence changed");
                expect(result.options.count(expected_key == "instruct" ? "instruction" : "instruct") == 0,
                       "lower-priority instruction alias survived");
                expect(result.options.at("num_steps") == "8", "unrelated preset option changed");
                expect(result.options.at("speaker_embedding_file") == "/voices/narrator.speaker.safetensors",
                       "unrelated voice option changed");
            };
            check(R"({"input":"x","voice":"styled"})", no_default, key, "calm");
            check(R"({"input":"x"})", styled, key, "calm");
            for (const std::string request_key : {key, alias}) {
                check("{\"input\":\"x\",\"voice\":\"styled\",\"options\":{\"" + request_key + "\":\"cheerful\"}}",
                      no_default, request_key, "cheerful");
                check("{\"input\":\"x\",\"options\":{\"" + request_key + "\":\"cheerful\"}}",
                      styled, request_key, "cheerful");
                const std::string field = request_key == "instruct" ? "instruct" : "instructions";
                check("{\"input\":\"x\",\"voice\":\"styled\",\"" + field + "\":\"cheerful\"}",
                      no_default, request_key, "cheerful");
            }
        }
    } catch (const std::exception & error) {
        std::cerr << "voice preset test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "server voice preset passed\n";
    return 0;
}
