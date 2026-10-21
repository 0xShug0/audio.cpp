#include "engine/models/ming3_audio/talker_frontend.h"

#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <map>

namespace engine::models::ming3_audio {

TalkerFrontend::TalkerFrontend(const MingAudioAssets & assets)
    : tokenizer_({{}, {}, assets.resources.require_file("talker_tokenizer_config"),
        assets.resources.require_file("talker_tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Qwen2}) {}

TalkerPrompt TalkerFrontend::build(const std::string & text, const std::string & voice, const std::string & language,
                                   bool reference, const std::string & reference_text, int64_t reference_patches) const {
    std::string user_prompt = "Please generate speech based on the following description.\n";
    std::string key = io::trim_ascii_whitespace(language);
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
    if (!key.empty() && key != "en" && key != "english" && key != "en-us" && key != "en-gb" &&
        key != "zh" && key != "zh-cn" && key != "chinese") {
        static const std::map<std::string, std::string> languages{
            {"ja", "Japanese"}, {"ko", "Korean"}, {"de", "German"}, {"fr", "French"}, {"es", "Spanish"}};
        const auto found = languages.find(key);
        user_prompt += "Target language: " + (found == languages.end() ? language : found->second) + ".\n";
    }
    std::vector<std::string> pieces{"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "<|im_start|>user\n", user_prompt};
    if (reference) {
        pieces.insert(pieces.end(), {"  speaker_1:", "<|vision_start|>", "<|vision_pad|>", "<|vision_end|>\n"});
    }
    pieces.emplace_back(" Text input:\n");
    if (reference_patches > 0 && !reference_text.empty()) pieces.push_back(reference_text);
    pieces.insert(pieces.end(), {text, "<|im_end|>\n", "<|im_start|>assistant\n"});
    if (!reference) {
        const std::array<const char *, 8> keys{
            u8"\u5e8f\u53f7", u8"\u8bf4\u8bdd\u4eba", u8"\u65b9\u8a00", u8"\u98ce\u683c",
            u8"\u8bed\u901f", u8"\u57fa\u9891", u8"\u97f3\u91cf", u8"\u60c5\u611f"};
        std::string instruction = "{\"audio_sequence\": [{" + io::json::stringify_string(keys[0]) + ": 1, " +
            io::json::stringify_string(keys[1]) + ": \"speaker_1\"";
        for (size_t i = 2; i < keys.size(); ++i) instruction += ", " + io::json::stringify_string(keys[i]) + ": null";
        instruction += ", \"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, \"ENV\": null, \"SNR\": null}, \"IP\": " +
            io::json::stringify_string(voice) + "}]}";
        pieces.insert(pieces.end(), {instruction, "<|im_end|>"});
    }
    pieces.emplace_back("<audio>");
    TalkerPrompt prompt;
    for (const auto & piece : pieces) {
        const auto ids = tokenizer_.encode(piece, true);
        prompt.ids.insert(prompt.ids.end(), ids.begin(), ids.end());
    }
    if (reference) {
        const auto start = tokenizer_.find_token_id("<|vision_start|>").value();
        const auto found = std::find(prompt.ids.begin(), prompt.ids.end(), start);
        prompt.speaker_position = std::distance(prompt.ids.begin(), found) + 1;
    }
    if (reference_patches > 0 && !reference_text.empty()) {
        prompt.latent_position = prompt.ids.size();
        prompt.ids.insert(prompt.ids.end(), reference_patches, tokenizer_.find_token_id("<audioPatch>").value());
    }
    return prompt;
}

}  // namespace engine::models::ming3_audio
