// LFM3-Audio prompts: the voice and language tables, the system prompts, the
// ASR / chat / TTS templates on a synthetic tokenizer, and the mode a prompt
// starts generation in.
#include "engine/community_models/lfm3_audio/prompt.h"
#include "lfm3_audio_test_tokenizer.h"
#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
using engine::test::require;
using engine::test::require_eq;
using lfm3_audio_test::join;
using lfm3_audio_test::require_ids;
using lfm3_audio_test::require_request_error;

void test_voices() {
    const auto & voices = lfm3::lfm3_tts_voices();
    require_eq(voices.size(), size_t{38}, "19 languages, two voices each");
    std::set<std::string> ids;
    for (const auto & voice : voices) {
        ids.insert(voice.id);
        require(voice.gender == "female" || voice.gender == "male", voice.id + " gender");
        require(voice.id.size() > voice.gender.size() && voice.id.substr(voice.id.size() - voice.gender.size()) == voice.gender, voice.id + " ends in its gender");
    }

    require_eq(ids.size(), size_t{38}, "voice ids are unique");
    require_eq(voices.front().id, std::string("arabic_female"), "first voice");
    require_eq(voices.back().id, std::string("chinese_male"), "last voice");

    require_eq(lfm3::lfm3_tts_voice("", "auto").id, std::string("american_english_female"), "the default voice");
    require_eq(lfm3::lfm3_tts_voice("", "").id, std::string("american_english_female"), "no language");
    require_eq(lfm3::lfm3_tts_voice("", "en").id, std::string("american_english_female"), "English");
    require_eq(lfm3::lfm3_tts_voice("", "ja").id, std::string("japanese_female"), "a language's default voice");
    require_eq(lfm3::lfm3_tts_voice("british_english_male", "ja").id, std::string("british_english_male"), "a voice wins over a language");
    require_request_error([] { (void)lfm3::lfm3_tts_voice("klingon_female", "auto"); }, "has no voice klingon_female", "an unknown voice");
    require_request_error([] { (void)lfm3::lfm3_tts_voice("", "xx"); }, "does not know language \"xx\"", "an unknown language");

    require_eq(lfm3::lfm3_tts_system_prompt(lfm3::lfm3_tts_voice("", "en")),
        std::string("Perform TTS in American English using a female voice."), "TTS system prompt");
    require_eq(lfm3::lfm3_tts_system_prompt(lfm3::lfm3_tts_voice("japanese_male", "")),
        std::string("Perform TTS in Japanese using a male voice."), "Japanese spelled right");
    require_eq(lfm3::lfm3_tts_system_prompt(lfm3::lfm3_tts_voice("british_english_female", "")),
        std::string("Perform TTS in British English using a female voice."), "British English");

    require_eq(lfm3::lfm3_asr_system_prompt("auto"), std::string("Perform ASR."), "ASR auto");
    require_eq(lfm3::lfm3_asr_system_prompt("en"), std::string("Perform ASR in English."), "ASR English");
    require_eq(lfm3::lfm3_asr_system_prompt("zh"), std::string("Perform ASR in Chinese."), "ASR Chinese");
    require_request_error([] { (void)lfm3::lfm3_asr_system_prompt("english"); }, "does not know language \"english\"", "a language name");
}

struct Synthetic {
    lfm3::Lfm3Tokenizer tokenizer;
    const lfm3::Lfm3TokenIds & ids;
    explicit Synthetic(const std::string & name)
        : tokenizer(lfm3_audio_test::write_tokenizer(lfm3_audio_test::fresh_directory(name), true)), ids(tokenizer.ids()) {}

    std::vector<int32_t> text(const std::string & s) const { return tokenizer.encode(s); }
};

std::vector<int32_t> repeat(int32_t id, size_t n) {
    return std::vector<int32_t>(n, id);
}

void test_templates() {
    const Synthetic s("lfm3_audio_prompt_test_templates");
    const auto & id = s.ids;
    const std::vector<int32_t> nl = {id.nl};

    const auto asr = lfm3::make_lfm3_asr_prompt(s.tokenizer, "Perform ASR.", 3);
    require_ids(asr.ids,
        join({{id.bos, id.im_start}, s.text("system"), nl, s.text("Perform ASR."), {id.im_end}, nl, {id.im_start}, s.text("user"), nl,
              repeat(id.audio, 3), {id.im_end}, nl, {id.im_start}, s.text("assistant"), nl}),
        "ASR prompt");
    require(asr.frames.empty(), "ASR prompt has no frames");
    require_ids(lfm3::make_lfm3_asr_prompt(s.tokenizer, "", 1).ids,
        join({{id.bos, id.im_start}, s.text("user"), nl, {id.audio, id.im_end}, nl, {id.im_start}, s.text("assistant"), nl}),
        "an empty system prompt renders no system block");

    const auto chat_tail = join({{id.im_end}, nl, {id.im_start}, s.text("assistant"), nl});
    const auto chat_head = join({{id.bos, id.im_start}, s.text("user"), nl, repeat(id.audio, 2)});
    require_ids(lfm3::make_lfm3_chat_prompt(s.tokenizer, 2, false).ids, join({chat_head, chat_tail, {id.think, id.think_end}}), "chat");
    require_ids(lfm3::make_lfm3_chat_prompt(s.tokenizer, 2, true).ids, join({chat_head, chat_tail, {id.think}}), "chat with thinking");

    const auto tts = lfm3::make_lfm3_tts_prompt(s.tokenizer, "Perform TTS.", "abc <think>");
    require_ids(tts.ids,
        join({{id.bos, id.im_start}, s.text("system"), nl, s.text("Perform TTS."), {id.im_end}, nl, {id.im_start}, s.text("user"), nl,
              {lfm3_audio_test::kAbc, ' ', id.think, id.im_end}, nl, {id.im_start}, s.text("assistant"), nl, {id.audio_start}}),
        "TTS prompt, with an added token in the user text as HF parses it");
    // The placeholders would take slots no audio fills: the request's error, not the backbone's.
    for (const char * text : {"Say <|audio|> now", "Say <|audio_out|> now"}) {
        require_request_error([&] { (void)lfm3::make_lfm3_tts_prompt(s.tokenizer, "Perform TTS.", text); }, "holds an audio placeholder token",
            std::string("TTS text \"") + text + "\"");
    }

    for (const int64_t tokens : {int64_t{0}, int64_t{-1}}) {
        lfm3_audio_test::require_throws_with([&] { (void)lfm3::make_lfm3_asr_prompt(s.tokenizer, "x", tokens); }, "needs its audio",
            "an ASR prompt with " + std::to_string(tokens) + " audio tokens");
    }
}

void test_initial_mode() {
    const Synthetic s("lfm3_audio_prompt_test_mode");
    const auto & id = s.ids;
    const int32_t eoa = 2016;
    const auto mode = [&](std::vector<int32_t> ids, std::vector<std::vector<int32_t>> frames = {}) {
        return lfm3::lfm3_initial_mode(lfm3::Lfm3Prompt{std::move(ids), std::move(frames)}, id, eoa);
    };
    require(mode({id.bos, id.audio_start}) == lfm3::Lfm3Mode::Audio, "<|audio_start|> starts in audio");
    require(mode({id.bos, id.mixed_start}) == lfm3::Lfm3Mode::Interleaved, "<|mixed_start|> starts interleaved");
    require(mode({id.bos, id.nl}) == lfm3::Lfm3Mode::Text, "anything else starts in text");
    require(mode({id.bos, id.think, id.think_end}) == lfm3::Lfm3Mode::Text, "the chat prompt starts in text");
    require(mode({}) == lfm3::Lfm3Mode::Text, "an empty prompt");
    require(mode({id.audio_out}, {{1, 2, 3}}) == lfm3::Lfm3Mode::Audio, "a replayed frame goes on in audio");
    require(mode({id.audio_out}, {{eoa, eoa, eoa}}) == lfm3::Lfm3Mode::Text, "a replayed end-of-audio frame goes on in text");
    require(mode({id.audio_out}, {{1, eoa, 3}}) == lfm3::Lfm3Mode::Text, "end-of-audio in any codebook ends the audio");
}

}  // namespace

int main() {
    try {
        test_voices();
        test_templates();
        test_initial_mode();
        std::cout << "lfm3_audio_prompt_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_prompt_test: " << error.what() << '\n';
        return 1;
    }
}
