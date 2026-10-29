#include "engine/community_models/lfm3_audio/prompt.h"

#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <stdexcept>

namespace engine::community_models::lfm3_audio {

namespace {

// The TTS languages, in the model card's order; `iso` picks the default voice
// ("" for British English, which no code picks).
struct Language {
    const char * slug;
    const char * name;
    const char * iso;
};

constexpr Language kLanguages[] = {
    {"arabic", "Arabic", "ar"},
    {"danish", "Danish", "da"},
    {"german", "German", "de"},
    {"greek", "Greek", "el"},
    {"british_english", "British English", ""},
    {"american_english", "American English", "en"},
    {"finnish", "Finnish", "fi"},
    {"french", "French", "fr"},
    {"hebrew", "Hebrew", "he"},
    {"italian", "Italian", "it"},
    {"japanese", "Japanese", "ja"},
    {"korean", "Korean", "ko"},
    {"dutch", "Dutch", "nl"},
    {"polish", "Polish", "pl"},
    {"portuguese", "Portuguese", "pt"},
    {"spanish", "Spanish", "es"},
    {"swedish", "Swedish", "sv"},
    {"turkish", "Turkish", "tr"},
    {"chinese", "Chinese", "zh"},
};

constexpr const char * kDefaultVoice = "american_english_female";

bool is_auto(const std::string & language) {
    return language.empty() || language == "auto";
}

const Language & language_of(const std::string & iso) {
    for (const auto & language : kLanguages) {
        if (iso == language.iso) {
            return language;
        }
    }

    std::string known;
    for (const auto & language : kLanguages) {
        if (*language.iso != '\0') {
            known += (known.empty() ? "" : ", ") + std::string(language.iso);
        }
    }

    throw runtime::InvalidRequestError("LFM3-Audio does not know language \"" + iso + "\"; it takes auto or one of " + known);
}

void append(std::vector<int32_t> & out, const std::vector<int32_t> & ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

// The template's system block, which it leaves out when the prompt is empty.
std::string system_block(const std::string & system_prompt) {
    return system_prompt.empty() ? std::string() : "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
}

// Tokenizing around the audio equals tokenizing the expanded text: special tokens split the text first.
Lfm3Prompt with_audio(const Lfm3Tokenizer & tokenizer, const std::string & prefix, int64_t audio_tokens, const std::string & suffix) {
    if (audio_tokens <= 0) {
        throw std::runtime_error("LFM3-Audio prompt needs its audio, got " + std::to_string(audio_tokens) + " audio tokens");
    }

    Lfm3Prompt out;
    out.ids = tokenizer.encode(prefix);
    out.ids.insert(out.ids.end(), static_cast<size_t>(audio_tokens), tokenizer.ids().audio);
    append(out.ids, tokenizer.encode(suffix));
    return out;
}

}  // namespace

const std::vector<Lfm3Voice> & lfm3_tts_voices() {
    static const std::vector<Lfm3Voice> voices = [] {
        std::vector<Lfm3Voice> out;
        for (const auto & language : kLanguages) {
            for (const char * gender : {"female", "male"}) {
                out.push_back({std::string(language.slug) + "_" + gender, language.name, gender});
            }
        }

        return out;
    }();
    return voices;
}

const Lfm3Voice & lfm3_tts_voice(const std::string & voice_id, const std::string & language) {
    std::string id = voice_id;
    if (id.empty()) {
        id = is_auto(language) ? kDefaultVoice : std::string(language_of(language).slug) + "_female";
    }

    const auto & voices = lfm3_tts_voices();
    const auto it = std::find_if(voices.begin(), voices.end(), [&](const Lfm3Voice & voice) { return voice.id == id; });
    if (it == voices.end()) {
        throw runtime::InvalidRequestError("LFM3-Audio has no voice " + id + "; its voices are <language>_female and <language>_male, e.g. " +
                                           kDefaultVoice);
    }

    return *it;
}

std::string lfm3_tts_system_prompt(const Lfm3Voice & voice) {
    return "Perform TTS in " + voice.language_name + " using a " + voice.gender + " voice.";
}

std::string lfm3_asr_system_prompt(const std::string & language) {
    if (is_auto(language)) {
        return "Perform ASR.";
    }

    // English has two TTS voices, but ASR names the language once.
    const std::string name = language == "en" ? "English" : language_of(language).name;
    return "Perform ASR in " + name + ".";
}

Lfm3Prompt make_lfm3_asr_prompt(const Lfm3Tokenizer & tokenizer, const std::string & system_prompt, int64_t audio_tokens) {
    return with_audio(tokenizer, "<|startoftext|>" + system_block(system_prompt) + "<|im_start|>user\n", audio_tokens,
                      "<|im_end|>\n<|im_start|>assistant\n");
}

Lfm3Prompt make_lfm3_chat_prompt(const Lfm3Tokenizer & tokenizer, int64_t audio_tokens, bool thinking) {
    return with_audio(tokenizer, "<|startoftext|><|im_start|>user\n", audio_tokens,
                      std::string("<|im_end|>\n<|im_start|>assistant\n") + (thinking ? "<think>" : "<think></think>"));
}

Lfm3Prompt make_lfm3_tts_prompt(const Lfm3Tokenizer & tokenizer, const std::string & system_prompt, const std::string & text) {
    Lfm3Prompt out;
    out.ids = tokenizer.encode("<|startoftext|>" + system_block(system_prompt) + "<|im_start|>user\n" + text +
                               "<|im_end|>\n<|im_start|>assistant\n<|audio_start|>");
    require_lfm3_no_audio_placeholders(out.ids, tokenizer.ids());  // the template has none
    return out;
}

void require_lfm3_no_audio_placeholders(const std::vector<int32_t> & ids, const Lfm3TokenIds & token_ids) {
    // Each one would take a slot that no audio or frame fills.
    if (std::any_of(ids.begin(), ids.end(), [&](int32_t id) { return id == token_ids.audio || id == token_ids.audio_out; })) {
        throw runtime::InvalidRequestError("LFM3-Audio text holds an audio placeholder token (<|audio|> or <|audio_out|>), which it cannot take as text");
    }
}

Lfm3Mode lfm3_initial_mode(const Lfm3Prompt & prompt, const Lfm3TokenIds & ids, int32_t end_of_audio) {
    if (prompt.ids.empty()) {
        return Lfm3Mode::Text;
    }

    const int32_t last = prompt.ids.back();
    if (last == ids.mixed_start) {
        return Lfm3Mode::Interleaved;
    }

    if (last == ids.audio_start) {
        return Lfm3Mode::Audio;
    }

    if (last == ids.audio_out) {
        // A replayed turn that ended on its end-of-audio frame goes on in text.
        const bool ended = !prompt.frames.empty() &&
                           std::find(prompt.frames.back().begin(), prompt.frames.back().end(), end_of_audio) != prompt.frames.back().end();
        return ended ? Lfm3Mode::Text : Lfm3Mode::Audio;
    }

    return Lfm3Mode::Text;
}

}  // namespace engine::community_models::lfm3_audio
