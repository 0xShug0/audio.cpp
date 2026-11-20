#pragma once

#include "engine/community_models/kale_tts/assets.h"
#include "engine/community_models/kale_tts/ettin_tokenizer.h"
#include "engine/framework/text/espeak_phonemizer.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::kale_tts {

// Text -> phoneme ids. Words come from the model's own checked word list (CMUdict-based, with hand-checked overrides and
// spoken forms of technical terms); words it does not have, and numbers, go through eSpeak-ng. The IPA is then written
// in Kokoro's own symbols, as the model was trained.
class KaleFrontend {
public:
    KaleFrontend(const KaleAssets & assets, std::filesystem::path espeak_library, std::filesystem::path espeak_data);
    ~KaleFrontend();

    struct Sentence {
        std::string text;
        std::vector<int32_t> ids;                       // no pads
        std::vector<std::vector<int32_t>> word_ids;     // per word of `text`
    };
    std::vector<Sentence> sentences(const std::string & text, const std::string & accent) const;
    // the sentences of the text, as they are written (after the term list); no phonemes
    std::vector<std::string> sentence_texts(const std::string & text) const;
    // Kokoro-symbol phonemes (a caller's own G2P) -> ids; throws on a symbol the model does not have.
    std::vector<int32_t> phoneme_ids(const std::string & phonemes) const;
    const EttinTokenizer * ettin() const { return ettin_.get(); }

private:
    std::vector<int32_t> ids(const std::string & ipa, bool british) const;
    std::string word_ipa(const std::string & word, bool british) const;
    const KaleAssets & assets_;
    std::unordered_map<std::string, std::string> words_us_, words_gb_, terms_;
    std::filesystem::path espeak_library_, espeak_data_;
    mutable std::unique_ptr<engine::audio::EspeakPhonemizer> espeak_us_, espeak_gb_;
    std::unique_ptr<EttinTokenizer> ettin_;
};

std::vector<std::string> kokoro_native(const std::vector<std::string> & chars, bool british);

}  // namespace engine::models::kale_tts
