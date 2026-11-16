#pragma once

#include "engine/community_models/tiny_voca/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::tiny_voca {

// Text -> phoneme ids, with dizione, the model's own phonemizer. This is the
// only place where the phonemizer (src/community_models/tiny_voca/dizione,
// which knows nothing of audio.cpp) meets the framework.
class TinyVocaFrontend {
public:
    struct Utterance {
        std::vector<int32_t> ids;
        std::vector<int32_t> stress;
    };

    explicit TinyVocaFrontend(const TinyVocaAssets & assets);
    ~TinyVocaFrontend();

    // The phonemes of one utterance; nothing when the text has no word,
    // number or letter to say (another script, signs only).
    std::optional<Utterance> phonemize(const std::string & text);

    // The sentences of a text, without a break after a title or an initial.
    static std::vector<std::string> sentences(const std::string & text);
    // The text with a full stop added when it ends without a mark.
    static std::string finished(const std::string & text);
    // The text in smaller pieces: cut after commas and the like if it has
    // any, else in two between words. Empty for a single word.
    static std::vector<std::string> shorter(const std::string & text);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::tiny_voca
