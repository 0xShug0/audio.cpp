#pragma once

// Byte-level BPE tokenizer built from the backbone GGUF's tokenizer.ggml.*
// metadata, so the package needs no separate tokenizer files.

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/framework/runtime/partial_text.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_tokenizer_vendor {
struct BpeVocabulary;
}

namespace engine::community_models::lfm2_audio {

class Lfm2TextTokenizer {
public:
    explicit Lfm2TextTokenizer(const Lfm2TextVocabulary & vocabulary);

    // Special tokens such as <|im_start|> are matched whole.
    [[nodiscard]] std::vector<int32_t> encode(const std::string & text) const;
    // Drops control and unused tokens. The result is the tokens' bytes, which
    // need not be whole UTF-8 characters (see lfm2_take_text).
    [[nodiscard]] std::string decode(const std::vector<int32_t> & token_ids) const;
    [[nodiscard]] int32_t require_token_id(const std::string & token) const;
    [[nodiscard]] bool is_control_token(int32_t token_id) const;

private:
    std::shared_ptr<const llama_tokenizer_vendor::BpeVocabulary> vocab_;
    std::vector<bool> dropped_in_decode_;
};

// Takes the text off the front of decoded bytes, as liquid-audio's Hugging
// Face tokenizer decodes them (String::from_utf8_lossy): well-formed UTF-8 is
// kept, and each maximal subpart of an ill-formed sequence becomes one U+FFFD
// (Unicode 3.9). A token can end inside a character, so a trailing sequence
// that later bytes could still finish stays in `bytes`; when no more bytes
// come it is left out, as runtime::transcript_publishable_end ends a cut
// transcript at its last whole character (liquid-audio shows U+FFFD there).
// Later bytes only add to the text taken before them.
[[nodiscard]] std::string lfm2_take_text(std::string & bytes);

// The text of a reply that comes a few tokens at a time, as the S2S stream's
// events carry it. Each event gets the whole characters written since the
// last one, and together they are the text lfm2_take_text makes of all the
// reply's bytes.
class Lfm2StreamedText {
public:
    // Adds an event's decoded bytes and returns its text. A character they
    // leave open waits for the next event's bytes, and is left out if the
    // reply ends first.
    [[nodiscard]] std::string add(const std::string & bytes);
    // The text of the events so far, without a character still open.
    [[nodiscard]] const std::string & text() const { return partials_.published(); }

private:
    std::string bytes_;                       // a character the bytes so far leave open
    std::string text_;                        // the reply's text so far, all of it in events
    runtime::PartialTextPublisher partials_;  // publishes it; open characters wait in bytes_
};

}  // namespace engine::community_models::lfm2_audio
