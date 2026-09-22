#pragma once

// Prompt construction for MOSS-TTSD, the dialogue member of the moss_tts_delay
// family.
//
// It renders the same <user_inst> block as MOSS-TTS-v1.5 and splices reference
// audio in the same way -- both are handled by the family code in
// engine/framework/decoders/moss_tts_delay/prompt.h -- and differs in four
// places, all of them taken from this checkpoint's processing_moss_tts.py:
//
//   1. "- Tokens:" is the literal "None". The field is baked into this
//      checkpoint's template rather than substituted, so there is no duration
//      budget to expose: TTSD stops when the dialogue is spoken.
//   2. A "- Scene:" slot sits between Language and Text, and is ALSO always
//      "None" -- the message dataclass carries a `scene` field but the renderer
//      does .replace("{scene}", "None") unconditionally, so it never reaches the
//      prompt. Exposing it as an option would be exposing something the model
//      has never been given.
//   3. A speaker may appear WITHOUT a reference, rendering "[S<n>]: None"
//      instead of an audio span. That is how a dialogue asks for one voice to be
//      cloned and the other invented.
//   4. The text is NOT normalised. v1.5 normalises as it builds the message;
//      this checkpoint renders str(text). The text here carries [S1]/[S2]
//      speaker tags, which a normaliser has no business rewriting.

#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"
#include "engine/framework/decoders/moss_tts_delay/prompt.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::moss_ttsd {

using ReferenceAudio = decoders::MossReferenceAudio;

// One turn of the dialogue prompt. `references` is positional: entry i is
// speaker [S(i+1)], and an entry without audio renders as "[S<n>]: None".
struct PromptFields {
    std::string text;                                        // "[S1] ... [S2] ..."
    std::optional<std::string> instruction;
    std::optional<std::string> quality;
    std::optional<std::string> sound_event;
    std::optional<std::string> ambient_sound;
    std::optional<std::string> language;
    std::vector<std::optional<ReferenceAudio>> references;
};

// The <user_inst> block, with "<|audio|>" standing where each reference span goes.
std::string render_user_inst(const PromptFields & fields);

// The whole generation prefix: the chat wrapper, the rendered block with its
// reference spans expanded into rows, and the assistant header.
codecs::MossTokenRows build_generation_prefix(
    const PromptFields & fields,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer);

// The cloning prefix: the same user turn, plus an assistant turn carrying the
// reference recording that the generated dialogue continues from. This is the
// mode the model card uses, and `assistant_audio` is normally the speakers'
// references concatenated in the order they are introduced.
codecs::MossTokenRows build_continuation_prefix(
    const PromptFields & fields,
    const ReferenceAudio & assistant_audio,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer);

}  // namespace engine::models::moss_ttsd
