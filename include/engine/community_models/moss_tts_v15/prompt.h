#pragma once

// Prompt construction for MOSS-TTS-v1.5.
//
// The user turn is the moss_tts_delay family's eight-field <user_inst> block.
// moss_voicegen renders the same eight fields but fills only the instruction and
// the language, leaving the rest at "None"; v1.5 additionally uses the reference
// slot for voice cloning and the token budget for duration, so the fields are
// carried here as options rather than hard-coded.
//
// A reference recording enters the prompt as its codec codes: the "- Reference(s):"
// slot renders "[S<n>]:" followed by an audio span, and the span is
// audio_start, one audio_user_slot row per delayed frame, audio_end. The codes
// are delay-patterned the same way the model emits them, so the span is
// frames + n_vq - 1 rows rather than frames.

#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"
#include "engine/framework/decoders/moss_tts_delay/prompt.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::moss_tts_v15 {

// One speaker's reference codes, [n_vq][frames] as the codec returns them. The
// family's type: the assembly that consumes it is shared with the other
// checkpoints, so the rendering here hands back exactly what that expects.
using ReferenceAudio = decoders::MossReferenceAudio;

struct PromptFields {
    std::string text;
    std::optional<std::string> instruction;
    std::optional<std::string> tokens;         // duration budget in codec frames
    std::optional<std::string> quality;
    std::optional<std::string> sound_event;
    std::optional<std::string> ambient_sound;
    std::optional<std::string> language;
    std::vector<ReferenceAudio> references;    // empty for instruction-only
};

// The <user_inst> block, with "<|audio|>" standing where each reference span goes.
std::string render_user_inst(const PromptFields & fields);

// The whole generation prefix: the chat wrapper, the rendered block with its
// reference spans expanded into rows, and the assistant header.
codecs::MossTokenRows build_generation_prefix(
    const PromptFields & fields,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer);

// Delay-patterns one reference's codes. Kept as a name in this namespace because
// the prompt parity test reads it; the implementation is the family's.
inline std::vector<std::vector<int32_t>> apply_delay_pattern(
    const ReferenceAudio & reference, int64_t num_codebooks, int32_t audio_pad_code) {
    return decoders::moss_apply_delay_pattern(reference, num_codebooks, audio_pad_code);
}

}  // namespace engine::models::moss_tts_v15
