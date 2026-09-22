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
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::moss_tts_v15 {

// One speaker's reference codes, [frames][n_vq] as the codec returns them.
struct ReferenceAudio {
    std::vector<std::vector<int32_t>> codes;  // [n_vq][frames], codec layout
    int64_t frames = 0;
};

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

// Delay-patterns one reference's codes: row t of codebook v carries frame t - v,
// and the rest is audio_pad_code. Returns frames + n_vq - 1 rows.
std::vector<std::vector<int32_t>> apply_delay_pattern(
    const ReferenceAudio & reference, int64_t num_codebooks, int32_t audio_pad_code);

}  // namespace engine::models::moss_tts_v15
