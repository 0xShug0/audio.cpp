#include "engine/community_models/moss_tts_v15/prompt.h"

#include "engine/framework/text/moss_tts_normalization.h"

#include <string>

namespace engine::models::moss_tts_v15 {
namespace {

// The block's own scaffolding. Everything else -- the chat wrapper, the delay
// pattern, the placeholder splicing -- is the family's, in
// engine/framework/decoders/moss_tts_delay/prompt.h.
constexpr const char * kUserInstPrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserInstSuffix = "\n</user_inst>";
constexpr const char * kNoneValue = "None";

std::string render_references(const std::vector<ReferenceAudio> & references) {
    if (references.empty()) {
        return kNoneValue;
    }
    std::string out;
    for (size_t i = 0; i < references.size(); ++i) {
        if (i != 0) {
            out += "\n";
        }
        out += "[S" + std::to_string(i + 1) + "]:\n" + decoders::kMossAudioPlaceholder;
    }
    return out;
}

}  // namespace

std::string render_user_inst(const PromptFields & fields) {
    // The reference normalises the text as it builds the message
    // (processing_moss_tts.py, build_user_message), not later, so the prompt the
    // model sees is the normalised one. Clean prose is unchanged by this.
    //
    // ⚠ THIS IS v1.5-ONLY. The other checkpoints in the family render the text
    // with str(), and MOSS-TTSD in particular carries speaker tags that a
    // normaliser has no business touching -- which is why the rendering stays
    // per-model while the assembly is shared.
    const std::string text = engine::text::normalize_moss_tts_text(fields.text);
    return std::string(kUserInstPrefix) + render_references(fields.references)
        + "\n- Instruction:\n" + decoders::moss_prompt_field(fields.instruction)
        + "\n- Tokens:\n" + decoders::moss_prompt_field(fields.tokens)
        + "\n- Quality:\n" + decoders::moss_prompt_field(fields.quality)
        + "\n- Sound Event:\n" + decoders::moss_prompt_field(fields.sound_event)
        + "\n- Ambient Sound:\n" + decoders::moss_prompt_field(fields.ambient_sound)
        + "\n- Language:\n" + decoders::moss_prompt_field(fields.language)
        + "\n- Text:\n" + text
        + kUserInstSuffix;
}

codecs::MossTokenRows build_generation_prefix(
    const PromptFields & fields,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer) {
    return decoders::build_moss_generation_prefix(
        render_user_inst(fields), fields.references, config, tokenizer, "MOSS-TTS-v1.5");
}

}  // namespace engine::models::moss_tts_v15
