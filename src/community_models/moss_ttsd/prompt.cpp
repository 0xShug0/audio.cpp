#include "engine/community_models/moss_ttsd/prompt.h"

#include <string>

namespace engine::models::moss_ttsd {
namespace {

constexpr const char * kUserInstPrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserInstSuffix = "\n</user_inst>";
constexpr const char * kNoneValue = "None";

// "None" when the caller named no speakers at all -- the reference's
// `reference is None` branch -- and otherwise one entry per speaker, in order.
std::string render_references(const std::vector<std::optional<ReferenceAudio>> & references) {
    if (references.empty()) {
        return kNoneValue;
    }
    std::string out;
    for (size_t i = 0; i < references.size(); ++i) {
        if (i != 0) {
            out += "\n";
        }
        const std::string tag = "[S" + std::to_string(i + 1) + "]";
        // ⚠ THE TWO BRANCHES ARE NOT THE SAME SHAPE, and the difference is not
        // cosmetic. A speaker with audio is "[S1]:" then a newline then the
        // placeholder; one without is "[S1]: None" on a single line, with the
        // colon followed by a space rather than a newline. Rendering the second
        // like the first shifts every token after it.
        out += references[i].has_value() ? (tag + ":\n" + decoders::kMossAudioPlaceholder)
                                         : (tag + ": " + kNoneValue);
    }
    return out;
}

// Only the speakers that brought audio produce a placeholder, so only those are
// handed to the assembly -- in order, which is what pairs each span with its own
// "[S<n>]:" tag.
std::vector<decoders::MossReferenceAudio> present_references(const PromptFields & fields) {
    std::vector<decoders::MossReferenceAudio> present;
    present.reserve(fields.references.size());
    for (const auto & reference : fields.references) {
        if (reference.has_value()) {
            present.push_back(*reference);
        }
    }
    return present;
}

}  // namespace

std::string render_user_inst(const PromptFields & fields) {
    return std::string(kUserInstPrefix) + render_references(fields.references)
        + "\n- Instruction:\n" + decoders::moss_prompt_field(fields.instruction)
        // Baked into this checkpoint's template, not substituted. See prompt.h.
        + "\n- Tokens:\n" + kNoneValue
        + "\n- Quality:\n" + decoders::moss_prompt_field(fields.quality)
        + "\n- Sound Event:\n" + decoders::moss_prompt_field(fields.sound_event)
        + "\n- Ambient Sound:\n" + decoders::moss_prompt_field(fields.ambient_sound)
        + "\n- Language:\n" + decoders::moss_prompt_field(fields.language)
        // Substituted with the literal "None" unconditionally by the reference.
        + "\n- Scene:\n" + kNoneValue
        // str(text), not normalised: the text carries [S1]/[S2] speaker tags.
        + "\n- Text:\n" + fields.text
        + kUserInstSuffix;
}

codecs::MossTokenRows build_generation_prefix(
    const PromptFields & fields,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer) {
    return decoders::build_moss_generation_prefix(
        render_user_inst(fields), present_references(fields), config, tokenizer, "MOSS-TTSD");
}

codecs::MossTokenRows build_continuation_prefix(
    const PromptFields & fields,
    const ReferenceAudio & assistant_audio,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer) {
    return decoders::build_moss_continuation_prefix(
        render_user_inst(fields), present_references(fields), assistant_audio,
        config, tokenizer, "MOSS-TTSD");
}

}  // namespace engine::models::moss_ttsd
