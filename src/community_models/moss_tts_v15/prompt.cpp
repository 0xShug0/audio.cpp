#include "engine/community_models/moss_tts_v15/prompt.h"

#include <stdexcept>

namespace engine::models::moss_tts_v15 {
namespace {

// Template fragments copied verbatim from MossTTSDelayProcessor so the encoded
// prompt matches the reference token for token. moss_voicegen carries the same
// eight fields; the difference here is which of them get filled.
constexpr const char * kImStartToken = "<|im_start|>";
constexpr const char * kImEndToken = "<|im_end|>";
constexpr const char * kUserRolePrefix = "user\n";
constexpr const char * kAssistantTurnPrefix = "\n";
constexpr const char * kAssistantRolePrefix = "assistant\n";
constexpr const char * kUserInstPrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserInstSuffix = "\n</user_inst>";
constexpr const char * kNoneValue = "None";
constexpr const char * kAudioPlaceholder = "<|audio|>";

// An unset field renders as the literal "None", and so does one that is present
// but blank -- the reference does str(None) on the dataclass field, and a field
// trimmed to nothing is not a value the template can carry.
std::string field(const std::optional<std::string> & value) {
    if (!value.has_value()) {
        return kNoneValue;
    }
    const auto first = value->find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return kNoneValue;
    }
    const auto last = value->find_last_not_of(" \t\r\n");
    return value->substr(first, last - first + 1);
}

std::string render_references(const std::vector<ReferenceAudio> & references) {
    if (references.empty()) {
        return kNoneValue;
    }
    std::string out;
    for (size_t i = 0; i < references.size(); ++i) {
        if (i != 0) {
            out += "\n";
        }
        out += "[S" + std::to_string(i + 1) + "]:\n" + kAudioPlaceholder;
    }
    return out;
}

}  // namespace

std::string render_user_inst(const PromptFields & fields) {
    return std::string(kUserInstPrefix) + render_references(fields.references)
        + "\n- Instruction:\n" + field(fields.instruction)
        + "\n- Tokens:\n" + field(fields.tokens)
        + "\n- Quality:\n" + field(fields.quality)
        + "\n- Sound Event:\n" + field(fields.sound_event)
        + "\n- Ambient Sound:\n" + field(fields.ambient_sound)
        + "\n- Language:\n" + field(fields.language)
        + "\n- Text:\n" + fields.text
        + kUserInstSuffix;
}

std::vector<std::vector<int32_t>> apply_delay_pattern(
    const ReferenceAudio & reference, int64_t num_codebooks, int32_t audio_pad_code) {
    if (static_cast<int64_t>(reference.codes.size()) != num_codebooks) {
        throw std::runtime_error("MOSS-TTS-v1.5 reference has the wrong codebook count");
    }
    // frames + n_vq - 1, not frames + n_vq: codebook v occupies row t + v, so the
    // last row used is (frames - 1) + (n_vq - 1). The reference builds the same span
    // as `length` gen slots followed by `n_vq - 1` delay slots.
    const int64_t rows = reference.frames + num_codebooks - 1;
    std::vector<std::vector<int32_t>> out(
        static_cast<size_t>(rows), std::vector<int32_t>(static_cast<size_t>(num_codebooks), audio_pad_code));
    for (int64_t v = 0; v < num_codebooks; ++v) {
        for (int64_t t = 0; t < reference.frames; ++t) {
            // Codebook v is delayed by v steps, which is the layout the model
            // both consumes and emits.
            out[static_cast<size_t>(t + v)][static_cast<size_t>(v)] =
                reference.codes[static_cast<size_t>(v)][static_cast<size_t>(t)];
        }
    }
    return out;
}

codecs::MossTokenRows build_generation_prefix(
    const PromptFields & fields,
    const decoders::MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer) {
    codecs::MossTokenRowBuilder builder(
        config.num_codebooks, static_cast<int32_t>(config.audio_pad_code));

    const std::string user_inst = render_user_inst(fields);

    // Encode either side of each placeholder as one span, not fragment by
    // fragment: the reference processor encodes the whole turn in a single pass,
    // and splitting at an arbitrary point would break BPE merges across the seam.
    const std::string head = std::string(kImStartToken) + kUserRolePrefix;
    std::string pending = head + user_inst;

    size_t reference_index = 0;
    for (;;) {
        const auto at = pending.find(kAudioPlaceholder);
        if (at == std::string::npos) {
            break;
        }
        if (reference_index >= fields.references.size()) {
            throw std::runtime_error("MOSS-TTS-v1.5 prompt has more audio slots than references");
        }
        builder.push_text_tokens(tokenizer.encode(pending.substr(0, at), true));
        builder.push_text_token(static_cast<int32_t>(config.audio_start_token_id));
        const auto delayed = apply_delay_pattern(
            fields.references[reference_index], config.num_codebooks,
            static_cast<int32_t>(config.audio_pad_code));
        for (const auto & row : delayed) {
            builder.push_audio_row(
                static_cast<int32_t>(config.audio_user_slot_token_id),
                row.data(),
                config.num_codebooks);
        }
        builder.push_text_token(static_cast<int32_t>(config.audio_end_token_id));
        pending = pending.substr(at + std::string(kAudioPlaceholder).size());
        ++reference_index;
    }

    pending += std::string(kImEndToken) + kAssistantTurnPrefix + kImStartToken + kAssistantRolePrefix;
    builder.push_text_tokens(tokenizer.encode(pending, true));

    // The audio-start token is not seeded here: the model emits it itself on the
    // first step, and generate() reads the last text token to decide whether this
    // is a continuation, so appending it would be taken as one.
    return builder.finish();
}

}  // namespace engine::models::moss_tts_v15
