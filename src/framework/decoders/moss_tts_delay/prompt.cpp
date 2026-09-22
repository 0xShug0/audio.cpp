#include "engine/framework/decoders/moss_tts_delay/prompt.h"

#include <stdexcept>
#include <string>

namespace engine::decoders {
namespace {

// Template fragments copied verbatim from MossTTSDelayProcessor so the encoded
// prompt matches the reference token for token.
constexpr const char * kImStartToken = "<|im_start|>";
constexpr const char * kImEndToken = "<|im_end|>";
constexpr const char * kUserRolePrefix = "user\n";
constexpr const char * kAssistantTurnPrefix = "\n";
constexpr const char * kAssistantRolePrefix = "assistant\n";
constexpr const char * kNoneValue = "None";

}  // namespace

std::string moss_prompt_field(const std::optional<std::string> & value) {
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

std::vector<std::vector<int32_t>> moss_apply_delay_pattern(
    const MossReferenceAudio & reference, int64_t num_codebooks, int32_t audio_pad_code) {
    if (static_cast<int64_t>(reference.codes.size()) != num_codebooks) {
        throw std::runtime_error("MOSS reference has the wrong codebook count");
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

namespace {

// The user turn plus the assistant header, which both modes share. Left as a
// builder rather than finished rows so continuation can append its span.
codecs::MossTokenRowBuilder build_prefix_through_assistant_header(
    const std::string & user_inst,
    const std::vector<MossReferenceAudio> & references,
    const MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view model_label) {
    codecs::MossTokenRowBuilder builder(
        config.num_codebooks, static_cast<int32_t>(config.audio_pad_code));

    // Encode either side of each placeholder as one span, not fragment by
    // fragment: the reference processor encodes the whole turn in a single pass,
    // and splitting at an arbitrary point would break BPE merges across the seam.
    const std::string head = std::string(kImStartToken) + kUserRolePrefix;
    std::string pending = head + user_inst;

    size_t reference_index = 0;
    for (;;) {
        const auto at = pending.find(kMossAudioPlaceholder);
        if (at == std::string::npos) {
            break;
        }
        if (reference_index >= references.size()) {
            throw std::runtime_error(
                std::string(model_label) + " prompt has more audio slots than references");
        }
        builder.push_text_tokens(tokenizer.encode(pending.substr(0, at), true));
        builder.push_text_token(static_cast<int32_t>(config.audio_start_token_id));
        const auto delayed = moss_apply_delay_pattern(
            references[reference_index], config.num_codebooks,
            static_cast<int32_t>(config.audio_pad_code));
        for (const auto & row : delayed) {
            builder.push_audio_row(
                static_cast<int32_t>(config.audio_user_slot_token_id),
                row.data(),
                config.num_codebooks);
        }
        builder.push_text_token(static_cast<int32_t>(config.audio_end_token_id));
        pending = pending.substr(at + std::string(kMossAudioPlaceholder).size());
        ++reference_index;
    }
    // A rendered block that named fewer speakers than the caller supplied codes
    // for would silently drop a voice, so say so rather than generate with it.
    if (reference_index != references.size()) {
        throw std::runtime_error(
            std::string(model_label) + " prompt has fewer audio slots than references");
    }

    pending += std::string(kImEndToken) + kAssistantTurnPrefix + kImStartToken + kAssistantRolePrefix;
    builder.push_text_tokens(tokenizer.encode(pending, true));
    return builder;
}

}  // namespace

codecs::MossTokenRows build_moss_generation_prefix(
    const std::string & user_inst,
    const std::vector<MossReferenceAudio> & references,
    const MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view model_label) {
    auto builder = build_prefix_through_assistant_header(
        user_inst, references, config, tokenizer, model_label);
    // The audio-start token is not seeded here: the model emits it itself on the
    // first step, and generate() reads the last text token to decide whether this
    // is a continuation, so appending it would be taken as one.
    return builder.finish();
}

codecs::MossTokenRows build_moss_continuation_prefix(
    const std::string & user_inst,
    const std::vector<MossReferenceAudio> & references,
    const MossReferenceAudio & assistant_audio,
    const MossTtsDelayConfig & config,
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view model_label) {
    auto builder = build_prefix_through_assistant_header(
        user_inst, references, config, tokenizer, model_label);

    // Here the audio-start token IS written: this turn's audio has begun, and it
    // is the caller supplying it rather than the model.
    builder.push_text_token(static_cast<int32_t>(config.audio_start_token_id));
    const auto delayed = moss_apply_delay_pattern(
        assistant_audio, config.num_codebooks, static_cast<int32_t>(config.audio_pad_code));
    // Only the first `frames` rows -- see the note in the header on why the span
    // stops short of the delay slots and the audio_end.
    for (int64_t row = 0; row < assistant_audio.frames; ++row) {
        builder.push_audio_row(
            static_cast<int32_t>(config.audio_assistant_gen_slot_token_id),
            delayed[static_cast<size_t>(row)].data(),
            config.num_codebooks);
    }
    return builder.finish();
}

}  // namespace engine::decoders
