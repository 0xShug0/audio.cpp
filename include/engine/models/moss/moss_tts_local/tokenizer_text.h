#pragma once

#include "engine/models/moss/shared/token_rows.h"
#include "engine/models/moss/moss_tts_local/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::moss_tts_local {

// Decoder input for a generation request: the text channel (input_ids[..., 0]) plus the
// n_vq audio channels flattened row-major as [seq, n_vq] (input_ids[..., 1:]). Every audio
// slot of the prompt carries audio_pad_token_id, matching MossTTSLocalProcessor._build_text_rows.
using MossGenerationPrefix = moss::TokenRows;

// Reproduces the direct-generation branch of MossTTSLocalProcessor: it renders the
// <user_inst> template, byte-level BPE encodes each piece with the Qwen tokenizer, and
// splices in the im_start/im_end/audio_start ids to open the assistant audio turn. The
// voice-clone variant additionally embeds a reference speaker's RLFQ codes under
// "- Reference(s):" (audio_start, audio_user_slot rows carrying the codes, audio_end),
// mirroring MossTTSLocalProcessor._build_generation_or_voice_clone_codes.
class MossTextProcessor {
public:
    explicit MossTextProcessor(std::shared_ptr<const MossTTSLocalAssets> assets);
    ~MossTextProcessor();

    MossTextProcessor(const MossTextProcessor &) = delete;
    MossTextProcessor & operator=(const MossTextProcessor &) = delete;

    MossGenerationPrefix build_generation_prefix(
        const std::string & text,
        const std::optional<std::string> & language = std::nullopt) const;

    // Builds a voice-clone prompt. reference_codes is [num_codebooks][frames] as produced
    // by MossAudioTokenizerEncoder for the reference speaker.
    MossGenerationPrefix build_clone_prefix(
        const std::string & text,
        const std::vector<std::vector<int32_t>> & reference_codes,
        const std::optional<std::string> & language = std::nullopt) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_tts_local
