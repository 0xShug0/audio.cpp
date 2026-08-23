#pragma once

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/models/moss/shared/token_rows.h"

#include <memory>
#include <optional>
#include <string>

namespace engine::models::moss_voicegen {

// Builds the <user_inst> prompt for MOSS-VoiceGenerator. The family shares its template
// with the rest of MOSS-TTS, but this is the voice-design path: the "- Instruction:" slot
// carries the written voice description and there is no reference audio, so the
// "- Reference(s):" slot stays "None".
class MossVoiceGenTextProcessor {
public:
    explicit MossVoiceGenTextProcessor(std::shared_ptr<const MossVoiceGenAssets> assets);
    ~MossVoiceGenTextProcessor();

    MossVoiceGenTextProcessor(const MossVoiceGenTextProcessor &) = delete;
    MossVoiceGenTextProcessor & operator=(const MossVoiceGenTextProcessor &) = delete;

    // `instruction` describes the speaker to design. `language` must be the full language
    // name the model was trained on ("English", not "en"); an empty value renders "None".
    moss::TokenRows build_generation_prefix(
        const std::string & text,
        const std::optional<std::string> & instruction,
        const std::optional<std::string> & language) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_voicegen
