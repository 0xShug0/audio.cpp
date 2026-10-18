#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"

#include <optional>
#include <unordered_map>
#include <vector>

namespace engine::models::plapre {

struct PlapreAudioReference {
    std::vector<int32_t> text;
    std::vector<int32_t> codes;
};

struct PlaprePromptControls {
    std::optional<PlapreAudioReference> voice_reference;
    std::optional<std::vector<int32_t>> context_text;
    std::optional<std::vector<int32_t>> context_codes;
    bool same_speaker = true;
    std::optional<std::vector<int64_t>> duration_frames;
    std::vector<PlapreAudioReference> pronunciations;
};

class PlaprePromptBuilder {
public:
    explicit PlaprePromptBuilder(const tokenizers::LlamaBpeTokenizer & tokenizer);

    std::vector<int32_t> generate(const std::vector<int32_t> & text,
        const PlaprePromptControls & controls) const;
    std::vector<int32_t> edit(const std::vector<int32_t> & text,
        const std::vector<int32_t> & original_codes, int64_t start, int64_t end,
        const std::vector<PlapreAudioReference> & pronunciations) const;
    std::vector<int32_t> stop_ids(bool editing) const;
    std::vector<int32_t> audio_codes(const std::vector<int32_t> & generated) const;

private:
    std::vector<int32_t> pronunciation_prefix(const std::vector<PlapreAudioReference> & references) const;
    void append_audio(std::vector<int32_t> & prompt, const std::vector<int32_t> & codes,
        size_t start, size_t end) const;

    std::unordered_map<std::string, int32_t> tokens_;
};

}  // namespace engine::models::plapre
