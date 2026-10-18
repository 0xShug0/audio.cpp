#include "engine/models/plapre/prompt.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::plapre {

PlaprePromptBuilder::PlaprePromptBuilder(const tokenizers::LlamaBpeTokenizer & tokenizer) {
    for (const auto * name : {"<text>", "<audio>", "</audio>", "<audio_0>", "<dur_0>",
        "<dur>", "</dur>", "<ref_text>", "</ref_text>", "<ref_audio>", "</ref_audio>",
        "<context>", "</context>", "<spk_same>", "<spk_diff>", "<ctx_audio>", "</ctx_audio>",
        "<edit_mask>", "<edit_fill>", "<edit_end>", "<lex>", "</lex>", "<lex_word>",
        "</lex_word>", "<lex_audio>", "</lex_audio>", "<eos>"}) {
        const auto id = tokenizer.find_token_id(name);
        if (!id) throw std::runtime_error("Plapre tokenizer is missing " + std::string(name));
        tokens_.emplace(name, *id);
    }
    for (int i = 0; i < 12800; ++i) {
        if (tokenizer.find_token_id("<audio_" + std::to_string(i) + ">") != tokens_.at("<audio_0>") + i) {
            throw std::runtime_error("Plapre audio vocabulary is not contiguous");
        }
    }
    for (int i = 0; i < 400; ++i) {
        if (tokenizer.find_token_id("<dur_" + std::to_string(i) + ">") != tokens_.at("<dur_0>") + i) {
            throw std::runtime_error("Plapre duration vocabulary is not contiguous");
        }
    }
}

void PlaprePromptBuilder::append_audio(std::vector<int32_t> & prompt, const std::vector<int32_t> & codes,
    size_t start, size_t end) const {
    for (size_t i = start; i < end; ++i) {
        if (codes[i] < 0 || codes[i] >= 12800) throw std::runtime_error("invalid Kanade audio code");
        prompt.push_back(tokens_.at("<audio_0>") + codes[i]);
    }
}

std::vector<int32_t> PlaprePromptBuilder::pronunciation_prefix(
    const std::vector<PlapreAudioReference> & references) const {
    if (references.empty()) return {};
    std::vector<int32_t> prompt{tokens_.at("<lex>")};
    size_t total = 0;
    for (size_t i = 0; i < std::min<size_t>(10, references.size()) && total < 200; ++i) {
        const auto & ref = references[i];
        const auto count = std::min({size_t{60}, size_t{200} - total, ref.codes.size()});
        prompt.push_back(tokens_.at("<lex_word>"));
        prompt.insert(prompt.end(), ref.text.begin(), ref.text.end());
        prompt.push_back(tokens_.at("</lex_word>"));
        prompt.push_back(tokens_.at("<lex_audio>"));
        append_audio(prompt, ref.codes, 0, count);
        prompt.push_back(tokens_.at("</lex_audio>"));
        total += count;
    }
    prompt.push_back(tokens_.at("</lex>"));
    return prompt;
}

std::vector<int32_t> PlaprePromptBuilder::generate(const std::vector<int32_t> & text,
    const PlaprePromptControls & controls) const {
    auto prompt = pronunciation_prefix(controls.pronunciations);
    if (controls.voice_reference) {
        const auto & ref = *controls.voice_reference;
        prompt.push_back(tokens_.at("<ref_text>"));
        prompt.insert(prompt.end(), ref.text.begin(), ref.text.end());
        prompt.push_back(tokens_.at("</ref_text>"));
        prompt.push_back(tokens_.at("<ref_audio>"));
        append_audio(prompt, ref.codes, 0, std::min<size_t>(250, ref.codes.size()));
        prompt.push_back(tokens_.at("</ref_audio>"));
    }
    if (controls.context_text) {
        prompt.push_back(tokens_.at("<context>"));
        prompt.push_back(tokens_.at(controls.same_speaker ? "<spk_same>" : "<spk_diff>"));
        prompt.insert(prompt.end(), controls.context_text->begin(), controls.context_text->end());
        if (controls.context_codes) {
            prompt.push_back(tokens_.at("<ctx_audio>"));
            append_audio(prompt, *controls.context_codes, 0, std::min<size_t>(150, controls.context_codes->size()));
            prompt.push_back(tokens_.at("</ctx_audio>"));
        }
        prompt.push_back(tokens_.at("</context>"));
    }
    prompt.push_back(tokens_.at("<text>"));
    prompt.insert(prompt.end(), text.begin(), text.end());
    if (controls.duration_frames) {
        prompt.push_back(tokens_.at("<dur>"));
        for (const auto frames : *controls.duration_frames) {
            prompt.push_back(tokens_.at("<dur_0>") + static_cast<int32_t>(std::clamp<int64_t>(frames, 0, 399)));
        }
        prompt.push_back(tokens_.at("</dur>"));
    }
    prompt.push_back(tokens_.at("<audio>"));
    return prompt;
}

std::vector<int32_t> PlaprePromptBuilder::edit(const std::vector<int32_t> & text,
    const std::vector<int32_t> & original_codes, int64_t start, int64_t end,
    const std::vector<PlapreAudioReference> & pronunciations) const {
    if (start < 0 || end < start || end > static_cast<int64_t>(original_codes.size())) {
        throw std::runtime_error("Plapre edit span is outside the reference audio");
    }
    auto prompt = pronunciation_prefix(pronunciations);
    prompt.push_back(tokens_.at("<text>"));
    prompt.insert(prompt.end(), text.begin(), text.end());
    prompt.push_back(tokens_.at("<audio>"));
    append_audio(prompt, original_codes, 0, static_cast<size_t>(start));
    prompt.push_back(tokens_.at("<edit_mask>"));
    append_audio(prompt, original_codes, static_cast<size_t>(end), original_codes.size());
    prompt.push_back(tokens_.at("<edit_fill>"));
    prompt.push_back(tokens_.at("<edit_mask>"));
    return prompt;
}

std::vector<int32_t> PlaprePromptBuilder::stop_ids(bool editing) const {
    return {tokens_.at("<eos>"), tokens_.at(editing ? "<edit_end>" : "</audio>")};
}

std::vector<int32_t> PlaprePromptBuilder::audio_codes(const std::vector<int32_t> & generated) const {
    std::vector<int32_t> codes;
    const auto base = tokens_.at("<audio_0>");
    for (const auto token : generated) {
        if (token >= base && token < base + 12800) codes.push_back(token - base);
    }
    return codes;
}

}  // namespace engine::models::plapre
