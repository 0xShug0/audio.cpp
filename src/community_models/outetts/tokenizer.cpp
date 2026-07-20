#include "engine/community_models/outetts/tokenizer.h"

#include "engine/framework/text/utf8.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace engine::models::outetts {
namespace {

int32_t require_id(const engine::tokenizers::LlamaBpeTokenizer &tokenizer,
                   const std::string &token) {
  const auto id = tokenizer.find_token_id(token);
  if (!id.has_value()) {
    throw std::runtime_error("OuteTTS tokenizer is missing token: " + token);
  }
  return *id;
}

bool is_punctuation(char value) {
  return value == ',' || value == '.' || value == '?' || value == '!' ||
         value == ':' || value == ';';
}

std::string trim_text(const std::string &input) {
  const auto first = std::find_if_not(
      input.begin(), input.end(),
      [](unsigned char value) { return std::isspace(value); });
  const auto last = std::find_if_not(
      input.rbegin(), input.rend(),
      [](unsigned char value) { return std::isspace(value); })
                        .base();
  return first < last ? std::string(first, last) : std::string{};
}

std::string normalize_text(const std::string &input) {
  std::string text;
  text.reserve(input.size());
  bool pending_space = false;
  for (const unsigned char value : input) {
    if (value < 0x20 || value == 0x7f || value == '"')
      continue;
    if (std::isspace(value)) {
      pending_space = !text.empty();
      continue;
    }
    const char character = static_cast<char>(value);
    if (is_punctuation(character)) {
      while (!text.empty() && text.back() == ' ')
        text.pop_back();
      if ((character == '?' || character == '!') && !text.empty() &&
          text.back() == character) {
        pending_space = true;
        continue;
      }
      if (character == '.' && text.size() >= 3 &&
          text[text.size() - 1] == '.' && text[text.size() - 2] == '.' &&
          text[text.size() - 3] == '.') {
        pending_space = true;
        continue;
      }
      text.push_back(character);
      pending_space = true;
      continue;
    }
    if (pending_space && !text.empty())
      text.push_back(' ');
    text.push_back(character);
    pending_space = false;
  }
  while (!text.empty() && text.back() == ' ')
    text.pop_back();
  return text;
}

std::string speaker_separator(const std::string &text) {
  if (text.empty())
    return {};
  const char last = text.back();
  return last == '.' || last == '?' || last == '!' ? " " : ". ";
}

std::string profile_codes(const OuteTTSVoiceProfile &profile,
                          const std::string &separator) {
  std::ostringstream out;
  for (size_t word_index = 0; word_index < profile.words.size(); ++word_index) {
    const auto &word = profile.words[word_index];
    std::string word_text = trim_text(word.text);
    if (word_index + 1 == profile.words.size())
      word_text += trim_text(separator);
    out << "<|word_start|>" << word_text << "<|features|><|t_" << std::fixed
        << std::setprecision(2) << word.duration << "|>"
        << "<|energy_" << word.features.energy << "|>"
        << "<|spectral_centroid_" << word.features.spectral_centroid << "|>"
        << "<|pitch_" << word.features.pitch << "|>"
        << "<|code|>";
    const size_t frames =
        std::min(word.codebook1.size(), word.codebook2.size());
    for (size_t frame = 0; frame < frames; ++frame) {
      out << "<|c1_" << word.codebook1[frame] << "|>"
          << "<|c2_" << word.codebook2[frame] << "|>";
    }
    out << "<|word_end|>";
    if (word_index + 1 != profile.words.size())
      out << '\n';
  }
  return out.str();
}

} // namespace

OuteTTSTextGenerationBudget
estimate_text_generation_budget(std::string_view text) {
  OuteTTSTextGenerationBudget out;
  size_t position = 0;
  while (position < text.size()) {
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])) != 0) {
      ++position;
    }
    if (position >= text.size())
      break;
    const size_t begin = position;
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])) == 0) {
      ++position;
    }
    ++out.words;
    out.non_whitespace_codepoints += static_cast<int64_t>(
        engine::text::utf8_codepoint_count(text.substr(begin, position - begin),
                                           "OuteTTS text"));
  }
  const int64_t by_words = out.words * 72;
  const int64_t by_characters = out.non_whitespace_codepoints * 12;
  out.recommended_max_new_tokens =
      std::max<int64_t>({256, by_words, by_characters}) + 128;
  return out;
}

struct OuteTTSTokenizer::Impl {
  explicit Impl(const OuteTTSAssets &assets)
      : tokenizer({
            {},
            {},
            assets.resources.require_file("tokenizer_config"),
            assets.resources.require_file("tokenizer"),
            engine::tokenizers::LlamaBpePreTokenizer::Llama3,
        }),
        eos(require_id(tokenizer, "<|im_end|>")),
        audio_end(require_id(tokenizer, "<|audio_end|>")),
        word_end(require_id(tokenizer, "<|word_end|>")) {
    for (int32_t code = 0; code <= 1024; ++code) {
      c1.emplace(require_id(tokenizer, "<|c1_" + std::to_string(code) + "|>"),
                 code);
      c2.emplace(require_id(tokenizer, "<|c2_" + std::to_string(code) + "|>"),
                 code);
    }
  }

  engine::tokenizers::LlamaBpeTokenizer tokenizer;
  int32_t eos = 0;
  int32_t audio_end = 0;
  int32_t word_end = 0;
  std::unordered_map<int32_t, int32_t> c1;
  std::unordered_map<int32_t, int32_t> c2;
};

OuteTTSTokenizer::OuteTTSTokenizer(
    std::shared_ptr<const OuteTTSAssets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("OuteTTS tokenizer requires assets");
  }
  impl_ = std::make_shared<Impl>(*assets);
}

std::vector<int32_t>
OuteTTSTokenizer::build_prompt(const std::string &text) const {
  if (text.empty()) {
    throw std::runtime_error("OuteTTS requires non-empty text");
  }
  const std::string prompt_text = normalize_text(text);
  const std::string prompt = "<|im_start|>\n<|text_start|>" + prompt_text +
                             "<|text_end|>\n<|audio_start|>\n";
  return impl_->tokenizer.encode(prompt, true);
}

std::vector<int32_t>
OuteTTSTokenizer::build_clone_prompt(const std::string &text,
                                     const OuteTTSVoiceProfile &profile) const {
  const std::string prompt_text = normalize_text(text);
  const std::string reference_text = trim_text(profile.text);
  if (prompt_text.empty() || reference_text.empty() || profile.words.empty()) {
    throw std::runtime_error("OuteTTS voice cloning requires text, "
                             "reference_text, and reference codec frames");
  }
  const std::string separator = speaker_separator(reference_text);
  const std::string merged = reference_text + separator + prompt_text;
  const std::string prompt = "<|im_start|>\n<|text_start|>" + merged +
                             "<|text_end|>\n<|audio_start|>\n" +
                             profile_codes(profile, separator) +
                             "\n<|word_start|>";
  return impl_->tokenizer.encode(prompt, true);
}

bool OuteTTSTokenizer::is_stop_token(int32_t token) const noexcept {
  return token == impl_->eos || token == impl_->audio_end;
}

int32_t OuteTTSTokenizer::eos_id() const noexcept { return impl_->eos; }

int32_t OuteTTSTokenizer::audio_end_id() const noexcept {
  return impl_->audio_end;
}

bool OuteTTSTokenizer::append_audio_code(
    int32_t token, std::vector<int32_t> &codebook1,
    std::vector<int32_t> &codebook2) const {
  if (const auto it = impl_->c1.find(token); it != impl_->c1.end()) {
    if (it->second < 1024) {
      codebook1.push_back(it->second);
    }
    return true;
  }
  if (const auto it = impl_->c2.find(token); it != impl_->c2.end()) {
    if (it->second < 1024) {
      codebook2.push_back(it->second);
    }
    return true;
  }
  return false;
}

} // namespace engine::models::outetts
