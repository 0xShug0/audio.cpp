#include "engine/models/fun_asr_nano/prompt.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::fun_asr_nano {
namespace {

std::string language_name(const std::string &language) {
  if (language.empty() || language == "auto") {
    return {};
  }
  if (language == "zh") {
    return "中文";
  }
  if (language == "en") {
    return "英文";
  }
  if (language == "ja") {
    return "日文";
  }
  throw std::runtime_error("unsupported Fun-ASR-Nano prompt language: " +
                           language);
}

} // namespace

FunAsrNanoPromptBuilder::FunAsrNanoPromptBuilder(
    std::shared_ptr<const FunAsrNanoAssets> assets)
    : assets_(std::move(assets)),
      tokenizer_(std::make_shared<FunAsrNanoTextTokenizer>(assets_)) {
  if (assets_ == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano prompt builder requires assets");
  }
}

std::string FunAsrNanoPromptBuilder::prompt_text(
    const FunAsrNanoPromptRequest &request) const {
  if (!request.prompt.empty()) {
    return request.prompt;
  }
  std::string prompt = "语音转写";
  const std::string language = language_name(request.language);
  if (!language.empty()) {
    prompt += "成" + language;
  }
  if (!request.enable_itn) {
    prompt += "，不进行文本规整";
  }
  return prompt + "：";
}

FunAsrNanoPrompt
FunAsrNanoPromptBuilder::build(const FunAsrNanoPromptRequest &request,
                               int64_t audio_tokens) const {
  if (audio_tokens <= 0 ||
      audio_tokens >
          static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        "Fun-ASR-Nano prompt requires a valid positive audio token count");
  }
  const std::string chat =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n" +
      prompt_text(request) +
      "<|object_ref_start|><|im_end|>\n<|im_start|>assistant\n";
  const auto ids = tokenizer_->encode(chat);
  const int32_t audio_token =
      static_cast<int32_t>(assets_->config.audio_token_id);
  if (std::count(ids.begin(), ids.end(), audio_token) != 1) {
    throw std::runtime_error(
        "Fun-ASR-Nano prompt must contain exactly one audio placeholder");
  }

  FunAsrNanoPrompt result;
  result.input_ids.reserve(ids.size() + static_cast<size_t>(audio_tokens - 1));
  for (const int32_t token_id : ids) {
    if (token_id == audio_token) {
      for (int64_t index = 0; index < audio_tokens; ++index) {
        result.audio_token_positions.push_back(
            static_cast<int32_t>(result.input_ids.size()));
        result.input_ids.push_back(token_id);
      }
    } else {
      result.input_ids.push_back(token_id);
    }
  }
  result.attention_mask.assign(result.input_ids.size(), 1);
  return result;
}

} // namespace engine::models::fun_asr_nano
