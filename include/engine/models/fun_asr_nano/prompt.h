#pragma once

#include "engine/models/fun_asr_nano/tokenizer_text.h"
#include "engine/models/fun_asr_nano/types.h"

#include <cstdint>
#include <memory>
#include <string>

namespace engine::models::fun_asr_nano {

class FunAsrNanoPromptBuilder {
public:
  explicit FunAsrNanoPromptBuilder(
      std::shared_ptr<const FunAsrNanoAssets> assets);

  FunAsrNanoPrompt build(const FunAsrNanoPromptRequest &request,
                         int64_t audio_tokens) const;
  std::string prompt_text(const FunAsrNanoPromptRequest &request) const;

private:
  std::shared_ptr<const FunAsrNanoAssets> assets_;
  std::shared_ptr<const FunAsrNanoTextTokenizer> tokenizer_;
};

} // namespace engine::models::fun_asr_nano
