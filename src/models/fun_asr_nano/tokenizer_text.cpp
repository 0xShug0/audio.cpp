#include "engine/models/fun_asr_nano/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fun_asr_nano {

struct FunAsrNanoTextTokenizer::Impl {
  std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

namespace {

std::shared_ptr<const FunAsrNanoTextTokenizer::Impl>
load_impl(const FunAsrNanoAssets &assets) {
  engine::tokenizers::LlamaBpeTokenizerSpec spec;
  spec.tokenizer_config_path =
      assets.resources.require_file("tokenizer_config");
  if (const auto *path = assets.resources.find_file("vocab")) {
    spec.vocab_path = *path;
  }
  if (const auto *path = assets.resources.find_file("merges")) {
    spec.merges_path = *path;
  }
  spec.tokenizer_json_path = assets.resources.require_file("tokenizer_json");
  spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;

  auto impl = std::make_shared<FunAsrNanoTextTokenizer::Impl>();
  impl->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
  return impl;
}

} // namespace

FunAsrNanoTextTokenizer::FunAsrNanoTextTokenizer(
    std::shared_ptr<const FunAsrNanoAssets> assets)
    : assets_(std::move(assets)) {
  if (assets_ == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano tokenizer requires assets");
  }
  impl_ = load_impl(*assets_);
  if (require_token_id("<|object_ref_start|>") !=
          assets_->config.audio_token_id ||
      require_token_id("<|im_end|>") != assets_->config.text.eos_token_id) {
    throw std::runtime_error(
        "Fun-ASR-Nano tokenizer special token ids do not match config");
  }
}

std::vector<int32_t>
FunAsrNanoTextTokenizer::encode(const std::string &text) const {
  return impl_->tokenizer->encode(text);
}

std::string
FunAsrNanoTextTokenizer::decode(const std::vector<int32_t> &token_ids) const {
  std::vector<int32_t> filtered;
  filtered.reserve(token_ids.size());
  for (const int32_t token_id : token_ids) {
    if (token_id == assets_->config.text.eos_token_id ||
        impl_->tokenizer->is_control_token_id(token_id)) {
      continue;
    }
    filtered.push_back(token_id);
  }
  return impl_->tokenizer->decode(filtered);
}

int32_t
FunAsrNanoTextTokenizer::require_token_id(const std::string &token) const {
  const auto token_id = impl_->tokenizer->find_token_id(token);
  if (!token_id.has_value()) {
    throw std::runtime_error("Fun-ASR-Nano tokenizer is missing token: " +
                             token);
  }
  return *token_id;
}

} // namespace engine::models::fun_asr_nano
