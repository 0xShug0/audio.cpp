#pragma once

#include "engine/models/fun_asr_nano/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::fun_asr_nano {

class FunAsrNanoTextTokenizer {
public:
  struct Impl;

  explicit FunAsrNanoTextTokenizer(
      std::shared_ptr<const FunAsrNanoAssets> assets);

  std::vector<int32_t> encode(const std::string &text) const;
  std::string decode(const std::vector<int32_t> &token_ids) const;
  int32_t require_token_id(const std::string &token) const;

private:
  std::shared_ptr<const FunAsrNanoAssets> assets_;
  std::shared_ptr<const Impl> impl_;
};

} // namespace engine::models::fun_asr_nano
