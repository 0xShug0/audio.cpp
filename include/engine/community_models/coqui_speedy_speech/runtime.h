#pragma once

#include "engine/community_models/coqui_speedy_speech/assets.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <vector>

namespace engine::community_models::coqui_speedy_speech {

class NativeRuntime {
public:
  NativeRuntime(std::shared_ptr<const Assets> assets,
                engine::core::BackendConfig backend);
  ~NativeRuntime();
  engine::runtime::AudioBuffer synthesize(const std::vector<int32_t> &tokens,
                                          float speaking_rate);

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace engine::community_models::coqui_speedy_speech
