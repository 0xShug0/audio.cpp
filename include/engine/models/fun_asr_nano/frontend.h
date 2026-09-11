#pragma once

#include "engine/models/fun_asr_nano/assets.h"
#include "engine/models/fun_asr_nano/types.h"

#include <vector>

namespace engine::models::fun_asr_nano {

class FunAsrNanoFrontend {
public:
  explicit FunAsrNanoFrontend(FunAsrNanoFrontendConfig config);

  FunAsrNanoAudioFeatures extract(const std::vector<float> &audio,
                                  int sample_rate) const;

private:
  FunAsrNanoFrontendConfig config_;
};

} // namespace engine::models::fun_asr_nano
