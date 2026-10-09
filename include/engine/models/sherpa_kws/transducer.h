#pragma once

#include "engine/models/sherpa_kws/assets.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::sherpa_kws {

class TransducerScorer {
public:
    explicit TransducerScorer(std::shared_ptr<const SherpaKwsAssets> assets);

    std::vector<float> score(
        const float * encoder_frame,
        const std::array<int32_t, 2> & context) const;

private:
    std::vector<float> predictor(const std::array<int32_t, 2> & context) const;

    std::shared_ptr<const SherpaKwsAssets> assets_;
    std::vector<float> embedding_;
    std::vector<float> convolution_;
    std::vector<float> decoder_projection_;
    std::vector<float> decoder_bias_;
    std::vector<float> joiner_projection_;
    std::vector<float> joiner_bias_;
    mutable std::vector<float> activated_;
};

}  // namespace engine::models::sherpa_kws
