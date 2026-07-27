#pragma once

#include "engine/community_models/kroko_asr/assets.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::kroko_asr {

struct KrokoDecodedTokens {
    std::vector<int32_t> ids;
    std::vector<int64_t> frame_indices;
};

class KrokoGreedyDecoder {
public:
    explicit KrokoGreedyDecoder(std::shared_ptr<const KrokoASRAssets> assets);

    void reset();
    const KrokoDecodedTokens & append(
        const std::vector<float> & encoder_output,
        int64_t frames,
        int64_t hidden_size);
    const KrokoDecodedTokens & decoded() const noexcept;
    KrokoDecodedTokens decode(
        const std::vector<float> & encoder_output,
        int64_t frames,
        int64_t hidden_size);

private:
    std::array<float, 512> predictor(const std::array<int32_t, 2> & context) const;
    int32_t join(
        const float * encoder_frame,
        const std::array<float, 512> & decoder_output) const;

    std::shared_ptr<const KrokoASRAssets> assets_;
    std::vector<float> embedding_;
    std::vector<float> conv_;
    std::vector<float> decoder_projection_;
    std::vector<float> decoder_bias_;
    std::vector<float> joiner_projection_;
    std::vector<float> joiner_bias_;
    std::array<int32_t, 2> context_{};
    std::array<float, 512> decoder_output_{};
    KrokoDecodedTokens decoded_;
    int64_t decoded_frames_ = 0;
};

}  // namespace engine::models::kroko_asr
