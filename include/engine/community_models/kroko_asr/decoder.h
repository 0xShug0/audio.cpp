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

enum class KrokoDecodingMethod {
    GreedySearch,
    ModifiedBeamSearch,
};

struct KrokoDecoderOptions {
    KrokoDecodingMethod method = KrokoDecodingMethod::GreedySearch;
    int32_t max_active_paths = 4;
    float blank_penalty = 0.0F;
    float hotwords_score = 1.5F;
    std::vector<std::vector<int32_t>> hotwords;
};

class KrokoTransducerDecoder {
public:
    explicit KrokoTransducerDecoder(std::shared_ptr<const KrokoASRAssets> assets);
    ~KrokoTransducerDecoder();

    void configure(KrokoDecoderOptions options);
    void reset(int64_t frame_offset = 0);
    void reset_segment(int64_t frame_offset);
    const KrokoDecodedTokens & append(
        const std::vector<float> & encoder_output,
        int64_t frames,
        int64_t hidden_size);
    const KrokoDecodedTokens & decoded() const noexcept;
    int64_t decoded_frames() const noexcept;
    int64_t trailing_blank_frames() const noexcept;
    KrokoDecodedTokens decode(
        const std::vector<float> & encoder_output,
        int64_t frames,
        int64_t hidden_size);

private:
    struct Impl;

    std::array<float, 512> predictor(const std::array<int32_t, 2> & context) const;
    void join_scores(
        const float * encoder_frame,
        const std::array<float, 512> & decoder_output,
        std::vector<float> & scores) const;
    void append_greedy(
        const std::vector<float> & encoder_output,
        int64_t frames,
        int64_t hidden_size);
    void append_modified_beam(
        const std::vector<float> & encoder_output,
        int64_t frames,
        int64_t hidden_size);

    std::shared_ptr<const KrokoASRAssets> assets_;
    std::vector<float> embedding_;
    std::vector<float> conv_;
    std::vector<float> decoder_projection_;
    std::vector<float> decoder_bias_;
    std::vector<float> joiner_projection_;
    std::vector<float> joiner_bias_;
    mutable std::vector<float> joiner_scores_;
    std::array<int32_t, 2> context_{};
    std::array<float, 512> decoder_output_{};
    KrokoDecodedTokens decoded_;
    int64_t decoded_frames_ = 0;
    int64_t trailing_blank_frames_ = 0;
    KrokoDecoderOptions options_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kroko_asr
