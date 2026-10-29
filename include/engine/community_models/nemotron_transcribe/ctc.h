// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted for audio.cpp from NVIDIA/NeMo-Speech.cpp, revision 66d50d91b75a406becd95fbaf39d2c9c13e95d7c.
#pragma once

#include "engine/community_models/nemotron_transcribe/alignment.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::assets { class TensorSource; }
namespace engine::core { class ExecutionContext; }

namespace engine::models::nemotron_transcribe {

// ctc.* metadata and tensors in the encoder GGUF. ConvTranspose1d is converted to
// ctc.upsample.{weight,bias}, a D -> U*D linear layer with consecutive output rows
// j*D+o = bias[o] + sum_i source_weight[i,o,j]*x[i]. All head weights remain F32.
struct CtcConfig {
    int dimension = 1280;
    int heads = 16;
    int layers = 2;
    int ff_hidden = 640;
    int rotary_dims = 40;
    int vocabulary = 23863;
    int blank = 23862;
    int upsampling = 2;
    bool qk_norm = true;
    bool qkv_bias = false;
    float residual_scale = 0.1f;
    float norm_epsilon = 1.0e-5f;
    std::string model_name = "Nemotron3-CTC-Timestamp-Adapter_23k";
};

struct CtcOptions {
    // Windows and their context are in 80 ms encoder frames, before upsampling.
    int head_window = 500;
    int head_left_context = 50;
    int head_right_context = 50;
    bool bf16_emissions = true;
    std::vector<std::string> special_tokens;
};

// Transcript-independent evidence, kept on the host so the LLM can use the device
// while decoding. Only h (D floats per CTC frame), logsumexp and maximum are retained.
// The full recording x vocabulary emission matrix is never materialized.
struct CtcEvidence {
    int encoder_frames = 0;
    int dimension = 0;
    int upsampling = 0;
    int vocabulary = 0;
    std::vector<float> hidden;     // [encoder_frames*upsampling, dimension]
    std::vector<float> logsumexp;  // one value per CTC frame
    std::vector<float> maximum;    // one value per CTC frame
    int frames() const noexcept { return encoder_frames * upsampling; }
};

struct CtcSpeakerEvidence {
    // Sortformer rows on the 80 ms grid. Fit to encoder_frames by truncation or
    // last-row repetition, then linearly resampled to the CTC grid by alignment.
    const float * probabilities = nullptr;
    int probability_frames = 0;
    int speakers = 0;
    // Activity on the original 10 ms mel grid; thresholded at >0.5 for segments.
    const float * activity10 = nullptr;
    int activity10_frames = 0;
    double duration_seconds = 0.0;
    double time_offset = 0.0;
};

struct CtcTimestampResult {
    // "aligned", "partial" (some speaker streams failed), or "unavailable".
    std::string status = "unavailable";
    std::string failure_reason;
    std::string error;
    std::string model_name;
    std::string cleaned_transcript;
    timestamps::AlignmentTargets targets;
    // Words are in public order, rounded to milliseconds. Failed streams produce
    // no word rows. Diarization can still be returned when alignment fails.
    timestamps::AlignmentResult alignment;
};

class CtcTimestampHead {
public:
    // serialized_tokenizer is the decoded ctc.tokenizer.spm_model SentencePiece
    // blob, not Base64. Empty bytes allow head-only component tests; timestamps
    // then reports ctc_tokenizer_unavailable. ExecutionContext must outlive this.
    CtcTimestampHead(std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution, CtcConfig config,
        const std::string & serialized_tokenizer);
    ~CtcTimestampHead();
    CtcTimestampHead(const CtcTimestampHead &) = delete;
    CtcTimestampHead & operator=(const CtcTimestampHead &) = delete;

    const CtcConfig & config() const noexcept;
    CtcEvidence prepare(const float * asr_states, int encoder_frames,
        const CtcOptions & options = {});
    // Log probabilities for the requested vocabulary IDs, [CTC frames, columns].
    std::vector<float> emissions(const CtcEvidence & evidence,
        const std::vector<int> & columns, const CtcOptions & options = {});
    CtcTimestampResult timestamps(const CtcEvidence & evidence,
        const std::string & transcript, const CtcSpeakerEvidence & speakers,
        const CtcOptions & options = {});

    // Per-window CPU/reference checks. Input/output are frame-major; the bridge
    // returns upsampling*encoder_frames rows. Each call has local RoPE positions.
    std::vector<float> bridge(const float * asr_states, int encoder_frames);
    std::vector<float> project(const float * hidden, int ctc_frames,
        const std::vector<int> & columns);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::nemotron_transcribe
