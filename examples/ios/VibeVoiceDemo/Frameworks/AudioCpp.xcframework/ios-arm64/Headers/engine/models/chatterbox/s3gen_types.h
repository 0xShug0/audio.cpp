#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::chatterbox {

struct S3FlowEncoderOutputs {
    std::vector<float> hidden;
    int64_t frames = 0;
    int64_t storage_frames = 0;
    int64_t hidden_size = 0;
};

struct S3Token2MelOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct S3FlowDecoderOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
    int64_t storage_frames = 0;
};

struct S3FlowCFMOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
    int64_t storage_frames = 0;
};

struct S3GenInferenceOutputs {
    std::vector<float> waveform;
    int64_t samples = 0;
    std::vector<float> source;
    int64_t source_channels = 0;
    int64_t source_frames = 0;
    std::vector<float> mel;
    int64_t mel_channels = 0;
    int64_t mel_frames = 0;
};

struct S3SpeakerEncoderWeights;
struct S3TokenizerV2Weights;
struct S3FlowEncoderWeights;
struct S3FlowDecoderWeights;
struct S3HiFTVocoderWeights;

}  // namespace engine::models::chatterbox
