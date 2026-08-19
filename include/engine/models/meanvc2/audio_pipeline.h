#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::models::meanvc2 {

struct MeanVC2FbankWindow {
    std::vector<float> values;
    int64_t offset = 0;
};

struct MeanVC2PreparedAudio {
    std::vector<float> mono_16k;
};

class MeanVC2StreamingFrontend {
public:
    MeanVC2StreamingFrontend();

    void reset();
    std::vector<MeanVC2FbankWindow> encode_chunk(const float * samples, int64_t sample_count);

private:
    std::vector<float> samples_cache_;
    std::vector<float> fbank_cache_;
    bool has_fbank_cache_ = false;
    int64_t asr_offset_ = 8;
};

class MeanVC2BnStreamAdapter {
public:
    void reset();
    std::vector<float> append_encoded_frames(const std::vector<float> & encoded_frames);

private:
    std::vector<float> last_frame_;
};

MeanVC2PreparedAudio prepare_meanvc2_audio_16k(const runtime::AudioBuffer & audio);
MeanVC2PreparedAudio prepare_meanvc2_source_audio_16k(const runtime::AudioBuffer & audio);

}  // namespace engine::models::meanvc2
