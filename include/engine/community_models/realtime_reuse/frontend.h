#pragma once

#include "engine/community_models/realtime_reuse/runtime.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"

#include <complex>
#include <deque>

namespace engine::models::realtime_reuse {

struct Spectrum {
    audio::STFTConfig config;
    audio::AudioTensor spectrum;
    std::vector<float> features;
    int64_t samples = 0, frames = 0, bins = 0;
};
Spectrum analyze_spectrum(const std::vector<float> & waveform, int sample_rate, size_t threads);
std::vector<float> synthesize_spectrum(Spectrum & input, const std::array<std::vector<float>, 3> & heads, size_t threads);
std::vector<float> restore(Runtime & runtime, const std::vector<float> & waveform, int sample_rate, size_t threads);

class StreamingFrontend {
public:
    StreamingFrontend(Runtime & runtime, int sample_rate, size_t threads);
    std::vector<float> push(const std::vector<float> & samples, bool final);

private:
    void analyze(bool final, std::vector<float> & output);
    void infer(std::vector<float> & output, bool final = false);
    Runtime & runtime_;
    size_t threads_;
    int64_t fft_, hop_, bins_;
    int64_t received_ = 0, input_offset_ = 0, analyzed_ = 0, synthesized_ = 0, emitted_ = 0;
    bool finished_ = false;
    int64_t batch_frames_ = 1;
    std::shared_ptr<audio::RealFFTPlan> plan_;
    std::vector<float> window_, frame_, overlap_, denominator_;
    std::vector<std::complex<float>> spectrum_;
    std::deque<float> input_, pending_audio_;
    std::deque<std::vector<float>> features_;
};

} // namespace engine::models::realtime_reuse
