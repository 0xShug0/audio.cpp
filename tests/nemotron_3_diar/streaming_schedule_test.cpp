#include "engine/models/nemotron_3_diar/streaming.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace engine::models::nemotron_3_diar;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<StreamWindow> run(int64_t samples, float value = 0.5F) {
    FeatureConfig frontend;
    StreamingConfig streaming;
    streaming.chunk_len = 9;
    streaming.chunk_right_context = 4;
    StreamScheduler scheduler(frontend, streaming, 8);
    engine::runtime::AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.samples.assign(static_cast<size_t>(samples), value);
    auto windows = scheduler.push_audio(chunk);
    auto tail = scheduler.finalize();
    windows.insert(windows.end(), tail.begin(), tail.end());
    return windows;
}

// NeMo FilterbankFeatures.get_seq_len: floor(samples / hop) valid mel frames.
void test_final_frame_count() {
    for (const int64_t samples : {16000LL, 16090LL, 16159LL, 16160LL, 23999LL}) {
        const auto windows = run(samples);
        require(!windows.empty(), "no windows");
        const auto & last = windows.back();
        const int64_t end = last.mel_start + last.mel_frames;
        require(end == samples / 160,
            "final mel end " + std::to_string(end) + " != floor(" + std::to_string(samples) + " / 160)");
    }
}

// Right padding must be zero after pre-emphasis, as with torch.stft's
// constant padding: x[n] - 0.97 * x[n - 1] == 0 for every synthetic sample.
void test_tail_is_silent_after_preemphasis() {
    const int64_t samples = 16090;
    const auto windows = run(samples);
    const auto & last = windows.back();
    const int64_t feature_start = last.mel_start == 0 ? 0 : last.mel_start * 160 - 256;
    const int64_t real = samples - feature_start;
    require(real > 0 && real < static_cast<int64_t>(last.mono_samples.size()), "final window does not reach the tail");
    for (size_t i = static_cast<size_t>(real); i < last.mono_samples.size(); ++i) {
        const float emphasized = last.mono_samples[i] - 0.97F * last.mono_samples[i - 1];
        require(std::fabs(emphasized) < 1.0e-6F, "tail sample " + std::to_string(i) + " is not silent after pre-emphasis");
    }
}

}  // namespace

int main() {
    try {
        test_final_frame_count();
        test_tail_is_silent_after_preemphasis();
    } catch (const std::exception & error) {
        std::cerr << "nemotron_3_diar_streaming_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "nemotron_3_diar_streaming_test passed\n";
    return 0;
}
