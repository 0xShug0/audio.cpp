#include "waveform.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char * name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

}  // namespace

int main() {
    std::vector<float> partial_final_window(250, 0.0F);
    for (size_t i = 0; i < 200; ++i) {
        partial_final_window[i] = 0.1F;
    }
    partial_final_window[240] = 0.2F;
    engine::models::ming3_audio::detail::trim_trailing_silence(partial_final_window, 1000);
    expect(partial_final_window.size() == 250, "keep the audible partial final window");
    expect(partial_final_window[240] == 0.2F, "preserve audio samples in the partial final window");

    std::vector<float> trailing_silence(500, 0.0F);
    for (size_t i = 0; i < 100; ++i) {
        trailing_silence[i] = 0.1F;
    }
    engine::models::ming3_audio::detail::trim_trailing_silence(trailing_silence, 1000);
    expect(trailing_silence.size() == 400, "trim silence while retaining the 300 ms tail");

    bool rejected_invalid_rate = false;
    try {
        std::vector<float> samples(1, 0.1F);
        engine::models::ming3_audio::detail::trim_trailing_silence(samples, 0);
    } catch (const std::invalid_argument &) {
        rejected_invalid_rate = true;
    }
    expect(rejected_invalid_rate, "reject an invalid sample rate");

    std::printf("%s\n", failures == 0 ? "PASS: Ming3 Audio waveform trimming" : "FAIL: Ming3 Audio waveform trimming");
    return failures == 0 ? 0 : 1;
}
