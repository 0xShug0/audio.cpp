#include "engine/models/firered_vad/runtime.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace engine::models::firered_vad;

void check(const std::vector<float> & probabilities, const DetectionOptions & options,
           const std::vector<std::pair<int64_t, int64_t>> & expected) {
    const int64_t samples = probabilities.empty() ? 0 : (probabilities.size() - 1) * 160 + 400;
    auto actual = segment_probabilities(probabilities, samples, options);
    if (actual.size() != expected.size()) throw std::runtime_error("VAD segment count differs from upstream fixture");
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i].span.start_sample != expected[i].first || actual[i].span.end_sample != expected[i].second)
            throw std::runtime_error("VAD timestamps differ from upstream fixture: segment=" + std::to_string(i) +
                " actual=" + std::to_string(actual[i].span.start_sample) + "," + std::to_string(actual[i].span.end_sample) +
                " expected=" + std::to_string(expected[i].first) + "," + std::to_string(expected[i].second));
}

int main() {
    try {
        DetectionOptions defaults;
        std::vector<float> speech(110, 0.0f);
        std::fill(speech.begin() + 40, speech.begin() + 80, 1.0f);
        check(speech, defaults, {{5760, 16480}});
        std::vector<float> short_speech(20, 0.0f);
        std::fill(short_speech.begin() + 10, short_speech.begin() + 15, 1.0f);
        check(short_speech, defaults, {});
        check({}, defaults, {});
        check(std::vector<float>(100, 0.0f), defaults, {});
        DetectionOptions split;
        split.smooth_window_frames = 3; split.min_speech_frames = 3;
        split.max_speech_frames = 10; split.min_silence_frames = 2;
        check(std::vector<float>(50, 1.0f), split,
              {{0,800},{960,1760},{1920,2720},{2880,3680},{3840,4640},{4800,5600},{5760,6560},{6720,8240}});
        split.max_speech_frames = 100; split.merge_silence_frames = 15; split.extend_speech_frames = 2;
        std::vector<float> merge(50, 0.0f);
        std::fill(merge.begin() + 5, merge.begin() + 15, 1.0f);
        std::fill(merge.begin() + 25, merge.begin() + 35, 1.0f);
        check(merge, split, {{160,6400}});
        std::cout << "FireRed VAD rules match official postprocessor fixtures\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
