#include "engine/community_models/nemotron_transcribe/alignment.h"
#include "engine/community_models/nemotron_transcribe/ctc.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace model = engine::models::nemotron_transcribe;
namespace ts = model::timestamps;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function&& function, const char* message) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

int main() {
    try {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        ts::AlignmentTargets targets;
        targets.columns = {10, 3};
        targets.words = {{"test", 0, 0}};
        targets.streams = {{0, {0}, {0, 1, 0}, {-1, 0, -1}}};
        std::vector<float> scores{0, -2, -2, 0, 0, -2};
        ts::AlignmentInputs inputs;
        inputs.log_probs = scores.data();
        inputs.frames = 3;
        inputs.duration = .12;
        const auto valid = ts::align_record(targets, inputs);
        require(valid.words.size() == 1 && valid.failed_streams.empty(), "valid one-word alignment");

        auto bad = targets;
        bad.streams[0].labels[1] = 2;
        rejects([&] { ts::align_record(bad, inputs); }, "out-of-range class accepted");
        bad = targets;
        bad.streams[0].state_to_word[1] = 1;
        rejects([&] { ts::align_record(bad, inputs); }, "out-of-range state word accepted");
        bad = targets;
        bad.streams[0].state_to_word.pop_back();
        rejects([&] { ts::align_record(bad, inputs); }, "short state mapping accepted");
        bad = targets;
        bad.streams[0].words[0] = -1;
        rejects([&] { ts::align_record(bad, inputs); }, "negative word index accepted");
        bad = targets;
        bad.words[0].tag = 1;
        rejects([&] { ts::align_record(bad, inputs); }, "speaker tag/stream mismatch accepted");
        for (float invalid : {nan, inf}) {
            scores[0] = invalid;
            rejects([&] { ts::align_record(targets, inputs); }, "invalid emissions accepted");
        }
        scores[0] = 0;
        auto bad_inputs = inputs;
        bad_inputs.duration = std::numeric_limits<double>::infinity();
        rejects([&] { ts::align_record(targets, bad_inputs); }, "infinite clock accepted");
        ts::AlignmentConfig config;
        config.maximum_token_len = nan;
        rejects([&] { ts::align_record(targets, inputs, config); }, "NaN token duration accepted");
        config = {};
        config.epsilon = 0;
        rejects([&] { ts::align_record(targets, inputs, config); }, "zero logarithm epsilon accepted");
        std::vector<float> activity{1, nan, 1};
        bad_inputs = inputs;
        bad_inputs.speaker_probs = activity.data();
        bad_inputs.speaker_frames = 3;
        bad_inputs.speakers = 1;
        rejects([&] { ts::align_record(targets, bad_inputs); }, "NaN speaker evidence accepted");
        rejects([&] { ts::diarization_segments(activity.data(), 3, 1, .01, .03); },
            "NaN activity became an inactive segment");
        rejects([&] { ts::resample_speaker_probs(activity.data(), 3, 1, 6); },
            "NaN resampling accepted");
        {
            // Beyond 2^24, float32 linspace can round rows-1 up to rows.
            // Endpoint gathering must stay inside the original matrix.
            std::vector<float> large(16777220, .25f);
            large.front() = .75f;
            const auto endpoints = ts::resample_speaker_probs(large.data(), int(large.size()), 1, 2);
            require(endpoints == std::vector<float>({.75f, .25f}), "float32 endpoint gathered past input rows");
        }
        rejects([&] { ts::banded_ctc_search(scores.data(), 2, {0, 2, 0}, {0, 1}, nullptr, {0, 1}); },
            "search accepted invalid class");
        for (const std::vector<int> positions : {std::vector<int>{-1}, {2}, {0, 0}, {1, 0}})
            rejects([&] { ts::banded_ctc_search(scores.data(), 2, {0, 1, 0}, positions, nullptr, {0, 1}); },
                "search accepted invalid timeline position");
        rejects([&] { ts::speaker_mapping({2}, 2, targets.streams, activity.data(), 1, 1, .5f); },
            "speaker mapping accepted invalid class");

        std::fill(scores.begin(), scores.end(), -inf);
        const auto unavailable = ts::align_record(targets, inputs);
        require(unavailable.words.empty() && unavailable.failed_streams.size() == 1,
            "impossible negative-infinity emissions fabricated a word");
        require(ts::banded_ctc_search(nullptr, 2, {0, 1, 0}, {}, nullptr, {}).path.empty(),
            "an empty search must remain an unavailable path");

        // More streams than available speaker columns preserves the aligned
        // stream and reports the unassigned stream, rather than changing tags.
        targets.columns = {10, 3, 4};
        targets.words = {{"first", 7, 0}, {"second", 2, 1}};
        targets.streams = {{7, {0}, {0, 1, 0}, {-1, 0, -1}},
                           {2, {1}, {0, 2, 0}, {-1, 0, -1}}};
        scores = {0, -2, -4, -2, 0, -4, 0, -2, -4};
        activity = {1, 1, 1};
        inputs.log_probs = scores.data();
        inputs.speaker_probs = activity.data();
        inputs.speaker_frames = 3;
        inputs.speakers = 1;
        const auto partial = ts::align_record(targets, inputs);
        require(partial.words.size() == 1 && partial.failed_streams.size() == 1 &&
            partial.words[0].tag == 7 && partial.words[0].column == 0 && partial.failed_streams[0].tag == 2,
            "partial alignment lost tag/column distinction");

        const auto windows = ts::head_windows(INT_MAX, INT_MAX - 1, INT_MAX, INT_MAX);
        require(windows.size() == 2 && windows[0].end == INT_MAX - 1 &&
            windows[1].start == INT_MAX - 1 && windows[1].end == INT_MAX &&
            windows[1].context_start == 0 && windows[1].context_end == INT_MAX,
            "window arithmetic overflowed");
        rejects([&] { ts::head_windows(10, 5, -1, 0); }, "negative window context accepted");
        model::CtcEvidence evidence;
        evidence.encoder_frames = INT_MAX;
        evidence.upsampling = 2;
        rejects([&] { evidence.frames(); }, "evidence clock overflow accepted");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Nemotron CTC/alignment safety checks passed\n";
}
