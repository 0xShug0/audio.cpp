#include "engine/framework/decoders/ctc_host_decoder.h"
#include "engine/framework/runtime/host_ops.h"
#include "engine/framework/debug/profiler.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace engine;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

// Independent full-state recurrence, without banding or packed backpointers.
float reference(const std::vector<float> & scores, int frames, int classes,
    const std::vector<int> & labels, int blank) {
    const float inf = -std::numeric_limits<float>::infinity();
    std::vector<float> previous(labels.size(), inf), next(previous);
    previous[0] = scores[labels[0]];
    if (labels.size() > 1) previous[1] = scores[labels[1]];
    for (int t = 1; t < frames; ++t) {
        for (size_t s = 0; s < labels.size(); ++s) {
            float best = previous[s];
            if (s) best = std::max(best, previous[s - 1]);
            if (s > 1 && labels[s] != blank && labels[s] != labels[s - 2])
                best = std::max(best, previous[s - 2]);
            next[s] = best + scores[static_cast<size_t>(t * classes + labels[s])];
        }
        previous.swap(next);
    }
    return labels.size() > 1 ? std::max(previous.back(), previous[previous.size() - 2]) : previous.back();
}

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        debug::LoggingConfig logging;
        if (argc == 3 && std::string(argv[1]) == "--log") {
            logging.enabled = true;
            logging.file_path = argv[2];
        }
        debug::configure_logging(logging);
        decoders::CtcHostDecoder ctc(0);
        const auto spans = ctc.decode({0, 2, 2, 0, 2, 1, 1, 0});
        require(spans.size() == 3 && spans[0].id == 2 && spans[0].start_frame == 1 && spans[0].end_frame == 3 &&
            spans[1].start_frame == 4 && spans[2].end_frame == 7, "CTC collapse lost blank separation or frame spans");
        require(ctc.decode({}).empty() && ctc.decode({0, 0}).empty(), "empty/all-blank CTC");
        const auto shifted = decoders::CtcHostDecoder(2).decode({2, 0, 0, 2, 0});
        require(shifted.size() == 2 && shifted[0].id == 0, "nonzero blank discarded valid token zero");
        std::vector<float> tied{0, 0, -1, -1, 1, 1};
        require(decoders::CtcHostDecoder::frame_ids(tied.data(), 2, 3) == std::vector<int32_t>({0, 1}), "argmax ties");
        // Cover every short frame sequence, including blank-separated repeats,
        // for callers retaining IDs, first frames, or complete nonblank runs.
        for (int blank : {0, 3}) {
            decoders::CtcHostDecoder decoder(blank);
            for (int pattern = 0; pattern < 4096; ++pattern) {
                std::vector<int32_t> ids(6);
                for (size_t t = 0; t < ids.size(); ++t) ids[t] = (pattern >> (2 * t)) & 3;
                std::vector<decoders::CtcTokenSpan> expected;
                for (size_t begin = 0; begin < ids.size();) {
                    size_t end = begin + 1;
                    while (end < ids.size() && ids[end] == ids[begin]) ++end;
                    if (ids[begin] != blank)
                        expected.push_back({ids[begin], static_cast<int64_t>(begin), static_cast<int64_t>(end)});
                    begin = end;
                }
                const auto actual = decoder.decode(ids);
                require(actual.size() == expected.size(), "CTC run count differs");
                for (size_t i = 0; i < actual.size(); ++i)
                    require(actual[i].id == expected[i].id && actual[i].start_frame == expected[i].start_frame &&
                        actual[i].end_frame == expected[i].end_frame, "CTC run ID or boundary differs");
            }
        }
        // Host argmax keeps the first winner, matching Sense-ASR's original
        // strict-greater comparison, including nonfinite logits.
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        const std::vector<float> logits{nan, 3, 4, 1, nan, 2, -inf, -inf, -inf, 1, inf, inf};
        require(decoders::CtcHostDecoder::frame_ids(logits.data(), 4, 3) == std::vector<int32_t>({0, 2, 0, 1}),
            "CTC host argmax changed nonfinite/tie behavior");
        const auto padded = runtime::CTCDecoder().compute({0, 2, 0, 3, 0, 0, 0, 2, 0, 3, 0, 0}, 2, 2, 3, 0);
        require(padded.values == std::vector<int32_t>({1, 0, 1, 0}), "legacy batched CTC output changed");
        std::mt19937 rng(17);
        for (int blank : {0, 2}) {
            decoders::CtcHostDecoder decoder(blank);
            for (int trial = 0; trial < 100; ++trial) {
                const int frames = 8 + trial % 9, classes = 4;
                const int token = blank == 0 ? 1 : 0;
                const std::vector<int> labels{blank, token, blank, token, blank, 3, blank};
                std::vector<float> scores(static_cast<size_t>(frames * classes));
                for (auto & score : scores) score = -float(rng() % 100) / 13;
                const auto greedy = decoders::CtcHostDecoder::frame_ids(scores.data(), frames, classes);
                std::vector<int> positions(static_cast<size_t>(frames));
                std::iota(positions.begin(), positions.end(), 0);
                const auto full = decoder.align(scores.data(), classes, labels, positions, nullptr, greedy,
                    decoders::CtcAlignmentSearch::Full);
                const auto band = decoder.align(scores.data(), classes, labels, positions, nullptr, greedy);
                require(full.score == reference(scores, frames, classes, labels, blank), "CTC differs from independent recurrence");
                require(full.path == band.path && full.score == band.score, "full-width band changed alignment");
                require(full.path.front() <= 1 && full.path.back() >= int(labels.size()) - 2, "invalid CTC endpoints");
                for (size_t t = 1; t < full.path.size(); ++t) {
                    const int delta = full.path[t] - full.path[t - 1];
                    require(delta >= 0 && delta <= 2, "invalid CTC transition");
                    if (delta == 2) require(labels[full.path[t]] != blank && labels[full.path[t]] != labels[full.path[t - 1]],
                        "CTC skipped repeated label or blank");
                }
            }
        }
        // Exercise the narrow band and prior on a long monotonic path.
        {
            const int frames = 1200, classes = 4;
            std::vector<float> scores(static_cast<size_t>(frames * classes), -5);
            std::vector<int> greedy(frames), positions(frames), labels{0};
            std::vector<float> prior(frames, -0.25f);
            for (int t = 0; t < frames; ++t) {
                greedy[t] = t % 4;
                scores[static_cast<size_t>(t * classes + greedy[t])] = 0;
                positions[t] = t;
                if (t % 4) { labels.push_back(greedy[t]); labels.push_back(0); }
            }
            const auto band = ctc.align(scores.data(), classes, labels, positions, prior.data(), greedy);
            const auto full = ctc.align(scores.data(), classes, labels, positions, prior.data(), greedy, decoders::CtcAlignmentSearch::Full);
            require(band.path == full.path && band.score == full.score, "anchored band/prior differs on monotonic path");
        }
        require(ctc.align(nullptr, 2, {0, 1, 0}, {}, nullptr, {}).path.empty(), "empty searched frame set");
        ctc.release_workspace();
        std::cout << "ctc_host_decoder_test: PASS\n";
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
