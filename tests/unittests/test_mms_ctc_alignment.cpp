#include "engine/community_models/mms_forced_aligner/ctc_alignment.h"
#include "test_assert.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

using engine::community_models::mms_forced_aligner::CtcAlignmentLimits;
using engine::community_models::mms_forced_aligner::CtcForcedAlignment;
using engine::community_models::mms_forced_aligner::MmsLogProbabilities;
using engine::community_models::mms_forced_aligner::MmsPreparedText;
using engine::community_models::mms_forced_aligner::ctc_forced_align;
using engine::community_models::mms_forced_aligner::mms_word_spans_from_alignment;

constexpr int32_t kBlank = 0;
constexpr int32_t kA = 4;
constexpr int32_t kI = 5;
constexpr int32_t kStar = 31;
constexpr int64_t kClasses = 32;
constexpr float kOffPath = -100.0F;

// Builds log probabilities where the desired label path costs exactly 0 and
// every off-path entry costs -100, making the path uniquely optimal.
std::vector<float> zero_cost_matrix(int64_t frames, const std::vector<int32_t> & labels) {
    require_eq(static_cast<int64_t>(labels.size()), frames, "label path length");
    std::vector<float> matrix(static_cast<size_t>(frames * kClasses), kOffPath);
    for (int64_t frame = 0; frame < frames; ++frame) {
        matrix[static_cast<size_t>(frame * kClasses + labels[static_cast<size_t>(frame)])] = 0.0F;
    }
    return matrix;
}

CtcForcedAlignment align(
    const std::vector<float> & matrix,
    const std::vector<int32_t> & targets,
    CtcAlignmentLimits limits = {}) {
    MmsLogProbabilities log_probs{matrix.data(), static_cast<int64_t>(matrix.size()) / kClasses, kClasses};
    return ctc_forced_align(log_probs, targets, kBlank, limits);
}

MmsPreparedText two_word_prepared() {
    MmsPreparedText prepared;
    prepared.canonical_language = "eng";
    prepared.original_words = {"a", "i"};
    prepared.normalized_words = {"a", "i"};
    prepared.target_ids = {kStar, kA, kStar, kI};
    prepared.target_to_word = {-1, 0, -1, 1};
    return prepared;
}

void test_single_token_blanks() {
    const std::vector<int32_t> targets{kA};
    const std::vector<int32_t> labels{kBlank, kA, kA, kBlank, kBlank};
    const auto matrix = zero_cost_matrix(5, labels);
    const auto alignment = align(matrix, targets);
    require_eq(static_cast<int64_t>(alignment.label_path.size()), int64_t{5}, "path length");
    for (size_t i = 0; i < labels.size(); ++i) {
        require_eq(alignment.label_path[i], labels[i], "label path");
    }
    require_eq(alignment.state_path[0], int32_t{0}, "leading blank state");
    require_eq(alignment.state_path[1], int32_t{1}, "target state");
    require_eq(alignment.state_path[4], int32_t{2}, "trailing blank state");
    require_close(alignment.score, 0.0F, 1.0e-6F, "path score");
}

void test_repeated_letters_need_blank() {
    const std::vector<int32_t> targets{kA, kA};
    const std::vector<int32_t> labels{kBlank, kA, kBlank, kA};
    const auto matrix = zero_cost_matrix(4, labels);
    const auto alignment = align(matrix, targets);
    for (size_t i = 0; i < labels.size(); ++i) {
        require_eq(alignment.label_path[i], labels[i], "repeated letter path");
    }
    require_eq(alignment.state_path[2], int32_t{2}, "blank between repeats");
}

void test_stay_tie_wins() {
    const std::vector<int32_t> targets{kA};
    // f0 and f1 both tie between blank and target; stay beats advance-one.
    std::vector<float> matrix(2 * kClasses, kOffPath);
    matrix[0] = 0.0F;
    matrix[static_cast<size_t>(kA)] = 0.0F;
    matrix[static_cast<size_t>(kClasses + kBlank)] = 0.0F;
    matrix[static_cast<size_t>(kClasses + kA)] = 0.0F;
    const auto alignment = align(matrix, targets);
    require_eq(alignment.state_path[0], int32_t{1}, "f0 target via stay tie");
    require_eq(alignment.state_path[1], int32_t{1}, "f1 target via stay tie");
}

void test_terminal_tie_prefers_target() {
    const std::vector<int32_t> targets{kA};
    std::vector<float> matrix(kClasses, kOffPath);
    matrix[0] = 0.0F;
    matrix[static_cast<size_t>(kA)] = 0.0F;
    const auto alignment = align(matrix, targets);
    require_eq(alignment.state_path[0], int32_t{1}, "final target wins the tie");
    require_eq(alignment.label_path[0], kA, "label is the target");
    require_close(alignment.score, 0.0F, 1.0e-6F, "score");
}

void test_word_spans_with_star_sharing() {
    const std::vector<int32_t> targets{kStar, kA, kStar, kI};
    const std::vector<int32_t> labels{kBlank, kStar, kBlank, kA, kBlank, kStar, kBlank, kI, kBlank};
    const auto matrix = zero_cost_matrix(9, labels);
    const auto alignment = align(matrix, targets);
    const auto spans = mms_word_spans_from_alignment(alignment, two_word_prepared(), kBlank, 0.0F);
    require_eq(static_cast<int64_t>(spans.size()), int64_t{2}, "two word spans");
    require_eq(spans[0].word_index, int32_t{0}, "first word index");
    require_eq(spans[0].start_frame, int64_t{2}, "first word start shares full leading blank");
    require_eq(spans[0].end_frame, int64_t{4}, "first word end at blank midpoint");
    require_eq(spans[1].word_index, int32_t{1}, "second word index");
    require_eq(spans[1].start_frame, int64_t{6}, "second word start at blank midpoint");
    require_eq(spans[1].end_frame, int64_t{8}, "last word end takes full trailing blank");
    require_close(spans[0].log_prob, 0.0F, 1.0e-6F, "mean path score");
}

void test_odd_blank_midpoint() {
    const std::vector<int32_t> targets{kStar, kA, kStar, kI};
    // 3-frame blanks around the middle star: word0's trailing blank spans
    // frames 3..5 (midpoint 4); word1's leading blank spans 7..9 (midpoint 8).
    const std::vector<int32_t> labels{
        kBlank, kStar, kA, kBlank, kBlank, kBlank, kStar, kBlank, kBlank, kBlank, kI, kBlank};
    const auto matrix = zero_cost_matrix(12, labels);
    const auto alignment = align(matrix, targets);
    const auto spans = mms_word_spans_from_alignment(alignment, two_word_prepared(), kBlank, 0.0F);
    require_eq(spans[0].start_frame, int64_t{2}, "first word starts at its letter segment");
    require_eq(spans[0].end_frame, int64_t{4}, "midpoint of frames 3..5");
    require_eq(spans[1].start_frame, int64_t{8}, "midpoint of frames 7..9");
    require_eq(spans[1].end_frame, int64_t{11}, "last word takes the full trailing blank");
}

void test_word_first_interval_midpoint() {
    // Segment mode: the leading <star> is interval 0, so the first word's
    // multi-frame leading blank is shared at its midpoint, not taken whole.
    const std::vector<int32_t> targets{kStar, kA, kStar, kI};
    const std::vector<int32_t> labels{
        kBlank, kStar, kBlank, kBlank, kBlank, kA, kBlank, kStar, kBlank, kI, kBlank};
    const auto matrix = zero_cost_matrix(11, labels);
    const auto alignment = align(matrix, targets);
    const auto spans = mms_word_spans_from_alignment(alignment, two_word_prepared(), kBlank, 0.0F);
    require_eq(spans[0].start_frame, int64_t{3}, "first word leading blank at midpoint, not full");
    require_eq(spans[0].end_frame, int64_t{6}, "first word trailing blank midpoint");
    require_eq(spans[1].start_frame, int64_t{8}, "second word leading blank midpoint");
    require_eq(spans[1].end_frame, int64_t{10}, "last word retains the full trailing blank (segment mode)");
}

MmsPreparedText edges_prepared() {
    MmsPreparedText prepared;
    prepared.canonical_language = "eng";
    prepared.original_words = {"a", "i"};
    prepared.normalized_words = {"a", "i"};
    prepared.target_ids = {kStar, kA, kI, kStar};
    prepared.target_to_word = {-1, 0, 1, -1};
    return prepared;
}

void test_edges_mode_last_interval_midpoint() {
    // Edges mode: a trailing <star> is the last interval, so the final word's
    // multi-frame trailing blank is shared at the midpoint, not taken whole.
    const std::vector<int32_t> targets{kStar, kA, kI, kStar};
    const std::vector<int32_t> labels{
        kBlank, kStar, kBlank, kA, kI, kBlank, kBlank, kBlank, kStar, kBlank};
    const auto matrix = zero_cost_matrix(10, labels);
    const auto alignment = align(matrix, targets);
    const auto spans = mms_word_spans_from_alignment(alignment, edges_prepared(), kBlank, 0.0F);
    require_eq(spans[0].start_frame, int64_t{2}, "first word leading blank midpoint");
    require_eq(spans[0].end_frame, int64_t{3}, "first word span");
    require_eq(spans[1].start_frame, int64_t{4}, "last word leading blank midpoint");
    require_eq(spans[1].end_frame, int64_t{6}, "edges-mode last word trailing blank at midpoint, not full");
}

void test_merge_threshold() {
    const std::vector<int32_t> targets{kStar, kA, kStar, kI};
    // No blank between star and letters: the star segment (frame 1 and 3)
    // leaves a 1-frame gap (inclusive end frames) between the word spans.
    const std::vector<int32_t> labels{kBlank, kStar, kA, kStar, kI, kBlank};
    const auto matrix = zero_cost_matrix(6, labels);
    const auto alignment = align(matrix, targets);
    // Third gap is 1 frame at a 3-frame threshold: merged start must be
    // previous.end + 1 so exclusive-end samples stay contiguous (clipping to
    // previous.end would overlap the earlier word by one 320-sample frame).
    const auto spans = mms_word_spans_from_alignment(alignment, two_word_prepared(), kBlank, 0.05F);
    require_eq(spans[0].end_frame, int64_t{2}, "first word span");
    require_eq(spans[1].start_frame, int64_t{3}, "merged start is previous end + 1");
    // Gap count is off-by-one sensitive at a 1-frame threshold: the real gap
    // is exactly 1 frame, so a 0.02 s (1-frame) threshold must still merge.
    const auto tight = mms_word_spans_from_alignment(alignment, two_word_prepared(), kBlank, 0.02F);
    require_eq(tight[1].start_frame, int64_t{3}, "1-frame gap merges at a 1-frame threshold");
}

void expect_rejected(const std::vector<int32_t> & targets, int64_t frames, CtcAlignmentLimits limits = {}) {
    std::vector<float> matrix(static_cast<size_t>(frames * kClasses), 0.0F);
    bool threw = false;
    try {
        (void) align(matrix, targets, limits);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "expected alignment rejection");
}

void test_impossible_frame_count() {
    // L=2 with one repeat needs T >= 3.
    expect_rejected({kA, kA}, 2);
}

void test_target_contains_blank() {
    expect_rejected({kBlank, kA}, 4);
}

void test_target_out_of_range() {
    expect_rejected({kA, kClasses}, 4);
    expect_rejected({-1}, 4);
}

void test_empty_targets() {
    expect_rejected({}, 4);
}

void test_max_target_tokens() {
    CtcAlignmentLimits limits;
    limits.max_target_tokens = 2;
    expect_rejected({kStar, kA, kStar}, 6, limits);
}

void test_max_alignment_cells() {
    CtcAlignmentLimits limits;
    limits.max_alignment_cells = 10;
    // 6 frames x 3 states = 18 cells > 10.
    expect_rejected({kA}, 6, limits);
}

void test_null_log_probs() {
    bool threw = false;
    try {
        MmsLogProbabilities log_probs{nullptr, 4, kClasses};
        (void) ctc_forced_align(log_probs, {kA}, kBlank, {});
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "null log probs must be rejected");
}

void test_star_excluded_from_spans() {
    const std::vector<int32_t> targets{kStar, kA};
    const std::vector<int32_t> labels{kBlank, kStar, kBlank, kA, kBlank};
    const auto matrix = zero_cost_matrix(5, labels);
    const auto alignment = align(matrix, targets);
    auto prepared = two_word_prepared();
    prepared.original_words = {"a"};
    prepared.normalized_words = {"a"};
    prepared.target_ids = {kStar, kA};
    prepared.target_to_word = {-1, 0};
    const auto spans = mms_word_spans_from_alignment(alignment, prepared, kBlank, 0.0F);
    require_eq(static_cast<int64_t>(spans.size()), int64_t{1}, "only the real word survives");
    require_eq(spans[0].start_frame, int64_t{2}, "full leading blank for first word");
    require_eq(spans[0].end_frame, int64_t{4}, "full trailing blank for last word");
}

}  // namespace

int main() {
    try {
        test_single_token_blanks();
        test_repeated_letters_need_blank();
        test_stay_tie_wins();
        test_terminal_tie_prefers_target();
        test_word_spans_with_star_sharing();
        test_odd_blank_midpoint();
        test_word_first_interval_midpoint();
        test_edges_mode_last_interval_midpoint();
        test_merge_threshold();
        test_impossible_frame_count();
        test_target_contains_blank();
        test_target_out_of_range();
        test_empty_targets();
        test_max_target_tokens();
        test_max_alignment_cells();
        test_null_log_probs();
        test_star_excluded_from_spans();
        std::cout << "mms_ctc_alignment_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "mms_ctc_alignment_test: %s\n", error.what());
        return 1;
    }
}
