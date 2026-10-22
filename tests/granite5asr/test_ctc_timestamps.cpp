#include "engine/community_models/granite5asr/ctc_timestamps.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::community_models::granite5asr::CtcArgmaxFrame;
using engine::community_models::granite5asr::CtcPiece;
using engine::community_models::granite5asr::DecodedCtcPiece;
using engine::community_models::granite5asr::Granite5ASRConfig;
using engine::community_models::granite5asr::ctc_greedy_pieces;
using engine::community_models::granite5asr::language_code_from_token;
using engine::community_models::granite5asr::samples_per_ctc_frame;
using engine::community_models::granite5asr::scale_sample_to_rate;
using engine::community_models::granite5asr::words_from_decoded_ctc_pieces;

constexpr int32_t kBlank = 0;
constexpr int64_t kRate = 16000;

CtcArgmaxFrame frame(int32_t id, float probability) {
    return CtcArgmaxFrame{id, probability};
}

DecodedCtcPiece decoded_from(const CtcPiece & piece, std::string text) {
    return DecodedCtcPiece{
        std::move(text),
        piece.start_frame,
        piece.end_frame,
        piece.log_prob_sum,
        piece.frames,
    };
}

void test_blank_gap_between_labels() {
    const std::vector<CtcArgmaxFrame> frames = {
        frame(1, 0.9f), frame(1, 0.9f), frame(1, 0.9f),
        frame(kBlank, 0.9f), frame(kBlank, 0.9f), frame(kBlank, 0.9f),
        frame(2, 0.8f),
    };
    const auto pieces = ctc_greedy_pieces(frames, kBlank);
    engine::test::require_eq(pieces.size(), size_t(2), "A and B are separate pieces");
    engine::test::require_eq(pieces[0].id, int32_t(1), "first piece is A");
    engine::test::require_eq(pieces[0].start_frame, int64_t(0), "A starts at frame 0");
    engine::test::require_eq(pieces[0].end_frame, int64_t(3), "A ends at the first blank");
    engine::test::require_eq(pieces[1].id, int32_t(2), "second piece is B");
    engine::test::require_eq(pieces[1].start_frame, int64_t(6), "B starts after the blanks");
    engine::test::require_eq(pieces[1].frames, int64_t(1), "trailing blanks are outside B");
}

void test_repeated_label_is_one_piece() {
    std::vector<CtcArgmaxFrame> frames(50, frame(4, 0.5f));
    const auto pieces = ctc_greedy_pieces(frames, kBlank);
    engine::test::require_eq(pieces.size(), size_t(1), "a repeated label is one piece");
    engine::test::require_eq(pieces[0].end_frame - pieces[0].start_frame, int64_t(50), "piece covers the run");
    engine::test::require_eq(pieces[0].frames, int64_t(50), "every frame of the run is scored");
}

void test_subword_gap_stays_inside_one_word() {
    std::vector<CtcArgmaxFrame> frames;
    frames.insert(frames.end(), 2, frame(7, 0.5f));
    frames.insert(frames.end(), 39, frame(kBlank, 0.99f));
    frames.insert(frames.end(), 2, frame(8, 0.5f));
    const auto pieces = ctc_greedy_pieces(frames, kBlank);
    engine::test::require_eq(pieces.size(), size_t(2), "hel and lo collapse separately");
    engine::test::require_eq(pieces[0].start_frame, int64_t(0), "hel starts at 0");
    engine::test::require_eq(pieces[0].end_frame, int64_t(2), "hel ends at the blank");
    engine::test::require_eq(pieces[1].start_frame, int64_t(41), "lo starts at frame 41");
    engine::test::require_eq(pieces[1].end_frame, int64_t(43), "lo ends at frame 43");
    engine::test::require_eq(pieces[0].frames + pieces[1].frames, int64_t(4), "blank frames are not scored");

    const auto words = words_from_decoded_ctc_pieces(
        {decoded_from(pieces[0], "hel"), decoded_from(pieces[1], "lo")},
        1,
        0,
        1000,
        kRate,
        kRate);
    engine::test::require_eq(words.size(), size_t(1), "no leading space keeps one word");
    engine::test::require(words[0].word == "hello", "subwords concatenate");
    engine::test::require_eq(words[0].span.start_sample, int64_t(0), "word starts at the first piece");
    engine::test::require_eq(words[0].span.end_sample, int64_t(43), "word ends at the last piece");
    engine::test::require_close(words[0].confidence, 0.5f, 1.0e-5f, "confidence ignores blank frames");
}

void test_punctuation_does_not_extend_the_word() {
    const DecodedCtcPiece effort{"effort", 0, 2, 2.0 * std::log(0.9), 2};
    const DecodedCtcPiece period{".", 2, 80, std::log(0.5), 78};
    const DecodedCtcPiece toward{" toward", 80, 82, std::log(0.8), 2};
    const auto words = words_from_decoded_ctc_pieces(
        {effort, period, toward}, 1, 0, 1000, kRate, kRate);
    engine::test::require_eq(words.size(), size_t(2), "punctuation stays on the previous word");
    engine::test::require(words[0].word == "effort.", "the period is kept on the word");
    engine::test::require_eq(words[0].span.start_sample, int64_t(0), "word still starts at the letters");
    engine::test::require_eq(words[0].span.end_sample, int64_t(2), "the period's pause is not part of the word");
    engine::test::require_close(words[0].confidence, 0.9f, 1.0e-5f, "punctuation frames are not scored");
    engine::test::require(words[1].word == "toward", "the next word is unchanged");
    engine::test::require_eq(words[1].span.start_sample, int64_t(80), "the next word keeps its own start");

    const DecodedCtcPiece spaced_period{" .", 2, 80, 0.0, 1};
    const auto spaced = words_from_decoded_ctc_pieces(
        {effort, spaced_period}, 1, 0, 1000, kRate, kRate);
    engine::test::require_eq(spaced.size(), size_t(1), "a spaced period is not its own word");
    engine::test::require(spaced[0].word == "effort.", "a spaced period is written onto the word");
    engine::test::require_eq(spaced[0].span.end_sample, int64_t(2), "a spaced period does not move the end");

    const DecodedCtcPiece mark{"。", 2, 40, 0.0, 1};
    const auto cjk = words_from_decoded_ctc_pieces({effort, mark}, 1, 0, 1000, kRate, kRate);
    engine::test::require(cjk[0].word == "effort。", "ideographic punctuation is kept");
    engine::test::require_eq(cjk[0].span.end_sample, int64_t(2), "ideographic punctuation does not move the end");
}

void test_leading_space_leaves_blanks_between_words() {
    std::vector<CtcArgmaxFrame> frames;
    frames.insert(frames.end(), 2, frame(7, 0.8f));
    frames.insert(frames.end(), 39, frame(kBlank, 0.8f));
    frames.insert(frames.end(), 2, frame(8, 0.8f));
    const auto pieces = ctc_greedy_pieces(frames, kBlank);
    const auto words = words_from_decoded_ctc_pieces(
        {decoded_from(pieces[0], "hel"), decoded_from(pieces[1], " lo")},
        1,
        0,
        1000,
        kRate,
        kRate);
    engine::test::require_eq(words.size(), size_t(2), "leading space flushes the previous word");
    engine::test::require(words[0].word == "hel", "first word text");
    engine::test::require(words[1].word == "lo", "leading spaces are stripped");
    engine::test::require_eq(words[0].span.end_sample, int64_t(2), "first word ends at its piece");
    engine::test::require_eq(words[1].span.start_sample, int64_t(41), "second word starts at its piece");
    engine::test::require(words[1].span.start_sample > words[0].span.end_sample, "blanks sit in neither word");
}

void test_non_positive_probability_zeros_confidence() {
    const auto zero = ctc_greedy_pieces({frame(9, 0.f)}, kBlank);
    const auto negative = ctc_greedy_pieces({frame(9, -0.2f)}, kBlank);
    engine::test::require_eq(zero[0].frames, int64_t(0), "non-positive probability contributes no frames");
    engine::test::require(!std::isfinite(zero[0].log_prob_sum), "non-positive probability yields -inf");
    engine::test::require_eq(negative[0].frames, int64_t(0), "negative probability contributes no frames");

    const auto words = words_from_decoded_ctc_pieces(
        {decoded_from(zero[0], "hel"), decoded_from(negative[0], "lo")},
        1280,
        0,
        100000,
        kRate,
        kRate);
    engine::test::require_eq(words.size(), size_t(1), "zeroed pieces still form the word");
    engine::test::require_close(words[0].confidence, 0.f, 0.f, "no contributing frames means confidence 0");

    const auto good = ctc_greedy_pieces({frame(9, 0.9f)}, kBlank);
    const auto mixed = words_from_decoded_ctc_pieces(
        {decoded_from(good[0], "hel"), decoded_from(zero[0], "lo")},
        1280,
        0,
        100000,
        kRate,
        kRate);
    engine::test::require_close(mixed[0].confidence, 0.f, 0.f, "a zeroed piece makes the word sum non-finite");
}

void test_probability_above_one_clamps() {
    const auto pieces = ctc_greedy_pieces({frame(3, 1.5f), frame(3, 2.f)}, kBlank);
    engine::test::require_eq(pieces[0].frames, int64_t(2), "clamped frames still count");
    engine::test::require_close(static_cast<float>(pieces[0].log_prob_sum), 0.f, 0.f, "probability above 1 scores as 1");
}

void test_end_sample_clamps_to_chunk_end() {
    const DecodedCtcPiece tail{"tail", 12, 13, 0.0, 1};
    const auto words = words_from_decoded_ctc_pieces({tail}, 1280, 16000, 32000, kRate, kRate);
    engine::test::require_eq(words[0].span.start_sample, int64_t(16000 + 12 * 1280), "offset start");
    engine::test::require_eq(words[0].span.end_sample, int64_t(32000), "end clamps to the chunk end");
}

void test_sample_stride_and_offset() {
    Granite5ASRConfig config;
    config.frontend.hop_length = 160;
    config.frontend.stack_factor = 2;
    config.encoder.subsample_layers = {0, 1};
    const int64_t stride = samples_per_ctc_frame(config);
    engine::test::require_eq(stride, int64_t(1280), "16 kHz frame is 160 * 2 * 4 samples");

    const DecodedCtcPiece piece{"hi", 0, 1, 0.0, 1};
    const auto origin = words_from_decoded_ctc_pieces({piece}, stride, 0, 1000000, kRate, kRate);
    engine::test::require_eq(origin[0].span.start_sample, int64_t(0), "frame 0 is sample 0");
    engine::test::require_eq(origin[0].span.end_sample, int64_t(1280), "frame 1 is sample 1280");

    const auto shifted = words_from_decoded_ctc_pieces({piece}, stride, 16000, 1000000, kRate, kRate);
    engine::test::require_eq(shifted[0].span.start_sample, int64_t(16000), "offset is added to the start");
    engine::test::require_eq(shifted[0].span.end_sample, int64_t(17280), "offset is added to the end");
}

void test_frames_convert_to_input_rate_samples() {
    const DecodedCtcPiece piece{"hi", 1, 2, 0.0, 1};
    const auto high = words_from_decoded_ctc_pieces({piece}, 1280, 48000, 1000000, kRate, 48000);
    engine::test::require_eq(high[0].span.start_sample, int64_t(48000 + 3840), "48 kHz frame 1 is sample 3840");
    engine::test::require_eq(high[0].span.end_sample, int64_t(48000 + 7680), "48 kHz frame 2 is sample 7680");

    const auto low = words_from_decoded_ctc_pieces({piece}, 1280, 8000, 1000000, kRate, 8000);
    engine::test::require_eq(low[0].span.start_sample, int64_t(8000 + 640), "8 kHz frame 1 is sample 640");
    engine::test::require_eq(low[0].span.end_sample, int64_t(8000 + 1280), "8 kHz frame 2 is sample 1280");

    const DecodedCtcPiece tail{"tents", 43, 44, 0.0, 1};
    const auto clamped = words_from_decoded_ctc_pieces({tail}, 1280, 0, 28040, kRate, 8000);
    engine::test::require_eq(clamped[0].span.start_sample, int64_t(43 * 640), "8 kHz start stays on the input clock");
    engine::test::require_eq(clamped[0].span.end_sample, int64_t(28040), "8 kHz end clamps to the input buffer");
}

void test_streaming_samples_use_the_input_rate() {
    engine::test::require_eq(scale_sample_to_rate(16000, 16000, 16000), int64_t(16000), "16 kHz stays on its own clock");
    engine::test::require_eq(scale_sample_to_rate(16000, 16000, 48000), int64_t(48000), "one 16 kHz second is 48000 input samples");
    engine::test::require_eq(scale_sample_to_rate(16000, 16000, 8000), int64_t(8000), "one 16 kHz second is 8000 input samples");
    engine::test::require_eq(scale_sample_to_rate(0, 16000, 48000), int64_t(0), "the origin stays at 0");
}

void test_language_tokens_are_recognized() {
    engine::test::require(language_code_from_token("<|en|>") == std::string("en"), "en tag");
    engine::test::require(language_code_from_token("<|fr|>") == std::string("fr"), "fr tag");
    engine::test::require(language_code_from_token("<|ja|>") == std::string("ja"), "ja tag");
    engine::test::require(language_code_from_token("<|zh|>").empty(), "unsupported tags are not a language");
    engine::test::require(language_code_from_token("hello").empty(), "ordinary text is not a language");
    engine::test::require(language_code_from_token("<|timestamp|>").empty(), "the timestamp control is not a language");
}

void test_end_sample_clamps_and_keeps_one_sample() {
    const DecodedCtcPiece inside{"hi", 0, 10, 0.0, 1};
    const auto clamped = words_from_decoded_ctc_pieces({inside}, 1280, 0, 1000, kRate, kRate);
    engine::test::require_eq(clamped[0].span.start_sample, int64_t(0), "start stays inside the buffer");
    engine::test::require_eq(clamped[0].span.end_sample, int64_t(1000), "end clamps to the buffer length");

    const DecodedCtcPiece at_end{"hi", 1000, 1008, 0.0, 1};
    const auto bumped = words_from_decoded_ctc_pieces({at_end}, 1, 0, 1000, kRate, kRate);
    engine::test::require_eq(bumped[0].span.start_sample, int64_t(1000), "start is the piece start");
    engine::test::require_eq(
        bumped[0].span.end_sample,
        bumped[0].span.start_sample + 1,
        "an empty clamped span is one sample");
}

}  // namespace

int main() {
    try {
        test_blank_gap_between_labels();
        test_repeated_label_is_one_piece();
        test_subword_gap_stays_inside_one_word();
        test_punctuation_does_not_extend_the_word();
        test_leading_space_leaves_blanks_between_words();
        test_non_positive_probability_zeros_confidence();
        test_sample_stride_and_offset();
        test_frames_convert_to_input_rate_samples();
        test_end_sample_clamps_and_keeps_one_sample();
        test_probability_above_one_clamps();
        test_end_sample_clamps_to_chunk_end();
        test_streaming_samples_use_the_input_rate();
        test_language_tokens_are_recognized();
        std::cout << "PASS: granite5asr CTC word timestamps\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
