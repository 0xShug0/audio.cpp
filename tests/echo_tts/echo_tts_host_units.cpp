// Host-side unit tests for the Echo-TTS port.
//
// These cover the parts of the pipeline that run on the CPU and need neither a
// GPU nor the 5.5 GB checkpoint: the byte tokenizer and its WhisperD
// normalisation, the PCA forward/inverse pair, and the flattening-point crop
// that sets the output duration.
//
// Every expected value here was produced by executing the reference
// implementation at tts-bench/venvs/echo/src/inference.py -- `tokenizer_encode`
// for the token streams and `find_flattening_point` for the crop indices -- not
// by reasoning about what it ought to return.
//
// Two review findings shaped the fixtures, and both are worth stating so they
// are not "simplified" back out:
//
//   * The PCA basis here is RECTANGULAR and non-symmetric on purpose. An
//     identity basis is its own transpose, so a round trip over one cannot
//     distinguish `components[c * features + k]` from the transposed indexing,
//     and a mean or scale dropped on both legs cancels. Both projection and
//     inversion are therefore pinned against independently computed values
//     rather than against each other.
//
//   * The flattening fixtures include a tail that is quiet but NOT zero and a
//     tail that is flat but too loud. Only all-zero fixtures would let an
//     implementation that merely searches for a zero window pass without ever
//     evaluating the standard-deviation and mean thresholds.

#include "engine/community_models/echo_tts/config.h"
#include "engine/community_models/echo_tts/latent_post.h"
#include "engine/community_models/echo_tts/tokenizer.h"

#include "../unittests/test_assert.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

using namespace engine::models::echo_tts;

constexpr int64_t kMaxLength = 768;  // upstream's hard cap

// The shared require_close compares `fabs(a - b) > tolerance`, which is FALSE
// for a NaN difference, so a NaN silently passes every float assertion. Reject
// non-finite values explicitly before deferring to it.
void require_close(float actual, float expected, float tolerance, const std::string & label) {
    if (!std::isfinite(actual)) {
        throw std::runtime_error(label + " is not finite");
    }
    engine::test::require_close(actual, expected, tolerance, label);
}

void require_ids(const std::vector<int32_t> & actual,
                 const std::vector<int32_t> & expected,
                 const std::string & label) {
    require_eq(static_cast<int64_t>(actual.size()), static_cast<int64_t>(expected.size()),
               label + " length");
    for (size_t i = 0; i < expected.size(); ++i) {
        require_eq(actual[i], expected[i], label + " id " + std::to_string(i));
    }
}

std::vector<int32_t> encode(const std::string & text) {
    return tokenize_echo_text(text, kMaxLength).input_ids;
}

// --- tokenizer -------------------------------------------------------------

void test_normalisation_matches_reference() {
    require_eq(normalize_echo_text("Hello world."), std::string("[S1] Hello world."),
               "bare text gets the [S1] tag");

    require_eq(normalize_echo_text("[S1] Already tagged."), std::string("[S1] Already tagged."),
               "an existing tag is not doubled");

    require_eq(normalize_echo_text("(parenthesised start)"), std::string("(parenthesised start)"),
               "a leading paren suppresses the tag");

    // Upstream's tag check is a bare substring search for "S1"/"S2" anywhere in
    // the string, not a prefix check, so prose containing those two characters
    // loses the speaker tag. Faithful, and pinned so it stays faithful.
    require_eq(normalize_echo_text("This is S1 talking"), std::string("This is S1 talking"),
               "a bare S1 anywhere suppresses the tag, as upstream does");

    // Colons and semicolons become commas, an em dash becomes ", ", an ellipsis
    // becomes "...", a right single quote becomes an apostrophe, and a newline
    // becomes a space.
    //
    // The asymmetric quote handling is deliberate and is reproduced from
    // upstream: the RIGHT double quote is rewritten to ASCII, the LEFT one is
    // not. Upstream applies the right-quote replacement twice, which is a
    // no-op, and never touches U+201C. If someone "fixes" that asymmetry the
    // token stream silently stops matching the reference, so it is pinned here.
    require_eq(
        normalize_echo_text("Time: 3; place \xE2\x80\x94 here\xE2\x80\xA6 he said "
                            "\xE2\x80\x9Cgo\xE2\x80\x9D and it\xE2\x80\x99s fine.\nNext line."),
        std::string("[S1] Time, 3, place ,  here... he said \xE2\x80\x9Cgo\" and it's fine. Next line."),
        "punctuation rewrites match the reference");
}

void test_tokenisation_matches_reference() {
    // Full id vectors, not lengths. A length-preserving rewrite -- signed char
    // sign-extension on the multibyte U+201C below being the obvious one --
    // passes a count check and fails these.
    require_ids(encode("Hello world."),
                {0, 91, 83, 49, 93, 32, 72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 46},
                "hello");

    require_ids(encode("[S1] Already tagged."),
                {0, 91, 83, 49, 93, 32, 65, 108, 114, 101, 97, 100, 121, 32, 116, 97, 103, 103,
                 101, 100, 46},
                "pre-tagged");

    require_ids(encode("(parenthesised start)"),
                {0, 40, 112, 97, 114, 101, 110, 116, 104, 101, 115, 105, 115, 101, 100, 32, 115,
                 116, 97, 114, 116, 41},
                "paren");

    require_ids(encode("This is S1 talking"),
                {0, 84, 104, 105, 115, 32, 105, 115, 32, 83, 49, 32, 116, 97, 108, 107, 105, 110,
                 103},
                "bare S1");

    // 226/128/156 is the untouched left double quote: three unsigned bytes.
    require_ids(encode("Time: 3; place \xE2\x80\x94 here\xE2\x80\xA6 he said "
                       "\xE2\x80\x9Cgo\xE2\x80\x9D and it\xE2\x80\x99s fine.\nNext line."),
                {0, 91, 83, 49, 93, 32, 84, 105, 109, 101, 44, 32, 51, 44, 32, 112, 108, 97, 99,
                 101, 32, 44, 32, 32, 104, 101, 114, 101, 46, 46, 46, 32, 104, 101, 32, 115, 97,
                 105, 100, 32, 226, 128, 156, 103, 111, 34, 32, 97, 110, 100, 32, 105, 116, 39,
                 115, 32, 102, 105, 110, 101, 46, 32, 78, 101, 120, 116, 32, 108, 105, 110, 101,
                 46},
                "punctuation-heavy");
}

void test_tokeniser_truncates_at_max_length() {
    const std::string long_text(4000, 'a');
    const auto tokens = tokenize_echo_text(long_text, kMaxLength);
    require_eq(static_cast<int64_t>(tokens.input_ids.size()), kMaxLength,
               "truncated length includes the BOS");
    require(tokens.truncated, "over-long input is reported as truncated");
    require_eq(tokens.input_ids.front(), static_cast<int32_t>(0), "BOS survives truncation");

    // The surviving ids are the *leading* bytes, not zero padding: everything
    // after the BOS is the tag "[S1] " and then 'a'.
    const std::vector<int32_t> head{0, 91, 83, 49, 93, 32, 97, 97};
    for (size_t i = 0; i < head.size(); ++i) {
        require_eq(tokens.input_ids[i], head[i], "truncated head id " + std::to_string(i));
    }
    require_eq(tokens.input_ids.back(), static_cast<int32_t>('a'), "truncation keeps a real byte");
    for (const float m : tokens.mask) {
        require_close(m, 1.0F, 1e-6F, "a fully truncated sequence has no padding");
    }

    const auto shortish = tokenize_echo_text("Hello world.", kMaxLength);
    require(!shortish.truncated, "short input is not reported as truncated");
}

void test_mask_marks_real_tokens() {
    const auto tokens = tokenize_echo_text("Hello world.", kMaxLength);
    require_eq(tokens.mask.size(), tokens.input_ids.size(), "mask and ids are the same length");
    for (size_t i = 0; i < tokens.mask.size(); ++i) {
        require_close(tokens.mask[i], 1.0F, 1e-6F, "unpadded mask entry " + std::to_string(i));
    }
}

void test_pad_to_max_zeroes_the_tail() {
    const auto padded = tokenize_echo_text("Hello world.", kMaxLength, true, true);
    require_eq(static_cast<int64_t>(padded.input_ids.size()), kMaxLength, "padded id length");
    require_eq(static_cast<int64_t>(padded.mask.size()), kMaxLength, "padded mask length");

    constexpr int64_t kReal = 18;  // "[S1] Hello world." plus the BOS
    for (int64_t i = 0; i < kMaxLength; ++i) {
        const auto idx = static_cast<size_t>(i);
        if (i < kReal) {
            require_close(padded.mask[idx], 1.0F, 1e-6F, "real mask " + std::to_string(i));
        } else {
            require_close(padded.mask[idx], 0.0F, 1e-6F, "pad mask " + std::to_string(i));
            require_eq(padded.input_ids[idx], static_cast<int32_t>(0),
                       "pad id " + std::to_string(i));
        }
    }
}

// --- PCA -------------------------------------------------------------------

// A rectangular, non-symmetric orthonormal basis: 2 components over 4 features.
// Rectangular so a transposed read is a different computation rather than the
// same one; orthonormal so the projected coefficients are exact in binary
// floating point and can be written down by hand.
//
//   b0 = [ 0.5,  0.5,  0.5,  0.5]
//   b1 = [ 0.5, -0.5,  0.5, -0.5]
//   mean = [1, 2, 3, 4], latent_scale = 0.5
//
// Because 2 components cannot span 4 features, the round trip is deliberately
// LOSSY -- it returns the projection of the input onto span{b0, b1}, which is
// what the real (80, 1024) basis does too. An identity fixture hides that.
constexpr int64_t kFeatures = 4;
constexpr int64_t kComponents = 2;

EchoPcaState rectangular_pca() {
    EchoPcaState pca;
    pca.components = {0.5F, 0.5F, 0.5F, 0.5F,
                      0.5F, -0.5F, 0.5F, -0.5F};
    pca.mean = {1.0F, 2.0F, 3.0F, 4.0F};
    pca.latent_scale = 0.5F;
    return pca;
}

EchoTtsConfig rectangular_config() {
    EchoTtsConfig config;
    config.latent_size = kComponents;
    config.ae_latent_dim = kFeatures;
    return config;
}

void test_pca_projection_matches_hand_computed_values() {
    const auto pca = rectangular_pca();
    const auto config = rectangular_config();

    // Frame 0: z_q - mean = [1, 2, 3, 4]; dot(b0) = 5, dot(b1) = -1; scaled by 0.5.
    // Frame 1: z_q - mean = [-1, -2, -3, -4]; dot(b0) = -5, dot(b1) = 1.
    const std::vector<float> z_q{2.0F, 4.0F, 6.0F, 8.0F,
                                 0.0F, 0.0F, 0.0F, 0.0F};
    const std::vector<float> expected{2.5F, -0.5F,
                                      -2.5F, 0.5F};

    const auto latents = pca_project(pca, config, z_q, 2);
    require_eq(static_cast<int64_t>(latents.size()), static_cast<int64_t>(expected.size()),
               "projected size");
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(latents[i], expected[i], 1e-6F, "projection element " + std::to_string(i));
    }
}

void test_pca_inversion_matches_hand_computed_values() {
    const auto pca = rectangular_pca();
    const auto config = rectangular_config();

    // Pinned independently of the forward pass so a mean or scale dropped on
    // both legs cannot cancel:
    //   frame 0: coeffs [2.5, -0.5] / 0.5 = [5, -1]
    //            mean + 5*b0 - 1*b1 = [1,2,3,4] + [2.5]*4 + [-0.5, 0.5, -0.5, 0.5]
    //                               = [3, 5, 5, 7]
    //   frame 1: coeffs [-5, 1] -> [1,2,3,4] + [-2.5]*4 + [0.5,-0.5,0.5,-0.5]
    //                            = [-1, -1, 1, 1]
    const std::vector<float> latents{2.5F, -0.5F,
                                     -2.5F, 0.5F};
    const std::vector<float> expected{3.0F, 5.0F, 5.0F, 7.0F,
                                      -1.0F, -1.0F, 1.0F, 1.0F};

    const auto recovered = pca_unproject(pca, config, latents, 2);
    require_eq(static_cast<int64_t>(recovered.size()), static_cast<int64_t>(expected.size()),
               "unprojected size");
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(recovered[i], expected[i], 1e-6F, "inversion element " + std::to_string(i));
    }
}

void test_pca_round_trip_recovers_an_in_subspace_vector() {
    const auto pca = rectangular_pca();
    const auto config = rectangular_config();

    // Exactly on span{b0, b1} once the mean is removed, so the lossy projection
    // is an identity here and the round trip must be exact -- 1e-6, not 1e-3.
    const std::vector<float> z_q{1.0F + 2.0F, 2.0F + 1.0F, 3.0F + 2.0F, 4.0F + 1.0F};

    const auto latents = pca_project(pca, config, z_q, 1);
    const auto recovered = pca_unproject(pca, config, latents, 1);
    for (size_t i = 0; i < z_q.size(); ++i) {
        require_close(recovered[i], z_q[i], 1e-6F, "round trip element " + std::to_string(i));
    }
}

void test_pca_rejects_mis_shaped_buffers() {
    const auto pca = rectangular_pca();
    const auto config = rectangular_config();

    bool projected_threw = false;
    try {
        pca_project(pca, config, std::vector<float>(7, 0.0F), 2);
    } catch (const std::exception &) {
        projected_threw = true;
    }
    require(projected_threw, "a mis-shaped z_q buffer is rejected rather than read out of bounds");

    bool inverted_threw = false;
    try {
        pca_unproject(pca, config, std::vector<float>(3, 0.0F), 2);
    } catch (const std::exception &) {
        inverted_threw = true;
    }
    require(inverted_threw, "a mis-shaped latent buffer is rejected");

    bool zero_scale_threw = false;
    try {
        EchoPcaState broken = pca;
        broken.latent_scale = 0.0F;
        pca_unproject(broken, config, std::vector<float>(2, 0.0F), 1);
    } catch (const std::exception &) {
        zero_scale_threw = true;
    }
    require(zero_scale_threw, "a zero latent_scale is rejected rather than dividing by zero");
}

// --- flattening point ------------------------------------------------------

constexpr int64_t kCropFrames = 60;
constexpr int64_t kCropLatent = 4;

// Loud alternating +-1 for `active_frames`, then a constant `tail` value.
std::vector<float> loud_then_tail(int64_t active_frames, float tail) {
    std::vector<float> out(static_cast<size_t>(kCropFrames * kCropLatent), tail);
    for (int64_t f = 0; f < active_frames; ++f) {
        for (int64_t c = 0; c < kCropLatent; ++c) {
            out[static_cast<size_t>(f * kCropLatent + c)] = ((f + c) % 2 == 0) ? 1.0F : -1.0F;
        }
    }
    return out;
}

int64_t crop(const std::vector<float> & latents) {
    return find_flattening_point(latents, kCropFrames, kCropLatent);
}

void test_flattening_point_matches_reference() {
    // Active for 30 frames, then silent. Reference returns 30.
    require_eq(crop(loud_then_tail(30, 0.0F)), static_cast<int64_t>(30),
               "crop lands where the signal goes flat");

    // Never flattens: the reference falls through to len(data).
    require_eq(crop(loud_then_tail(kCropFrames, 0.0F)), kCropFrames,
               "a latent that never flattens keeps every frame");

    // Flat from the first frame.
    require_eq(crop(loud_then_tail(0, 0.0F)), static_cast<int64_t>(0),
               "an all-silent latent crops to nothing");

    // Quiet but NOT zero: std is 0 and |mean - 0| = 0.02 < 0.1, so this must
    // still crop at 30. An implementation that looks for an all-zero window
    // rather than evaluating both thresholds fails here.
    require_eq(crop(loud_then_tail(30, 0.02F)), static_cast<int64_t>(30),
               "a quiet non-zero tail still counts as flat");

    // Flat but too loud: std is 0, yet |mean - 0| = 0.5 exceeds 0.1, so no
    // window qualifies and the crop falls through to every frame. This is the
    // case that pins the mean threshold rather than just the std threshold.
    require_eq(crop(loud_then_tail(30, 0.5F)), kCropFrames,
               "a flat but loud tail is not a flattening point");
}

}  // namespace

int main() {
    try {
        test_normalisation_matches_reference();
        test_tokenisation_matches_reference();
        test_tokeniser_truncates_at_max_length();
        test_mask_marks_real_tokens();
        test_pad_to_max_zeroes_the_tail();
        test_pca_projection_matches_hand_computed_values();
        test_pca_inversion_matches_hand_computed_values();
        test_pca_round_trip_recovers_an_in_subspace_vector();
        test_pca_rejects_mis_shaped_buffers();
        test_flattening_point_matches_reference();
        std::cout << "echo_tts_host_units: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "echo_tts_host_units: " << ex.what() << "\n";
        return 1;
    }
}
