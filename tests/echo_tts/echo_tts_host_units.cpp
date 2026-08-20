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

#include "engine/community_models/echo_tts/config.h"
#include "engine/community_models/echo_tts/latent_post.h"
#include "engine/community_models/echo_tts/tokenizer.h"

#include "../unittests/test_assert.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

using namespace engine::models::echo_tts;

constexpr int64_t kMaxLength = 768;  // upstream's hard cap

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
    // A BOS 0 followed by the raw UTF-8 bytes of the normalised string.
    const auto hello = encode("Hello world.");
    require_eq(static_cast<int64_t>(hello.size()), static_cast<int64_t>(18), "hello token count");
    const std::vector<int32_t> hello_prefix{0, 91, 83, 49, 93, 32, 72, 101, 108, 108, 111, 32};
    for (size_t i = 0; i < hello_prefix.size(); ++i) {
        require_eq(hello[i], hello_prefix[i], "hello token " + std::to_string(i));
    }

    require_eq(static_cast<int64_t>(encode("[S1] Already tagged.").size()),
               static_cast<int64_t>(21), "pre-tagged token count");

    const auto paren = encode("(parenthesised start)");
    require_eq(static_cast<int64_t>(paren.size()), static_cast<int64_t>(22), "paren token count");
    require_eq(paren[1], static_cast<int32_t>('('), "paren text is not re-tagged");

    require_eq(static_cast<int64_t>(
                   encode("Time: 3; place \xE2\x80\x94 here\xE2\x80\xA6 he said "
                          "\xE2\x80\x9Cgo\xE2\x80\x9D and it\xE2\x80\x99s fine.\nNext line.")
                       .size()),
               static_cast<int64_t>(72), "punctuation-heavy token count");
}

void test_tokeniser_truncates_at_max_length() {
    const std::string long_text(4000, 'a');
    const auto tokens = tokenize_echo_text(long_text, kMaxLength);
    require_eq(static_cast<int64_t>(tokens.input_ids.size()), kMaxLength,
               "truncated length includes the BOS");
    require(tokens.truncated, "over-long input is reported as truncated");
    require_eq(tokens.input_ids.front(), static_cast<int32_t>(0), "BOS survives truncation");

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

// --- PCA -------------------------------------------------------------------

// A square identity basis makes project/unproject an exact bijection, so any
// round-trip error is the implementation's own rather than the subspace's.
EchoPcaState identity_pca(int64_t dim, float scale) {
    EchoPcaState pca;
    pca.components.assign(static_cast<size_t>(dim * dim), 0.0F);
    for (int64_t i = 0; i < dim; ++i) {
        pca.components[static_cast<size_t>(i * dim + i)] = 1.0F;
    }
    pca.mean.assign(static_cast<size_t>(dim), 0.0F);
    for (int64_t i = 0; i < dim; ++i) {
        pca.mean[static_cast<size_t>(i)] = 0.25F * static_cast<float>(i);
    }
    pca.latent_scale = scale;
    return pca;
}

void test_pca_round_trip_is_lossless_on_an_orthonormal_basis() {
    constexpr int64_t kDim = 16;
    constexpr int64_t kFrames = 5;

    EchoTtsConfig config;
    config.latent_size = kDim;
    config.ae_latent_dim = kDim;

    // The real checkpoint ships latent_scale = 1/18; use it rather than 1.0 so
    // a dropped scale on either leg shows up.
    const auto pca = identity_pca(kDim, 0.0555555559694767F);

    std::vector<float> z_q(static_cast<size_t>(kFrames * kDim));
    for (int64_t f = 0; f < kFrames; ++f) {
        for (int64_t k = 0; k < kDim; ++k) {
            z_q[static_cast<size_t>(f * kDim + k)] =
                static_cast<float>(f) - 2.0F + 0.5F * static_cast<float>(k);
        }
    }

    const auto latents = pca_project(pca, config, z_q, kFrames);
    require_eq(static_cast<int64_t>(latents.size()), kFrames * kDim, "projected size");

    const auto recovered = pca_unproject(pca, config, latents, kFrames);
    require_eq(static_cast<int64_t>(recovered.size()), kFrames * kDim, "unprojected size");

    for (size_t i = 0; i < z_q.size(); ++i) {
        require_close(recovered[i], z_q[i], 1e-3F, "pca round trip element " + std::to_string(i));
    }
}

void test_pca_applies_the_latent_scale() {
    constexpr int64_t kDim = 4;
    EchoTtsConfig config;
    config.latent_size = kDim;
    config.ae_latent_dim = kDim;

    EchoPcaState pca = identity_pca(kDim, 0.5F);
    pca.mean.assign(static_cast<size_t>(kDim), 0.0F);  // isolate the scale

    const std::vector<float> z_q{2.0F, 4.0F, 6.0F, 8.0F};
    const auto latents = pca_project(pca, config, z_q, 1);
    for (size_t i = 0; i < z_q.size(); ++i) {
        require_close(latents[i], z_q[i] * 0.5F, 1e-6F,
                      "projection scales by latent_scale, element " + std::to_string(i));
    }
}

void test_pca_rejects_mis_shaped_buffers() {
    EchoTtsConfig config;
    config.latent_size = 4;
    config.ae_latent_dim = 4;
    const auto pca = identity_pca(4, 1.0F);

    bool threw = false;
    try {
        pca_project(pca, config, std::vector<float>(7, 0.0F), 2);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "a mis-shaped z_q buffer is rejected rather than read out of bounds");
}

// --- flattening point ------------------------------------------------------

std::vector<float> alternating(int64_t frames, int64_t latent_size, int64_t active_frames) {
    std::vector<float> out(static_cast<size_t>(frames * latent_size), 0.0F);
    for (int64_t f = 0; f < active_frames; ++f) {
        for (int64_t c = 0; c < latent_size; ++c) {
            out[static_cast<size_t>(f * latent_size + c)] = ((f + c) % 2 == 0) ? 1.0F : -1.0F;
        }
    }
    return out;
}

void test_flattening_point_matches_reference() {
    constexpr int64_t kFrames = 60;
    constexpr int64_t kLatent = 4;

    // Active for 30 frames, then silent. Reference returns 30.
    require_eq(find_flattening_point(alternating(kFrames, kLatent, 30), kFrames, kLatent),
               static_cast<int64_t>(30), "crop lands where the signal goes flat");

    // Never flattens: the reference falls through to len(data).
    require_eq(find_flattening_point(alternating(kFrames, kLatent, kFrames), kFrames, kLatent),
               kFrames, "a latent that never flattens keeps every frame");

    // Flat from the first frame.
    require_eq(find_flattening_point(alternating(kFrames, kLatent, 0), kFrames, kLatent),
               static_cast<int64_t>(0), "an all-silent latent crops to nothing");
}

}  // namespace

int main() {
    try {
        test_normalisation_matches_reference();
        test_tokenisation_matches_reference();
        test_tokeniser_truncates_at_max_length();
        test_mask_marks_real_tokens();
        test_pca_round_trip_is_lossless_on_an_orthonormal_basis();
        test_pca_applies_the_latent_scale();
        test_pca_rejects_mis_shaped_buffers();
        test_flattening_point_matches_reference();
        std::cout << "echo_tts_host_units: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "echo_tts_host_units: " << ex.what() << "\n";
        return 1;
    }
}
