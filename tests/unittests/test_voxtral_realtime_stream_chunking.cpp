#include "engine/models/voxtral_realtime/frontend.h"

#include "test_assert.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::models::voxtral_realtime::VoxtralRealtimeAssets;
using engine::models::voxtral_realtime::VoxtralRealtimeFrontend;
using engine::models::voxtral_realtime::VoxtralRealtimeFrontendStreamState;
using engine::test::require;
using engine::test::require_eq;

// The frontend only needs the config to size chunks and build its mel filterbank, so the whole
// suite runs without a model, weights or a backend.
VoxtralRealtimeFrontend make_frontend() {
    auto assets = std::make_shared<VoxtralRealtimeAssets>();
    return VoxtralRealtimeFrontend(std::move(assets));
}

engine::runtime::AudioBuffer mono_audio(int64_t samples) {
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.channels = 1;
    audio.samples.assign(static_cast<size_t>(samples), 0.25F);
    return audio;
}

bool throws(const std::function<void()> & call) {
    try {
        call();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// A steady chunk holds `steady_tokens` audio tokens of 8 * 160 = 1280 samples (80 ms), plus one
// win_length tail shared by the whole batch. Batching must not change what a single token costs.
void test_steady_chunk_samples_scale_per_token() {
    const auto frontend = make_frontend();
    require_eq(frontend.steady_stream_chunk_samples(1), 1680, "steady chunk samples at 1 token");
    for (const int64_t tokens : {1, 2, 4, 7}) {
        require_eq(
            frontend.steady_stream_chunk_samples(tokens),
            tokens * 1280 + 400,
            "steady chunk samples at " + std::to_string(tokens) + " tokens");
    }
}

// The advance is what the session consumes per chunk, and it is what
// `streaming_steps_processed_ += stream_batch_tokens_` is accounted against: a batched chunk must
// advance by exactly as much as the same number of single-token chunks would.
void test_steady_advance_is_linear_in_tokens() {
    const auto frontend = make_frontend();
    const int64_t single = frontend.steady_stream_chunk_advance_samples(1);
    require_eq(single, 1280, "steady advance at 1 token");
    for (const int64_t tokens : {1, 2, 4, 7}) {
        require_eq(
            frontend.steady_stream_chunk_advance_samples(tokens),
            tokens * single,
            "steady advance at " + std::to_string(tokens) + " tokens");
    }
}

// Every caller that does not batch must be bit-for-bit unaffected by the new parameter.
void test_defaults_match_single_token() {
    const auto frontend = make_frontend();
    require_eq(frontend.steady_stream_chunk_samples(), 1680, "default steady chunk samples");
    require_eq(frontend.steady_stream_chunk_advance_samples(), 1280, "default steady advance");
    // The first chunk is never batched: it primes the decoder with the delay-token lookahead.
    require_eq(frontend.first_stream_chunk_samples(), 9000, "first chunk samples");
    require_eq(frontend.first_stream_chunk_advance_samples(), 8760, "first chunk advance");
}

void test_non_positive_token_counts_are_rejected() {
    const auto frontend = make_frontend();
    for (const int64_t tokens : {static_cast<int64_t>(0), static_cast<int64_t>(-1)}) {
        require(
            throws([&] { (void) frontend.steady_stream_chunk_samples(tokens); }),
            "steady_stream_chunk_samples must reject " + std::to_string(tokens));
        require(
            throws([&] { (void) frontend.steady_stream_chunk_advance_samples(tokens); }),
            "steady_stream_chunk_advance_samples must reject " + std::to_string(tokens));
        require(
            throws([&] {
                VoxtralRealtimeFrontendStreamState state;
                (void) frontend.extract_stream_chunk(mono_audio(1680), false, state, tokens);
            }),
            "extract_stream_chunk must reject " + std::to_string(tokens));
    }
}

// One STFT pass over an N-token chunk must yield 8N feature frames, which the encoder's conv
// stack turns into 4N encoder steps and, after downsample_factor 4, exactly N audio tokens. The
// first steady chunk and the ones after it take different STFT routes -- the first computes the
// whole magnitude, later ones reuse the previous chunk's cached frame as their prefix -- so both
// are exercised here and must agree.
void test_batched_chunk_yields_one_token_worth_of_frames_each() {
    const auto frontend = make_frontend();
    for (const int64_t tokens : {1, 2, 4}) {
        VoxtralRealtimeFrontendStreamState state;
        const auto audio = mono_audio(frontend.steady_stream_chunk_samples(tokens));
        const std::string label = " at " + std::to_string(tokens) + " tokens";

        const auto uncached = frontend.extract_stream_chunk(audio, false, state, tokens);
        require_eq(uncached.frames, tokens * 8, "uncached steady frames" + label);
        require_eq(uncached.mel_bins, 128, "uncached steady mel bins" + label);
        require(state.cached_frame_ready, "steady chunk must cache its STFT tail" + label);

        const auto cached = frontend.extract_stream_chunk(audio, false, state, tokens);
        require_eq(cached.frames, uncached.frames, "cached steady frames" + label);
        require_eq(cached.mel_bins, uncached.mel_bins, "cached steady mel bins" + label);
        require_eq(
            static_cast<int64_t>(cached.values.size()),
            cached.frames * cached.mel_bins,
            "cached steady feature size" + label);
    }
}

}  // namespace

int main() {
    try {
        test_steady_chunk_samples_scale_per_token();
        test_steady_advance_is_linear_in_tokens();
        test_defaults_match_single_token();
        test_non_positive_token_counts_are_rejected();
        test_batched_chunk_yields_one_token_worth_of_frames_each();
        std::cout << "voxtral_realtime_stream_chunking_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "voxtral_realtime_stream_chunking_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
