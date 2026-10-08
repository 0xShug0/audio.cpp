/*
 * Rules behind the MOSS-TTS-Nano reference-voice cache.
 *
 * The cache exists because encoding a reference runs the audio tokenizer encoder, and
 * a session that keeps one voice should pay that once. These checks are model-free on
 * purpose: they run anywhere, including build configurations with no model downloaded.
 */

#include "engine/models/moss/moss_tts_nano/session.h"

#include <cstdio>
#include <optional>
#include <vector>

namespace {

namespace nano = engine::models::moss_tts_nano;
namespace runtime = engine::runtime;

int g_failures = 0;

void check(bool condition, const char * what) {
    std::printf("  [%s] %s\n", condition ? "ok" : "FAIL", what);
    if (!condition) {
        ++g_failures;
    }
}

runtime::AudioBuffer make_audio(float bias) {
    runtime::AudioBuffer buffer;
    buffer.sample_rate = 48000;
    buffer.channels = 2;
    buffer.samples.resize(64);
    for (size_t i = 0; i < buffer.samples.size(); ++i) {
        buffer.samples[i] = bias + static_cast<float>(i) * 1e-4F;
    }
    return buffer;
}

nano::MossTTSNanoAudioCodes make_codes(int64_t codebooks) {
    nano::MossTTSNanoAudioCodes codes;
    codes.frames = 4;
    codes.codebooks = codebooks;
    codes.token_ids.assign(static_cast<size_t>(codes.frames * codebooks), 0);
    return codes;
}

}  // namespace

int main() {
    const runtime::AudioBuffer voice_a = make_audio(0.10F);
    const runtime::AudioBuffer voice_b = make_audio(0.20F);

    std::optional<runtime::AudioBuffer> cached_audio;
    std::optional<nano::MossTTSNanoAudioCodes> cached_codes;

    std::printf("first prepare records the audio, encode happens on first use\n");
    nano::invalidate_reference_codes_if_changed(cached_audio, cached_codes, voice_a);
    check(cached_audio.has_value(), "audio recorded");
    check(!cached_codes.has_value(), "prepare does not encode");
    check(!nano::reference_codes_reusable(cached_audio, cached_codes, voice_a, 16),
          "without codes nothing is reusable");

    std::printf("after the first encode the same voice is reused\n");
    cached_codes = make_codes(16);
    check(nano::reference_codes_reusable(cached_audio, cached_codes, voice_a, 16),
          "same audio and codebook count -> reusable");
    check(!nano::reference_codes_reusable(cached_audio, cached_codes, voice_a, 8),
          "different codebook count -> not reusable");
    check(!nano::reference_codes_reusable(cached_audio, cached_codes, voice_b, 16),
          "different audio -> not reusable");

    std::printf("preparing the same voice again keeps the codes\n");
    nano::invalidate_reference_codes_if_changed(cached_audio, cached_codes, voice_a);
    check(cached_codes.has_value(), "codes kept");
    check(nano::reference_codes_reusable(cached_audio, cached_codes, voice_a, 16), "still reusable");

    std::printf("changing the voice invalidates them\n");
    nano::invalidate_reference_codes_if_changed(cached_audio, cached_codes, voice_b);
    check(!cached_codes.has_value(), "codes dropped, so the new voice cannot reuse the old codes");
    cached_codes = make_codes(16);
    check(nano::reference_codes_reusable(cached_audio, cached_codes, voice_b, 16), "the new voice is cached");

    std::printf("near misses are not treated as the same voice\n");
    runtime::AudioBuffer tweaked = voice_b;
    tweaked.samples.back() += 0.5F;
    check(!nano::reference_codes_reusable(cached_audio, cached_codes, tweaked, 16),
          "one changed sample -> not reusable");
    runtime::AudioBuffer mono = voice_b;
    mono.channels = 1;
    check(!nano::reference_codes_reusable(cached_audio, cached_codes, mono, 16),
          "different channel count -> not reusable");
    runtime::AudioBuffer resampled = voice_b;
    resampled.sample_rate = 44100;
    check(!nano::reference_codes_reusable(cached_audio, cached_codes, resampled, 16),
          "different sample rate -> not reusable");

    std::printf("%s\n", g_failures == 0 ? "PASS: moss_tts_nano reference cache" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
