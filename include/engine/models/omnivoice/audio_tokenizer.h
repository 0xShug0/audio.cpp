#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::omnivoice {

struct OmniVoiceReferenceAudioOptions {
    bool preprocess_prompt = true;
    // Longest reference the model is conditioned on. Everything past the last
    // pause inside this window is dropped. Zero disables the limit.
    float reference_max_seconds = 15.0F;
    // Exact digital silence written onto both ends of a clamped reference.
    int64_t reference_pad_ms = 150;
};

struct OmniVoiceReferenceClipOptions {
    float max_seconds = 15.0F;
    int64_t pad_ms = 150;
};

struct OmniVoiceReferenceClipResult {
    std::vector<float> samples;
    // True when speech was actually dropped off the end of the reference.
    bool clamped = false;
    // True when the reference is over the limit but holds no pause to cut at,
    // so the whole clip was kept rather than a word being sliced in half.
    bool no_pause_found = false;
    // Silence-run tier that produced the cut, in milliseconds; 0 when no cut.
    int64_t pause_tier_ms = 0;
    int64_t content_start_sample = 0;
    int64_t cut_sample = 0;
    double input_seconds = 0.0;
    double output_seconds = 0.0;
};

// Bounds a reference clip to `options.max_seconds` of speech.
//
// A reference within the limit is returned byte-for-byte unchanged. A longer
// one is cut at the *start* of the last silence run that finishes inside the
// budget -- never at the quietest single window, which lands mid-word -- and
// the surviving body is re-padded with exact digital silence at both ends.
// Silence runs are searched in tiers (250 ms, then 150 ms, then 80 ms); when
// no tier matches, the clip is kept whole and `no_pause_found` is set so the
// caller can warn instead of slicing a word.
//
// This is a port of the `clip_reference` shell function in
// `scripts/omnivoice_studio.sh`, which is the known-good implementation.
OmniVoiceReferenceClipResult clip_reference_mono(
    const std::vector<float> & mono,
    int sample_rate,
    const OmniVoiceReferenceClipOptions & options);

struct OmniVoiceAudioTokenizerRuntimeStats {
    bool encoder_graph_rebuilt = false;
    int64_t encoder_frame_capacity = 0;
    int64_t encoder_acoustic_sample_capacity = 0;
    int64_t encoder_semantic_sample_capacity = 0;
    double encoder_rebuild_ms = 0.0;
    bool decoder_graph_rebuilt = false;
    int64_t decoder_frame_capacity = 0;
    int64_t decoder_codebook_capacity = 0;
    double decoder_rebuild_ms = 0.0;
};

class OmniVoiceAudioTokenizerRuntime {
public:
    OmniVoiceAudioTokenizerRuntime(
        std::shared_ptr<const OmniVoiceAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~OmniVoiceAudioTokenizerRuntime();

    OmniVoiceAudioTokens encode_reference_audio(
        const runtime::AudioBuffer & audio,
        const OmniVoiceReferenceAudioOptions & options);
    runtime::AudioBuffer decode_audio_tokens(const OmniVoiceGeneratedAudioTokens & audio_tokens);
    void release_runtime_graphs();
    const OmniVoiceAudioTokenizerRuntimeStats & last_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::omnivoice
