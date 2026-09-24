#pragma once

#include "engine/models/nemotron_asr/decoder.h"
#include "engine/models/nemotron_asr/encoder.h"
#include "engine/models/nemotron_asr/frontend.h"
#include "engine/models/nemotron_asr/speaker_tagging.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

// Masked speaker-tagged ASR, a port of NeMo SpeakerTaggedASR.perform_parallel_streaming_stt_spk
// with masked_asr=true: one cache-aware ASR stream per active speaker, fed with that
// speaker's masked features in NeMo's pad_and_drop_preencoded chunking.
namespace engine::models::nemotron_asr {

// Log-mel features equal to one full-file extract(center=true) with NeMo's zeroed
// padding frame, computed incrementally as audio arrives.
class MelFeatureSource {
public:
    explicit MelFeatureSource(const NemotronFrontend & frontend, int64_t hop_length = 160, int64_t n_fft = 512);

    void append(const float * samples, size_t count);
    void append(std::vector<float> && samples);
    // Frames computable now; after the last audio: floor(samples / hop) + 1.
    int64_t available_frames(bool final) const noexcept;
    // Time-major [end - begin, features]. keep_group, when set, zeroes the audio of
    // 80 ms groups it rejects before the frontend runs (speaker_mask=audio).
    std::vector<float> frames(int64_t begin, int64_t end, bool final,
                              const std::function<bool(int64_t)> * keep_group = nullptr) const;
    void discard_before(int64_t frame);
    int64_t feature_dim() const noexcept { return feature_dim_; }

private:
    const NemotronFrontend * frontend_;
    int64_t hop_;
    int64_t n_fft_;
    mutable int64_t feature_dim_ = 128;
    // Frames [cache_begin_, cache_end_) computed in one frontend call.
    mutable std::vector<float> cache_;
    mutable int64_t cache_begin_ = 0;
    mutable int64_t cache_end_ = 0;
    std::vector<float> compute(int64_t begin, int64_t end, bool final,
                               const std::function<bool(int64_t)> * keep_group) const;
    std::vector<float> audio_;
    int64_t base_ = 0;  // sample index of audio_[0]
    int64_t received_ = 0;
};


// NeMo _find_active_speakers: speakers whose 80 ms activity exceeds 0.5 anywhere
// in [begin, end), ascending. value(group, speaker) reads the timeline.
std::vector<int> active_speakers(
    const std::function<float(int64_t, int64_t)> & value, int64_t speakers, int64_t begin, int64_t end);
// NeMo mask_features mask: each 80 ms decision repeated 8 times, truncated from the
// right or left-padded with zeros to `length` feature frames.
std::vector<float> feature_mask(const std::vector<bool> & active_groups, int64_t length);
// NeMo mask_features, in place on time-major features: masked frames become 0.0 and
// frames that were already exactly 0.0 become the log floor (-16.6355).
void apply_feature_mask(std::vector<float> & features, const std::vector<float> & mask, int64_t dim);

class MaskedSpeakerStreams {
public:
    MaskedSpeakerStreams(
        NemotronEncoderRuntime & encoder,
        NemotronDecoderRuntime & decoder,
        const NemotronFrontend & frontend,
        const SpeakerProbabilities & probabilities,
        const SpeakerTaggingOptions & options,
        int64_t prompt_id,
        int64_t lookahead_tokens,
        NemotronDecodeOptions decode_options);

    void push_audio(const float * samples, size_t count);
    void push_audio(std::vector<float> && samples);
    // Runs every chunk whose audio is complete; final also runs the tail.
    void process(bool final);
    SpeakerSegmentBuilder & segments() { return segments_; }
    // Start time of the next chunk: nothing earlier can still change.
    double stream_time() const;

private:
    struct Speaker {
        NemotronEncoderSpeakerState encoder;
        NemotronDecoderStreamState decoder;
    };
    void run_chunk(int64_t chunk, int64_t new_frames, bool final);
    float group_value(int64_t group, int64_t speaker) const;

    NemotronEncoderRuntime * encoder_;
    NemotronDecoderRuntime * decoder_;
    const SpeakerProbabilities * probabilities_;
    bool audio_mask_;
    int64_t prompt_id_;
    int64_t lookahead_;
    NemotronDecodeOptions decode_options_;
    MelFeatureSource features_;
    SpeakerSegmentBuilder segments_;
    std::array<std::optional<Speaker>, 8> speakers_;
    int64_t next_chunk_ = 0;
    int64_t stream_groups_ = 0;  // NeMo diar_pred_out_stream length
    bool done_ = false;
};

}  // namespace engine::models::nemotron_asr
