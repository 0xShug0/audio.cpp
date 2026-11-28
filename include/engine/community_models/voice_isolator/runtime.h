#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace engine::models::voice_isolator {

struct ModelState;
class StreamingSession;

struct Options {
    // Streaming checkpoints: look-ahead in 10 ms frames, one of the trained
    // positions (0, 2 or 4 for voice-isolator-v1-realtime); -1 takes the
    // checkpoint's default.
    int64_t lookahead_frames = -1;
    // Stream frames computed together: sets throughput, adds no latency.
    int64_t block_frames = 4;
    // Offline render windows in seconds; 0 takes the checkpoint's own.
    double chunk_seconds = 0.0;
    double overlap_seconds = 0.0;
    // Storage of the large projection matrices. Native: BF16 on CPUs with
    // AVX512-BF16, where it runs 10-35% faster, and F32 (the reference numerics)
    // elsewhere, including CPUs without it, where BF16 is slower.
    assets::TensorStorageType weight_storage = assets::TensorStorageType::Native;
    // Storage of the streaming attention keys and values: F32 (the reference
    // numerics), F16, Q8_0, Q5_0 or Q4_0. Every frame reads the whole cache, so
    // smaller storage is faster where memory bandwidth limits. Native takes
    // Q8_0 on the CPU and F16 elsewhere.
    assets::TensorStorageType attention_cache_storage = assets::TensorStorageType::Native;
};

struct Waveform {
    int sample_rate = 48000;
    std::vector<float> samples;
};

// Inworld Voice Isolator v1: 48 kHz voice isolation. Two-sided checkpoints
// render offline in overlapping windows; causal checkpoints stream with a
// trained look-ahead.
class Model {
public:
    // config_json is the architecture the checkpoint describes itself with. The
    // execution context is the session's and must outlive the model.
    static Model load(
        std::string_view config_json,
        const assets::TensorSource & source,
        core::ExecutionContext & execution,
        const Options & options = {});
    // A released checkpoint, which carries that JSON as its "config" metadata.
    static Model load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::ExecutionContext & execution,
        const Options & options = {});

    Model();
    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    // Whether the checkpoint is causal: it streams, and isolate_mono_48k runs
    // a stream over the clip.
    bool streams() const;
    int64_t lookahead_frames() const;

    // The isolated voice of a mono 48 kHz clip, same length and level.
    Waveform isolate_mono_48k(const std::vector<float> & waveform) const;
    // A live stream; causal checkpoints only.
    std::unique_ptr<StreamingSession> create_streaming_session() const;

private:
    explicit Model(std::shared_ptr<ModelState> state);

    std::shared_ptr<ModelState> state_;
};

// Isolates a voice as its audio arrives. Each output sample is released once
// latency_samples() more input samples have arrived.
class StreamingSession {
public:
    explicit StreamingSession(std::shared_ptr<const ModelState> model);
    ~StreamingSession();
    StreamingSession(StreamingSession &&) noexcept;
    StreamingSession & operator=(StreamingSession &&) noexcept;
    StreamingSession(const StreamingSession &) = delete;
    StreamingSession & operator=(const StreamingSession &) = delete;

    // New mono 48 kHz samples in pieces of any size; returns the output samples that became final.
    std::vector<float> push(const float * samples, size_t count);
    std::vector<float> push(const std::vector<float> & samples);
    // Ends the stream and returns every remaining output sample.
    std::vector<float> flush();
    // Starts a new stream.
    void reset();
    int64_t latency_samples() const;

private:
    struct State;

    std::shared_ptr<const ModelState> model_;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::voice_isolator
