#pragma once

#include "engine/framework/core/execution_context.h"
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::dettato {
struct DettatoAssets;

struct Note {
    double onset = 0, offset = 0;
    int pitch = 0, velocity = 100;
    float confidence = 0;
    int voice = 0;
    bool inserted = false;
};
struct Track {
    std::string name;
    int program = 0;
    bool is_drum = false;
    std::vector<Note> notes;
};
struct Transcription {
    std::vector<Track> tracks;
    double tuning_cents = 0;
    double drum_router = 0;
    std::string drum_model;
    size_t transcribed = 0, dropped = 0, inserted = 0;
};

// One pipeline per session. Coordinates independently owned neural components
// and merges their pitched/drum tracks. Requests never share mutable state.
class DettatoPipeline {
  public:
    DettatoPipeline(std::shared_ptr<const DettatoAssets> assets, const core::ExecutionContext &execution,
                    int window_hop_sec = 2);
    ~DettatoPipeline();
    Transcription transcribe(const std::vector<float> &mono, int sample_rate);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::community_models::dettato
