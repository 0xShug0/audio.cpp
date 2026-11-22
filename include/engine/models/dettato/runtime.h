#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::dettato {

struct DettatoAssets {
    assets::ResourceBundle resources;
    io::json::Value config;
    std::array<std::shared_ptr<const assets::TensorSource>, 4> weights;
};
std::shared_ptr<const DettatoAssets> load_dettato_assets(const std::filesystem::path &path);

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

// One runtime per session. Owns the four weight banks, cached PreparedGraphs,
// frontend constants and execution buffers. Requests never share mutable state.
class DettatoRuntime {
  public:
    DettatoRuntime(std::shared_ptr<const DettatoAssets> assets, const core::ExecutionContext &execution, int window_hop_sec = 2);
    ~DettatoRuntime();
    Transcription transcribe(const std::vector<float> &mono, int sample_rate);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::models::dettato
