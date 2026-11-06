#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::mulacover {

struct ChordEvent {
    std::string label;
    double onset_seconds = 0;
    double offset_seconds = 0;
};

struct ChordGridEvent {
    std::string label;
    int64_t onset = 0;
    int64_t offset = 0;
};

// One instance per session. Execution must outlive this object; calls are serial.
// Sources preserve the upstream net state-dictionary names (without "net.").
class ChordNet {
public:
    ChordNet(core::ExecutionContext & execution,
             const std::array<std::shared_ptr<const assets::TensorSource>, 5> & sources,
             const std::filesystem::path & vocabulary_path);
    ~ChordNet();
    ChordNet(const ChordNet &) = delete;
    ChordNet & operator=(const ChordNet &) = delete;

    // Row-major, already cropped CQT magnitudes [T,252]. No CQT frontend here.
    // Frame duration matches upstream hop_length/sample_rate (512/22050).
    std::vector<ChordEvent> forward(const std::vector<float> & cqt);
    // Upstream whole-beat grid, including N and zero-duration rounded intervals.
    std::vector<ChordGridEvent> forward(const std::vector<float> & cqt, double bpm);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
