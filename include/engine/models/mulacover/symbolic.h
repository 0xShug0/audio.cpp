#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace engine::models::mulacover {

// Positions and durations use the upstream sixteenth-note grid, not seconds.
struct SymbolicNote {
    int64_t onset = 0;
    int64_t pitch = 0;
    int64_t duration = 0;
};

struct SymbolicInputs {
    int64_t length = 0;
    std::vector<float> pianoroll;
    std::vector<float> drum_pianoroll;
    std::vector<float> chord;
};

SymbolicInputs build_symbolic_inputs(
    const std::vector<SymbolicNote> & melody,
    const std::vector<SymbolicNote> & chords,
    const std::vector<SymbolicNote> & drums);

// Merge melody/chord tracks excluding channel 9; an explicit drum file uses all channels.
SymbolicInputs load_symbolic_midi(
    const std::filesystem::path & melody,
    const std::filesystem::path & chord,
    const std::optional<std::filesystem::path> & drums = std::nullopt);

}  // namespace engine::models::mulacover
