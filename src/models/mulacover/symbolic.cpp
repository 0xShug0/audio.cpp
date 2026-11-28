#include "engine/models/mulacover/symbolic.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/midi/midi_file.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::mulacover {

namespace {

std::vector<SymbolicNote> read_symbolic_midi(const std::filesystem::path & path, bool include_drum_channel) {
    const auto sequence = midi::read_standard_midi_file(io::read_binary_file(path));
    const auto ppq = sequence.ticks_per_beat;
    const auto quantize = [ppq](int64_t tick) {
        if (tick > (std::numeric_limits<int64_t>::max() - ppq) / 8) {
            throw std::invalid_argument("MuLaCover MIDI tick exceeds the symbolic timing range");
        }
        return (tick * 8 + ppq) / (2 * ppq);
    };
    std::vector<SymbolicNote> notes;
    for (const auto & note : sequence.notes) {
        if (!include_drum_channel && note.channel == 9) continue;
        const auto start = quantize(note.onset_tick);
        notes.push_back({start, note.pitch, quantize(note.offset_tick) - start});
    }
    return notes;
}

}  // namespace

SymbolicInputs load_symbolic_midi(
    const std::filesystem::path & melody,
    const std::filesystem::path & chord,
    const std::optional<std::filesystem::path> & drums) {
    const auto melody_notes = read_symbolic_midi(melody, false);
    const auto chord_notes = read_symbolic_midi(chord, false);
    const auto drum_notes = drums ? read_symbolic_midi(*drums, true) : std::vector<SymbolicNote>{};
    return build_symbolic_inputs(melody_notes, chord_notes, drum_notes);
}

SymbolicInputs build_symbolic_inputs(
    const std::vector<SymbolicNote> & melody,
    const std::vector<SymbolicNote> & chords,
    const std::vector<SymbolicNote> & drums) {
    constexpr int64_t max_length = 5000;
    if (melody.empty()) {
        throw std::invalid_argument("MuLaCover melody must contain at least one note");
    }
    SymbolicInputs inputs;
    for (const auto * notes : {&melody, &chords, &drums}) {
        for (const auto & note : *notes) {
            if (notes == &chords && note.duration == 0 && note.onset >= 0 &&
                note.pitch >= 0 && note.pitch <= 127) continue;
            if (note.onset < 0 || note.pitch < 0 || note.pitch > 127 ||
                note.duration < 0 || note.onset > max_length ||
                note.duration > max_length - note.onset) {
                throw std::invalid_argument("MuLaCover symbolic note is outside the pitch or timing limits");
            }
            const auto duration = notes == &chords ? note.duration : std::max(int64_t{1}, note.duration);
            if (duration > max_length - note.onset) {
                throw std::invalid_argument("MuLaCover symbolic input exceeds 5000 sixteenth notes");
            }
            inputs.length = std::max(inputs.length, note.onset + duration);
        }
    }
    inputs.pianoroll.resize(static_cast<size_t>(inputs.length * 256));
    inputs.drum_pianoroll.resize(static_cast<size_t>(inputs.length * 128));
    inputs.chord.resize(static_cast<size_t>(inputs.length * 12));
    for (const auto & note : melody) {
        inputs.pianoroll[static_cast<size_t>(note.onset * 256 + note.pitch * 2)] = 1.0F;
        for (int64_t i = note.onset + 1; i < note.onset + std::max(int64_t{1}, note.duration); ++i) {
            inputs.pianoroll[static_cast<size_t>(i * 256 + note.pitch * 2 + 1)] = 1.0F;
        }
    }
    for (const auto & note : drums) {
        for (int64_t i = note.onset; i < note.onset + std::max(int64_t{1}, note.duration); ++i) {
            inputs.drum_pianoroll[static_cast<size_t>(i * 128 + note.pitch)] = 1.0F;
        }
    }
    for (const auto & note : chords) {
        for (int64_t i = note.onset; i < note.onset + note.duration; ++i) {
            inputs.chord[static_cast<size_t>(i * 12 + note.pitch % 12)] = 1.0F;
        }
    }
    return inputs;
}

}  // namespace engine::models::mulacover
