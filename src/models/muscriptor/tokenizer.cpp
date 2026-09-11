#include "engine/models/muscriptor/tokenizer.h"

#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/midi/midi_file.h"

#include <cstddef>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace engine::models::muscriptor {
namespace {

constexpr int kDrumProgram = 128;
constexpr double kFrameRate = 100.0;
constexpr double kMinimumNoteDuration = 0.01;

std::string event_key(const std::string & type, int value) {
    return type + ":" + std::to_string(value);
}

std::string readable_instrument_name(std::string value) {
    for (char & ch : value) {
        if (ch == '_') {
            ch = ' ';
        }
    }
    return value;
}

int program_for_instrument(const std::string & instrument) {
    static const std::unordered_map<std::string, int> kInstrumentToProgram = {
        {"acoustic_piano", 0}, {"electric_piano", 2}, {"chromatic_percussion", 8}, {"organ", 16},
        {"acoustic_guitar", 24}, {"clean_electric_guitar", 26}, {"distorted_electric_guitar", 29},
        {"acoustic_bass", 32}, {"electric_bass", 33}, {"violin", 40}, {"viola", 41}, {"cello", 42},
        {"contrabass", 43}, {"orchestral_harp", 46}, {"timpani", 47}, {"string_ensemble", 48},
        {"synth_strings", 50}, {"voice", 52}, {"orchestra_hit", 55}, {"trumpet", 56},
        {"trombone", 57}, {"tuba", 58}, {"french_horn", 60}, {"brass_section", 61},
        {"soprano_and_alto_sax", 64}, {"tenor_sax", 66}, {"baritone_sax", 67}, {"oboe", 68},
        {"english_horn", 69}, {"bassoon", 70}, {"clarinet", 71}, {"flutes", 72},
        {"synth_lead", 80}, {"synth_pad", 88}, {"drums", engine::midi::kDrumProgram},
    };
    if (const auto it = kInstrumentToProgram.find(instrument); it != kInstrumentToProgram.end()) {
        return it->second;
    }
    constexpr std::string_view prefix = "program_";
    if (instrument.rfind(prefix.data(), 0) == 0) {
        return std::stoi(instrument.substr(prefix.size()));
    }
    throw std::runtime_error("unknown MuScriptor instrument for MIDI: " + instrument);
}

std::vector<std::string> split_csv(const std::string & csv) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream input(csv);
    while (std::getline(input, item, ',')) {
        item = engine::io::trim_ascii_whitespace(std::move(item));
        if (item.empty()) {
            continue;
        }
        out.push_back(std::move(item));
    }
    return out;
}

struct NoteKey {
    int program = 0;
    int pitch = 0;

    bool operator<(const NoteKey & other) const noexcept {
        return program == other.program ? pitch < other.pitch : program < other.program;
    }
};

}  // namespace

MuScriptorTokenizer::MuScriptorTokenizer() {
    vocab_.push_back({"PAD", 0});
    vocab_.push_back({"EOS", 0});
    vocab_.push_back({"UNK", 0});
    for (int shift = 0; shift < 1001; ++shift) {
        vocab_.push_back({"shift", shift});
    }
    for (int pitch = 0; pitch < 128; ++pitch) {
        vocab_.push_back({"pitch", pitch});
    }
    for (int velocity = 0; velocity < 2; ++velocity) {
        vocab_.push_back({"velocity", velocity});
    }
    vocab_.push_back({"tie", 0});
    for (int program = 0; program < 130; ++program) {
        vocab_.push_back({"program", program});
    }
    for (int drum = 0; drum < 128; ++drum) {
        vocab_.push_back({"drum", drum});
    }

    group_name_to_id_ = {
        {"acoustic_piano", 0}, {"electric_piano", 1}, {"chromatic_percussion", 2}, {"organ", 3},
        {"acoustic_guitar", 4}, {"clean_electric_guitar", 5}, {"distorted_electric_guitar", 6},
        {"acoustic_bass", 7}, {"electric_bass", 8}, {"violin", 9}, {"viola", 10}, {"cello", 11},
        {"contrabass", 12}, {"orchestral_harp", 13}, {"timpani", 14}, {"string_ensemble", 15},
        {"synth_strings", 16}, {"voice", 17}, {"orchestra_hit", 18}, {"trumpet", 19},
        {"trombone", 20}, {"tuba", 21}, {"french_horn", 22}, {"brass_section", 23},
        {"soprano_and_alto_sax", 24}, {"tenor_sax", 25}, {"baritone_sax", 26}, {"oboe", 27},
        {"english_horn", 28}, {"bassoon", 29}, {"clarinet", 30}, {"flutes", 31},
        {"synth_lead", 32}, {"synth_pad", 33}, {"drums", 36},
    };
    group_programs_ = {
        {0, {0, 1, 3, 6, 7}}, {1, {2, 4, 5}}, {2, {8, 9, 10, 11, 12, 13, 14, 15}},
        {3, {16, 17, 18, 19, 20, 21, 22, 23}}, {4, {24, 25}}, {5, {26, 27, 28}},
        {6, {29, 30, 31}}, {7, {32, 35}}, {8, {33, 34, 36, 37, 38, 39}}, {9, {40}},
        {10, {41}}, {11, {42}}, {12, {43}}, {13, {46}}, {14, {47}},
        {15, {48, 49, 44, 45}}, {16, {50, 51}}, {17, {52, 53, 54}}, {18, {55}},
        {19, {56, 59}}, {20, {57}}, {21, {58}}, {22, {60}}, {23, {61, 62, 63}},
        {24, {64, 65}}, {25, {66}}, {26, {67}}, {27, {68}}, {28, {69}}, {29, {70}},
        {30, {71}}, {31, {72, 73, 74, 75, 76, 77, 78, 79}},
        {32, {80, 81, 82, 83, 84, 85, 86, 87}}, {33, {88, 89, 90, 91, 92, 93, 94, 95}},
        {34, {100}}, {35, {101}},
    };
    for (const auto & [name, group] : group_name_to_id_) {
        if (name == "drums") {
            continue;
        }
        const auto it = group_programs_.find(group);
        if (it != group_programs_.end() && !it->second.empty()) {
            program_to_name_.emplace(it->second.front(), name);
        }
    }
}

int32_t MuScriptorTokenizer::eos_id() const noexcept {
    return 1;
}

const std::vector<MuScriptorTokenEvent> & MuScriptorTokenizer::vocab() const noexcept {
    return vocab_;
}

std::vector<int32_t> MuScriptorTokenizer::condition_instrument_ids(const std::string & csv) const {
    const auto names = split_csv(csv);
    if (names.empty()) {
        return {1};
    }
    std::vector<int32_t> out;
    out.reserve(names.size());
    for (const auto & item : names) {
        const auto it = group_name_to_id_.find(item);
        if (it == group_name_to_id_.end()) {
            throw std::runtime_error("unknown MuScriptor instrument: " + item);
        }
        out.push_back(static_cast<int32_t>(it->second + 2));
    }
    return out;
}

std::vector<int32_t> MuScriptorTokenizer::forbidden_token_ids(const std::string & csv) const {
    const auto names = split_csv(csv);
    if (names.empty()) {
        return {};
    }
    bool allow_drums = false;
    std::set<int> allowed_programs;
    for (const auto & name : names) {
        const auto group_it = group_name_to_id_.find(name);
        if (group_it == group_name_to_id_.end()) {
            throw std::runtime_error("unknown MuScriptor instrument: " + name);
        }
        if (name == "drums") {
            allow_drums = true;
            continue;
        }
        const auto program_it = group_programs_.find(group_it->second);
        if (program_it != group_programs_.end() && !program_it->second.empty()) {
            allowed_programs.insert(program_it->second.front());
        }
    }

    std::vector<int32_t> forbidden;
    for (int32_t id = 0; id < static_cast<int32_t>(vocab_.size()); ++id) {
        const auto & event = vocab_[static_cast<size_t>(id)];
        if (event.type == "program" && !allowed_programs.count(event.value)) {
            forbidden.push_back(id);
        } else if (event.type == "drum" && !allow_drums) {
            forbidden.push_back(id);
        }
    }
    return forbidden;
}

int32_t MuScriptorTokenizer::token_id(const std::string & type, int value) const {
    for (int32_t id = 0; id < static_cast<int32_t>(vocab_.size()); ++id) {
        const auto & event = vocab_[static_cast<size_t>(id)];
        if (event.type == type && event.value == value) {
            return id;
        }
    }
    throw std::runtime_error("MuScriptor token not found: " + event_key(type, value));
}

std::vector<int32_t> MuScriptorTokenizer::tie_section_token_ids(const std::set<std::pair<int, int>> & open_notes) const {
    std::vector<int32_t> tokens;
    std::optional<int> current_program;
    for (const auto & [program, pitch] : open_notes) {
        if (!current_program.has_value() || *current_program != program) {
            tokens.push_back(token_id("program", program));
            current_program = program;
        }
        tokens.push_back(token_id("pitch", pitch));
    }
    tokens.push_back(token_id("tie", 0));
    return tokens;
}

std::string MuScriptorTokenizer::instrument_for_program(int program) const {
    if (program == kDrumProgram) {
        return "drums";
    }
    const auto it = program_to_name_.find(program);
    if (it != program_to_name_.end()) {
        return it->second;
    }
    return "program_" + std::to_string(program);
}

std::set<std::pair<int, int>> MuScriptorTokenizer::open_note_keys(
    const std::vector<MuScriptorGeneratedChunk> & chunks) const {
    std::set<std::pair<int, int>> open;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const auto & chunk = chunks[chunk_index];
        const double seek = static_cast<double>(chunk_index) * 5.0;
        const std::optional<double> next_seek(static_cast<double>(chunk_index + 1) * 5.0);
        const int start_tick = static_cast<int>(std::llround(seek * kFrameRate));
        int tick = start_tick;
        std::optional<int> program;
        std::optional<int> velocity;
        bool in_prologue = true;
        bool skip_rest = false;
        std::set<std::pair<int, int>> tie_set;
        for (const int32_t token : chunk.tokens) {
            if (token < 0 || token >= static_cast<int32_t>(vocab_.size())) {
                continue;
            }
            const auto & event = vocab_[static_cast<size_t>(token)];
            if (in_prologue) {
                if (event.type == "tie") {
                    in_prologue = false;
                    velocity.reset();
                    std::vector<std::pair<int, int>> to_close;
                    for (const auto & key : open) {
                        if (!tie_set.count(key)) {
                            to_close.push_back(key);
                        }
                    }
                    for (const auto & key : to_close) {
                        open.erase(key);
                    }
                    continue;
                }
                if (event.type == "shift") {
                    in_prologue = false;
                    skip_rest = true;
                    open.clear();
                    continue;
                }
                if (event.type == "program") {
                    program = event.value;
                } else if (event.type == "pitch" && program.has_value()) {
                    tie_set.insert({*program, event.value});
                }
                continue;
            }
            if (skip_rest) {
                continue;
            }
            if (event.type == "shift" && event.value > 0) {
                tick = start_tick + event.value;
            } else if (event.type == "program") {
                program = event.value;
            } else if (event.type == "velocity") {
                velocity = event.value;
            } else if (event.type == "pitch" && program.has_value() && velocity.has_value()) {
                const double time = static_cast<double>(tick) / kFrameRate;
                if (next_seek.has_value() && time >= *next_seek) {
                    continue;
                }
                const std::pair<int, int> key{*program, event.value};
                open.erase(key);
                if (*velocity > 0) {
                    open.insert(key);
                }
            }
        }
        if (in_prologue) {
            open.clear();
        }
    }
    return open;
}

std::vector<MuScriptorEvent> MuScriptorTokenizer::decode_chunks(const std::vector<MuScriptorGeneratedChunk> & chunks) const {
    std::vector<MuScriptorEvent> out;
    std::map<NoteKey, double> open;
    std::map<NoteKey, int> key_to_index;
    int next_index = 0;

    auto close_note = [&](int program, int pitch, double time) {
        const NoteKey key{program, pitch};
        const auto open_it = open.find(key);
        if (open_it == open.end()) {
            return;
        }
        open.erase(open_it);
        const auto index_it = key_to_index.find(key);
        if (index_it != key_to_index.end()) {
            const int matched_index = index_it->second;
            key_to_index.erase(index_it);
            MuScriptorEvent event;
            event.kind = MuScriptorEvent::Kind::End;
            event.end = {time, matched_index};
            out.push_back(std::move(event));
        }
    };

    auto start_note = [&](int program, int pitch, double time) {
        const auto instrument = instrument_for_program(program);
        MuScriptorEvent event;
        event.kind = MuScriptorEvent::Kind::Start;
        event.start = {pitch, time, next_index, instrument};
        open[{program, pitch}] = time;
        key_to_index[{program, pitch}] = next_index;
        ++next_index;
        out.push_back(std::move(event));
    };

    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const double seek = static_cast<double>(chunk_index) * 5.0;
        const std::optional<double> next_seek = chunk_index + 1 < chunks.size()
            ? std::optional<double>(static_cast<double>(chunk_index + 1) * 5.0)
            : std::nullopt;
        int tick = static_cast<int>(std::llround(seek * kFrameRate));
        std::optional<int> program;
        std::optional<int> velocity;
        bool in_prologue = true;
        bool skip_rest = false;
        std::set<NoteKey> tie_set;
        for (const int32_t token : chunks[chunk_index].tokens) {
            if (token < 0 || token >= static_cast<int32_t>(vocab_.size())) {
                continue;
            }
            const auto & event = vocab_[static_cast<size_t>(token)];
            if (in_prologue) {
                if (event.type == "tie") {
                    in_prologue = false;
                    velocity.reset();
                    std::vector<NoteKey> to_close;
                    for (const auto & item : open) {
                        if (!tie_set.count(item.first)) {
                            to_close.push_back(item.first);
                        }
                    }
                    for (const auto & key : to_close) {
                        close_note(key.program, key.pitch, seek);
                    }
                    continue;
                }
                if (event.type == "shift") {
                    in_prologue = false;
                    skip_rest = true;
                    std::vector<NoteKey> to_close;
                    for (const auto & item : open) {
                        to_close.push_back(item.first);
                    }
                    for (const auto & key : to_close) {
                        close_note(key.program, key.pitch, seek);
                    }
                    continue;
                }
                if (event.type == "program") {
                    program = event.value;
                } else if (event.type == "pitch" && program.has_value()) {
                    tie_set.insert({*program, event.value});
                }
                continue;
            }
            if (skip_rest) {
                continue;
            }
            if (event.type == "shift" && event.value > 0) {
                tick = static_cast<int>(std::llround(seek * kFrameRate)) + event.value;
            } else if (event.type == "program") {
                program = event.value;
            } else if (event.type == "velocity") {
                velocity = event.value;
            } else if (event.type == "drum") {
                const double time = static_cast<double>(tick) / kFrameRate;
                if (!next_seek.has_value() || time < *next_seek) {
                    start_note(kDrumProgram, event.value, time);
                    close_note(kDrumProgram, event.value, time + kMinimumNoteDuration);
                }
            } else if (event.type == "pitch" && program.has_value() && velocity.has_value()) {
                const double time = static_cast<double>(tick) / kFrameRate;
                if (next_seek.has_value() && time >= *next_seek) {
                    continue;
                }
                const NoteKey key{*program, event.value};
                if (open.count(key)) {
                    close_note(key.program, key.pitch, time);
                }
                if (*velocity > 0) {
                    start_note(key.program, key.pitch, time);
                }
            }
        }
        if (in_prologue) {
            std::vector<NoteKey> to_close;
            for (const auto & item : open) {
                to_close.push_back(item.first);
            }
            for (const auto & key : to_close) {
                close_note(key.program, key.pitch, seek);
            }
        }
    }
    std::vector<NoteKey> to_close;
    for (const auto & item : open) {
        to_close.push_back(item.first);
    }
    for (const auto & key : to_close) {
        close_note(key.program, key.pitch, open[key] + kMinimumNoteDuration);
    }
    return out;
}

std::string muscriptor_event_to_json(const MuScriptorEvent & event) {
    std::ostringstream out;
    out << std::setprecision(17);
    if (event.kind == MuScriptorEvent::Kind::Start) {
        out << "{\"type\":\"start\",\"pitch\":" << event.start.pitch
            << ",\"start_time\":" << event.start.start_time
            << ",\"index\":" << event.start.index
            << ",\"instrument\":" << engine::io::json::stringify_string(event.start.instrument) << "}";
    } else {
        out << "{\"type\":\"end\",\"end_time\":" << event.end.end_time
            << ",\"start_event_index\":" << event.end.start_event_index << "}";
    }
    return out.str();
}

std::string muscriptor_events_to_json(const std::vector<MuScriptorEvent> & events) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < events.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << muscriptor_event_to_json(events[i]);
    }
    out << "]";
    return out.str();
}

std::vector<std::byte> muscriptor_events_to_midi_bytes(const std::vector<MuScriptorEvent> & events) {
    std::map<int, engine::midi::MidiNote> open_notes;
    std::vector<engine::midi::MidiNote> notes;
    for (const auto & event : events) {
        if (event.kind == MuScriptorEvent::Kind::Start) {
            const bool is_drum = event.start.instrument == "drums";
            engine::midi::MidiNote note;
            note.is_drum = is_drum;
            note.program = is_drum ? engine::midi::kDrumProgram : program_for_instrument(event.start.instrument);
            note.onset_seconds = event.start.start_time;
            note.offset_seconds = event.start.start_time;
            note.pitch = event.start.pitch;
            note.track_name = readable_instrument_name(event.start.instrument);
            open_notes[event.start.index] = std::move(note);
        } else {
            const auto it = open_notes.find(event.end.start_event_index);
            if (it == open_notes.end()) {
                throw std::runtime_error("MuScriptor MIDI event stream has unmatched note end");
            }
            it->second.offset_seconds = event.end.end_time;
            notes.push_back(std::move(it->second));
            open_notes.erase(it);
        }
    }
    return engine::midi::write_standard_midi_file(notes);
}

}  // namespace engine::models::muscriptor
