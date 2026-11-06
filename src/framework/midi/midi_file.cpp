#include "engine/framework/midi/midi_file.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace engine::midi {
namespace {

constexpr double kMinimumNoteDurationSeconds = 0.01;

struct NoteEvent {
    bool is_drum = false;
    int program = 0;
    double time_seconds = 0.0;
    int velocity = 0;
    int pitch = 0;
    std::string track_name;
};

void require_u7(int value, const char * name) {
    if (value < 0 || value > 127) {
        throw std::runtime_error(std::string(name) + " must be in [0, 127]");
    }
}

void append_u8(std::vector<std::byte> & out, uint8_t value) {
    out.push_back(static_cast<std::byte>(value));
}

void append_u16_be(std::vector<std::byte> & out, uint16_t value) {
    append_u8(out, static_cast<uint8_t>((value >> 8) & 0xff));
    append_u8(out, static_cast<uint8_t>(value & 0xff));
}

void append_u24_be(std::vector<std::byte> & out, uint32_t value) {
    append_u8(out, static_cast<uint8_t>((value >> 16) & 0xff));
    append_u8(out, static_cast<uint8_t>((value >> 8) & 0xff));
    append_u8(out, static_cast<uint8_t>(value & 0xff));
}

void append_u32_be(std::vector<std::byte> & out, uint32_t value) {
    append_u8(out, static_cast<uint8_t>((value >> 24) & 0xff));
    append_u8(out, static_cast<uint8_t>((value >> 16) & 0xff));
    append_u8(out, static_cast<uint8_t>((value >> 8) & 0xff));
    append_u8(out, static_cast<uint8_t>(value & 0xff));
}

void append_ascii(std::vector<std::byte> & out, const char * text) {
    while (*text != '\0') {
        append_u8(out, static_cast<uint8_t>(*text));
        ++text;
    }
}

void append_bytes(std::vector<std::byte> & out, const std::string & text) {
    for (const unsigned char ch : text) {
        append_u8(out, ch);
    }
}

void append_varlen(std::vector<std::byte> & out, uint32_t value) {
    uint8_t buffer[5] = {};
    int index = 4;
    buffer[index] = static_cast<uint8_t>(value & 0x7f);
    while ((value >>= 7) != 0) {
        --index;
        buffer[index] = static_cast<uint8_t>((value & 0x7f) | 0x80);
    }
    for (; index < 5; ++index) {
        append_u8(out, buffer[index]);
    }
}

int64_t seconds_to_tick(double seconds, const MidiFileOptions & options) {
    const double ticks = seconds *
        static_cast<double>(options.ticks_per_beat) *
        1000000.0 /
        static_cast<double>(options.tempo_us_per_beat);
    return static_cast<int64_t>(std::llround(ticks));
}

int note_key_program(const MidiNote & note) {
    return note.is_drum ? kDrumProgram : note.program;
}

int event_key_program(const NoteEvent & event) {
    return event.is_drum ? kDrumProgram : event.program;
}

std::vector<NoteEvent> note_events_from_notes(const std::vector<MidiNote> & notes) {
    std::vector<NoteEvent> events;
    events.reserve(notes.size() * 2);
    for (const auto & note : notes) {
        require_u7(note.pitch, "MIDI note pitch");
        if (!note.is_drum) {
            require_u7(note.program, "MIDI program");
        }
        events.push_back(NoteEvent{
            note.is_drum,
            note.program,
            note.onset_seconds,
            1,
            note.pitch,
            note.track_name,
        });
        events.push_back(NoteEvent{
            note.is_drum,
            note.program,
            note.is_drum ? note.onset_seconds + kMinimumNoteDurationSeconds : note.offset_seconds,
            0,
            note.pitch,
            note.track_name,
        });
    }
    std::sort(events.begin(), events.end(), [](const NoteEvent & lhs, const NoteEvent & rhs) {
        return std::tie(lhs.time_seconds, lhs.is_drum, lhs.program, lhs.velocity, lhs.pitch) <
            std::tie(rhs.time_seconds, rhs.is_drum, rhs.program, rhs.velocity, rhs.pitch);
    });
    return events;
}

void append_midi_track(std::vector<std::byte> & file, const std::vector<std::byte> & track) {
    append_ascii(file, "MTrk");
    append_u32_be(file, static_cast<uint32_t>(track.size()));
    file.insert(file.end(), track.begin(), track.end());
}

}  // namespace

MidiTickSequence read_standard_midi_file(const std::vector<std::byte> & bytes) {
    size_t position = 0;
    size_t limit = bytes.size();
    const auto fail = [&]() {
        throw std::invalid_argument("Malformed or unsupported Standard MIDI File");
    };
    const auto read_byte = [&]() -> uint8_t {
        if (position >= limit) fail();
        return std::to_integer<uint8_t>(bytes[position++]);
    };
    const auto read_be = [&](int count) {
        uint32_t value = 0;
        for (int i = 0; i < count; ++i) value = (value << 8) | read_byte();
        return value;
    };
    const auto read_vlq = [&]() -> uint32_t {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const auto byte = read_byte();
            value = (value << 7) | (byte & 0x7f);
            if (!(byte & 0x80)) return value;
        }
        throw std::invalid_argument("Invalid MIDI variable-length quantity");
    };
    if (read_be(4) != 0x4d546864) fail();  // MThd
    const auto header_length = read_be(4);
    if (header_length < 6 || header_length > limit - position) fail();
    const auto header_end = position + header_length;
    const auto format = read_be(2);
    const auto track_count = read_be(2);
    const auto ppq = read_be(2);
    if (format > 1 || track_count == 0 || (format == 0 && track_count != 1) ||
        ppq == 0 || (ppq & 0x8000)) fail();
    position = header_end;

    MidiTickSequence sequence;
    sequence.ticks_per_beat = static_cast<int>(ppq);
    for (uint32_t track = 0; track < track_count; ++track) {
        if (read_be(4) != 0x4d54726b) fail();  // MTrk
        const auto track_length = read_be(4);
        if (track_length > limit - position) fail();
        limit = position + track_length;
        uint64_t tick = 0;
        uint8_t running_status = 0;
        std::map<int, std::deque<uint64_t>> active;
        while (position < limit) {
            const auto delta = read_vlq();
            if (tick > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - delta) {
                fail();
            }
            tick += delta;
            auto status = read_byte();
            if (status < 0x80) {
                if (running_status < 0x80 || running_status >= 0xf0) fail();
                --position;
                status = running_status;
            } else if (status != 0xff) {
                running_status = status;
            }
            if (status == 0xff) {
                const auto type = read_byte();
                if (type >= 0x80) fail();
                const auto length = read_vlq();
                if (length > limit - position) fail();
                // Fixed-width meta events must have their complete payload.
                int expected_length = -1;
                switch (type) {
                    case 0x00: expected_length = 2; break;
                    case 0x20: case 0x21: expected_length = 1; break;
                    case 0x2f: expected_length = 0; break;
                    case 0x51: expected_length = 3; break;
                    case 0x54: expected_length = 5; break;
                    case 0x58: expected_length = 4; break;
                    case 0x59: expected_length = 2; break;
                }
                if (expected_length >= 0 && length != static_cast<uint32_t>(expected_length)) fail();
                if ((type == 0x20 && std::to_integer<uint8_t>(bytes[position]) > 15) ||
                    (type == 0x21 && std::to_integer<uint8_t>(bytes[position]) > 127)) fail();
                if (type == 0x59) {
                    const auto key = std::to_integer<uint8_t>(bytes[position]);
                    if ((key > 7 && key < 249) || std::to_integer<uint8_t>(bytes[position + 1]) > 1) fail();
                }
                if (type == 0x51) {
                    if (read_be(3) == 0) fail();
                } else {
                    position += length;
                }
                continue;
            }
            if (status == 0xf0 || status == 0xf7) {
                const auto length = read_vlq();
                if (length > limit - position) fail();
                // F7 escape payloads may contain arbitrary MIDI status bytes.
                position += length;
                continue;
            }
            if (status >= 0xf0) fail();
            const auto kind = status >> 4;
            const auto channel = status & 0x0f;
            const auto data1 = read_byte();
            const auto data2 = (kind == 0xc || kind == 0xd) ? 0 : read_byte();
            if (data1 >= 0x80 || data2 >= 0x80) fail();
            if (kind != 0x8 && kind != 0x9) continue;
            auto & pending = active[channel * 128 + data1];
            if (kind == 0x9 && data2 > 0) {
                pending.push_back(tick);
            } else {
                if (pending.empty()) {
                    throw std::invalid_argument("Unmatched note-off in MIDI track " + std::to_string(track));
                }
                const auto onset = pending.front();
                pending.pop_front();
                sequence.notes.push_back({static_cast<int64_t>(onset), static_cast<int64_t>(tick), channel, data1});
            }
        }
        for (const auto & entry : active) {
            if (!entry.second.empty()) {
                throw std::invalid_argument("Unterminated note in MIDI track " + std::to_string(track));
            }
        }
        limit = bytes.size();
    }
    return sequence;
}


std::vector<MidiNote> clean_midi_notes(std::vector<MidiNote> notes) {
    for (auto & note : notes) {
        if (note.is_drum) {
            note.program = kDrumProgram;
            note.offset_seconds = note.onset_seconds + kMinimumNoteDurationSeconds;
        } else if (note.onset_seconds > note.offset_seconds ||
                   note.offset_seconds - note.onset_seconds < kMinimumNoteDurationSeconds) {
            note.offset_seconds = note.onset_seconds + kMinimumNoteDurationSeconds;
        }
    }
    std::map<std::tuple<int, int, bool>, std::vector<MidiNote>> channels;
    for (auto & note : notes) {
        channels[{note_key_program(note), note.pitch, note.is_drum}].push_back(std::move(note));
    }
    std::vector<MidiNote> out;
    out.reserve(notes.size());
    for (auto & [key, channel_notes] : channels) {
        (void)key;
        std::sort(channel_notes.begin(), channel_notes.end(), [](const MidiNote & lhs, const MidiNote & rhs) {
            return lhs.onset_seconds < rhs.onset_seconds;
        });
        for (size_t i = 1; i < channel_notes.size(); ++i) {
            if (channel_notes[i - 1].offset_seconds > channel_notes[i].onset_seconds) {
                channel_notes[i - 1].offset_seconds = channel_notes[i].onset_seconds;
            }
        }
        for (auto & note : channel_notes) {
            if (note.is_drum || note.onset_seconds < note.offset_seconds) {
                out.push_back(std::move(note));
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const MidiNote & lhs, const MidiNote & rhs) {
        return std::tie(lhs.onset_seconds, lhs.is_drum, lhs.program, lhs.pitch, lhs.offset_seconds) <
            std::tie(rhs.onset_seconds, rhs.is_drum, rhs.program, rhs.pitch, rhs.offset_seconds);
    });
    return out;
}

std::vector<std::byte> write_standard_midi_file(
    const std::vector<MidiNote> & notes,
    const MidiFileOptions & options) {
    if (options.ticks_per_beat <= 0) {
        throw std::runtime_error("MIDI ticks_per_beat must be positive");
    }
    if (options.tempo_us_per_beat <= 0 || options.tempo_us_per_beat > 0x00ffffff) {
        throw std::runtime_error("MIDI tempo_us_per_beat must fit in a tempo meta event");
    }
    require_u7(options.velocity, "MIDI velocity");

    const auto clean_notes = clean_midi_notes(notes);
    const auto events = note_events_from_notes(clean_notes);

    std::vector<std::byte> meta_track;
    append_u8(meta_track, 0x00);
    append_u8(meta_track, 0xff);
    append_u8(meta_track, 0x51);
    append_u8(meta_track, 0x03);
    append_u24_be(meta_track, static_cast<uint32_t>(options.tempo_us_per_beat));
    append_u8(meta_track, 0x00);
    append_u8(meta_track, 0xff);
    append_u8(meta_track, 0x2f);
    append_u8(meta_track, 0x00);

    std::map<int, std::vector<std::byte>> tracks;
    std::map<int, int64_t> track_ticks;
    std::map<int, int> channels;
    std::vector<int> available_channels{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15};
    std::vector<int> track_order;
    for (const auto & event : events) {
        const int key = event_key_program(event);
        if (!tracks.count(key)) {
            track_order.push_back(key);
            auto & track = tracks[key];
            track_ticks[key] = 0;
            const bool is_drum = key == kDrumProgram;
            const int channel = is_drum
                ? 9
                : (available_channels.empty() ? 15 : available_channels.front());
            if (!is_drum && !available_channels.empty()) {
                available_channels.erase(available_channels.begin());
            }
            channels[key] = channel;
            const std::string name = !event.track_name.empty()
                ? event.track_name
                : (is_drum ? std::string("drums") : "program " + std::to_string(key));
            append_u8(track, 0x00);
            append_u8(track, 0xff);
            append_u8(track, 0x03);
            append_varlen(track, static_cast<uint32_t>(name.size()));
            append_bytes(track, name);
            append_u8(track, 0x00);
            append_u8(track, static_cast<uint8_t>(0xc0 | channels[key]));
            append_u8(track, static_cast<uint8_t>(is_drum ? 0 : key));
        }
        const int64_t absolute_tick = seconds_to_tick(event.time_seconds, options);
        if (absolute_tick < 0) {
            throw std::runtime_error("MIDI note time must be non-negative");
        }
        const int64_t delta_tick = absolute_tick - track_ticks[key];
        if (delta_tick < 0) {
            throw std::runtime_error("MIDI events for a track must be time sorted");
        }
        track_ticks[key] = absolute_tick;
        auto & track = tracks[key];
        append_varlen(track, static_cast<uint32_t>(delta_tick));
        append_u8(track, static_cast<uint8_t>((event.velocity > 0 ? 0x90 : 0x80) | channels[key]));
        append_u8(track, static_cast<uint8_t>(event.pitch));
        append_u8(track, static_cast<uint8_t>(event.velocity > 0 ? options.velocity : 0));
    }
    for (const int key : track_order) {
        auto & track = tracks[key];
        append_u8(track, 0x00);
        append_u8(track, 0xff);
        append_u8(track, 0x2f);
        append_u8(track, 0x00);
    }

    std::vector<std::byte> file;
    append_ascii(file, "MThd");
    append_u32_be(file, 6);
    append_u16_be(file, 1);
    append_u16_be(file, static_cast<uint16_t>(1 + track_order.size()));
    append_u16_be(file, static_cast<uint16_t>(options.ticks_per_beat));
    append_midi_track(file, meta_track);
    for (const int key : track_order) {
        append_midi_track(file, tracks[key]);
    }
    return file;
}

}  // namespace engine::midi
