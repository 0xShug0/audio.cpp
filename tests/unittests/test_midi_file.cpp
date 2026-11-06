#include "engine/framework/midi/midi_file.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include "test_assert.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

struct ParsedEvent {
    uint32_t tick = 0;
    uint8_t status = 0;
    uint8_t a = 0;
    uint8_t b = 0;
};

struct ParsedTrack {
    std::string name;
    int program = -1;
    int channel = -1;
    std::vector<ParsedEvent> events;
};

uint8_t byte_at(const std::vector<std::byte> & bytes, size_t offset) {
    if (offset >= bytes.size()) {
        throw std::runtime_error("MIDI parse offset out of range");
    }
    return static_cast<uint8_t>(bytes[offset]);
}

uint16_t read_u16_be(const std::vector<std::byte> & bytes, size_t offset) {
    return static_cast<uint16_t>((byte_at(bytes, offset) << 8) | byte_at(bytes, offset + 1));
}

uint32_t read_u32_be(const std::vector<std::byte> & bytes, size_t offset) {
    return (static_cast<uint32_t>(byte_at(bytes, offset)) << 24) |
        (static_cast<uint32_t>(byte_at(bytes, offset + 1)) << 16) |
        (static_cast<uint32_t>(byte_at(bytes, offset + 2)) << 8) |
        static_cast<uint32_t>(byte_at(bytes, offset + 3));
}

std::string read_ascii(const std::vector<std::byte> & bytes, size_t offset, size_t size) {
    std::string out;
    out.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        out.push_back(static_cast<char>(byte_at(bytes, offset + i)));
    }
    return out;
}

uint32_t read_varlen(const std::vector<std::byte> & bytes, size_t & offset, size_t end) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        if (offset >= end) {
            throw std::runtime_error("truncated MIDI variable length value");
        }
        const uint8_t ch = byte_at(bytes, offset++);
        value = (value << 7) | (ch & 0x7f);
        if ((ch & 0x80) == 0) {
            return value;
        }
    }
    throw std::runtime_error("MIDI variable length value is too long");
}

ParsedTrack parse_track(const std::vector<std::byte> & bytes, size_t offset, size_t size) {
    ParsedTrack out;
    const size_t end = offset + size;
    uint32_t tick = 0;
    while (offset < end) {
        tick += read_varlen(bytes, offset, end);
        const uint8_t status = byte_at(bytes, offset++);
        if (status == 0xff) {
            const uint8_t meta_type = byte_at(bytes, offset++);
            const uint32_t meta_size = read_varlen(bytes, offset, end);
            if (meta_type == 0x03) {
                out.name = read_ascii(bytes, offset, meta_size);
            }
            offset += meta_size;
            continue;
        }
        const uint8_t type = status & 0xf0;
        const uint8_t channel = status & 0x0f;
        if (type == 0xc0) {
            out.channel = channel;
            out.program = byte_at(bytes, offset++);
            continue;
        }
        if (type == 0x90 || type == 0x80) {
            ParsedEvent event;
            event.tick = tick;
            event.status = status;
            event.a = byte_at(bytes, offset++);
            event.b = byte_at(bytes, offset++);
            out.events.push_back(event);
            continue;
        }
        throw std::runtime_error("unexpected MIDI event status");
    }
    return out;
}

std::vector<ParsedTrack> parse_midi(const std::vector<std::byte> & bytes) {
    require_eq(read_ascii(bytes, 0, 4), std::string("MThd"), "MIDI header chunk");
    require_eq(read_u32_be(bytes, 4), uint32_t{6}, "MIDI header length");
    require_eq(read_u16_be(bytes, 8), uint16_t{1}, "MIDI format");
    require_eq(read_u16_be(bytes, 12), uint16_t{480}, "MIDI ticks per beat");
    const uint16_t track_count = read_u16_be(bytes, 10);
    std::vector<ParsedTrack> tracks;
    size_t offset = 14;
    for (uint16_t track = 0; track < track_count; ++track) {
        require_eq(read_ascii(bytes, offset, 4), std::string("MTrk"), "MIDI track chunk");
        const uint32_t size = read_u32_be(bytes, offset + 4);
        offset += 8;
        tracks.push_back(parse_track(bytes, offset, size));
        offset += size;
    }
    require_eq(offset, bytes.size(), "MIDI parsed byte count");
    return tracks;
}

std::vector<std::byte> bytes_from_hex(const std::string & hex) {
    require(hex.size() % 2 == 0, "hex fixture length must be even");
    std::vector<std::byte> out;
    out.reserve(hex.size() / 2);
    auto nibble = [](char ch) -> uint8_t {
        if (ch >= '0' && ch <= '9') {
            return static_cast<uint8_t>(ch - '0');
        }
        if (ch >= 'a' && ch <= 'f') {
            return static_cast<uint8_t>(ch - 'a' + 10);
        }
        if (ch >= 'A' && ch <= 'F') {
            return static_cast<uint8_t>(ch - 'A' + 10);
        }
        throw std::runtime_error("invalid hex fixture");
    };
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<std::byte>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
    }
    return out;
}

void require_bytes_eq(
    const std::vector<std::byte> & actual,
    const std::vector<std::byte> & expected,
    const std::string & label) {
    require_eq(actual.size(), expected.size(), label + " size");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(label + " byte mismatch at " + std::to_string(i));
        }
    }
}

void test_empty_midi_has_header_and_tempo_track() {
    const auto midi = engine::midi::write_standard_midi_file({});
    const auto tracks = parse_midi(midi);
    require_eq(tracks.size(), size_t{1}, "empty MIDI track count");
    require(tracks.front().events.empty(), "tempo track should not contain note events");
}

void test_multi_track_notes_and_drums() {
    const auto midi = engine::midi::write_standard_midi_file({
        {false, 0, 0.0, 0.5, 60, "acoustic piano"},
        {true, engine::midi::kDrumProgram, 0.25, 0.25, 36, "drums"},
    });
    const auto mido_golden = bytes_from_hex(
        "4d546864000000060001000301e0"
        "4d54726b0000000b00ff510307a12000ff2f00"
        "4d54726b0000002200ff030e61636f7573746963207069616e6f00c00000903c648360803c0000ff2f00"
        "4d54726b0000001900ff03056472756d7300c90081709924640a89240000ff2f00");
    require_bytes_eq(midi, mido_golden, "MIDI bytes from Python mido fixture");
    const auto tracks = parse_midi(midi);
    require_eq(tracks.size(), size_t{3}, "piano plus drums track count");
    require_eq(tracks[1].name, std::string("acoustic piano"), "piano track name");
    require_eq(tracks[1].channel, 0, "piano channel");
    require_eq(tracks[1].program, 0, "piano program");
    require_eq(tracks[1].events.size(), size_t{2}, "piano event count");
    require_eq(tracks[1].events[0].tick, uint32_t{0}, "piano note on tick");
    require_eq(tracks[1].events[0].status, uint8_t{0x90}, "piano note on status");
    require_eq(tracks[1].events[1].tick, uint32_t{480}, "piano note off tick");
    require_eq(tracks[1].events[1].status, uint8_t{0x80}, "piano note off status");

    require_eq(tracks[2].name, std::string("drums"), "drum track name");
    require_eq(tracks[2].channel, 9, "drum channel");
    require_eq(tracks[2].program, 0, "drum program change");
    require_eq(tracks[2].events.size(), size_t{2}, "drum event count");
    require_eq(tracks[2].events[0].tick, uint32_t{240}, "drum note on tick");
    require_eq(tracks[2].events[1].tick, uint32_t{250}, "drum note off tick");
}

void test_overlap_is_trimmed_before_serializing() {
    const auto midi = engine::midi::write_standard_midi_file({
        {false, 40, 0.0, 1.0, 69, "violin"},
        {false, 40, 0.25, 0.3, 70, "violin"},
        {false, 40, 0.5, 0.75, 69, "violin"},
    });
    const auto tracks = parse_midi(midi);
    require_eq(tracks.size(), size_t{2}, "overlap MIDI track count");
    require_eq(tracks[1].events.size(), size_t{6}, "overlap note event count");
    require_eq(tracks[1].events[3].tick, uint32_t{480}, "trimmed first note off tick");
    require_eq(tracks[1].events[3].status, uint8_t{0x80}, "trimmed first note off status");
    require_eq(tracks[1].events[4].tick, uint32_t{480}, "second note on tick");
    require_eq(tracks[1].events[4].status, uint8_t{0x90}, "second note on status");
}

void test_reader_roundtrip() {
    const auto sequence = engine::midi::read_standard_midi_file(engine::midi::write_standard_midi_file({
        {false, 0, 0.0, 0.5, 60, "piano"},
        {true, engine::midi::kDrumProgram, 0.25, 0.25, 36, "drums"},
    }));
    require_eq(sequence.ticks_per_beat, 480, "read PPQ");
    require_eq(sequence.notes.size(), size_t{2}, "read note count");
    require_eq(sequence.notes[0].channel, 0, "read piano channel");
    require_eq(sequence.notes[0].pitch, 60, "read piano pitch");
    require_eq(sequence.notes[0].onset_tick, int64_t{0}, "read piano onset");
    require_eq(sequence.notes[0].offset_tick, int64_t{480}, "read piano offset");
    require_eq(sequence.notes[1].channel, 9, "read drum channel");
    require_eq(sequence.notes[1].onset_tick, int64_t{240}, "read drum onset");
    require_eq(sequence.notes[1].offset_tick, int64_t{250}, "read drum offset");
}

void test_reader_running_status_and_overlapping_notes() {
    // Program change, overlapping same-pitch note-ons, text meta event,
    // velocity-zero note-off using running status, explicit note-off, EOT.
    const auto sequence = engine::midi::read_standard_midi_file(bytes_from_hex(
        "4d54686400000006000000010060"
        "4d54726b0000001a"
        "00c00500903c64183c5a0cff0101610c3c0018803c0000ff2f00"));
    require_eq(sequence.ticks_per_beat, 96, "nondefault PPQ");
    require_eq(sequence.notes.size(), size_t{2}, "overlap note count");
    require_eq(sequence.notes[0].onset_tick, int64_t{0}, "FIFO first onset");
    require_eq(sequence.notes[0].offset_tick, int64_t{48}, "FIFO first offset");
    require_eq(sequence.notes[1].onset_tick, int64_t{24}, "FIFO second onset");
    require_eq(sequence.notes[1].offset_tick, int64_t{72}, "FIFO second offset");
}

void test_reader_keeps_tick_timing_across_tempo_changes() {
    const auto sequence = engine::midi::read_standard_midi_file(bytes_from_hex(
        "4d54686400000006000000010060"
        "4d54726b0000001a"
        "00ff510307a12000903c6430ff51030f424030803c0000ff2f00"));
    require_eq(sequence.notes.size(), size_t{1}, "tempo-change note count");
    require_eq(sequence.notes[0].onset_tick, int64_t{0}, "tempo-change onset");
    require_eq(sequence.notes[0].offset_tick, int64_t{96}, "tempo-change offset stays in ticks");
}

void test_reader_skips_escaped_sysex() {
    const auto sequence = engine::midi::read_standard_midi_file(bytes_from_hex(
        "4d54686400000006000000010060"
        "4d54726b00000012"
        "00f703903c6400903c6460803c0000ff2f00"));
    require_eq(sequence.notes.size(), size_t{1}, "only channel events become notes");
    require_eq(sequence.notes[0].offset_tick, int64_t{96}, "note following escaped SysEx");
}

void test_reader_rejects_malformed_data() {
    const std::vector<std::string> malformed{
        "4d546864", // Truncated header.
        "4d546864000000060002000100604d54726b0000000400ff2f00", // Format 2.
        "4d546864000000060000000180604d54726b0000000400ff2f00", // SMPTE division.
        "4d546864000000060000000100604d54726b0000000300903c", // Missing velocity.
        "4d546864000000060000000100604d54726b0000000800803c0000ff2f00", // Unmatched off.
        "4d546864000000060000000100604d54726b0000000800903c6400ff2f00", // Unterminated on.
        "4d546864000000060000000100604d54726b00000009818080800000ff2f00", // Long VLQ.
        "4d546864000000060000000100604d54726b0000000b00ff510300000000ff2f00", // Zero tempo.
    };
    for (const auto & hex : malformed) {
        bool rejected = false;
        try {
            engine::midi::read_standard_midi_file(bytes_from_hex(hex));
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "malformed MIDI must be rejected: " + hex);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--log")) {
            throw std::invalid_argument("Usage: midi_file_test [--log FILE]");
        }
        engine::debug::configure_logging({argc == 3, argc == 3 ? argv[2] : ""});
        test_empty_midi_has_header_and_tempo_track();
        test_multi_track_notes_and_drums();
        test_overlap_is_trimmed_before_serializing();
        test_reader_roundtrip();
        test_reader_running_status_and_overlapping_notes();
        test_reader_keeps_tick_timing_across_tempo_changes();
        test_reader_skips_escaped_sysex();
        test_reader_rejects_malformed_data();
        engine::debug::log_message("validation.status=passed");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    } catch (const std::exception & ex) {
        std::cerr << "midi_file_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "midi_file_test passed\n";
    return 0;
}
