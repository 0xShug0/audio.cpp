#include "engine/models/sheetsage/processing.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        // Expected spellings from the Python notation implementation.
        const std::vector<std::array<std::string, 4>> cases = {
            {"A#:major", "A#:maj/3", "Bb", "A#/C##"},
            {"D#:major", "D#:maj7/7", "Eb", "D#maj7/C##"},
            {"G#:major", "G:min7/5", "Ab", "Gm7/D"},
            {"G:minor", "G:min/b3", "Gm", "Gm/Bb"},
            {"A#:minor", "A#:maj/5", "A#m", "A#/E#"},
            {"C:major", "C:maj/2", "C", "C/D"},
            {"G:major", "G:maj", "G", "G"},
        };
        for (const auto & test : cases) {
            std::vector<engine::models::sheetsage::SheetSage2Event> events(9);
            for (size_t i = 0; i < events.size(); ++i) {
                auto & event = events[i];
                event.time = static_cast<float>(i) * 0.5F;
                event.subbeat = event.source_subbeat = event.global_subbeat = static_cast<int64_t>(i) * 8;
                event.timestamp = event.time;
                event.meter = std::make_pair(4, 4);
                event.eighth_position = static_cast<int64_t>(i % 4) * 2;
            }
            events[0].key = test[0];
            events[0].chord = test[1];
            events[0].notes.push_back({70, 0, 3, 4});
            events[0].note_end_times.push_back(0.5F);
            const auto abc = engine::models::sheetsage::events_to_abc(events, 4.0);
            if (abc.find("K:" + test[2] + "\n") == std::string::npos ||
                abc.find("\"" + test[3] + "\"") == std::string::npos) {
                throw std::runtime_error("notation mismatch for " + test[0] + " " + test[1] + "\n" + abc);
            }
        }
        std::cout << "PASS SheetSage2 key signatures and chord inversions\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
