#pragma once

#include "engine/models/muscriptor/types.h"

#include <cstddef>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::muscriptor {

struct MuScriptorTokenEvent {
    std::string type;
    int value = 0;
};

class MuScriptorTokenizer {
public:
    MuScriptorTokenizer();

    int32_t eos_id() const noexcept;
    const std::vector<MuScriptorTokenEvent> & vocab() const noexcept;

    std::vector<int32_t> condition_instrument_ids(const std::string & csv) const;
    std::vector<int32_t> forbidden_token_ids(const std::string & csv) const;
    std::vector<int32_t> tie_section_token_ids(const std::set<std::pair<int, int>> & open_notes) const;
    std::set<std::pair<int, int>> open_note_keys(const std::vector<MuScriptorGeneratedChunk> & chunks) const;
    std::string instrument_for_program(int program) const;
    std::vector<MuScriptorEvent> decode_chunks(const std::vector<MuScriptorGeneratedChunk> & chunks) const;

private:
    int32_t token_id(const std::string & type, int value) const;
    std::vector<MuScriptorTokenEvent> vocab_;
    std::unordered_map<std::string, int> group_name_to_id_;
    std::unordered_map<int, std::vector<int>> group_programs_;
    std::unordered_map<int, std::string> program_to_name_;
};

std::string muscriptor_events_to_json(const std::vector<MuScriptorEvent> & events);
std::string muscriptor_event_to_json(const MuScriptorEvent & event);
std::vector<std::byte> muscriptor_events_to_midi_bytes(const std::vector<MuScriptorEvent> & events);

}  // namespace engine::models::muscriptor
