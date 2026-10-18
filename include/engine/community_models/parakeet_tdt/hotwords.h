#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::parakeet_tdt {

// Phonon-2's bonus-only, piece-level Aho-Corasick decoder policy. Ported from
// fermion-research 0.2.11 (Apache-2.0), fermion/_speech/hotwords.py.
struct PhononHotword {
    std::string word;
    std::vector<std::string> spoken;
};

std::vector<PhononHotword> parse_phonon_hotwords(const std::string & value);

class PhononHotwordAutomaton {
public:
    PhononHotwordAutomaton(const std::vector<PhononHotword> & words,
                          const std::vector<std::string> & vocabulary,
                          int32_t blank, float strength);
    bool empty() const { return nodes_.size() <= 1; }
    int32_t select(const std::vector<float> & scores, int32_t state) const;
    int32_t step(int32_t state, int32_t token) const;
    float bonus(int32_t state, int32_t token) const;
    size_t states() const { return nodes_.size(); }

private:
    struct Node {
        std::unordered_map<int32_t, int32_t> children;
        int32_t fail = 0;
        int32_t depth = 0;
    };
    std::vector<Node> nodes_{1};
    std::vector<int32_t> active_tokens_;
    int32_t blank_;
    float strength_;
};

} // namespace engine::community_models::parakeet_tdt
