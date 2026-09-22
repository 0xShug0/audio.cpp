#pragma once

#include <cstdint>
#include <string>

namespace engine::models::vieneu_v3_turbo {

// Upper bound on generated frames for one chunk of SEA-G2P phonemes, mirroring
// `vieneu_utils.core_utils.max_expected_frames` in the Python package: it stops
// runaway generation ("babble") when the model misses EOS. The bound is
// slack + 2 frames per phoneme, tightened for chunks of at most four syllables and
// for cue-only chunks (an emotion tag with no speech).
int64_t max_expected_frames(const std::string & phonemes);

// Estimated syllable count of a SEA-G2P phoneme string (Vietnamese: one per word;
// English words: one per vowel cluster / stress mark).
int64_t syllable_count(const std::string & phonemes);

}  // namespace engine::models::vieneu_v3_turbo
