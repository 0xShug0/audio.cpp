#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::micro_wake_word {

// Stateful TensorFlow Lite Micro speech frontend used by microWakeWord.
// One 160-sample hop may produce one 40-value feature row after the initial
// 30 ms window has filled. Values include the upstream 0.0390625 scaling.
class MicroSpeechFrontend {
public:
    MicroSpeechFrontend();
    ~MicroSpeechFrontend();

    MicroSpeechFrontend(const MicroSpeechFrontend &) = delete;
    MicroSpeechFrontend & operator=(const MicroSpeechFrontend &) = delete;

    void reset();
    std::vector<float> process_hop(const int16_t * samples, size_t count);
    std::vector<float> compute(const std::vector<float> & samples);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::micro_wake_word
