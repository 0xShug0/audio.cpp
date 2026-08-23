#pragma once

#include <stdexcept>
#include <string>

namespace engine::runtime {

// A request the device cannot serve AT THIS SIZE -- e.g. a transcription
// prompt plus audio whose prefill graph does not fit in VRAM. Distinct from a
// genuine internal fault: the caller can fix it by sending less, so servers
// should surface it as a client error rather than an opaque 500.
class CapacityError : public std::runtime_error {
public:
    explicit CapacityError(const std::string & message) : std::runtime_error(message) {}
};

}  // namespace engine::runtime
