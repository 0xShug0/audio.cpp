#pragma once

#include <stdexcept>
#include <string>

namespace engine::runtime {

// A request has a problem that the client can fix by changing request
// parameters. Distinct from a genuine internal fault: so servers should
// surface it as a client error rather than an opaque 500.
class RequestValidationError : public std::runtime_error {
public:
    explicit RequestValidationError(const std::string & message) : std::runtime_error(message) {}
};

// A request the device cannot serve AT THIS SIZE -- e.g. a transcription
// prompt plus audio whose prefill graph does not fit in VRAM.
class CapacityError : public RequestValidationError {
public:
    explicit CapacityError(const std::string & message) : RequestValidationError(message) {}
};

}  // namespace engine::runtime
