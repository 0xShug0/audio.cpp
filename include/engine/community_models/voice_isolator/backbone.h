#pragma once

#include <cstdint>

namespace engine::models::voice_isolator {

// The network between band packing and the spectral change: the band encoders,
// the dual-path blocks and the band heads. An offline backbone sees a whole
// chunk per run; a streaming one carries its state from one run to the next.
class Backbone {
public:
    virtual ~Backbone() = default;

    // packed: [bands][frames][encoder_inputs]; raw: [bands][frames][decoder_outputs].
    virtual void run(const float * packed, int64_t frames, float * raw) = 0;
    // Forgets a stream's past; an offline backbone holds none.
    virtual void reset() = 0;
};

}  // namespace engine::models::voice_isolator
