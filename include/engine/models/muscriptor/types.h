#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::muscriptor {

struct MuScriptorConditioning {
    std::vector<float> values;
    int64_t batch = 1;
    int64_t steps = 0;
    int64_t dim = 0;
};

struct MuScriptorAudioChunk {
    std::vector<float> log_mel;
    std::vector<int32_t> mask;
};

struct MuScriptorGeneratedChunk {
    std::vector<int32_t> tokens;
    bool emitted_eos = false;
};

struct MuScriptorGenerationOptions {
    bool use_sampling = false;
    float temperature = 1.0F;
    float guidance_scale = 1.0F;
    int64_t batch_size = 1;
    int64_t num_beams = 1;
    bool prelude_forcing = true;
    uint64_t seed = 0;
};

struct MuScriptorNoteStart {
    int pitch = 0;
    double start_time = 0.0;
    int index = 0;
    std::string instrument;
};

struct MuScriptorNoteEnd {
    double end_time = 0.0;
    int start_event_index = 0;
};

struct MuScriptorEvent {
    enum class Kind {
        Start,
        End,
    };

    Kind kind = Kind::Start;
    MuScriptorNoteStart start;
    MuScriptorNoteEnd end;
};

}  // namespace engine::models::muscriptor
