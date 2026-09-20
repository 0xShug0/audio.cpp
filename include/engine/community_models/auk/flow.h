#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <map>
#include <vector>

namespace engine::models::auk {

struct FlowEmbeddings {
    std::vector<float> time;
    std::vector<float> text;
    std::vector<float> audio;
    std::vector<float> first_block_audio;
    std::vector<float> first_block_text;
    std::vector<float> double_stream_audio;
    std::vector<float> double_stream_text;
    std::vector<float> single_stream;
    std::vector<float> velocity;
    std::vector<float> guided_velocity;
    std::map<std::string, std::vector<float>> first_block_boundaries;
};

class FlowRuntime {
public:
    FlowRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
                int64_t audio_frames, int64_t text_tokens, bool cfg = false, bool flash_attention = true,
                bool capture_intermediates = false, int64_t reference_frames = 0, int64_t valid_reference_frames = 0,
                int schedule_steps = 32);
    ~FlowRuntime();
    FlowRuntime(const FlowRuntime &) = delete;
    FlowRuntime & operator=(const FlowRuntime &) = delete;

    void prepare(int64_t audio_frames, int64_t text_tokens);
    void prepare(int64_t audio_frames, int64_t text_tokens, bool cfg,
                 int64_t reference_frames = 0, int64_t valid_reference_frames = 0, int schedule_steps = 32);

    FlowEmbeddings embed(const std::vector<float> & audio, const std::vector<float> & text, float time,
                         float cfg_strength = 2.0F, const std::vector<float> & reference = {});
    std::vector<float> sample(const std::vector<float> & initial_noise, const std::vector<float> & text,
                              int steps = 32, float sway = -1.0F, float cfg_strength = 2.0F,
                              const std::vector<float> & reference = {}, bool distilled_flash = false);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::auk
