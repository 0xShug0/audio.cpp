#pragma once

#include "decode.h"
#include "frontend.h"
#include "transcriber.h"

namespace engine::models::dettato::detail {
struct WindowTranscription {
    int64_t frames = 0;
    std::array<std::vector<float>, 6> maps, grouping_maps;
    // Time, piano key, channel. Match the upstream F16 overlap accumulation.
    // Session-owned readback storage, overwritten by the next run(). The
    // 4-second schedule uses the same hidden average for both consumers.
    std::shared_ptr<std::vector<ggml_fp16_t>> hidden = std::make_shared<std::vector<ggml_fp16_t>>();
    std::shared_ptr<std::vector<ggml_fp16_t>> grouping_hidden = std::make_shared<std::vector<ggml_fp16_t>>();
    std::shared_ptr<std::vector<ggml_fp16_t>> grouping_stem = std::make_shared<std::vector<ggml_fp16_t>>();
};

class WindowTranscriber {
  public:
    WindowTranscriber(const assets::TensorSource &source, const core::ExecutionContext &execution, int window_hop_sec = 2);
    ~WindowTranscriber();
    void run_device(const std::vector<float> &audio16k);
    WindowTranscription readback();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::models::dettato::detail
