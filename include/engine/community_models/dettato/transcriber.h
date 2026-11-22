#pragma once

#include "engine/community_models/dettato/assets.h"

#include "engine/framework/core/prepared_graph.h"
#include "engine/community_models/dettato/frontend.h"
#include "engine/community_models/dettato/note_repair.h"
#include <array>

namespace engine::community_models::dettato::detail {

struct LaneOutputs {
    std::vector<float> logits; // B, 6, T, 88
    std::vector<float> stem;   // B, C, T, 96
    std::vector<float> hidden; // B, C, T, 88
    int64_t batch = 0, frames = 0;
};

// Session-owned transcriber, including its derived pitch operators and one
// bounded graph shape. The normal window runner always uses the same shape.
class LaneTranscriber {
  public:
    LaneTranscriber(TranscriberWeights weights, const core::ExecutionContext &execution);
    ~LaneTranscriber();
    LaneOutputs run(const std::vector<float> &hcqt, int64_t batch, int64_t frames);
    // Borrowed until the next call or destruction. Device consumers finish
    // before the next execution and need no intermediate host readback.
    std::array<core::TensorValue, 3> run_device(const std::vector<float> &hcqt, int64_t batch, int64_t frames);

  private:
    static constexpr int64_t kAttentionHeads = 8;
    const core::ExecutionContext &execution_;
    TranscriberWeights weights_;
    std::vector<std::pair<core::TensorValue, std::vector<float>>> constants_;
    std::unique_ptr<core::PreparedGraph> graph_;
    ggml_backend_set_graph_fusions_t set_fusions_ = nullptr;
    core::TensorValue input_, logits_, stem_, hidden_, rotary_positions_;
    int64_t batch_ = 0, frames_ = 0;
    void prepare(int64_t batch, int64_t frames);
    core::TensorValue conv(core::ModuleBuildContext &, const core::TensorValue &, const modules::Conv2dWeights &,
                           int stride = 1, int dilation = 1, bool depthwise = false, int frequency_stride = 1);
    core::TensorValue norm(core::ModuleBuildContext &, const core::TensorValue &, const TranscriberNormWeights &);
    core::TensorValue linear(core::ModuleBuildContext &, const core::TensorValue &, const modules::LinearWeights &,
                             bool use_bias = true);
    core::TensorValue lane(core::ModuleBuildContext &, const core::TensorValue &, const LaneBlockWeights &,
                           int dilation);
    core::TensorValue attention(core::ModuleBuildContext &, const core::TensorValue &, const LaneAttentionWeights &,
                                bool time);
    core::TensorValue recurrent(core::ModuleBuildContext &, const core::TensorValue &, const LaneRecurrentWeights &);
};

struct WindowTranscription {
    int64_t frames = 0;
    bool compact_grouping = false;
    std::vector<int32_t> grouping_rows; // Sorted flattened time/key rows when compact.
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
    WindowTranscriber(TranscriberWeights weights, const core::ExecutionContext &execution, int window_hop_sec = 2);
    ~WindowTranscriber();
    // The fixed-size device ring normalizes finalized frames. Grouping readback
    // selects the repaired notes from the session-owned host feature storage.
    void run_device(const std::vector<float> &audio16k, bool full_grouping = true);
    WindowTranscription readback();
    void readback_grouping(WindowTranscription &transcription, const std::vector<Note> &notes);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::community_models::dettato::detail
