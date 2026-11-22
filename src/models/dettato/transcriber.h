#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/weight_binding.h"
#include <array>
#include <unordered_map>

namespace engine::models::dettato::detail {

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
    LaneTranscriber(const assets::TensorSource &source, const core::ExecutionContext &execution);
    LaneOutputs run(const std::vector<float> &hcqt, int64_t batch, int64_t frames);
    // Borrowed until the next call or destruction. Device consumers finish
    // before the next execution and need no intermediate host readback.
    std::array<core::TensorValue, 3> run_device(const std::vector<float> &hcqt, int64_t batch, int64_t frames);

  private:
    static constexpr int64_t kAttentionHeads = 8;
    const core::ExecutionContext &execution_;
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> weights_;
    std::unordered_map<std::string, std::vector<float>> host_relative_bias_;
    std::vector<std::pair<core::TensorValue, std::vector<float>>> constants_;
    std::unique_ptr<core::PreparedGraph> graph_;
    core::TensorValue input_, logits_, stem_, hidden_, rotary_positions_;
    int64_t batch_ = 0, frames_ = 0;
    void prepare(int64_t batch, int64_t frames);
    core::TensorValue conv(core::ModuleBuildContext &, const core::TensorValue &, const std::string &, int stride = 1,
                           int dilation = 1, bool depthwise = false);
    core::TensorValue norm(core::ModuleBuildContext &, const core::TensorValue &, const std::string &);
    core::TensorValue linear(core::ModuleBuildContext &, const core::TensorValue &, const std::string &, bool use_bias = true);
    core::TensorValue lane(core::ModuleBuildContext &, const core::TensorValue &, const std::string &, int dilation);
    core::TensorValue attention(core::ModuleBuildContext &, const core::TensorValue &, const std::string &, bool time);
    core::TensorValue recurrent(core::ModuleBuildContext &, const core::TensorValue &, const std::string &);
};
} // namespace engine::models::dettato::detail
