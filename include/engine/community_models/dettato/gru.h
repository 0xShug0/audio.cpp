#pragma once

#include "engine/framework/core/module.h"

namespace engine::community_models::dettato::detail {

enum class GRURecurrenceLowering { Unrolled, HostRecurrence, DeviceScan, DeviceCell };
struct BidirectionalGRUConfig {
    int64_t input_size = 0;
    int64_t hidden_size = 0;
    // Unrolled preserves the original graph. HostRecurrence needs a CPU backend.
    // DeviceCell keeps GGML recurrent GEMMs and fuses cell math; DeviceScan runs
    // F32 recurrence in one device operation. Input projection remains batched.
    GRURecurrenceLowering lowering = GRURecurrenceLowering::Unrolled;
};

// PyTorch reset-after GRU, with gate order reset/update/new. The two biases
// remain separate: the reset gate multiplies the recurrent new-gate bias too.
struct GRUWeights {
    core::TensorValue weight_ih;
    core::TensorValue weight_hh;
    core::TensorValue bias_ih;
    core::TensorValue bias_hh;
};

struct BidirectionalGRUOutputs {
    core::TensorValue sequence;
    core::TensorValue forward_hidden;
    core::TensorValue reverse_hidden;
};

// Graph composition only. Input: [time, batch, features]; output: [time, batch,
// 2*hidden]. Each initial/final state is [batch, hidden]. Both output directions
// are chronological.
// The caller owns the graph, weights, initial state and allocation lifecycle.
class BidirectionalGRU {
  public:
    explicit BidirectionalGRU(BidirectionalGRUConfig config);
    const BidirectionalGRUConfig &config() const noexcept {
        return config_;
    }
    BidirectionalGRUOutputs build(core::ModuleBuildContext &ctx, const core::TensorValue &input,
                                  const core::TensorValue &forward_initial, const core::TensorValue &reverse_initial,
                                  const GRUWeights &forward_weights, const GRUWeights &reverse_weights) const;

  private:
    BidirectionalGRUConfig config_;
};

} // namespace engine::community_models::dettato::detail
