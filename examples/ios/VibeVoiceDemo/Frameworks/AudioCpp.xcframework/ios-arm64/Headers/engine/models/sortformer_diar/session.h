#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/graph.h"
#include "engine/models/sortformer_diar/types.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::models::sortformer_diar {

class SortformerDiarSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SortformerDiarSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SortformerAssets> assets);
    ~SortformerDiarSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::MappedGraphCapacityAdapter make_graph_capacity_adapter();
    int64_t base_graph_capacity_samples() const;
    std::vector<int64_t> prepared_graph_capacities() const;
    void prepare_graph_capacity(int64_t capacity);
    runtime::TaskResult run_offline_diarization(
        const runtime::AudioBuffer & audio,
        const SortformerPostprocessConfig & config);

    runtime::TaskSpec task_;
    std::shared_ptr<const SortformerAssets> assets_;
    std::shared_ptr<const SortformerDiarWeights> weights_;
    SortformerPostprocessConfig default_postprocess_;
    size_t graph_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType conv_weight_storage_type_ = assets::TensorStorageType::Native;
    runtime::GraphCapacityController graph_capacity_controller_;
    SortformerFixedContextContract base_context_;
    std::unordered_map<int64_t, SortformerFixedContextContract> prepared_contexts_;
    std::unordered_map<int64_t, std::unique_ptr<SortformerInferenceGraph>> inference_graphs_;
    std::vector<float> probabilities_;
};

}  // namespace engine::models::sortformer_diar
