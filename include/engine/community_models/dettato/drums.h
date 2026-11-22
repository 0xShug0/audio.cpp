#pragma once

#include "engine/community_models/dettato/assets.h"

#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/community_models/dettato/pipeline.h"
#include <unordered_map>

namespace engine::community_models::dettato::detail {
struct DrumProbabilities {
    double router = 0;
    std::vector<float> kit, kit_velocity, orchestral, timpani, velocity;
};
Track decode_drum_hits(const DrumProbabilities &probabilities, const io::json::Value &config);

// Shared by the kit and each orchestral expert: CNN over time/bands, followed
// by the short bidirectional band GRU and projection to frame embeddings.
class DrumEmbedder {
  public:
    DrumEmbedder(DrumEmbedderWeights weights, const io::json::Value &config, const core::ExecutionContext &execution,
                 core::GraphWorkspace *workspace = nullptr);
    // Input [frames,160]; output [frames/2,width]. Long inputs use upstream
    // context windows; one session-owned graph set caches their shapes.
    // Build/register before the pipeline finalizes its shared workspace.
    void prepare(int64_t frames, int64_t long_above = 8400);
    std::vector<float> run(const std::vector<float> &mel, int64_t long_above = 8400);

  private:
    const core::ExecutionContext &execution_;
    core::GraphWorkspace *workspace_;
    DrumEmbedderWeights weights_;
    std::string activation_;
    int64_t width_, band_hidden_;
    struct Ports {
        core::TensorValue input, output, initial;
    };
    std::vector<int64_t> shapes_;
    std::vector<Ports> ports_;
    std::vector<float> window_;
    std::unique_ptr<core::PreparedGraph> graph_;
    void prepare(const std::vector<int64_t> &shapes);
};

// Recording-level temporal convolution, recurrent/attention blocks and family
// heads. Persistent outputs survive reuse of the pipeline's shared scratch.
class DrumTrunk {
  public:
    DrumTrunk(DrumTrunkWeights weights, const io::json::Value &config, const core::ExecutionContext &execution,
              core::GraphWorkspace *workspace = nullptr);
    void prepare(int64_t frames);
    std::array<core::TensorValue, 3> run(const std::vector<float> &embeddings);

  private:
    const core::ExecutionContext &execution_;
    core::GraphWorkspace *workspace_;
    std::unique_ptr<core::ExecutionContext> host_execution_;
    DrumTrunkWeights weights_;
    std::vector<std::vector<int>> groups_;
    std::unordered_map<int64_t, core::TensorValue> zero_states_;
    int64_t width_, family_hidden_, frames_ = 0;
    std::unique_ptr<core::PreparedGraph> graph_;
    core::TensorValue input_, indices_;
    std::array<core::TensorValue, 3> outputs_;
};

class OrchestralDrumExpert {
  public:
    OrchestralDrumExpert(DrumAdapterWeights weights, const io::json::Value &config,
                         const core::ExecutionContext &execution, core::GraphWorkspace *workspace = nullptr);
    void prepare(int64_t frames);
    // Raw base/drum/timpani/velocity/router logits, in upstream order.
    std::array<core::TensorValue, 5> run(const std::vector<float> &features);

  private:
    const core::ExecutionContext &execution_;
    core::GraphWorkspace *workspace_;
    std::unique_ptr<core::ExecutionContext> host_execution_;
    DrumEmbedder embedder_;
    DrumTrunk trunk_;
    DrumAdapterWeights weights_;
    int64_t hidden_, layers_, heads_, base_width_, frames_ = 0;
    std::unique_ptr<core::PreparedGraph> graph_;
    core::TensorValue cq_input_, hidden_input_, base_input_, initial_;
    std::array<core::TensorValue, 3> outputs_;
};

struct DrumResult {
    Track track;
    double router = 0;
    std::string model;
};
class DrumTranscriber {
  public:
    DrumTranscriber(DrumWeights weights, const io::json::Value &config, const core::ExecutionContext &execution);
    ~DrumTranscriber();
    DrumProbabilities probabilities(const std::vector<float> &features);
    DrumResult transcribe_features(const std::vector<float> &features);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::community_models::dettato::detail
