#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/speech_encoders/rope_transformer_encoder_runtime.h"
#include "engine/models/dettato/runtime.h"
#include "grouping_decode.h"
#include <array>
#include <unordered_map>

namespace engine::models::dettato::detail {
struct HarmonicFeatures;
struct WindowTranscription;
struct NoteThresholds;
struct GroupingFeatures {
    std::vector<float> notes; // N, 12, sorted by onset/pitch.
    std::vector<ggml_fp16_t> taps; // N, 3, 368, matching collate's F16 storage.
};
GroupingFeatures collect_grouping_features(const WindowTranscription & transcription,const std::vector<Note> & notes,
    const NoteThresholds & thresholds);
struct GrouperInputs {
    std::vector<int32_t> pitch;
    std::vector<float> scalars, harmonics, profile, lanes, taps, cosine, sine;
};
GrouperInputs make_grouper_inputs(const HarmonicFeatures & features,const std::vector<float> & notes,
    const std::vector<float> & taps,const std::vector<float> & harmonic_prior,const std::vector<float> & lane_reads);

// Lane CNN, note-token projections, shared RoPE encoder and grouping heads.
// Fixed-shape lane CNN and note-count-dependent encoder retain separate
// session-owned plans, so changing note count does not rebuild the lane CNN.
class GrouperNetwork {
  public:
    GrouperNetwork(const assets::TensorSource &source, const core::ExecutionContext &execution);
    // Instrument logits, normalized track embeddings, keep logits, note states,
    // instrument probabilities. Softmax stays in the neural graph.
    std::array<std::vector<float>, 5> run(const GrouperInputs &inputs);
    // Model's lane CNN and per-note onset/mean reads, before lanes.read.
    std::vector<float> lane_features(const HarmonicFeatures & features,const std::vector<float> & notes);

  private:
    const core::ExecutionContext &execution_;
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> weights_;
    modules::RopeTransformerEncoderWeights encoder_weights_;
    std::unique_ptr<core::PreparedGraph> graph_, lane_graph_;
    std::array<core::TensorValue, 8> inputs_;
    std::array<core::TensorValue, 5> outputs_;
    core::TensorValue lane_input_, raster_input_, lane_output_;
    std::vector<std::pair<core::TensorValue,std::vector<float>>> constants_;
    std::unordered_map<std::string,std::vector<float>> relative_bias_;
    int64_t notes_ = 0;
    int64_t frames_ = 0;
    void prepare(int64_t notes);
    void prepare_lanes(int64_t frames);
    void build_lanes();
    core::TensorValue linear(core::ModuleBuildContext &, const core::TensorValue &, const std::string &);
};

class WindowGrouper {
  public:
    WindowGrouper(const assets::TensorSource & source, const io::json::Value & config,
        const core::ExecutionContext & execution);
    ~WindowGrouper();
    GroupingPrediction predict(const std::vector<float> & audio16, const GroupingFeatures & features);
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::models::dettato::detail
