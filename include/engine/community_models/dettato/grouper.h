#pragma once

#include "engine/community_models/dettato/assets.h"

#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/speech_encoders/rope_transformer_encoder_runtime.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/community_models/dettato/pipeline.h"
#include <array>
#include <vector>

namespace engine::community_models::dettato::detail {

struct GroupingWindow {
    double start = 0;
    std::vector<int> notes;
    std::vector<float> embeddings;    // notes x 32
    std::vector<float> probabilities; // notes x 73
};
struct GroupingPrediction {
    std::vector<GroupingWindow> windows;
    std::vector<float> embeddings, probabilities, hidden; // N x 32/73/176
    double window_seconds = 24;
};

// Tracklet clustering and overlap matching from the official grouping decoder.
// Uncovered notes retain -1. Window order is chronological, stable on ties.
std::vector<int> chain_grouping_windows(const std::vector<GroupingWindow> &windows, const std::vector<double> &onsets,
                                        double window_seconds = 24, double threshold = .1,
                                        float probability_weight = .5f);

// Merge song-wide tracklets by size-weighted average linkage. Embeddings are
// already unit vectors; instrument distributions are normalized here.
std::vector<int> merge_grouping_tracklets(const std::vector<int> &tracklets, const std::vector<float> &embeddings,
                                          const std::vector<float> &probabilities, double threshold = .4,
                                          double probability_weight = .5);

class GroupingDecoder {
  public:
    GroupingDecoder(GroupingDecoderWeights weights, const io::json::Value &config);
    std::vector<Track> decode(const GroupingPrediction &prediction, const std::vector<Note> &notes,
                              double time_ratio = 1) const;

  private:
    std::vector<TrackClassifierLayer> head_;
    std::vector<float> credit_;
    std::vector<std::string> names_;
    std::vector<int> programs_;
    double threshold_, merge_threshold_, lambda_, alpha_, absorb_, absorb_fraction_;
    int minimum_notes_, refine_;
    bool expected_credit_, merge_;
};

struct HarmonicFeatures;
struct WindowTranscription;
struct NoteThresholds;
struct GroupingFeatures {
    std::vector<float> notes;      // N, 12, sorted by onset/pitch.
    std::vector<ggml_fp16_t> taps; // N, 3, 368, matching collate's F16 storage.
};
GroupingFeatures collect_grouping_features(const WindowTranscription &transcription, const std::vector<Note> &notes,
                                           const NoteThresholds &thresholds);
struct GrouperInputs {
    std::vector<int32_t> pitch;
    std::vector<float> scalars, harmonics, profile, lanes, taps, cosine, sine;
};
GrouperInputs make_grouper_inputs(const HarmonicFeatures &features, const std::vector<float> &notes,
                                  const std::vector<float> &taps, const std::vector<float> &harmonic_prior,
                                  const std::vector<float> &lane_reads);

// Lane CNN, note-token projections, shared RoPE encoder and grouping heads.
// Fixed-shape lane CNN and note-count-dependent encoder retain separate
// session-owned plans, so changing note count does not rebuild the lane CNN.
class GrouperNetwork {
  public:
    GrouperNetwork(GrouperWeights weights, const core::ExecutionContext &execution);
    // Instrument logits, normalized track embeddings, keep logits, note states,
    // instrument probabilities. Softmax stays in the neural graph.
    std::array<std::vector<float>, 5> run(const GrouperInputs &inputs);
    // Model's lane CNN and per-note onset/mean reads, before lanes.read.
    std::vector<float> lane_features(const HarmonicFeatures &features, const std::vector<float> &notes);

  private:
    const core::ExecutionContext &execution_;
    GrouperWeights weights_;
    struct GraphState {
        std::unique_ptr<core::PreparedGraph> graph;
        std::array<core::TensorValue, 8> inputs;
        std::array<core::TensorValue, 5> outputs;
    };
    runtime::CacheSlots<int64_t, std::unique_ptr<GraphState>> graphs_;
    std::unique_ptr<core::PreparedGraph> lane_graph_;
    core::TensorValue lane_input_, raster_input_, lane_output_;
    std::vector<std::pair<core::TensorValue, std::vector<float>>> constants_;
    int64_t frames_ = 0;
    GraphState &prepare(int64_t notes);
    void prepare_lanes(int64_t frames);
    void build_lanes();
    core::TensorValue linear(core::ModuleBuildContext &, const core::TensorValue &, const modules::LinearWeights &);
};

class WindowGrouper {
  public:
    WindowGrouper(GrouperWeights weights, const io::json::Value &config, const core::ExecutionContext &execution);
    ~WindowGrouper();
    std::vector<Track> group(const std::vector<float> &audio16, const WindowTranscription &transcription,
                             const std::vector<Note> &notes, const NoteThresholds &thresholds, double time_ratio = 1);
    GroupingPrediction predict(const std::vector<float> &audio16, const GroupingFeatures &features);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::community_models::dettato::detail
