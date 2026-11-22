#pragma once

#include <vector>
#include "engine/models/dettato/runtime.h"

namespace engine::models::dettato::detail {
struct GroupingWindow {
    double start = 0;
    std::vector<int> notes;
    std::vector<float> embeddings; // notes x 32
    std::vector<float> probabilities; // notes x 73
};
struct GroupingPrediction {
    std::vector<GroupingWindow> windows;
    std::vector<float> embeddings, probabilities, hidden; // N x 32/73/176
    double window_seconds = 24;
};

// Tracklet clustering and overlap matching from the official grouping decoder.
// Uncovered notes retain -1. Window order is chronological, stable on ties.
std::vector<int> chain_grouping_windows(const std::vector<GroupingWindow> & windows,
    const std::vector<double> & onsets, double window_seconds = 24,
    double threshold = .1, float probability_weight = .5f);

// Merge song-wide tracklets by size-weighted average linkage. Embeddings are
// already unit vectors; instrument distributions are normalized here.
std::vector<int> merge_grouping_tracklets(const std::vector<int> & tracklets,
    const std::vector<float> & embeddings, const std::vector<float> & probabilities,
    double threshold = .4, double probability_weight = .5);

class GroupingDecoder {
  public:
    GroupingDecoder(const assets::TensorSource & source, const io::json::Value & config);
    std::vector<Track> decode(const GroupingPrediction & prediction, const std::vector<Note> & notes,
        double time_ratio = 1) const;
  private:
    struct DenseLayer { int input = 0, output = 0; std::vector<double> weight, bias; };
    std::vector<DenseLayer> head_;
    std::vector<float> credit_;
    std::vector<std::string> names_;
    std::vector<int> programs_;
    double threshold_, merge_threshold_, lambda_, alpha_, absorb_, absorb_fraction_;
    int minimum_notes_, refine_;
    bool expected_credit_, merge_;
};
} // namespace engine::models::dettato::detail
