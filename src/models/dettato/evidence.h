#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/models/dettato/runtime.h"
#include <array>
#include <unordered_map>

namespace engine::models::dettato::detail {
struct WindowTranscription;
std::vector<float> extract_note_evidence(const WindowTranscription &transcription, const std::vector<Note> &notes, size_t threads = 1);
struct EvidenceInputs {
    std::vector<int32_t> instrument, pitch, velocity_bin;
    std::vector<float> continuous;        // N, 110
    std::vector<float> evidence;          // N, 769
    std::vector<float> cosine, sine;      // N, 14; continuous onset/pitch rotary axes
    std::vector<float> onsets, durations; // Quantized note times on the window axis.
};
EvidenceInputs make_evidence_inputs(const std::vector<Note> &notes, const std::vector<float> &evidence,
                                    float window_start);

class EvidenceNetwork {
  public:
    EvidenceNetwork(const assets::TensorSource &source, const core::ExecutionContext &execution);
    // Keep, onset correction, duration ratio, insertion slot 0, insertion slot 1.
    std::array<std::vector<float>, 5> run(const EvidenceInputs &inputs, bool probabilities = false);

  private:
    const core::ExecutionContext &execution_;
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> weights_;
    std::unique_ptr<core::PreparedGraph> graph_;
    std::array<core::TensorValue, 7> inputs_;
    std::array<core::TensorValue, 5> outputs_;
    std::array<core::TensorValue, 5> probabilities_;
    int64_t notes_ = 0;
    void prepare(int64_t notes);
    core::TensorValue linear(core::ModuleBuildContext &, const core::TensorValue &, const std::string &);
    core::TensorValue rms(core::ModuleBuildContext &, const core::TensorValue &, const std::string &);
};
std::vector<Note> evidence_candidates(const EvidenceInputs &inputs,
                                      const std::array<std::vector<float>, 5> &probabilities);
struct RepairResult {
    std::vector<Note> notes;
    size_t dropped = 0, inserted = 0;
};
RepairResult repair_notes(EvidenceNetwork &network, const std::vector<Note> &notes, const std::vector<float> &evidence,
                          float keep_threshold = 0.4f, float insertion_threshold = 0.6f);
} // namespace engine::models::dettato::detail
