#pragma once

#include "engine/community_models/dettato/assets.h"

#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/community_models/dettato/pipeline.h"
#include <array>

namespace engine::community_models::dettato::detail {

struct NoteThresholds {
    float onset = 0.25f, second_onset = 0.3f, offset = 0.3f, frame = 0.4f;
    int low_frames = 2, onset_nms = 3;
};

// Six probability maps in upstream head order, each [frames, 88].
std::vector<Note> decode_notes(const std::array<std::vector<float>, 6> &maps, int64_t frames, double hop,
                               const NoteThresholds &thresholds);

struct WindowTranscription;
std::vector<float> extract_note_evidence(const WindowTranscription &transcription, const std::vector<Note> &notes,
                                         size_t threads = 1);
struct NoteRepairInputs {
    std::vector<int32_t> instrument, pitch, velocity_bin;
    std::vector<float> continuous;        // N, 110
    std::vector<float> evidence;          // N, 769
    std::vector<float> cosine, sine;      // N, 14; continuous onset/pitch rotary axes
    std::vector<float> onsets, durations; // Quantized note times on the window axis.
};
NoteRepairInputs make_note_repair_inputs(const std::vector<Note> &notes, const std::vector<float> &evidence,
                                         float window_start);

class NoteRepairNetwork {
  public:
    NoteRepairNetwork(NoteRepairWeights weights, const core::ExecutionContext &execution);
    // Keep, onset correction, duration ratio, insertion slot 0, insertion slot 1.
    std::array<std::vector<float>, 5> run(const NoteRepairInputs &inputs, bool probabilities = false);

  private:
    const core::ExecutionContext &execution_;
    NoteRepairWeights weights_;
    struct GraphState {
        std::unique_ptr<core::PreparedGraph> graph;
        std::array<core::TensorValue, 7> inputs;
        std::array<core::TensorValue, 5> outputs, probabilities;
    };
    runtime::CacheSlots<int64_t, std::unique_ptr<GraphState>> graphs_;
    GraphState &prepare(int64_t notes);
    core::TensorValue linear(core::ModuleBuildContext &, const core::TensorValue &, const modules::LinearWeights &);
    core::TensorValue rms(core::ModuleBuildContext &, const core::TensorValue &, const modules::NormWeights &);
};
std::vector<Note> decode_repair_candidates(const NoteRepairInputs &inputs,
                                           const std::array<std::vector<float>, 5> &probabilities);
struct RepairResult {
    std::vector<Note> notes;
    size_t dropped = 0, inserted = 0;
};
RepairResult repair_notes(NoteRepairNetwork &network, const std::vector<Note> &notes,
                          const std::vector<float> &evidence, float keep_threshold = 0.4f,
                          float insertion_threshold = 0.6f);
} // namespace engine::community_models::dettato::detail
