#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/midi/midi_file.h"
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::rosvot {

struct EncoderConfig {
    int64_t hidden_size = 256, mel_bins = 40, down_layers = 4, conformer_layers = 2;
    int64_t conformer_kernel = 9, pitch_heads = 4, note_num = 85, note_start = 30;
    float boundary_temperature = 0.2f, pitch_temperature = 0.01f;
    int boundary_gap = 17, reference_gap = 8;
};

struct RosvotAssets {
    assets::ResourceBundle resources;
    EncoderConfig notes, words;
    std::shared_ptr<const assets::TensorSource> note_weights, word_weights, pitch_weights;
};
std::shared_ptr<const RosvotAssets> load_rosvot_assets(const std::filesystem::path & path);

struct Transcription {
    std::vector<midi::MidiNote> notes;  // Includes rests (pitch 0) in the JSON timeline.
    std::vector<int64_t> note_to_word;
    std::vector<double> word_durations;
};

// Host decision between frame and note graphs is part of the model algorithm.
// All model graphs and weights are owned by this session runtime and reused.
class RosvotRuntime {
public:
    RosvotRuntime(std::shared_ptr<const RosvotAssets> assets, const core::ExecutionContext & execution);
    ~RosvotRuntime();
    Transcription transcribe(const std::vector<float> & mono24k, float note_threshold,
        float word_threshold, const std::vector<double> & word_durations = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::rosvot
