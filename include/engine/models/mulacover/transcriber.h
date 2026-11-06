#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <memory>
#include <vector>

namespace engine::models::mulacover {

struct TranscribedNote {
    double onset_seconds = 0.0;
    int pitch = 0;
    double duration = 0.0;
    int program = 0;
    bool is_drum = false;
};

// Fixed YourMT3 YPTF.MoE+Multi (noPS) inference checkpoint. One instance belongs
// to one session, retains both graphs and all KV caches, and is not reentrant.
// TensorSource names are the original inference.py state_dict names, without
// a component prefix. The converter writes these names unchanged.
class Transcriber {
public:
    Transcriber(std::shared_ptr<const assets::TensorSource> source,
                const core::BackendConfig & backend,
                assets::TensorStorageType storage_type = assets::TensorStorageType::Native);
    ~Transcriber();
    Transcriber(const Transcriber &) = delete;
    Transcriber & operator=(const Transcriber &) = delete;

    // Float mono at any positive sample rate. Returns all 13 channels in
    // seconds, including program 100 (lead singing) and program 128 (drums).
    // Selecting tracks, tempo quantization, and chord inference belong to caller.
    std::vector<TranscribedNote> transcribe(const std::vector<float> & mono, int sample_rate);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
