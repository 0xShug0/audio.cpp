#pragma once

#include "engine/community_models/parakeet_tdt/frontend.h"
#include "engine/community_models/parakeet_tdt/predictor.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <string_view>

namespace engine::community_models::parakeet_tdt {

using SessionFactory = std::unique_ptr<runtime::IVoiceTaskSession> (*)(
    const runtime::TaskSpec &, const runtime::SessionOptions &,
    std::shared_ptr<const ParakeetTDTAssets>, std::shared_ptr<const model_spec::ModelContract>);

// Register family differences here; shared runtimes do not interpret variant labels.
struct ParakeetImplementation {
    SessionFactory offline;
    SessionFactory streaming;
    std::shared_ptr<const Predictor> (*load_predictor)(
        core::BackendWeightStore &, const ParakeetTDTAssets &, assets::TensorStorageType);
    runtime::GraphOptimizationOptions (*encoder_graph_options)(runtime::GraphOptimizationBackend);
    std::unique_ptr<WindowFrontend> (*window_frontend)(std::shared_ptr<const ParakeetTDTAssets>) = nullptr;
    int64_t input_prefix_samples = 0;
    bool hotwords = false;
    const char * hotword_name = "Phonon";
    bool invalid_request_errors = false;
    void (*window_timestamps)(std::vector<runtime::WordTimestamp> &, int64_t) = nullptr;
};

extern const ParakeetImplementation kParakeetImplementation;
extern const ParakeetImplementation kPhononImplementation;
extern const ParakeetImplementation kBarkImplementation;
extern const ParakeetImplementation kHaasImplementation;

const ParakeetImplementation & implementation_for_family(std::string_view family);

}  // namespace engine::community_models::parakeet_tdt
