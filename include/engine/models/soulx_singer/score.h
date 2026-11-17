#pragma once
#include "engine/framework/io/json.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/convnext_modules.h"
#include <map>
#include "engine/framework/runtime/cache_slots.h"

namespace engine::models::soulx_singer {
struct SingingScore {
    std::vector<int32_t> phonemes, pitches, types, frame_notes;
    std::vector<float> f0;
};
SingingScore parse_score(const io::json::Value &metadata, const std::map<std::string, int32_t> &phones);

class ScoreConditioner {
  public:
    ScoreConditioner(const assets::TensorSource &source, core::ExecutionContext &execution);
    std::vector<float> encode(const SingingScore &score, int shift, bool melody);

  private:
    core::ExecutionContext &execution_;
    core::BackendWeightStore weights_;
    core::TensorValue text_, pitch_, type_;
    std::vector<modules::ConvNeXt1dWeights> blocks_;
    struct Graph {
        std::unique_ptr<core::PreparedGraph> owner;
        core::TensorValue text, pitch, type, frames, output;
    };
    runtime::CacheSlots<std::pair<size_t, size_t>, Graph> graphs_{2};
};
} // namespace engine::models::soulx_singer
