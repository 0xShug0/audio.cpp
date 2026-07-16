#pragma once

#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/types.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::vevo2 {

struct Vevo2VocoderWeights;
struct Vevo2VocoderGraph;

class Vevo2VocoderRuntime final {
public:
    Vevo2VocoderRuntime(
        const Vevo2Assets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Vevo2VocoderRuntime();

    runtime::AudioBuffer decode(const Vevo2MelSequence & mel) const;

private:
    Vevo2VocoderConfig config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> weight_source_;
    std::shared_ptr<const Vevo2VocoderWeights> weights_;
    mutable std::unique_ptr<Vevo2VocoderGraph> graph_;
    std::string name_;
};

}  // namespace engine::models::vevo2
