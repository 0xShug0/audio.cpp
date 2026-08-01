#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/types.h"

#include <cstddef>
#include <cstdint>
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

struct Vevo2FMWeights;
struct Vevo2FMGraph;
struct Vevo2FMStepGraph;

class Vevo2FlowMatchingRuntime final {
public:
    Vevo2FlowMatchingRuntime(
        const Vevo2Assets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Vevo2FlowMatchingRuntime();

    Vevo2MelSequence generate_mel(
        const runtime::AudioBuffer & timbre_ref_audio,
        const Vevo2TokenSequence & timbre_tokens,
        const Vevo2TokenSequence & generated_tokens,
        const Vevo2GenerationOptions & generation) const;

private:
    struct TimbreMelCacheKey {
        uint64_t hash = 0;
        int sample_rate = 0;
        int channels = 0;
        size_t samples = 0;
    };

    struct TimbreMelCacheKeyEqual {
        bool operator()(const TimbreMelCacheKey & lhs, const TimbreMelCacheKey & rhs) const noexcept {
            return lhs.hash == rhs.hash &&
                lhs.sample_rate == rhs.sample_rate &&
                lhs.channels == rhs.channels &&
                lhs.samples == rhs.samples;
        }
    };

    struct TimbreMelCacheValue {
        Vevo2MelSequence mel;
    };

    Vevo2MelSequence cached_timbre_mel(const runtime::AudioBuffer & timbre_ref_audio) const;

    Vevo2FMConfig config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> weight_source_;
    std::shared_ptr<const Vevo2FMWeights> weights_;
    mutable std::unique_ptr<Vevo2FMGraph> graph_;
    mutable std::unique_ptr<Vevo2FMStepGraph> step_graph_;
    mutable runtime::CacheSlots<TimbreMelCacheKey, TimbreMelCacheValue, TimbreMelCacheKeyEqual> timbre_mel_cache_;
    std::string name_;
};

}  // namespace engine::models::vevo2
