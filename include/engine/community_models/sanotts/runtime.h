#pragma once

#include "engine/community_models/sanotts/assets.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace engine::models::sanotts {

struct SanoTtsGenerationOptions {
    /** Duration multiplier applied before rounding; larger is slower. */
    float speaking_rate = 1.0F;
    /** Decoder noise seed; the session resolves the derive-from-text default
     *  before calling the runtime, so this is always the final seed. */
    uint64_t seed = 0;
};

/** Which code runs the frame stage + decoder on a CPU backend
 *  (session option sanotts.cpu_decoder). */
enum class SanoTtsCpuDecoder {
    Auto,   // NEON on supported AArch64 CPU sessions, otherwise GGML
    Ggml,   // the GGML graph (every platform and backend)
    Neon,   // hand-written ARM NEON kernels (AArch64 CPU backend only)
};

class SanoTtsNativeRuntime {
public:
    SanoTtsNativeRuntime(
        std::shared_ptr<const SanoTtsAssets> assets,
        core::BackendConfig backend_config,
        SanoTtsCpuDecoder cpu_decoder = SanoTtsCpuDecoder::Auto);
    ~SanoTtsNativeRuntime();

    runtime::AudioBuffer synthesize(
        const std::vector<int32_t> & token_ids,
        const SanoTtsGenerationOptions & options);

private:
    struct State;
    std::unique_ptr<State> state_;
};

/** int.from_bytes(sha256(text).digest()[:8], "big") -- the seed the reference
 *  implementations derive when the caller does not pass one. */
uint64_t sanotts_text_seed(const std::string & text);

}  // namespace engine::models::sanotts
