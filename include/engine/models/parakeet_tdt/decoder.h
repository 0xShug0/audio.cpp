#pragma once

#include "engine/framework/decoders/tdt_decoder_algorithm.h"
#include "engine/models/parakeet_tdt/assets.h"
#include "engine/models/parakeet_tdt/weights.h"

#include <memory>
#include <cstddef>
#include <string>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::parakeet_tdt {

struct ParakeetDecodeResult {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> token_timestamps;
    std::vector<int32_t> token_durations;
    std::string text;
};

struct ParakeetStreamingDecoderState {
    bool initialized = false;
    std::vector<float> h0;
    std::vector<float> c0;
    std::vector<float> h1;
    std::vector<float> c1;
    std::vector<float> decoder_output;
};

enum class ParakeetTraceContractMode {
    Offline,
    Longform,
    Streaming,
};

class ParakeetSharedDecoder {
public:
    ParakeetSharedDecoder(const ParakeetAssets & assets, const ParakeetTDTWeights & weights);
    ~ParakeetSharedDecoder();

    ParakeetSharedDecoder(const ParakeetSharedDecoder &) = delete;
    ParakeetSharedDecoder & operator=(const ParakeetSharedDecoder &) = delete;
    ParakeetSharedDecoder(ParakeetSharedDecoder &&) noexcept;
    ParakeetSharedDecoder & operator=(ParakeetSharedDecoder &&) noexcept;

    ParakeetDecodeResult run(
        core::ExecutionContext & execution_context,
        const std::vector<float> & encoder_projected,
        int64_t frames,
        decoders::TdtDecoderAlgorithm algorithm);

    ParakeetDecodeResult run_streaming_chunk(
        core::ExecutionContext & execution_context,
        const std::vector<float> & encoder_projected,
        int64_t frames,
        decoders::TdtDecoderAlgorithm algorithm,
        ParakeetStreamingDecoderState & state);

    void prepare(core::ExecutionContext & execution_context, int64_t capacity_frames);

    size_t joint_graph_count() const;
    int64_t active_joint_frames() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::parakeet_tdt
