#pragma once

#include "engine/models/higgs_audio_tts/ar.h"
#include "engine/models/higgs_audio_tts/codec.h"
#include "engine/models/higgs_audio_tts/sampler.h"
#include "engine/models/higgs_audio_tts/tokenizer_text.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::higgs_audio_tts {

struct HiggsGenerationOptions {
    int64_t max_tokens = 2048;
    float temperature = 1.0F;
    std::optional<float> top_p;
    std::optional<int64_t> top_k;
    float repetition_penalty = 1.0F;
    std::optional<uint64_t> seed;
};

struct HiggsGenerationRequest {
    std::string text;
    std::string reference_text;
    std::vector<int32_t> reference_codes;
    int64_t reference_frames = 0;
    int64_t reference_codebooks = 0;
    HiggsGenerationOptions options;
};

struct HiggsGenerationResult {
    HiggsCodecDecodeOutput audio;
    std::vector<int32_t> delayed_codes;
    int64_t delayed_frames = 0;
    std::vector<int32_t> raw_codes;
    int64_t raw_frames = 0;
};

class HiggsGenerator {
public:
    HiggsGenerator(std::shared_ptr<const HiggsAssets> assets,
                   std::shared_ptr<HiggsARRuntime> ar,
                   std::shared_ptr<HiggsCodecRuntime> codec,
                   size_t ar_decode_graph_arena_bytes);

    void prepare(const HiggsGenerationRequest & request);
    HiggsGenerationResult generate(const HiggsGenerationRequest & request);

private:
    struct ReferencePrefixCache {
        std::string reference_text;
        std::vector<int32_t> reference_codes;
        int64_t reference_frames = 0;
        int64_t reference_codebooks = 0;
        std::vector<int32_t> delayed_reference_codes;
        int64_t delayed_reference_frames = 0;
        std::vector<int32_t> prefix_tokens;
        int64_t prefix_steps = 0;
    };

    std::shared_ptr<const HiggsAssets> assets_;
    std::shared_ptr<HiggsARRuntime> ar_;
    std::shared_ptr<HiggsCodecRuntime> codec_;
    HiggsTextTokenizer tokenizer_;
    size_t ar_decode_graph_arena_bytes_ = 0;
    std::optional<ReferencePrefixCache> reference_prefix_cache_;
    bool reference_kv_ready_ = false;
    std::optional<HiggsCudaSamplingPolicy> cuda_sampling_policy_;
    std::unique_ptr<HiggsARKVCache> ar_kv_cache_;
    std::unique_ptr<HiggsARPrefillGraph> prefill_graph_;
    std::unique_ptr<HiggsARDecodeGraph> decode_graph_;
};

} // namespace engine::models::higgs_audio_tts
