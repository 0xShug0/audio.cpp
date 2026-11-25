#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::reverb {

// The backbone shape, read from the checkpoint's config.json (llm_config and reverb.architecture).
struct ReverbArchitecture {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    // Rank of the factorized text embedding ([vocab, rank] table, then rank -> hidden); 0 for a plain
    // [vocab, hidden] table.
    int64_t embedding_rank = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    int64_t codebooks = 8;
    int64_t codes = 1025;  // entries per codebook: 1024 codes and the mask
    int32_t mask_id = 1024;
};

// reverb_tts.sampler.SamplerConfig: the upstream defaults are the tuned setup.
struct ReverbSamplerConfig {
    int64_t steps = 8;
    bool remasking = true;
    float remask_ratio = 0.25F;
    float t_shift = 0.1F;
    float codebook_penalty = 5.0F;
    float position_temperature = 5.0F;
    float polish_bias = 1.0F;
    float polish_guidance = 0.5F;
};

struct ReverbGeneratorStats {
    double prefix_ms = 0.0;
    double passes_ms = 0.0;
    double sampling_ms = 0.0;
    int64_t passes = 0;
};

// The masked-diffusion generator: one prefix pass (text, then reference frames) whose keys and
// values are kept, then `steps` main passes that fill codebooks 0-1 and two polish passes that fill
// codebooks 2-7, each over the target frames only.
class ReverbGenerator {
public:
    ReverbGenerator(
        std::shared_ptr<const assets::TensorSource> weights,
        const ReverbArchitecture & architecture,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType storage_type);
    ~ReverbGenerator();

    // text_ids: the tokenized prompt. reference_codes: [codebooks, reference_frames], codebook-major.
    // Returns the generated codes as [codebooks, target_frames], codebook-major.
    std::vector<int32_t> generate(
        const std::vector<int32_t> & text_ids,
        const std::vector<int32_t> & reference_codes,
        int64_t reference_frames,
        int64_t target_frames,
        uint32_t seed,
        const ReverbSamplerConfig & config);

    const ReverbArchitecture & architecture() const noexcept;
    const ReverbGeneratorStats & last_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The commit schedule's time points, computed in float32 exactly as upstream's
// torch.linspace + t_shift warp; exposed for tests.
std::vector<double> reverb_time_steps(int64_t steps, float t_shift);

}  // namespace engine::community_models::reverb
