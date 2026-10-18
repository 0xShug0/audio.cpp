#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/vieneu_v3_turbo/assets.h"
#include "engine/community_models/vieneu_v3_turbo/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vieneu_v3_turbo {

enum class VieNeuTalkerPromptMode {
    VoiceClone,
    VoiceDesign,
    CustomVoice,
};

struct VieNeuTalkerPrefill {
    VieNeuTalkerPromptMode prompt_mode = VieNeuTalkerPromptMode::VoiceClone;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> instruct_ids;
    std::vector<int32_t> reference_ids;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::optional<VieNeuSpeakerEmbedding> speaker_embedding = std::nullopt;
    std::string speaker;
    std::string language = "Auto";
    bool icl_mode = false;
    bool x_vector_only_mode = false;
};

struct VieNeuTalkerCodes {
    Qwen3SpeechCodes generated_codes;
    Qwen3SpeechCodes decoder_input_codes;
};

class VieNeuTalkerWeightsRuntime;
class VieNeuTalkerStepRuntime;

class VieNeuTalkerStepRuntime {
public:
    class Impl;
    explicit VieNeuTalkerStepRuntime(std::unique_ptr<Impl> impl);
    ~VieNeuTalkerStepRuntime();

    VieNeuTalkerCodes generate(
        const VieNeuTalkerPrefill & prefill,
        const VieNeuTTSGenerationOptions & options,
        float repetition_penalty = 1.05F);
    int64_t release_cached_step_graph();

private:
    std::unique_ptr<Impl> impl_;
};

class VieNeuTalker {
public:
    explicit VieNeuTalker(VieNeuTTSTalkerConfig config);

    const VieNeuTTSTalkerConfig & config() const noexcept;

    std::shared_ptr<const VieNeuTalkerWeightsRuntime> create_weights_runtime(
        std::shared_ptr<const VieNeuTTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t graph_arena_bytes,
        size_t talker_constant_context_bytes,
        size_t code_predictor_constant_context_bytes,
        engine::assets::TensorStorageType weight_storage_type) const;

    std::shared_ptr<VieNeuTalkerStepRuntime> create_step_runtime(
        std::shared_ptr<const VieNeuTalkerWeightsRuntime> weights,
        int64_t prompt_capacity,
        int64_t generation_capacity) const;

private:
    VieNeuTTSTalkerConfig config_;
};

}  // namespace engine::models::vieneu_v3_turbo
