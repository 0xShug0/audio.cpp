#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/vietneu_tts/assets.h"
#include "engine/community_models/vietneu_tts/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vietneu_tts {

enum class VietneuTalkerPromptMode {
    VoiceClone,
    VoiceDesign,
    CustomVoice,
};

struct VietneuTalkerPrefill {
    VietneuTalkerPromptMode prompt_mode = VietneuTalkerPromptMode::VoiceClone;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> instruct_ids;
    std::vector<int32_t> reference_ids;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::optional<VietneuSpeakerEmbedding> speaker_embedding = std::nullopt;
    std::string speaker;
    std::string language = "Auto";
    bool icl_mode = false;
    bool x_vector_only_mode = false;
};

struct VietneuTalkerCodes {
    Qwen3SpeechCodes generated_codes;
    Qwen3SpeechCodes decoder_input_codes;
};

class VietneuTalkerWeightsRuntime;
class VietneuTalkerStepRuntime;

class VietneuTalkerStepRuntime {
public:
    class Impl;
    explicit VietneuTalkerStepRuntime(std::unique_ptr<Impl> impl);
    ~VietneuTalkerStepRuntime();

    VietneuTalkerCodes generate(
        const VietneuTalkerPrefill & prefill,
        const VietneuTTSGenerationOptions & options,
        float repetition_penalty = 1.05F);
    int64_t release_cached_step_graph();

private:
    std::unique_ptr<Impl> impl_;
};

class VietneuTalker {
public:
    explicit VietneuTalker(VietneuTTSTalkerConfig config);

    const VietneuTTSTalkerConfig & config() const noexcept;

    std::shared_ptr<const VietneuTalkerWeightsRuntime> create_weights_runtime(
        std::shared_ptr<const VietneuTTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t graph_arena_bytes,
        size_t talker_constant_context_bytes,
        size_t code_predictor_constant_context_bytes,
        engine::assets::TensorStorageType weight_storage_type) const;

    std::shared_ptr<VietneuTalkerStepRuntime> create_step_runtime(
        std::shared_ptr<const VietneuTalkerWeightsRuntime> weights,
        int64_t prompt_capacity,
        int64_t generation_capacity) const;

private:
    VietneuTTSTalkerConfig config_;
};

}  // namespace engine::models::vietneu_tts
