#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/audio/dsp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::auk {

audio::WhisperLogMelFeatures extract_audio_features(
    const std::vector<float> & samples, int sample_rate, int channels, size_t threads = 8);

class AudioConditioningRuntime {
public:
    AudioConditioningRuntime(core::ExecutionContext & execution, const assets::TensorSource & qwen, int64_t frames);
    ~AudioConditioningRuntime();
    AudioConditioningRuntime(const AudioConditioningRuntime &) = delete;
    AudioConditioningRuntime & operator=(const AudioConditioningRuntime &) = delete;
    void prepare(int64_t frames);
    // Valid log-mel frames in [128, frames]; returns [output_frames, 2048].
    std::vector<float> encode(const std::vector<float> & features);
private:
    struct State;
    std::unique_ptr<State> state_;
};

struct ConditioningInput {
    std::string formatted_text;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> positions;
};

ConditioningInput prepare_conditioning(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & instruction, int64_t audio_tokens = 0);

struct ConditioningWeights {
    core::TensorValue embedding;
    modules::NormWeights final_norm;
    std::vector<modules::QwenDecoderLayerWeights> layers;
    core::TensorValue layer_weights;
    core::TensorValue layer_scale;
};

ConditioningWeights load_conditioning_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & qwen,
    const assets::TensorSource & auk);

core::TensorValue build_text_conditioning(
    core::ModuleBuildContext & ctx,
    const ConditioningWeights & weights,
    const core::TensorValue & embeddings,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const modules::QwenDecoderActivationCastPolicy & activation_cast = {},
    std::vector<core::TensorValue> * captured_layers = nullptr);

class ConditioningRuntime {
public:
    ConditioningRuntime(core::ExecutionContext & execution,
                        const assets::TensorSource & qwen,
                        const assets::TensorSource & auk,
                        int64_t tokens, bool capture_layers = false, bool bf16_autocast = true, int64_t audio_tokens = 0);
    ~ConditioningRuntime();
    ConditioningRuntime(const ConditioningRuntime &) = delete;
    ConditioningRuntime & operator=(const ConditioningRuntime &) = delete;
    void prepare(int64_t tokens, int64_t audio_tokens = 0);
    std::vector<float> encode(const ConditioningInput & input, const std::vector<float> & audio_embeddings = {});
    std::vector<std::vector<float>> captured_layers() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::auk
