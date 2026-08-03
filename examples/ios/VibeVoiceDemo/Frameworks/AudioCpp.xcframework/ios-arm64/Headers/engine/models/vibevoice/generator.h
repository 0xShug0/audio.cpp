#pragma once

#include "engine/models/vibevoice/connector.h"
#include "engine/models/vibevoice/decoder.h"
#include "engine/models/vibevoice/diffusion_head.h"
#include "engine/models/vibevoice/tokenizer_audio.h"
#include "engine/models/vibevoice/tokenizer_text.h"
#include "engine/models/vibevoice/types.h"

#include <cstdint>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoicePreparedPrompt {
    VibeVoicePromptEncoding encoding;
    std::vector<float> embeddings;
    int64_t steps = 0;
    int64_t hidden_size = 0;
    uint64_t next_rng_index = 0;
};

VibeVoicePreparedPrompt prepare_vibevoice_prompt(
    const VibeVoiceRequest & request,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceDecoderWeightsRuntime & decoder,
    uint64_t seed,
    uint64_t start_rng_index,
    const std::vector<float> * prompt_noise_values = nullptr);

VibeVoiceResult generate_vibevoice(
    const VibeVoiceRequest & request,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceDecoderWeightsRuntime & decoder,
    const VibeVoiceDiffusionHeadWeightsRuntime & diffusion_head);

VibeVoiceResult generate_vibevoice(
    const VibeVoiceRequest & request,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceDecoderWeightsRuntime & decoder,
    const VibeVoiceDiffusionHeadWeightsRuntime & diffusion_head,
    VibeVoiceDecoderCachedState & positive_cache,
    VibeVoiceDecoderCachedState & negative_cache);

std::vector<VibeVoiceResult> generate_vibevoice_batch(
    const std::vector<VibeVoiceRequest> & requests,
    const VibeVoiceTextTokenizer & text_tokenizer,
    const VibeVoiceTokenizerWeightsRuntime & audio_tokenizer,
    const VibeVoiceConnectorWeightsRuntime & connector,
    const VibeVoiceDecoderWeightsRuntime & decoder,
    const VibeVoiceDiffusionHeadWeightsRuntime & diffusion_head);

}  // namespace engine::models::vibevoice
