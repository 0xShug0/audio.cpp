#pragma once

// AuK's message format, as Qwen2.5-Omni's chat template renders it.
//
// The processor produces exactly this for a user turn (verified against
// Qwen2_5OmniProcessor.apply_chat_template):
//
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
//   <|im_start|>user\n<|audio_bos|><|AUDIO|>...<|audio_eos|>INSTRUCTION<|im_end|>\n
//   <|im_start|>assistant\n
//
// ⚠ <|AUDIO|> is repeated once per audio TOKEN, not once per clip: the template is
// expanded by the processor after it knows how long the audio is. Four mel frames
// become one token, so the count has to be derived from the waveform before the text
// can be built.

#include <cstdint>
#include <string>

namespace engine::models::auk {

// Audio tokens for a 16 kHz clip: mel frames (hop 160), then conv2's stride 2 and the
// average pool, matching _get_feat_extract_output_lengths.
int64_t audio_token_count(int64_t samples_16k);

// The templated string. `audio_tokens` of 0 renders a text-only turn.
std::string build_user_message(const std::string & instruction, int64_t audio_tokens);

}  // namespace engine::models::auk
