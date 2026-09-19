#include "engine/models/auk/message.h"

namespace engine::models::auk {

int64_t audio_token_count(int64_t samples_16k) {
    if (samples_16k <= 0) return 0;
    const int64_t mel_frames = samples_16k / 160;          // hop 160 at 16 kHz
    const int64_t after_conv = (mel_frames - 1) / 2 + 1;   // conv2 stride 2
    return (after_conv - 2) / 2 + 1;                       // average pool
}

std::string build_user_message(const std::string & instruction, int64_t audio_tokens) {
    std::string text =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n";
    if (audio_tokens > 0) {
        text += "<|audio_bos|>";
        text.reserve(text.size() + static_cast<size_t>(audio_tokens) * 9);
        for (int64_t index = 0; index < audio_tokens; ++index) text += "<|AUDIO|>";
        text += "<|audio_eos|>";
    }
    text += instruction;
    text += "<|im_end|>\n<|im_start|>assistant\n";
    return text;
}

}  // namespace engine::models::auk
