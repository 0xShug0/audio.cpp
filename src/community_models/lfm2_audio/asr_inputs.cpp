#include "engine/community_models/lfm2_audio/asr_inputs.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

constexpr int kSampleRate = 16000;
// Never looked up: the audio embeddings overwrite these rows before prefill.
constexpr int32_t kAudioPlaceholderId = 0;

// The system prompts each checkpoint was trained with (liquid-audio README /
// README_JP).
std::string asr_system_prompt(const std::string & language) {
    if (language == "en") {
        return "Perform ASR.";
    }

    if (language == "ja") {
        return "Perform ASR in japanese.";
    }

    throw std::runtime_error("LFM2-Audio has no ASR prompt for language " + language);
}

void append(std::vector<int32_t> & out, const std::vector<int32_t> & ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

}  // namespace

std::vector<float> lfm2_audio_mono_16k(const runtime::AudioBuffer & audio) {
    if (audio.samples.empty() || audio.channels <= 0) {
        throw std::runtime_error("LFM2-Audio requires non-empty audio");
    }

    // One NaN or Inf sample turns every feature of its chunk into NaN, and the
    // encoder's first ReLU maps NaN to 0 (ggml_vec_relu_f32), so the chunk would
    // be transcribed from constant input without an error.
    if (!std::all_of(audio.samples.begin(), audio.samples.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LFM2-Audio input audio has non-finite samples");
    }

    auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate == kSampleRate) {
        return mono;
    }

    // liquid-audio's ChatState.add_audio uses torchaudio.functional.resample
    // on float32 audio.
    return audio::resample_mono_torchaudio_sinc_hann(
        mono, audio.sample_rate, kSampleRate, audio::torchaudio_sinc_hann_float32_options());
}

Lfm2Prompt Lfm2AsrPrompt::with_audio(int64_t audio_tokens) const {
    Lfm2Prompt out;
    out.input_ids = prefix;
    for (int64_t i = 0; i < audio_tokens; ++i) {
        out.audio_positions.push_back(static_cast<int32_t>(out.input_ids.size()));
        out.input_ids.push_back(kAudioPlaceholderId);
    }

    append(out.input_ids, suffix);
    return out;
}

Lfm2AsrPrompt make_lfm2_asr_prompt(const Lfm2TextTokenizer & tokenizer, const std::string & language) {
    const auto system_prompt = asr_system_prompt(language);

    // The pieces below spell these; without them the tokenizer would quietly
    // split the markup into bytes.
    for (const char * token : {"<|startoftext|>", "<|im_start|>", "<|im_end|>"}) {
        (void)tokenizer.require_token_id(token);
    }

    // liquid-audio's ChatState encodes each piece on its own, so the prompt is
    // assembled from the same pieces rather than from one string.
    Lfm2AsrPrompt out;
    for (const char * piece : {"<|startoftext|>", "<|im_start|>system\n"}) {
        append(out.prefix, tokenizer.encode(piece));
    }
    append(out.prefix, tokenizer.encode(system_prompt));
    for (const char * piece : {"<|im_end|>\n", "<|im_start|>user\n"}) {
        append(out.prefix, tokenizer.encode(piece));
    }

    for (const char * piece : {"<|im_end|>\n", "<|im_start|>assistant\n"}) {
        append(out.suffix, tokenizer.encode(piece));
    }

    // <|audio_start|> would switch generate_sequential to audio output, which
    // ASR does not produce.
    out.stop_token_ids = {tokenizer.require_token_id("<|im_end|>"), tokenizer.require_token_id("<|audio_start|>")};
    return out;
}

}  // namespace engine::community_models::lfm2_audio
