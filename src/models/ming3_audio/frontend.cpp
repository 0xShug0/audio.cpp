#include "engine/models/ming3_audio/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>

namespace engine::models::ming3_audio {

Frontend::Frontend(std::shared_ptr<const MingAudioAssets> assets)
    : assets_(std::move(assets)), tokenizer_({{}, {}, assets_->resources.require_file("tokenizer_config"),
        assets_->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::BailingMoe}),
      mel_({16000, 400, 160, assets_->config.audio.encoder.n_mels}) {}

AudioFeatures Frontend::extract(const runtime::AudioBuffer & input, size_t threads) const {
    auto samples = audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
    if (input.sample_rate != 16000) {
        samples = audio::resample_mono_torchaudio_sinc_hann(samples, input.sample_rate, 16000,
            audio::torchaudio_sinc_hann_float32_options());
    }
    const auto features = mel_.compute(samples, threads);
    AudioFeatures output;
    output.frames = features.frames;
    output.log_mel.resize(features.values.size());
    for (int64_t frame = 0; frame < features.frames; ++frame) {
        for (int64_t bin = 0; bin < features.mel_bins; ++bin) {
            output.log_mel[frame * features.mel_bins + bin] = features.values[bin * features.frames + frame];
        }
    }
    return output;
}

std::vector<int32_t> Frontend::prompt(const std::string & instruction, int64_t mel_frames,
                                    const std::string & system_prompt, bool enable_thinking) const {
    const auto & c = assets_->config.audio;
    std::string content;
    if (mel_frames > 0) {
        content = "<|audio_start|>";
        for (int64_t offset = 0; offset < mel_frames; offset += c.chunk_frames) {
            int64_t frames = (std::min(c.chunk_frames, mel_frames - offset) + 1) / 2;
            for (size_t i = 0; i < c.projector_kernels.size(); ++i) {
                const auto kernel = c.projector_kernels[i];
                frames = (frames + 2 * (kernel / 2) - kernel) / c.projector_strides[i] + 1;
            }
            for (int64_t i = 0; i < frames; ++i) content += "<|audio_pad|>";
        }
        content += "<|audio_end|>";
    }
    content = system_prompt.empty() ? content + instruction : instruction + content;
    std::string system = system_prompt;
    if (system.find("detailed thinking on") == std::string::npos &&
        system.find("detailed thinking off") == std::string::npos) {
        if (!system.empty() && system.back() != '\n') system += '\n';
        system += enable_thinking ? "detailed thinking on" : "detailed thinking off";
    }
    return tokenizer_.encode("<role>SYSTEM</role>" + system + "<|role_end|><role>HUMAN</role>" +
        content + "<|role_end|><role>ASSISTANT</role>\n<think>" + (enable_thinking ? "" : "</think>"), true);
}

std::string Frontend::decode(const std::vector<int32_t> & ids) const { return tokenizer_.decode(ids, true); }

}  // namespace engine::models::ming3_audio
