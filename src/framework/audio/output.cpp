#include "engine/framework/audio/output.h"

#include "engine/framework/audio/wav_writer.h"

#include <string>

namespace engine::audio {

std::string WavPcm16Sink::family() const {
    return "wav_pcm16";
}

void WavPcm16Sink::write(const std::filesystem::path & path, const AudioBuffer & audio) const {
    write_pcm16_wav(path, audio.sample_rate, audio.channel_count, audio.samples);
}

std::string WavSink::family() const {
    return std::string("wav_") + wav_sample_format_name(options_.format);
}

void WavSink::write(const std::filesystem::path & path, const AudioBuffer & audio) const {
    write_wav(path, audio.sample_rate, audio.channel_count, audio.samples, options_);
}

}  // namespace engine::audio
