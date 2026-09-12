#include "engine/framework/audio/decode.h"

// Must match the feature set miniaudio_impl.c is compiled with so both
// translation units agree on which parts of the API are declared.
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_THREADING
#define MA_NO_GENERATION

#include "miniaudio.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine::audio {
namespace {

bool starts_with(std::string_view data, std::string_view prefix) {
    return data.size() >= prefix.size() && data.substr(0, prefix.size()) == prefix;
}

bool looks_like_wav(std::string_view data) {
    return data.size() >= 12 && starts_with(data, "RIFF") && data.substr(8, 4) == "WAVE";
}

bool looks_like_ogg(std::string_view data) {
    return starts_with(data, "OggS");
}

bool looks_like_webm(std::string_view data) {
    // EBML magic number, the container header shared by WebM and Matroska.
    return data.size() >= 4 &&
        static_cast<uint8_t>(data[0]) == 0x1A && static_cast<uint8_t>(data[1]) == 0x45 &&
        static_cast<uint8_t>(data[2]) == 0xDF && static_cast<uint8_t>(data[3]) == 0xA3;
}

[[noreturn]] void throw_unsupported_container(const char * container_name) {
    throw std::runtime_error(
        std::string(container_name) +
        " audio uploads are not supported yet (browsers typically record microphone input as "
        "WebM/Opus or Ogg/Opus, which audio.cpp cannot decode without an additional Opus codec "
        "backend). Convert the upload to WAV, MP3, or FLAC before sending it, for example with "
        "`ffmpeg -i input -ar 16000 -ac 1 output.wav`.");
}

}  // namespace

WavData decode_audio_upload_f32(std::string_view bytes) {
    if (bytes.empty()) {
        throw std::runtime_error("audio upload is empty");
    }
    if (looks_like_wav(bytes)) {
        return read_wav_f32(bytes);
    }
    if (looks_like_ogg(bytes)) {
        throw_unsupported_container("Ogg");
    }
    if (looks_like_webm(bytes)) {
        throw_unsupported_container("WebM");
    }

    ma_decoder decoder;
    const ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
    const ma_result init_result = ma_decoder_init_memory(bytes.data(), bytes.size(), &config, &decoder);
    if (init_result != MA_SUCCESS) {
        throw std::runtime_error(
            "unrecognized or corrupt audio upload; supported formats are WAV, MP3, and FLAC");
    }

    ma_uint64 total_frames = 0;
    const ma_result length_result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
    if (length_result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        throw std::runtime_error("failed to determine length of decoded audio upload");
    }

    WavData result;
    result.sample_rate = static_cast<int>(decoder.outputSampleRate);
    result.channels = static_cast<int>(decoder.outputChannels);
    result.samples.resize(static_cast<size_t>(total_frames) * static_cast<size_t>(result.channels));

    ma_uint64 frames_read = 0;
    const ma_result read_result = total_frames == 0
        ? MA_SUCCESS
        : ma_decoder_read_pcm_frames(&decoder, result.samples.data(), total_frames, &frames_read);
    ma_decoder_uninit(&decoder);
    if (read_result != MA_SUCCESS) {
        throw std::runtime_error("failed to decode audio upload");
    }
    result.samples.resize(static_cast<size_t>(frames_read) * static_cast<size_t>(result.channels));
    return result;
}

}  // namespace engine::audio
