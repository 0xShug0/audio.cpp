// The server voice library resolves "voice" to an Irodori-TTS Speaker Inversion
// embedding (<name>.speaker.safetensors) before a wav of the same name, only for
// families that use such files.
#include "server_protocol.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace fs = std::filesystem;
using minitts::server::RuntimeVoicePreset;
using minitts::server::SpeechRequestContext;

void expect(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_file(const fs::path & path, const std::string & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    expect(static_cast<bool>(out), "could not write " + path.string());
}

void put_u16(std::string & out, uint16_t value) {
    out.push_back(static_cast<char>(value & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
}

void put_u32(std::string & out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

// 16 kHz mono PCM16 silence.
std::string tiny_wav() {
    constexpr uint32_t kSamples = 160;
    std::string out = "RIFF";
    put_u32(out, 36 + kSamples * 2);
    out += "WAVEfmt ";
    put_u32(out, 16);
    put_u16(out, 1);
    put_u16(out, 1);
    put_u32(out, 16000);
    put_u32(out, 16000 * 2);
    put_u16(out, 2);
    put_u16(out, 16);
    out += "data";
    put_u32(out, kSamples * 2);
    out.append(kSamples * 2, '\0');
    return out;
}

engine::runtime::TaskRequest speech(const fs::path & voice_dir, bool uses_speaker_embeddings,
                                    const std::string & body_text) {
    const std::optional<fs::path> dir = voice_dir;
    const std::unordered_map<std::string, RuntimeVoicePreset> presets;
    const std::optional<RuntimeVoicePreset> default_preset;
    const std::unordered_map<std::string, std::string> default_options;
    return minitts::server::build_speech_request(
        engine::io::json::parse(body_text),
        SpeechRequestContext{voice_dir, dir, presets, default_preset, true, true, true,
                             default_options, uses_speaker_embeddings});
}

bool has_audio(const engine::runtime::TaskRequest & request) {
    return request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value();
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() /
        ("audiocpp-voice-library-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(dir);
        // The embedding is never opened here: the model reads it.
        write_file(dir / "both.speaker.safetensors", "not read by the server");
        write_file(dir / "both.wav", tiny_wav());
        write_file(dir / "embedding_only.speaker.safetensors", "not read by the server");
        write_file(dir / "wav_only.wav", tiny_wav());
        write_file(dir / "plain.safetensors", "not a speaker embedding name");

        expect(minitts::server::uses_voice_library_speaker_embeddings("irodori_tts"),
               "irodori_tts should use voice library speaker embeddings");
        expect(!minitts::server::uses_voice_library_speaker_embeddings("vieneu_v3_turbo"),
               "vieneu_v3_turbo reads another embedding format");

        // An embedding wins over a wav of the same name and replaces an embedding
        // given in options under either name.
        auto request = speech(dir, true,
            R"({"input":"x","voice":"both","options":{"speaker_embedding_path":"other.safetensors"}})");
        expect(request.options.at("speaker_embedding_file") ==
                   (dir / "both.speaker.safetensors").string(),
               "voice did not resolve to the library embedding");
        expect(request.options.count("speaker_embedding_path") == 0,
               "legacy embedding option was kept next to the resolved one");
        expect(!has_audio(request), "an embedding voice also loaded reference audio");

        request = speech(dir, true, R"({"input":"x","voice":"embedding_only"})");
        expect(request.options.at("speaker_embedding_file") ==
                   (dir / "embedding_only.speaker.safetensors").string(),
               "embedding-only voice did not resolve");

        request = speech(dir, true, R"({"input":"x","voice":"wav_only"})");
        expect(has_audio(request) && request.options.count("speaker_embedding_file") == 0,
               "wav-only voice did not resolve to reference audio");

        // Other families keep the wav, and never get the embedding.
        request = speech(dir, false, R"({"input":"x","voice":"both"})");
        expect(has_audio(request) && request.options.count("speaker_embedding_file") == 0,
               "a family without embeddings did not get the wav");
        request = speech(dir, false, R"({"input":"x","voice":"embedding_only"})");
        expect(request.options.count("speaker_embedding_file") == 0 &&
                   request.voice.has_value() && request.voice->speaker.has_value() &&
                   request.voice->speaker->cached_voice_id == "embedding_only",
               "a family without embeddings did not fall back to the cached voice id");

        // An explicit voice_ref bypasses the library, as for wavs.
        request = speech(dir, true,
            R"({"input":"x","voice":"embedding_only","voice_ref":")" +
                (dir / "wav_only.wav").generic_string() + R"("})");
        expect(request.options.count("speaker_embedding_file") == 0 && has_audio(request),
               "voice_ref did not bypass the library embedding");

        // Names that are paths never leave the library.
        request = speech(dir, true, R"({"input":"x","voice":"../embedding_only"})");
        expect(request.options.count("speaker_embedding_file") == 0,
               "a path-like voice name resolved to an embedding");

        auto names = minitts::server::voice_library_speaker_embedding_names(dir);
        std::sort(names.begin(), names.end());
        expect(names == std::vector<std::string>{"both", "embedding_only"},
               "embedding names should be the .speaker.safetensors stems only");
        expect(minitts::server::voice_library_speaker_embedding_names(dir / "missing").empty(),
               "a missing voice library should list no embeddings");
    } catch (const std::exception & error) {
        std::error_code ec;
        fs::remove_all(dir, ec);
        std::cerr << "voice library test failed: " << error.what() << "\n";
        return 1;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::cout << "server voice library passed\n";
    return 0;
}
