#include "engine/framework/audio/decode.h"
#include "engine/framework/audio/wav_writer.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_throws_containing(
    const std::function<void()> & action,
    const std::string & needle,
    const std::string & label) {
    try {
        action();
    } catch (const std::exception & ex) {
        const std::string what = ex.what();
        require(
            what.find(needle) != std::string::npos,
            label + ": expected message containing '" + needle + "', got '" + what + "'");
        return;
    }
    throw std::runtime_error(label + ": expected an exception, none was thrown");
}

std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "failed to open " + path.string());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() / "audiocpp_audio_decode_test";
        std::filesystem::create_directories(root);

        // --- WAV bytes decode identically through the new entry point --------
        {
            const auto path = root / "tone.wav";
            const int sample_rate = 22050;
            const int channels = 1;
            std::vector<float> samples(256);
            for (size_t i = 0; i < samples.size(); ++i) {
                samples[i] = 0.25f * std::sin(2.0f * 3.14159265f * 440.0f *
                    static_cast<float>(i) / static_cast<float>(sample_rate));
            }
            engine::audio::write_pcm16_wav(path, sample_rate, channels, samples);

            const auto bytes = read_file(path);
            // Disambiguates against the std::filesystem::path overload: a std::string
            // is an equally-valid implicit conversion target for both, which some
            // compilers (Clang) reject outright as an ambiguous call.
            const auto direct = engine::audio::read_wav_f32(std::string_view(bytes));
            const auto via_decode = engine::audio::decode_audio_upload_f32(bytes);
            require(direct.sample_rate == via_decode.sample_rate, "WAV passthrough: sample_rate mismatch");
            require(direct.channels == via_decode.channels, "WAV passthrough: channels mismatch");
            require(direct.samples.size() == via_decode.samples.size(), "WAV passthrough: sample count mismatch");
            for (size_t i = 0; i < direct.samples.size(); ++i) {
                require(direct.samples[i] == via_decode.samples[i], "WAV passthrough: sample value mismatch");
            }
        }

        // --- Empty upload ------------------------------------------------------
        require_throws_containing(
            [&] { (void)engine::audio::decode_audio_upload_f32(std::string_view()); },
            "empty", "empty upload rejection");

        // --- Ogg container (Vorbis/Opus payload audio.cpp cannot decode yet) ---
        {
            const std::string ogg_bytes = std::string("OggS") + std::string(23, '\0');
            require_throws_containing(
                [&] { (void)engine::audio::decode_audio_upload_f32(ogg_bytes); },
                "Ogg", "Ogg container rejection");
        }

        // --- WebM/Matroska container (what browser MediaRecorder sends) --------
        {
            const std::string webm_bytes =
                std::string({static_cast<char>(0x1A), static_cast<char>(0x45),
                             static_cast<char>(0xDF), static_cast<char>(0xA3)}) +
                std::string(16, '\0');
            require_throws_containing(
                [&] { (void)engine::audio::decode_audio_upload_f32(webm_bytes); },
                "WebM", "WebM container rejection");
        }

        // --- Unrecognizable garbage ----------------------------------------
        {
            const std::string garbage = "not an audio file, just text bytes to sniff";
            require_throws_containing(
                [&] { (void)engine::audio::decode_audio_upload_f32(garbage); },
                "unrecognized or corrupt", "garbage upload rejection");
        }

        std::cout << "audio_decode_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "audio_decode_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
