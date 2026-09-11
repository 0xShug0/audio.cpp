// The WAV 'data' chunk_size is a 32-bit field read straight from the file, so a
// few-byte WAV can claim up to 4 GiB. The reader used to resize() the whole
// amount before reading a single byte of it, turning a truncated or hostile
// header into an out-of-memory condition rather than a parse error.
//
// The reader now grows only as fast as data actually arrives, so an
// unbacked claim fails as a read error after one block.

#include "engine/framework/audio/wav_reader.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void put_u32(std::string & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

void put_u16(std::string & out, uint16_t value) {
    for (int i = 0; i < 2; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

/// A minimal 16-bit mono PCM WAV whose 'data' chunk declares `declared_size`
/// while actually carrying `actual` bytes.
std::string make_wav(uint32_t declared_size, const std::vector<char> & actual) {
    std::string wav;
    wav += "RIFF";
    put_u32(wav, 0);  // riff size, skipped by the reader
    wav += "WAVE";

    wav += "fmt ";
    put_u32(wav, 16);
    put_u16(wav, 1);      // PCM
    put_u16(wav, 1);      // mono
    put_u32(wav, 16000);  // sample rate
    put_u32(wav, 32000);  // byte rate
    put_u16(wav, 2);      // block align
    put_u16(wav, 16);     // bits per sample

    wav += "data";
    put_u32(wav, declared_size);
    wav.append(actual.data(), actual.size());
    return wav;
}

void test_oversized_chunk_size_is_rejected_not_allocated() {
    // Claims 4 GiB, carries four bytes. Must fail as a read error rather than
    // attempting the allocation.
    const std::string wav = make_wav(0xFFFFFFFFu, {0, 0, 0, 0});
    bool threw = false;
    try {
        (void) engine::audio::read_wav_f32(std::string_view(wav));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "a data chunk larger than the file must be rejected");
}

void test_truncated_chunk_is_rejected() {
    const std::string wav = make_wav(1024, std::vector<char>(16, 0));
    bool threw = false;
    try {
        (void) engine::audio::read_wav_f32(std::string_view(wav));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "a truncated data chunk must be rejected");
}

void test_valid_wav_still_reads() {
    // 8 samples of 16-bit mono PCM, honestly declared.
    std::vector<char> pcm(16, 0);
    for (size_t i = 0; i < 8; ++i) {
        const int16_t sample = static_cast<int16_t>(i * 1000);
        std::memcpy(pcm.data() + i * 2, &sample, sizeof(sample));
    }
    const std::string wav = make_wav(static_cast<uint32_t>(pcm.size()), pcm);
    const auto decoded = engine::audio::read_wav_f32(std::string_view(wav));

    require(decoded.sample_rate == 16000, "sample rate should round-trip");
    require(decoded.channels == 1, "channel count should round-trip");
    require(decoded.samples.size() == 8, "all 8 samples should decode");
}

}  // namespace

int main() {
    try {
        test_oversized_chunk_size_is_rejected_not_allocated();
        test_truncated_chunk_is_rejected();
        test_valid_wav_still_reads();
    } catch (const std::exception & error) {
        std::cerr << "wav chunk bounds test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "wav chunk bounds test passed\n";
    return 0;
}
