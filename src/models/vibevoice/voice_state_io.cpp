#include "engine/models/vibevoice/voice_state_io.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::vibevoice {
namespace {

constexpr std::array<char, 8> kMagic = {'V', 'V', 'S', 'T', 'A', 'T', 'E', '1'};
constexpr uint32_t kVersion = 1;

void write_u32(std::ostream & out, uint32_t value) {
    std::array<char, 4> bytes {
        static_cast<char>(value & 0xFFu),
        static_cast<char>((value >> 8u) & 0xFFu),
        static_cast<char>((value >> 16u) & 0xFFu),
        static_cast<char>((value >> 24u) & 0xFFu),
    };
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_i64(std::ostream & out, int64_t value) {
    auto bits = static_cast<uint64_t>(value);
    std::array<char, 8> bytes {};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((bits >> (8u * i)) & 0xFFu);
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

uint32_t read_u32(std::istream & in, const std::filesystem::path & path) {
    std::array<unsigned char, 4> bytes {};
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
        throw std::runtime_error("failed to read VibeVoice state file: " + path.string());
    }
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8u) |
           (static_cast<uint32_t>(bytes[2]) << 16u) |
           (static_cast<uint32_t>(bytes[3]) << 24u);
}

int64_t read_i64(std::istream & in, const std::filesystem::path & path) {
    std::array<unsigned char, 8> bytes {};
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
        throw std::runtime_error("failed to read VibeVoice state file: " + path.string());
    }
    uint64_t bits = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        bits |= static_cast<uint64_t>(bytes[i]) << (8u * i);
    }
    return static_cast<int64_t>(bits);
}

void validate_state(const VibeVoiceReferenceVoiceState & state, const std::filesystem::path & path) {
    if (state.frames <= 0 || state.dim <= 0 || state.speech_tokens <= 0) {
        throw std::runtime_error("invalid VibeVoice reference state shape: " + path.string());
    }
    if (state.frames > std::numeric_limits<int64_t>::max() / state.dim) {
        throw std::runtime_error("VibeVoice reference state shape overflows: " + path.string());
    }
    const auto expected = static_cast<size_t>(state.frames * state.dim);
    if (state.acoustic_mean.size() != expected) {
        throw std::runtime_error("VibeVoice reference state acoustic mean size mismatch: " + path.string());
    }
}

}  // namespace

VibeVoiceReferenceVoiceState load_vibevoice_reference_voice_state(
    const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open VibeVoice state file: " + path.string());
    }
    std::array<char, kMagic.size()> magic {};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != kMagic) {
        throw std::runtime_error("invalid VibeVoice state file magic: " + path.string());
    }
    const uint32_t version = read_u32(in, path);
    if (version != kVersion) {
        throw std::runtime_error("unsupported VibeVoice state file version: " + path.string());
    }

    VibeVoiceReferenceVoiceState state;
    state.frames = read_i64(in, path);
    state.dim = read_i64(in, path);
    state.speech_tokens = read_i64(in, path);
    if (state.frames <= 0 || state.dim <= 0) {
        throw std::runtime_error("invalid VibeVoice reference state shape: " + path.string());
    }
    if (state.frames > std::numeric_limits<int64_t>::max() / state.dim) {
        throw std::runtime_error("VibeVoice reference state shape overflows: " + path.string());
    }
    const auto value_count = static_cast<size_t>(state.frames * state.dim);
    state.acoustic_mean.resize(value_count);
    if (value_count > 0) {
        in.read(
            reinterpret_cast<char *>(state.acoustic_mean.data()),
            static_cast<std::streamsize>(value_count * sizeof(float)));
        if (!in) {
            throw std::runtime_error("failed to read VibeVoice reference state values: " + path.string());
        }
    }
    validate_state(state, path);
    return state;
}

void save_vibevoice_reference_voice_state(
    const std::filesystem::path & path,
    const VibeVoiceReferenceVoiceState & state) {
    validate_state(state, path);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open VibeVoice state output file: " + path.string());
    }
    out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_u32(out, kVersion);
    write_i64(out, state.frames);
    write_i64(out, state.dim);
    write_i64(out, state.speech_tokens);
    out.write(
        reinterpret_cast<const char *>(state.acoustic_mean.data()),
        static_cast<std::streamsize>(state.acoustic_mean.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("failed to write VibeVoice reference state file: " + path.string());
    }
}

}  // namespace engine::models::vibevoice
