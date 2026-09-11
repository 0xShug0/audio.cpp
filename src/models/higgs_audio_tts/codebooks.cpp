#include "engine/models/higgs_audio_tts/codebooks.h"

#include <stdexcept>
#include <string>

namespace engine::models::higgs_audio_tts {
namespace {

void require_codebook_matrix(
    const std::vector<int32_t> & codes,
    int64_t frames,
    int64_t codebooks,
    const char * label) {
    if (frames <= 0 || codebooks <= 0) {
        throw std::runtime_error(std::string("Higgs TTS ") + label + " requires positive frames and codebooks");
    }
    if (static_cast<int64_t>(codes.size()) != frames * codebooks) {
        throw std::runtime_error(std::string("Higgs TTS ") + label + " code matrix shape mismatch");
    }
}

size_t flat_index(int64_t frame, int64_t codebook, int64_t codebooks) {
    return static_cast<size_t>(frame * codebooks + codebook);
}

}  // namespace

int64_t higgs_delayed_frame_count(int64_t raw_frames, int64_t codebooks) {
    if (raw_frames <= 0 || codebooks <= 0) {
        throw std::runtime_error("Higgs TTS delayed frame count requires positive dimensions");
    }
    return raw_frames + codebooks - 1;
}

std::vector<int32_t> apply_higgs_delay_pattern(
    const std::vector<int32_t> & raw_codes,
    int64_t raw_frames,
    int64_t codebooks) {
    require_codebook_matrix(raw_codes, raw_frames, codebooks, "delay pattern input");
    const int64_t delayed_frames = higgs_delayed_frame_count(raw_frames, codebooks);
    std::vector<int32_t> delayed(static_cast<size_t>(delayed_frames * codebooks), kHiggsEocId);
#ifdef _OPENMP
#pragma omp parallel for if (raw_frames * codebooks > 1024)
#endif
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        for (int64_t frame = 0; frame < codebook; ++frame) {
            delayed[flat_index(frame, codebook, codebooks)] = kHiggsBocId;
        }
        for (int64_t frame = 0; frame < raw_frames; ++frame) {
            delayed[flat_index(codebook + frame, codebook, codebooks)] =
                raw_codes[flat_index(frame, codebook, codebooks)];
        }
    }
    return delayed;
}

std::vector<int32_t> reverse_higgs_delay_pattern(
    const std::vector<int32_t> & delayed_codes,
    int64_t delayed_frames,
    int64_t codebooks) {
    require_codebook_matrix(delayed_codes, delayed_frames, codebooks, "reverse delay pattern input");
    const int64_t raw_frames = delayed_frames - (codebooks - 1);
    if (raw_frames <= 0) {
        throw std::runtime_error("Higgs TTS delayed codes must include at least one recoverable raw frame");
    }
    std::vector<int32_t> raw(static_cast<size_t>(raw_frames * codebooks), 0);
#ifdef _OPENMP
#pragma omp parallel for if (raw_frames * codebooks > 1024)
#endif
    for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
        for (int64_t frame = 0; frame < raw_frames; ++frame) {
            raw[flat_index(frame, codebook, codebooks)] =
                delayed_codes[flat_index(codebook + frame, codebook, codebooks)];
        }
    }
    return raw;
}

}  // namespace engine::models::higgs_audio_tts
