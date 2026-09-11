#include "engine/models/moss/moss_tts_nano/generator.h"

#include <random>
#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_nano {

MossTTSNanoGenerator::MossTTSNanoGenerator(
    MossTTSNanoGlobalTransformerRuntime & global_transformer,
    MossTTSNanoLocalFrameDecoderRuntime & local_frame_decoder)
    : global_transformer_(global_transformer),
      local_frame_decoder_(local_frame_decoder) {}

MossTTSNanoAudioCodes MossTTSNanoGenerator::generate(
    const MossTTSNanoPrompt & prompt,
    const MossTTSNanoGenerationOptions & options) {
    if (prompt.rows <= 0 || prompt.row_width <= 1) {
        throw std::runtime_error("MOSS-TTS-Nano generator requires a non-empty prompt");
    }
    if (options.active_codebooks <= 0 || options.max_new_frames <= 0) {
        throw std::runtime_error("MOSS-TTS-Nano generator requires positive generation limits");
    }
    global_transformer_.prepare_prefill(prompt.rows);
    global_transformer_.prepare_decode(prompt.rows + options.max_new_frames);
    local_frame_decoder_.prepare(options.active_codebooks);
    std::vector<int32_t> rows = prompt.input_ids;
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(options.max_new_frames * (prompt.row_width - 1)));
    std::mt19937 rng(options.seed);
    uint64_t sample_call_index = 0;
    bool stopped_on_eoc = false;
    for (int64_t step = 0; step < options.max_new_frames; ++step) {
        const int64_t row_count = static_cast<int64_t>(rows.size()) / prompt.row_width;
        const auto hidden = global_transformer_.last_hidden(rows, row_count, prompt.row_width);
        auto frame = local_frame_decoder_.generate_frame(
            hidden,
            options.active_codebooks,
            generated,
            options.sampling,
            rng,
            options.seed,
            sample_call_index);
        if (frame.empty()) {
            stopped_on_eoc = true;
            break;
        }
        if (static_cast<int64_t>(frame.size()) != prompt.row_width - 1) {
            throw std::runtime_error("MOSS-TTS-Nano local decoder returned an invalid frame width");
        }
        generated.insert(generated.end(), frame.begin(), frame.end());
        rows.push_back(local_frame_decoder_.assistant_slot_token_id());
        rows.insert(rows.end(), frame.begin(), frame.end());
    }
    if (generated.empty()) {
        throw std::runtime_error("MOSS-TTS-Nano generation produced no audio frames");
    }
    MossTTSNanoAudioCodes out;
    out.frames = static_cast<int64_t>(generated.size()) / (prompt.row_width - 1);
    out.codebooks = prompt.row_width - 1;
    out.hit_max_new_frames = !stopped_on_eoc && out.frames >= options.max_new_frames;
    out.token_ids = std::move(generated);
    return out;
}

}  // namespace engine::models::moss_tts_nano
