#include "engine/models/moss/moss_tts_nano/generator.h"

#include "engine/framework/debug/trace.h"

#include <algorithm>
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
    const MossTTSNanoGenerationOptions & options,
    const std::function<void(const std::vector<int32_t> &)> & on_frame) {
    if (prompt.rows <= 0 || prompt.row_width <= 1) {
        throw std::runtime_error("MOSS-TTS-Nano generator requires a non-empty prompt");
    }
    if (options.active_codebooks <= 0 || options.max_new_frames <= 0) {
        throw std::runtime_error("MOSS-TTS-Nano generator requires positive generation limits");
    }
    // The global transformer keeps a K/V cache: the prompt is forwarded once and every later frame
    // only appends its own single row, so the per-step cost no longer grows with the transcript.
    global_transformer_.begin_generation(prompt.rows + options.max_new_frames);
    local_frame_decoder_.prepare(options.active_codebooks);
    // The chat template dominates the prompt (70+ rows) and is identical for every utterance in a
    // session, so the K/V rows shared with the previous prompt are reused instead of re-forwarded.
    // begin_generation() above clears the cache whenever it had to rebuild the graph, in which case
    // cached_positions() is 0 and nothing is reused.
    int64_t reuse_rows = 0;
    if (previous_row_width_ == prompt.row_width && previous_prompt_rows_ > 0) {
        const int64_t limit = std::min(prompt.rows, previous_prompt_rows_);
        while (reuse_rows < limit) {
            const auto offset = static_cast<std::ptrdiff_t>(reuse_rows * prompt.row_width);
            if (!std::equal(
                    prompt.input_ids.begin() + offset,
                    prompt.input_ids.begin() + offset + prompt.row_width,
                    previous_prompt_ids_.begin() + offset)) {
                break;
            }
            ++reuse_rows;
        }
    }
    reuse_rows = std::min(reuse_rows, global_transformer_.cached_positions());
    debug::trace_log_scalar("moss_tts_nano.gen.prompt_rows", prompt.rows);
    debug::trace_log_scalar("moss_tts_nano.gen.prefix_reuse_rows", reuse_rows);

    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(options.max_new_frames * (prompt.row_width - 1)));
    std::mt19937 rng(options.seed);
    uint64_t sample_call_index = 0;
    bool stopped_on_eoc = false;
    std::vector<float> hidden =
        global_transformer_.prefill(prompt.input_ids, prompt.rows, prompt.row_width, reuse_rows);
    for (int64_t step = 0; step < options.max_new_frames; ++step) {
        const auto frame = local_frame_decoder_.generate_frame(
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
        global_transformer_.step_into(
            local_frame_decoder_.assistant_slot_token_id(),
            frame,
            hidden);
        if (on_frame) on_frame(frame);
    }
    if (generated.empty()) {
        throw std::runtime_error("MOSS-TTS-Nano generation produced no audio frames");
    }
    previous_prompt_ids_ = prompt.input_ids;
    previous_prompt_rows_ = prompt.rows;
    previous_row_width_ = prompt.row_width;
    MossTTSNanoAudioCodes out;
    out.frames = static_cast<int64_t>(generated.size()) / (prompt.row_width - 1);
    out.codebooks = prompt.row_width - 1;
    out.hit_max_new_frames = !stopped_on_eoc && out.frames >= options.max_new_frames;
    out.token_ids = std::move(generated);
    return out;
}

}  // namespace engine::models::moss_tts_nano
