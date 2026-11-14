#include "engine/community_models/zonos3/generator.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::zonos3 {

Zonos3Codes generate_codes(const Zonos3Hparams & hp, Zonos3Backbone & backbone, const Zonos3Prompt & prompt,
                           const Zonos3Prompt * twin, float cfg_scale, int max_frames,
                           const Zonos3SamplingOptions & sampling, const Zonos3RowCallback & on_row) {
    if (max_frames <= 0) throw std::runtime_error("zonos3: max_frames must be positive");
    const int ncb = hp.n_codebooks;
    const size_t logit_count = static_cast<size_t>(hp.audio_vocab) * static_cast<size_t>(ncb);
    const int slots = twin ? 2 : 1;
    backbone.begin(slots, std::max(prompt.n_rows, twin ? twin->n_rows : 0) + max_frames);
    std::vector<float> logits, twin_logits, guided(twin ? logit_count : 0);
    backbone.prefill(0, prompt, logits);
    if (twin) backbone.prefill(1, *twin, twin_logits);

    Zonos3Sampler sampler(sampling, ncb, hp.audio_vocab, hp.codebook_size, hp.audio_pad_id, hp.eoa_id);
    Zonos3Codes out;
    std::vector<int32_t> frame(static_cast<size_t>(ncb));
    std::vector<int32_t> next(static_cast<size_t>(ncb) + 1);
    std::vector<float> step_logits;
    int countdown = -1;
    for (int step = 0;; ++step) {
        const float * source = logits.data();
        if (twin) {
            zonos3_cfg_mix(logits.data(), twin_logits.data(), cfg_scale, logit_count, guided.data());
            source = guided.data();
        }
        for (int cb = 0; cb < ncb; ++cb) {
            frame[static_cast<size_t>(cb)] = sampler.sample(source + static_cast<size_t>(cb) * static_cast<size_t>(hp.audio_vocab), cb);
        }
        // Cold start: codebooks whose delayed stream has not started (the stream already holds the prefill rows).
        const int row = prompt.prefill_frames + step;
        for (int cb = row + 1; cb < ncb; ++cb) frame[static_cast<size_t>(cb)] = hp.audio_pad_id;
        sampler.accept(frame);
        out.rows.insert(out.rows.end(), frame.begin(), frame.end());
        ++out.count;
        int max_eoa = -1;
        for (int cb = 0; cb < ncb; ++cb) {
            if (frame[static_cast<size_t>(cb)] == hp.eoa_id) max_eoa = cb;
        }
        if (out.eos_frame < 0 && max_eoa >= 0) {
            out.eos_frame = std::max(0, row - max_eoa);
            countdown = ncb + 1;
        }
        if (on_row) on_row(frame.data(), out.eos_frame);
        const bool finished = countdown > 0 && --countdown == 0;
        if (finished) break;
        if (out.count >= max_frames) {
            out.reached_limit = out.eos_frame < 0;
            break;
        }
        std::copy(frame.begin(), frame.end(), next.begin());
        next[static_cast<size_t>(ncb)] = hp.text_vocab;
        backbone.step({next.data(), next.data()}, slots, step_logits);
        std::copy(step_logits.begin(), step_logits.begin() + static_cast<std::ptrdiff_t>(logit_count), logits.begin());
        if (twin) {
            twin_logits.assign(step_logits.begin() + static_cast<std::ptrdiff_t>(logit_count), step_logits.end());
        }
    }
    return out;
}

}  // namespace engine::models::zonos3
