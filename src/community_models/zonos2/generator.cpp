#include "engine/community_models/zonos2/generator.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::zonos2 {

Zonos2Codes generate_zonos2_codes(const Zonos2Hparams & hp, Zonos2Backbone & backbone, const Zonos2Prompt & prompt,
                                  const float * speaker, int max_frames, const Zonos2SamplingOptions & sampling,
                                  const Zonos2RowCallback & on_row) {
    if (max_frames <= 0) throw std::runtime_error("zonos2: max_frames must be positive");
    const int ncb = hp.n_codebooks;
    backbone.begin(prompt.n_rows + max_frames + 8);
    std::vector<float> logits;
    backbone.prefill(prompt, speaker, logits);

    Zonos2Sampler sampler(sampling, ncb, hp.audio_vocab, hp.codebook_size);
    Zonos2Codes out;
    std::vector<int32_t> frame(static_cast<size_t>(ncb));
    std::vector<int32_t> next(static_cast<size_t>(ncb) + 1);
    int countdown = -1;
    for (int step = 0;; ++step) {
        for (int cb = 0; cb < ncb; ++cb) {
            frame[static_cast<size_t>(cb)] = sampler.sample(logits.data() + static_cast<size_t>(cb) * static_cast<size_t>(hp.audio_vocab), cb);
        }
        sampler.accept(frame);
        out.rows.insert(out.rows.end(), frame.begin(), frame.end());
        ++out.count;
        int max_eoa = -1;
        for (int cb = 0; cb < ncb; ++cb) {
            if (frame[static_cast<size_t>(cb)] == hp.eoa_id) max_eoa = cb;
        }
        if (out.eos_frame < 0 && max_eoa >= 0) {
            out.eos_frame = std::max(0, step - max_eoa);
            countdown = kZonos2EosDrainFrames;
        }
        if (on_row) on_row(frame.data(), out.eos_frame);
        if (countdown > 0 && --countdown == 0) break;
        if (out.count >= max_frames) {
            out.reached_limit = out.eos_frame < 0;
            break;
        }
        std::copy(frame.begin(), frame.end(), next.begin());
        next[static_cast<size_t>(ncb)] = hp.text_vocab;
        backbone.step(next.data(), logits);
    }
    return out;
}

}  // namespace engine::models::zonos2
