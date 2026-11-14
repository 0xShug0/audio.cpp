#include "engine/community_models/zonos3/generator.h"

#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::models::zonos3 {

Zonos3Codes generate_codes(const Zonos3Hparams & hp, Zonos3Backbone & backbone, const Zonos3Prompt & prompt,
                           const Zonos3Prompt * twin, float guidance_scale, int max_frames,
                           const Zonos3SamplingOptions & sampling, const Zonos3RowCallback & on_row) {
    if (max_frames <= 0) throw std::runtime_error("zonos3: max_frames must be positive");
    const int ncb = hp.n_codebooks;
    const size_t logit_count = static_cast<size_t>(hp.audio_vocab) * static_cast<size_t>(ncb);
    const int slots = twin ? 2 : 1;
    const auto begin_started = std::chrono::steady_clock::now();
    backbone.begin(slots, std::max(prompt.n_rows, twin ? twin->n_rows : 0) + max_frames);
    std::vector<float> logits, twin_logits, guided(twin ? logit_count : 0);
    logits.reserve(static_cast<size_t>(slots) * logit_count);
    const auto prefill_started = std::chrono::steady_clock::now();
    const double begin_ms = std::chrono::duration<double, std::milli>(prefill_started - begin_started).count();
    backbone.prefill(0, prompt, logits);
    if (twin) backbone.prefill(1, *twin, twin_logits);
    const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefill_started).count();
    double sampling_ms = 0, decode_ms = 0, callback_ms = 0;
    const float * unconditional = twin_logits.data();

    Zonos3Sampler sampler(sampling, ncb, hp.audio_vocab, hp.codebook_size, hp.audio_pad_id, hp.eoa_id);
    Zonos3Codes out;
    std::vector<int32_t> frame(static_cast<size_t>(ncb));
    std::vector<int32_t> next(static_cast<size_t>(ncb) + 1);
    int countdown = -1;
    for (int step = 0;; ++step) {
        const auto sampling_started = std::chrono::steady_clock::now();
        const float * source = logits.data();
        if (twin) {
            zonos3_cfg_mix(logits.data(), unconditional, guidance_scale, logit_count, guided.data());
            source = guided.data();
        }
        for (int cb = 0; cb < ncb; ++cb) {
            frame[static_cast<size_t>(cb)] = sampler.sample(source + static_cast<size_t>(cb) * static_cast<size_t>(hp.audio_vocab), cb);
        }
        // Cold start: codebooks whose delayed stream has not started (the stream already holds the prefill rows).
        const int row = prompt.prefill_frames + step;
        for (int cb = row + 1; cb < ncb; ++cb) frame[static_cast<size_t>(cb)] = hp.audio_pad_id;
        sampler.accept(frame);
        ++out.count;
        int max_eoa = -1;
        for (int cb = 0; cb < ncb; ++cb) {
            if (frame[static_cast<size_t>(cb)] == hp.eoa_id) max_eoa = cb;
        }
        if (out.eos_frame < 0 && max_eoa >= 0) {
            out.eos_frame = std::max(0, row - max_eoa);
            countdown = ncb + 1;
        }
        sampling_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sampling_started).count();
        const auto callback_started = std::chrono::steady_clock::now();
        if (on_row) on_row(frame.data(), out.eos_frame);
        callback_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callback_started).count();
        const bool finished = countdown > 0 && --countdown == 0;
        if (finished) break;
        if (out.count >= max_frames) {
            out.reached_limit = out.eos_frame < 0;
            break;
        }
        std::copy(frame.begin(), frame.end(), next.begin());
        next[static_cast<size_t>(ncb)] = hp.text_vocab;
        const auto decode_started = std::chrono::steady_clock::now();
        backbone.step({next.data(), next.data()}, slots, logits);
        decode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_started).count();
        if (twin) unconditional = logits.data() + logit_count;
    }
    debug::timing_log_scalar("zonos3.backbone.prepare_ms", begin_ms);
    debug::timing_log_scalar("zonos3.backbone.prefill_ms", prefill_ms);
    debug::timing_log_scalar("zonos3.backbone.decode_ms", decode_ms);
    debug::timing_log_scalar("zonos3.sampling_ms", sampling_ms);
    debug::timing_log_scalar("zonos3.emit_ms", callback_ms);
    return out;
}

}  // namespace engine::models::zonos3
