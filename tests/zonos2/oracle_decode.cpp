// Teacher-forced decode logits from Zyphra/zonos2.cpp's own KV-cache runtime, for zonos2_parity backbone.
// Built and run by make_oracle.sh against a zonos2.cpp build tree; not part of the audio.cpp build.
//   zonos2-oracle-decode <model.gguf> <ids.npy> <prompt.npy> <out.npy> [<speaker.npy>]
// ids.npy is zonos2-cli --dump-ids output [n0 + frames, 10], prompt.npy the --build-prompt rows [n0, 10]. Prefills the
// prompt, then feeds every generated row through zonos2_batch_step; writes the post-softcap logits
// [1 + frames, 9, 1026] (row j predicts frame j) with a KV capacity of round_up_256(n0 + frames + 8).
#include "npy.h"
#include "zonos2.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <ids.npy> <prompt.npy> <out.npy> [<speaker.npy>]\n", argv[0]);
        return 1;
    }
    std::vector<float> ids, prompt, speaker;
    std::vector<int64_t> ids_shape, prompt_shape, speaker_shape;
    if (!npy::load_f32(argv[2], ids, ids_shape) || !npy::load_f32(argv[3], prompt, prompt_shape) ||
        ids_shape.size() != 2 || prompt_shape.size() != 2 || ids_shape[1] != prompt_shape[1] ||
        ids_shape[0] <= prompt_shape[0] || std::memcmp(ids.data(), prompt.data(), prompt.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "ids.npy must start with the prompt.npy rows and add generated rows\n");
        return 1;
    }
    if (argc == 6 && !npy::load_f32(argv[5], speaker, speaker_shape)) {
        std::fprintf(stderr, "cannot read %s\n", argv[5]);
        return 1;
    }
    zonos2_model model;
    if (!zonos2_model_load(model, argv[1], /*use_gpu=*/false)) return 1;
    const int width = static_cast<int>(ids_shape[1]);
    const int ncb = static_cast<int>(model.hp.n_codebooks);
    const int vocab = static_cast<int>(model.hp.audio_vocab);
    const int n0 = static_cast<int>(prompt_shape[0]);
    const int frames = static_cast<int>(ids_shape[0]) - n0;
    const size_t per_row = static_cast<size_t>(ncb) * static_cast<size_t>(vocab);
    std::vector<float> logits(per_row * static_cast<size_t>(frames + 1));
    zonos2_batch_ctx batch;
    if (!zonos2_batch_init(batch, model, 1, n0 + frames + 8) ||
        !zonos2_batch_slot_prefill(batch, 0, ids.data(), n0, speaker.empty() ? nullptr : speaker.data(), 0, logits.data())) {
        std::fprintf(stderr, "prefill failed\n");
        return 1;
    }
    zonos2_slot slot;
    slot.index = 0;
    slot.active = true;
    slot.n_past = n0;
    slot.next_ids.resize(static_cast<size_t>(width));
    const std::vector<zonos2_slot *> active = {&slot};
    for (int j = 0; j < frames; ++j) {
        for (int k = 0; k < width; ++k) {
            slot.next_ids[static_cast<size_t>(k)] = static_cast<int32_t>(std::lround(ids[static_cast<size_t>(n0 + j) * width + k]));
        }
        zonos2_batch_step(batch, active);
        std::memcpy(logits.data() + per_row * static_cast<size_t>(j + 1), zonos2_batch_slot_logits(batch, 0), per_row * sizeof(float));
    }
    npy::save_f32(argv[4], logits.data(), {static_cast<int64_t>(frames + 1), ncb, vocab});
    std::printf("wrote %s [%d, %d, %d]\n", argv[4], frames + 1, ncb, vocab);
    zonos2_batch_free(batch);
    zonos2_model_free(model);
    return 0;
}
