#pragma once

// Vendor zero-op (librknnrt rknn_matmul_api) backend for the OmniVoice
// diffusion-LM generator: the same 28-layer trunk and front-shard contract as
// the compiled rknn ladder, run on the vendor single-core matmul API with
// tile-assembled attention and a host fp32 softmax, so no compiled per-bucket
// engine files are needed: weights are packed once into the native tiled
// layouts (disk-cached) and the geometry is built at runtime.
//
// Contract = the front-shard contract (see rknn_generator.h): run() writes raw
// post-trunk hidden rows in the packed host layout, the CFG tail (final norm,
// audio heads, guided combine) stays in the host fp32 tail graph. Board gate
// (2026-11-15, RK3588): 28-layer step/prefill referee ladder 80.6-132.9 dB
// against the compiled front-shard ground truth; twin run vs compiled: smooth
// ladder decay, whisper-clean output; step ~22.9 s vs the compiled engine's
// 23.2 s at seq1536 (deep shapes at compiled parity; the twin bucket is
// slower, ~7.5 s/step at seq896 geometry, prefill included).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/models/omnivoice/rknn_generator.h"

namespace engine::models::omnivoice {

// One layer's host weight views, straight out of the runtime's weight store
// (generator.cpp Qwen3DiffusionLayerWeights). `qkv`/`gate_up` carry the packed
// row order [q|k|v] / [gate|up] when `packed` is set; otherwise q/k/v and
// gate/up point at the separate tensors. `*_is_f16` says whether the buffer
// holds fp16 bits (GGML_TYPE_F16) or fp32 floats to convert at pack time.
struct OmniVoiceZeroopLayerView {
    const void * qkv, * q, * k, * v;
    const void * o;
    const void * gate_up;  // packed row order [gate | up]
    const void * gate, * up, * down;
    const void * n_in, * n_post, * n_qn, * n_kn;  // always fp32 (load_f32_tensor)
    bool packed = false;
    bool qkv_is_f16 = true, o_is_f16 = true, gu_is_f16 = true, down_is_f16 = true;
    bool norms_is_f16 = true;
};

struct OmniVoiceZeroopWeightViews {
    std::vector<OmniVoiceZeroopLayerView> layers;
    const void * text_embedding = nullptr;   // [text_embedding_rows, hidden]
    int64_t text_embedding_rows = 0;
    const void * audio_embedding = nullptr;  // [audio_embedding_rows, hidden]
    int64_t audio_embedding_rows = 0;
    bool text_embedding_is_f16 = true, audio_embedding_is_f16 = true;
};

struct OmniVoiceZeroopModelDims {
    int64_t codebooks = 8;
    int64_t audio_vocab = 1025;
    int64_t heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t hidden = 1024;
    int64_t intermediate = 3072;
    double rope_theta = 1000000.0;
};

class OmniVoiceRknnZeroopGenerator : public OmniVoiceRknnGenerator {
public:
    // Views must stay valid for the generator's lifetime (they do: the weight
    // store outlives the sidecar in OmniVoiceQwen3DiffusionRuntime::Impl).
    OmniVoiceRknnZeroopGenerator(
        const OmniVoiceZeroopWeightViews & weights,
        const OmniVoiceZeroopModelDims & dims,
        const std::string & weight_cache_dir,
        std::string fingerprint);
    ~OmniVoiceRknnZeroopGenerator() override;

    int64_t max_tokens() const override;
    int64_t engine_count() const override;
    bool front_shard() const override;
    int64_t bucket_for(int64_t total_tokens) const override;
    void run(int64_t bucket, const OmniVoiceRknnStepInputs & inputs, float * out, int64_t row_capacity) override;
    // New request: the next run() re-prefills (caches are per-request state).
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::omnivoice
