// ovz_engine.h - OmniVoice 28-layer diffusion-LM trunk on the VENDOR zero-op
// rknn_matmul API (librknnrt, RK3588).
//
// Usage: open() -> set_weights() -> pack_weights() -> ensure_geometry() ->
// prefill() once (full-canvas forward; per-layer KV caches end holding the
// prompt prefix = canvas rows [0, seq_pad - tgt_pad)) -> step() per diffusion
// step (target rows only; target KV appended + truncated per layer).
// GEMMs: per-(layer,shape) rknn_matmul contexts, fp16-in/fp32-out, weights
// prepacked once into the native tiled B layout (M-independent -> all buckets
// share); disk cache keyed by fingerprint under an ovz_ filename prefix. The
// vendor zero-op matmul is single-core (core_mask accepted-then-ignored, so it
// is never set). fp32 host chain (rmsnorm/rope/silu) between GEMMs. Attention
// is host-assembled from tiles: per (batch, head) QK^T + fp32 host softmax +
// PV over the FULL cache width (the additive mask selects the visible set,
// the window is never truncated); the dLLM step canvas masks make every target
// row self-only (visible set = its own column), and for that case the step
// attention is the exact identity on V: ctx[row] = V_scaled[row], zero NPU
// work. V prescaled by powers of two to stay in fp16 LUT-exp headroom, the
// inverse folded into the residual add (inert under the host softmax, kept so
// ctxh semantics and dbg_p stay stable across geometries).
//
// Cacheable rknn_create_mem buffers + explicit rknn_mem_sync (TO_DEVICE after
// every CPU write into A/B, FROM_DEVICE after run before any CPU read of C):
// a missing sync is the deterministic-garbage bug the gate bench measured.
//
// Numerics validated by the gate-passing bench fullstep_v.cpp (seq1536
// geometry: SEQ 1536 = OFF 1024 + TGT 512).
#ifndef OVZ_ENGINE_H
#define OVZ_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

namespace ovz {

constexpr int kHid = 1024;
constexpr int kNH = 16;
constexpr int kNKV = 8;
constexpr int kHD = 128;
constexpr int kInter = 3072;
constexpr int kLayers = 28;
constexpr int kQrows = 128;  // attention query chunk (vendor tile size)

// One layer's host weights as fp16 bit patterns, row-major as in safetensors
// ([N,K] for projections, [K] for norms). Pointers must stay valid through
// pack_weights() (which packs them into the native tiled B layout) and through
// prefill/step (the norm weights are applied on the host chain).
struct LayerW {
    const uint16_t *q, *k, *v, *o, *gate, *up, *down;  // [N,K]
    const uint16_t *n_in, *n_post, *n_qn, *n_kn;       // [Hid],[Hid],[D],[D]
};

struct Timings {  // microseconds, accumulated per call
    double gemm = 0, attn = 0, cpu = 0, kv = 0, pack = 0, unpack = 0;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // npu_dev: ignored (the vendor API has no device node; contexts are per
    // (M,K,N) shape, created in pack_weights()/ensure_geometry()).
    // weight_cache_dir: nullptr disables the packed-weight disk cache;
    // fingerprint identifies the exact weight set (e.g. sha256 of the
    // safetensors header).
    int open(const char* npu_dev, const char* weight_cache_dir, const std::string& fingerprint);
    void close();

    void set_weights(const std::vector<LayerW>& w);  // exactly kLayers entries

    // Pack per-(layer,shape) B operands into the native tiled layout
    // (M-independent -> all buckets share) and sync to device. Warm: loads the
    // disk cache (files "ovz_<fingerprint>.<layer>.<shape>.bin" = packed body
    // + FNV-1a-64 trailer, verified per file; any miss falls back to a cold
    // repack); cold: packs and writes it (creating the cache dir; each file is
    // published via temp+fsync+rename so a crashed write can never pass the
    // warm checks; cache I/O problems are non-fatal and land in
    // last_warning()).
    int pack_weights();

    // Vendor contexts + activation buffers for a canvas geometry.
    // seq_pad/tgt_pad: canvas / target rows per batch, each a multiple of
    // kQrows, tgt<=seq. Growing the geometry rebuilds the matmul contexts.
    int ensure_geometry(int seq_pad, int tgt_pad);

    // Full-canvas forward, fills the per-layer caches.
    // h_canvas: [2, seq_pad, kHid] fp32, rows [seq..seq_pad) must be zero.
    // cos2/sin2: [2*seq, kHD] fp32 RoPE tables (CFG-doubled, canvas order).
    // mask_bias: [2, seq, seq] additive float attention bias (keep > -1000;
    // canvas pad rows/cols become fully hidden, pad rows self-diagonal).
    // h_out: [2, seq_pad, kHid] post-trunk canvas states.
    int prefill(const float* h_canvas, const float* cos2, const float* sin2,
                const float* mask_bias, int seq, int seq_pad, int tgt,
                float* h_out, Timings* t = nullptr);

    // Target-rows-only forward over rows [seq-tgt, seq_pad) of the canvas.
    // h_tgt: [2, tgt_pad, kHid] fp32 (rows [tgt..tgt_pad) zero); h_out:
    // [2, tgt_pad, kHid]. cos/sin: [2*seq, kHD] plain rope tables - entry
    // b*seq+row carries the position that pass's step rows read (cond = canvas
    // row, uncond = block-local row), NOT prefill's canvas/dual-zone tables
    // (they collide on canvas rows [seq-tgt, tgt) when the target block is
    // bigger than the prompt); mask_bias: as in prefill. Caches keep the
    // prompt prefix between steps.
    int step(const float* h_tgt, const float* cos2, const float* sin2,
             const float* mask_bias, int seq, int tgt, int tgt_pad,
             float* h_out, Timings* t = nullptr);

    int caches_truncate(int len);  // all 56 caches -> len

    const std::string& last_error() const;
    // non-fatal disk-cache problems from pack_weights() (mkdir/write/flush
    // failures); the engine runs normally with them, the caller logs it
    const std::string& last_warning() const;

    // --- debug gate hooks (valid after step(); bench-SNR validation only) ---
    void set_layer_hook(void (*fn)(int layer, void* ud), void* ud);
    const uint16_t* dbg_q16() const;    // [2,NH,seq_pad,HD] rope'd (head stride seq_pad*HD, target rows at their canvas row)
    const uint16_t* dbg_k16() const;    // [2,NKV,seq_pad,HD] rope'd, unscaled (canvas-strided)
    const uint16_t* dbg_ctxh() const;   // [2,nrows,NH*HD] attention ctx, V-scaled by p(b); nrows = seq_pad after prefill, tgt_pad after step
    const float* dbg_h() const;         // [2,seq_pad,kHid] in-place canvas; after step only rows [seq-tgt, seq_pad) are fresh
    float dbg_p(int b, int l = 0) const;  // V prescale of (batch, layer), valid inside the layer hook

private:
    struct Impl;
    Impl* im_;
};

}  // namespace ovz
#endif  // OVZ_ENGINE_H
