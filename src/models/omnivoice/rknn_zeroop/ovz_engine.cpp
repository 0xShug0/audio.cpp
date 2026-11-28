#if defined(ENGINE_HAVE_RKNN)  // vendor zero-op rknn_matmul API (librknnrt): AGPL-clean path
// ovz_engine.cpp - see ovz_engine.h. Port of the gate-passing vendor bench
// fullstep_v.cpp: same fp32 host chain
// (rmsnorm/rope/silu/residual in NEON fp32, fp16 only at GEMM inputs), same
// per-layer host KV caches with the power-of-two V prescale, same
// self-only-step identity on V. Attention is host-assembled from tiles:
// QK^T + host-stable fp32 softmax + PV tiles over the full cache width
// (mask selects the visible set, the window is never truncated). Weights are
// packed once into the official demo native layouts (A perf [K1,M,subK],
// B [N1,K1,subN,subK], C perf [N1,M,subN]) with the bench's pack helpers.
#include "ovz_engine.h"
#if defined(__aarch64__)

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <system_error>
#include <tuple>
#include <time.h>
#include <unistd.h>
#include <utility>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

namespace ovz {
namespace {

constexpr int kNqkv = (kNH + 2 * kNKV) * kHD;  // 4096
constexpr int kNo = kHid, kNgu = 2 * kInter, kNd = kHid;
constexpr int kKqkv = kHid, kKo = kNH * kHD, kKgu = kHid, kKd = kInter;
constexpr int kSn[4] = {kNqkv, kNo, kNgu, kNd}, kSk[4] = {kKqkv, kKo, kKgu, kKd};
constexpr float kEps = 1e-6f;
constexpr float kScale = 1.0f / sqrtf((float)kHD);
constexpr float kHidMask = -1e30f;  // additive stand-in for a fully hidden slot

inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
inline uint16_t f2h(float x) { __fp16 v = (__fp16)x; uint16_t r; memcpy(&r, &v, 2); return r; }
inline float h2f(uint16_t x) { __fp16 v; memcpy(&v, &x, 2); return (float)v; }

void rmsnorm_neon(const float* x, const uint16_t* w, uint16_t* o, long rows, int cols) {
    for (long r = 0; r < rows; ++r) {
        const float* p = x + (size_t)r * cols;
        float32x4_t acc = vdupq_n_f32(0);
        for (int c = 0; c < cols; c += 4) acc = vmlaq_f32(acc, vld1q_f32(p + c), vld1q_f32(p + c));
        float k = 1.0f / sqrtf((acc[0] + acc[1] + acc[2] + acc[3]) / cols + kEps);
        float32x4_t kq = vdupq_n_f32(k);
        for (int c = 0; c < cols; c += 4) {
            float32x4_t v = vmulq_f32(vld1q_f32(p + c), vmulq_f32(kq, vcvt_f32_f16(vld1_f16((const __fp16*)w + c))));
            vst1_f16((__fp16*)o + (size_t)r * cols + c, vcvt_f16_f32(v));
        }
    }
}
void silu_neon(const float* g, const float* u, uint16_t* o, long n) {
    long i = 0;
    for (; i + 4 <= n; i += 4) {
        const float* g4 = g + i;
        float r[4] = {g4[0] / (1 + expf(-g4[0])), g4[1] / (1 + expf(-g4[1])), g4[2] / (1 + expf(-g4[2])), g4[3] / (1 + expf(-g4[3]))};
        float32x4_t mv = vmulq_f32(vld1q_f32(r), vld1q_f32(u + i));
        vst1_f16((__fp16*)o + i, vcvt_f16_f32(mv));
    }
    for (; i < n; ++i) o[i] = f2h((g[i] / (1.0f + expf(-g[i]))) * u[i]);
}

// ---- vendor zero-op pack/unpack (NATIVE layouts, official demo packers) ----
// A perf [K1, M, subK] from src [M,K]
void pack_a(const uint16_t* src, uint16_t* dst, int M, int K, int subK) {
    int K1 = (K + subK - 1) / subK;
    for (int i = 0; i < K1; ++i)
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < subK; ++j) {
                int ki = i * subK + j;
                dst[(size_t)i * M * subK + (size_t)m * subK + j] = ki < K ? src[(size_t)m * K + ki] : (uint16_t)0;
            }
}
// B native [N1, K1, subN, subK] from src [N,K] rows (B[n][k] = src[n*K+k])
void pack_b(const uint16_t* src, uint16_t* dst, int K, int N, int subN, int subK) {
    int N1 = (N + subN - 1) / subN, K1 = (K + subK - 1) / subK;
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < K1; ++j)
            for (int n = 0; n < subN; ++n) {
                int ni = i * subN + n;
                size_t base = ((size_t)i * K1 + j) * subN * subK;
                for (int k = 0; k < subK; ++k) {
                    int ki = j * subK + k;
                    dst[base + (size_t)n * subK + k] = (ni < N && ki < K) ? src[(size_t)ni * K + ki] : (uint16_t)0;
                }
            }
}
// B native [N1, K1, subN, subK] where B[n][k] = V[k][n]; src is V [K, N] (K rows of N)
void pack_b_v(const uint16_t* src, uint16_t* dst, int K, int N, int subN, int subK) {
    int N1 = (N + subN - 1) / subN, K1 = (K + subK - 1) / subK;
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < K1; ++j)
            for (int n = 0; n < subN; ++n) {
                int ni = i * subN + n;
                size_t base = ((size_t)i * K1 + j) * subN * subK;
                for (int k = 0; k < subK; ++k) {
                    int ki = j * subK + k;
                    dst[base + (size_t)n * subK + k] = (ni < N && ki < K) ? src[(size_t)ki * N + ni] : (uint16_t)0;
                }
            }
}
// C perf [N1, M, subN] -> dst [M,N]
void unpack_c(const float* src, float* dst, int M, int N, int N1, int subN) {
    for (int i = 0; i < N1; ++i)
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < subN; ++j) {
                int ni = i * subN + j;
                if (ni < N) dst[(size_t)m * N + ni] = src[(size_t)i * M * subN + (size_t)m * subN + j];
            }
}

}  // namespace

struct Engine::Impl {
    std::vector<LayerW> w;
    std::string cache_dir, fp, err, warn;
    bool opened = false, packed = false;

    // main GEMM slots: per-(M-slot,shape) 28 layer ctx sharing A/C (bench
    // mainslot); [0]=prefill M=2*seq_pad, [1]=step M=2*tgt_pad
    struct MS {
        int M = 0;  // 0 = not created
        int K = 0, N = 0;
        int subKA = 0, subNB = 0, subKB = 0, subNC = 0, N1C = 0;
        rknn_matmul_io_attr io{};
        rknn_tensor_mem* A = nullptr;
        rknn_tensor_mem* C = nullptr;
        rknn_matmul_ctx ctx[kLayers]{};
    };
    MS ms[2][4]{};
    struct Wt {
        rknn_tensor_mem* mem = nullptr;
        rknn_matmul_ctx sync = 0;  // creating ctx, used for rknn_mem_sync
    };
    Wt wt[kLayers][4]{};
    rknn_matmul_ctx pack_ctx[4]{};   // io-size query + weight-mem creator (kept alive: Wb mems outlive it)
    rknn_matmul_io_attr pack_io[4]{};

    // attention slots: shared A/B/C scratch per (M,K,N), per-shape ctx
    struct AS {
        int M = 0, K = 0, N = 0;
        int subKA = 0, subNB = 0, subKB = 0, subNC = 0, N1C = 0;
        rknn_matmul_ctx ctx = 0;
        rknn_matmul_io_attr io{};
        rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
    };
    std::map<std::tuple<int, int, int>, AS> asl;  // node-stable: pointers survive insert

    // geometry state (max ever requested)
    int seq_pad = 0, tgt_pad = 0, cap = 0;
    int klen[2] = {0, 0};  // logical KV cache length per batch (cap-stride caches)
    std::vector<uint16_t> kcache, vcache;  // [l][b][kvhead][cap][HD] fp16, V stored = v/p
    std::vector<uint16_t> x16, q16, k16, v16, ctxh, mid16, P16;
    std::vector<float> qkv32, o32, gu32, d32, h, ht, Sbuf, ctxbuf, maskfm;
    std::vector<uint8_t> maskb;  // [2, sp, sp] boolean, rebuilt per call
    int mask_sp = 0;             // maskfm/maskb row stride (canvas seq_pad)
    float p_scale[2][kLayers]{};
    void (*hook)(int, void*) = nullptr;
    void* hook_ud = nullptr;

    void ms_destroy(MS& m) {
        if (m.A) rknn_destroy_mem(m.ctx[0], m.A);
        if (m.C) rknn_destroy_mem(m.ctx[0], m.C);
        for (int l = 0; l < kLayers; ++l)
            if (m.ctx[l]) { rknn_matmul_destroy(m.ctx[l]); m.ctx[l] = 0; }
        m = MS{};
    }

    // (Re)create the M-slots for the current max geometry and bind A/C + the
    // persistent weights on every layer ctx (bench mainslot_init + Wb bind).
    // core_mask is accepted-then-ignored by the vendor single-core zero-op path,
    // so it is never set.
    int ms_init_all() {
        const int Ms[2] = {2 * seq_pad, 2 * tgt_pad};
        for (int mi = 0; mi < 2; ++mi)
            for (int s = 0; s < 4; ++s) {
                MS& m = ms[mi][s];
                if (m.M == Ms[mi]) continue;
                ms_destroy(m);
                if (!Ms[mi]) continue;  // tgt_pad == 0: no step slots
                rknn_matmul_info info{};
                info.M = Ms[mi]; info.K = kSk[s]; info.N = kSn[s];
                info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
                info.B_layout = RKNN_MM_LAYOUT_NATIVE;
                info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
                for (int l = 0; l < kLayers; ++l)
                    if (rknn_matmul_create(&m.ctx[l], &info, &m.io)) {
                        err = "mcreate M" + std::to_string(Ms[mi]) + " s" + std::to_string(s);
                        return 1;
                    }
                m.M = Ms[mi];
                m.K = kSk[s]; m.N = kSn[s];
                m.subKA = m.io.A.dims[2]; m.subNB = m.io.B.dims[2]; m.subKB = m.io.B.dims[3];
                m.subNC = m.io.C.dims[2]; m.N1C = m.io.C.dims[0];
                m.A = rknn_create_mem(m.ctx[0], m.io.A.size);
                m.C = rknn_create_mem(m.ctx[0], m.io.C.size);
                if (!m.A || !m.C) { err = "alloc A/C M" + std::to_string(Ms[mi]); return 1; }
                for (int l = 0; l < kLayers; ++l) {
                    rknn_matmul_set_io_mem(m.ctx[l], m.A, &m.io.A);
                    rknn_matmul_set_io_mem(m.ctx[l], m.C, &m.io.C);
                }
            }
        for (int mi = 0; mi < 2; ++mi)
            for (int s = 0; s < 4; ++s)
                for (int l = 0; l < kLayers; ++l)
                    if (ms[mi][s].M &&
                        rknn_matmul_set_io_mem(ms[mi][s].ctx[l], wt[l][s].mem, &ms[mi][s].io.B)) {
                        err = "bind wt l" + std::to_string(l) + " s" + std::to_string(s);
                        return 1;
                    }
        return 0;
    }

    AS* asget(int M, int K, int N) {
        auto it = asl.find(std::make_tuple(M, K, N));
        if (it != asl.end()) return &it->second;
        AS& a = asl[std::make_tuple(M, K, N)];
        a.M = M; a.K = K; a.N = N;
        rknn_matmul_info info{};
        info.M = M; info.K = K; info.N = N;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
        info.iommu_domain_id = 2;  // attention slots live in a separate 4GB IOMMU domain
                                   // (the 224 mslot ctxs exhaust domain 0)
        if (rknn_matmul_create(&a.ctx, &info, &a.io)) {
            asl.erase(std::make_tuple(M, K, N));
            err = "aslot create M" + std::to_string(M) + " K" + std::to_string(K) + " N" + std::to_string(N);
            return nullptr;
        }
        a.A = rknn_create_mem(a.ctx, a.io.A.size);
        a.B = rknn_create_mem(a.ctx, a.io.B.size);
        a.C = rknn_create_mem(a.ctx, a.io.C.size);
        if (!a.A || !a.B || !a.C) { err = "aslot mem"; return nullptr; }
        rknn_matmul_set_io_mem(a.ctx, a.A, &a.io.A);
        rknn_matmul_set_io_mem(a.ctx, a.B, &a.io.B);
        rknn_matmul_set_io_mem(a.ctx, a.C, &a.io.C);
        a.subKA = a.io.A.dims[2]; a.subNB = a.io.B.dims[2]; a.subKB = a.io.B.dims[3];
        a.subNC = a.io.C.dims[2]; a.N1C = a.io.C.dims[0];
        return &a;
    }

    // mi selects the M-slot ([0]=prefill, [1]=step). Every buffer the CPU wrote
    // is synced TO_DEVICE before run, and C FROM_DEVICE before the CPU reads it:
    // the bench's deterministic-garbage lesson. A/C belong to ctx[0] (shared).
    // Note: ms slots run at the MAX geometry M, so for a call with fewer rows
    // the D rows [m_calls, M) of the output carry stale values; no caller reads
    // past its own row count, so the slack is inert (pack_a zero-pads A there,
    // the vendor GEMM still writes a valid C row, just a meaningless one).
    int gemm(int mi, int l, int s, const uint16_t* A, float* D, Timings* t) {
        MS& m = ms[mi][s];
        if (!m.M) { err = "gemm: geometry missing"; return 1; }
        double t0 = now_us();
        pack_a(A, (uint16_t*)m.A->virt_addr, m.M, m.K, m.subKA);
        if (rknn_mem_sync(m.ctx[0], m.A, RKNN_MEMORY_SYNC_TO_DEVICE)) { err = "syncA"; return 1; }
        if (t) t->pack += now_us() - t0;
        t0 = now_us();
        if (rknn_matmul_run(m.ctx[l]) != 0) {
            err = "gemm run l" + std::to_string(l) + " s" + std::to_string(s);
            return 1;
        }
        if (t) t->gemm += now_us() - t0;
        t0 = now_us();
        if (rknn_mem_sync(m.ctx[0], m.C, RKNN_MEMORY_SYNC_FROM_DEVICE)) { err = "syncC"; return 1; }
        unpack_c((const float*)m.C->virt_addr, D, m.M, m.N, m.N1C, m.subNC);
        if (t) t->unpack += now_us() - t0;
        return 0;
    }

    // boolean canvas mask [2,sp,sp] from additive bias, kept in both forms:
    // maskb for the self-only scan, maskfm (additive) for the tile softmax;
    // pad rows self-diagonal (compiled convention: a pad token attends to
    // itself only, never a full row)
    void build_mask(const float* mask_bias, int seq, int sp) {
        mask_sp = sp;
        maskb.assign((size_t)2 * sp * sp, 0);
        maskfm.assign((size_t)2 * sp * sp, kHidMask);
        for (int b = 0; b < 2; ++b) {
            for (int r = 0; r < seq; ++r)
                for (int cl = 0; cl < seq; ++cl) {
                    float bias = mask_bias[((size_t)b * seq + r) * seq + cl];
                    maskb[((size_t)b * sp + r) * sp + cl] = bias > -1000.0f;
                    maskfm[((size_t)b * sp + r) * sp + cl] = bias;
                }
            for (int r = seq; r < sp; ++r) {
                maskb[((size_t)b * sp + r) * sp + r] = 1;
                maskfm[((size_t)b * sp + r) * sp + r] = 0.0f;
            }
            // fully-masked rows (the compiled-convention gap the open library
            // degrades to uniform: e.g. the b1 prompt block of a step canvas)
            // must not run a sum=0 softmax: inv=1/0=inf, f2h(0*inf)=NaN, and
            // the NaN rows poison the caches. Force the same self-diagonal
            // pad convention on any row whose visible set is empty.
            for (int r = 0; r < sp; ++r) {
                uint8_t* prow = &maskb[((size_t)b * sp + r) * sp];
                if (!memchr(prow, 1, (size_t)sp)) {
                    prow[r] = 1;
                    maskfm[((size_t)b * sp + r) * sp + r] = 0.0f;
                }
            }
        }
    }

    // One (batch, head) attention tile: QK^T [qrows,K=HD]x[HD,ctx_rows] +
    // host-stable fp32 softmax (scale + additive mask row, exp accumulated in
    // double, per bench) + PV [qrows,ctx_rows]x[ctx_rows,HD]. kb/vb are
    // kv-head cache slices (V prescaled), mrow points at the (b, mrow0)
    // maskfm row; results land in ctxh rows [b*nrows + crow0, +qrows) as fp16
    // V-scaled ctx (prescaled V, same semantics the compiled ladder sees).
    int attn_tile(int b, int hh, const uint16_t* q, int qrows,
                  const uint16_t* kb, const uint16_t* vb, int ctx_rows,
                  const float* mrow, int qrow0, long crow0, int nrows, Timings* t) {
        // GQA: head hh shares kv-head hh/2. Bucket the cache width to a
        // multiple of kQrows so per-shape aslots (and their device buffers)
        // are reused across prompt lengths instead of leaking 2 per length.
        // The slack columns [ctx_rows, cw) are neutralized by the mask row
        // (fully hidden there -> exp(-1e30-mx)=0 -> P=0), so the stale cache
        // rows the QK^T/PV tail packs contribute exactly 0.
        const int cw = (ctx_rows + kQrows - 1) / kQrows * kQrows;
        AS* s1 = asget(qrows, kHD, cw);
        if (!s1) return 1;
        double t0 = now_us();
        pack_a(q, (uint16_t*)s1->A->virt_addr, qrows, kHD, s1->subKA);
        pack_b(kb, (uint16_t*)s1->B->virt_addr, kHD, cw, s1->subNB, s1->subKB);
        if (rknn_mem_sync(s1->ctx, s1->A, RKNN_MEMORY_SYNC_TO_DEVICE)) { err = "attn syncA"; return 1; }
        if (rknn_mem_sync(s1->ctx, s1->B, RKNN_MEMORY_SYNC_TO_DEVICE)) { err = "attn syncB"; return 1; }
        if (t) t->pack += now_us() - t0;
        t0 = now_us();
        if (rknn_matmul_run(s1->ctx) != 0) { err = "attn QK run"; return 1; }
        if (t) t->attn += now_us() - t0;
        t0 = now_us();
        if (rknn_mem_sync(s1->ctx, s1->C, RKNN_MEMORY_SYNC_FROM_DEVICE)) { err = "attn syncC"; return 1; }
        unpack_c((const float*)s1->C->virt_addr, Sbuf.data(), qrows, cw, s1->N1C, s1->subNC);
        if (t) t->unpack += now_us() - t0;
        t0 = now_us();
        for (int r = 0; r < qrows; ++r) {
            const float* mb = mrow + (size_t)(qrow0 + r) * mask_sp;
            float* p = &Sbuf[(size_t)r * cw];
            float mx = -1e30f;
            for (int c = 0; c < cw; ++c) { p[c] = p[c] * kScale + mb[c]; if (p[c] > mx) mx = p[c]; }
            double sum = 0;
            for (int c = 0; c < cw; ++c) { p[c] = expf(p[c] - mx); sum += (double)p[c]; }
            float inv = (float)(1.0 / sum);
            for (int c = 0; c < cw; ++c) P16[(size_t)r * cw + c] = f2h(p[c] * inv);
        }
        if (t) t->cpu += now_us() - t0;
        AS* s2 = asget(qrows, cw, kHD);
        if (!s2) return 1;
        t0 = now_us();
        pack_a(P16.data(), (uint16_t*)s2->A->virt_addr, qrows, cw, s2->subKA);
        pack_b_v(vb, (uint16_t*)s2->B->virt_addr, cw, kHD, s2->subNB, s2->subKB);
        if (rknn_mem_sync(s2->ctx, s2->A, RKNN_MEMORY_SYNC_TO_DEVICE)) { err = "attn syncA2"; return 1; }
        if (rknn_mem_sync(s2->ctx, s2->B, RKNN_MEMORY_SYNC_TO_DEVICE)) { err = "attn syncB2"; return 1; }
        if (t) t->pack += now_us() - t0;
        t0 = now_us();
        if (rknn_matmul_run(s2->ctx) != 0) { err = "attn PV run"; return 1; }
        if (t) t->attn += now_us() - t0;
        t0 = now_us();
        if (rknn_mem_sync(s2->ctx, s2->C, RKNN_MEMORY_SYNC_FROM_DEVICE)) { err = "attn syncC2"; return 1; }
        unpack_c((const float*)s2->C->virt_addr, ctxbuf.data(), qrows, kHD, s2->N1C, s2->subNC);
        if (t) t->unpack += now_us() - t0;
        // scatter head-major chunk into row-major ctxh (fp32 -> fp16, V-scaled)
        for (int r = 0; r < qrows; ++r) {
            uint16_t* dc = &ctxh[((size_t)(b * nrows + crow0 + r)) * kNH * kHD + (size_t)hh * kHD];
            for (int c = 0; c < kHD; ++c) dc[c] = f2h(ctxbuf[(size_t)r * kHD + c]);
        }
        return 0;
    }

    // head-major rope of one canvas row from its qkv32 row (q/k norm+rope, v
    // raw); dst head stride hstride (seq_pad*HD in prefill, same in step -
    // target rows land on their canvas rows)
    void rope_row(int l, const float* row, const float* cs, const float* sn,
                  uint16_t* qdst, uint16_t* kdst, uint16_t* vdst, long hstride) {
        const uint16_t* nqn = w[l].n_qn;
        const uint16_t* nkn = w[l].n_kn;
        for (int hh = 0; hh < kNH; ++hh) {
            const float* p = row + (size_t)hh * kHD;
            float ss = 0;
            for (int c = 0; c < kHD; ++c) ss += p[c] * p[c];
            float k2 = 1.0f / sqrtf(ss / kHD + kEps);
            uint16_t* dst = qdst ? qdst + (size_t)hh * hstride : nullptr;
            for (int j = 0; j < kHD / 2; ++j) {
                float a = p[j] * k2 * h2f(nqn[j]), z = p[j + kHD / 2] * k2 * h2f(nqn[j + kHD / 2]);
                if (dst) {
                    dst[j] = f2h(a * cs[j] - z * sn[j]);
                    dst[j + kHD / 2] = f2h(z * cs[j + kHD / 2] + a * sn[j + kHD / 2]);
                }
            }
        }
        for (int hh = 0; hh < kNKV; ++hh) {
            const float* p = row + (size_t)kNH * kHD + (size_t)hh * kHD;
            float ss = 0;
            for (int c = 0; c < kHD; ++c) ss += p[c] * p[c];
            float k2 = 1.0f / sqrtf(ss / kHD + kEps);
            uint16_t* dst = kdst + (size_t)hh * hstride;
            for (int j = 0; j < kHD / 2; ++j) {
                float a = p[j] * k2 * h2f(nkn[j]), z = p[j + kHD / 2] * k2 * h2f(nkn[j + kHD / 2]);
                dst[j] = f2h(a * cs[j] - z * sn[j]);
                dst[j + kHD / 2] = f2h(z * cs[j + kHD / 2] + a * sn[j + kHD / 2]);
            }
            const float* pv = row + (size_t)(kNH + kNKV) * kHD + (size_t)hh * kHD;
            uint16_t* dv = vdst + (size_t)hh * hstride;
            for (int c = 0; c < kHD; ++c) dv[c] = f2h(pv[c]);
        }
    }

    uint16_t* kslice(int l, int b, int kh) { return &kcache[(((size_t)l * 2 + b) * kNKV + kh) * (size_t)cap * kHD]; }
    uint16_t* vslice(int l, int b, int kh) { return &vcache[(((size_t)l * 2 + b) * kNKV + kh) * (size_t)cap * kHD]; }
};

Engine::Engine() : im_(new Impl()) {}
Engine::~Engine() {
    close();
    delete im_;
    im_ = nullptr;
}

void Engine::close() {
    Impl& I = *im_;
    for (int mi = 0; mi < 2; ++mi)
        for (int s = 0; s < 4; ++s) I.ms_destroy(I.ms[mi][s]);
    for (auto& kv : I.asl) {
        Impl::AS& a = kv.second;
        if (a.A) rknn_destroy_mem(a.ctx, a.A);
        if (a.B) rknn_destroy_mem(a.ctx, a.B);
        if (a.C) rknn_destroy_mem(a.ctx, a.C);
        if (a.ctx) rknn_matmul_destroy(a.ctx);
    }
    I.asl.clear();
    for (int l = 0; l < kLayers; ++l)
        for (int s = 0; s < 4; ++s)
            if (I.wt[l][s].mem) {
                rknn_destroy_mem(I.wt[l][s].sync, I.wt[l][s].mem);
                I.wt[l][s] = Impl::Wt{};
            }
    for (int s = 0; s < 4; ++s)
        if (I.pack_ctx[s]) { rknn_matmul_destroy(I.pack_ctx[s]); I.pack_ctx[s] = 0; }
    // geometry is gone with the contexts: reset it so a reopen rebuilds the
    // ms slots (ensure_geometry's early return must not skip ms_init_all)
    I.seq_pad = I.tgt_pad = I.cap = 0;
    I.klen[0] = I.klen[1] = 0;
    I.opened = false;
    I.packed = false;
}

int Engine::open(const char* npu_dev, const char* weight_cache_dir, const std::string& fingerprint) {
    (void)npu_dev;  // vendor API: no device node, contexts are per shape
    Impl& I = *im_;
    I.cache_dir = weight_cache_dir ? weight_cache_dir : "";
    I.fp = fingerprint;
    I.opened = true;
    return 0;
}

const std::string& Engine::last_error() const { return im_->err; }
void Engine::set_layer_hook(void (*fn)(int, void*), void* ud) { im_->hook = fn; im_->hook_ud = ud; }
const uint16_t* Engine::dbg_q16() const { return im_->q16.data(); }
const uint16_t* Engine::dbg_k16() const { return im_->k16.data(); }
const uint16_t* Engine::dbg_ctxh() const { return im_->ctxh.data(); }
const float* Engine::dbg_h() const { return im_->h.data(); }
float Engine::dbg_p(int b, int l) const { return im_->p_scale[b][l]; }

void Engine::set_weights(const std::vector<LayerW>& w) {
    im_->w = w;
    im_->packed = false;
}

int Engine::pack_weights() {
    Impl& I = *im_;
    if (!I.opened) { I.err = "open first"; return 1; }
    if (I.w.size() != kLayers) { I.err = "weights: need 28 layers"; return 1; }
    I.warn.clear();
    const bool cache = !I.cache_dir.empty() && !I.fp.empty();
    if (cache) {
        // a fresh install has no cache dir; mkdir it so the disk cache is
        // usable from the first run (fopen below does not create parents)
        std::error_code ec;
        std::filesystem::create_directories(I.cache_dir, ec);
        if (ec) I.warn = "cache: mkdir " + I.cache_dir + ": " + ec.message();
        // sweep temp files a process killed mid-pack left behind (only exact
        // final names are ever read back, so those are a pure disk leak; a
        // concurrent packer's live temp is indistinguishable and dies with it
        // - that run repacks on its next start, concurrent cold packs of one
        // cache dir on one NPU are not a supported board scenario anyway)
        std::error_code SEC;
        for (const auto& e : std::filesystem::directory_iterator(I.cache_dir, SEC))
            if (e.path().filename().string().find(".tmp.") != std::string::npos) {
                std::error_code REC;
                std::filesystem::remove(e.path(), REC);
            }
    }
    const int Mw = 2 * kQrows;  // io.B.size is M-independent; any M queries it
    for (int s = 0; s < 4; ++s) {
        if (I.pack_ctx[s]) continue;
        rknn_matmul_info info{};
        info.M = Mw; info.K = kSk[s]; info.N = kSn[s];
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NATIVE;
        if (rknn_matmul_create(&I.pack_ctx[s], &info, &I.pack_io[s])) { I.err = "query wt"; return 1; }
    }
    std::vector<uint32_t> wsz(4);
    for (int s = 0; s < 4; ++s) {
        wsz[s] = I.pack_io[s].B.size;
        for (int l = 0; l < kLayers; ++l)
            if (!I.wt[l][s].mem) {
                I.wt[l][s].mem = rknn_create_mem(I.pack_ctx[s], wsz[s]);
                I.wt[l][s].sync = I.pack_ctx[s];
                if (!I.wt[l][s].mem) { I.err = "alloc wt"; return 1; }
            }
    }
    bool warm = cache;
    std::string fn;
    auto path = [&](int l, int s) {
        fn = I.cache_dir + "/ovz_" + I.fp + "." + std::to_string(l) + "." + std::to_string(s) + ".bin";
    };
    // cache file = [packed body wsz][FNV-1a-64 trailer 8] published atomically;
    // the size check catches truncated files, the trailer catches full-size
    // content corruption (bit-rot, half-committed fs blocks after power loss).
    // Caches written without a trailer fail the size check and repack, so the
    // format change invalidates them without touching the fingerprint tag.
    auto fnv64 = [](const uint8_t* b, size_t n) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ULL; }
        return h;
    };
    for (int l = 0; warm && l < kLayers; ++l)
        for (int s = 0; s < 4; ++s) {
            path(l, s);
            FILE* f = fopen(fn.c_str(), "rb");
            if (!f) { warm = false; break; }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fclose(f);
            if (sz != (long)wsz[s] + 8) { warm = false; break; }
        }
    if (warm)  // warm load with per-file checksum verify; any miss -> cold repack
        for (int l = 0; warm && l < kLayers; ++l)
            for (int s = 0; s < 4 && warm; ++s) {
                path(l, s);
                uint16_t* dst = (uint16_t*)I.wt[l][s].mem->virt_addr;
                uint64_t cs = 0;
                FILE* f = fopen(fn.c_str(), "rb");
                bool bad = !f || fread(dst, 1, wsz[s], f) != wsz[s] || fread(&cs, 1, 8, f) != 8 ||
                           cs != fnv64((const uint8_t*)dst, wsz[s]);
                int src = bad ? 0 : rknn_mem_sync(I.wt[l][s].sync, I.wt[l][s].mem, RKNN_MEMORY_SYNC_TO_DEVICE);
                if (f) fclose(f);
                if (bad || src) {  // surface the reason: an all-warm pack_ms is the operator's
                                   // only signal, and a driver-side sync failure must not
                                   // masquerade as a plain cache miss
                    I.warn = "cache: warm miss at " + fn + (bad ? " (read/checksum)" : " (device sync)") +
                             ", repacking";
                    warm = false;
                }
            }
    std::vector<uint16_t> fused;
    for (int l = 0; !warm && l < kLayers; ++l) {
        const LayerW& w = I.w[l];
        for (int s = 0; s < 4; ++s) {
            const uint16_t* src = nullptr;
            // the vendor pack_b wants src [N,K] rows directly (no transpose):
            // w.k == nullptr: q is the fused [q|k|v] block (audio.cpp packed
            // store layout), w.up == nullptr: gate is fused [gate|up]
            switch (s) {
                case 0:
                    if (w.k) {
                        fused.assign((size_t)kNqkv * kHid, 0);
                        memcpy(fused.data(), w.q, (size_t)kNH * kHD * kHid * 2);
                        memcpy(fused.data() + (size_t)kNH * kHD * kHid, w.k, (size_t)kNKV * kHD * kHid * 2);
                        memcpy(fused.data() + (size_t)(kNH + kNKV) * kHD * kHid, w.v, (size_t)kNKV * kHD * kHid * 2);
                        src = fused.data();
                    } else src = w.q;
                    break;
                case 1: src = w.o; break;
                case 2:
                    if (w.up) {
                        fused.assign((size_t)kNgu * kHid, 0);
                        memcpy(fused.data(), w.gate, (size_t)kInter * kHid * 2);
                        memcpy(fused.data() + (size_t)kInter * kHid, w.up, (size_t)kInter * kHid * 2);
                        src = fused.data();
                    } else src = w.gate;
                    break;
                case 3: src = w.down; break;
            }
            memset(I.wt[l][s].mem->virt_addr, 0, (size_t)wsz[s]);  // grid N1*K1*subN*subK can
            // trail B.size (tile rounding in other librknnrt builds): zero it so the file and
            // its trailer never carry uninitialized device-pool bytes
            pack_b(src, (uint16_t*)I.wt[l][s].mem->virt_addr, kSk[s], kSn[s],
                   I.pack_io[s].B.dims[2], I.pack_io[s].B.dims[3]);
            if (rknn_mem_sync(I.wt[l][s].sync, I.wt[l][s].mem, RKNN_MEMORY_SYNC_TO_DEVICE)) {
                I.err = "sync wt l" + std::to_string(l) + " s" + std::to_string(s);
                return 1;
            }
            if (cache) {
                path(l, s);
                const uint8_t* buf = (const uint8_t*)I.wt[l][s].mem->virt_addr;
                const uint64_t cs = fnv64(buf, wsz[s]);
                // atomic publish: write a unique temp (mkstemp: unique even
                // across pid namespaces sharing a bind-mounted cache dir),
                // fsync, rename over the final name; a half-written/crashed
                // pack never masquerades as a complete cache file. On any
                // failure drop the temp and warn - the engine still runs (and
                // repacks next run).
                std::string tmplt = fn + ".tmp.XXXXXX";
                std::vector<char> tpl(tmplt.begin(), tmplt.end());
                tpl.push_back('\0');
                bool ok = false;
                FILE* f = fdopen(mkstemp(tpl.data()), "wb");
                if (f) {
                    ok = fwrite(buf, 1, (size_t)wsz[s], f) == (size_t)wsz[s] &&
                         fwrite(&cs, 1, 8, f) == 8 && fflush(f) == 0 && fsync(fileno(f)) == 0;
                    fclose(f);
                }
                if (!ok || rename(tpl.data(), fn.c_str()) != 0) {
                    remove(tpl.data());
                    if (I.warn.empty()) I.warn = "cache: write failed for " + fn;
                }
            }
        }
    }
    I.packed = true;
    return 0;
}

const std::string& Engine::last_warning() const { return im_->warn; }

int Engine::ensure_geometry(int seq_pad, int tgt_pad) {
    Impl& I = *im_;
    if (!I.packed) { I.err = "pack_weights first"; return 1; }
    if (seq_pad % kQrows || tgt_pad % kQrows || tgt_pad > seq_pad) { I.err = "geometry: alignment"; return 1; }
    if (seq_pad <= I.seq_pad && tgt_pad <= I.tgt_pad) return 0;
    I.seq_pad = std::max(I.seq_pad, seq_pad);
    I.tgt_pad = std::max(I.tgt_pad, tgt_pad);
    if (I.ms_init_all()) return 1;
    int M = 2 * I.seq_pad;
    I.x16.assign((size_t)M * kHid, 0);
    I.q16.assign((size_t)2 * kNH * I.seq_pad * kHD, 0);
    I.k16.assign((size_t)2 * kNKV * I.seq_pad * kHD, 0);
    I.v16.assign((size_t)2 * kNKV * I.seq_pad * kHD, 0);
    I.ctxh.assign((size_t)M * kNH * kHD, 0);
    I.mid16.assign((size_t)M * kInter, 0);
    I.qkv32.assign((size_t)M * kNqkv, 0);
    I.o32.assign((size_t)M * kHid, 0);
    I.gu32.assign((size_t)M * kNgu, 0);
    I.d32.assign((size_t)M * kHid, 0);
    I.h.assign((size_t)M * kHid, 0);
    I.ht.assign((size_t)2 * I.tgt_pad * kHid, 0);
    I.Sbuf.assign((size_t)kQrows * I.seq_pad, 0);
    I.P16.assign((size_t)kQrows * I.seq_pad, 0);
    I.ctxbuf.assign((size_t)kQrows * kHD, 0);
    I.cap = I.seq_pad;  // host caches keep the max geometry stride
    I.kcache.assign((size_t)kLayers * 2 * kNKV * I.cap * kHD, 0);
    I.vcache.assign((size_t)kLayers * 2 * kNKV * I.cap * kHD, 0);
    I.klen[0] = I.klen[1] = 0;
    // seed 1.0f, not 0: the growth loop divides by p (0 -> inf/NaN, and
    // while(vmax/0 > 31) never terminates)
    std::fill(&I.p_scale[0][0], &I.p_scale[0][0] + 2 * kLayers, 1.0f);
    return 0;
}

int Engine::prefill(const float* h_canvas, const float* cos2, const float* sin2,
                    const float* mask_bias, int seq, int seq_pad, int tgt,
                    float* h_out, Timings* t) {
    Impl& I = *im_;
    (void)tgt;
    if (!I.packed || seq_pad > I.seq_pad) { I.err = "call ensure_geometry"; return 1; }
    const int sp = seq_pad, m1 = 2 * sp, prefix = seq - tgt;
    I.build_mask(mask_bias, seq, sp);
    memcpy(I.h.data(), h_canvas, (size_t)m1 * kHid * 4);
    for (int l = 0; l < kLayers; ++l) {
        rmsnorm_neon(I.h.data(), I.w[l].n_in, I.x16.data(), m1, kHid);
        if (I.gemm(0, l, 0, I.x16.data(), I.qkv32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < sp; ++s) {
                int cix = std::min(s, seq - 1);  // pad rows clamp to last real position
                I.rope_row(l, &I.qkv32[(size_t)(b * sp + s) * kNqkv], &cos2[(size_t)(b * seq + cix) * kHD],
                           &sin2[(size_t)(b * seq + cix) * kHD],
                           &I.q16[((size_t)b * kNH) * (long)sp * kHD + (size_t)s * kHD],
                           &I.k16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)s * kHD],
                           &I.v16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)s * kHD], (long)sp * kHD);
            }
        // fill full-canvas KV (V prescaled down by pow2 p)
        double kv0 = now_us();
        for (int b = 0; b < 2; ++b) {
            const uint16_t* vb = &I.v16[(size_t)b * kNKV * sp * kHD];
            float vmax = 0;
            // abs scan: keep the prescale gate identical across layers (the
            // LUT-exp gate there is max|V|; a signed max skews neg layers)
            for (long i = 0; i < (long)kNKV * sp * kHD; ++i) vmax = std::max(vmax, std::fabs(h2f(vb[i])));
            float p = 1.0f;
            while (vmax / p > 31.0f) p *= 2.0f;
            I.p_scale[b][l] = p;
            for (int kh = 0; kh < kNKV; ++kh) {
                const uint16_t* sk = &I.k16[((size_t)b * kNKV + kh) * (long)sp * kHD];
                const uint16_t* sv = &I.v16[((size_t)b * kNKV + kh) * (long)sp * kHD];
                memcpy(I.kslice(l, b, kh), sk, (size_t)sp * kHD * 2);
                uint16_t* dv = I.vslice(l, b, kh);
                for (long i = 0; i < (long)sp * kHD; ++i) dv[i] = f2h(h2f(sv[i]) / p);
            }
            I.klen[b] = sp;
        }
        if (t) t->kv += now_us() - kv0;
        // full-canvas tile attention; pad rows self-diagonal -> never a full-masked softmax
        for (int b = 0; b < 2; ++b) {
            const float* mrow = &I.maskfm[((size_t)b * sp) * sp];
            for (int hh = 0; hh < kNH; ++hh) {
                const uint16_t* qb = &I.q16[((size_t)b * kNH + hh) * (long)sp * kHD];
                const uint16_t* kb = I.kslice(l, b, hh / 2);
                const uint16_t* vb = I.vslice(l, b, hh / 2);
                for (int ch = 0; ch < sp / kQrows; ++ch)
                    if (I.attn_tile(b, hh, qb + (size_t)ch * kQrows * kHD, kQrows, kb, vb, sp,
                                    mrow, ch * kQrows, (long)ch * kQrows, sp, t)) return 1;
            }
        }
        // caches keep only the cond prompt prefix between steps; the uncond row
        // has no prompt (its keys are its own target block, rebuilt per step).
        // cap-stride caches: truncate is logical, rows above klen get
        // overwritten by the next step's append (no repack).
        I.klen[0] = prefix;
        I.klen[1] = 0;
        if (I.gemm(0, l, 1, I.ctxh.data(), I.o32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b) {
            float p = I.p_scale[b][l];  // ctx was computed from V/p -> residual scales by p
            for (int s = 0; s < sp; ++s) {
                float* dst = &I.h[(size_t)(b * sp + s) * kHid];
                const float* src = &I.o32[(size_t)(b * sp + s) * kHid];
                for (int c = 0; c < kHid; c += 4) vst1q_f32(dst + c, vmlaq_f32(vld1q_f32(dst + c), vld1q_f32(src + c), vdupq_n_f32(p)));
            }
        }
        rmsnorm_neon(I.h.data(), I.w[l].n_post, I.x16.data(), m1, kHid);
        if (I.gemm(0, l, 2, I.x16.data(), I.gu32.data(), t)) return 1;
        for (long r = 0; r < m1; ++r) silu_neon(&I.gu32[(size_t)r * kNgu], &I.gu32[(size_t)r * kNgu + kInter], &I.mid16[(size_t)r * kInter], kInter);
        if (I.gemm(0, l, 3, I.mid16.data(), I.d32.data(), t)) return 1;
        for (long r = 0; r < m1; ++r)
            for (int c = 0; c < kHid; c += 4) vst1q_f32(&I.h[(size_t)r * kHid] + c, vaddq_f32(vld1q_f32(&I.h[(size_t)r * kHid] + c), vld1q_f32(&I.d32[(size_t)r * kHid] + c)));
        if (I.hook) I.hook(l, I.hook_ud);
    }
    memcpy(h_out, I.h.data(), (size_t)m1 * kHid * 4);
    return 0;
}

int Engine::step(const float* h_tgt, const float* cos2, const float* sin2,
                 const float* mask_bias, int seq, int tgt, int tgt_pad,
                 float* h_out, Timings* t) {
    Impl& I = *im_;
    if (!I.packed || tgt_pad > I.tgt_pad) { I.err = "call ensure_geometry"; return 1; }
    const int sp = I.seq_pad, tp = tgt_pad, m2 = 2 * tp, prefix = seq - tgt;
    if (prefix < 0 || prefix + tp > sp) { I.err = "step: target block overruns canvas"; return 1; }
    float* hc = I.h.data();  // canvas [2,sp,kHid] seeded by prefill(); we own target rows
    I.build_mask(mask_bias, seq, sp);
    // b1's block-local pad rows [tgt,tp) are all-hidden in mask_bias (canvas space);
    // give them the pad self-diagonal, the same convention build_mask applies to
    // canvas pad rows, so they never run a fully-masked softmax
    for (int r = tgt; r < tp; ++r) {
        I.maskb[(size_t)(sp + r) * sp + r] = 1;
        I.maskfm[(size_t)(sp + r) * sp + r] = 0.0f;
    }
    for (int b = 0; b < 2; ++b)
        for (int s = 0; s < tp; ++s)  // rows [tgt,tp) are pad (zero) -> canvas pad rows re-zeroed
            memcpy(&hc[(size_t)(b * sp + seq - tgt + s) * kHid], &h_tgt[(size_t)(b * tp + s) * kHid], kHid * 4);
    // dLLM canvas masks are self-only for every step query row (the target block is
    // unmasked one token per step, so each row attends to exactly itself). Then the
    // step attention is an identity on V: ctx[row] = V_scaled[row]. Detect it
    // once per call; any non-self-only mask (causal, block) keeps the tile path.
    int selfonly = 1;
    for (int b = 0; b < 2 && selfonly; ++b) {
        const int rb0 = b == 0 ? prefix : 0;
        for (int s = 0; s < tp && selfonly; ++s) {
            const uint8_t* mrow = &I.maskb[((size_t)b * sp + rb0 + s) * sp];
            const uint8_t* v0 = (const uint8_t*)memchr(mrow, 1, (size_t)sp);
            if (!v0 || v0 != mrow + rb0 + s || (v0 + 1 < mrow + sp && memchr(v0 + 1, 1, (size_t)(sp - (rb0 + s) - 1)))) selfonly = 0;
        }
    }
    for (int l = 0; l < kLayers; ++l) {
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s)
                memcpy(&I.ht[(size_t)(b * tp + s) * kHid], &hc[(size_t)(b * sp + seq - tgt + s) * kHid], kHid * 4);
        for (int b = 0; b < 2; ++b)
            rmsnorm_neon(&I.ht[(size_t)b * tp * kHid], I.w[l].n_in, &I.x16[(size_t)b * tp * kHid], tp, kHid);
        if (I.gemm(1, l, 0, I.x16.data(), I.qkv32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s) {
                // rope'd target rows land on their canvas rows in canvas head-stride sp,
                // exactly where the attention reads them: cond at [seq-tgt, seq-tgt+tp),
                // uncond block-local at [0, tp)
                int row = (b == 0 ? prefix : 0) + s;
                int cix = std::min(row, seq - 1);
                I.rope_row(l, &I.qkv32[(size_t)(b * tp + s) * kNqkv], &cos2[(size_t)(b * seq + cix) * kHD],
                           &sin2[(size_t)(b * seq + cix) * kHD],
                           &I.q16[((size_t)b * kNH) * (long)sp * kHD + (size_t)row * kHD],
                           &I.k16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)row * kHD],
                           &I.v16[((size_t)b * kNKV) * (long)sp * kHD + (size_t)row * kHD], (long)sp * kHD);
            }
        double kv0 = now_us();
        for (int b = 0; b < 2; ++b) {
            const int pb = b == 0 ? prefix : 0;  // uncond cache holds only the appended block
            float vmax = 0;  // abs, see prefill
            for (int kh = 0; kh < kNKV; ++kh) {
                const uint16_t* vc = I.vslice(l, b, kh);
                for (long i = 0; i < (long)pb * kHD; ++i) vmax = std::max(vmax, std::fabs(h2f(vc[i])));
                const uint16_t* sv = &I.v16[((size_t)b * kNKV + kh) * (long)sp * kHD + (size_t)pb * kHD];
                for (long i = 0; i < (long)tp * kHD; ++i) vmax = std::max(vmax, std::fabs(h2f(sv[i])));
            }
            float p = I.p_scale[b][l];
            while (vmax / p > 31.0f) p *= 2.0f;
            if (p != I.p_scale[b][l]) {  // prescale grew: rescale stored V
                float f = I.p_scale[b][l] / p;
                for (int kh = 0; kh < kNKV; ++kh) {
                    uint16_t* vc = I.vslice(l, b, kh);
                    for (long i = 0; i < (long)pb * kHD; ++i) vc[i] = f2h(h2f(vc[i]) * f);
                }
                I.p_scale[b][l] = p;
            }
            // append target KV at rows [pb, pb+tp) (klen[b] == pb after the last truncate)
            for (int kh = 0; kh < kNKV; ++kh) {
                const uint16_t* sk = &I.k16[((size_t)b * kNKV + kh) * (long)sp * kHD + (size_t)pb * kHD];
                const uint16_t* sv = &I.v16[((size_t)b * kNKV + kh) * (long)sp * kHD + (size_t)pb * kHD];
                uint16_t* dk = I.kslice(l, b, kh) + (size_t)pb * kHD;
                uint16_t* dv = I.vslice(l, b, kh) + (size_t)pb * kHD;
                for (int s = 0; s < tp; ++s) {
                    memcpy(dk + (size_t)s * kHD, sk + (size_t)s * kHD, kHD * 2);
                    for (int c = 0; c < kHD; ++c) dv[(size_t)s * kHD + c] = f2h(h2f(sv[(size_t)s * kHD + c]) / p);
                }
            }
            I.klen[b] = pb + tp;
        }
        if (t) t->kv += now_us() - kv0;
        // cond step rows read their canvas mask rows; uncond rows are block-local
        // (maskb b1 rows [0,tgt) = visible [0,tgt) = the appended target block)
        if (selfonly) {
            double a0 = now_us();
            for (int b = 0; b < 2; ++b) {
                const float inv = 1.0f / I.p_scale[b][l];
                const int rb0 = b == 0 ? prefix : 0;
                for (int hh = 0; hh < kNH; ++hh)
                    for (int s = 0; s < tp; ++s) {
                        const uint16_t* sv = &I.v16[((size_t)b * kNKV + hh / 2) * (long)sp * kHD + (size_t)(rb0 + s) * kHD];
                        uint16_t* dc = &I.ctxh[(((size_t)b * tp + s) * kNH + hh) * kHD];
                        for (int c = 0; c < kHD; ++c) dc[c] = f2h(h2f(sv[c]) * inv);
                    }
            }
            if (t) t->attn += now_us() - a0;
        } else
            // tile path over the FULL cache width (klen[b] = pb+tp): the mask
            // selects the visible set, the window is never truncated
            for (int b = 0; b < 2; ++b) {
                const int rb0 = b == 0 ? prefix : 0;
                const float* mrow = &I.maskfm[((size_t)b * sp + rb0) * sp];
                for (int hh = 0; hh < kNH; ++hh) {
                    const uint16_t* qb = &I.q16[((size_t)b * kNH + hh) * (long)sp * kHD + (size_t)rb0 * kHD];
                    const uint16_t* kb = I.kslice(l, b, hh / 2);
                    const uint16_t* vb = I.vslice(l, b, hh / 2);
                    for (int ch = 0; ch < tp / kQrows; ++ch)
                        if (I.attn_tile(b, hh, qb + (size_t)ch * kQrows * kHD, kQrows, kb, vb, I.klen[b],
                                        mrow, ch * kQrows, (long)ch * kQrows, tp, t)) return 1;
                }
            }
        if (I.gemm(1, l, 1, I.ctxh.data(), I.o32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s) {
                float* dst = &hc[(size_t)(b * sp + seq - tgt + s) * kHid];
                const float* src = &I.o32[(size_t)(b * tp + s) * kHid];
                for (int c = 0; c < kHid; c += 4) vst1q_f32(dst + c, vmlaq_f32(vld1q_f32(dst + c), vld1q_f32(src + c), vdupq_n_f32(I.p_scale[b][l])));
            }
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s)
                memcpy(&I.ht[(size_t)(b * tp + s) * kHid], &hc[(size_t)(b * sp + seq - tgt + s) * kHid], kHid * 4);
        for (int b = 0; b < 2; ++b)
            rmsnorm_neon(&I.ht[(size_t)b * tp * kHid], I.w[l].n_post, &I.x16[(size_t)b * tp * kHid], tp, kHid);
        if (I.gemm(1, l, 2, I.x16.data(), I.gu32.data(), t)) return 1;
        for (long r = 0; r < m2; ++r) silu_neon(&I.gu32[(size_t)r * kNgu], &I.gu32[(size_t)r * kNgu + kInter], &I.mid16[(size_t)r * kInter], kInter);
        if (I.gemm(1, l, 3, I.mid16.data(), I.d32.data(), t)) return 1;
        for (int b = 0; b < 2; ++b)
            for (int s = 0; s < tp; ++s) {
                float* dst = &hc[(size_t)(b * sp + seq - tgt + s) * kHid];
                const float* src = &I.d32[(size_t)(b * tp + s) * kHid];
                for (int c = 0; c < kHid; c += 4) vst1q_f32(dst + c, vaddq_f32(vld1q_f32(dst + c), vld1q_f32(src + c)));
            }
        // truncate back to the prefix; appended rows are overwritten by the next step
        I.klen[0] = prefix;
        I.klen[1] = 0;
        if (I.hook) I.hook(l, I.hook_ud);
    }
    for (int b = 0; b < 2; ++b)
        for (int s = 0; s < tp; ++s)
            memcpy(&h_out[(size_t)(b * tp + s) * kHid], &hc[(size_t)(b * sp + seq - tgt + s) * kHid], kHid * 4);
    return 0;
}

int Engine::caches_truncate(int len) {
    Impl& I = *im_;
    for (int b = 0; b < 2; ++b) I.klen[b] = std::max(0, std::min(len, I.cap));
    return 0;
}

}  // namespace ovz

#else  // !__aarch64__: the engine is ARMv8.2-A fp16 NEON (arm_neon.h, __fp16)
       // and the rknn_matmul contexts need a real NPU. ENGINE_ENABLE_RKNN is
       // arch-neutral, so non-aarch64 builds compile this stub; open() fails
       // and the generator surfaces last_error().
namespace ovz {

Engine::Engine() : im_(nullptr) {}
Engine::~Engine() {}
int Engine::open(const char*, const char*, const std::string&) { return -1; }
void Engine::close() {}
void Engine::set_weights(const std::vector<LayerW>&) {}
int Engine::pack_weights() { return -1; }
int Engine::ensure_geometry(int, int) { return -1; }
int Engine::prefill(const float*, const float*, const float*, const float*, int, int, int, float*, Timings*) { return -1; }
int Engine::step(const float*, const float*, const float*, const float*, int, int, int, float*, Timings*) { return -1; }
int Engine::caches_truncate(int) { return 0; }
const std::string& Engine::last_error() const {
    static const std::string e = "omnivoice rknn_zeroop engine requires aarch64";
    return e;
}
const std::string& Engine::last_warning() const {
    static const std::string w;
    return w;
}
void Engine::set_layer_hook(void (*)(int, void*), void*) {}
const uint16_t* Engine::dbg_q16() const { return nullptr; }
const uint16_t* Engine::dbg_k16() const { return nullptr; }
const uint16_t* Engine::dbg_ctxh() const { return nullptr; }
const float* Engine::dbg_h() const { return nullptr; }
float Engine::dbg_p(int, int) const { return 0.0f; }

}  // namespace ovz

#endif  // __aarch64__

#endif  // ENGINE_HAVE_RKNN
