// OmniVoice generator on the vendor rknn_matmul_api zero-op matmul calls
// (librknnrt only, AGPL-clean). See rknn_zeroop_generator.h for the contract.
//
// One request: first run() = full-canvas prefill (fills the per-layer KV caches
// and returns step 0's target hidden rows), later run()s = target-rows-only
// steps (~14.5 s vs the compiled ladder's 23 s at seq1536, board-measured). The
// packed CFG pair is one canvas of two rows: conditional row = style+text+
// reference+target, unconditional row = the target block alone.
#include "engine/models/omnivoice/rknn_zeroop_generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "rknn_zeroop/ovz_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_HAVE_RKNN)

namespace engine::models::omnivoice {

namespace {

constexpr const char * kLabel = "OmniVoice generator rknn-zeroop:";
constexpr float kMaskValue = -10000.0F;

using Clock = std::chrono::steady_clock;

// fp16 bit conversion: the ARMv8.2-A scalar half type on the board. On other
// targets the ovz engine is a stub whose open() fails, so the generator throws
// before any conversion runs and these bodies are never reached.
inline uint16_t f2h(float x) {
#if defined(__aarch64__)
    __fp16 v = static_cast<__fp16>(x);
    uint16_t r;
    std::memcpy(&r, &v, 2);
    return r;
#else
    (void)x;
    return 0;
#endif
}
inline float h2f(uint16_t x) {
#if defined(__aarch64__)
    __fp16 v;
    std::memcpy(&v, &x, 2);
    return static_cast<float>(v);
#else
    (void)x;
    return 0.0F;
#endif
}

int64_t round128(int64_t n) {
    return (n + ovz::kQrows - 1) / ovz::kQrows * ovz::kQrows;
}

// FNV-1a-64 over the weight views, sampling 1 KiB every 64 KiB (whole-store
// hashing of the ~1.1 GiB set would cost a second per load): the cache
// fingerprint must identify the weight CONTENT, so a checkpoint replaced in
// the same model directory can never silently reuse the packed cache of the
// previous one.
void fnv_mix(uint64_t & h, const void * data, int64_t elems, int elem_size) {
    h = (h ^ static_cast<uint64_t>(elems)) * 0x100000001b3ULL;
    const uint8_t * b = static_cast<const uint8_t *>(data);
    const uint64_t bytes = static_cast<uint64_t>(elems) * static_cast<uint64_t>(elem_size);
    for (uint64_t off = 0; off < bytes; off += 65536) {
        const uint64_t n = std::min<uint64_t>(1024, bytes - off);
        for (uint64_t i = 0; i < n; ++i) {
            h ^= b[off + i];
            h *= 0x100000001b3ULL;
        }
    }
}

std::string weight_set_hash(const OmniVoiceZeroopWeightViews & w, const OmniVoiceZeroopModelDims & d) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const int64_t H = d.hidden, HD = d.head_dim;
    const int64_t qkv_rows = (d.heads + 2 * d.kv_heads) * HD;
    const int64_t gu_rows = 2 * d.intermediate;
    for (const auto & l : w.layers) {
        const int esz = l.qkv_is_f16 ? 2 : 4;
        if (l.packed) {
            fnv_mix(h, l.qkv, qkv_rows * H, esz);
            fnv_mix(h, l.gate_up, gu_rows * H, l.gu_is_f16 ? 2 : 4);
        } else {
            fnv_mix(h, l.q, d.heads * HD * H, esz);
            fnv_mix(h, l.k, d.kv_heads * HD * H, esz);
            fnv_mix(h, l.v, d.kv_heads * HD * H, esz);
            fnv_mix(h, l.gate, d.intermediate * H, l.gu_is_f16 ? 2 : 4);
            fnv_mix(h, l.up, d.intermediate * H, l.gu_is_f16 ? 2 : 4);
        }
        fnv_mix(h, l.o, H * d.heads * HD, l.o_is_f16 ? 2 : 4);
        fnv_mix(h, l.down, H * d.intermediate, l.down_is_f16 ? 2 : 4);
        const int nesz = l.norms_is_f16 ? 2 : 4;
        fnv_mix(h, l.n_in, H, nesz);
        fnv_mix(h, l.n_post, H, nesz);
        fnv_mix(h, l.n_qn, HD, nesz);
        fnv_mix(h, l.n_kn, HD, nesz);
    }
    fnv_mix(h, w.text_embedding, w.text_embedding_rows * H, w.text_embedding_is_f16 ? 2 : 4);
    fnv_mix(h, w.audio_embedding, w.audio_embedding_rows * H, w.audio_embedding_is_f16 ? 2 : 4);
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
    return hex;
}

}  // namespace

struct OmniVoiceRknnZeroopGenerator::Impl {
    OmniVoiceZeroopModelDims dims;
    ovz::Engine engine;
    bool ready = false;
    bool prefilled = false;

    // per-request geometry and constant feeds (rebuilt when the shape changes)
    int64_t seq = 0;      // conditional row length (total_tokens)
    int64_t target = 0;   // target frames
    int64_t sp = 0;       // padded canvas rows per batch
    int64_t tp = 0;       // padded target rows per batch
    std::vector<float> cos2, sin2;  // [2*seq, head_dim] fp32, canvas order (prefill)
    std::vector<float> cosS, sinS;  // step-pass rope tables: plain positions, both batches
    std::vector<float> mask_p, mask_s;  // [2, seq, seq] additive, per pass: prefill / step
    std::vector<float> h_canvas, h_target, hidden;
    std::vector<uint16_t> arena;    // f32->f16 weight conversions live here
    std::vector<ovz::LayerW> layers;
    const uint16_t * text_emb = nullptr;   // column-major [hidden, text_vocab] fp16 bits
    const uint16_t * audio_emb = nullptr;  // column-major [hidden, codebooks*audio_vocab] fp16 bits
    int64_t text_emb_stride = 0;   // table rows (ggml column-major: a token row is strided by rows)
    int64_t audio_emb_stride = 0;

    void build_const_feeds(const OmniVoiceRknnStepInputs & in) {
        const int64_t seq = in.total_tokens, tgt = in.target_frames;
        const int64_t hd = dims.head_dim, half = hd / 2;
        cos2.assign(static_cast<size_t>(2) * seq * hd, 0.0F);
        sin2.assign(cos2.size(), 0.0F);
        cosS.assign(cos2.size(), 0.0F);
        sinS.assign(cos2.size(), 0.0F);
        for (int64_t row = 0; row < 2 * seq; ++row) {
            // rope position = canvas index (Slot::run tables). The uncond batch is
            // seen twice: prefill reads its canvas rows [seq-tgt, seq) (content
            // lives in the target block) and needs the entry at that canvas index
            // to carry the block-local position; step() reads block-local rows
            // [0, tp) directly. The two readings collide on indices
            // [seq-tgt, tgt) (target block bigger than the prompt), so step gets
            // its own plain table: entry i carries position i for both batches.
            int64_t token = row % seq;
            if (row >= seq) { token = token >= seq - tgt ? token - (seq - tgt) : (token < tgt ? token : 0); }
            const int64_t tokenS = row % seq;
            for (int64_t d = 0; d < half; ++d) {
                const double inv =
                    std::pow(dims.rope_theta, -2.0 * static_cast<double>(d) / static_cast<double>(hd));
                const double angle = std::fmod(static_cast<double>(token) * inv,
                                               2.0 * 3.14159265358979323846);
                const double angleS = std::fmod(static_cast<double>(tokenS) * inv,
                                                2.0 * 3.14159265358979323846);
                const float c = static_cast<float>(std::cos(angle));
                const float s = static_cast<float>(std::sin(angle));
                const float cS = static_cast<float>(std::cos(angleS));
                const float sS = static_cast<float>(std::sin(angleS));
                cos2[static_cast<size_t>(row * hd + d)] = c;
                cos2[static_cast<size_t>(row * hd + d + half)] = c;
                sin2[static_cast<size_t>(row * hd + d)] = s;
                sin2[static_cast<size_t>(row * hd + d + half)] = s;
                cosS[static_cast<size_t>(row * hd + d)] = cS;
                cosS[static_cast<size_t>(row * hd + d + half)] = cS;
                sinS[static_cast<size_t>(row * hd + d)] = sS;
                sinS[static_cast<size_t>(row * hd + d + half)] = sS;
            }
        }
        // additive masks, ONE PER PASS (the compiled-engine convention, NOT the
        // HF canvas mask whose target rows are self-only: an HF-mask experiment
        // on this sampler rendered non-speech at every rung, 2026-11-15):
        // prefill b0 rows attend [0, seq); prefill b1 attends with its
        // CANVAS-TARGET rows [seq-tgt, seq) over [seq-tgt, seq); step b0 reads
        // the same all-to-all rows (its cond batch lives at canvas positions);
        // step b1 attends with its BLOCK-LOCAL rows [0, tgt) over [0, tgt).
        // The two b1 conventions must not share one buffer: merged, their
        // overlap rows (target block bigger than the prompt) get the union =
        // full-row visibility, a third semantics no engine implements.
        const int64_t prefix = seq - tgt;
        mask_p.assign(static_cast<size_t>(2) * seq * seq, kMaskValue);
        mask_s.assign(static_cast<size_t>(2) * seq * seq, kMaskValue);
        for (int64_t r = 0; r < seq; ++r) {
            std::fill(mask_p.data() + static_cast<size_t>(r * seq),
                      mask_p.data() + static_cast<size_t>((r + 1) * seq), 0.0F);
            std::fill(mask_s.data() + static_cast<size_t>(r * seq),
                      mask_s.data() + static_cast<size_t>((r + 1) * seq), 0.0F);
        }
        float * p1 = mask_p.data() + static_cast<size_t>(seq * seq);
        for (int64_t r = prefix; r < seq; ++r)
            std::fill(p1 + r * seq + prefix, p1 + r * seq + seq, 0.0F);
        float * s1 = mask_s.data() + static_cast<size_t>(seq * seq);
        for (int64_t r = 0; r < tgt; ++r)
            std::fill(s1 + r * seq, s1 + r * seq + tgt, 0.0F);
    }

    static void layer_hook(int l, void * ud) {
        auto * self = static_cast<Impl *>(ud);
        const int64_t ctx_rows = self->prefilled ? self->tp : self->sp;  // ctxh is packed nrows/batch per pass
        auto mx16 = [](const uint16_t * p, size_t n) { float m = 0.0F; for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(h2f(p[i]))); return m; };
        auto mxf = [](const float * p, size_t n) { float m = 0.0F; for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(p[i])); return m; };
        // q16 lives at canvas head-stride sp in both passes (step's cond rows sit
        // at canvas rows [seq-tgt, seq-tgt+tp)), so scan the full buffer
        std::fprintf(stderr, "[ovzdbg] l%d q16=%.4g ctx=%.4g h=%.4g p=%.4g\n", l,
                     mx16(self->engine.dbg_q16(), static_cast<size_t>(2) * ovz::kNH * self->sp * ovz::kHD),
                     mx16(self->engine.dbg_ctxh(), static_cast<size_t>(2) * ctx_rows * ovz::kNH * ovz::kHD),
                     mxf(self->engine.dbg_h(), static_cast<size_t>(2) * self->sp * ovz::kHid),
                     self->engine.dbg_p(0, l));
        std::fflush(stderr);
    }

    // The embedding tables are stored row-major (HF layout): token row = hidden
    // contiguous fp16 values at row * hidden (validated against the checkpoint by
    // the AUDIOCPP_RKNN_ZEROOP_DEBUG probe).
    void embed_text(float * dst, int32_t id) const {
        const uint16_t * src = text_emb + static_cast<size_t>(id) * dims.hidden;
        for (int64_t c = 0; c < dims.hidden; ++c) dst[c] = h2f(src[c]);
    }
    // audio row = sum of the 8 codebook embedding rows; PackedInputs ids are already
    // global codes (code + cb*audio_vocab, the same values the compiled slot's baked
    // cb_offsets produce), the table has 8*audio_vocab rows
    void embed_audio(float * dst, const std::array<std::vector<int32_t>, 8> & ids, int64_t pos) const {
        for (int64_t cb = 0; cb < dims.codebooks; ++cb) {
            const uint16_t * s = audio_emb +
                static_cast<size_t>(ids[static_cast<size_t>(cb)][static_cast<size_t>(pos)]) * dims.hidden;
            for (int64_t c = 0; c < dims.hidden; ++c) {
                dst[c] = cb == 0 ? h2f(s[c]) : dst[c] + h2f(s[c]);
            }
        }
    }
};

OmniVoiceRknnZeroopGenerator::OmniVoiceRknnZeroopGenerator(
    const OmniVoiceZeroopWeightViews & weights,
    const OmniVoiceZeroopModelDims & dims,
    const std::string & weight_cache_dir,
    std::string fingerprint)
    : OmniVoiceRknnGenerator({}, dims.codebooks, dims.audio_vocab, dims.head_dim,
                             static_cast<float>(dims.rope_theta), dims.hidden),
      impl_(std::make_unique<Impl>()) {
    auto & I = *impl_;
    I.dims = dims;
    if (dims.heads != ovz::kNH || dims.kv_heads != ovz::kNKV || dims.head_dim != ovz::kHD ||
        dims.hidden != ovz::kHid || dims.intermediate != ovz::kInter ||
        static_cast<int64_t>(weights.layers.size()) != ovz::kLayers) {
        throw std::runtime_error(std::string(kLabel) + " engine is built for the 0.6B trunk shape");
    }
    // the caller's string is a schema tag; the cache key must also pin the
    // weight CONTENT (size-only cache checks cannot tell two checkpoints apart)
    fingerprint += "-" + weight_set_hash(weights, dims);
    auto to16 = [&](const void * data, bool is_f16, int64_t n) -> const uint16_t * {
        if (is_f16) {
            return static_cast<const uint16_t *>(data);
        }
        const size_t base = I.arena.size();
        I.arena.resize(base + static_cast<size_t>(n));
        const float * src = static_cast<const float *>(data);
        for (int64_t i = 0; i < n; ++i) I.arena[base + static_cast<size_t>(i)] = f2h(src[i]);
        return I.arena.data() + base;
    };
    const int64_t H = dims.hidden, HD = dims.head_dim;
    const int64_t qkv_rows = (dims.heads + 2 * dims.kv_heads) * HD;
    const int64_t qkvo_rows = dims.heads * HD;
    const int64_t gu_rows = 2 * dims.intermediate;
    I.layers.resize(weights.layers.size());
    // arena pointers handed out by to16() must stay valid: pre-reserve the exact
    // conversion volume so the vector never reallocates
    size_t arena_need = 0;
    for (const auto & w : weights.layers) {
        if (!w.packed) { arena_need += (qkv_rows + gu_rows) * H; }  // fused [q|k|v] / [gate|up] blocks
        if (!w.packed && !w.qkv_is_f16) { arena_need += (dims.heads + 2 * dims.kv_heads) * HD * H; }
        if (!w.packed && !w.gu_is_f16) { arena_need += gu_rows * H; }
        if (!w.o_is_f16) { arena_need += H * qkvo_rows; }
        if (!w.down_is_f16) { arena_need += H * dims.intermediate; }
        if (!w.norms_is_f16) { arena_need += 2 * H + 2 * HD; }
    }
    if (!weights.text_embedding_is_f16) { arena_need += static_cast<size_t>(weights.text_embedding_rows) * H; }
    if (!weights.audio_embedding_is_f16) { arena_need += static_cast<size_t>(weights.audio_embedding_rows) * H; }
    I.arena.reserve(arena_need);
    for (size_t l = 0; l < weights.layers.size(); ++l) {
        const auto & w = weights.layers[l];
        auto & dst = I.layers[l];
        if (w.packed) {  // store rows are [q|k|v] and [gate|up], engine fusion order
            dst.q = to16(w.qkv, w.qkv_is_f16, qkv_rows * H);
            dst.gate = to16(w.gate_up, w.gu_is_f16, gu_rows * H);
        } else {
            // fuse the trio on the host once: [q|k|v] / [gate|up] contiguous
            const uint16_t * q16 = to16(w.q, w.qkv_is_f16, dims.heads * HD * H);
            const uint16_t * k16 = to16(w.k, w.qkv_is_f16, dims.kv_heads * HD * H);
            const uint16_t * v16 = to16(w.v, w.qkv_is_f16, dims.kv_heads * HD * H);
            const size_t base = I.arena.size();
            I.arena.resize(base + static_cast<size_t>(qkv_rows) * H);
            std::memcpy(I.arena.data() + base, q16, static_cast<size_t>(dims.heads * HD) * H * 2);
            std::memcpy(I.arena.data() + base + static_cast<size_t>(dims.heads * HD) * H, k16,
                        static_cast<size_t>(dims.kv_heads * HD) * H * 2);
            std::memcpy(I.arena.data() + base + static_cast<size_t>(dims.heads + dims.kv_heads) * HD * H, v16,
                        static_cast<size_t>(dims.kv_heads * HD) * H * 2);
            dst.q = I.arena.data() + base;
            const uint16_t * g16 = to16(w.gate, w.gu_is_f16, dims.intermediate * H);
            const uint16_t * u16 = to16(w.up, w.gu_is_f16, dims.intermediate * H);
            const size_t gbase = I.arena.size();
            I.arena.resize(gbase + static_cast<size_t>(gu_rows) * H);
            std::memcpy(I.arena.data() + gbase, g16, static_cast<size_t>(dims.intermediate) * H * 2);
            std::memcpy(I.arena.data() + gbase + static_cast<size_t>(dims.intermediate) * H, u16,
                        static_cast<size_t>(dims.intermediate) * H * 2);
            dst.gate = I.arena.data() + gbase;
        }
        dst.k = nullptr;
        dst.v = nullptr;
        dst.up = nullptr;
        dst.o = to16(w.o, w.o_is_f16, H * qkvo_rows);
        dst.down = to16(w.down, w.down_is_f16, H * dims.intermediate);
        dst.n_in = to16(w.n_in, w.norms_is_f16, H);
        dst.n_post = to16(w.n_post, w.norms_is_f16, H);
        dst.n_qn = to16(w.n_qn, w.norms_is_f16, HD);
        dst.n_kn = to16(w.n_kn, w.norms_is_f16, HD);
    }
    // The caller hands embeddings through the layer-view arena path too:
    I.text_emb = to16(weights.text_embedding, weights.text_embedding_is_f16, weights.text_embedding_rows * H);
    I.audio_emb = to16(weights.audio_embedding, weights.audio_embedding_is_f16, weights.audio_embedding_rows * H);
    I.text_emb_stride = weights.text_embedding_rows;
    I.audio_emb_stride = weights.audio_embedding_rows;
    if (std::getenv("AUDIOCPP_RKNN_ZEROOP_DEBUG") != nullptr) {
        std::fprintf(stderr, "[ovzdbg] views text=%p audio=%p text_rows=%lld audio_rows=%lld text_f16=%d audio_f16=%d\n",
                     (void *) I.text_emb, (void *) I.audio_emb,
                     static_cast<long long>(weights.text_embedding_rows),
                     static_cast<long long>(weights.audio_embedding_rows),
                     weights.text_embedding_is_f16, weights.audio_embedding_is_f16);
        for (int64_t row = 0; row < 2; ++row) {
            std::fprintf(stderr, "[ovzdbg] text row %lld:", static_cast<long long>(row));
            for (int64_t c = 0; c < 4; ++c) std::fprintf(stderr, " %.4g", h2f(I.text_emb[static_cast<size_t>(row * H + c)]));
            std::fprintf(stderr, " | audio row %lld:", static_cast<long long>(row));
            for (int64_t c = 0; c < 4; ++c) std::fprintf(stderr, " %.4g", h2f(I.audio_emb[static_cast<size_t>(row * H + c)]));
            std::fprintf(stderr, " | audio col-slice %lld:", static_cast<long long>(row));
            for (int64_t c = 0; c < 4; ++c) std::fprintf(stderr, " %.4g", h2f(I.audio_emb[static_cast<size_t>(row + c * weights.audio_embedding_rows)]));
            std::fprintf(stderr, "\n");
        }
        std::fflush(stderr);
    }
    if (std::getenv("AUDIOCPP_RKNN_ZEROOP_DEBUG") != nullptr) {
        I.engine.set_layer_hook(&Impl::layer_hook, &I);
    }
    I.engine.set_weights(I.layers);
    const auto pack_start = Clock::now();
    if (I.engine.open(nullptr, weight_cache_dir.empty() ? nullptr : weight_cache_dir.c_str(), fingerprint)) {
        throw std::runtime_error(std::string(kLabel) + " " + I.engine.last_error());
    }
    if (I.engine.pack_weights()) {
        throw std::runtime_error(std::string(kLabel) + " pack: " + I.engine.last_error());
    }
    debug::timing_log_scalar("omnivoice.generator.rknn_zeroop.pack_ms", debug::elapsed_ms(pack_start));
    // non-fatal disk-cache problems (mkdir/write/flush): surface them, the
    // engine runs normally and just repacks on every run
    if (const std::string & warn = I.engine.last_warning(); !warn.empty()) {
        debug::log_message(debug::LogLevel::Warning, "omnivoice.rknn_zeroop", warn);
    }
    I.ready = true;
}

OmniVoiceRknnZeroopGenerator::~OmniVoiceRknnZeroopGenerator() {
    impl_->engine.close();
}

int64_t OmniVoiceRknnZeroopGenerator::max_tokens() const {
    return 2176;  // policy/memory cap (the ovz caches are per-geometry, cap =
                  // seq_pad: ~528 MB of KV cache at 2304 padded rows)
}

int64_t OmniVoiceRknnZeroopGenerator::engine_count() const {
    return 1;
}

bool OmniVoiceRknnZeroopGenerator::front_shard() const {
    return true;
}

int64_t OmniVoiceRknnZeroopGenerator::bucket_for(int64_t total_tokens) const {
    const int64_t bucket = round128(total_tokens) + ovz::kQrows;  // slack covers the target pad block
    return bucket <= max_tokens() + ovz::kQrows ? bucket : 0;
}

void OmniVoiceRknnZeroopGenerator::reset() {
    impl_->prefilled = false;
}

void OmniVoiceRknnZeroopGenerator::run(
    int64_t bucket,
    const OmniVoiceRknnStepInputs & inputs,
    float * out,
    int64_t row_capacity) {
    auto & I = *impl_;
    if (!I.ready) {
        throw std::runtime_error(std::string(kLabel) + " not initialized");
    }
    const auto start = Clock::now();
    const int64_t total = inputs.total_tokens;
    const int64_t target = inputs.target_frames;
    const int64_t sp = bucket;
    const int64_t tp = round128(target);
    // the engine needs the padded canvas to fit prompt + padded target block
    if (sp < total - target + tp) {
        throw std::runtime_error(std::string(kLabel) + " bucket smaller than the request");
    }
    if (I.seq != total || I.target != target) {
        I.seq = total;
        I.target = target;
        I.sp = sp;
        I.tp = tp;
        I.build_const_feeds(inputs);
        I.prefilled = false;
    }
    if (I.engine.ensure_geometry(static_cast<int>(sp), static_cast<int>(tp))) {
        throw std::runtime_error(std::string(kLabel) + " geometry: " + I.engine.last_error());
    }
    const int64_t H = I.dims.hidden;
    if (!I.prefilled) {
        // canvas h0 [2, sp, H]: cond rows = text/audio embeddings, uncond rows =
        // the 8-row audio sum; every pad row stays zero
        I.h_canvas.assign(static_cast<size_t>(2) * sp * H, 0.0F);
        for (int64_t s = 0; s < total; ++s) {
            float * dst = I.h_canvas.data() + static_cast<size_t>(s) * H;
            if (s < inputs.audio_start) {
                I.embed_text(dst, inputs.conditional_text_ids[static_cast<size_t>(s)]);
            } else {
                I.embed_audio(dst, inputs.conditional_ids, s);
            }
        }
        for (int64_t s = 0; s < target; ++s) {
            // uncond content goes to the target block, matching the engine's
            // prefix [0, total-target) / target split
            I.embed_audio(I.h_canvas.data() + static_cast<size_t>((sp + total - target + s) * H),
                          inputs.unconditional_ids, s);
        }
        if (std::getenv("AUDIOCPP_RKNN_ZEROOP_DEBUG") != nullptr) {
            size_t nf = 0; float mx = 0.0F;
            for (float v : I.h_canvas) { if (!std::isfinite(v)) { ++nf; } mx = std::max(mx, std::fabs(v)); }
            std::fprintf(stderr, "[ovzdbg] canvas total=%lld target=%lld sp=%lld tp=%lld astart=%lld max=%.4g nonfinite=%zu\n",
                         static_cast<long long>(total), static_cast<long long>(target),
                         static_cast<long long>(sp), static_cast<long long>(tp),
                         static_cast<long long>(inputs.audio_start), mx, nf);
        }
        // feed dumps stand on their own: a twin harness sets DUMP alone
        if (const char * dump = std::getenv("AUDIOCPP_RKNN_ZEROOP_DUMP"); dump != nullptr) {
            const char * names[7] = {"/hcanvas.bin", "/cos2.bin", "/sin2.bin", "/mask.bin",
                                     "/maskS.bin", "/cosS.bin", "/sinS.bin"};
            const std::vector<float> * bufs[7] = {&I.h_canvas, &I.cos2, &I.sin2, &I.mask_p,
                                                  &I.mask_s, &I.cosS, &I.sinS};
            for (int f = 0; f < 7; ++f) {
                const std::string path = std::string(dump) + names[f];
                FILE * fh = std::fopen(path.c_str(), "wb");
                if (fh != nullptr) {
                    std::fwrite(bufs[f]->data(), sizeof(float), bufs[f]->size(), fh);
                    std::fclose(fh);
                }
            }
        }
        I.hidden.assign(static_cast<size_t>(2) * sp * H, 0.0F);
        if (I.engine.prefill(I.h_canvas.data(), I.cos2.data(), I.sin2.data(), I.mask_p.data(),
                             static_cast<int>(total), static_cast<int>(sp), static_cast<int>(target),
                             I.hidden.data())) {
            throw std::runtime_error(std::string(kLabel) + " prefill: " + I.engine.last_error());
        }
        I.prefilled = true;
        debug::timing_log_scalar("omnivoice.generator.rknn_zeroop.prefill_ms", debug::elapsed_ms(start));
        // crop the target block: cond rows [total-target, total), uncond rows
        // [sp+total-target, sp+total) (uncond content lives in the target block)
        for (int64_t s = 0; s < target; ++s) {
            std::memcpy(out + static_cast<size_t>((total - target + s) * H),
                        &I.hidden[static_cast<size_t>((total - target + s) * H)], H * sizeof(float));
            std::memcpy(out + static_cast<size_t>((row_capacity + s) * H),
                        &I.hidden[static_cast<size_t>((sp + total - target + s) * H)], H * sizeof(float));
        }
    } else {
        I.h_target.assign(static_cast<size_t>(2) * tp * H, 0.0F);
        for (int64_t s = 0; s < target; ++s) {
            I.embed_audio(I.h_target.data() + static_cast<size_t>(s * H), inputs.conditional_ids,
                          total - target + s);
            I.embed_audio(I.h_target.data() + static_cast<size_t>((tp + s) * H), inputs.unconditional_ids, s);
        }
        I.hidden.assign(static_cast<size_t>(2) * tp * H, 0.0F);
        if (I.engine.step(I.h_target.data(), I.cosS.data(), I.sinS.data(), I.mask_s.data(),
                          static_cast<int>(total), static_cast<int>(target), static_cast<int>(tp),
                          I.hidden.data())) {
            throw std::runtime_error(std::string(kLabel) + " step: " + I.engine.last_error());
        }
        for (int64_t s = 0; s < target; ++s) {
            std::memcpy(out + static_cast<size_t>((total - target + s) * H),
                        &I.hidden[static_cast<size_t>(s * H)], H * sizeof(float));
            std::memcpy(out + static_cast<size_t>((row_capacity + s) * H),
                        &I.hidden[static_cast<size_t>((tp + s) * H)], H * sizeof(float));
        }
    }
    if (const char * d = std::getenv("AUDIOCPP_RKNN_ZEROOP_DUMP")) {  // twin-dump for A/B
        static int n = 0;
        char path[512];
        std::snprintf(path, sizeof path, "%s/ovz.%03d.bin", d, n++);
        if (FILE * fh = std::fopen(path, "wb")) {
            std::fwrite(I.hidden.data(), sizeof(float), I.hidden.size(), fh);
            std::fclose(fh);
        }
    }
    debug::timing_log_scalar("omnivoice.generator.rknn.step_ms", debug::elapsed_ms(start));
    debug::trace_log_scalar("omnivoice.generator.rknn.engine_tokens", sp);
}

}  // namespace engine::models::omnivoice

#else  // ENGINE_HAVE_RKNN

#include <stdexcept>

namespace engine::models::omnivoice {

struct OmniVoiceRknnZeroopGenerator::Impl {};

OmniVoiceRknnZeroopGenerator::OmniVoiceRknnZeroopGenerator(
    const OmniVoiceZeroopWeightViews &, const OmniVoiceZeroopModelDims &, const std::string &, std::string)
    : OmniVoiceRknnGenerator({}, 8, 1025, 128, 1000000.0F, 1024) {
    throw std::runtime_error("OmniVoice generator rknn-zeroop: built without ENGINE_ENABLE_RKNN");
}

OmniVoiceRknnZeroopGenerator::~OmniVoiceRknnZeroopGenerator() = default;
int64_t OmniVoiceRknnZeroopGenerator::max_tokens() const { return 0; }
int64_t OmniVoiceRknnZeroopGenerator::engine_count() const { return 0; }
bool OmniVoiceRknnZeroopGenerator::front_shard() const { return true; }
int64_t OmniVoiceRknnZeroopGenerator::bucket_for(int64_t) const { return 0; }
void OmniVoiceRknnZeroopGenerator::reset() {}
void OmniVoiceRknnZeroopGenerator::run(int64_t, const OmniVoiceRknnStepInputs &, float *, int64_t) {}

}  // namespace engine::models::omnivoice

#endif
