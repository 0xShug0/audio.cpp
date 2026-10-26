// Earshot VAD float kernels. runtime.cpp includes this file once per instruction
// set, each time in its own namespace. This file defines V4, a 4-lane float
// vector; with EARSHOT_WIDE_VF set, the including namespace has already defined
// a wider VF and quantize_q15(VF, int16_t *), otherwise VF is V4.
//
// Each kernel applies the same IEEE single-precision operations to each value,
// in the same order, whatever the vector width, so every instruction set gives
// bit-identical scores. Vector max follows std::max(a, b): a unless a < b.

#if defined(EARSHOT_SSE2)
struct V4 {
    static constexpr int64_t kWidth = 4;
    __m128 v;
    static V4 load(const float * p) { return {_mm_loadu_ps(p)}; }
    void store(float * p) const { _mm_storeu_ps(p, v); }
    static V4 set1(float x) { return {_mm_set1_ps(x)}; }
    static V4 zero() { return {_mm_setzero_ps()}; }
    static V4 max_of(const V4 & a, const V4 & b) { return {_mm_max_ps(b.v, a.v)}; }
    static V4 neg(const V4 & a) { return {_mm_xor_ps(a.v, _mm_set1_ps(-0.0f))}; }
    static V4 reverse(const V4 & a) { return {_mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(0, 1, 2, 3))}; }
    // even = p[0], p[2], p[4], p[6]; odd = p[1], p[3], p[5], p[7].
    static void deinterleave(const float * p, V4 & even, V4 & odd) {
        const __m128 lo = _mm_loadu_ps(p);
        const __m128 hi = _mm_loadu_ps(p + 4);
        even.v = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
        odd.v = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
    }
    static void transpose(V4 & a, V4 & b, V4 & c, V4 & d) { _MM_TRANSPOSE4_PS(a.v, b.v, c.v, d.v); }
    // (v0 + v2) + (v1 + v3)
    static float hsum(const V4 & a) {
        const __m128 pairs = _mm_add_ps(a.v, _mm_movehl_ps(a.v, a.v));
        return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
    }
    // x = m * 2^e with m in [sqrt(1/2), sqrt(2)), for positive normal x.
    static void split_log2(const V4 & x, V4 & m, V4 & e) {
        const __m128i bits = _mm_sub_epi32(_mm_castps_si128(x.v), _mm_set1_epi32(kSqrtHalfBits));
        e.v = _mm_cvtepi32_ps(_mm_srai_epi32(bits, 23));
        m.v = _mm_castsi128_ps(_mm_add_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x007fffff)), _mm_set1_epi32(kSqrtHalfBits)));
    }
    friend V4 operator+(const V4 & a, const V4 & b) { return {_mm_add_ps(a.v, b.v)}; }
    friend V4 operator-(const V4 & a, const V4 & b) { return {_mm_sub_ps(a.v, b.v)}; }
    friend V4 operator*(const V4 & a, const V4 & b) { return {_mm_mul_ps(a.v, b.v)}; }
};

// Clamp to the int16 range, then round half away from zero like std::round:
// truncate, and step away from zero when the exact remainder is at least 0.5.
inline void quantize_q15(const V4 & x, int16_t * out) {
    const __m128 v = _mm_min_ps(_mm_max_ps(x.v, _mm_set1_ps(-32768.0f)), _mm_set1_ps(32767.0f));
    __m128i t = _mm_cvttps_epi32(v);
    const __m128 frac = _mm_sub_ps(v, _mm_cvtepi32_ps(t));
    t = _mm_sub_epi32(t, _mm_castps_si128(_mm_cmpge_ps(frac, _mm_set1_ps(0.5f))));
    t = _mm_add_epi32(t, _mm_castps_si128(_mm_cmple_ps(frac, _mm_set1_ps(-0.5f))));
    _mm_storel_epi64(reinterpret_cast<__m128i *>(out), _mm_packs_epi32(t, t));
}
#elif defined(EARSHOT_NEON)
struct V4 {
    static constexpr int64_t kWidth = 4;
    float32x4_t v;
    static V4 load(const float * p) { return {vld1q_f32(p)}; }
    void store(float * p) const { vst1q_f32(p, v); }
    static V4 set1(float x) { return {vdupq_n_f32(x)}; }
    static V4 zero() { return {vdupq_n_f32(0.0f)}; }
    static V4 max_of(const V4 & a, const V4 & b) { return {vbslq_f32(vcltq_f32(a.v, b.v), b.v, a.v)}; }
    static V4 neg(const V4 & a) { return {vnegq_f32(a.v)}; }
    static V4 reverse(const V4 & a) {
        const float32x4_t r = vrev64q_f32(a.v);
        return {vcombine_f32(vget_high_f32(r), vget_low_f32(r))};
    }
    static void deinterleave(const float * p, V4 & even, V4 & odd) {
        const float32x4x2_t pair = vld2q_f32(p);
        even.v = pair.val[0];
        odd.v = pair.val[1];
    }
    static void transpose(V4 & a, V4 & b, V4 & c, V4 & d) {
        const float32x4x2_t ab = vtrnq_f32(a.v, b.v);
        const float32x4x2_t cd = vtrnq_f32(c.v, d.v);
        a.v = vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0]));
        b.v = vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1]));
        c.v = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
        d.v = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
    }
    static float hsum(const V4 & a) {
        const float32x2_t pairs = vadd_f32(vget_low_f32(a.v), vget_high_f32(a.v));
        return vget_lane_f32(pairs, 0) + vget_lane_f32(pairs, 1);
    }
    static void split_log2(const V4 & x, V4 & m, V4 & e) {
        const int32x4_t bits = vsubq_s32(vreinterpretq_s32_f32(x.v), vdupq_n_s32(kSqrtHalfBits));
        e.v = vcvtq_f32_s32(vshrq_n_s32(bits, 23));
        m.v = vreinterpretq_f32_s32(vaddq_s32(vandq_s32(bits, vdupq_n_s32(0x007fffff)), vdupq_n_s32(kSqrtHalfBits)));
    }
    friend V4 operator+(const V4 & a, const V4 & b) { return {vaddq_f32(a.v, b.v)}; }
    friend V4 operator-(const V4 & a, const V4 & b) { return {vsubq_f32(a.v, b.v)}; }
    friend V4 operator*(const V4 & a, const V4 & b) { return {vmulq_f32(a.v, b.v)}; }
};

inline void quantize_q15(const V4 & x, int16_t * out) {
    const float32x4_t v = vminq_f32(vmaxq_f32(x.v, vdupq_n_f32(-32768.0f)), vdupq_n_f32(32767.0f));
    int32x4_t t = vcvtq_s32_f32(v);
    const float32x4_t frac = vsubq_f32(v, vcvtq_f32_s32(t));
    t = vsubq_s32(t, vreinterpretq_s32_u32(vcgeq_f32(frac, vdupq_n_f32(0.5f))));
    t = vaddq_s32(t, vreinterpretq_s32_u32(vcleq_f32(frac, vdupq_n_f32(-0.5f))));
    vst1_s16(out, vqmovn_s32(t));
}
#else
struct V4 {
    static constexpr int64_t kWidth = 4;
    float v[4];
    static V4 load(const float * p) { return {{p[0], p[1], p[2], p[3]}}; }
    void store(float * p) const { std::memcpy(p, v, sizeof(v)); }
    static V4 set1(float x) { return {{x, x, x, x}}; }
    static V4 zero() { return set1(0.0f); }
    template <class Op>
    static V4 map(const V4 & a, const V4 & b, Op op) { return {{op(a.v[0], b.v[0]), op(a.v[1], b.v[1]), op(a.v[2], b.v[2]), op(a.v[3], b.v[3])}}; }
    static V4 max_of(const V4 & a, const V4 & b) { return map(a, b, [](float x, float y) { return std::max(x, y); }); }
    static V4 neg(const V4 & a) { return {{-a.v[0], -a.v[1], -a.v[2], -a.v[3]}}; }
    static V4 reverse(const V4 & a) { return {{a.v[3], a.v[2], a.v[1], a.v[0]}}; }
    static void deinterleave(const float * p, V4 & even, V4 & odd) {
        even = {{p[0], p[2], p[4], p[6]}};
        odd = {{p[1], p[3], p[5], p[7]}};
    }
    static void transpose(V4 & a, V4 & b, V4 & c, V4 & d) {
        const V4 r[4] = {a, b, c, d};
        a = {{r[0].v[0], r[1].v[0], r[2].v[0], r[3].v[0]}};
        b = {{r[0].v[1], r[1].v[1], r[2].v[1], r[3].v[1]}};
        c = {{r[0].v[2], r[1].v[2], r[2].v[2], r[3].v[2]}};
        d = {{r[0].v[3], r[1].v[3], r[2].v[3], r[3].v[3]}};
    }
    static float hsum(const V4 & a) { return (a.v[0] + a.v[2]) + (a.v[1] + a.v[3]); }
    static void split_log2(const V4 & x, V4 & m, V4 & e) {
        for (int i = 0; i < 4; ++i) {
            int32_t bits;
            std::memcpy(&bits, &x.v[i], sizeof(bits));
            bits -= kSqrtHalfBits;
            e.v[i] = static_cast<float>(bits >> 23);
            const int32_t mant = (bits & 0x007fffff) + kSqrtHalfBits;
            std::memcpy(&m.v[i], &mant, sizeof(mant));
        }
    }
    friend V4 operator+(const V4 & a, const V4 & b) { return map(a, b, [](float x, float y) { return x + y; }); }
    friend V4 operator-(const V4 & a, const V4 & b) { return map(a, b, [](float x, float y) { return x - y; }); }
    friend V4 operator*(const V4 & a, const V4 & b) { return map(a, b, [](float x, float y) { return x * y; }); }
};

inline void quantize_q15(const V4 & x, int16_t * out) {
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<int16_t>(std::clamp(std::round(x.v[i]), -32768.0f, 32767.0f));
    }
}
#endif

#if !defined(EARSHOT_WIDE_VF)
using VF = V4;
#endif

// v = x * w for split complex vectors.
template <class V>
inline void cmul(const V & xr, const V & xi, const V & wr, const V & wi, V & vr, V & vi) {
    vr = xr * wr - xi * wi;
    vi = xr * wi + xi * wr;
}

// One radix-2 DIT stage with half-size h (h >= V::kWidth).
template <class V>
inline void fft_stage(float * re, float * im, int64_t h) {
    const auto & t = fft_tables();
    const float * wr = t.twiddle_re.data() + (h - 1);
    const float * wi = t.twiddle_im.data() + (h - 1);
    for (int64_t base = 0; base < kHalfFft; base += 2 * h) {
        float * ar = re + base;
        float * ai = im + base;
        for (int64_t j = 0; j < h; j += V::kWidth) {
            V vr, vi;
            cmul(V::load(ar + h + j), V::load(ai + h + j), V::load(wr + j), V::load(wi + j), vr, vi);
            const V ur = V::load(ar + j);
            const V ui = V::load(ai + j);
            (ur + vr).store(ar + j);
            (ui + vi).store(ai + j);
            (ur - vr).store(ar + h + j);
            (ui - vi).store(ai + h + j);
        }
    }
}

// Stages h and 2h in one pass over memory. The arithmetic is exactly that of
// two fft_stage() calls; only the loads and stores in between are skipped.
template <class V>
inline void fft_stage_pair(float * re, float * im, int64_t h) {
    const auto & t = fft_tables();
    const float * w1r = t.twiddle_re.data() + (h - 1);
    const float * w1i = t.twiddle_im.data() + (h - 1);
    const float * w2r = t.twiddle_re.data() + (2 * h - 1);
    const float * w2i = t.twiddle_im.data() + (2 * h - 1);
    for (int64_t base = 0; base < kHalfFft; base += 4 * h) {
        float * pr = re + base;
        float * pi = im + base;
        for (int64_t j = 0; j < h; j += V::kWidth) {
            const V c1r = V::load(w1r + j);
            const V c1i = V::load(w1i + j);
            const V ar = V::load(pr + j);
            const V ai = V::load(pi + j);
            const V cr = V::load(pr + 2 * h + j);
            const V ci = V::load(pi + 2 * h + j);
            V vr, vi;
            cmul(V::load(pr + h + j), V::load(pi + h + j), c1r, c1i, vr, vi);
            const V a1r = ar + vr;
            const V a1i = ai + vi;
            const V b1r = ar - vr;
            const V b1i = ai - vi;
            cmul(V::load(pr + 3 * h + j), V::load(pi + 3 * h + j), c1r, c1i, vr, vi);
            const V c1r_out = cr + vr;
            const V c1i_out = ci + vi;
            const V d1r = cr - vr;
            const V d1i = ci - vi;
            cmul(c1r_out, c1i_out, V::load(w2r + j), V::load(w2i + j), vr, vi);
            (a1r + vr).store(pr + j);
            (a1i + vi).store(pi + j);
            (a1r - vr).store(pr + 2 * h + j);
            (a1i - vi).store(pi + 2 * h + j);
            cmul(d1r, d1i, V::load(w2r + h + j), V::load(w2i + h + j), vr, vi);
            (b1r + vr).store(pr + h + j);
            (b1i + vi).store(pi + h + j);
            (b1r - vr).store(pr + 3 * h + j);
            (b1i - vi).store(pi + 3 * h + j);
        }
    }
}

// Bit-reversal permutation fused with the DIT stages h = 1 and h = 2. Write the
// complex input index as n = a * 128 + b * 4 + c (a, c < 4, b < 32); z[n] goes to
// position rev2(c) * 128 + rev5(b) * 4 + rev2(a). For one b, the rows
// M[a] = z[a * 128 + b * 4 + (0..3)] hold four whole radix-4 groups, lane c of
// row rev2(l) being lane l of group c. So the radix-4 butterflies run
// elementwise across the rows, then a 4x4 transpose turns each column into its
// group. Row a = 3 is the zero padding (n >= 384). The arithmetic matches the
// scalar radix-4 pass exactly.
inline void bitrev_radix4(const float * windowed, float * re, float * im) {
    const auto & t = fft_tables();
    const V4 zero = V4::zero();
    for (int64_t b = 0; b < 32; ++b) {
        V4 r0, i0, r1, i1, r2, i2;
        V4::deinterleave(windowed + 2 * (b * 4), r0, i0);
        V4::deinterleave(windowed + 2 * (128 + b * 4), r1, i1);
        V4::deinterleave(windowed + 2 * (256 + b * 4), r2, i2);
        // Group lanes 0..3 are rows 0, 2, 1 and 3 (zero); twiddles 1 and -i.
        const V4 t0r = r0 + r2;
        const V4 t0i = i0 + i2;
        const V4 t1r = r0 - r2;
        const V4 t1i = i0 - i2;
        const V4 t2r = r1 + zero;
        const V4 t2i = i1 + zero;
        const V4 t3r = r1 - zero;
        const V4 t3i = i1 - zero;
        V4 g0r = t0r + t2r;
        V4 g0i = t0i + t2i;
        V4 g2r = t0r - t2r;
        V4 g2i = t0i - t2i;
        V4 g1r = t1r + t3i;
        V4 g1i = t1i - t3r;
        V4 g3r = t1r - t3i;
        V4 g3i = t1i + t3r;
        V4::transpose(g0r, g1r, g2r, g3r);
        V4::transpose(g0i, g1i, g2i, g3i);
        // Group c starts at rev2(c) * 128 + rev5(b) * 4; rev5(b) * 4 = bitrev9(4 b).
        const int64_t offset = t.bitrev[b * 4];
        g0r.store(re + offset);
        g0i.store(im + offset);
        g1r.store(re + 256 + offset);
        g1i.store(im + 256 + offset);
        g2r.store(re + 128 + offset);
        g2i.store(im + 128 + offset);
        g3r.store(re + 384 + offset);
        g3i.store(im + 384 + offset);
    }
}

// Scaled |X[k]|^2 / 32768^2 of the 1024-point DFT of `windowed` (768 samples,
// zero-padded), for k in [0, w.mel_bins).
inline void power_spectrum(const EarshotWeights & w, const float * windowed, float * power) {
    const auto & t = fft_tables();
    alignas(64) float re[kHalfFft];
    alignas(64) float im[kHalfFft];
#if defined(EARSHOT_STAGE48)
    bitrev_radix4_wide(windowed, re, im);
#else
    bitrev_radix4(windowed, re, im);
#endif
    static_assert(VF::kWidth <= 16, "stage pairs from h = 16 need at most 16 lanes");
#if defined(EARSHOT_STAGE48)
    fft_stage48(re, im);
#else
    fft_stage_pair<V4>(re, im, 4);
#endif
    fft_stage_pair<VF>(re, im, 16);
    fft_stage_pair<VF>(re, im, 64);
    fft_stage<VF>(re, im, 256);

    // Split Z = FFT512(even + i * odd) into the real-input spectrum X.
    const int64_t end = w.mel_bins;
    const float dc_re = re[0];
    const float dc_im = im[0];
    power[0] = ((dc_re + dc_im) * (dc_re + dc_im)) * kPowerScale;
    const VF half = VF::set1(0.5f);
    const VF neg_half = VF::set1(-0.5f);
    const VF scale = VF::set1(kPowerScale);
    int64_t k = 1;
    for (; k < end && k + VF::kWidth <= kHalfFft; k += VF::kWidth) {
        const int64_t mirror = kHalfFft - k - (VF::kWidth - 1);
        const VF zr = VF::load(re + k);
        const VF zi = VF::load(im + k);
        const VF cr = VF::reverse(VF::load(re + mirror));
        const VF ci = VF::neg(VF::reverse(VF::load(im + mirror)));
        const VF er = half * (zr + cr);
        const VF ei = half * (zi + ci);
        const VF odd_re = half * (zi - ci);
        const VF odd_im = neg_half * (zr - cr);
        const VF pr = VF::load(t.post_re.data() + k);
        const VF pi = VF::load(t.post_im.data() + k);
        const VF xr = er + (pr * odd_re - pi * odd_im);
        const VF xi = ei + (pr * odd_im + pi * odd_re);
        ((xr * xr + xi * xi) * scale).store(power + k);
    }
    for (; k < end && k < kHalfFft; ++k) {
        const float zr = re[k];
        const float zi = im[k];
        const float cr = re[kHalfFft - k];
        const float ci = -im[kHalfFft - k];
        const float er = 0.5f * (zr + cr);
        const float ei = 0.5f * (zi + ci);
        const float odd_re = 0.5f * (zi - ci);
        const float odd_im = -0.5f * (zr - cr);
        const float xr = er + (t.post_re[k] * odd_re - t.post_im[k] * odd_im);
        const float xi = ei + (t.post_re[k] * odd_im + t.post_im[k] * odd_re);
        power[k] = (xr * xr + xi * xi) * kPowerScale;
    }
    if (end > kHalfFft) {
        power[kHalfFft] = ((dc_re - dc_im) * (dc_re - dc_im)) * kPowerScale;
    }
}

// Natural log of positive normal floats: Cephes logf, relative error below
// 1e-7. Plain float operations, so it rounds the same on every instruction set
// and platform, unlike the C library's logf.
inline VF log_positive(const VF & x) {
    VF m;
    VF e;
    VF::split_log2(x, m, e);
    const VF f = m - VF::set1(1.0f);
    const VF z = f * f;
    VF p = VF::set1(7.0376836292e-2f);
    p = p * f + VF::set1(-1.1514610310e-1f);
    p = p * f + VF::set1(1.1676998740e-1f);
    p = p * f + VF::set1(-1.2420140846e-1f);
    p = p * f + VF::set1(1.4249322787e-1f);
    p = p * f + VF::set1(-1.6668057665e-1f);
    p = p * f + VF::set1(2.0000714765e-1f);
    p = p * f + VF::set1(-2.4999993993e-1f);
    p = p * f + VF::set1(3.3333331174e-1f);
    VF y = p * f * z;
    y = y + e * VF::set1(-2.12194440e-4f);
    y = y + z * VF::set1(-0.5f);
    return (f + y) + e * VF::set1(0.693359375f);
}

#if !defined(EARSHOT_MEL8)
// Dot product of a zero-padded mel band, `groups` groups of eight: eight
// interleaved partial sums, then ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7)).
// Every ISA uses this exact order.
inline float mel_sum(const float * bins, const float * coeffs, int32_t groups) {
    V4 lo = V4::zero();
    V4 hi = V4::zero();
    for (int32_t g = 0; g < groups; ++g) {
        lo = lo + V4::load(bins + 8 * g) * V4::load(coeffs + 8 * g);
        hi = hi + V4::load(bins + 8 * g + 4) * V4::load(coeffs + 8 * g + 4);
    }
    return V4::hsum(lo + hi);
}
#endif

// Mel log-energies of one Hann-windowed frame, then the RMS norm applied by
// DefaultPredictor::normalize.
inline void window_to_features(const EarshotWeights & w, const float * windowed, float * out) {
    alignas(64) float power[kBins + 16];
    power_spectrum(w, windowed, power);
    alignas(64) float energy[kMelsPadded];
    for (int64_t band = 0; band < kMels; ++band) {
        energy[band] = mel_sum(power + w.mel_start[band], w.mel_coeffs.data() + w.mel_offset[band],
                               w.mel_groups[band]) + 1e-20f;
    }
    for (int64_t band = kMels; band < kMelsPadded; ++band) {
        energy[band] = 1.0f;
    }
    for (int64_t band = 0; band < kMelsPadded; band += VF::kWidth) {
        log_positive(VF::load(energy + band)).store(energy + band);
    }
    std::memcpy(out, energy, kMels * sizeof(float));
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < kMels; ++i) {
        sum_sq += out[i] * out[i];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(kMels));
    for (int64_t i = 0; i < kMels; ++i) {
        out[i] = w.norm_weight[i] * out[i] * inv_rms;
    }
}

// Pre-emphasis and Hann window over 768 samples; src[-1] must be readable.
inline void emphasis_window(const float * src, const float * hann, float * out) {
    const VF scale = VF::set1(32768.0f);
    const VF coeff = VF::set1(kPreEmphasis);
    for (int64_t i = 0; i < kWindowSize; i += VF::kWidth) {
        const VF sample = VF::load(src + i) * scale;
        const VF prev = VF::load(src + i - 1) * scale;
        ((sample - coeff * prev) * VF::load(hann + i)).store(out + i);
    }
}

// Pointwise 16x16 conv of N columns at once: N independent accumulator chains
// keep the adders busy, and each weight row is loaded once for all N. Every
// output still sums its 16 inputs in upstream order.
constexpr int64_t kConvParts = kChannels / VF::kWidth;
constexpr int64_t kConvGroup = kConvParts >= 8 ? 1 : 8 / kConvParts;

template <int64_t N>
inline void pointwise_columns(const float * dw, const float * weight_t, float * out) {
    VF acc[N][kConvParts];
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t p = 0; p < kConvParts; ++p) {
            acc[i][p] = VF::zero();
        }
    }
    for (int64_t c = 0; c < kChannels; ++c) {
        for (int64_t p = 0; p < kConvParts; ++p) {
            const VF wv = VF::load(weight_t + c * kChannels + p * VF::kWidth);
            for (int64_t i = 0; i < N; ++i) {
                acc[i][p] = acc[i][p] + VF::set1(dw[i * kChannels + c]) * wv;
            }
        }
    }
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t p = 0; p < kConvParts; ++p) {
            acc[i][p].store(out + i * kChannels + p * VF::kWidth);
        }
    }
}

template <int64_t Cols, int64_t Done = 0>
inline void pointwise_all(const float * dw, const float * weight_t, float * out) {
    if constexpr (Done < Cols) {
        constexpr int64_t n = Cols - Done < kConvGroup ? Cols - Done : kConvGroup;
        pointwise_columns<n>(dw + Done * kChannels, weight_t, out + Done * kChannels);
        pointwise_all<Cols, Done + n>(dw, weight_t, out);
    }
}

// Depthwise 3-tap conv (stride 2, pad 1) of a [InCols, 16] map, then the 16x16
// pointwise conv; out is [OutCols, 16]. Sums run in upstream order.
template <int64_t InCols, int64_t OutCols>
inline void separable_layer(const float * input, const float * kernel_t, const float * weight_t, float * out) {
    alignas(64) float dw[OutCols * kChannels];
    for (int64_t ox = 0; ox < OutCols; ++ox) {
        for (int64_t p = 0; p < kConvParts; ++p) {
            VF sum = VF::zero();
            for (int64_t kw = 0; kw < 3; ++kw) {
                const int64_t ix = ox * 2 + kw - 1;
                if (ix >= 0 && ix < InCols) {
                    sum = sum + VF::load(input + ix * kChannels + p * VF::kWidth) *
                                    VF::load(kernel_t + kw * kChannels + p * VF::kWidth);
                }
            }
            sum.store(dw + ox * kChannels + p * VF::kWidth);
        }
    }
    pointwise_all<OutCols>(dw, weight_t, out);
}

// Conv stack over a [3, 40] feature context; writes the Q15 [5, 16] RNN input.
inline void conv_stack(const EarshotWeights & w, const float * context, int16_t * out) {
    constexpr int64_t kParts = kChannels / VF::kWidth;
    alignas(64) float row[kDepthwiseCols] = {};
    for (int64_t kh = 0; kh < 3; ++kh) {
        for (int64_t kw = 0; kw < 3; ++kw) {
            const float k = w.layer1_kernel[kh * 3 + kw];
            const float * src = context + kh * kMels + kw;
            const VF kv = VF::set1(k);
            int64_t ox = 0;
            for (; ox + VF::kWidth <= kDepthwiseCols; ox += VF::kWidth) {
                (VF::load(row + ox) + VF::load(src + ox) * kv).store(row + ox);
            }
            for (; ox < kDepthwiseCols; ++ox) {
                row[ox] += src[ox] * k;
            }
        }
    }

    // Per-channel affine, then a zero-initialised max pool (kernel 3, stride 2)
    // that doubles as ReLU. Layout [col, channel].
    alignas(64) float layer1[kPooledCols * kChannels];
    for (int64_t p = 0; p < kParts; ++p) {
        const VF scale = VF::load(w.layer1_weight.data() + p * VF::kWidth);
        const VF bias = VF::load(w.layer1_bias.data() + p * VF::kWidth);
        VF prev = VF::set1(row[0]) * scale + bias;
        for (int64_t q = 0; q < kPooledCols; ++q) {
            const VF s1 = VF::set1(row[q * 2 + 1]) * scale + bias;
            const VF s2 = VF::set1(row[q * 2 + 2]) * scale + bias;
            VF::max_of(VF::max_of(VF::max_of(VF::zero(), prev), s1), s2)
                .store(layer1 + q * kChannels + p * VF::kWidth);
            prev = s2;
        }
    }

    alignas(64) float layer2[kLayer2Cols * kChannels];
    separable_layer<kPooledCols, kLayer2Cols>(layer1, w.layer2_kernel_t.data(), w.layer2_weight_t.data(), layer2);
    for (int64_t ox = 0; ox < kLayer2Cols; ++ox) {
        for (int64_t p = 0; p < kParts; ++p) {
            float * cell = layer2 + ox * kChannels + p * VF::kWidth;
            VF::max_of(VF::load(cell) + VF::load(w.layer2_bias.data() + p * VF::kWidth), VF::zero()).store(cell);
        }
    }

    // No bias or ReLU; the output is already [col, channel].
    alignas(64) float layer3[kLayer3Cols * kChannels];
    separable_layer<kLayer2Cols, kLayer3Cols>(layer2, w.layer3_kernel_t.data(), w.layer3_weight_t.data(), layer3);
    const VF q15 = VF::set1(32768.0f);
    for (int64_t i = 0; i < kLayer3Cols * kChannels; i += VF::kWidth) {
        quantize_q15(VF::load(layer3 + i) * q15, out + i);
    }
}
