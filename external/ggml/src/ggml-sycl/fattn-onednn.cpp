#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fattn-onednn.hpp"
#include "fattn-tile.hpp"

// Shorter queries do not repay the partition compile and the F16 staging.
#define GGML_SYCL_FA_ONEDNN_MIN_Q 32

bool ggml_sycl_flash_attn_ext_onednn_supported(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!g_ggml_sycl_fa_onednn || g_ggml_sycl_disable_dnn) {
        return false;
    }
    // without a fused SDPA kernel oneDNN falls back to the same reference matmul
    if (!ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (sinks != nullptr) {
        return false;
    }
    // The mask node is left out of the SDPA graph when there is no mask. With one, it is bound as a
    // dense [q, seq] plane of F16 values, so its rows must be exactly the active KV length.
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
                            mask->ne[2] != 1 || mask->ne[3] != 1 || !ggml_is_contiguous(mask))) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int64_t d = K->ne[0];
    if (Q->ne[0] != d || V->ne[0] != d || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    if (Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        return false;
    }
    // SDPA with 64-wide heads fails on Alchemist.
    const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    const bool is_alchemist = arch == gpu_arch::intel_gpu_acm_g10 || arch == gpu_arch::intel_gpu_acm_g11 ||
                              arch == gpu_arch::intel_gpu_acm_g12;
    if (is_alchemist && d == 64) {
        return false;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Very long sequences
    // can make the fused SDPA slow enough to risk the driver watchdog on some stacks.
    if (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        return false;
    }
    return true;
#endif
}

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl;
using namespace dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
template <typename src_t>
static void cont_to_f16_sycl(const char * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t i0 = i % ne0; i /= ne0;
        const int64_t i1 = i % ne1; i /= ne1;
        const int64_t i2 = i % ne2; const int64_t i3 = i / ne2;
        const src_t * p = (const src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
static void permute_sdpa_out_sycl(const sycl::half * out, float * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    const int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t e = i % d; i /= d;
        const int64_t t = i % q; i /= q;
        const int64_t h = i % H; const int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (float) out[gid];
    });
}

struct sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    bool   ok = false;
};

// Build + compile the GQA SDPA graph: MatMul -> Divide -> [Add mask] -> SoftMax -> MatMul.
static sdpa_partition build_sdpa(const engine & eng, int H, int Hkv, int q, int seq, int d, bool has_mask,
                                 const std::array<int64_t, 5> & k_str, const std::array<int64_t, 5> & v_str) try {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    fi = dt::f32, t = dt::f16;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    const ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto key    = logical_tensor(id++, t,  kv_sz, k_st);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);          // key is [.., seq, d]
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({has_mask ? masked : scaled}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, v_st);
    // An f16 output is required for the fused micro-kernel; f32 materializes the score matrix.
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv);
    if (has_mask) {
        g.add_op(madd);
    }
    g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        return E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = true;
    return E;
}
catch (const std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    return {};
}

void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t d   = K->ne[0];   // head_dim
    const int64_t seq = K->ne[1];   // n_kv
    const int64_t Hkv = K->ne[2];   // n_head_kv
    const int64_t H   = Q->ne[2];   // n_head
    const int64_t q   = Q->ne[1];   // n_tok
    const int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    float kq_scale = 1.0f;
    memcpy(&kq_scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    // Q: always f32 -- copy to dense f16.
    ggml_sycl_pool_alloc<sycl::half> Qf_pool(ctx.pool(), (size_t) H * q * d);
    sycl::half * Qf_ptr = Qf_pool.get();
    cont_to_f16_sycl<float>((const char *) Q->data, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    // K/V: bind the f16 tensors in place when their strides are whole elements; otherwise stage a
    // dense copy, the layout the default strides describe.
    auto bindable = [](const ggml_tensor * t) {
        return t->nb[0] == sizeof(sycl::half) && t->nb[1] % sizeof(sycl::half) == 0 &&
               t->nb[2] % sizeof(sycl::half) == 0 && t->nb[3] % sizeof(sycl::half) == 0;
    };
    auto elem_strides = [](const ggml_tensor * t) {
        const int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        const int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        const int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        // dims are {mb=1, Hkv, rep=1, seq, d}; the size-1 dims at 0 and 2 never advance an address.
        return std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    sycl::half * K_ptr = nullptr;
    sycl::half * V_ptr = nullptr;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    if (bindable(K) && bindable(V)) {
        K_ptr = (sycl::half *) K->data;
        V_ptr = (sycl::half *) V->data;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } else {
        Kf_pool.emplace(ctx.pool(), (size_t) Hkv * seq * d);
        Vf_pool.emplace(ctx.pool(), (size_t) Hkv * seq * d);
        K_ptr = Kf_pool->get();
        V_ptr = Vf_pool->get();
        cont_to_f16_sycl<sycl::half>((const char *) K->data, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<sycl::half>((const char *) V->data, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    }

    // divide-by-(1/scale) reproduces ggml's score *= kq_scale. The scale is written from a kernel so
    // that no host memory has to outlive this call while the K/V staging kernels are still queued.
    const sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    ggml_sycl_pool_alloc<sycl::half> scale_pool(ctx.pool(), 1);
    sycl::half * scale_dev = scale_pool.get();
    stream->single_task([=]() { *scale_dev = scale_h; });

    // f16 contiguous SDPA out [mb,H,q,d]
    ggml_sycl_pool_alloc<sycl::half> outf_pool(ctx.pool(), (size_t) H * q * d);
    sycl::half * outf_ptr = outf_pool.get();

    // compile once per (device, shape, mask presence, KV strides), reuse across layers and calls.
    if (!ctx.sdpa_partitions) {
        ctx.sdpa_partitions = std::make_shared<std::unordered_map<std::string, sdpa_partition>>();
    }
    auto & cache = *static_cast<std::unordered_map<std::string, sdpa_partition> *>(ctx.sdpa_partitions.get());
    char keyb[256];
    snprintf(keyb, sizeof(keyb), "%d:%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ctx.device,
             mask != nullptr ? 1 : 0, (long long) H, (long long) Hkv, (long long) q, (long long) seq, (long long) d,
             (long long) k_str[0], (long long) k_str[1], (long long) k_str[3],
             (long long) v_str[0], (long long) v_str[1], (long long) v_str[3]);
    auto it = cache.find(keyb);
    if (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (int) H, (int) Hkv, (int) q, (int) seq, (int) d, mask != nullptr, k_str, v_str)).first;
    }
    sdpa_partition & E = it->second;
    if (!E.ok) {
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        return;
    }

    auto id2ptr = [&](size_t r) -> void * {
        if (r == E.id_q)     return Qf_ptr;
        if (r == E.id_k)     return K_ptr;
        if (r == E.id_v)     return V_ptr;
        if (r == E.id_scale) return scale_dev;
        if (r == E.id_mask && mask != nullptr) return (void *) mask->data;
        return nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    for (auto & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor to(E.out, eng, outf_ptr);
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf_ptr, (float *) dst->data, mb, H, q, d, stream);
    // The dnnl stream wraps this same in-order queue, so the SDPA is sequenced between the staging
    // kernels before it and the permute and pool reuse after it.
}
catch (const std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#endif // GGML_SYCL_DNNL
