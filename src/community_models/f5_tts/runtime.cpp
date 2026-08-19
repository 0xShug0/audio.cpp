#include "engine/community_models/f5_tts/runtime.h"

#include "engine/community_models/f5_tts/dit_modules.h"
#include "engine/community_models/f5_tts/weights.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <map>
#include <thread>
#include <unordered_map>

namespace engine::models::f5_tts {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;

// Column convention throughout: tensors are [features, seq] (ggml ne0 =
// features), so ggml_norm == LayerNorm over features and ggml_mul_mat(W, X)
// matches torch X @ W^T with torch weights stored [out, in].

struct F5Linear {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct F5ConvNeXt {
    modules::DepthwiseConv1dWeights dwconv;
    core::TensorValue norm_w, norm_b;
    F5Linear pw1, pw2;
    std::vector<float> grn_gamma, grn_beta;
};

struct F5Block {
    F5Linear attn_norm;  // -> 6*dim
    F5Linear to_q, to_k, to_v, to_out;
    F5Linear ff0, ff2;
};

struct F5Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;  // [512, 2731]
    F5Linear input_proj;
    modules::Conv1dWeights cpe0, cpe2;  // grouped k31 g16
    F5Linear time0, time2;
    std::vector<float> inv_freq;
    std::vector<F5ConvNeXt> text_blocks;
    std::vector<F5Block> blocks;
    F5Linear norm_out;
    F5Linear proj_out;
};

struct LoadedModel {
    F5Weights w;
    F5Architecture arch;
    ggml_backend_t backend = nullptr;            // backend owning the weight store
    core::BackendType backend_type = core::BackendType::Cpu;
};

struct BackendOwner {
    ggml_backend_t value = nullptr;
    // Intentionally never freed: CUDA backends must be released before the
    // driver shuts down at static destruction, which we cannot order reliably.
    // Leaking a few backends at process exit is harmless.
    ~BackendOwner() = default;
};

F5Weights load_weights(
    const engine::assets::TensorSource & source,
    ggml_backend_t backend,
    core::BackendType backend_type,
    bool fp16_linears) {
    F5Weights w;
    w.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "f5_tts.weights", 2ULL * 1024ULL * 1024ULL * 1024ULL);
    const auto f32 = [&](const std::string & n) {
        return w.store->load_f32_tensor(
            source, n, source.require_metadata(n).shape);
    };
    // Linear weights consumed only by ggml_mul_mat: FP16 storage halves GEMM
    // time on tensor cores (mul_mat returns F32 regardless of weight dtype).
    // Everything else (embeddings consumed by get_rows, dwconv kernels read
    // on host, biases/norms broadcast) stays F32.
    const auto gemm_w = [&](const std::string & n) {
        if (fp16_linears) {
            return w.store->load_tensor(
                source, n, engine::assets::TensorStorageType::F16,
                source.require_metadata(n).shape);
        }
        return f32(n);
    };
    const auto lin_f32bias = [&](const std::string & n) {
        return F5Linear{gemm_w(n + ".weight"), f32(n + ".bias")};
    };

    w.text_embedding = f32("text_embed.text_embed.weight");  // get_rows consumer: keep F32
    w.input_proj = lin_f32bias("input_embed.proj");
    w.cpe0.weight = gemm_w("input_embed.conv_pos_embed.conv1d.0.weight");
    w.cpe0.bias = f32("input_embed.conv_pos_embed.conv1d.0.bias");
    w.cpe2.weight = gemm_w("input_embed.conv_pos_embed.conv1d.2.weight");
    w.cpe2.bias = f32("input_embed.conv_pos_embed.conv1d.2.bias");
    w.time0 = lin_f32bias("time_embed.time_mlp.0");
    w.time2 = lin_f32bias("time_embed.time_mlp.2");
    w.inv_freq = source.require_f32("rotary_embed.inv_freq");

    w.text_blocks.reserve(4);
    for (int i = 0; i < 4; ++i) {
        const std::string p = "text_embed.text_blocks." + std::to_string(i);
        F5ConvNeXt b;
        b.dwconv.weight = f32(p + ".dwconv.weight");   // host-side kernel read
        b.dwconv.bias = f32(p + ".dwconv.bias");
        b.norm_w = f32(p + ".norm.weight");
        b.norm_b = f32(p + ".norm.bias");
        b.pw1 = lin_f32bias(p + ".pwconv1");
        b.pw2 = lin_f32bias(p + ".pwconv2");
        b.grn_gamma = source.require_f32(p + ".grn.gamma");
        b.grn_beta = source.require_f32(p + ".grn.beta");
        w.text_blocks.push_back(std::move(b));
    }
    w.blocks.reserve(22);
    for (int i = 0; i < 22; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i);
        F5Block b;
        b.attn_norm = lin_f32bias(p + ".attn_norm.linear");
        b.to_q = lin_f32bias(p + ".attn.to_q");
        b.to_k = lin_f32bias(p + ".attn.to_k");
        b.to_v = lin_f32bias(p + ".attn.to_v");
        b.to_out = lin_f32bias(p + ".attn.to_out.0");
        b.ff0 = lin_f32bias(p + ".ff.ff.0.0");
        b.ff2 = lin_f32bias(p + ".ff.ff.2");
        w.blocks.push_back(std::move(b));
    }
    w.norm_out = lin_f32bias("norm_out.linear");
    w.proj_out = lin_f32bias("proj_out");
    w.store->upload();
    source.release_storage();
    return w;
}

// write a small view that strips that dotted prefix.
class StrippedView final : public engine::assets::TensorSource {
public:
    static constexpr std::string_view kPrefix = "ema_model.transformer.";

    explicit StrippedView(std::shared_ptr<const engine::assets::TensorSource> inner)
        : inner_(std::move(inner)) {
        for (const auto & t : inner_->tensors()) {
            if (t.name.rfind(kPrefix, 0) == 0) {
                routes_.emplace(t.name.substr(kPrefix.size()), t.name);
            }
        }
        if (routes_.empty()) {
            throw std::runtime_error("F5 checkpoint has no ema_model.transformer.* tensors");
        }
    }
    const std::filesystem::path & source_path() const noexcept override {
        return inner_->source_path();
    }
    bool has_tensor(std::string_view name) const noexcept override {
        return routes_.find(std::string(name)) != routes_.end();
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        auto m = inner_->require_metadata(require(name));
        m.name = std::string(name);
        return m;
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        std::vector<engine::assets::TensorMetadata> out;
        out.reserve(routes_.size());
        for (const auto & [n, _] : routes_) {
            out.push_back(require_metadata(n));
        }
        return out;
    }
    void release_storage() const override { inner_->release_storage(); }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        auto d = inner_->require_tensor_data(require(name));
        d.metadata.name = std::string(name);
        return d;
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return inner_->require_f32(require(name), expected_shape);
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto found = routes_.find(std::string(name));
        if (found == routes_.end()) {
            return std::nullopt;
        }
        return inner_->optional_f32(found->second, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        return inner_->require_i64_scalar(require(name));
    }

private:
    const std::string & require(std::string_view name) const {
        const auto found = routes_.find(std::string(name));
        if (found == routes_.end()) {
            throw std::runtime_error("F5 missing tensor: " + std::string(name));
        }
        return found->second;
    }
    std::shared_ptr<const engine::assets::TensorSource> inner_;
    std::unordered_map<std::string, std::string> routes_;
};

const LoadedModel & load_model_once(const std::string & path, const F5ComputeDevice & dev);

// Module-typed DiT weights cache (per path + device). Leaked at exit like
// the raw-weight cache: CUDA buffers cannot be freed after driver shutdown.
const F5DiTWeights & load_dit_weights_once(
    const std::string & path, const F5ComputeDevice & dev) {
    struct Entry { F5DiTWeights w; };
    static auto * cache = new std::unordered_map<std::string, Entry>();
    const std::string key =
        (dev.use_cuda ? "cuda" + std::to_string(dev.device) : "cpu") + ":" + path;
    const auto found = cache->find(key);
    if (found != cache->end()) {
        return found->second.w;
    }
    const auto & model = load_model_once(path, dev);
    auto source = engine::assets::open_tensor_source(path);
    auto stripped = std::make_shared<StrippedView>(source);
    Entry entry;
    entry.w = load_dit_weights(*stripped, model.backend, model.backend_type);
    return cache->emplace(key, std::move(entry)).first->second.w;
}

const LoadedModel & load_model_once(const std::string & path, const F5ComputeDevice & dev) {
    // Deliberately leaked at exit (never destroyed): CUDA weight buffers must
    // be freed before driver shutdown; static destruction order cannot
    // guarantee that. Model caches are process-lifetime anyway.
    static auto * cache = new std::unordered_map<std::string, LoadedModel>();
    // The backend (and its weight buffer) must outlive the cache entry, so it
    // is owned by a static owner freed after the cache at exit.
    static std::vector<std::unique_ptr<BackendOwner>> owners;
    const std::string key = (dev.use_cuda ? "cuda" + std::to_string(dev.device) : "cpu") + ":" + path;
    if (const auto found = cache->find(key); found != cache->end()) {
        return found->second;
    }
    auto source = engine::assets::open_tensor_source(path);
    auto stripped = std::make_shared<StrippedView>(source);
    auto owner = std::make_unique<BackendOwner>();
    const core::BackendType type = dev.use_cuda ? core::BackendType::Cuda : core::BackendType::Cpu;
    core::BackendConfig cfg{type, dev.use_cuda ? dev.device : 0, dev.use_cuda ? 1 : std::max(1, dev.threads)};
    owner->value = core::init_backend(cfg);
    if (!dev.use_cuda) {
        core::set_backend_threads(owner->value, std::max(1, dev.threads));
    }
    LoadedModel model;
    model.arch = F5Architecture{};
    model.backend = owner->value;
    model.backend_type = type;
    // FP16 linear weights on CUDA only (CPU mul_mat with F16 weights is slow
    // via the fallback path; CUDA hits tensor cores).
    model.w = load_weights(*stripped, owner->value, type, dev.use_cuda && dev.fp16_weights);
    owners.push_back(std::move(owner));
    return cache->emplace(key, std::move(model)).first->second;
}

// ---- graph helpers (column convention) --------------------------------------

ggml_tensor * lin_apply(
    ggml_context * ctx,
    const F5Linear & w,
    ggml_tensor * x) {  // x [in, T] -> [out, T]
    auto * out = ggml_mul_mat(ctx, w.weight.tensor, x);
    // bias [out] -> [out, 1] broadcast via repeat
    auto * b2 = ggml_reshape_2d(ctx, w.bias.tensor, ggml_nelements(w.bias.tensor), 1);
    auto * b_rep = ggml_repeat(ctx, b2, out);
    return ggml_add(ctx, out, b_rep);
}

ggml_tensor * affine_norm(
    ggml_context * ctx,
    ggml_tensor * x,  // [D, T]
    ggml_tensor * gamma,
    ggml_tensor * beta) {
    auto * n = ggml_norm(ctx, x, 1e-6F);
    auto * g2 = ggml_reshape_2d(ctx, gamma, ggml_nelements(gamma), 1);
    auto * b2 = ggml_reshape_2d(ctx, beta, ggml_nelements(beta), 1);
    auto * g_rep = ggml_repeat(ctx, g2, n);
    auto * b_rep = ggml_repeat(ctx, b2, n);
    return ggml_add(ctx, ggml_mul(ctx, n, g_rep), b_rep);
}

// chunk i of an [6*D, 1] embedding -> [D, 1]
ggml_tensor * chunk_col(ggml_context * ctx, ggml_tensor * emb, int64_t idx, int64_t d) {
    const int64_t stride = d * static_cast<int64_t>(sizeof(float));
    auto * v = ggml_view_2d(ctx, emb, d, 1, stride, idx * stride);
    return ggml_cont(ctx, v);
}

// x * (1 + scale) + shift, scale/shift [D, 1], x [D, T]
ggml_tensor * modulate(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * scale,
    ggml_tensor * shift,
    ggml_tensor * ones_d1) {
    auto * one_plus = ggml_add(ctx, scale, ones_d1);
    auto * s_rep = ggml_repeat(ctx, one_plus, x);
    auto * sh_rep = ggml_repeat(ctx, shift, x);
    return ggml_add(ctx, ggml_mul(ctx, x, s_rep), sh_rep);
}

// Depthwise conv1d k=7 pad=3, stride 1 (verified against numpy reference at
// cosine 1.0). x: [C, T] columns; w: store tensor with torch [C,1,7] raw
// bytes (host-readable on CPU backend); b: [C] bias tensor.
ggml_tensor * depthwise_conv7(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    const std::function<void(ggml_tensor *, const void *, size_t)> & leaf_write,
    const std::function<void(ggml_tensor *, size_t)> & leaf_zero) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    // copy kernels out of the store tensor into host memory: torch [C,1,7].
    // On CUDA the store tensor is device memory, so go through tensor_get.
    std::vector<float> w_host(static_cast<size_t>(C) * 7);
    if (w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer)) {
        std::memcpy(w_host.data(), w->data, w_host.size() * sizeof(float));
    } else {
        ggml_backend_tensor_get(w, w_host.data(), 0, w_host.size() * sizeof(float));
    }
    const auto * raw = w_host.data();
    std::vector<float> wk(static_cast<size_t>(C));
    auto * zl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 3);
    leaf_zero(zl, static_cast<size_t>(C) * 3 * sizeof(float));
    auto * zr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 3);
    leaf_zero(zr, static_cast<size_t>(C) * 3 * sizeof(float));
    auto * xpad = ggml_concat(ctx, ggml_concat(ctx, zl, x, 1), zr, 1);  // [C, T+6]
    ggml_tensor * acc = nullptr;
    for (int k = 0; k < 7; ++k) {
        for (int64_t c = 0; c < C; ++c) {
            wk[static_cast<size_t>(c)] = raw[static_cast<size_t>(c) * 7 + static_cast<size_t>(k)];
        }
        auto * wk_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 1);
        leaf_write(wk_t, wk.data(), wk.size() * sizeof(float));
        auto * shift = ggml_view_2d(ctx, xpad, C, T, xpad->nb[1], k * xpad->nb[1]);
        auto * term = ggml_mul(ctx, shift, wk_t);
        acc = acc == nullptr ? term : ggml_add(ctx, acc, term);
    }
    auto * b2 = ggml_reshape_2d(ctx, b, C, 1);
    return ggml_add(ctx, acc, ggml_repeat(ctx, b2, acc));
}

// Grouped 1D conv (stride 1, pad k/2, dilation 1, bias) via per-group im2col.
// input: rows layout ne [T, C_in, 1]; weight: torch logical [C_out, C_in/g, k]
// loaded as ggml ne [k, C_in/g, C_out]; bias: [C_out].
ggml_tensor * grouped_conv1d(
    ggml_context * ctx,
    ggml_tensor * input_rows,  // ne [T, C_in, 1]
    ggml_tensor * weight,      // ne [k, C_in/g, C_out]
    ggml_tensor * bias,        // ne [C_out]
    int64_t c_in,
    int64_t c_out,
    int64_t groups,
    int64_t kernel) {
    const int64_t t = input_rows->ne[0];
    const int64_t cg_in = c_in / groups;
    const int64_t cg_out = c_out / groups;
    ggml_tensor * out = nullptr;
    for (int64_t g = 0; g < groups; ++g) {
        // input group slice: rows [T, cg_in] — ne1 offset via view_3d advance
        auto * in_g = ggml_view_3d(
            ctx,
            input_rows,
            t,
            cg_in,
            1,
            input_rows->nb[1],
            input_rows->nb[2],
            g * cg_in * input_rows->nb[1]);
        // im2col with the group's kernel: view weight ne [k, cg_in, cg_out]
        auto * w_g = ggml_view_3d(
            ctx,
            weight,
            kernel,
            cg_in,
            cg_out,
            weight->nb[1],
            weight->nb[2],
            g * cg_out * weight->nb[2]);
        auto * cols = ggml_im2col(ctx, w_g, in_g, 1, 1, kernel / 2, 0, 1, 1, false, GGML_TYPE_F32);
        // 1D im2col result: ne [cg_in*k, T, 1, 1] columns; matmul w2d
        auto * w2 = ggml_reshape_2d(ctx, w_g, cg_in * kernel, cg_out);
        auto * y = ggml_mul_mat(ctx, w2, cols);  // [cg_out, T]
        // bias per-group slice
        auto * b_g = ggml_view_1d(ctx, bias, cg_out, g * cg_out * bias->nb[0]);
        auto * b2 = ggml_reshape_2d(ctx, b_g, cg_out, 1);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
        // columns [cg_out, t] -> rows [t, cg_out]
        auto * y_rows = ggml_cont(ctx, ggml_transpose(ctx, y));
        out = out == nullptr
            ? y_rows
            : ggml_concat(ctx, out, y_rows, 1);  // stack groups on channel dim
    }
    return out;  // rows ne [t, c_out, 1]
}

}  // namespace


// Batched CFG forward: one graph, ne3=2 batch (half 0 = conditioned with
// text_ids, half 1 = uncond with all-zero ids). Same per-half math as two
// f5_dit_forward calls; halves share weights/time-embed/positions, differ
// only in text ids (and optionally cond zeroing, which callers handle on the
// host by uploading a zeroed cond for half 1 if drop_audio_cond).
// Returns {cond, null} mel-major [MEL*N] each.
// ---- batched-CFG helpers (tensors carry B=2 at ne3) ----

// view of one half (ne3 slice) of a [.., .., .., 2] tensor
ggml_tensor * view_of_4d_half(ggml_context * ctx, ggml_tensor * t, int half) {
    // slice along the LAST populated axis (works for [.., k] rank-3/4 batch)
    if (t->ne[3] > 1) {
        return ggml_view_3d(
            ctx, t, t->ne[0], t->ne[1], t->ne[2],
            t->nb[1], t->nb[2], static_cast<size_t>(half) * t->nb[3]);
    }
    // rank-3 batch [F, T, 2]: stride nb[2] is the half size
    return ggml_view_2d(
        ctx, t, t->ne[0], t->ne[1],
        t->nb[1], static_cast<size_t>(half) * t->nb[2]);
}

// concat two 2D halves [F, T] back into [F, T, 1, 2]
ggml_tensor * concat_halves(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return ggml_concat(
        ctx,
        ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, 1),
        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, 1),
        3);  // [F, T, 1, 2] — matches the leaf rank used downstream
}

// lin_apply over a 3D/4D activation: mul_mat handles ne2/ne3 as batch dims;
// bias [out] -> [out, 1, 1, 1] broadcast via repeat
ggml_tensor * lin_apply4(
    ggml_context * ctx,
    const F5Linear & w,
    ggml_tensor * x) {
    auto * out = ggml_mul_mat(ctx, w.weight.tensor, x);
    auto * b2 = ggml_reshape_2d(ctx, w.bias.tensor, ggml_nelements(w.bias.tensor), 1);
    auto * b_rep = ggml_repeat(ctx, b2, out);
    return ggml_add(ctx, out, b_rep);
}

// modulate over batched h: scale/shift [D,1] broadcast over [D, N, 1, 2]
ggml_tensor * modulate4(
    ggml_context * ctx,
    ggml_tensor * h,
    ggml_tensor * scale,
    ggml_tensor * shift,
    ggml_tensor * ones_d1) {
    // ones [D,1] -> [D,1,1,1] to match h rank for the add
    auto * ones4 = ggml_reshape_4d(ctx, ones_d1, ones_d1->ne[0], 1, 1, 1);
    auto * one_rep = ggml_repeat(ctx, ones4, h);
    auto * s4 = ggml_reshape_4d(ctx, scale, scale->ne[0], 1, 1, 1);
    auto * sh4 = ggml_reshape_4d(ctx, shift, shift->ne[0], 1, 1, 1);
    auto * s_rep = ggml_repeat(ctx, s4, h);
    auto * sh_rep = ggml_repeat(ctx, sh4, h);
    return ggml_add(
        ctx, ggml_mul(ctx, h, ggml_add(ctx, one_rep, s_rep)), sh_rep);
}

std::pair<std::vector<float>, std::vector<float>> f5_dit_forward_cfg(
    const std::string & weights_path,
    const std::vector<float> & x_in,
    const std::vector<float> & cond_in,
    const std::vector<int32_t> & text_in,
    float time_value,
    int seq_len,
    const F5Architecture & arch,
    const F5ComputeDevice * device) {
    static const F5ComputeDevice kDefaultDevice{};
    const F5ComputeDevice & dev = device != nullptr ? *device : kDefaultDevice;
    const auto & model = load_model_once(weights_path, dev);
    const auto & W = model.w;
    const int N = seq_len;
    const int MEL = arch.mel_dim;
    const int D = arch.dim;
    const int HEADS = arch.heads;
    const int DH = arch.head_dim;
    const int TD = arch.text_dim;
    const int NT = static_cast<int>(text_in.size());
    const bool is_cuda = model.backend_type == core::BackendType::Cuda;

    struct CfgGraph {
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_buffer_t io_buffer = nullptr;
        ggml_tensor * x = nullptr;        // [MEL, N, 1, 2]
        ggml_tensor * cond = nullptr;     // [MEL, N, 1, 2]
        ggml_tensor * text_ids = nullptr; // [NT, 1, 2, 1] (per-half ids)
        ggml_tensor * th_t = nullptr;     // [256, 1]
        ggml_tensor * output = nullptr;   // [MEL, N] (half 0, conditioned)
        ggml_tensor * out_u = nullptr;    // [MEL, N] (half 1, uncond)
    };
    static auto * cache = new std::map<std::tuple<const LoadedModel *, int, int>, std::unique_ptr<CfgGraph>>();
    const auto ckey = std::make_tuple(&model, N, NT);
    auto it = cache->find(ckey);
    if (it == cache->end()) {
        auto gnew = std::make_unique<CfgGraph>();
        const size_t ctx_bytes = std::min<size_t>(
            std::max<size_t>(1536ULL << 20, static_cast<size_t>(N) * (8ULL << 20)),
            6144ULL << 20);
        gnew->ctx = ggml_init({ctx_bytes, nullptr, is_cuda});
        ggml_context * ctx = gnew->ctx;
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> pending_uploads;
        const auto leaf_write = [&](ggml_tensor * t, const void * src, size_t bytes) {
            if (!is_cuda) {
                std::memcpy(t->data, src, bytes);
            } else {
                const auto * b = static_cast<const uint8_t *>(src);
                pending_uploads.emplace_back(t, std::vector<uint8_t>(b, b + bytes));
            }
        };
        const auto leaf_zero = [&](ggml_tensor * t, size_t bytes) {
            if (!is_cuda) {
                std::memset(t->data, 0, bytes);
            } else {
                pending_uploads.emplace_back(t, std::vector<uint8_t>(bytes, 0));
            }
        };
        // ---- module-composed batched-CFG graph (B=2) ----
        // Leaves: x/cond [2, N, 100] (same values in both halves), ids
        // [2*NT] (cond half, then uncond half), time input [1, 256].
        // Memory layout of [2, N, 100] rows == the old 4d (MEL, N, 1, 2)
        // interleaved halves? No: halves are now batch-major. The per-call
        // upload path below fills each half explicitly.
        const F5DiTWeights & dit_w = load_dit_weights_once(weights_path, dev);
        std::vector<engine::models::f5_tts::ConstStage> * cfg_staged = nullptr;
        auto * stage = const_stage_begin();
        auto io = build_dit_cfg_modules_graph(
            ctx, dit_w, arch, N, NT, model.backend_type);
        const_stage_end(stage);
        cfg_staged = stage;
        ggml_tensor * output = io.output.tensor;   // [2, N, 100]
        ggml_tensor * th_t = io.time_input.tensor;
        gnew->output = output;                     // cond half via view at read time
        gnew->out_u = output;                      // uncond half: same tensor, offset read
        gnew->x = io.x.tensor;
        gnew->cond = io.cond.tensor;
        gnew->text_ids = io.text_ids.tensor;
        gnew->th_t = th_t;
        gnew->graph = ggml_new_graph_custom(ctx, 262144, false);
        ggml_build_forward_expand(gnew->graph, output);
        core::validate_backend_graph_supported(model.backend, gnew->graph, "f5_dit_cfg");
        if (!is_cuda) {
            const int threads = dev.threads > 0 ? dev.threads : static_cast<int>(std::thread::hardware_concurrency());
            core::set_backend_threads(model.backend, threads);
        }
        if (is_cuda) {
            // gallocr-only flow (see f5_dit_forward): no ctx-tensor buffer,
            // the arena owns leaves + constants + intermediates.
            gnew->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (gnew->gallocr == nullptr || !ggml_gallocr_reserve(gnew->gallocr, gnew->graph) ||
                !ggml_gallocr_alloc_graph(gnew->gallocr, gnew->graph)) {
                throw std::runtime_error("F5 DiT CUDA graph alloc failed");
            }
            for (auto & leaf : pending_uploads) {
                ggml_backend_tensor_set(leaf.first, leaf.second.data(), 0, leaf.second.size());
            }
            if (cfg_staged != nullptr) {
                const_stage_upload(cfg_staged);
                const_stage_end(cfg_staged);
            }
        }
        it = cache->emplace(ckey, std::move(gnew)).first;
        // bound the graph cache: each entry holds a CUDA io buffer + gallocr
        // arena (~1-5 GiB at N~1000). Keep at most 2; evict the oldest
        // (map order == insertion order), never the entry just added.
        while (cache->size() > 2) {
            bool evicted = false;
            for (auto cit = cache->begin(); cit != cache->end(); ++cit) {
                if (cit->first != ckey) {
                    cache->erase(cit);
                    evicted = true;
                    break;
                }
            }
            if (!evicted) break;
        }
    }
    CfgGraph & g = *it->second;

    // ---- per-call uploads: x/cond duplicated to both halves, ids per half ----
    {
        const size_t half_bytes = static_cast<size_t>(N) * MEL * sizeof(float);
        std::vector<float> xb(x_in.size() * 2);
        std::memcpy(xb.data(), x_in.data(), half_bytes);
        std::memcpy(xb.data() + x_in.size(), x_in.data(), half_bytes);
        std::vector<float> cb(cond_in.size() * 2);
        std::memcpy(cb.data(), cond_in.data(), half_bytes);
        std::memcpy(cb.data() + cond_in.size(), cond_in.data(), half_bytes);
        std::vector<int32_t> ids(NT * 2);
        for (int i = 0; i < NT; ++i) {
            ids[i] = text_in[i] + 1;          // cond half
            ids[NT + i] = 0;                  // uncond half (drop_text)
        }
        std::vector<float> th(256);
        {
            const float log_base = std::log(10000.0F) / 127.0F;
            for (int i = 0; i < 128; ++i) {
                const float f = 1000.0F * time_value * std::exp(-log_base * i);
                th[i] = std::sin(f);
                th[128 + i] = std::cos(f);
            }
        }
        if (is_cuda) {
            ggml_backend_tensor_set(g.x, xb.data(), 0, xb.size() * sizeof(float));
            ggml_backend_tensor_set(g.cond, cb.data(), 0, cb.size() * sizeof(float));
            ggml_backend_tensor_set(g.text_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(g.th_t, th.data(), 0, th.size() * sizeof(float));
        } else {
            std::memcpy(g.x->data, xb.data(), xb.size() * sizeof(float));
            std::memcpy(g.cond->data, cb.data(), cb.size() * sizeof(float));
            std::memcpy(g.text_ids->data, ids.data(), ids.size() * sizeof(int32_t));
            std::memcpy(g.th_t->data, th.data(), th.size() * sizeof(float));
        }
    }

    // ---- compute + read both halves ----
    const auto status = is_cuda
        ? core::compute_backend_graph(model.backend, g.graph, nullptr, "f5_dit_cfg")
        : ggml_graph_compute_with_ctx(g.ctx, g.graph,
              dev.threads > 0 ? dev.threads : static_cast<int>(std::thread::hardware_concurrency()));
    if (is_cuda) ggml_backend_synchronize(model.backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("F5 DiT CFG graph compute failed");
    }
    std::pair<std::vector<float>, std::vector<float>> out;
    // single [2, N, 100] output tensor, batch-major halves
    const size_t half_floats = ggml_nelements(g.output) / 2;
    out.first.resize(half_floats);
    out.second.resize(half_floats);
    if (is_cuda) {
        ggml_backend_tensor_get(g.output, out.first.data(), 0, half_floats * sizeof(float));
        ggml_backend_tensor_get(g.output, out.second.data(), half_floats * sizeof(float), half_floats * sizeof(float));
    } else {
        const float * all = reinterpret_cast<const float *>(ggml_get_data(g.output));
        std::memcpy(out.first.data(), all, half_floats * sizeof(float));
        std::memcpy(out.second.data(), all + half_floats, half_floats * sizeof(float));
    }
    return out;
}

}  // namespace engine::models::f5_tts


namespace engine::models::f5_tts {


std::vector<float> f5_dit_forward(
    const std::string & weights_path,
    const std::vector<float> & x_in,
    const std::vector<float> & cond_in,
    const std::vector<int32_t> & text_in,
    float time_value,
    int seq_len,
    const F5Architecture & arch,
    bool drop_audio_cond,
    bool drop_text,
    const F5DebugTaps * taps,
    const F5ComputeDevice * device) {
    static const F5ComputeDevice kDefaultDevice{};
    const F5ComputeDevice & dev = device != nullptr ? *device : kDefaultDevice;
    const auto & model = load_model_once(weights_path, dev);
    const auto & W = model.w;
    const int N = seq_len;
    const int MEL = arch.mel_dim;
    const int D = arch.dim;
    const int HEADS = arch.heads;
    const int DH = arch.head_dim;
    const int TD = arch.text_dim;
    const int NT = static_cast<int>(text_in.size());
    const bool is_cuda = model.backend_type == core::BackendType::Cuda;

    // ---- cached graph per (model, N, NT, with/without taps) ----
    // Taps change the graph (extra roots), so key on their presence. The
    // per-call leaves (x, cond, text_ids, th) are uploaded each invocation;
    // everything else (pos ids, pe, ones, constants) is uploaded once.
    struct DiTGraph {
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_buffer_t io_buffer = nullptr;  // CUDA leaves buffer
        core::HostGraphPlan host_plan;              // CPU plan reuse
        ggml_tensor * x = nullptr;
        ggml_tensor * cond = nullptr;
        ggml_tensor * text_ids = nullptr;
        ggml_tensor * th_t = nullptr;
        ggml_tensor * output = nullptr;
        ggml_tensor * tap_text_embed = nullptr;
        ggml_tensor * tap_text_convnext = nullptr;
        ggml_tensor * tap_text_padded = nullptr;
        ggml_tensor * tap_input_embed = nullptr;
        ggml_tensor * tap_time_embed = nullptr;
        ggml_tensor * tap_block0 = nullptr;
        ggml_tensor * tap_block21 = nullptr;
        ~DiTGraph() {
            // Leaked by design when cached (freed only on program-graph reset);
            // destroyed here only when construction throws mid-build.
            if (gallocr != nullptr) ggml_gallocr_free(gallocr);
            if (io_buffer != nullptr) ggml_backend_buffer_free(io_buffer);
            if (ctx != nullptr) ggml_free(ctx);
        }
    };

    // Graph cache is process-lifetime (CUDA buffers cannot be safely freed
    // after driver shutdown in static destruction).
    static auto * graph_cache =
        new std::map<std::tuple<const LoadedModel *, int, int, bool>, std::unique_ptr<DiTGraph>>();
    const bool want_taps = taps != nullptr;
    const auto cache_key = std::make_tuple(&model, N, NT, want_taps);
    auto it = graph_cache->find(cache_key);
    if (it == graph_cache->end()) {
        auto gnew = std::make_unique<DiTGraph>();
        const size_t ctx_bytes = std::min<size_t>(
            std::max<size_t>(1536ULL << 20, static_cast<size_t>(N) * (6ULL << 20)),
            6144ULL << 20);
        gnew->ctx = ggml_init({ctx_bytes, nullptr, is_cuda});
        ggml_context * ctx = gnew->ctx;
        // On CUDA the ctx is no_alloc: leaf tensors get device storage after
        // ggml_backend_alloc_ctx_tensors, values uploaded from staging vectors.
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> pending_uploads;
        const auto leaf_write = [&](ggml_tensor * t, const void * src, size_t bytes) {
            if (!is_cuda) {
                std::memcpy(t->data, src, bytes);
            } else {
                const auto * b = static_cast<const uint8_t *>(src);
                pending_uploads.emplace_back(t, std::vector<uint8_t>(b, b + bytes));
            }
        };
        const auto leaf_zero = [&](ggml_tensor * t, size_t bytes) {
            if (!is_cuda) {
                std::memset(t->data, 0, bytes);
            } else {
                pending_uploads.emplace_back(t, std::vector<uint8_t>(bytes, 0));
            }
        };
        (void)MEL; (void)D; (void)HEADS; (void)DH; (void)TD;
    ggml_tensor * output = nullptr;
    // NOTE: stage taps are not wired in the module-composed graph; the parity
    // harness compares the final output (and uses the raw path where needed).
    ggml_tensor * tap_text_embed = nullptr;
    ggml_tensor * tap_text_convnext = nullptr;
    ggml_tensor * tap_text_padded = nullptr;
    ggml_tensor * tap_input_embed = nullptr;
    ggml_tensor * tap_time_embed = nullptr;
    ggml_tensor * tap_block0 = nullptr;
    ggml_tensor * tap_block21 = nullptr;
    // ---- module-composed graph build (dev-branch reviewer pattern) ----
    // All ops expressed via framework modules; leaves are [1, N, 100]
    // row-major (same memory as the old mel-major columns) so the upload
    // and readback paths are unchanged.
    const F5DiTWeights & dit_w = load_dit_weights_once(weights_path, dev);
    std::vector<engine::models::f5_tts::ConstStage> * staged_module_consts = nullptr;
    auto * stage = const_stage_begin();
    auto io = build_dit_modules_graph(
        ctx, dit_w, arch, N, NT, model.backend_type);
    const_stage_end(stage);
    // staged constants upload after ggml_backend_alloc_ctx_tensors below
    staged_module_consts = stage;
    output = io.output.tensor;
    gnew->x = io.x.tensor;
    gnew->cond = io.cond.tensor;
    gnew->text_ids = io.text_ids.tensor;
    gnew->th_t = io.time_input.tensor;
    gnew->output = output;
        gnew->graph = ggml_new_graph_custom(ctx, 262144, false);
        ggml_build_forward_expand(gnew->graph, output);
        for (ggml_tensor * tap :
             {tap_text_embed, tap_text_convnext, tap_text_padded, tap_input_embed,
              tap_time_embed, tap_block0, tap_block21}) {
            if (tap != nullptr) {
                ggml_build_forward_expand(gnew->graph, tap);
            }
        }
        core::validate_backend_graph_supported(model.backend, gnew->graph, "f5_dit");
        if (!is_cuda) {
            const int threads = dev.threads > 0
                ? dev.threads
                : static_cast<int>(std::thread::hardware_concurrency());
            core::set_backend_threads(model.backend, threads);
        }
        if (is_cuda) {
            // Standard no_alloc flow: the gallocr owns ALL tensors (leaves,
            // constants, intermediates) in one arena sized by liveness; the
            // former ggml_backend_alloc_ctx_tensors double-allocation was
            // fatal for the module graph (~3x more ctx tensors).
            gnew->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (gnew->gallocr == nullptr || !ggml_gallocr_reserve(gnew->gallocr, gnew->graph) ||
                !ggml_gallocr_alloc_graph(gnew->gallocr, gnew->graph)) {
                throw std::runtime_error("F5 DiT CUDA graph alloc failed");
            }
            for (auto & leaf : pending_uploads) {
                ggml_backend_tensor_set(leaf.first, leaf.second.data(), 0, leaf.second.size());
            }
            if (staged_module_consts != nullptr) {
                const_stage_upload(staged_module_consts);
                const_stage_end(staged_module_consts);
                staged_module_consts = nullptr;
            }
        } else {
            if (staged_module_consts != nullptr) {
                const_stage_end(staged_module_consts);
                staged_module_consts = nullptr;
            }
            // CPU: graph data tensors live in the ctx pool (inline alloc);
            // ggml_graph_compute_with_ctx replays without realloc.
        }
        it = graph_cache->emplace(cache_key, std::move(gnew)).first;
    }  // if build
    DiTGraph & g = *it->second;

    // ---- per-call leaf uploads ----
    {
        // x/cond arrive mel-major [N][mel]; ggml [MEL, N] wants (m, n) at
        // n * MEL + m -- identical layout, direct copy. drop_audio_cond
        // zeroes cond on the host before upload.
        std::vector<float> cond_col;
        const float * cond_src = cond_in.data();
        if (drop_audio_cond) {
            cond_col.assign(cond_in.size(), 0.0F);
            cond_src = cond_col.data();
        }
        if (is_cuda) {
            ggml_backend_tensor_set(g.x, x_in.data(), 0, x_in.size() * sizeof(float));
            ggml_backend_tensor_set(g.cond, cond_src, 0, cond_in.size() * sizeof(float));
        } else {
            std::memcpy(g.x->data, x_in.data(), x_in.size() * sizeof(float));
            std::memcpy(g.cond->data, cond_src, cond_in.size() * sizeof(float));
        }
        std::vector<int32_t> ids(NT);
        for (int i = 0; i < NT; ++i) {
            ids[i] = drop_text ? 0 : (text_in[i] + 1);
        }
        if (is_cuda) {
            ggml_backend_tensor_set(g.text_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        } else {
            std::memcpy(g.text_ids->data, ids.data(), ids.size() * sizeof(int32_t));
        }
        std::vector<float> th(256);
        {
            const float log_base = std::log(10000.0F) / 127.0F;
            for (int i = 0; i < 128; ++i) {
                const float f = 1000.0F * time_value * std::exp(-log_base * i);
                th[i] = std::sin(f);
                th[128 + i] = std::cos(f);
            }
        }
        if (is_cuda) {
            ggml_backend_tensor_set(g.th_t, th.data(), 0, th.size() * sizeof(float));
        } else {
            std::memcpy(g.th_t->data, th.data(), th.size() * sizeof(float));
        }
    }

    // ---- compute ----
    std::vector<float> out;
    const auto status = is_cuda
        ? core::compute_backend_graph(model.backend, g.graph, nullptr, "f5_dit")
        : ggml_graph_compute_with_ctx(g.ctx, g.graph,
              dev.threads > 0 ? dev.threads
                              : static_cast<int>(std::thread::hardware_concurrency()));
    if (is_cuda) {
        ggml_backend_synchronize(model.backend);
    }
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("F5 DiT graph compute failed");
    }
    out.resize(ggml_nelements(g.output));
    if (is_cuda) {
        ggml_backend_tensor_get(g.output, out.data(), 0, out.size() * sizeof(float));
    } else {
        std::memcpy(out.data(), ggml_get_data(g.output), out.size() * sizeof(float));
    }
    if (taps != nullptr) {
        const auto read_tap = [&](ggml_tensor * t, std::vector<float> * dst) {
            if (t != nullptr && dst != nullptr) {
                dst->resize(ggml_nelements(t));
                if (is_cuda) {
                    ggml_backend_tensor_get(t, dst->data(), 0, dst->size() * sizeof(float));
                } else {
                    std::memcpy(dst->data(), ggml_get_data(t), dst->size() * sizeof(float));
                }
            }
        };
        read_tap(g.tap_text_embed, taps->text_embed);
        read_tap(g.tap_text_convnext, taps->text_convnext);
        read_tap(g.tap_text_padded, taps->text_padded);
        read_tap(g.tap_input_embed, taps->input_embed);
        read_tap(g.tap_time_embed, taps->time_embed);
        read_tap(g.tap_block0, taps->block0);
        read_tap(g.tap_block21, taps->block21);
    }
    return out;  // [MEL * N] mel-major columns: out[m * N + n]
}


}  // namespace engine::models::f5_tts

