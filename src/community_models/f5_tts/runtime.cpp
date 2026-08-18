#include "engine/community_models/f5_tts/runtime.h"

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
    core::BackendType backend_type) {
    F5Weights w;
    w.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "f5_tts.weights", 2ULL * 1024ULL * 1024ULL * 1024ULL);
    const auto f32 = [&](const std::string & n) {
        return w.store->load_f32_tensor(
            source, n, source.require_metadata(n).shape);
    };
    const auto lin = [&](const std::string & n) {
        return F5Linear{f32(n + ".weight"), f32(n + ".bias")};
    };

    w.text_embedding = f32("text_embed.text_embed.weight");
    w.input_proj = lin("input_embed.proj");
    w.cpe0.weight = f32("input_embed.conv_pos_embed.conv1d.0.weight");
    w.cpe0.bias = f32("input_embed.conv_pos_embed.conv1d.0.bias");
    w.cpe2.weight = f32("input_embed.conv_pos_embed.conv1d.2.weight");
    w.cpe2.bias = f32("input_embed.conv_pos_embed.conv1d.2.bias");
    w.time0 = lin("time_embed.time_mlp.0");
    w.time2 = lin("time_embed.time_mlp.2");
    w.inv_freq = source.require_f32("rotary_embed.inv_freq");

    w.text_blocks.reserve(4);
    for (int i = 0; i < 4; ++i) {
        const std::string p = "text_embed.text_blocks." + std::to_string(i);
        F5ConvNeXt b;
        b.dwconv.weight = f32(p + ".dwconv.weight");
        b.dwconv.bias = f32(p + ".dwconv.bias");
        b.norm_w = f32(p + ".norm.weight");
        b.norm_b = f32(p + ".norm.bias");
        b.pw1 = lin(p + ".pwconv1");
        b.pw2 = lin(p + ".pwconv2");
        b.grn_gamma = source.require_f32(p + ".grn.gamma");
        b.grn_beta = source.require_f32(p + ".grn.beta");
        w.text_blocks.push_back(std::move(b));
    }
    w.blocks.reserve(22);
    for (int i = 0; i < 22; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i);
        F5Block b;
        b.attn_norm = lin(p + ".attn_norm.linear");
        b.to_q = lin(p + ".attn.to_q");
        b.to_k = lin(p + ".attn.to_k");
        b.to_v = lin(p + ".attn.to_v");
        b.to_out = lin(p + ".attn.to_out.0");
        b.ff0 = lin(p + ".ff.ff.0.0");
        b.ff2 = lin(p + ".ff.ff.2");
        w.blocks.push_back(std::move(b));
    }
    w.norm_out = lin("norm_out.linear");
    w.proj_out = lin("proj_out");
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
    model.w = load_weights(*stripped, owner->value, type);
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

    // CPU path: inline-allocating context (constants written at build time;
    // weights are host memory on the CPU-backend store).
    // CUDA path: no_alloc context; constants/inputs are uploaded via
    // ggml_backend_tensor_set after ggml_backend_alloc_ctx_tensors.
    const bool is_cuda = model.backend_type == core::BackendType::Cuda;
    const size_t ctx_bytes = std::min<size_t>(
        std::max<size_t>(1536ULL << 20, static_cast<size_t>(N) * (6ULL << 20)),
        6144ULL << 20);
    ggml_context * ctx = ggml_init({ctx_bytes, nullptr, is_cuda});
    // On CUDA the ctx is no_alloc: leaf tensors get device storage after
    // ggml_backend_alloc_ctx_tensors, and their values are uploaded from these
    // staging vectors. On CPU the writes below go directly into ctx memory.
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
    ggml_tensor * output = nullptr;
    ggml_tensor * tap_text_embed = nullptr;
    ggml_tensor * tap_text_convnext = nullptr;
    ggml_tensor * tap_text_padded = nullptr;
    ggml_tensor * tap_input_embed = nullptr;
    ggml_tensor * tap_time_embed = nullptr;
    ggml_tensor * tap_block0 = nullptr;
    ggml_tensor * tap_block21 = nullptr;
    {
        // ---- inputs [MEL, N] ----
        // x_in/cond_in arrive mel-major [N][mel]; transpose on host into
        // column-major [mel][n].
        std::vector<float> x_col(static_cast<size_t>(N) * MEL);
        std::vector<float> cond_col(static_cast<size_t>(N) * MEL);
        // ggml [MEL, N] tensor memory: element (m, n) at n * MEL + m
        // (ne0 = MEL is the fastest axis).
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < MEL; ++m) {
                x_col[static_cast<size_t>(n) * MEL + m] = x_in[static_cast<size_t>(n) * MEL + m];
                const float cv = drop_audio_cond ? 0.0F : cond_in[static_cast<size_t>(n) * MEL + m];
                cond_col[static_cast<size_t>(n) * MEL + m] = cv;
            }
        }
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MEL, N);
        leaf_write(x, x_col.data(), x_col.size() * sizeof(float));
        auto * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MEL, N);
        leaf_write(cond, cond_col.data(), cond_col.size() * sizeof(float));

        // ---- text embed ----
        auto * text_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NT);
        {
            std::vector<int32_t> ids(NT);
            for (int i = 0; i < NT; ++i) {
                ids[i] = drop_text ? 0 : (text_in[i] + 1);
            }
            leaf_write(text_ids, ids.data(), ids.size() * sizeof(int32_t));
        }
        auto * te = ggml_get_rows(ctx, W.text_embedding.tensor, text_ids);  // [TD, NT]
        if (taps != nullptr && taps->text_embed != nullptr) {
            tap_text_embed = ggml_cont(ctx, te);
            ggml_set_output(tap_text_embed);
        }

        // sinus position embedding (precompute_freqs_cis: cat(cos, sin))
        {
            // Layout: pe[col t][row i] stored t-major; we need [TD, NT] column
            // tensor: element (i, t) at data[t * TD + i].
            std::vector<float> pe(static_cast<size_t>(TD) * NT);
            const int half = TD / 2;
            for (int pos = 0; pos < NT; ++pos) {
                for (int i = 0; i < half; ++i) {
                    const float inv = std::pow(10000.0F, -2.0F * i / static_cast<float>(TD));
                    const float f = pos * inv;
                    pe[static_cast<size_t>(pos) * TD + i] = std::cos(f);
                    pe[static_cast<size_t>(pos) * TD + half + i] = std::sin(f);
                }
            }
            auto * pe_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, TD, NT);
            leaf_write(pe_t, pe.data(), pe.size() * sizeof(float));
            te = ggml_add(ctx, te, pe_t);
        }

        // ---- 4x ConvNeXt over text (column layout throughout) ----
        for (int bi = 0; bi < 4; ++bi) {
            const auto & B = W.text_blocks[bi];
            auto * dw = depthwise_conv7(
                ctx, te, B.dwconv.weight.tensor, B.dwconv.bias->tensor,
                leaf_write, leaf_zero);
            auto * nrm = affine_norm(ctx, dw, B.norm_w.tensor, B.norm_b.tensor);
            auto * h1 = lin_apply(ctx, B.pw1, nrm);  // [1024, NT]
            h1 = ggml_gelu(ctx, h1);                  // exact erf
            // GRN: per-feature L2 over sequence
            {
                auto * sq = ggml_sqr(ctx, h1);
                // sum over sequence (ne1): transpose to [NT, 1024], sum_rows -> [1, 1024]
                auto * tr = ggml_cont(ctx, ggml_transpose(ctx, sq));  // [NT, 1024]
                auto * ssum = ggml_sum_rows(ctx, tr);                  // [1, 1024]
                auto * gx = ggml_sqrt(ctx, ssum);                      // [1, 1024]
                // ggml_mean on [1, N] is identity (row-wise over ne0); use
                // sum + scale for a true scalar mean over features.
                auto * mean = ggml_scale(ctx, ggml_sum(ctx, gx), 1.0F / 1024);  // [1,1]
                auto * eps_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                {
                    const float eps_val = 1e-6F;
                    leaf_write(eps_t, &eps_val, sizeof(float));
                }
                auto * nx = ggml_div(ctx, gx, ggml_add(ctx, mean, eps_t));
                auto * nx_col = ggml_cont(ctx, ggml_transpose(ctx, nx));  // [1024, 1]
                auto * nx_rep = ggml_repeat(ctx, nx_col, h1);
                auto * scaled = ggml_mul(ctx, h1, nx_rep);
                static thread_local std::vector<float> gbuf, bbuf;
                gbuf = B.grn_gamma;
                bbuf = B.grn_beta;
                auto * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
                leaf_write(gamma, gbuf.data(), 1024 * sizeof(float));
                auto * beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
                leaf_write(beta, bbuf.data(), 1024 * sizeof(float));
                auto * g2 = ggml_reshape_2d(ctx, gamma, 1024, 1);
                auto * b2 = ggml_reshape_2d(ctx, beta, 1024, 1);
                auto * g_rep = ggml_repeat(ctx, g2, scaled);
                auto * b_rep = ggml_repeat(ctx, b2, scaled);
                auto * grn_out = ggml_add(
                    ctx, ggml_add(ctx, ggml_mul(ctx, g_rep, scaled), b_rep), h1);
                auto * h2 = lin_apply(ctx, B.pw2, grn_out);  // [512, NT]
                te = ggml_add(ctx, te, h2);
            }
        }

        if (taps != nullptr && taps->text_convnext != nullptr) {
            tap_text_convnext = ggml_cont(ctx, te);
            ggml_set_output(tap_text_convnext);
        }

        // ---- pad/curtail text to N (pure graph ops; te data is not valid at
        // build time) ----
        ggml_tensor * te_pad;
        if (NT >= N) {
            te_pad = ggml_cont(ctx, ggml_view_2d(ctx, te, TD, N, te->nb[1], 0));
        } else {
            // zero-pad columns: concat te with a zeros [TD, N-NT] constant
            auto * zeros = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, TD, N - NT);
            leaf_zero(zeros, static_cast<size_t>(TD) * (N - NT) * sizeof(float));
            te_pad = ggml_concat(ctx, te, zeros, 1);  // [TD, N]
        }

        if (taps != nullptr && taps->text_padded != nullptr) {
            tap_text_padded = ggml_cont(ctx, te_pad);
            ggml_set_output(tap_text_padded);
        }

        // ---- input embed: concat rows [MEL; MEL; TD] -> 712 ----
        auto * cat0 = ggml_concat(ctx, x, cond, 0);  // [200, N]
        auto * cat1 = ggml_concat(ctx, cat0, te_pad, 0);  // [712, N]
        auto * inp = lin_apply(ctx, W.input_proj, cat1);  // [1024, N]

        // ---- conv pos embed (grouped k31 g16, Mish x2) ----
        // Verified layout: ggml conv path wants ne [T, C, 1] with TIME as the
        // fastest axis (element (t,c) at t + c*T). inp is [D, N] columns
        // (feature-fastest); ggml_transpose -> [N, D] is exactly time-fastest.
        {
            auto * rows = ggml_cont(ctx, ggml_transpose(ctx, inp));  // ne [N, D]
            auto * r0 = grouped_conv1d(
                ctx, ggml_reshape_3d(ctx, rows, N, D, 1),
                W.cpe0.weight.tensor, W.cpe0.bias->tensor, D, D, 16, 31);
            // Mish: x * tanh(softplus(x))
            r0 = ggml_mul(ctx, r0, ggml_tanh(ctx, ggml_softplus(ctx, r0)));
            auto * r1 = grouped_conv1d(
                ctx, ggml_reshape_3d(ctx, r0, N, D, 1),
                W.cpe2.weight.tensor, W.cpe2.bias->tensor, D, D, 16, 31);
            r1 = ggml_mul(ctx, r1, ggml_tanh(ctx, ggml_softplus(ctx, r1)));
            // r1 is ne [N, D, 1] time-fastest; back to columns [D, N]
            auto * c1_cols = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, r1, N, D)));
            inp = ggml_add(ctx, inp, c1_cols);
        }

        if (taps != nullptr && taps->input_embed != nullptr) {
            tap_input_embed = ggml_cont(ctx, inp);
            ggml_set_output(tap_input_embed);
        }

        // ---- time embed ----
        std::vector<float> th(256);
        {
            const float log_base = std::log(10000.0F) / 127.0F;
            for (int i = 0; i < 128; ++i) {
                const float f = 1000.0F * time_value * std::exp(-log_base * i);
                th[i] = std::sin(f);
                th[128 + i] = std::cos(f);
            }
        }
        auto * th_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        leaf_write(th_t, th.data(), th.size() * sizeof(float));
        auto * t0 = lin_apply(ctx, W.time0, th_t);
        t0 = ggml_silu(ctx, t0);
        auto * t_emb = lin_apply(ctx, W.time2, t0);  // [1024, 1]

        auto * ones_d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        {
            static const std::vector<float> ones(D, 1.0F);
            leaf_write(ones_d, ones.data(), D * sizeof(float));
        }
        auto * ones_d1 = ggml_reshape_2d(ctx, ones_d, D, 1);

        if (taps != nullptr && taps->time_embed != nullptr) {
            tap_time_embed = ggml_cont(ctx, t_emb);
            ggml_set_output(tap_time_embed);
        }

        // ---- RoPE positions ----
        auto * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        {
            static thread_local std::vector<int32_t> pos;
            pos.resize(N);
            for (int i = 0; i < N; ++i) {
                pos[i] = i;
            }
            leaf_write(pos_ids, pos.data(), N * sizeof(int32_t));
        }

        // ---- 22 DiT blocks ----
        auto * h = inp;
        for (int bi = 0; bi < arch.depth; ++bi) {
            const auto & B = W.blocks[bi];
            auto * emb = lin_apply(ctx, B.attn_norm, ggml_silu(ctx, t_emb));  // [6144, 1]
            auto * shift_msa = chunk_col(ctx, emb, 0, D);
            auto * scale_msa = chunk_col(ctx, emb, 1, D);
            auto * gate_msa = chunk_col(ctx, emb, 2, D);
            auto * shift_mlp = chunk_col(ctx, emb, 3, D);
            auto * scale_mlp = chunk_col(ctx, emb, 4, D);
            auto * gate_mlp = chunk_col(ctx, emb, 5, D);

            auto * norm = modulate(ctx, ggml_norm(ctx, h, 1e-6F), scale_msa, shift_msa, ones_d1);
            auto * q = lin_apply(ctx, B.to_q, norm);
            auto * k = lin_apply(ctx, B.to_k, norm);
            auto * v = lin_apply(ctx, B.to_v, norm);
            // [1024, N] -> [DH, H, N]: rope layout (positions at ne2)
            q = ggml_reshape_3d(ctx, q, DH, HEADS, N);
            k = ggml_reshape_3d(ctx, k, DH, HEADS, N);
            v = ggml_reshape_3d(ctx, v, DH, HEADS, N);
            // interleaved (pair) RoPE over head dim, theta 10000 = F5 inv_freq
            q = ggml_rope_ext(
                ctx, q, pos_ids, nullptr, DH, GGML_ROPE_TYPE_NORMAL, 0,
                10000.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
            k = ggml_rope_ext(
                ctx, k, pos_ids, nullptr, DH, GGML_ROPE_TYPE_NORMAL, 0,
                10000.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
            // rope layout [DH, H, N] -> flash-attn layout [DH, N, H]
            q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
            auto * attn = ggml_flash_attn_ext(
                ctx, q, k, v, nullptr,
                1.0F / std::sqrt(static_cast<float>(DH)), 0.0F, 0.0F);
            // res: [DH, H, N] permuted -> flatten to [D, N]
            auto * attn2 = ggml_reshape_2d(ctx, ggml_cont(ctx, attn), D, N);
            auto * proj = lin_apply(ctx, B.to_out, attn2);
            h = ggml_add(ctx, h, ggml_mul(ctx, proj, ggml_repeat(ctx, gate_msa, proj)));

            auto * norm2 = modulate(ctx, ggml_norm(ctx, h, 1e-6F), scale_mlp, shift_mlp, ones_d1);
            auto * f1 = lin_apply(ctx, B.ff0, norm2);
            // FeedForward(approximate="tanh"):
            // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
            {
                auto * cube = ggml_mul(ctx, f1, ggml_mul(ctx, f1, f1));
                auto * inner = ggml_add(ctx, f1, ggml_scale(ctx, cube, 0.044715F));
                auto * tanh_part = ggml_tanh(
                    ctx, ggml_scale(ctx, inner, 0.7978845608028654F));
                // +1 via adding ones of matching shape
                auto * one_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f1->ne[0], f1->ne[1]);
                {
                    const size_t need = ggml_nelements(one_t);
                    static thread_local std::vector<float> ones_fill;
                    if (ones_fill.size() < need) {
                        ones_fill.assign(need, 1.0F);
                    }
                    leaf_write(one_t, ones_fill.data(), need * sizeof(float));
                }
                f1 = ggml_scale(
                    ctx, ggml_mul(ctx, f1, ggml_add(ctx, tanh_part, one_t)), 0.5F);
            }
            auto * f2 = lin_apply(ctx, B.ff2, f1);
            h = ggml_add(ctx, h, ggml_mul(ctx, f2, ggml_repeat(ctx, gate_mlp, f2)));
            if (taps != nullptr && taps->block0 != nullptr && bi == 0) {
                tap_block0 = ggml_cont(ctx, h);
                ggml_set_output(tap_block0);
            }
            if (taps != nullptr && taps->block21 != nullptr && bi == arch.depth - 1) {
                tap_block21 = ggml_cont(ctx, h);
                ggml_set_output(tap_block21);
            }
        }

        // ---- final adaLN + proj ----
        {
            auto * emb = lin_apply(ctx, W.norm_out, ggml_silu(ctx, t_emb));  // [2048, 1]
            auto * scale = chunk_col(ctx, emb, 0, D);
            auto * shift = chunk_col(ctx, emb, 1, D);
            auto * norm = modulate(ctx, ggml_norm(ctx, h, 1e-6F), scale, shift, ones_d1);
            output = lin_apply(ctx, W.proj_out, norm);  // [100, N]
        }
    }

    // ---- compute ----
    std::vector<float> out;
    if (!is_cuda) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 262144, false);
        ggml_build_forward_expand(graph, output);
        for (ggml_tensor * tap :
             {tap_text_embed, tap_text_convnext, tap_text_padded, tap_input_embed,
              tap_time_embed, tap_block0, tap_block21}) {
            if (tap != nullptr) {
                ggml_build_forward_expand(graph, tap);
            }
        }
        const int threads = dev.threads > 0 ? dev.threads
                                            : static_cast<int>(std::thread::hardware_concurrency());
        const auto status = ggml_graph_compute_with_ctx(ctx, graph, threads);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            throw std::runtime_error("F5 DiT graph compute failed");
        }
        out.resize(ggml_nelements(output));
        std::memcpy(out.data(), ggml_get_data(output), out.size() * sizeof(float));
        const auto read_tap = [](ggml_tensor * t, std::vector<float> * dst) {
            if (t != nullptr && dst != nullptr) {
                dst->resize(ggml_nelements(t));
                std::memcpy(dst->data(), ggml_get_data(t), dst->size() * sizeof(float));
            }
        };
        read_tap(tap_text_embed, taps == nullptr ? nullptr : taps->text_embed);
        read_tap(tap_text_convnext, taps == nullptr ? nullptr : taps->text_convnext);
        read_tap(tap_text_padded, taps == nullptr ? nullptr : taps->text_padded);
        read_tap(tap_input_embed, taps == nullptr ? nullptr : taps->input_embed);
        read_tap(tap_time_embed, taps == nullptr ? nullptr : taps->time_embed);
        read_tap(tap_block0, taps == nullptr ? nullptr : taps->block0);
        read_tap(tap_block21, taps == nullptr ? nullptr : taps->block21);
    } else {
        // CUDA: mark leaves as inputs, allocate into a backend buffer, upload
        // data, build graph, compute via gallocr, read back.
        ggml_backend_buffer_t io_buffer =
            ggml_backend_alloc_ctx_tensors(ctx, model.backend);
        if (io_buffer == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error("F5 DiT CUDA io buffer alloc failed");
        }
        for (auto & leaf : pending_uploads) {
            ggml_backend_tensor_set(
                leaf.first, leaf.second.data(), 0, leaf.second.size());
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 262144, false);
        ggml_build_forward_expand(graph, output);
        for (ggml_tensor * tap :
             {tap_text_embed, tap_text_convnext, tap_text_padded, tap_input_embed,
              tap_time_embed, tap_block0, tap_block21}) {
            if (tap != nullptr) {
                ggml_build_forward_expand(graph, tap);
            }
        }
        core::validate_backend_graph_supported(model.backend, graph, "f5_dit");
        ggml_gallocr_t allocator =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (allocator == nullptr || !ggml_gallocr_reserve(allocator, graph) ||
            !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator != nullptr) ggml_gallocr_free(allocator);
            ggml_backend_buffer_free(io_buffer);
            ggml_free(ctx);
            throw std::runtime_error("F5 DiT CUDA graph alloc failed");
        }
        const auto status = core::compute_backend_graph(model.backend, graph, nullptr, "f5_dit");
        ggml_backend_synchronize(model.backend);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(allocator);
            ggml_backend_buffer_free(io_buffer);
            ggml_free(ctx);
            throw std::runtime_error("F5 DiT CUDA graph compute failed");
        }
        out.resize(ggml_nelements(output));
        ggml_backend_tensor_get(output, out.data(), 0, out.size() * sizeof(float));
        const auto read_tap = [&](ggml_tensor * t, std::vector<float> * dst) {
            if (t != nullptr && dst != nullptr) {
                dst->resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, dst->data(), 0, dst->size() * sizeof(float));
            }
        };
        read_tap(tap_text_embed, taps == nullptr ? nullptr : taps->text_embed);
        read_tap(tap_text_convnext, taps == nullptr ? nullptr : taps->text_convnext);
        read_tap(tap_text_padded, taps == nullptr ? nullptr : taps->text_padded);
        read_tap(tap_input_embed, taps == nullptr ? nullptr : taps->input_embed);
        read_tap(tap_time_embed, taps == nullptr ? nullptr : taps->time_embed);
        read_tap(tap_block0, taps == nullptr ? nullptr : taps->block0);
        read_tap(tap_block21, taps == nullptr ? nullptr : taps->block21);
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(io_buffer);
    }
    ggml_free(ctx);
    return out;  // [MEL * N] mel-major columns: out[m * N + n]
}

}  // namespace engine::models::f5_tts
