#include "engine/community_models/zonos3/ssl_teacher.h"

#include "engine/community_models/zonos3/resample.h"
#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos3 {
namespace {

using core::TensorValue;

constexpr int kLayers = 16;
constexpr int kEmbd = 1024;
constexpr int kHeads = 16;
constexpr int kFf = 4096;
constexpr int kRelLeft = 64;
constexpr int kRelRight = 8;
constexpr int kConvKernel = 31;
constexpr int kMels = 80;
constexpr int kFrameLength = 400;
constexpr int kHop = 160;
constexpr int kFft = 512;
constexpr int kStride = 2;
constexpr int kFeature = kMels * kStride;
constexpr int kMaxSeconds = 60;
constexpr int kMaxFrames = 3000;
constexpr float kEps = 1e-5f;
constexpr float kMelFloor = 1.192092955078125e-07f;
constexpr int kGraphNodes = 8192;

// Logical (row-major) view of any ggml tensor: the reverse of its ne over ggml_n_dims axes.
TensorValue tv(ggml_tensor * x) {
    const int rank = ggml_n_dims(x);
    core::TensorShape shape;
    shape.rank = static_cast<size_t>(rank);
    for (int i = 0; i < rank; ++i) shape.dims[static_cast<size_t>(i)] = x->ne[rank - 1 - i];
    return core::wrap_tensor(x, shape);
}

// CUDA autocast materializes BF16 tensors at these boundaries only.
ggml_tensor * bf16(core::ModuleBuildContext & ctx, ggml_tensor * x) {
    return modules::Bf16RoundModule().build(ctx, tv(x)).tensor;
}

// BF16 operands, F32 accumulation (autocast GEMM).
ggml_tensor * matmul_bf16(ggml_context * g, ggml_tensor * a, ggml_tensor * b) {
    if (a->type != GGML_TYPE_BF16) a = ggml_cast(g, a, GGML_TYPE_BF16);
    auto * out = ggml_mul_mat(g, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

// NumPy's spectrogram computes an F64 FFT (n = 512), then materializes its complex64 result.
void fft512(std::vector<std::complex<double>> & a) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[static_cast<size_t>(i)], a[static_cast<size_t>(j)]);
    }
    static const std::array<std::complex<double>, 256> twiddle = [] {
        std::array<std::complex<double>, 256> t{};
        for (int i = 0; i < 256; ++i) {
            const double angle = -2.0 * std::acos(-1.0) * i / 512;
            t[static_cast<size_t>(i)] = {std::cos(angle), std::sin(angle)};
        }
        return t;
    }();
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                const auto u = a[static_cast<size_t>(i + j)];
                const auto v = a[static_cast<size_t>(i + j + len / 2)] * twiddle[static_cast<size_t>(j * (n / len))];
                a[static_cast<size_t>(i + j)] = u + v;
                a[static_cast<size_t>(i + j + len / 2)] = u - v;
            }
        }
    }
}

// The extractor reduces each contiguous frame column of its transposed fbank with NumPy's eight-accumulator,
// 128-element pairwise summation; sequential F32 accumulation changes long padded-lane statistics.
template <bool squared>
float pairwise_sum(const float * x, int n, int stride, float mean = 0.0f) {
    auto value = [&](int i) {
        if constexpr (squared) {
            const float delta = x[static_cast<size_t>(i) * static_cast<size_t>(stride)] - mean;
            // NumPy materializes the F32 squared deviations before reducing them.
            volatile const float square = delta * delta;
            return static_cast<float>(square);
        } else {
            return x[static_cast<size_t>(i) * static_cast<size_t>(stride)];
        }
    };
    if (n < 8) {
        float sum = -0.0f;
        for (int i = 0; i < n; ++i) sum += value(i);
        return sum;
    }
    if (n <= 128) {
        float sums[8];
        for (int j = 0; j < 8; ++j) sums[j] = value(j);
        const int end = n - n % 8;
        int i = 8;
        for (; i < end; i += 8) {
            for (int j = 0; j < 8; ++j) sums[j] += value(i + j);
        }
        float sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < n; ++i) sum += value(i);
        return sum;
    }
    int half = n / 2;
    half -= half % 8;
    return pairwise_sum<squared>(x, half, stride, mean) +
           pairwise_sum<squared>(x + static_cast<size_t>(half) * static_cast<size_t>(stride), n - half, stride, mean);
}

std::vector<double> f64_table(const assets::TensorSource & source, const std::string & name, const std::vector<int64_t> & shape) {
    const auto raw = source.require_tensor_data(name);
    if (raw.metadata.dtype != "f64" || raw.metadata.shape != shape) {
        throw std::runtime_error("zonos3: ssl-teacher GGUF: " + name + " must be an F64 table");
    }
    std::vector<double> out(raw.bytes.size() / sizeof(double));
    std::memcpy(out.data(), raw.bytes.data(), out.size() * sizeof(double));
    return out;
}

struct LayerWeights {
    std::array<TensorValue, 2> ffn_ln_w, ffn_ln_b, ffn_up, ffn_up_b, ffn_down, ffn_down_b;
    TensorValue attn_ln_w, attn_ln_b, dist;
    std::array<TensorValue, 4> attn, attn_b;  // q, k, v, o
    TensorValue conv_ln_w, conv_ln_b, pw1, dw, dw_ln_w, dw_ln_b, pw2, final_ln_w, final_ln_b;
};

}  // namespace

class Zonos3SslTeacher::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : source_(std::move(source)), execution_(execution) {
        validate_metadata();
        load_weights();
    }

    ~Impl() { graph_.reset(); }

    std::vector<float> encode(const std::vector<float> & mono_44k, int & frames) const {
        if (mono_44k.empty() || mono_44k.size() > static_cast<size_t>(44100) * kMaxSeconds) {
            throw std::runtime_error("zonos3: the SSL teacher takes 1 sample to 60 s of 44.1 kHz audio");
        }
        std::vector<float> features;
        std::vector<int32_t> mask;
        compute_features(zonos3_resample_44k_to_16k(resample_kernel_, mono_44k), features, frames, mask);
        if (frames > kMaxFrames) throw std::runtime_error("zonos3: SSL teacher input exceeds 3000 frames");
        if (!graph_ || frames_ != frames) build_graph(frames);
        const int T = frames;
        std::vector<float> valid(static_cast<size_t>(T));
        for (int i = 0; i < T; ++i) valid[static_cast<size_t>(i)] = mask[static_cast<size_t>(i)] ? 1.0f : 0.0f;
        // Invalid keys get -max(bf16), the autocast attention mask value.
        const float blocked = -ggml_bf16_to_fp32(ggml_bf16_t{0x7f7f});
        std::vector<float> attn_mask(static_cast<size_t>(T) * static_cast<size_t>(T));
        for (int q = 0; q < T; ++q) {
            for (int k = 0; k < T; ++k) attn_mask[static_cast<size_t>(q) * T + k] = valid[static_cast<size_t>(k)] != 0.0f ? 0.0f : blocked;
        }
        ggml_backend_tensor_set(input_, features.data(), 0, static_cast<size_t>(T) * kFeature * sizeof(float));
        ggml_backend_tensor_set(valid_, valid.data(), 0, valid.size() * sizeof(float));
        ggml_backend_tensor_set(mask_, attn_mask.data(), 0, attn_mask.size() * sizeof(float));
        graph_->compute();
        std::vector<float> out(static_cast<size_t>(T) * kEmbd);
        ggml_backend_tensor_get(output_, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

private:
    void validate_metadata() const {
        const assets::GgufMetadata meta(source_->source_path());
        if (meta.require_string("general.architecture") != "w2vbert_ssl") {
            throw std::runtime_error("zonos3: ssl-teacher GGUF: general.architecture must be w2vbert_ssl");
        }
        const std::array<std::pair<const char *, int64_t>, 14> ints{{
            {"sample_rate", 16000}, {"max_seconds", kMaxSeconds}, {"n_layer", kLayers}, {"n_embd", kEmbd},
            {"n_head", kHeads}, {"n_ff", kFf}, {"rel_left", kRelLeft}, {"rel_right", kRelRight},
            {"conv_kernel", kConvKernel}, {"n_mels", kMels}, {"frame_length", kFrameLength},
            {"hop_length", kHop}, {"fft_length", kFft}, {"stride", kStride}}};
        for (const auto & [key, expected] : ints) {
            const std::string full = std::string("w2vbert_ssl.") + key;
            if (meta.require_int(full.c_str()) != expected) {
                throw std::runtime_error("zonos3: ssl-teacher GGUF: " + full + " must be " + std::to_string(expected));
            }
        }
        const std::array<std::pair<const char *, float>, 4> floats{{
            {"eps", kEps}, {"preemphasis", 0.97f}, {"padding_value", 1.0f}, {"mel_floor", kMelFloor}}};
        for (const auto & [key, expected] : floats) {
            const std::string full = std::string("w2vbert_ssl.") + key;
            if (meta.require_f32(full.c_str()) != expected) {
                throw std::runtime_error("zonos3: ssl-teacher GGUF: " + full + " has an unsupported value");
            }
        }
    }

    void load_weights() {
        const auto & source = *source_;
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos3.ssl_teacher.weights", 4 * 1024 * 1024);
        size_t loaded = 0;
        auto f32 = [&](const std::string & name, std::vector<int64_t> shape) {
            ++loaded;
            return store_->load_f32_tensor(source, name, shape);
        };
        // Autocast GEMM operands: BF16 storage is exactly the in-graph BF16 cast of the F32 checkpoint.
        auto bf16_matrix = [&](const std::string & name, std::vector<int64_t> shape) {
            ++loaded;
            return store_->load_tensor(source, name, assets::TensorStorageType::BF16, shape);
        };
        fp_ln_w_ = f32("fp.ln.weight", {kFeature});
        fp_ln_b_ = f32("fp.ln.bias", {kFeature});
        fp_proj_ = bf16_matrix("fp.proj.weight", {kEmbd, kFeature});
        fp_proj_b_ = f32("fp.proj.bias", {kEmbd});
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "enc." + std::to_string(l) + ".";
            LayerWeights w;
            for (size_t j = 0; j < 2; ++j) {
                const std::string f = p + "ffn" + std::to_string(j + 1);
                w.ffn_ln_w[j] = f32(f + "_ln.weight", {kEmbd});
                w.ffn_ln_b[j] = f32(f + "_ln.bias", {kEmbd});
                w.ffn_up[j] = bf16_matrix(f + ".up.weight", {kFf, kEmbd});
                w.ffn_up_b[j] = f32(f + ".up.bias", {kFf});
                w.ffn_down[j] = bf16_matrix(f + ".down.weight", {kEmbd, kFf});
                w.ffn_down_b[j] = f32(f + ".down.bias", {kEmbd});
            }
            w.attn_ln_w = f32(p + "attn_ln.weight", {kEmbd});
            w.attn_ln_b = f32(p + "attn_ln.bias", {kEmbd});
            const std::array<const char *, 4> names{"q", "k", "v", "o"};
            for (size_t i = 0; i < names.size(); ++i) {
                w.attn[i] = bf16_matrix(p + "attn." + names[i] + ".weight", {kEmbd, kEmbd});
                w.attn_b[i] = f32(p + "attn." + names[i] + ".bias", {kEmbd});
            }
            w.dist = bf16_matrix(p + "attn.dist", {kRelLeft + kRelRight + 1, kEmbd / kHeads});
            w.conv_ln_w = f32(p + "conv_ln.weight", {kEmbd});
            w.conv_ln_b = f32(p + "conv_ln.bias", {kEmbd});
            w.pw1 = bf16_matrix(p + "conv.pw1.weight", {2 * kEmbd, kEmbd});
            w.dw = f32(p + "conv.dw.weight", {kEmbd, kConvKernel});
            w.dw_ln_w = f32(p + "conv.dw_ln.weight", {kEmbd});
            w.dw_ln_b = f32(p + "conv.dw_ln.bias", {kEmbd});
            w.pw2 = bf16_matrix(p + "conv.pw2.weight", {kEmbd, kEmbd});
            w.final_ln_w = f32(p + "final_ln.weight", {kEmbd});
            w.final_ln_b = f32(p + "final_ln.bias", {kEmbd});
            layers_.push_back(std::move(w));
        }
        window_ = f64_table(source, "frontend.window", {kFrameLength});
        mel_filters_ = f64_table(source, "frontend.mel_filters", {kFft / 2 + 1, kMels});
        resample_kernel_ = source.require_f32("frontend.resample_44100", {160, 475});
        loaded += 3;
        if (source.tensors().size() != loaded) {
            throw std::runtime_error("zonos3: ssl-teacher GGUF: tensor count does not match the 16-layer teacher");
        }
        store_->upload();
        source_->release_storage();
    }

    // SeamlessM4T fbank: per frame in double, x*32768 minus its mean, backwards pre-emphasis, Povey window,
    // F64 radix-2 FFT rounded to complex64, mel power, log; then per-mel normalization and stride-2 stacking.
    void compute_features(const std::vector<float> & wav, std::vector<float> & feats, int & T, std::vector<int32_t> & mask) const {
        const int n = static_cast<int>(wav.size());
        const int nf = n < kFrameLength ? 0 : 1 + (n - kFrameLength) / kHop;
        if (nf < 2) throw std::runtime_error("zonos3: reference audio is too short for the SSL teacher");
        const int bins = kFft / 2 + 1;
        std::vector<float> fb(static_cast<size_t>(nf) * kMels);
        std::vector<float> mean(kMels, 0.0f);
        std::vector<float> var(kMels, 0.0f);
        std::vector<std::complex<double>> z(kFft);
        std::vector<double> powers(static_cast<size_t>(bins));
        for (int fr = 0; fr < nf; ++fr) {
            const float * frame = wav.data() + static_cast<size_t>(fr) * kHop;
            double avg = 0;
            for (int i = 0; i < kFrameLength; ++i) avg += static_cast<double>(frame[i]) * 32768.0;
            avg /= kFrameLength;
            for (int i = 0; i < kFrameLength; ++i) z[static_cast<size_t>(i)] = static_cast<double>(frame[i]) * 32768.0 - avg;
            for (int i = kFrameLength - 1; i > 0; --i) z[static_cast<size_t>(i)] -= 0.97 * z[static_cast<size_t>(i - 1)];
            z[0] *= 1.0 - 0.97;
            for (int i = 0; i < kFrameLength; ++i) z[static_cast<size_t>(i)] *= window_[static_cast<size_t>(i)];
            for (int i = kFrameLength; i < kFft; ++i) z[static_cast<size_t>(i)] = 0;
            fft512(z);
            for (int k = 0; k < bins; ++k) {
                const double re = static_cast<float>(z[static_cast<size_t>(k)].real());
                const double im = static_cast<float>(z[static_cast<size_t>(k)].imag());
                powers[static_cast<size_t>(k)] = re * re + im * im;
            }
            for (int mel = 0; mel < kMels; ++mel) {
                double power = 0.0;
                for (int k = 0; k < bins; ++k) power += powers[static_cast<size_t>(k)] * mel_filters_[static_cast<size_t>(k) * kMels + mel];
                fb[static_cast<size_t>(fr) * kMels + mel] = static_cast<float>(std::log(std::max(power, static_cast<double>(kMelFloor))));
            }
        }
        for (int j = 0; j < kMels; ++j) {
            mean[static_cast<size_t>(j)] = pairwise_sum<false>(fb.data() + j, nf, kMels) / static_cast<float>(nf);
            var[static_cast<size_t>(j)] = std::sqrt(pairwise_sum<true>(fb.data() + j, nf, kMels, mean[static_cast<size_t>(j)]) /
                                                        static_cast<float>(nf - 1) + 1e-7f);
        }
        T = (nf + kStride - 1) / kStride;
        feats.assign(static_cast<size_t>(T) * kFeature, 1.0f);
        mask.assign(static_cast<size_t>(T), 1);
        if (nf % kStride) mask.back() = 0;
        for (int i = 0; i < nf; ++i) {
            for (int j = 0; j < kMels; ++j) {
                const size_t at = static_cast<size_t>(i) * kMels + static_cast<size_t>(j);
                feats[at] = (fb[at] - mean[static_cast<size_t>(j)]) / var[static_cast<size_t>(j)];
            }
        }
    }

    ggml_tensor * layer_norm(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & w, const TensorValue & b) const {
        modules::NormConfig config;
        config.hidden_size = x->ne[0];
        config.eps = kEps;
        return modules::LayerNormModule(config).build(ctx, tv(x), {w, b}).tensor;
    }

    // Linear output rounds to BF16 after its bias (autocast F.linear).
    ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & w,
                         const std::optional<TensorValue> & b) const {
        modules::LinearConfig config;
        config.in_features = w.shape.dims[1];
        config.out_features = w.shape.dims[0];
        config.use_bias = b.has_value();
        config.precision = GGML_PREC_F32;
        return bf16(ctx, modules::LinearModule(config).build(ctx, tv(x), {w, b}).tensor);
    }

    ggml_tensor * ffn(core::ModuleBuildContext & ctx, ggml_tensor * x, const LayerWeights & w, size_t j) const {
        auto * norm = layer_norm(ctx, x, w.ffn_ln_w[j], w.ffn_ln_b[j]);
        auto * up = linear(ctx, norm, w.ffn_up[j], w.ffn_up_b[j]);
        auto * act = bf16(ctx, modules::SiluModule().build(ctx, tv(up)).tensor);
        return linear(ctx, act, w.ffn_down[j], w.ffn_down_b[j]);
    }

    void build_graph(int T) const {
        graph_.reset();
        frames_ = 0;
        const size_t metadata = ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead();
        graph_ = std::make_unique<core::PreparedGraph>(execution_, core::PreparedGraphConfig{
            "zonos3.ssl_teacher", metadata, kGraphNodes, 8 * ggml_tensor_overhead()});
        auto * pg = graph_->persistent_build_context().ggml;
        input_ = ggml_new_tensor_2d(pg, GGML_TYPE_F32, kFeature, T);
        valid_ = ggml_new_tensor_2d(pg, GGML_TYPE_F32, 1, T);
        mask_ = ggml_new_tensor_2d(pg, GGML_TYPE_F32, T, T);
        auto * ridx = ggml_new_tensor_1d(pg, GGML_TYPE_I32, 2 * T - 1);
        for (auto * t : {input_, valid_, mask_, ridx}) ggml_set_input(t);
        graph_->allocate_persistent();
        std::vector<int32_t> idx(static_cast<size_t>(2 * T - 1));
        for (int j = 0; j < 2 * T - 1; ++j) idx[static_cast<size_t>(j)] = std::clamp(j - (T - 1), -kRelLeft, kRelRight) + kRelLeft;
        ggml_backend_tensor_set(ridx, idx.data(), 0, idx.size() * sizeof(int32_t));

        auto & ctx = graph_->build_context();
        auto * g = ctx.ggml;
        constexpr int D = kEmbd;
        constexpr int H = kHeads;
        constexpr int HD = D / H;
        constexpr int R = kRelLeft + kRelRight + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        auto * x = layer_norm(ctx, bf16(ctx, input_), fp_ln_w_, fp_ln_b_);
        x = ggml_mul(g, linear(ctx, x, fp_proj_, fp_proj_b_), valid_);
        for (int l = 0; l < kLayers; ++l) {
            const auto & w = layers_[static_cast<size_t>(l)];
            // Only layer 0's residual stream is BF16; final_layer_norm is F32, which promotes later residuals.
            auto residual = [&](ggml_tensor * sum) { return l == 0 ? bf16(ctx, sum) : sum; };
            x = residual(ggml_add(g, x, bf16(ctx, ggml_scale(g, ffn(ctx, x, w, 0), 0.5f))));

            auto * an = layer_norm(ctx, x, w.attn_ln_w, w.attn_ln_b);
            auto proj = [&](size_t i) { return ggml_reshape_3d(g, linear(ctx, an, w.attn[i], w.attn_b[i]), HD, H, T); };
            auto * q = proj(0);
            auto * k = proj(1);
            auto * v = proj(2);
            auto * qp = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));
            auto * kp = ggml_cont(g, ggml_permute(g, k, 0, 2, 1, 3));
            // matmul -> division and einsum -> division are separate BF16 results before the score sum.
            auto * score = bf16(ctx, ggml_scale(g, bf16(ctx, matmul_bf16(g, kp, qp)), scale));
            auto * rel = bf16(ctx, ggml_scale(g, bf16(ctx, matmul_bf16(g, w.dist.tensor, qp)), scale));
            auto * relT = ggml_cont(g, ggml_transpose(g, ggml_reshape_2d(g, rel, R, static_cast<int64_t>(T) * H)));
            auto * full = ggml_cont(g, ggml_transpose(g, ggml_get_rows(g, relT, ridx)));
            auto * bias = ggml_cont(g, ggml_view_3d(g, full, T, T, H,
                static_cast<size_t>(std::max(1, 2 * T - 2)) * sizeof(float),
                static_cast<size_t>(T) * static_cast<size_t>(2 * T - 1) * sizeof(float),
                static_cast<size_t>(T - 1) * sizeof(float)));
            score = bf16(ctx, ggml_add(g, score, bias));
            score = ggml_soft_max(g, bf16(ctx, ggml_add(g, score, mask_)));
            auto * vp = ggml_cont(g, ggml_permute(g, v, 1, 2, 0, 3));
            auto * attn = ggml_cont(g, ggml_permute(g, bf16(ctx, matmul_bf16(g, vp, score)), 0, 2, 1, 3));
            x = residual(ggml_add(g, x, linear(ctx, ggml_reshape_2d(g, attn, D, T), w.attn[3], w.attn_b[3])));

            auto * conv = ggml_mul(g, layer_norm(ctx, x, w.conv_ln_w, w.conv_ln_b), valid_);
            conv = linear(ctx, conv, w.pw1, std::nullopt);
            // nn.GLU computes the sigmoid and product in F32 and rounds only its output.
            auto rows = [&](ggml_tensor * t, int start, int count) {
                return ggml_cont(g, ggml_view_2d(g, t, count, t->ne[1], t->nb[1], static_cast<size_t>(start) * t->nb[0]));
            };
            auto * gate = modules::SigmoidModule().build(ctx, tv(rows(conv, D, D))).tensor;
            conv = bf16(ctx, ggml_mul(g, rows(conv, 0, D), gate));
            // Causal depthwise k31: left pad 30 on the contiguous time axis.
            auto * padded = ggml_pad_ext(g, ggml_cont(g, ggml_transpose(g, conv)), kConvKernel - 1, 0, 0, 0, 0, 0, 0, 0);
            const auto input = core::wrap_tensor(padded, core::TensorShape::from_dims({1, D, T + kConvKernel - 1}));
            const auto kernel = core::reshape_tensor(ctx, w.dw, core::TensorShape::from_dims({D, 1, kConvKernel}));
            conv = modules::DepthwiseConv1dModule({D, kConvKernel, 1, 0, 1, false})
                       .build(ctx, input, {kernel, std::nullopt}).tensor;
            conv = ggml_cont(g, ggml_transpose(g, ggml_reshape_2d(g, conv, T, D)));
            conv = layer_norm(ctx, bf16(ctx, conv), w.dw_ln_w, w.dw_ln_b);
            conv = linear(ctx, modules::SiluModule().build(ctx, tv(conv)).tensor, w.pw2, std::nullopt);
            x = residual(ggml_add(g, x, conv));

            x = residual(ggml_add(g, x, bf16(ctx, ggml_scale(g, ffn(ctx, x, w, 1), 0.5f))));
            x = layer_norm(ctx, x, w.final_ln_w, w.final_ln_b);
        }
        output_ = x;
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_->graph(), output_);
        graph_->allocate();
        frames_ = T;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    std::unique_ptr<core::BackendWeightStore> store_;
    TensorValue fp_ln_w_, fp_ln_b_, fp_proj_, fp_proj_b_;
    std::vector<LayerWeights> layers_;
    std::vector<double> window_, mel_filters_;
    std::vector<float> resample_kernel_;

    mutable std::unique_ptr<core::PreparedGraph> graph_;
    mutable int frames_ = 0;
    mutable ggml_tensor * input_ = nullptr;
    mutable ggml_tensor * valid_ = nullptr;
    mutable ggml_tensor * mask_ = nullptr;
    mutable ggml_tensor * output_ = nullptr;
};

Zonos3SslTeacher::Zonos3SslTeacher(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), execution)) {}

Zonos3SslTeacher::~Zonos3SslTeacher() = default;

std::vector<float> Zonos3SslTeacher::encode(const std::vector<float> & mono_44k, int & frames) const {
    return impl_->encode(mono_44k, frames);
}

}  // namespace engine::models::zonos3
