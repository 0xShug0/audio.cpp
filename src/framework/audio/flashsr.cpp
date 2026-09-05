#include "engine/framework/audio/flashsr.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::audio {
namespace {

constexpr int kFlashSrOutputSampleRate = 48000;
constexpr int kFlashSrChannels = 32;
constexpr int kFlashSrActivationKernel = 12;
constexpr int kFlashSrActivationRatio = 2;
constexpr float kFlashSrOutputScale = 0.9990000128746033f;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct BackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

std::string shape_string(const engine::assets::TensorDataF32 & tensor) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tensor.shape.rank; ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << tensor.shape.dims[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<float> require_channel_param(
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    const std::string & name) {
    const auto tensor = source->require_f32_tensor(name);
    if (tensor.shape.rank != 3 || tensor.shape.dims[0] != 1 ||
        tensor.shape.dims[1] != kFlashSrChannels || tensor.shape.dims[2] != 1) {
        throw std::runtime_error("FlashSR tensor shape mismatch for " + name + ": " + shape_string(tensor));
    }
    return tensor.values;
}

std::string conv_weight_name(const std::string & block_id, int group, int index) {
    return "resblocks." + block_id + ".convs" + std::to_string(group) + "." +
           std::to_string(index) + ".weight";
}

std::string conv_bias_name(const std::string & block_id, int group, int index) {
    return "resblocks." + block_id + ".convs" + std::to_string(group) + "." +
           std::to_string(index) + ".bias";
}

struct Conv1dLayer {
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel = 0;
    modules::Conv1dWeights conv;
};

struct ActivationParams {
    core::TensorValue alpha;
    core::TensorValue inv_beta;
};

struct ResBlockWeights {
    Conv1dLayer convs1[3];
    Conv1dLayer convs2[3];
    ActivationParams activations[6];
};

Conv1dLayer load_conv(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & weight_name,
    const std::string & bias_name,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel) {
    Conv1dLayer layer;
    layer.out_channels = out_channels;
    layer.in_channels = in_channels;
    layer.kernel = kernel;
    layer.conv.weight = store.load_f32_tensor(source, weight_name, {out_channels, in_channels, kernel});
    if (!bias_name.empty()) {
        layer.conv.bias = store.load_f32_tensor(source, bias_name, {out_channels});
    }
    return layer;
}

ActivationParams load_activation(
    core::BackendWeightStore & store,
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    const std::string & prefix) {
    return {
        store.make_f32(core::TensorShape::from_dims({kFlashSrChannels}), require_channel_param(source, prefix + ".alpha")),
        store.make_f32(core::TensorShape::from_dims({kFlashSrChannels}), require_channel_param(source, prefix + ".inv_beta")),
    };
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    const std::string & block_id,
    int64_t kernel) {
    ResBlockWeights block;
    for (int i = 0; i < 3; ++i) {
        block.convs1[i] = load_conv(
            store,
            *source,
            conv_weight_name(block_id, 1, i),
            conv_bias_name(block_id, 1, i),
            kFlashSrChannels,
            kFlashSrChannels,
            kernel);
        block.convs2[i] = load_conv(
            store,
            *source,
            conv_weight_name(block_id, 2, i),
            conv_bias_name(block_id, 2, i),
            kFlashSrChannels,
            kFlashSrChannels,
            kernel);
    }
    for (int i = 0; i < 6; ++i) {
        block.activations[i] = load_activation(store, source, "resblocks." + block_id + ".activations." + std::to_string(i));
    }
    return block;
}

std::vector<float> make_diagonal_filter_weights(const std::vector<float> & filter, bool upsample) {
    if (filter.size() != kFlashSrActivationKernel) {
        throw std::runtime_error("FlashSR lowpass filter shape mismatch");
    }
    std::vector<float> weights(static_cast<size_t>(kFlashSrChannels * kFlashSrChannels * kFlashSrActivationKernel), 0.0f);
    for (int64_t c = 0; c < kFlashSrChannels; ++c) {
        for (int64_t k = 0; k < kFlashSrActivationKernel; ++k) {
            const size_t index = static_cast<size_t>((c * kFlashSrChannels + c) * kFlashSrActivationKernel + k);
            const int64_t source_k = upsample ? (kFlashSrActivationKernel - 1 - k) : k;
            weights[index] = filter[static_cast<size_t>(source_k)] * (upsample ? static_cast<float>(kFlashSrActivationRatio) : 1.0f);
        }
    }
    return weights;
}

}  // namespace

struct FlashSrWeights {
    std::unique_ptr<ggml_backend, BackendDeleter> backend;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::shared_ptr<core::BackendWeightStore> store;
    Conv1dLayer conv_pre;
    Conv1dLayer conv_post;
    ResBlockWeights resblock2;
    ResBlockWeights resblock0;
    ActivationParams activation_post;
    modules::Conv1dWeights downsample_filter;
    modules::ConvTranspose1dWeights upsample_filter;
};

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv1dLayer & layer,
    int64_t padding,
    int64_t dilation) {
    return modules::Conv1dModule({
        layer.in_channels,
        layer.out_channels,
        layer.kernel,
        1,
        static_cast<int>(padding),
        static_cast<int>(dilation),
        true,
    }).build(ctx, input, layer.conv);
}

core::TensorValue replicate_pad1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left,
    int64_t right) {
    if (left < 0 || right < 0) {
        throw std::runtime_error("FlashSR replicate padding must be non-negative");
    }
    auto output = input;
    constexpr int axis = 2;
    if (left > 0) {
        auto edge = modules::SliceModule({axis, 0, 1}).build(ctx, input);
        auto pad_shape = edge.shape;
        pad_shape.dims[axis] = left;
        auto pad = modules::RepeatModule({pad_shape}).build(ctx, edge);
        output = modules::ConcatModule({axis}).build(ctx, pad, output);
    }
    if (right > 0) {
        auto edge = modules::SliceModule({axis, input.shape.dims[axis] - 1, 1}).build(ctx, input);
        auto pad_shape = edge.shape;
        pad_shape.dims[axis] = right;
        auto pad = modules::RepeatModule({pad_shape}).build(ctx, edge);
        output = modules::ConcatModule({axis}).build(ctx, output, pad);
    }
    return output;
}

core::TensorValue activation1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ActivationParams & params,
    const FlashSrWeights & weights) {
    constexpr int64_t pad = kFlashSrActivationKernel / kFlashSrActivationRatio - 1;
    constexpr int64_t crop_left = pad * kFlashSrActivationRatio +
                                  (kFlashSrActivationKernel - kFlashSrActivationRatio) / 2;
    constexpr int64_t crop_right = pad * kFlashSrActivationRatio +
                                   (kFlashSrActivationKernel - kFlashSrActivationRatio + 1) / 2;
    auto padded = replicate_pad1d(ctx, input, pad, pad);
    auto up = modules::ConvTranspose1dModule({
        kFlashSrChannels,
        kFlashSrChannels,
        kFlashSrActivationKernel,
        kFlashSrActivationRatio,
        0,
        1,
        false,
    }).build(ctx, padded, weights.upsample_filter);
    up = modules::SliceModule({2, crop_left, up.shape.dims[2] - crop_left - crop_right}).build(ctx, up);

    auto alpha = core::reshape_tensor(ctx, params.alpha, core::TensorShape::from_dims({1, kFlashSrChannels, 1}));
    alpha = core::wrap_tensor(ggml_repeat(ctx.ggml, alpha.tensor, up.tensor), up.shape, GGML_TYPE_F32);
    auto inv_beta = core::reshape_tensor(ctx, params.inv_beta, core::TensorShape::from_dims({1, kFlashSrChannels, 1}));
    inv_beta = core::wrap_tensor(ggml_repeat(ctx.ggml, inv_beta.tensor, up.tensor), up.shape, GGML_TYPE_F32);
    auto periodic = core::wrap_tensor(
        ggml_sqr(ctx.ggml, ggml_sin(ctx.ggml, ggml_mul(ctx.ggml, up.tensor, alpha.tensor))),
        up.shape,
        GGML_TYPE_F32);
    auto snake = core::wrap_tensor(
        ggml_add(ctx.ggml, up.tensor, ggml_mul(ctx.ggml, periodic.tensor, inv_beta.tensor)),
        up.shape,
        GGML_TYPE_F32);

    auto down_padded = replicate_pad1d(
        ctx,
        snake,
        kFlashSrActivationKernel / 2 - 1,
        kFlashSrActivationKernel / 2);
    return modules::Conv1dModule({
        kFlashSrChannels,
        kFlashSrChannels,
        kFlashSrActivationKernel,
        kFlashSrActivationRatio,
        0,
        1,
        false,
    }).build(ctx, down_padded, weights.downsample_filter);
}

core::TensorValue resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResBlockWeights & block,
    const FlashSrWeights & weights,
    int64_t kernel) {
    auto output = input;
    constexpr int kDilations[3] = {1, 3, 5};
    for (int i = 0; i < 3; ++i) {
        auto xt = activation1d(ctx, output, block.activations[i * 2], weights);
        xt = conv1d(ctx, xt, block.convs1[i], (kernel * kDilations[i] - kDilations[i]) / 2, kDilations[i]);
        xt = activation1d(ctx, xt, block.activations[i * 2 + 1], weights);
        xt = conv1d(ctx, xt, block.convs2[i], (kernel - 1) / 2, 1);
        output = modules::AddModule().build(ctx, output, xt);
    }
    return output;
}

std::vector<float> normalize_output(const std::vector<float> & input) {
    float max_abs = 0.0f;
    for (float value : input) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    if (max_abs <= 0.0f) {
        throw std::runtime_error("FlashSR output normalization has zero peak");
    }
    std::vector<float> output(input.size());
    const float scale = kFlashSrOutputScale / max_abs;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale;
    }
    return output;
}

// ---------------------------------------------------------------------------
// CPU fast path for the FlashSR U-Net (enabled by default; set
// AUDIOCPP_FLASHSR_DSP=0 to compare against the generic graph).
//
// The generic ggml graph expresses every convolution as im2col + mul_mat and
// every replicate padding as slice/repeat/concat, so a single utterance moves
// several GB of temporary data (profiled: ~2.4 s for a 3.2 s utterance on 32
// threads, of which ~1 s per resblock). This class re-implements the same f32
// math directly: zero-padded direct convolutions (no im2col), per-channel
// fused upsample+snake+downsample activation (no pad copies), and the exact
// ggml bilinear expression for the 3x interpolate. Numerically it matches the
// ggml path to ~1e-6 relative (identical f32 ops and sinf; only the
// convolution reduction order differs).

struct FlashSrDspWeights {
    std::vector<float> conv_pre_w;   // [32][7]
    std::vector<float> conv_pre_b;   // [32]
    std::vector<float> conv_post_w;  // [32][7]
    struct Block {
        int kernel = 0;
        std::vector<float> w1[3];    // [32][32][K]
        std::vector<float> b1[3];    // [32]
        std::vector<float> w2[3];
        std::vector<float> b2[3];
        std::vector<float> alpha[6];     // [32]
        std::vector<float> inv_beta[6];  // [32]
    };
    Block block2;
    Block block0;
    std::vector<float> alpha_post;    // [32]
    std::vector<float> inv_beta_post; // [32]
    std::vector<float> filter;        // [12] raw lowpass filter
    std::vector<float> upsample_w;    // [12] = 2 * filter[11-k]
    std::vector<float> downsample_w;  // [12] = filter[k]
};

inline std::vector<float> read_f32_tensor(const core::TensorValue & value) {
    std::vector<float> out(static_cast<size_t>(value.shape.num_elements()));
    ggml_backend_tensor_get(value.tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

inline void flashsr_dsp_load_block(
    FlashSrDspWeights::Block & dst,
    const ResBlockWeights & src,
    int kernel) {
    dst.kernel = kernel;
    for (int i = 0; i < 3; ++i) {
        dst.w1[i] = read_f32_tensor(src.convs1[i].conv.weight);
        dst.b1[i] = read_f32_tensor(*src.convs1[i].conv.bias);
        dst.w2[i] = read_f32_tensor(src.convs2[i].conv.weight);
        dst.b2[i] = read_f32_tensor(*src.convs2[i].conv.bias);
    }
    for (int i = 0; i < 6; ++i) {
        dst.alpha[i] = read_f32_tensor(src.activations[i].alpha);
        dst.inv_beta[i] = read_f32_tensor(src.activations[i].inv_beta);
    }
}

// Zero-padded direct 1D convolution over [C][n] channel-major buffers, in the
// standard torch/ggml convention for (odd) kernels with symmetric padding:
//   out[oc][p] = bias[oc] + sum_ic sum_k w[oc][ic][k] * in[ic][p + dilation*(k - c)]
// where c = (kernel - 1) / 2 and out-of-range input positions read zero. The
// `pad` argument must equal dilation * c (true for every FlashSR convolution).
inline void conv1d_direct(
    int64_t n,
    int kernel,
    int dilation,
    int64_t pad,
    const float * in,
    const float * w,
    const float * bias,
    float * out) {
    (void)pad;
    constexpr int C = kFlashSrChannels;
    constexpr int kMaxTaps = 11;
    const int c = (kernel - 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t p0 = 0; p0 < n; p0 += 256) {
        const int64_t p1 = std::min<int64_t>(p0 + 256, n);
        for (int64_t p = p0; p < p1; ++p) {
            int taps[kMaxTaps];
            int64_t src[kMaxTaps];
            int ns = 0;
            for (int k = 0; k < kernel; ++k) {
                const int64_t s = p + static_cast<int64_t>(dilation) * (k - c);
                if (s < 0 || s >= n) {
                    continue;
                }
                taps[ns] = k;
                src[ns] = s;
                ++ns;
            }
            float window[C][kMaxTaps];
            for (int ic = 0; ic < C; ++ic) {
                const float * row = in + static_cast<size_t>(ic) * n;
                for (int j = 0; j < ns; ++j) {
                    window[ic][j] = row[src[j]];
                }
            }
            for (int oc = 0; oc < C; ++oc) {
                float acc = bias != nullptr ? bias[oc] : 0.0f;
                for (int ic = 0; ic < C; ++ic) {
                    const float * wr = w + (static_cast<size_t>(oc) * C + ic) * kernel;
                    for (int j = 0; j < ns; ++j) {
                        acc += wr[taps[j]] * window[ic][j];
                    }
                }
                out[static_cast<size_t>(oc) * n + p] = acc;
            }
        }
    }
}

// Replicates modules::Interpolate1dModule(Linear) == ggml BILINEAR for the
// [1][C][n] -> [1][C][3n] case: same scale factors and same arithmetic order
// as ggml_compute_forward_upscale_f32, so results match the ggml path bit-for-bit.
inline void interp_linear_3x(int64_t n, const float * in, float * out) {
    constexpr int C = kFlashSrChannels;
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < n * 3; ++i0) {
        const float x = (static_cast<float>(i0) + 0.5f) / 3.0f - 0.5f;
        int64_t x0 = static_cast<int64_t>(floorf(x));
        int64_t x1 = x0 + 1;
        x0 = std::max<int64_t>(0, std::min(x0, n - 1));
        x1 = std::max<int64_t>(0, std::min(x1, n - 1));
        float dx = x - static_cast<float>(x0);
        dx = std::max(0.0f, std::min(dx, 1.0f));
        const float dy = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float * row = in + static_cast<size_t>(c) * n;
            const float a = row[x0];
            const float b = row[x1];
            const float cc = row[x0];
            const float d = row[x1];
            out[static_cast<size_t>(c) * n * 3 + i0] =
                a * (1.0f - dx) * (1.0f - dy) + b * dx * (1.0f - dy) + cc * (1.0f - dx) * dy + d * dx * dy;
        }
    }
}

// Fused per-channel activation1d: replicate-pad(5,5) -> convT(k=12, s=2, x2,
// diagonal) -> crop[15:-15] -> snake(x + sin^2(x*alpha)*inv_beta)
// -> replicate-pad(5,6) -> conv1d(k=12, s=2, diagonal).
// Buffers: in/out [C][n], up [C][2n+30], snake [C][2n].
inline void activation1d_dsp(
    int64_t n,
    const float * in,
    const float * alpha,
    const float * inv_beta,
    const float * upsample_w,
    const float * downsample_w,
    float * up,
    float * snake,
    float * out) {
    // Full convT output length before the [15:-15] crop: 2*(n + 10) + (12 - 2) = 2n + 30.
    const int64_t lu = 2 * n + 30;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < kFlashSrChannels; ++c) {
        const float * x = in + static_cast<size_t>(c) * n;
        const int64_t lp = n + kFlashSrActivationKernel - 2;  // replicate pad 5/5
        float * up_c = up + static_cast<size_t>(c) * lu;
        for (int64_t j = 0; j < lu; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < kFlashSrActivationKernel; ++k) {
                const int64_t t = j - k;
                if (t < 0 || (t & 1) != 0) {
                    continue;
                }
                const int64_t idx = t / 2;
                if (idx >= lp) {
                    continue;
                }
                // idx is a position in the replicate-padded input (pad 5/5):
                // [0,5) -> x[0], [5, n+5) -> x[idx-5], [n+5, n+10) -> x[n-1].
                const int64_t sample = idx < 5 ? 0 : (idx >= n + 5 ? n - 1 : idx - 5);
                acc += upsample_w[k] * x[sample];
            }
            up_c[j] = acc;
        }
        const float * crop = up_c + 15;
        float * s = snake + static_cast<size_t>(c) * n * 2;
        const float a = alpha[c];
        const float ib = inv_beta[c];
        for (int64_t m = 0; m < n * 2; ++m) {
            const float u = crop[m];
            const float t1 = u * a;
            const float t2 = sinf(t1);
            const float t3 = t2 * t2;
            s[m] = u + t3 * ib;
        }
        for (int64_t p = 0; p < n; ++p) {
            float acc = 0.0f;
            for (int k = 0; k < kFlashSrActivationKernel; ++k) {
                int64_t idx = 2 * p + k - (kFlashSrActivationKernel / 2 - 1);
                idx = idx < 0 ? 0 : (idx >= n * 2 ? n * 2 - 1 : idx);
                acc += downsample_w[k] * s[idx];
            }
            out[static_cast<size_t>(c) * n + p] = acc;
        }
    }
}

inline void flashsr_dsp_resblock(
    int64_t n,
    const FlashSrDspWeights::Block & block,
    const float * x,
    const FlashSrDspWeights & weights,
    float * xt1,
    float * xt2,
    float * xt3,
    float * up_buf,
    float * snake_buf,
    float * out) {
    const int k = block.kernel;
    constexpr int kDilations[3] = {1, 3, 5};
    std::memcpy(out, x, static_cast<size_t>(kFlashSrChannels) * n * sizeof(float));
    for (int i = 0; i < 3; ++i) {
        activation1d_dsp(n, out, block.alpha[i * 2].data(), block.inv_beta[i * 2].data(),
                         weights.upsample_w.data(), weights.downsample_w.data(), up_buf, snake_buf, xt1);
        conv1d_direct(n, k, kDilations[i], (k * kDilations[i] - kDilations[i]) / 2,
                      xt1, block.w1[i].data(), block.b1[i].data(), xt2);
        activation1d_dsp(n, xt2, block.alpha[i * 2 + 1].data(), block.inv_beta[i * 2 + 1].data(),
                         weights.upsample_w.data(), weights.downsample_w.data(), up_buf, snake_buf, xt3);
        conv1d_direct(n, k, 1, (k - 1) / 2, xt3, block.w2[i].data(), block.b2[i].data(), xt1);
        for (size_t i2 = 0; i2 < static_cast<size_t>(kFlashSrChannels) * n; ++i2) {
            out[i2] += xt1[i2];
        }
    }
}

class FlashSrDsp {
public:
    explicit FlashSrDsp(const FlashSrWeights & weights) : weights_(make_weights(weights)) {
        omp_set_num_threads(std::max(1, weights.threads));
    }

    // waveform: [n] 16 kHz mono. Returns raw 48 kHz output (3n samples,
    // pre-normalization), matching FlashSrGraph::run.
    std::vector<float> run(const std::vector<float> & waveform) const {
        const int64_t n = static_cast<int64_t>(waveform.size());
        const int64_t l = n * 3;
        constexpr size_t C = kFlashSrChannels;
        std::vector<float> x_pre(C * static_cast<size_t>(n));
        std::vector<float> x3(C * static_cast<size_t>(l));
        std::vector<float> xs(C * static_cast<size_t>(l));
        std::vector<float> xs0(C * static_cast<size_t>(l));
        std::vector<float> z(C * static_cast<size_t>(l));
        std::vector<float> xt1(C * static_cast<size_t>(l));
        std::vector<float> xt2(C * static_cast<size_t>(l));
        std::vector<float> xt3(C * static_cast<size_t>(l));
        std::vector<float> up_buf(C * static_cast<size_t>(2 * l + 30));
        std::vector<float> snake_buf(C * static_cast<size_t>(2 * l));
        std::vector<float> za(C * static_cast<size_t>(l));
        std::vector<float> out(static_cast<size_t>(l));

        // conv_pre: [n] -> [C][n], kernel 7, zero pad 3, with bias.
        #pragma omp parallel for schedule(static)
        for (int64_t p = 0; p < n; ++p) {
            float win[7];
            for (int k = 0; k < 7; ++k) {
                const int64_t idx = p - 3 + k;
                win[k] = (idx >= 0 && idx < n) ? waveform[static_cast<size_t>(idx)] : 0.0f;
            }
            for (int c = 0; c < kFlashSrChannels; ++c) {
                float acc = weights_.conv_pre_b[static_cast<size_t>(c)];
                for (int k = 0; k < 7; ++k) {
                    acc += weights_.conv_pre_w[static_cast<size_t>(c) * 7 + k] * win[k];
                }
                x_pre[static_cast<size_t>(c) * n + p] = acc;
            }
        }
        interp_linear_3x(n, x_pre.data(), x3.data());

        flashsr_dsp_resblock(l, weights_.block2, x3.data(), weights_, xt1.data(), xt2.data(), xt3.data(),
                             up_buf.data(), snake_buf.data(), xs.data());
        flashsr_dsp_resblock(l, weights_.block0, x3.data(), weights_, xt1.data(), xt2.data(), xt3.data(),
                             up_buf.data(), snake_buf.data(), xs0.data());

        for (size_t i = 0; i < z.size(); ++i) {
            z[i] = (xs[i] + xs0[i]) * 0.5f;
        }
        activation1d_dsp(l, z.data(), weights_.alpha_post.data(), weights_.inv_beta_post.data(),
                         weights_.upsample_w.data(), weights_.downsample_w.data(), up_buf.data(), snake_buf.data(), za.data());

        // conv_post: [C][l] -> [l], kernel 7, zero pad 3, no bias, then tanh.
        #pragma omp parallel for schedule(static)
        for (int64_t p = 0; p < l; ++p) {
            float acc = 0.0f;
            for (int c = 0; c < kFlashSrChannels; ++c) {
                const float * row = za.data() + static_cast<size_t>(c) * l;
                for (int k = 0; k < 7; ++k) {
                    const int64_t idx = p - 3 + k;
                    if (idx >= 0 && idx < l) {
                        acc += weights_.conv_post_w[static_cast<size_t>(c) * 7 + k] * row[idx];
                    }
                }
            }
            out[static_cast<size_t>(p)] = tanhf(acc);
        }
        return out;
    }

private:
    static FlashSrDspWeights make_weights(const FlashSrWeights & weights) {
        FlashSrDspWeights w;
        w.conv_pre_w = read_f32_tensor(weights.conv_pre.conv.weight);
        w.conv_pre_b = read_f32_tensor(*weights.conv_pre.conv.bias);
        w.conv_post_w = read_f32_tensor(weights.conv_post.conv.weight);
        w.alpha_post = read_f32_tensor(weights.activation_post.alpha);
        w.inv_beta_post = read_f32_tensor(weights.activation_post.inv_beta);
        // Raw lowpass filter = diagonal of the downsample weight matrix.
        // The store layout is [ic][oc][k] (see make_diagonal_filter_weights), so the
        // channel-0 diagonal is simply the first kFlashSrActivationKernel values.
        auto down = read_f32_tensor(weights.downsample_filter.weight);
        w.filter.resize(kFlashSrActivationKernel);
        for (int k = 0; k < kFlashSrActivationKernel; ++k) {
            w.filter[static_cast<size_t>(k)] = down[static_cast<size_t>(k)];
        }
        w.upsample_w.resize(kFlashSrActivationKernel);
        w.downsample_w = w.filter;
        for (int k = 0; k < kFlashSrActivationKernel; ++k) {
            w.upsample_w[static_cast<size_t>(k)] =
                w.filter[static_cast<size_t>(kFlashSrActivationKernel - 1 - k)] * static_cast<float>(kFlashSrActivationRatio);
        }
        flashsr_dsp_load_block(w.block2, weights.resblock2, weights.resblock2.convs1[0].kernel);
        flashsr_dsp_load_block(w.block0, weights.resblock0, weights.resblock0.convs1[0].kernel);
        return w;
    }

    FlashSrDspWeights weights_;
};

class FlashSrGraph {
public:
    FlashSrGraph(const FlashSrWeights & weights, int64_t input_samples)
        : weights_(weights),
          input_samples_(input_samples) {
        if (input_samples_ <= 0) {
            throw std::runtime_error("FlashSR graph input length must be positive");
        }
        ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize FlashSR GGML context");
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "flashsr", weights.backend_type};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, input_samples_}));
        input_ = input.tensor;
        auto x = conv1d(build_ctx, input, weights.conv_pre, 3, 1);
        x = modules::Interpolate1dModule({input_samples_ * 3, modules::Interpolate1dMode::Linear}).build(build_ctx, x);
        auto xs = resblock(build_ctx, x, weights.resblock2, weights, 11);
        auto xs0 = resblock(build_ctx, x, weights.resblock0, weights, 3);
        xs = core::wrap_tensor(ggml_scale(build_ctx.ggml, modules::AddModule().build(build_ctx, xs, xs0).tensor, 0.5f), xs.shape, GGML_TYPE_F32);
        xs = activation1d(build_ctx, xs, weights.activation_post, weights);
        auto output = conv1d(build_ctx, xs, weights.conv_post, 3, 1);
        output = core::wrap_tensor(ggml_tanh(build_ctx.ggml, output.tensor), output.shape, GGML_TYPE_F32);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.backend.get()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate FlashSR GGML graph");
        }
        if (core::uses_host_graph_plan(weights.backend.get())) {
            plan_ = core::create_backend_graph_plan_if_host(weights.backend.get(), graph_);
            if (plan_ == nullptr) {
                throw std::runtime_error("failed to create FlashSR graph plan");
            }
        }
    }

    ~FlashSrGraph() {
        core::release_backend_graph_resources(weights_.backend.get(), graph_);
        if (plan_ != nullptr) {
            auto * backend = weights_.backend.get();
            core::free_backend_graph_plan(backend, plan_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t input_samples) const noexcept {
        return input_samples_ == input_samples;
    }

    std::vector<float> run(const std::vector<float> & waveform) {
        if (static_cast<int64_t>(waveform.size()) != input_samples_) {
            throw std::runtime_error("FlashSR input size changed without rebuilding graph");
        }
        ggml_backend_tensor_set(input_, waveform.data(), 0, waveform.size() * sizeof(float));
        core::set_backend_threads(weights_.backend.get(), weights_.threads);
        const auto status = core::compute_backend_graph(weights_.backend.get(), graph_, plan_, "FlashSR");
        ggml_backend_synchronize(weights_.backend.get());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FlashSR GGML graph compute failed");
        }
        std::vector<float> output(ggml_nelements(output_));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
    const FlashSrWeights & weights_;
    int64_t input_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

FlashSrModel::FlashSrModel() = default;
FlashSrModel::~FlashSrModel() = default;
FlashSrModel::FlashSrModel(FlashSrModel &&) noexcept = default;
FlashSrModel & FlashSrModel::operator=(FlashSrModel &&) noexcept = default;

FlashSrModel::FlashSrModel(std::shared_ptr<FlashSrWeights> weights) : weights_(std::move(weights)) {
    if (!weights_) {
        throw std::runtime_error("FlashSR weights are missing");
    }
}

FlashSrModel FlashSrModel::load_from_directory(const std::filesystem::path & model_dir) {
    return load_from_directory(model_dir, core::BackendConfig{});
}

FlashSrModel FlashSrModel::load_from_directory(
    const std::filesystem::path & model_dir,
    const core::BackendConfig & backend_config) {
    return load_from_tensor_source(
        engine::assets::open_tensor_source(model_dir / "flashsr.safetensors"),
        backend_config);
}

FlashSrModel FlashSrModel::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    const core::BackendConfig & backend_config) {
    if (!source) {
        throw std::runtime_error("FlashSR tensor source is missing");
    }
    auto weights = std::make_shared<FlashSrWeights>();
    weights->backend.reset(core::init_backend(backend_config));
    weights->backend_type = core::backend_type(weights->backend.get());
    weights->threads = std::max(1, backend_config.threads);
    weights->store = std::make_shared<core::BackendWeightStore>(weights->backend.get(), weights->backend_type, "FlashSR", 32ull * 1024ull * 1024ull);
    auto & store = *weights->store;
    weights->conv_pre = load_conv(store, *source, "conv_pre.weight", "conv_pre.bias", kFlashSrChannels, 1, 7);
    weights->conv_post = load_conv(store, *source, "conv_post.weight", "", 1, kFlashSrChannels, 7);
    weights->resblock2 = load_resblock(store, source, "2", 11);
    weights->resblock0 = load_resblock(store, source, "0", 3);
    weights->activation_post = load_activation(store, source, "activation_post");
    const auto filter_tensor = source->require_f32_tensor("activation_filter");
    if (filter_tensor.shape.rank != 3 || filter_tensor.shape.dims[0] != 1 ||
        filter_tensor.shape.dims[1] != 1 || filter_tensor.shape.dims[2] != kFlashSrActivationKernel) {
        throw std::runtime_error("FlashSR lowpass filter shape mismatch: " + shape_string(filter_tensor));
    }
    weights->downsample_filter.weight = store.make_f32(
        core::TensorShape::from_dims({kFlashSrChannels, kFlashSrChannels, kFlashSrActivationKernel}),
        make_diagonal_filter_weights(filter_tensor.values, false));
    weights->upsample_filter.weight = store.make_f32(
        core::TensorShape::from_dims({kFlashSrChannels, kFlashSrChannels, kFlashSrActivationKernel}),
        make_diagonal_filter_weights(filter_tensor.values, true));
    store.upload();
    return FlashSrModel(std::move(weights));
}

// The direct DSP path avoids the large im2col temporaries used by the generic
// graph. It is CPU-only and enabled by default; set AUDIOCPP_FLASHSR_DSP=0 to
// retain the generic graph for comparison or diagnostics.
bool flashsr_dsp_enabled(const FlashSrWeights & weights) {
    if (weights.backend_type != core::BackendType::Cpu) {
        return false;
    }
    const char * value = std::getenv("AUDIOCPP_FLASHSR_DSP");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

FlashSrOutput FlashSrModel::super_resolve_mono_16k(const std::vector<float> & waveform) const {
    if (!weights_) {
        throw std::runtime_error("FlashSR model is not loaded");
    }
    if (waveform.empty()) {
        throw std::runtime_error("FlashSR input waveform is empty");
    }
    const bool use_dsp = flashsr_dsp_enabled(*weights_);
    constexpr int64_t segment_samples = 16000 * 4;
    constexpr int64_t stride_samples = 16000 * 3;
    constexpr int64_t segment_threshold = segment_samples * 2;
    constexpr int64_t output_ratio = 3;
    const int64_t original_samples = static_cast<int64_t>(waveform.size());
    if (original_samples <= segment_threshold) {
        if (use_dsp) {
            if (!dsp_) {
                dsp_ = std::make_unique<FlashSrDsp>(*weights_);
            }
            return FlashSrOutput{
                kFlashSrOutputSampleRate,
                normalize_output(dsp_->run(waveform))};
        }
        if (!graph_ || !graph_->matches(original_samples)) {
            graph_ = std::make_unique<FlashSrGraph>(*weights_, original_samples);
        }
        return FlashSrOutput{kFlashSrOutputSampleRate, normalize_output(graph_->run(waveform))};
    }

    std::vector<float> padded = waveform;
    const int64_t remainder = (original_samples - segment_samples) % stride_samples;
    if (remainder != 0) {
        padded.insert(padded.end(), static_cast<size_t>(stride_samples - remainder), 0.0f);
    }
    const int64_t padded_samples = static_cast<int64_t>(padded.size());
    if (use_dsp) {
        if (!dsp_) {
            dsp_ = std::make_unique<FlashSrDsp>(*weights_);
        }
    } else if (!graph_ || !graph_->matches(segment_samples)) {
        graph_ = std::make_unique<FlashSrGraph>(*weights_, segment_samples);
    }
    std::vector<float> output(static_cast<size_t>(padded_samples * output_ratio), 0.0f);
    std::vector<float> weights(static_cast<size_t>(padded_samples * output_ratio), 0.0f);
    const int64_t segment_output_samples = segment_samples * output_ratio;
    const int64_t stride_output_samples = stride_samples * output_ratio;
    const int64_t overlap_output_samples = segment_output_samples - stride_output_samples;
    for (int64_t current = 0; current + segment_samples <= padded_samples; current += stride_samples) {
        std::vector<float> segment(
            padded.begin() + static_cast<std::ptrdiff_t>(current),
            padded.begin() + static_cast<std::ptrdiff_t>(current + segment_samples));
        const auto segment_output = use_dsp ? dsp_->run(segment) : graph_->run(segment);
        if (static_cast<int64_t>(segment_output.size()) != segment_output_samples) {
            throw std::runtime_error("FlashSR segmented output length mismatch");
        }
        const int64_t output_offset = current * output_ratio;
        for (int64_t i = 0; i < segment_output_samples; ++i) {
            float weight = 1.0f;
            if (current > 0 && i < overlap_output_samples) {
                weight = static_cast<float>(i + 1) / static_cast<float>(overlap_output_samples + 1);
            }
            if (current + segment_samples < padded_samples && i >= stride_output_samples) {
                weight = static_cast<float>(segment_output_samples - i) / static_cast<float>(overlap_output_samples + 1);
            }
            const size_t out_index = static_cast<size_t>(output_offset + i);
            output[out_index] += segment_output[static_cast<size_t>(i)] * weight;
            weights[out_index] += weight;
        }
    }
    output.resize(static_cast<size_t>(original_samples * output_ratio));
    weights.resize(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        if (weights[i] <= 0.0f) {
            throw std::runtime_error("FlashSR segmented synthesis produced an uncovered sample");
        }
        output[i] /= weights[i];
    }
    return FlashSrOutput{kFlashSrOutputSampleRate, normalize_output(output)};
}

}  // namespace engine::audio
