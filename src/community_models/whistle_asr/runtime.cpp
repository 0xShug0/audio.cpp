#include "engine/community_models/whistle_asr/runtime.h"

#include "engine/community_models/whistle_asr/frontend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine::community_models::whistle_asr {

class WhistleWeights {
public:
    explicit WhistleWeights(const assets::TensorSource & source) {
        for (const auto & metadata : source.tensors()) {
            if (metadata.dtype != "f32" && metadata.dtype != "F32" &&
                metadata.dtype != "float32") {
                throw std::runtime_error("Whistle currently requires FP32 checkpoint tensors");
            }
            auto values = source.require_f32(metadata.name, metadata.shape);
            tensors_.emplace(metadata.name, Tensor{metadata.shape, std::move(values)});
        }
    }

    struct View {
        const float * values;
        std::vector<int64_t> shape;
    };

    [[nodiscard]] View get(const std::string & name, int layer = -1) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Missing Whistle tensor: " + name);
        }
        const auto & tensor = it->second;
        if (layer < 0) {
            return {tensor.values.data(), tensor.shape};
        }
        if (tensor.shape.empty() || tensor.shape.front() != 8 || layer >= 8) {
            throw std::runtime_error("Invalid Whistle layer tensor: " + name);
        }
        const size_t stride = tensor.values.size() / 8;
        return {tensor.values.data() + layer * stride,
            {tensor.shape.begin() + 1, tensor.shape.end()}};
    }

private:
    struct Tensor {
        std::vector<int64_t> shape;
        std::vector<float> values;
    };
    std::unordered_map<std::string, Tensor> tensors_;
};

namespace {

constexpr size_t kDimension = 512;
constexpr size_t kLanes = 4;
constexpr int kLayers = 8;
constexpr int kTextVocabulary = 8192;
constexpr size_t kMaximumTokens = 320;
constexpr std::array<const char *, 7> kLanguages = {"en", "de", "fr", "es", "it", "nl", "pl"};
using View = WhistleWeights::View;
using Rows = std::vector<float>;

#ifdef _OPENMP
struct ScopedThreadCount {
    explicit ScopedThreadCount(int threads)
        : previous(omp_get_max_threads()) { omp_set_num_threads(threads); }
    ~ScopedThreadCount() { omp_set_num_threads(previous); }
    int previous;
};
#endif

View require(const WhistleWeights & weights, const std::string & name,
             std::initializer_list<int64_t> shape, int layer = -1) {
    auto result = weights.get(name, layer);
    if (result.shape != std::vector<int64_t>(shape)) {
        throw std::runtime_error("Whistle tensor has unexpected shape: " + name);
    }
    return result;
}

float sigmoid(float value) {
    return value >= 0
        ? 1.0f / (1.0f + std::exp(-value))
        : std::exp(value) / (1.0f + std::exp(value));
}

void softmax(float * values, size_t count) {
    const float peak = *std::max_element(values, values + count);
    float sum = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        values[index] = std::exp(values[index] - peak);
        sum += values[index];
    }
    for (size_t index = 0; index < count; ++index) {
        values[index] /= sum;
    }
}

Rows linear(const Rows & input, size_t rows, size_t width, View weight) {
    if (weight.shape.size() != 2 || weight.shape[0] != static_cast<int64_t>(width) ||
        input.size() != rows * width) {
        throw std::runtime_error("Whistle linear projection shape mismatch");
    }
    const size_t output_width = static_cast<size_t>(weight.shape[1]);
    Rows output(rows * output_width, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for if(rows > 8)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        const float * in = input.data() + static_cast<size_t>(row) * width;
        float * out = output.data() + static_cast<size_t>(row) * output_width;
        for (size_t col = 0; col < width; ++col) {
            const float scale = in[col];
            const float * weights = weight.values + col * output_width;
            for (size_t output_col = 0; output_col < output_width; ++output_col) {
                out[output_col] += scale * weights[output_col];
            }
        }
    }
    return output;
}

void normalize(Rows & values, size_t rows, size_t width, const float * scale = nullptr) {
    if (values.size() != rows * width) {
        throw std::runtime_error("Whistle normalization shape mismatch");
    }
#ifdef _OPENMP
#pragma omp parallel for if(rows > 8)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        float * data = values.data() + static_cast<size_t>(row) * width;
        double energy = 0.0;
        for (size_t col = 0; col < width; ++col) {
            energy += static_cast<double>(data[col]) * data[col];
        }
        const float norm = static_cast<float>(1.0 / std::sqrt(energy / width + 1.0e-6));
        for (size_t col = 0; col < width; ++col) {
            data[col] *= norm * (scale == nullptr ? 1.0f : 1.0f + scale[col]);
        }
    }
}

void add(Rows & destination, const Rows & source, float factor = 1.0f) {
    if (destination.size() != source.size()) {
        throw std::runtime_error("Whistle residual shape mismatch");
    }
    for (size_t index = 0; index < destination.size(); ++index) {
        destination[index] += factor * source[index];
    }
}

void kronecker(const float * input, const float * left, const float * right, float * output) {
    std::array<float, kDimension> intermediate{};
    for (size_t source_row = 0; source_row < 16; ++source_row) {
        for (size_t dest_row = 0; dest_row < 16; ++dest_row) {
            const float scale = left[source_row * 16 + dest_row];
            for (size_t column = 0; column < 32; ++column) {
                intermediate[dest_row * 32 + column] +=
                    scale * input[source_row * 32 + column];
            }
        }
    }
    std::fill(output, output + kDimension, 0.0f);
    for (size_t row = 0; row < 16; ++row) {
        for (size_t source_col = 0; source_col < 32; ++source_col) {
            const float scale = intermediate[row * 32 + source_col];
            for (size_t dest_col = 0; dest_col < 32; ++dest_col) {
                output[row * 32 + dest_col] +=
                    scale * right[source_col * 32 + dest_col];
            }
        }
    }
}

Rows hadamard(const Rows & input, size_t frames, const WhistleWeights & weights,
              const WhistleAssets & assets, const std::string & prefix, int layer) {
    const auto d1 = require(weights, prefix + "d1", {512}, layer);
    const auto d2 = require(weights, prefix + "d2", {512}, layer);
    const auto d3 = require(weights, prefix + "d3", {512}, layer);
    const auto d4 = require(weights, prefix + "d4", {512}, layer);
    const auto b2 = require(weights, prefix + "b2", {512}, layer);
    const auto cond_v = require(weights, prefix + "cond_v", {512, 8}, layer);
    const auto cond_u = require(weights, prefix + "cond_u", {8, 512}, layer);
    const auto w1a = require(weights, prefix + "w1a", {16, 16}, layer);
    const auto w1b = require(weights, prefix + "w1b", {32, 32}, layer);
    const auto w2a = require(weights, prefix + "w2a", {16, 16}, layer);
    const auto w2b = require(weights, prefix + "w2b", {32, 32}, layer);
    const auto w3a = require(weights, prefix + "w3a", {16, 16}, layer);
    const auto w3b = require(weights, prefix + "w3b", {32, 32}, layer);
    Rows result(input.size());
#ifdef _OPENMP
#pragma omp parallel for if(frames > 8)
#endif
    for (int64_t frame = 0; frame < static_cast<int64_t>(frames); ++frame) {
        const float * x = input.data() + static_cast<size_t>(frame) * kDimension;
        std::array<float, 8> cond_weights{};
        for (size_t index = 0; index < kDimension; ++index) {
            for (size_t hidden = 0; hidden < 8; ++hidden) {
                cond_weights[hidden] += x[index] * cond_v.values[index * 8 + hidden];
            }
        }
        softmax(cond_weights.data(), cond_weights.size());
        std::array<float, kDimension> z{};
        std::array<float, kDimension> next{};
        for (size_t index = 0; index < kDimension; ++index) {
            z[index] = x[index] * d1.values[index];
        }
        kronecker(z.data(), w1a.values, w1b.values, next.data());
        for (size_t index = 0; index < kDimension; ++index) {
            float condition = 1.0f;
            for (size_t hidden = 0; hidden < 8; ++hidden) {
                condition += cond_weights[hidden] * cond_u.values[hidden * kDimension + index];
            }
            const float value = next[assets.hadamard_permutations[0][index]];
            z[index] = d2.values[index] * condition * value + b2.values[index];
            z[index] *= sigmoid(z[index]);
        }
        kronecker(z.data(), w2a.values, w2b.values, next.data());
        for (size_t index = 0; index < kDimension; ++index) {
            z[index] = d3.values[index] * next[assets.hadamard_permutations[1][index]];
        }
        kronecker(z.data(), w3a.values, w3b.values,
            result.data() + static_cast<size_t>(frame) * kDimension);
        for (size_t index = 0; index < kDimension; ++index) {
            result[static_cast<size_t>(frame) * kDimension + index] *= d4.values[index];
        }
    }
    return result;
}

void sinkhorn(std::array<float, 16> & logits) {
    for (int iteration = 0; iteration < 20; ++iteration) {
        for (size_t row = 0; row < 4; ++row) {
            const float peak = *std::max_element(logits.begin() + row * 4, logits.begin() + row * 4 + 4);
            float sum = 0.0f;
            for (size_t col = 0; col < 4; ++col) {
                sum += std::exp(logits[row * 4 + col] - peak);
            }
            const float log_sum = peak + std::log(sum);
            for (size_t col = 0; col < 4; ++col) {
                logits[row * 4 + col] -= log_sum;
            }
        }
        for (size_t col = 0; col < 4; ++col) {
            float peak = -std::numeric_limits<float>::infinity();
            for (size_t row = 0; row < 4; ++row) {
                peak = std::max(peak, logits[row * 4 + col]);
            }
            float sum = 0.0f;
            for (size_t row = 0; row < 4; ++row) {
                sum += std::exp(logits[row * 4 + col] - peak);
            }
            const float log_sum = peak + std::log(sum);
            for (size_t row = 0; row < 4; ++row) {
                logits[row * 4 + col] -= log_sum;
            }
        }
    }
    for (auto & value : logits) {
        value = std::exp(value);
    }
}

template<typename Block>
Rows mhc(const Rows & state, size_t frames, const WhistleWeights & weights,
         const std::string & prefix, int layer, Block block) {
    const auto a_pre = require(weights, prefix + "a_pre", {}, layer).values[0];
    const auto a_post = require(weights, prefix + "a_post", {}, layer).values[0];
    const auto a_res = require(weights, prefix + "a_res", {}, layer).values[0];
    const auto b_pre = require(weights, prefix + "b_pre", {4}, layer);
    const auto b_post = require(weights, prefix + "b_post", {4}, layer);
    const auto b_res = require(weights, prefix + "b_res", {4, 4}, layer);
    const auto phi_pre = require(weights, prefix + "phi_pre", {2048, 4}, layer);
    const auto phi_post = require(weights, prefix + "phi_post", {2048, 4}, layer);
    const auto phi_res = require(weights, prefix + "phi_res", {2048, 16}, layer);
    if (state.size() != frames * kLanes * kDimension) {
        throw std::runtime_error("Whistle mHC state has the wrong shape");
    }
    Rows mixed(frames * kDimension, 0.0f);
    Rows hposts(frames * kLanes);
    Rows hres(frames * kLanes * kLanes);
#ifdef _OPENMP
#pragma omp parallel for if(frames > 8)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(frames); ++row) {
        const float * st = state.data() + static_cast<size_t>(row) * kLanes * kDimension;
        std::array<float, kLanes * kDimension> nx{};
        std::copy_n(st, nx.size(), nx.begin());
        double energy = 0.0;
        for (float value : nx) {
            energy += static_cast<double>(value) * value;
        }
        const float scale = static_cast<float>(1.0 / std::sqrt(energy / nx.size() + 1.0e-6));
        for (auto & value : nx) {
            value *= scale;
        }
        std::array<float, 4> pre{};
        std::array<float, 4> post{};
        std::array<float, 16> residual{};
        for (size_t index = 0; index < nx.size(); ++index) {
            for (size_t lane = 0; lane < 4; ++lane) {
                pre[lane] += nx[index] * phi_pre.values[index * 4 + lane];
                post[lane] += nx[index] * phi_post.values[index * 4 + lane];
            }
            for (size_t element = 0; element < 16; ++element) {
                residual[element] += nx[index] * phi_res.values[index * 16 + element];
            }
        }
        for (size_t lane = 0; lane < 4; ++lane) {
            const bool active = lane == static_cast<size_t>(layer % 4);
            const float hpre = sigmoid(a_pre * pre[lane] + b_pre.values[lane] + (active ? 4.0f : -4.0f));
            hposts[static_cast<size_t>(row) * 4 + lane] =
                2.0f * sigmoid(a_post * post[lane] + b_post.values[lane] + (active ? 0.0f : -4.0f));
            for (size_t col = 0; col < kDimension; ++col) {
                mixed[static_cast<size_t>(row) * kDimension + col] +=
                    hpre * st[lane * kDimension + col];
            }
        }
        for (size_t index = 0; index < 16; ++index) {
            residual[index] = a_res * residual[index] + b_res.values[index];
        }
        sinkhorn(residual);
        std::copy(residual.begin(), residual.end(), hres.begin() + static_cast<size_t>(row) * 16);
    }
    Rows delta = block(mixed);
    if (delta.size() != mixed.size()) {
        throw std::runtime_error("Whistle mHC block output has the wrong shape");
    }
    Rows next(state.size());
    for (size_t row = 0; row < frames; ++row) {
        for (size_t lane = 0; lane < 4; ++lane) {
            const float hpost = hposts[row * 4 + lane];
            for (size_t col = 0; col < kDimension; ++col) {
                float value = hpost * (delta[row * kDimension + col] - mixed[row * kDimension + col]);
                for (size_t source = 0; source < 4; ++source) {
                    value += hres[row * 16 + lane * 4 + source] *
                        state[row * 4 * kDimension + source * kDimension + col];
                }
                next[row * 4 * kDimension + lane * kDimension + col] = value;
            }
        }
    }
    return next;
}

struct Grid {
    size_t height;
    size_t width;
    size_t channels;
    Rows values;
};

Grid stem_conv(const Grid & input, View kernel, size_t output_channels, bool depthwise) {
    if (kernel.shape != std::vector<int64_t>({3, 3, 1, static_cast<int64_t>(output_channels)}) ||
        (!depthwise && input.channels != 1) || (depthwise && input.channels != output_channels)) {
        throw std::runtime_error("Whistle convolution shape mismatch");
    }
    Grid result{(input.height + 1) / 2, (input.width + 1) / 2,
        output_channels, Rows((input.height + 1) / 2 * ((input.width + 1) / 2) * output_channels)};
#ifdef _OPENMP
#pragma omp parallel for if(result.height > 8)
#endif
    for (int64_t row = 0; row < static_cast<int64_t>(result.height); ++row) {
        for (size_t col = 0; col < result.width; ++col) {
            for (size_t channel = 0; channel < output_channels; ++channel) {
                float sum = 0.0f;
                for (int dr = -1; dr <= 1; ++dr) {
                    const auto source_row = row * 2 + dr;
                    if (source_row < 0 || static_cast<size_t>(source_row) >= input.height) {
                        continue;
                    }
                    for (int dc = -1; dc <= 1; ++dc) {
                        const auto source_col = static_cast<int64_t>(col * 2) + dc;
                        if (source_col < 0 || static_cast<size_t>(source_col) >= input.width) {
                            continue;
                        }
                        sum += input.values[
                            (static_cast<size_t>(source_row) * input.width +
                            static_cast<size_t>(source_col)) * input.channels +
                            (depthwise ? channel : 0)] *
                            kernel.values[((dr + 1) * 3 + (dc + 1)) * output_channels + channel];
                    }
                }
                result.values[(static_cast<size_t>(row) * result.width + col) *
                    output_channels + channel] = sum;
            }
        }
    }
    return result;
}

Rows stem(const MelFeatures & mel, const WhistleWeights & weights) {
    if (mel.values.size() != mel.frames * 80) {
        throw std::runtime_error("Whistle mel features have the wrong shape");
    }
    Grid grid{mel.frames, 80, 1, mel.values};
    grid = stem_conv(grid, require(weights, "stem/w", {3, 3, 1, 128}), 128, false);
    for (float & value : grid.values) {
        value *= sigmoid(value);
    }
    grid = stem_conv(grid, require(weights, "stem/dw_1", {3, 3, 1, 128}), 128, true);
    grid.values = linear(grid.values, grid.height * grid.width, 128,
        require(weights, "stem/pw_1/kernel", {128, 128}));
    for (float & value : grid.values) {
        value *= sigmoid(value);
    }
    grid = stem_conv(grid, require(weights, "stem/dw_2", {3, 3, 1, 128}), 128, true);
    grid.values = linear(grid.values, grid.height * grid.width, 128,
        require(weights, "stem/pw_2/kernel", {128, 128}));
    for (float & value : grid.values) {
        value *= sigmoid(value);
    }
    if (grid.width != 10) {
        throw std::runtime_error("Whistle stem has an unexpected mel width");
    }
    Rows flattened(grid.height * 1280);
    // The trained stem flattens channels before mel-width bins, not the convolution layout.
    for (size_t frame = 0; frame < grid.height; ++frame) {
        for (size_t channel = 0; channel < 128; ++channel) {
            for (size_t column = 0; column < 10; ++column) {
                flattened[frame * 1280 + channel * 10 + column] =
                    grid.values[(frame * 10 + column) * 128 + channel];
            }
        }
    }
    return linear(flattened, grid.height, 1280,
        require(weights, "stem/out/kernel", {1280, 512}));
}

void apply_rope(Rows & values, size_t frames, size_t heads, size_t offset = 0) {
    const float theta = static_cast<float>(-std::log(100000.0) / 48.0);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t index = 0; index < heads; ++index) {
            float * head = values.data() + (frame * heads + index) * 48;
            for (size_t dim = 0; dim < 24; ++dim) {
                const float angle = static_cast<float>(frame + offset) * std::exp(2.0f * dim * theta);
                const float co = std::cos(angle);
                const float si = std::sin(angle);
                const float a = head[dim];
                const float b = head[dim + 24];
                head[dim] = a * co - b * si;
                head[dim + 24] = b * co + a * si;
            }
        }
    }
}

Rows encoder_attention(const Rows & input, size_t frames,
                       const WhistleWeights & weights, int layer) {
    const std::string prefix = "encoder/layers/block/self_attn/";
    Rows q = linear(input, frames, 512,
        require(weights, prefix + "q_proj/kernel", {512, 384}, layer));
    Rows k = linear(input, frames, 512,
        require(weights, prefix + "k_proj/kernel", {512, 96}, layer));
    Rows v = linear(input, frames, 512,
        require(weights, prefix + "v_proj/kernel", {512, 128}, layer));
    normalize(q, frames * 8, 48,
        require(weights, prefix + "q_norm/scale", {48}, layer).values);
    normalize(k, frames * 2, 48,
        require(weights, prefix + "k_norm/scale", {48}, layer).values);
    apply_rope(q, frames, 8);
    apply_rope(k, frames, 2);
    Rows context(frames * 512);
#ifdef _OPENMP
#pragma omp parallel for if(frames > 8)
#endif
    for (int64_t work = 0; work < static_cast<int64_t>(frames * 8); ++work) {
        const size_t time = static_cast<size_t>(work) / 8;
        const size_t head = static_cast<size_t>(work) % 8;
        const size_t kv = head / 4;
        std::vector<float> scores(frames);
        const float * query = q.data() + (time * 8 + head) * 48;
        for (size_t frame = 0; frame < frames; ++frame) {
            const float * key = k.data() + (frame * 2 + kv) * 48;
            float score = 0.0f;
            for (size_t col = 0; col < 48; ++col) {
                score += query[col] * key[col];
            }
            scores[frame] = score * (1.0f / std::sqrt(48.0f));
        }
        softmax(scores.data(), frames);
        float * out = context.data() + time * 512 + head * 64;
        for (size_t frame = 0; frame < frames; ++frame) {
            const float * value = v.data() + (frame * 2 + kv) * 64;
            for (size_t col = 0; col < 64; ++col) {
                out[col] += scores[frame] * value[col];
            }
        }
    }
    const Rows gates = linear(input, frames, 512,
        require(weights, prefix + "gate_proj/kernel", {512, 512}, layer));
    for (size_t index = 0; index < context.size(); ++index) {
        context[index] *= sigmoid(gates[index]);
    }
    return linear(context, frames, 512,
        require(weights, prefix + "out_proj/kernel", {512, 512}, layer));
}

Rows encoder_convolution(const Rows & input, size_t frames,
                         const WhistleWeights & weights, int layer) {
    const std::string prefix = "encoder/layers/block/";
    Rows c = input;
    normalize(c, frames, 512,
        require(weights, prefix + "conv_norm/scale", {512}, layer).values);
    c = linear(c, frames, 512,
        require(weights, prefix + "pw1/kernel", {512, 1024}, layer));
    Rows gated(frames * 512);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t col = 0; col < 512; ++col) {
            gated[frame * 512 + col] = c[frame * 1024 + col] *
                sigmoid(c[frame * 1024 + 512 + col]);
        }
    }
    const auto kernel = require(weights, prefix + "dw", {9, 1, 512}, layer);
    c.assign(frames * 512, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for if(frames > 8)
#endif
    for (int64_t frame = 0; frame < static_cast<int64_t>(frames); ++frame) {
        for (int tap = 0; tap < 9; ++tap) {
            const auto source = frame + tap - 4;
            if (source < 0 || static_cast<size_t>(source) >= frames) {
                continue;
            }
            for (size_t col = 0; col < 512; ++col) {
                c[static_cast<size_t>(frame) * 512 + col] +=
                    gated[static_cast<size_t>(source) * 512 + col] *
                    kernel.values[tap * 512 + col];
            }
        }
    }
    normalize(c, frames, 512,
        require(weights, prefix + "conv_out_norm/scale", {512}, layer).values);
    for (float & value : c) {
        value *= sigmoid(value);
    }
    return linear(c, frames, 512,
        require(weights, prefix + "pw2/kernel", {512, 512}, layer));
}

Rows encoder_block(const Rows & input, size_t frames,
                   const WhistleWeights & weights, const WhistleAssets & assets, int layer) {
    const std::string prefix = "encoder/layers/block/";
    Rows pre = input;
    normalize(pre, frames, 512,
        require(weights, prefix + "pre_hada_norm_0/scale", {512}, layer).values);
    Rows h = input;
    add(h, hadamard(pre, frames, weights, assets,
        prefix + "hadamard_mlp_0/", layer), 0.5f);
    Rows attention_input = h;
    normalize(attention_input, frames, 512,
        require(weights, prefix + "ZCRMSNorm_0/scale", {512}, layer).values);
    Rows attention = encoder_attention(attention_input, frames, weights, layer);
    normalize(attention, frames, 512,
        require(weights, prefix + "post_attn_norm/scale", {512}, layer).values);
    add(h, attention, sigmoid(require(weights, prefix + "attn_gate", {}, layer).values[0]));
    add(h, encoder_convolution(h, frames, weights, layer));
    pre = h;
    normalize(pre, frames, 512,
        require(weights, prefix + "pre_hada_norm/scale", {512}, layer).values);
    add(h, hadamard(pre, frames, weights, assets,
        prefix + "hadamard_mlp/", layer), 0.5f);
    return h;
}

struct EncoderOutput {
    size_t frames = 0;
    std::array<Rows, 8> cross_k;
    std::array<Rows, 8> cross_v;
};

EncoderOutput encode(const MelFeatures & mel, const WhistleWeights & weights,
                     const WhistleAssets & assets) {
    const Rows projected = stem(mel, weights);
    const size_t frames = projected.size() / 512;
    Rows state(frames * 4 * 512);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t lane = 0; lane < 4; ++lane) {
            std::copy_n(projected.begin() + frame * 512, 512,
                state.begin() + (frame * 4 + lane) * 512);
        }
    }
    for (int layer = 0; layer < kLayers; ++layer) {
        state = mhc(state, frames, weights, "encoder/mhc_", layer,
            [&](const Rows & mixed) { return encoder_block(mixed, frames, weights, assets, layer); });
    }
    Rows memory(frames * 512, 0.0f);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t lane = 0; lane < 4; ++lane) {
            for (size_t col = 0; col < 512; ++col) {
                memory[frame * 512 + col] += state[(frame * 4 + lane) * 512 + col] / 4.0f;
            }
        }
    }
    normalize(memory, frames, 512,
        require(weights, "encoder/final_norm/scale", {512}).values);
    const float gate = sigmoid(weights.get("pe_gate").values[0]);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t index = 0; index < 256; ++index) {
            const float angle = static_cast<float>(frame) *
                std::exp(static_cast<float>(index) * (-std::log(10000.0f) / 255.0f));
            memory[frame * 512 + index] += gate * std::sin(angle);
            memory[frame * 512 + 256 + index] += gate * std::cos(angle);
        }
    }
    EncoderOutput output;
    output.frames = frames;
    const std::string prefix = "stack/layers/block/cross_attn/";
    for (int layer = 0; layer < kLayers; ++layer) {
        output.cross_k[layer] = linear(memory, frames, 512,
            require(weights, prefix + "k_proj/kernel", {512, 384}, layer));
        normalize(output.cross_k[layer], frames * 8, 48,
            require(weights, prefix + "k_norm/scale", {48}, layer).values);
        output.cross_v[layer] = linear(memory, frames, 512,
            require(weights, prefix + "v_proj/kernel", {512, 512}, layer));
    }
    return output;
}

struct DecoderLayerState {
    std::array<float, 2 * 608> previous{};
    Rows keys;
    Rows values;
};

struct EngramOutput {
    Rows key;
    Rows value;
};

std::array<EngramOutput, 2> lookup_engrams(const std::vector<int32_t> & tokens,
                                           const WhistleWeights & weights) {
    const int64_t current = static_cast<int64_t>(tokens.size()) - 1;
    std::array<EngramOutput, 2> result;
    for (int site = 0; site < 2; ++site) {
        const std::string prefix = "engrams_" + std::to_string(site) + "/";
        const auto table = require(weights, prefix + "embedding", {4, 18432, 128});
        Rows feature(4 * 512, 0.0f);
        for (int tap = 0; tap < 4; ++tap) {
            const int64_t position = current - 3 * tap;
            if (position < 0) {
                continue;
            }
            for (int source = 0; source < 4; ++source) {
                const int order = source < 2 ? 2 : 3;
                if (position < order - 1) {
                    continue;
                }
                uint32_t hash = 0x9E3779B9u * static_cast<uint32_t>(source + 1);
                for (int history = 0; history < order; ++history) {
                    const auto at = position - history;
                    hash = (hash ^ static_cast<uint32_t>(at >= 0 ? tokens[static_cast<size_t>(at)] : 0)) *
                        0x01000193u;
                }
                hash ^= hash >> 15;
                const size_t row = (static_cast<size_t>(source) * 18432 + hash % 18432) * 128;
                std::copy_n(table.values + row, 128,
                    feature.begin() + static_cast<size_t>(tap * 512 + source * 128));
            }
        }
        result[site].key = linear(Rows(feature.begin(), feature.begin() + 512), 1, 512,
            require(weights, prefix + "key_proj/kernel", {512, 512}));
        const Rows projected = linear(feature, 4, 512,
            require(weights, prefix + "value_proj/kernel", {512, 512}));
        const auto taps = require(weights, prefix + "taps", {4, 512});
        result[site].value.assign(512, 0.0f);
        for (size_t tap = 0; tap < 4; ++tap) {
            for (size_t index = 0; index < 512; ++index) {
                result[site].value[index] +=
                    projected[tap * 512 + index] * taps.values[tap * 512 + index];
            }
        }
    }
    return result;
}

Rows decoder_self_attention(const Rows & input, const WhistleWeights & weights,
                            int layer, size_t position, DecoderLayerState & state) {
    const std::string prefix = "stack/layers/block/self_attn/";
    const Rows q = linear(input, 1, 512,
        require(weights, prefix + "q_proj/kernel", {512, 384}, layer));
    const Rows k = linear(input, 1, 512,
        require(weights, prefix + "k_proj/kernel", {512, 96}, layer));
    const Rows v = linear(input, 1, 512,
        require(weights, prefix + "v_proj/kernel", {512, 128}, layer));
    std::array<float, 608> current{};
    std::copy(q.begin(), q.end(), current.begin());
    std::copy(k.begin(), k.end(), current.begin() + 384);
    std::copy(v.begin(), v.end(), current.begin() + 480);
    const auto q_taps = require(weights, prefix + "q_taps", {3, 384}, layer);
    const auto k_taps = require(weights, prefix + "k_taps", {3, 96}, layer);
    const auto v_taps = require(weights, prefix + "v_taps", {3, 128}, layer);
    Rows query(384);
    Rows key(96);
    Rows value(128);
    const auto tap = [&](View weights_view, size_t width, size_t start, Rows & output) {
        for (size_t index = 0; index < width; ++index) {
            output[index] =
                current[start + index] * weights_view.values[index] +
                state.previous[608 + start + index] * weights_view.values[width + index] +
                state.previous[start + index] * weights_view.values[2 * width + index];
        }
    };
    tap(q_taps, 384, 0, query);
    tap(k_taps, 96, 384, key);
    tap(v_taps, 128, 480, value);
    std::copy_n(state.previous.begin() + 608, 608, state.previous.begin());
    std::copy(current.begin(), current.end(), state.previous.begin() + 608);
    normalize(query, 8, 48,
        require(weights, prefix + "q_norm/scale", {48}, layer).values);
    normalize(key, 2, 48,
        require(weights, prefix + "k_norm/scale", {48}, layer).values);
    apply_rope(query, 1, 8, position);
    apply_rope(key, 1, 2, position);
    state.keys.insert(state.keys.end(), key.begin(), key.end());
    state.values.insert(state.values.end(), value.begin(), value.end());
    const size_t steps = state.keys.size() / 96;
    Rows context(512, 0.0f);
    for (size_t head = 0; head < 8; ++head) {
        const size_t kv = head / 4;
        std::vector<float> scores(steps);
        for (size_t step = 0; step < steps; ++step) {
            float dot = 0.0f;
            for (size_t index = 0; index < 48; ++index) {
                dot += query[head * 48 + index] * state.keys[step * 96 + kv * 48 + index];
            }
            scores[step] = dot * (1.0f / std::sqrt(48.0f));
        }
        softmax(scores.data(), steps);
        for (size_t step = 0; step < steps; ++step) {
            for (size_t index = 0; index < 64; ++index) {
                context[head * 64 + index] +=
                    scores[step] * state.values[step * 128 + kv * 64 + index];
            }
        }
    }
    const Rows gates = linear(input, 1, 512,
        require(weights, prefix + "gate_proj/kernel", {512, 512}, layer));
    for (size_t index = 0; index < 512; ++index) {
        context[index] *= sigmoid(gates[index]);
    }
    return linear(context, 1, 512,
        require(weights, prefix + "out_proj/kernel", {512, 512}, layer));
}

Rows decoder_cross_attention(const Rows & input, const WhistleWeights & weights,
                             const EncoderOutput & encoder, int layer) {
    const std::string prefix = "stack/layers/block/cross_attn/";
    Rows query = linear(input, 1, 512,
        require(weights, prefix + "q_proj/kernel", {512, 384}, layer));
    normalize(query, 8, 48,
        require(weights, prefix + "q_norm/scale", {48}, layer).values);
    Rows context(512, 0.0f);
    const Rows & keys = encoder.cross_k[layer];
    const Rows & values = encoder.cross_v[layer];
    for (size_t head = 0; head < 8; ++head) {
        std::vector<float> scores(encoder.frames);
        for (size_t frame = 0; frame < encoder.frames; ++frame) {
            float dot = 0.0f;
            for (size_t col = 0; col < 48; ++col) {
                dot += query[head * 48 + col] * keys[(frame * 8 + head) * 48 + col];
            }
            scores[frame] = dot * (1.0f / std::sqrt(48.0f));
        }
        softmax(scores.data(), encoder.frames);
        for (size_t frame = 0; frame < encoder.frames; ++frame) {
            for (size_t col = 0; col < 64; ++col) {
                context[head * 64 + col] +=
                    scores[frame] * values[(frame * 8 + head) * 64 + col];
            }
        }
    }
    const Rows gates = linear(input, 1, 512,
        require(weights, prefix + "gate_proj/kernel", {512, 512}, layer));
    for (size_t index = 0; index < 512; ++index) {
        context[index] *= sigmoid(gates[index]);
    }
    return linear(context, 1, 512,
        require(weights, prefix + "out_proj/kernel", {512, 512}, layer));
}

Rows decoder_block(const Rows & input, const WhistleWeights & weights,
                   const WhistleAssets & assets, const EncoderOutput & encoder,
                   const std::array<EngramOutput, 2> & engram, int layer,
                   size_t position, DecoderLayerState & state) {
    const std::string prefix = "stack/layers/block/";
    Rows h = input;
    if (layer == 3 || layer == 7) {
        const auto & slot = engram[layer == 3 ? 0 : 1];
        Rows normalized = h;
        Rows key = slot.key;
        normalize(normalized, 1, 512);
        normalize(key, 1, 512);
        float dot = 0.0f;
        for (size_t index = 0; index < 512; ++index) {
            dot += normalized[index] * key[index];
        }
        add(h, slot.value, sigmoid(dot * (1.0f / std::sqrt(512.0f))));
    }
    Rows self_input = h;
    normalize(self_input, 1, 512,
        require(weights, prefix + "ZCRMSNorm_0/scale", {512}, layer).values);
    Rows self = decoder_self_attention(self_input, weights, layer, position, state);
    normalize(self, 1, 512,
        require(weights, prefix + "post_attn_norm/scale", {512}, layer).values);
    add(h, self, sigmoid(require(weights, prefix + "attn_gate", {}, layer).values[0]));
    Rows cross_input = h;
    normalize(cross_input, 1, 512,
        require(weights, prefix + "cross_norm/scale", {512}, layer).values);
    Rows cross = decoder_cross_attention(cross_input, weights, encoder, layer);
    normalize(cross, 1, 512,
        require(weights, prefix + "post_cross_norm/scale", {512}, layer).values);
    add(h, cross, sigmoid(require(weights, prefix + "cross_gate", {}, layer).values[0]));
    Rows ff_input = h;
    normalize(ff_input, 1, 512,
        require(weights, prefix + "pre_hada_norm/scale", {512}, layer).values);
    add(h, hadamard(ff_input, 1, weights, assets,
        prefix + "hadamard_mlp/", layer));
    return h;
}

Rows decoder_step(int32_t token, size_t position, const WhistleWeights & weights,
                  const WhistleAssets & assets, const EncoderOutput & encoder,
                  const std::array<EngramOutput, 2> & engram,
                  std::array<DecoderLayerState, 8> & cache) {
    const auto embedding = require(weights, "embedding/embedding", {8199, 512});
    Rows x(512);
    for (size_t index = 0; index < 512; ++index) {
        x[index] = embedding.values[static_cast<size_t>(token) * 512 + index] *
            std::sqrt(512.0f);
    }
    Rows state(4 * 512);
    for (size_t lane = 0; lane < 4; ++lane) {
        std::copy(x.begin(), x.end(), state.begin() + lane * 512);
    }
    for (int layer = 0; layer < kLayers; ++layer) {
        state = mhc(state, 1, weights, "stack/mhc_", layer,
            [&](const Rows & mixed) {
                return decoder_block(mixed, weights, assets, encoder,
                    engram, layer, position, cache[layer]);
            });
    }
    Rows h(512, 0.0f);
    for (size_t lane = 0; lane < 4; ++lane) {
        for (size_t index = 0; index < 512; ++index) {
            h[index] += state[lane * 512 + index] / 4.0f;
        }
    }
    normalize(h, 1, 512,
        require(weights, "stack/final_norm/scale", {512}).values);
    Rows logits(8199);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t id = 0; id < 8199; ++id) {
        float score = 0.0f;
        for (size_t index = 0; index < 512; ++index) {
            score += h[index] * embedding.values[static_cast<size_t>(id) * 512 + index];
        }
        logits[static_cast<size_t>(id)] = score;
    }
    return logits;
}

}  // namespace

WhistleRuntime::WhistleRuntime(std::shared_ptr<const WhistleAssets> assets, int threads)
    : assets_(std::move(assets)), threads_(threads) {
    if (!assets_ || !assets_->weights) {
        throw std::invalid_argument("Whistle runtime needs verified model assets");
    }
    if (threads_ < 1 || threads_ > 64) {
        throw std::invalid_argument("Whistle CPU thread count must be between 1 and 64");
    }
    weights_ = std::make_unique<WhistleWeights>(*assets_->weights);
}

WhistleRuntime::~WhistleRuntime() = default;

WhistleTranscript WhistleRuntime::transcribe(
    const runtime::AudioBuffer & audio, const std::string & language) const {
    if (audio.sample_rate != 16000 || audio.channels != 1) {
        throw std::invalid_argument("Whistle requires 16 kHz mono audio");
    }
    if (audio.samples.size() > 30 * 16000) {
        throw std::invalid_argument("Whistle audio exceeds the 30-second input limit");
    }
    if (!language.empty() &&
        std::find_if(kLanguages.begin(), kLanguages.end(),
            [&](const char * candidate) { return language == candidate; }) == kLanguages.end()) {
        throw std::invalid_argument("Whistle does not support the requested language");
    }
    double energy = 0.0;
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            throw std::invalid_argument("Whistle audio contains a non-finite sample");
        }
        energy += static_cast<double>(sample) * sample;
    }
    if (audio.samples.size() < 640) {
        return {"", ""};
    }
    if (energy == 0.0) {
        return {"", ""};
    }
#ifdef _OPENMP
    ScopedThreadCount thread_count(threads_);
#else
    if (threads_ != 1) {
        throw std::invalid_argument("Whistle was built without OpenMP; use one CPU thread");
    }
#endif
    const auto mel = WhistleFrontend(assets_->mel_filterbank).extract(audio.samples);
    const EncoderOutput encoder = encode(mel, *weights_, *assets_);
    std::array<DecoderLayerState, 8> cache;
    std::vector<int32_t> tokens{2};
    std::vector<int32_t> text;
    std::string detected_language;
    bool ended = false;
    for (size_t position = 0; position <= kMaximumTokens + 1; ++position) {
        const auto engram = lookup_engrams(tokens, *weights_);
        const Rows logits = decoder_step(tokens.back(), position, *weights_, *assets_, encoder, engram, cache);
        if (position == 0) {
            const size_t index = language.empty()
                ? static_cast<size_t>(std::max_element(logits.begin() + 8192,
                    logits.begin() + 8199) - (logits.begin() + 8192))
                : static_cast<size_t>(std::find_if(kLanguages.begin(), kLanguages.end(),
                    [&](const char * candidate) { return language == candidate; }) - kLanguages.begin());
            detected_language = kLanguages[index];
            tokens.push_back(static_cast<int32_t>(8192 + index));
            continue;
        }
        const int32_t next = static_cast<int32_t>(
            std::max_element(logits.begin(), logits.begin() + kTextVocabulary) - logits.begin());
        if (next == 1) {
            ended = true;
            break;
        }
        if (text.size() >= kMaximumTokens) {
            break;
        }
        text.push_back(next);
        tokens.push_back(next);
    }
    if (!ended) {
        // The C API has no partial-result status; do not present capped text as complete speech.
        throw std::runtime_error("Whistle decoding reached the token limit without an end token");
    }
    std::string transcript = decode_whistle_tokens(assets_->tokenizer_pieces, text);
    const auto first = transcript.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        transcript.clear();
    } else {
        const auto last = transcript.find_last_not_of(" \t\r\n");
        transcript = transcript.substr(first, last + 1 - first);
    }
    if (transcript.empty()) {
        detected_language.clear();
    }
    return {std::move(transcript), std::move(detected_language)};
}

}  // namespace engine::community_models::whistle_asr
