#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace engine::neon::detail {

#if defined(__aarch64__) && defined(__ARM_NEON)
inline float32x4_t exp4(float32x4_t x);
#endif

bool available() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
}

void quantize_dequantize(const float *input, float *output, size_t count, float scale,
                        float minimum, float maximum) noexcept {
    const float inverse = 1.0F / scale;
    size_t i = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; i + 4 <= count; i += 4) {
        auto value = vrndaq_f32(vmulq_n_f32(vld1q_f32(input + i), inverse));
        value = vmaxq_f32(vdupq_n_f32(minimum), vminq_f32(vdupq_n_f32(maximum), value));
        vst1q_f32(output + i, vmulq_n_f32(value, scale));
    }
#endif
    for (; i < count; ++i) output[i] = std::clamp(std::round(input[i] * inverse), minimum, maximum) * scale;
}

void depthwise_conv1d(const float *input, const float *weight, const float *bias, float *output,
                      const Conv1dConfig &config, int frames, int output_frames, WorkerContext &worker) noexcept {
    int begin, end;
    split_range(config.input_channels, worker.index, worker.count, begin, end);
    for (int channel = begin; channel < end; ++channel) {
        for (int frame = 0; frame < output_frames; ++frame) {
            float value = bias ? bias[channel] : 0.0F;
            for (int tap = 0; tap < config.kernel_size; ++tap) {
                const int at = frame * config.stride - config.padding_left + tap * config.dilation;
                if (at >= 0 && at < frames) value += input[channel * frames + at] * weight[channel * config.kernel_size + tap];
            }
            output[channel * output_frames + frame] = value;
        }
    }
}

PackedConvWeights pack_conv_weights(const float *w, const float *b, int cin, int cout, int kernel) {
    if (!w || cin <= 0 || cout <= 0 || cout > std::numeric_limits<int>::max() - 3 || kernel <= 0 ||
        cin > std::numeric_limits<int>::max() / kernel) {
        throw std::invalid_argument("ARM convolution weights require positive channels and kernel size");
    }
    PackedConvWeights c;
    c.input_channels = cin;
    c.output_channels = cout;
    c.kernel_size = kernel;
    c.padded_output_channels = (cout + 3) & ~3;
    c.bias.assign(static_cast<size_t>(c.padded_output_channels), 0.0F);
    if (b) {
        std::copy(b, b + cout, c.bias.begin());
    }
    const int rows = cin * kernel;
    c.values.assign(static_cast<size_t>(rows) * c.padded_output_channels, 0.0F);
    float *dst = c.values.data();
    for (int ob = 0; ob < c.padded_output_channels / 4; ++ob) {
        for (int r = 0; r < rows; ++r, dst += 4) {
            for (int j = 0; j < 4 && ob * 4 + j < cout; ++j) {
                dst[j] = w[static_cast<size_t>(ob * 4 + j) * rows + r];
            }
        }
    }
    return c;
}

void split_range(int n, int worker_index, int worker_count, int &lo, int &hi) {
    lo = static_cast<int>(static_cast<int64_t>(n) * worker_index / worker_count);
    hi = static_cast<int>(static_cast<int64_t>(n) * (worker_index + 1) / worker_count);
}

void grow(std::vector<float> &buffer, size_t size) {
    if (buffer.size() < size) {
        buffer.resize(size);
    }
}

void linear(const float *weight, const float *bias, const float *input, float *output, int input_features,
            int output_features, int frames, bool input_frame_major, bool output_frame_major,
            WorkerContext &worker) noexcept {
    int begin, end;
    split_range(output_features, worker.index, worker.count, begin, end);
    for (int row = begin; row < end; ++row) {
        const float *w = weight + static_cast<size_t>(row) * input_features;
        for (int frame = 0; frame < frames; ++frame) {
            float value = bias ? bias[row] : 0.0F;
            int feature = 0;
            if (input_frame_major || frames == 1) {
                const float *x = input + static_cast<size_t>(frame) * input_features;
#if defined(__aarch64__) && defined(__ARM_NEON)
                float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
                for (; feature + 16 <= input_features; feature += 16) {
                    a = vfmaq_f32(a, vld1q_f32(w + feature), vld1q_f32(x + feature));
                    b = vfmaq_f32(b, vld1q_f32(w + feature + 4), vld1q_f32(x + feature + 4));
                    c = vfmaq_f32(c, vld1q_f32(w + feature + 8), vld1q_f32(x + feature + 8));
                    d = vfmaq_f32(d, vld1q_f32(w + feature + 12), vld1q_f32(x + feature + 12));
                }
                value += vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
#endif
                for (; feature < input_features; ++feature) {
                    value += w[feature] * x[feature];
                }
            } else {
                for (; feature < input_features; ++feature) {
                    value += w[feature] * input[static_cast<size_t>(feature) * frames + frame];
                }
            }
            output[output_frame_major ? static_cast<size_t>(frame) * output_features + row
                                      : static_cast<size_t>(row) * frames + frame] = value;
        }
    }
}

PackedInt8Rows pack_int8_rows(const BlockInt8Weights &weight, int input_features, int output_features) {
    const int blocks = input_features / 32, tiles = (output_features + 3) / 4;
    PackedInt8Rows packed;
    packed.values.resize(static_cast<size_t>(tiles) * blocks * 128);
    packed.scales.resize(static_cast<size_t>(tiles) * blocks * 4);
    for (int tile = 0; tile < tiles; ++tile) {
        for (int block = 0; block < blocks; ++block) {
            const size_t offset = (static_cast<size_t>(tile) * blocks + block) * 128;
            for (int row = 0; row < 4 && tile * 4 + row < output_features; ++row) {
                const size_t source = static_cast<size_t>(tile * 4 + row) * input_features + block * 32;
                packed.scales[(static_cast<size_t>(tile) * blocks + block) * 4 + row] = weight.scales[source / 32];
                for (int group = 0; group < 8; ++group) {
                    std::copy_n(weight.values.data() + source + group * 4, 4,
                                packed.values.data() + offset + group * 16 + row * 4);
                }
            }
        }
    }
    return packed;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
inline int32x4_t dot_int8_rows(const int8_t *weight, const int8_t *input) {
#if defined(__ARM_FEATURE_DOTPROD)
    const auto x = vld1q_s8(input), y = vld1q_s8(input + 16);
    auto a = vdotq_laneq_s32(vdupq_n_s32(0), vld1q_s8(weight), x, 0);
    auto b = vdotq_laneq_s32(vdupq_n_s32(0), vld1q_s8(weight + 16), x, 1);
    auto c = vdotq_laneq_s32(vdupq_n_s32(0), vld1q_s8(weight + 32), x, 2);
    auto d = vdotq_laneq_s32(vdupq_n_s32(0), vld1q_s8(weight + 48), x, 3);
    a = vdotq_laneq_s32(a, vld1q_s8(weight + 64), y, 0);
    b = vdotq_laneq_s32(b, vld1q_s8(weight + 80), y, 1);
    c = vdotq_laneq_s32(c, vld1q_s8(weight + 96), y, 2);
    d = vdotq_laneq_s32(d, vld1q_s8(weight + 112), y, 3);
    return vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d));
#else
    auto sum = vdupq_n_s32(0);
    for (int group = 0; group < 8; ++group) {
        int32_t four;
        std::memcpy(&four, input + group * 4, 4);
        const auto x = vreinterpretq_s8_s32(vdupq_n_s32(four));
        const auto w = vld1q_s8(weight + group * 16);
        const auto low = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(x)));
        const auto high = vpaddlq_s16(vmull_s8(vget_high_s8(w), vget_high_s8(x)));
        sum = vaddq_s32(sum, vpaddq_s32(low, high));
    }
    return sum;
#endif
}
#endif

void quantized_linear(const PackedInt8Rows &weight, const float *bias, const float *input, float *output,
                      int input_features, int output_features, int frames, bool input_frame_major,
                      bool output_frame_major, Workspace &workspace, WorkerContext &worker) noexcept {
    const int blocks = input_features / 32;
    int begin, end;
    split_range(frames * blocks, worker.index, worker.count, begin, end);
    for (int block = begin; block < end; ++block) {
        const int frame = block / blocks, feature = (block % blocks) * 32;
#if defined(__aarch64__) && defined(__ARM_NEON)
        float gathered[32];
        const float *source;
        if (input_frame_major || frames == 1) {
            source = input + static_cast<size_t>(frame) * input_features + feature;
        } else {
            for (int i = 0; i < 32; ++i) {
                gathered[i] = input[static_cast<size_t>(feature + i) * frames + frame];
            }
            source = gathered;
        }
        float32x4_t values[8];
        float32x4_t maximum = vdupq_n_f32(0);
        uint32x4_t valid = vdupq_n_u32(UINT32_MAX);
        for (int i = 0; i < 8; ++i) {
            values[i] = vld1q_f32(source + i * 4);
            const auto magnitude = vabsq_f32(values[i]);
            maximum = vmaxq_f32(maximum, magnitude);
            valid = vandq_u32(valid, vcleq_f32(magnitude, vdupq_n_f32(std::numeric_limits<float>::max())));
        }
        const bool finite = vminvq_u32(valid) == UINT32_MAX;
        const float scale = vmaxvq_f32(maximum) / 127.0F;
        const float reciprocal = scale ? 1.0F / scale : 0;
        const float inverse = std::isfinite(reciprocal) ? reciprocal : 0;
        workspace.quantized_scales[block] = finite && scale <= 65504.0F
                                                ? vgetq_lane_f32(vcvt_f32_f16(vcvt_f16_f32(vdupq_n_f32(scale))), 0)
                                                : std::numeric_limits<float>::quiet_NaN();
        auto *destination = workspace.quantized_input.data() + static_cast<size_t>(block) * 32;
        if (!finite) {
            std::fill(destination, destination + 32, int8_t(0));
        } else {
            for (int i = 0; i < 8; i += 2) {
                const auto a = vcvtnq_s32_f32(vmulq_n_f32(values[i], inverse));
                const auto b = vcvtnq_s32_f32(vmulq_n_f32(values[i + 1], inverse));
                const auto bounded_a = vmaxq_s32(vdupq_n_s32(-127), vminq_s32(vdupq_n_s32(127), a));
                const auto bounded_b = vmaxq_s32(vdupq_n_s32(-127), vminq_s32(vdupq_n_s32(127), b));
                vst1_s8(destination + i * 4, vmovn_s16(vcombine_s16(vmovn_s32(bounded_a), vmovn_s32(bounded_b))));
            }
        }
#else
        float values[32], maximum = 0;
        bool finite = true;
        for (int i = 0; i < 32; ++i) {
            values[i] = input[input_frame_major ? static_cast<size_t>(frame) * input_features + feature + i
                                                : static_cast<size_t>(feature + i) * frames + frame];
            maximum = std::max(maximum, std::abs(values[i]));
            finite = finite && std::isfinite(values[i]);
        }
        const float scale = maximum / 127.0F;
        const float reciprocal = scale ? 1.0F / scale : 0;
        const float inverse = std::isfinite(reciprocal) ? reciprocal : 0;
        // Round the scale to binary16, as required by this block-int8 arithmetic.
        int exponent = 0;
        std::frexp(scale, &exponent);
        const float step = std::ldexp(1.0F, std::max(exponent - 11, -24));
        workspace.quantized_scales[block] =
            finite && scale <= 65504.0F ? std::nearbyint(scale / step) * step : std::numeric_limits<float>::quiet_NaN();
        for (int i = 0; i < 32; ++i) {
            workspace.quantized_input[static_cast<size_t>(block) * 32 + i] =
                finite ? static_cast<int8_t>(std::clamp(std::nearbyint(values[i] * inverse), -127.0F, 127.0F)) : 0;
        }
#endif
    }
    worker.synchronize();
    split_range((output_features + 3) / 4, worker.index, worker.count, begin, end);
    for (int tile = begin; tile < end; ++tile) {
        const int row = tile * 4, rows = std::min(4, output_features - row);
#if defined(__aarch64__) && defined(__ARM_NEON)
        float initial[4]{};
        for (int r = 0; r < rows; ++r) {
            initial[r] = bias ? bias[row + r] : 0;
        }
        for (int frame = 0; frame < frames; frame += 2) {
            auto first = vld1q_f32(initial), second = first;
            const auto *x = workspace.quantized_input.data() + static_cast<size_t>(frame) * input_features;
            for (int block = 0; block < blocks; ++block) {
                const size_t offset = static_cast<size_t>(tile) * blocks + block;
                const auto *w = weight.values.data() + offset * 128;
                const auto scale = vld1q_f32(weight.scales.data() + offset * 4);
                first = vfmaq_f32(first, vcvtq_f32_s32(dot_int8_rows(w, x + block * 32)),
                                  vmulq_n_f32(scale, workspace.quantized_scales[frame * blocks + block]));
                if (frame + 1 < frames) {
                    second = vfmaq_f32(second, vcvtq_f32_s32(dot_int8_rows(w, x + input_features + block * 32)),
                                       vmulq_n_f32(scale, workspace.quantized_scales[(frame + 1) * blocks + block]));
                }
            }
            float result[2][4];
            vst1q_f32(result[0], first);
            vst1q_f32(result[1], second);
            for (int f = 0; f < std::min(2, frames - frame); ++f) {
                for (int r = 0; r < rows; ++r) {
                    output[output_frame_major ? static_cast<size_t>(frame + f) * output_features + row + r
                                              : static_cast<size_t>(row + r) * frames + frame + f] = result[f][r];
                }
            }
        }
#else
        for (int frame = 0; frame < frames; ++frame) {
            for (int r = 0; r < rows; ++r) {
                float sum = bias ? bias[row + r] : 0;
                for (int block = 0; block < blocks; ++block) {
                    const size_t offset = static_cast<size_t>(tile) * blocks + block;
                    int dot = 0;
                    for (int i = 0; i < 32; ++i) {
                        dot += int(weight.values[offset * 128 + (i / 4) * 16 + r * 4 + i % 4]) *
                               int(workspace
                                       .quantized_input[static_cast<size_t>(frame) * input_features + block * 32 + i]);
                    }
                    sum += dot * (weight.scales[offset * 4 + r] * workspace.quantized_scales[frame * blocks + block]);
                }
                output[output_frame_major ? static_cast<size_t>(frame) * output_features + row + r
                                          : static_cast<size_t>(row + r) * frames + frame] = sum;
            }
        }
#endif
    }
}

void rotary_embedding(const float *input, float *output, int channels, int frames, bool frame_major, int head_dimension,
                      const std::vector<float> &frequencies, int position, WorkerContext &worker) noexcept {
    int begin, end;
    split_range((channels / head_dimension) * frames, worker.index, worker.count, begin, end);
    for (int row = begin; row < end; ++row) {
        const int head = row / frames, frame = row % frames;
        const auto offset = [&](int dimension) -> size_t {
            const int channel = head * head_dimension + dimension;
            return frame_major ? static_cast<size_t>(frame) * channels + channel
                               : static_cast<size_t>(channel) * frames + frame;
        };
        for (size_t pair = 0; pair < frequencies.size(); ++pair) {
            const float angle = (position + frame) * frequencies[pair];
            const float cosine = std::cos(angle), sine = std::sin(angle);
            const size_t a = offset(static_cast<int>(pair) * 2), b = offset(static_cast<int>(pair) * 2 + 1);
            output[a] = input[a] * cosine - input[b] * sine;
            output[b] = input[a] * sine + input[b] * cosine;
        }
        for (int dimension = static_cast<int>(frequencies.size()) * 2; dimension < head_dimension; ++dimension) {
            output[offset(dimension)] = input[offset(dimension)];
        }
    }
}

void attention(const float *query, const float *key, const float *value, const float *current_key,
               const float *current_value, float *output, int query_frames, int key_frames, int heads,
               int key_value_heads, int dimension, int key_capacity, bool key_head_major, bool query_frame_major,
               bool output_frame_major, Workspace &workspace, WorkerContext &worker) noexcept {
    int begin, end;
    split_range(heads * query_frames, worker.index, worker.count, begin, end);
    float *scores = workspace.attention.data() + static_cast<size_t>(worker.index) * workspace.attention_rows;
    float *accumulator = scores + key_frames;
    const float scale = 1.0F / std::sqrt(static_cast<float>(dimension));
    for (int row = begin; row < end; ++row) {
        const int head = row / query_frames, frame = row % query_frames;
        const int kv_head = head / (heads / key_value_heads);
        const float *q = query + (query_frame_major ? static_cast<size_t>(frame) * heads * dimension + head * dimension
                                                    : static_cast<size_t>(head) * dimension * query_frames + frame);
        float maximum = -std::numeric_limits<float>::infinity();
        for (int index = 0; index < key_frames; ++index) {
            const float *k =
                current_key && index == key_frames - 1
                    ? current_key + kv_head * dimension
                    : key + (key_head_major ? (static_cast<size_t>(kv_head) * key_capacity + index) * dimension
                                            : (static_cast<size_t>(index) * key_value_heads + kv_head) * dimension);
            float dot = 0;
            int d = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
            if (query_frame_major || query_frames == 1) {
                float32x4_t a = vdupq_n_f32(0), b = a, c = a, e = a;
                for (; d + 16 <= dimension; d += 16) {
                    a = vfmaq_f32(a, vld1q_f32(q + d), vld1q_f32(k + d));
                    b = vfmaq_f32(b, vld1q_f32(q + d + 4), vld1q_f32(k + d + 4));
                    c = vfmaq_f32(c, vld1q_f32(q + d + 8), vld1q_f32(k + d + 8));
                    e = vfmaq_f32(e, vld1q_f32(q + d + 12), vld1q_f32(k + d + 12));
                }
                auto sum = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, e));
                for (; d + 4 <= dimension; d += 4) {
                    sum = vfmaq_f32(sum, vld1q_f32(q + d), vld1q_f32(k + d));
                }
                dot = vaddvq_f32(sum);
            }
#endif
            for (; d < dimension; ++d) {
                dot += q[(query_frame_major || query_frames == 1) ? d : d * query_frames] * k[d];
            }
            scores[index] = dot * scale;
            maximum = std::max(maximum, scores[index]);
        }
        float denominator = 0;
        int score_index = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
        float32x4_t total = vdupq_n_f32(0);
        const float32x4_t max_score = vdupq_n_f32(maximum);
        for (; score_index + 4 <= key_frames; score_index += 4) {
            const auto probability = exp4(vsubq_f32(vld1q_f32(scores + score_index), max_score));
            vst1q_f32(scores + score_index, probability);
            total = vaddq_f32(total, probability);
        }
        denominator = vaddvq_f32(total);
#endif
        for (int index = score_index; index < key_frames; ++index) {
            scores[index] = std::exp(scores[index] - maximum);
            denominator += scores[index];
        }
        const float inverse_denominator = 1.0F / denominator;
        std::fill(accumulator, accumulator + dimension, 0.0F);
        int first_dimension = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
        for (; first_dimension + 16 <= dimension; first_dimension += 16) {
            float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
            for (int index = 0; index < key_frames; ++index) {
                const float *v =
                    current_value && index == key_frames - 1
                        ? current_value + kv_head * dimension
                        : value + (key_head_major
                                       ? (static_cast<size_t>(kv_head) * key_capacity + index) * dimension
                                       : (static_cast<size_t>(index) * key_value_heads + kv_head) * dimension);
                v += first_dimension;
                const float probability = scores[index] * inverse_denominator;
                a = vfmaq_n_f32(a, vld1q_f32(v), probability);
                b = vfmaq_n_f32(b, vld1q_f32(v + 4), probability);
                c = vfmaq_n_f32(c, vld1q_f32(v + 8), probability);
                d = vfmaq_n_f32(d, vld1q_f32(v + 12), probability);
            }
            vst1q_f32(accumulator + first_dimension, a);
            vst1q_f32(accumulator + first_dimension + 4, b);
            vst1q_f32(accumulator + first_dimension + 8, c);
            vst1q_f32(accumulator + first_dimension + 12, d);
        }
#endif
        for (int index = 0; index < key_frames; ++index) {
            const float *v =
                current_value && index == key_frames - 1
                    ? current_value + kv_head * dimension
                    : value + (key_head_major ? (static_cast<size_t>(kv_head) * key_capacity + index) * dimension
                                              : (static_cast<size_t>(index) * key_value_heads + kv_head) * dimension);
            const float probability = scores[index] * inverse_denominator;
            int d = first_dimension;
#if defined(__aarch64__) && defined(__ARM_NEON)
            for (; d + 4 <= dimension; d += 4) {
                vst1q_f32(accumulator + d, vfmaq_n_f32(vld1q_f32(accumulator + d), vld1q_f32(v + d), probability));
            }
#endif
            for (; d < dimension; ++d) {
                accumulator[d] += probability * v[d];
            }
        }
        for (int d = 0; d < dimension; ++d) {
            const int channel = head * dimension + d;
            output[output_frame_major ? static_cast<size_t>(frame) * heads * dimension + channel
                                      : static_cast<size_t>(channel) * query_frames + frame] = accumulator[d];
        }
    }
}

void apply_silu(float *values, size_t begin, size_t end) noexcept {
    for (size_t i = begin; i < end; ++i) {
        values[i] /= 1.0F + std::exp(-values[i]);
    }
}

void Workspace::prepare(int frame_count, int max_input_channels, int max_kernel) {
    if (frame_count <= 0 || max_input_channels <= 0 || max_kernel <= 0 || max_kernel % 2 == 0 ||
        ((static_cast<int64_t>(frame_count) + 15) & ~15LL) + max_kernel - 1 > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Invalid ARM workspace shape");
    }
    max_kernel_size = max_kernel;
    const int aligned_frames = (frame_count + 15) & ~15;
    const int packed_stride = aligned_frames + max_kernel - 1;
    grow(packed_input, static_cast<size_t>(max_input_channels) * packed_stride);
    grow(norm_mean, frame_count);
    grow(norm_inverse_std, frame_count);
}

void Workspace::reserve_panels(int rows, int workers) {
    if (rows <= 0 || workers <= 0) {
        throw std::invalid_argument("Invalid ARM panel shape");
    }
    panel_rows = std::max(panel_rows, static_cast<size_t>(rows));
    grow(panels, panel_rows * 16 * workers);
}

void Workspace::reserve_attention(int rows, int workers) {
    if (rows <= 0 || workers <= 0) {
        throw std::invalid_argument("Invalid ARM attention workspace shape");
    }
    attention_rows = std::max(attention_rows, static_cast<size_t>(rows));
    grow(attention, attention_rows * workers);
}

#if defined(__aarch64__) && defined(__ARM_NEON)
// 4 output channels x 16 frames per register block.
#define NEON_FMA16(xq0, xq1, xq2, xq3, wv)                                                                          \
    acc[0][0] = vfmaq_laneq_f32(acc[0][0], xq0, wv, 0);                                                                \
    acc[0][1] = vfmaq_laneq_f32(acc[0][1], xq1, wv, 0);                                                                \
    acc[0][2] = vfmaq_laneq_f32(acc[0][2], xq2, wv, 0);                                                                \
    acc[0][3] = vfmaq_laneq_f32(acc[0][3], xq3, wv, 0);                                                                \
    acc[1][0] = vfmaq_laneq_f32(acc[1][0], xq0, wv, 1);                                                                \
    acc[1][1] = vfmaq_laneq_f32(acc[1][1], xq1, wv, 1);                                                                \
    acc[1][2] = vfmaq_laneq_f32(acc[1][2], xq2, wv, 1);                                                                \
    acc[1][3] = vfmaq_laneq_f32(acc[1][3], xq3, wv, 1);                                                                \
    acc[2][0] = vfmaq_laneq_f32(acc[2][0], xq0, wv, 2);                                                                \
    acc[2][1] = vfmaq_laneq_f32(acc[2][1], xq1, wv, 2);                                                                \
    acc[2][2] = vfmaq_laneq_f32(acc[2][2], xq2, wv, 2);                                                                \
    acc[2][3] = vfmaq_laneq_f32(acc[2][3], xq3, wv, 2);                                                                \
    acc[3][0] = vfmaq_laneq_f32(acc[3][0], xq0, wv, 3);                                                                \
    acc[3][1] = vfmaq_laneq_f32(acc[3][1], xq1, wv, 3);                                                                \
    acc[3][2] = vfmaq_laneq_f32(acc[3][2], xq2, wv, 3);                                                                \
    acc[3][3] = vfmaq_laneq_f32(acc[3][3], xq3, wv, 3);

// Writes one 4-channel x 16-frame result block: channel-major (out[o * T + t]) or, with
// rows_ld > 0, frame rows (out[t * rows_ld + o]). Only the first m frames / no channels are real.
inline void store_block(const float32x4_t (&acc)[4][4], float *out, int rows_ld, int T, int t0, int m, int o0, int no) {
    float tmp[4][16];
    for (int j = 0; j < 4; ++j) {
        for (int q = 0; q < 4; ++q) {
            vst1q_f32(tmp[j] + 4 * q, acc[j][q]);
        }
    }
    if (rows_ld > 0) {
        for (int t = 0; t < m; ++t) {
            for (int j = 0; j < no; ++j) {
                out[static_cast<size_t>(t0 + t) * rows_ld + o0 + j] = tmp[j][t];
            }
        }
    } else {
        for (int j = 0; j < no; ++j) {
            std::memcpy(out + static_cast<size_t>(o0 + j) * T + t0, tmp[j], sizeof(float) * m);
        }
    }
}

inline float32x4_t exp4(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0F));
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341F)));
    float32x4_t q = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125F));
    q = vfmsq_f32(q, n, vdupq_n_f32(1.428606765330187045e-06F));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4F);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3F), p, q);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3F), p, q);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2F), p, q);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1F), p, q);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1F), p, q);
    p = vfmaq_f32(vaddq_f32(q, vdupq_n_f32(1.0F)), vmulq_f32(p, q), q);
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

/** Exact-erf GELU (nn.GELU()), erf by Abramowitz-Stegun 7.1.26, |error| < 1.5e-7. */
void apply_gelu(float *x, size_t lo, size_t hi) noexcept {
    size_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        const float32x4_t z = vmulq_f32(v, vdupq_n_f32(0.70710678118654752F));
        const float32x4_t az = vabsq_f32(z);
        const float32x4_t t = vdivq_f32(vdupq_n_f32(1.0F), vfmaq_f32(vdupq_n_f32(1.0F), az, vdupq_n_f32(0.3275911F)));
        float32x4_t p = vfmaq_f32(vdupq_n_f32(-1.453152027F), t, vdupq_n_f32(1.061405429F));
        p = vfmaq_f32(vdupq_n_f32(1.421413741F), p, t);
        p = vfmaq_f32(vdupq_n_f32(-0.284496736F), p, t);
        p = vfmaq_f32(vdupq_n_f32(0.254829592F), p, t);
        float32x4_t er = vfmsq_f32(vdupq_n_f32(1.0F), vmulq_f32(p, t), exp4(vnegq_f32(vmulq_f32(az, az))));
        er = vbslq_f32(vcltq_f32(z, vdupq_n_f32(0.0F)), vnegq_f32(er), er);
        vst1q_f32(x + i, vmulq_f32(vmulq_f32(vdupq_n_f32(0.5F), v), vaddq_f32(vdupq_n_f32(1.0F), er)));
    }
    for (; i < hi; ++i) {
        x[i] = 0.5F * x[i] * (1.0F + std::erf(x[i] * 0.70710678118654752F));
    }
}

void convolve_1d(const PackedConvWeights &c, const float *src, float *out, int rows_ld, int frames,
                 Workspace &workspace, WorkerContext &worker) noexcept {
    const int worker_index = worker.index;
    const int worker_count = worker.count;

    const int T = frames;
    const int frames16 = (T + 15) & ~15;
    const int padded = frames16 + workspace.max_kernel_size - 1;
    const int cin = c.input_channels;
    const int K = c.kernel_size;
    const int pad = K / 2;
    const int co4 = c.padded_output_channels;
    float *xp = workspace.packed_input.data();
    int lo = 0, hi = 0;
    if (K == 1) {
        // Frame tiling reuses each packed panel across output channels while it is in L1.
        split_range(frames16 / 16, worker_index, worker_count, lo, hi);
        for (int tb = lo; tb < hi; ++tb) {
            const int t0 = tb * 16;
            const int m = std::min(16, T - t0);
            float *blk = xp + static_cast<size_t>(tb) * cin * 16;
            for (int ci = 0; ci < cin; ++ci) {
                float *dst = blk + static_cast<size_t>(ci) * 16;
                std::memcpy(dst, src + static_cast<size_t>(ci) * T + t0, sizeof(float) * m);
                if (m < 16) {
                    std::memset(dst + m, 0, sizeof(float) * (16 - m));
                }
            }
            for (int ob = 0; ob < co4 / 4; ++ob) {
                const int o0 = ob * 4;
                const int no = std::min(4, c.output_channels - o0);
                float32x4_t acc[4][4];
                for (int j = 0; j < 4; ++j) {
                    const float32x4_t bj = vdupq_n_f32(c.bias[o0 + j]);
                    for (int q = 0; q < 4; ++q) {
                        acc[j][q] = bj;
                    }
                }
                const float *xq = blk;
                const float *wq = c.values.data() + static_cast<size_t>(ob) * cin * 4;
                for (int ci = 0; ci < cin; ++ci, xq += 16, wq += 4) {
                    const float32x4_t wv = vld1q_f32(wq);
                    const float32x4_t x0 = vld1q_f32(xq), x1 = vld1q_f32(xq + 4);
                    const float32x4_t x2 = vld1q_f32(xq + 8), x3 = vld1q_f32(xq + 12);
                    NEON_FMA16(x0, x1, x2, x3, wv)
                }
                store_block(acc, out, rows_ld, T, t0, m, o0, no);
            }
        }
        return;
    }
    // K > 1: every thread packs some zero-padded input rows (width `padded`), then all wait
    split_range(cin, worker_index, worker_count, lo, hi);
    for (int ci = lo; ci < hi; ++ci) {
        float *row = xp + static_cast<size_t>(ci) * padded;
        std::memset(row, 0, sizeof(float) * padded);
        std::memcpy(row + pad, src + static_cast<size_t>(ci) * T, sizeof(float) * T);
    }
    worker.synchronize();
    // 2. K > 1: threads split the 16-frame blocks; each computes every output-channel block for its
    //    frames, so the inputs it touches (cin rows x (16 + K - 1) floats) stay in L1.
    split_range(frames16 / 16, worker_index, worker_count, lo, hi);
    for (int tb = lo; tb < hi; ++tb) {
        const int t0 = tb * 16;
        const int m = std::min(16, T - t0);
        for (int ob = 0; ob < co4 / 4; ++ob) {
            const int o0 = ob * 4;
            const int no = std::min(4, c.output_channels - o0);
            float32x4_t acc[4][4];
            for (int j = 0; j < 4; ++j) {
                const float32x4_t bj = vdupq_n_f32(c.bias[o0 + j]);
                for (int q = 0; q < 4; ++q) {
                    acc[j][q] = bj;
                }
            }
            const float *wp = c.values.data() + static_cast<size_t>(ob) * cin * K * 4;
            for (int ci = 0; ci < cin; ++ci) {
                const float *xc = xp + static_cast<size_t>(ci) * padded + t0;
                for (int kk = 0; kk < K; ++kk, wp += 4) {
                    const float32x4_t wv = vld1q_f32(wp);
                    const float32x4_t x0 = vld1q_f32(xc + kk), x1 = vld1q_f32(xc + kk + 4);
                    const float32x4_t x2 = vld1q_f32(xc + kk + 8), x3 = vld1q_f32(xc + kk + 12);
                    NEON_FMA16(x0, x1, x2, x3, wv)
                }
            }
            store_block(acc, out, rows_ld, T, t0, m, o0, no);
        }
    }
}

#else
void apply_gelu(float *x, size_t lo, size_t hi) noexcept {
    for (size_t i = lo; i < hi; ++i) {
        x[i] = 0.5F * x[i] * (1.0F + std::erf(x[i] * 0.70710678118654752F));
    }
}
void convolve_1d(const PackedConvWeights &c, const float *src, float *out, int rows_ld, int frames,
                 Workspace &workspace, WorkerContext &worker) noexcept {
    const int worker_index = worker.index;
    const int worker_count = worker.count;

    int lo, hi;
    split_range(c.output_channels, worker_index, worker_count, lo, hi);
    const int T = frames;
    for (int o = lo; o < hi; ++o) {
        for (int t = 0; t < T; ++t) {
            float value = c.bias[o];
            for (int ci = 0; ci < c.input_channels; ++ci) {
                for (int k = 0; k < c.kernel_size; ++k) {
                    const int pos = t + k - c.kernel_size / 2;
                    if (pos < 0 || pos >= T) {
                        continue;
                    }
                    const auto &w = c.values;
                    const size_t ix =
                        (static_cast<size_t>(o / 4) * c.input_channels * c.kernel_size + ci * c.kernel_size + k) * 4 +
                        o % 4;
                    value = std::fma(src[static_cast<size_t>(ci) * T + pos], w[ix], value);
                }
            }
            out[rows_ld ? static_cast<size_t>(t) * rows_ld + o : static_cast<size_t>(o) * T + t] = value;
        }
    }
}
#endif

void normalize_channels(const float *src, const float *g, const float *b, float *out, int C, int T, float epsilon,
                        Workspace &workspace, WorkerContext &worker) noexcept {
    const int worker_index = worker.index;
    const int worker_count = worker.count;

    int lo = 0, hi = 0;
    split_range(T, worker_index, worker_count, lo, hi);
    float *m = workspace.norm_mean.data();
    float *v = workspace.norm_inverse_std.data();
    for (int t = lo; t < hi; ++t) {
        m[t] = 0.0F;
        v[t] = 0.0F;
    }
    for (int c = 0; c < C; ++c) {
        for (int t = lo; t < hi; ++t) {
            m[t] += src[static_cast<size_t>(c) * T + t];
        }
    }
    for (int t = lo; t < hi; ++t) {
        m[t] /= static_cast<float>(C);
    }
    for (int c = 0; c < C; ++c) {
        for (int t = lo; t < hi; ++t) {
            const float dd = src[static_cast<size_t>(c) * T + t] - m[t];
            v[t] += dd * dd;
        }
    }
    for (int t = lo; t < hi; ++t) {
        v[t] = 1.0F / std::sqrt(v[t] / static_cast<float>(C) + epsilon);
    }
    for (int c = 0; c < C; ++c) {
        for (int t = lo; t < hi; ++t) {
            out[static_cast<size_t>(c) * T + t] = (src[static_cast<size_t>(c) * T + t] - m[t]) * v[t] * g[c] + b[c];
        }
    }
}

void convolve_depthwise_1d(const float *input, const float *weight, const float *bias, float *output, int channels,
                           int frames, int kernel, WorkerContext &worker) noexcept {
    const int worker_index = worker.index;
    const int worker_count = worker.count;

    int lo = 0, hi = 0;
    split_range(channels, worker_index, worker_count, lo, hi);
    const int pad = kernel / 2;
    for (int c = lo; c < hi; ++c) {
        const float *xc = input + static_cast<size_t>(c) * frames;
        float *ec = output + static_cast<size_t>(c) * frames;
        std::fill_n(ec, frames, bias ? bias[c] : 0.0F);
        for (int k = 0; k < kernel; ++k) {
            const float wk = weight[static_cast<size_t>(c) * kernel + k];
            const int a0 = std::max(0, pad - k);
            const int b0 = std::min(frames, frames + pad - k);
            for (int t = a0; t < b0; ++t) {
                ec[t] += wk * xc[t + k - pad];
            }
        }
    }
}

// Same packed GEMM for 1D and 2D im2col panels; geometry stays in the input packer.
void multiply_panel(const float *weights, const float *panel, const float *bias, float *output, int rows, int cout,
                    int T, int t0, int count, int ob_begin, int ob_end) noexcept {
    for (int ob = ob_begin; ob < ob_end; ++ob) {
        const int o0 = ob * 4, no = std::min(4, cout - o0);
        const float *wp = weights + static_cast<size_t>(ob) * rows * 4;
#if defined(__aarch64__) && defined(__ARM_NEON)
        float32x4_t acc[4][4];
        for (auto &channel : acc) {
            for (auto &quad : channel) {
                quad = vdupq_n_f32(0);
            }
        }
        const float *x = panel;
        for (int r = 0; r < rows; ++r, x += 16, wp += 4) {
            const auto wv = vld1q_f32(wp);
            const auto x0 = vld1q_f32(x), x1 = vld1q_f32(x + 4);
            const auto x2 = vld1q_f32(x + 8), x3 = vld1q_f32(x + 12);
            NEON_FMA16(x0, x1, x2, x3, wv)
        }
        for (int j = 0; j < 4; ++j) {
            for (auto &quad : acc[j]) {
                quad = vaddq_f32(quad, vdupq_n_f32(bias[o0 + j]));
            }
        }
        store_block(acc, output, 0, T, t0, count, o0, no);
#else
        for (int j = 0; j < no; ++j) {
            for (int t = 0; t < count; ++t) {
                float value = 0;
                for (int r = 0; r < rows; ++r) {
                    value = std::fma(panel[r * 16 + t], wp[r * 4 + j], value);
                }
                output[static_cast<size_t>(o0 + j) * T + t0 + t] = value + bias[o0 + j];
            }
        }
#endif
    }
}

void convolve_1d_strided(const PackedConvWeights &weights, const float *input, int input_frames, float *output,
                         int stride, int dilation, int padding_left, int output_frames, Workspace &workspace,
                         WorkerContext &worker) noexcept {
    const int rows = weights.input_channels * weights.kernel_size;
    float *panel = workspace.panels.data() + static_cast<size_t>(worker.index) * workspace.panel_rows * 16;
    int begin, end;
    split_range((output_frames + 15) / 16, worker.index, worker.count, begin, end);
    for (int block = begin; block < end; ++block) {
        const int frame = block * 16;
        const int count = std::min(16, output_frames - frame);
        for (int channel = 0; channel < weights.input_channels; ++channel) {
            for (int tap = 0; tap < weights.kernel_size; ++tap) {
                float *row = panel + static_cast<size_t>(channel * weights.kernel_size + tap) * 16;
                for (int lane = 0; lane < 16; ++lane) {
                    const int64_t source = static_cast<int64_t>(frame + lane) * stride +
                                           static_cast<int64_t>(tap) * dilation - padding_left;
                    row[lane] = lane < count && source >= 0 && source < input_frames
                                    ? input[static_cast<size_t>(channel) * input_frames + source]
                                    : 0.0F;
                }
            }
        }
        multiply_panel(weights.values.data(), panel, weights.bias.data(), output, rows, weights.output_channels,
                       output_frames, frame, count, 0, weights.padded_output_channels / 4);
    }
}

void convolve_2d(const PackedConvWeights &weights, const Conv2dConfig &config, const float *input, int H, int W,
                 float *output, Workspace &workspace, WorkerContext &worker) noexcept {
    const int worker_index = worker.index;
    const int worker_count = worker.count;

    const auto &c = config;
    const int OH = static_cast<int>(
        (int64_t(H) + 2LL * c.padding_height - static_cast<int64_t>(c.dilation_height) * (c.kernel_height - 1) - 1) /
            c.stride_height +
        1);
    const int OW = static_cast<int>(
        (int64_t(W) + 2LL * c.padding_width - static_cast<int64_t>(c.dilation_width) * (c.kernel_width - 1) - 1) /
            c.stride_width +
        1);
    const int T = OH * OW, blocks = (T + 15) / 16;
    int tb_begin = 0, tb_end = blocks, ob_begin = 0, ob_end = weights.padded_output_channels / 4;
    // Spatial blocks are preferred for cache reuse; tiny planes split output channels instead.
    if (blocks >= worker_count) {
        split_range(blocks, worker_index, worker_count, tb_begin, tb_end);
    } else {
        split_range(weights.padded_output_channels / 4, worker_index, worker_count, ob_begin, ob_end);
    }
    if (ob_begin == ob_end) {
        return;
    }
    float *panel = workspace.panels.data() + static_cast<size_t>(worker_index) * workspace.panel_rows * 16;
    for (int tb = tb_begin; tb < tb_end; ++tb) {
        const int t0 = tb * 16, count = std::min(16, T - t0);
        int row = 0;
        for (int ci = 0; ci < c.input_channels; ++ci) {
            for (int kh = 0; kh < c.kernel_height; ++kh) {
                for (int kw = 0; kw < c.kernel_width; ++kw, ++row) {
                    float *packed = panel + static_cast<size_t>(row) * 16;
                    for (int t = 0; t < 16; ++t) {
                        const int64_t ih = int64_t((t0 + t) / OW) * c.stride_height - c.padding_height +
                                           int64_t(kh) * c.dilation_height;
                        const int64_t iw =
                            int64_t((t0 + t) % OW) * c.stride_width - c.padding_width + int64_t(kw) * c.dilation_width;
                        packed[t] = t < count && ih >= 0 && ih < H && iw >= 0 && iw < W
                                        ? input[(static_cast<size_t>(ci) * H + ih) * W + iw]
                                        : 0.0F;
                    }
                }
            }
        }
        multiply_panel(weights.values.data(), panel, weights.bias.data(), output, weights.input_channels,
                       weights.output_channels, T, t0, count, ob_begin, ob_end);
    }
}

void apply_leaky_relu_clamp(float *values, size_t begin, size_t end, float slope, float minimum) noexcept {
    size_t i = begin;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; i + 4 <= end; i += 4) {
        const auto value = vld1q_f32(values + i);
        const auto activated = vbslq_f32(vcltq_f32(value, vdupq_n_f32(0)), vmulq_n_f32(value, slope), value);
        vst1q_f32(values + i, vmaxq_f32(activated, vdupq_n_f32(minimum)));
    }
#endif
    for (; i < end; ++i) {
        values[i] = std::max(values[i] < 0 ? values[i] * slope : values[i], minimum);
    }
}

void pool_max_2d(const float *input, float *output, int channels, int H, int W, int KH, int KW, int SH, int SW,
                 WorkerContext &worker) noexcept {
    const int worker_index = worker.index;
    const int worker_count = worker.count;

    const int OH = (H - KH) / SH + 1, OW = (W - KW) / SW + 1;
    int lo, hi;
    split_range(channels * OH * OW, worker_index, worker_count, lo, hi);
    for (int index = lo; index < hi; ++index) {
        const int channel = index / (OH * OW), h = (index / OW) % OH, w = index % OW;
        float value = -std::numeric_limits<float>::infinity();
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                value = std::max(value, input[(static_cast<size_t>(channel) * H + h * SH + kh) * W + w * SW + kw]);
            }
        }
        output[index] = value;
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#undef NEON_FMA16
#endif

} // namespace engine::neon::detail
