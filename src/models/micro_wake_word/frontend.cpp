#include "engine/models/micro_wake_word/frontend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace engine::models::micro_wake_word {
namespace {

constexpr int kSampleRate = 16000;
constexpr size_t kWindowSamples = 480;
constexpr size_t kHopSamples = 160;
constexpr size_t kFftSize = 512;
constexpr size_t kChannels = 40;
constexpr int kWindowBits = 12;
constexpr int kFilterbankBits = 12;
constexpr int kNoiseBits = 14;
constexpr int kSmoothingBits = 10;
constexpr int kPcanSnrBits = 12;
constexpr int kPcanOutputBits = 6;
constexpr int kPcanGainBits = 21;
constexpr int kLogScaleBits = 16;
constexpr int kLogSegmentsBits = 7;
constexpr uint32_t kLogScale = 65536;
constexpr uint32_t kLogCoefficient = 45426;
constexpr float kOutputScale = 0.0390625F;

// The model was trained with the TensorFlow Lite Micro fixed-point frontend.
// Its FFT rounds at every radix stage, so a floating-point FFT changes the
// quantized features around decision boundaries. This specialized radix-4
// implementation preserves that arithmetic for the only supported size (512).
// The algorithm follows KissFFT (Copyright 2003-2010 Mark Borgerding), used by
// TensorFlow Lite Micro under its BSD 3-Clause license.
struct FixedComplex {
    int16_t real = 0;
    int16_t imaginary = 0;
};

int16_t fixed_round(int32_t value) {
    return static_cast<int16_t>((value + (1 << 14)) >> 15);
}

int16_t fixed_divide(int16_t value, int divisor) {
    return fixed_round(static_cast<int32_t>(value) * (32767 / divisor));
}

FixedComplex fixed_multiply(const FixedComplex & left, const FixedComplex & right) {
    return {
        fixed_round(static_cast<int32_t>(left.real) * right.real -
                    static_cast<int32_t>(left.imaginary) * right.imaginary),
        fixed_round(static_cast<int32_t>(left.real) * right.imaginary +
                    static_cast<int32_t>(left.imaginary) * right.real),
    };
}

FixedComplex fixed_add(const FixedComplex & left, const FixedComplex & right) {
    return {
        static_cast<int16_t>(left.real + right.real),
        static_cast<int16_t>(left.imaginary + right.imaginary),
    };
}

FixedComplex fixed_subtract(const FixedComplex & left, const FixedComplex & right) {
    return {
        static_cast<int16_t>(left.real - right.real),
        static_cast<int16_t>(left.imaginary - right.imaginary),
    };
}

class FixedRealFft512 {
  public:
    FixedRealFft512() {
        constexpr double pi = 3.141592653589793238462643383279502884;
        for (size_t index = 0; index < twiddles_.size(); ++index) {
            const double phase = -2.0 * pi * static_cast<double>(index) / twiddles_.size();
            twiddles_[index] = {
                static_cast<int16_t>(std::floor(0.5 + 32767.0 * std::cos(phase))),
                static_cast<int16_t>(std::floor(0.5 + 32767.0 * std::sin(phase))),
            };
        }
        for (size_t index = 0; index < real_twiddles_.size(); ++index) {
            const double phase = -pi *
                ((static_cast<double>(index) + 1.0) / twiddles_.size() + 0.5);
            real_twiddles_[index] = {
                static_cast<int16_t>(std::floor(0.5 + 32767.0 * std::cos(phase))),
                static_cast<int16_t>(std::floor(0.5 + 32767.0 * std::sin(phase))),
            };
        }
    }

    void forward(const std::array<int16_t, kFftSize> & input,
                 std::array<FixedComplex, kFftSize / 2 + 1> & output) {
        std::array<FixedComplex, kFftSize / 2> packed{};
        for (size_t index = 0; index < packed.size(); ++index) {
            packed[index] = {input[2 * index], input[2 * index + 1]};
        }
        complex_fft(tmp_.data(), packed.data(), 1, 64);

        FixedComplex dc = tmp_[0];
        dc.real = fixed_divide(dc.real, 2);
        dc.imaginary = fixed_divide(dc.imaginary, 2);
        output[0] = {
            static_cast<int16_t>(dc.real + dc.imaginary), 0};
        output[kFftSize / 2] = {
            static_cast<int16_t>(dc.real - dc.imaginary), 0};

        for (size_t index = 1; index <= kFftSize / 4; ++index) {
            FixedComplex positive = tmp_[index];
            FixedComplex negative = {
                tmp_[tmp_.size() - index].real,
                static_cast<int16_t>(-tmp_[tmp_.size() - index].imaginary),
            };
            positive.real = fixed_divide(positive.real, 2);
            positive.imaginary = fixed_divide(positive.imaginary, 2);
            negative.real = fixed_divide(negative.real, 2);
            negative.imaginary = fixed_divide(negative.imaginary, 2);
            const FixedComplex even = fixed_add(positive, negative);
            const FixedComplex odd = fixed_subtract(positive, negative);
            const FixedComplex rotated = fixed_multiply(odd, real_twiddles_[index - 1]);
            output[index] = {
                static_cast<int16_t>((even.real + rotated.real) >> 1),
                static_cast<int16_t>((even.imaginary + rotated.imaginary) >> 1),
            };
            output[tmp_.size() - index] = {
                static_cast<int16_t>((even.real - rotated.real) >> 1),
                static_cast<int16_t>((rotated.imaginary - even.imaginary) >> 1),
            };
        }
    }

  private:
    void butterfly4(FixedComplex * output, size_t stride, size_t count) const {
        const size_t second = 2 * count;
        const size_t third = 3 * count;
        for (size_t index = 0; index < count; ++index) {
            FixedComplex & first_value = output[index];
            FixedComplex & second_value = output[index + count];
            FixedComplex & third_value = output[index + second];
            FixedComplex & fourth_value = output[index + third];
            first_value.real = fixed_divide(first_value.real, 4);
            first_value.imaginary = fixed_divide(first_value.imaginary, 4);
            second_value.real = fixed_divide(second_value.real, 4);
            second_value.imaginary = fixed_divide(second_value.imaginary, 4);
            third_value.real = fixed_divide(third_value.real, 4);
            third_value.imaginary = fixed_divide(third_value.imaginary, 4);
            fourth_value.real = fixed_divide(fourth_value.real, 4);
            fourth_value.imaginary = fixed_divide(fourth_value.imaginary, 4);

            const FixedComplex first = fixed_multiply(second_value, twiddles_[index * stride]);
            const FixedComplex second_term = fixed_multiply(third_value, twiddles_[index * stride * 2]);
            const FixedComplex third_term = fixed_multiply(fourth_value, twiddles_[index * stride * 3]);
            const FixedComplex difference = fixed_subtract(first_value, second_term);
            first_value = fixed_add(first_value, second_term);
            const FixedComplex sum = fixed_add(first, third_term);
            const FixedComplex delta = fixed_subtract(first, third_term);
            third_value = fixed_subtract(first_value, sum);
            first_value = fixed_add(first_value, sum);
            second_value = {
                static_cast<int16_t>(difference.real + delta.imaginary),
                static_cast<int16_t>(difference.imaginary - delta.real),
            };
            fourth_value = {
                static_cast<int16_t>(difference.real - delta.imaginary),
                static_cast<int16_t>(difference.imaginary + delta.real),
            };
        }
    }

    void complex_fft(FixedComplex * output, const FixedComplex * input,
                     size_t stride, size_t count) const {
        if (count == 1) {
            for (size_t radix = 0; radix < 4; ++radix) {
                output[radix] = input[radix * stride];
            }
        } else {
            const size_t child_count = count / 4;
            for (size_t radix = 0; radix < 4; ++radix) {
                complex_fft(output + radix * count, input + radix * stride,
                            stride * 4, child_count);
            }
        }
        butterfly4(output, stride, count);
    }

    std::array<FixedComplex, kFftSize / 2> twiddles_{};
    std::array<FixedComplex, kFftSize / 4> real_twiddles_{};
    std::array<FixedComplex, kFftSize / 2> tmp_{};
};

constexpr std::array<uint16_t, 130> kLogLut = {
    0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163,
    2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
    3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
    5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
    5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
    5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
    5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
    4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
    3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
    2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
    1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
};

int significant_bits(uint64_t value) {
    if (value == 0) return 0;
    int result = 0;
    while (value != 0) {
        ++result;
        value >>= 1U;
    }
    return result;
}

uint32_t rounded_integer_sqrt(uint64_t value) {
    if (value == 0) return 0;
    uint64_t root = static_cast<uint64_t>(std::sqrt(static_cast<long double>(value)));
    while ((root + 1) <= std::numeric_limits<uint32_t>::max() &&
           (root + 1) * (root + 1) <= value) ++root;
    while (root * root > value) --root;
    if (value - root * root > root && root < std::numeric_limits<uint32_t>::max()) ++root;
    return static_cast<uint32_t>(root);
}

float hz_to_mel(float frequency) {
    return 1127.0F * std::log1p(frequency / 700.0F);
}

uint32_t integer_log(uint32_t value) {
    const uint32_t integer = static_cast<uint32_t>(significant_bits(value) - 1);
    int32_t fraction = static_cast<int32_t>(value - (uint64_t{1} << integer));
    if (integer < kLogScaleBits) fraction <<= kLogScaleBits - integer;
    else fraction >>= integer - kLogScaleBits;
    const uint32_t segment = static_cast<uint32_t>(fraction) >> (kLogScaleBits - kLogSegmentsBits);
    const uint32_t segment_unit = (uint32_t{1} << kLogScaleBits) >> kLogSegmentsBits;
    const int32_t c0 = kLogLut[segment];
    const int32_t c1 = kLogLut[segment + 1];
    const int32_t relative = ((c1 - c0) *
                              (fraction - static_cast<int32_t>(segment_unit * segment))) >> kLogScaleBits;
    const uint32_t log2 = (integer << kLogScaleBits) +
                          static_cast<uint32_t>(fraction + c0 + relative);
    const uint32_t round = kLogScale / 2;
    const uint32_t natural = static_cast<uint32_t>(
        (static_cast<uint64_t>(kLogCoefficient) * log2 + round) >> kLogScaleBits);
    return ((natural << 6U) + round) >> kLogScaleBits;
}

int16_t pcan_gain(uint32_t value) {
    const float scaled = static_cast<float>(value) / 128.0F;
    const float gain = static_cast<float>(uint32_t{1} << kPcanGainBits) *
                       std::pow(scaled + 80.0F, -0.95F);
    return gain > 32767.0F ? 32767 : static_cast<int16_t>(gain + 0.5F);
}

std::array<int16_t, 125> make_gain_lut() {
    std::array<int16_t, 125> result{};
    result[0] = pcan_gain(0);
    result[1] = pcan_gain(1);
    for (int interval = 2; interval <= 32; ++interval) {
        const uint32_t x0 = uint32_t{1} << (interval - 1);
        const uint32_t x1 = x0 + (x0 >> 1U);
        const uint32_t x2 = interval == 32 ? x0 + (x0 - 1) : 2 * x0;
        const int16_t y0 = pcan_gain(x0);
        const int16_t y1 = pcan_gain(x1);
        const int16_t y2 = pcan_gain(x2);
        const int32_t difference1 = static_cast<int32_t>(y1) - y0;
        const int32_t difference2 = static_cast<int32_t>(y2) - y0;
        result[static_cast<size_t>(4 * interval - 6)] = y0;
        result[static_cast<size_t>(4 * interval - 5)] =
            static_cast<int16_t>(4 * difference1 - difference2);
        result[static_cast<size_t>(4 * interval - 4)] =
            static_cast<int16_t>(difference2 - (4 * difference1 - difference2));
    }
    return result;
}

int16_t interpolate_gain(uint32_t value, const std::array<int16_t, 125> & lut) {
    if (value <= 2) return lut[value];
    const int interval = significant_bits(value);
    const size_t offset = static_cast<size_t>(4 * interval - 6);
    const int16_t fraction = static_cast<int16_t>(
        ((interval < 11 ? value << (11 - interval) : value >> (interval - 11)) & 0x3ffU));
    int32_t result = (static_cast<int32_t>(lut[offset + 2]) * fraction) >> 5;
    result += static_cast<int32_t>(static_cast<uint32_t>(lut[offset + 1]) << 5U);
    result *= fraction;
    result = (result + (1 << 14)) >> 15;
    result += lut[offset];
    return static_cast<int16_t>(result);
}

uint32_t pcan_shrink(uint32_t value) {
    if (value < (2U << kPcanSnrBits)) {
        return (value * value) >> (2 + 2 * kPcanSnrBits - kPcanOutputBits);
    }
    return (value >> (kPcanSnrBits - kPcanOutputBits)) - (1U << kPcanOutputBits);
}

struct FilterSegment {
    int start = 0;
    std::vector<int16_t> weights;
    std::vector<int16_t> unweights;
};

std::array<FilterSegment, kChannels + 1> make_filterbank() {
    std::array<FilterSegment, kChannels + 1> result;
    std::array<float, kChannels + 1> centers{};
    const float low = hz_to_mel(125.0F);
    const float high = hz_to_mel(7500.0F);
    const float spacing = (high - low) / static_cast<float>(kChannels + 1);
    for (size_t index = 0; index < centers.size(); ++index) {
        centers[index] = low + spacing * static_cast<float>(index + 1);
    }
    constexpr float hz_per_bin = 0.5F * kSampleRate / (kFftSize / 2);
    int segment_start = static_cast<int>(1.5F + 125.0F / hz_per_bin);
    for (size_t channel = 0; channel < result.size(); ++channel) {
        int end = segment_start;
        while (hz_to_mel(static_cast<float>(end) * hz_per_bin) <= centers[channel]) ++end;
        auto & segment = result[channel];
        segment.start = segment_start;
        segment.weights.resize(static_cast<size_t>(end - segment_start));
        segment.unweights.resize(segment.weights.size());
        const float denominator = channel == 0 ? low : centers[channel - 1];
        for (int frequency = segment_start; frequency < end; ++frequency) {
            const float weight =
                (centers[channel] - hz_to_mel(static_cast<float>(frequency) * hz_per_bin)) /
                (centers[channel] - denominator);
            const size_t offset = static_cast<size_t>(frequency - segment_start);
            segment.weights[offset] = static_cast<int16_t>(
                std::floor(weight * (1 << kFilterbankBits) + 0.5F));
            segment.unweights[offset] = static_cast<int16_t>(
                std::floor((1.0F - weight) * (1 << kFilterbankBits) + 0.5F));
        }
        segment_start = end;
    }
    return result;
}

}  // namespace

struct MicroSpeechFrontend::Impl {
    std::array<int16_t, kWindowSamples> coefficients{};
    std::array<int16_t, kWindowSamples> input{};
    size_t input_used = 0;
    std::array<uint32_t, kChannels> noise_estimate{};
    std::array<FilterSegment, kChannels + 1> filterbank = make_filterbank();
    std::array<int16_t, 125> gain_lut = make_gain_lut();
    FixedRealFft512 fft;

    Impl() {
        constexpr float pi = 3.14159265358979323846F;
        const float step = 2.0F * pi / static_cast<float>(kWindowSamples);
        for (size_t index = 0; index < coefficients.size(); ++index) {
            const float value = 0.5F - 0.5F * std::cos(step * (static_cast<float>(index) + 0.5F));
            coefficients[index] = static_cast<int16_t>(
                std::floor(value * (1 << kWindowBits) + 0.5F));
        }
    }

    std::vector<float> process(const int16_t * samples) {
        std::copy_n(samples, kHopSamples, input.data() + input_used);
        input_used += kHopSamples;
        if (input_used < kWindowSamples) return {};

        std::array<int16_t, kFftSize> fft_input{};
        int16_t maximum = 0;
        for (size_t index = 0; index < kWindowSamples; ++index) {
            const int16_t value = static_cast<int16_t>(
                (static_cast<int32_t>(input[index]) * coefficients[index]) >> kWindowBits);
            maximum = std::max<int16_t>(maximum, value < 0 ? static_cast<int16_t>(-value) : value);
            fft_input[index] = value;
        }
        std::move(input.begin() + kHopSamples, input.end(), input.begin());
        input_used -= kHopSamples;

        const int input_shift = 15 - significant_bits(static_cast<uint16_t>(maximum));
        for (int16_t & value : fft_input) {
            value = static_cast<int16_t>(static_cast<uint16_t>(value) << input_shift);
        }
        std::array<FixedComplex, kFftSize / 2 + 1> spectrum{};
        fft.forward(fft_input, spectrum);
        std::array<uint32_t, kFftSize / 2 + 1> energy{};
        for (size_t index = 0; index < energy.size(); ++index) {
            const int32_t real = spectrum[index].real;
            const int32_t imaginary = spectrum[index].imaginary;
            energy[index] = static_cast<uint32_t>(real * real + imaginary * imaginary);
        }

        std::array<uint64_t, kChannels + 1> accumulated{};
        uint64_t carried = 0;
        for (size_t channel = 0; channel < filterbank.size(); ++channel) {
            uint64_t weighted = 0;
            uint64_t unweighted = 0;
            const auto & segment = filterbank[channel];
            for (size_t index = 0; index < segment.weights.size(); ++index) {
                const uint64_t magnitude = energy[static_cast<size_t>(segment.start) + index];
                weighted += static_cast<uint64_t>(segment.weights[index]) * magnitude;
                unweighted += static_cast<uint64_t>(segment.unweights[index]) * magnitude;
            }
            accumulated[channel] = carried + weighted;
            carried = unweighted;
        }

        std::vector<float> result(kChannels);
        for (size_t channel = 0; channel < kChannels; ++channel) {
            uint32_t signal = rounded_integer_sqrt(accumulated[channel + 1]) >> input_shift;
            const uint32_t smoothing = channel % 2 == 0 ? 409U : 983U;
            const uint32_t scaled = signal << kSmoothingBits;
            uint32_t estimate = static_cast<uint32_t>(
                (static_cast<uint64_t>(scaled) * smoothing +
                 static_cast<uint64_t>(noise_estimate[channel]) * ((1U << kNoiseBits) - smoothing)) >>
                kNoiseBits);
            noise_estimate[channel] = estimate;
            estimate = std::min(estimate, scaled);
            const uint32_t floor = static_cast<uint32_t>(
                (static_cast<uint64_t>(signal) * 819U) >> kNoiseBits);
            signal = std::max((scaled - estimate) >> kSmoothingBits, floor);

            const uint32_t gain = static_cast<uint16_t>(
                interpolate_gain(noise_estimate[channel], gain_lut));
            const uint32_t snr = static_cast<uint32_t>(
                (static_cast<uint64_t>(signal) * gain) >> 6);
            signal = pcan_shrink(snr);
            signal <<= 3U;
            const uint32_t logged = signal > 1 ? integer_log(signal) : 0;
            result[channel] = static_cast<float>(std::min<uint32_t>(logged, 65535U)) * kOutputScale;
        }
        return result;
    }
};

MicroSpeechFrontend::MicroSpeechFrontend() : impl_(std::make_unique<Impl>()) {}
MicroSpeechFrontend::~MicroSpeechFrontend() = default;

void MicroSpeechFrontend::reset() {
    impl_->input.fill(0);
    impl_->input_used = 0;
    impl_->noise_estimate.fill(0);
}

std::vector<float> MicroSpeechFrontend::process_hop(const int16_t * samples, size_t count) {
    if (!samples || count != kHopSamples) {
        throw std::runtime_error("microWakeWord frontend requires exactly 160 PCM samples per hop");
    }
    return impl_->process(samples);
}

std::vector<float> MicroSpeechFrontend::compute(const std::vector<float> & samples) {
    reset();
    std::vector<float> result;
    std::array<int16_t, kHopSamples> pcm{};
    for (size_t offset = 0; offset + kHopSamples < samples.size(); offset += kHopSamples) {
        for (size_t index = 0; index < kHopSamples; ++index) {
            const float scaled = std::clamp(
                samples[offset + index] * 32768.0F, -32768.0F, 32767.0F);
            pcm[index] = static_cast<int16_t>(scaled);
        }
        auto row = process_hop(pcm.data(), pcm.size());
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

}  // namespace engine::models::micro_wake_word
