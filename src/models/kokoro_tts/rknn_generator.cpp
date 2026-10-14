#include "engine/models/kokoro_tts/rknn_generator.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_HAVE_RKNN)
#include <rknn_api.h>
#endif

namespace engine::models::kokoro_tts {

namespace {

using Clock = std::chrono::steady_clock;

// The generator's receptive field in latent frames, measured by `receptive_field()` in
// tools/prepare_kokoro_tts_rknn_generator.py: each conv_transpose scales the field by
// its stride and adds kernel-minus-stride, each resblock convolution adds
// (kernel-1)*dilation at its own rate, conv_post adds 6, all divided by the total
// upsampling. A window carrying this much real context on each side decodes its
// emitted frames identically to a full-length decode (measured: 136 dB).
constexpr int64_t kPadding = 28;
constexpr int64_t kStyleDim = 128;

#if defined(ENGINE_HAVE_RKNN)
std::string rknn_status(const char * what, int code) {
    std::ostringstream out;
    out << "Kokoro generator RKNN: " << what << " failed with code " << code;
    return out.str();
}

std::vector<char> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Kokoro generator RKNN: cannot open model " + path);
    }
    const auto size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("Kokoro generator RKNN: model " + path + " is empty");
    }
    std::vector<char> data(static_cast<size_t>(size));
    in.seekg(0);
    if (!in.read(data.data(), size)) {
        throw std::runtime_error("Kokoro generator RKNN: short read on " + path);
    }
    return data;
}
#endif

}  // namespace

struct KokoroTtsRknnGenerator::Impl {
    int64_t window = 0;
    int64_t emitted = 0;
    // Latent frames each window shares with its predecessor (the overlap walk); 0 keeps
    // the shipping walk, where windows abut and the seams are hard-written.
    int64_t overlap = 0;
#if defined(ENGINE_HAVE_RKNN)
    // One context, deliberately. The reference Kokoro engine runs three (one per NPU
    // core, `rknn_dup_context`) to overlap windows; measured here it is 1.15x on the
    // window loop (9650 -> 8375 ms for 17 windows) and it corrupts the output
    // (-7 dB SNR against the CPU decode, rms 2x the reference), because duplicated
    // contexts share the runtime's tensor staging. This engine streams ~75 MB of
    // weights per window, so the windows are DRAM-bound, not scheduler-bound: the
    // three cores cannot read the same weights three times faster than one.
    rknn_context ctx = 0;
    int64_t channels = 0;
    int64_t rows = 0;
    int64_t scale = 0;   // latent frames to spectrogram columns
    int64_t mid_rate = 0;  // latent frames to the stage-0 resblock rate
    uint32_t input_count = 0;
    std::vector<float> x;
    std::vector<float> har;
    std::vector<float> style;
    std::vector<float> mask_mid;
    std::vector<float> mask_out;
    std::vector<float> spec;

    ~Impl() {
        if (ctx != 0) {
            rknn_destroy(ctx);
        }
    }

    uint32_t input_elems(uint32_t index) {
        rknn_tensor_attr attr{};
        attr.index = index;
        if (const int q = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(input_attr)", q));
        }
        return attr.n_elems;
    }

    std::string input_name(uint32_t index) {
        rknn_tensor_attr attr{};
        attr.index = index;
        if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)) != 0) {
            return {};
        }
        return attr.name;
    }

    void load(const std::string & path, int64_t latent_channels, int64_t feature_rows,
              int64_t samples_per_frame) {
        auto model = read_file(path);
        const int ret = rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
        if (ret != 0) {
            throw std::runtime_error(rknn_status("rknn_init", ret));
        }
        // AUTO lets the driver pick, and for a graph this small it picks one core
        // (/sys/kernel/debug/rknpu/load stays at 100/0/0). Ask for all three by name:
        // one job can be split across the cores, and it is (Core0 94%, Core1/2 18%).
        if (const int m = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2); m != 0) {
            debug::timing_log_scalar("kokoro.decoder.rknn.core_mask_ret", m);
            if (const int a = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO); a != 0) {
                debug::timing_log_scalar("kokoro.decoder.rknn.core_mask_auto_ret", a);
            }
        }

        rknn_input_output_num io_num{};
        if (const int q = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(io_num)", q));
        }
        static const char * kNames[5] = {"x", "har", "style", "mask_mid", "mask_out"};
        if (io_num.n_input != 5 || io_num.n_output != 1) {
            throw std::runtime_error(
                "Kokoro generator RKNN: model " + path + " has " +
                std::to_string(io_num.n_input) + " inputs / " + std::to_string(io_num.n_output) +
                " outputs, expected the 5 inputs x/har/style/mask_mid/mask_out and one "
                "spectrogram output; compile it with "
                "tools/prepare_kokoro_tts_rknn_generator.py");
        }
        for (uint32_t index = 0; index < io_num.n_input; ++index) {
            if (input_name(index) != kNames[index]) {
                throw std::runtime_error(
                    "Kokoro generator RKNN: model " + path + " input " + std::to_string(index) +
                    " is '" + input_name(index) + "', expected '" + kNames[index] + "'");
            }
        }
        input_count = io_num.n_input;

        rknn_tensor_attr x_attr{};
        x_attr.index = 0;
        if (const int q = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &x_attr, sizeof(x_attr)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(input_attr)", q));
        }
        if (x_attr.n_dims != 3 || static_cast<int64_t>(x_attr.dims[1]) != latent_channels) {
            throw std::runtime_error(
                "Kokoro generator RKNN: model " + path + " takes " +
                (x_attr.n_dims == 3 ? std::to_string(x_attr.dims[1]) : std::string{"?"}) +
                " latent channels, this model has " + std::to_string(latent_channels) +
                "; compile the engine from this model "
                "(tools/prepare_kokoro_tts_rknn_generator.py)");
        }
        window = static_cast<int64_t>(x_attr.dims[2]);
        if (window <= 2 * kPadding) {
            throw std::runtime_error(
                "Kokoro generator RKNN: model " + path + " has a " + std::to_string(window) +
                "-frame window, which carries no emitted frames at the " + std::to_string(kPadding) +
                "-frame context");
        }
        emitted = window - 2 * kPadding;
        const uint32_t mid_elems = input_elems(3);
        const uint32_t out_elems = input_elems(4);
        if (mid_elems % static_cast<uint32_t>(window) != 0 ||
            (out_elems - 1) % static_cast<uint32_t>(window) != 0) {
            throw std::runtime_error(
                "Kokoro generator RKNN: model " + path + " masks " + std::to_string(mid_elems) +
                "/" + std::to_string(out_elems) + " frames per " + std::to_string(window) +
                "-frame window");
        }
        mid_rate = static_cast<int64_t>(mid_elems) / window;
        // The mask rates are the rates the graph normalizes at, in spectrogram columns, not
        // in audio samples: 60 columns per latent frame here, and the iSTFT head turns each
        // column into gen_istft_hop_size samples, so the model's 300 audio samples per frame
        // is scale * hop.
        scale = (static_cast<int64_t>(out_elems) - 1) / window;
        const int64_t columns = scale * window + 1;
        if (scale <= 0 || mid_rate <= 0 || samples_per_frame % scale != 0 ||
            static_cast<int64_t>(input_elems(1)) != feature_rows * columns ||
            static_cast<int64_t>(input_elems(2)) != kStyleDim) {
            throw std::runtime_error(
                "Kokoro generator RKNN: model " + path + " voices " + std::to_string(scale) +
                " spectrogram columns per latent frame, this model has " +
                std::to_string(samples_per_frame) + " audio samples per latent frame");
        }
        rknn_tensor_attr output{};
        output.index = 0;
        if (const int q = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(output_attr)", q));
        }
        if (static_cast<int64_t>(output.n_elems) != feature_rows * columns) {
            throw std::runtime_error(
                "Kokoro generator RKNN: model " + path + " returns " +
                std::to_string(output.n_elems) + " values per window, expected " +
                std::to_string(feature_rows * columns));
        }

        channels = latent_channels;
        rows = feature_rows;
        x.assign(static_cast<size_t>(channels * window), 0.0F);
        har.assign(static_cast<size_t>(rows * columns), 0.0F);
        style.assign(static_cast<size_t>(kStyleDim), 0.0F);
        mask_mid.assign(static_cast<size_t>(mid_rate * window), 0.0F);
        mask_out.assign(static_cast<size_t>(columns), 0.0F);
        spec.assign(static_cast<size_t>(rows * columns), 0.0F);
        debug::timing_log_scalar(
            "kokoro.decoder.rknn.io",
            std::string("window=") + std::to_string(window) + " emitted=" + std::to_string(emitted) +
                " scale=" + std::to_string(scale) + " mid_rate=" + std::to_string(mid_rate) +
                " -> " + std::string(output.name));
    }

    // One window: `count` latent frames of `source_x`/`source_har` starting at frame `lo`,
    // zero-padded to the compiled width and mask-zeroed over the padding.
    void run_window(
        const float * source_x,
        int64_t x_stride,
        const float * source_har,
        int64_t har_stride,
        const float * source_style,
        int64_t lo,
        int64_t count) {
        std::fill(x.begin(), x.end(), 0.0F);
        for (int64_t channel = 0; channel < channels; ++channel) {
            std::copy_n(
                source_x + channel * x_stride + lo,
                count,
                x.begin() + channel * window);
        }
        const int64_t columns = scale * window + 1;
        const int64_t har_cols = scale * count + 1;
        std::fill(har.begin(), har.end(), 0.0F);
        for (int64_t row = 0; row < rows; ++row) {
            std::copy_n(
                source_har + row * har_stride + scale * lo,
                har_cols,
                har.begin() + row * columns);
        }
        std::copy_n(source_style, static_cast<size_t>(kStyleDim), style.begin());
        // 1 over the real frames, 0 over the padding, at every rate the graph
        // normalizes at. AdaIN has to see the mask in its statistics *and* zero its
        // output there: a padded frame carrying the affine or a conv bias propagates
        // through the dilated convolutions into the last real frames.
        std::fill(mask_mid.begin(), mask_mid.end(), 0.0F);
        std::fill_n(mask_mid.begin(), static_cast<size_t>(mid_rate * count), 1.0F);
        std::fill(mask_out.begin(), mask_out.end(), 0.0F);
        std::fill_n(mask_out.begin(), static_cast<size_t>(har_cols), 1.0F);

        std::vector<rknn_input> inputs(static_cast<size_t>(input_count));
        const float * buffers[5] = {
            x.data(), har.data(), style.data(), mask_mid.data(), mask_out.data()};
        const size_t element_counts[5] = {
            x.size(), har.size(), style.size(), mask_mid.size(), mask_out.size()};
        for (size_t index = 0; index < inputs.size(); ++index) {
            inputs[index] = {};
            inputs[index].index = static_cast<uint32_t>(index);
            // The buffers are fp32 and pass_through = 0, so the width is the element
            // count times sizeof(float); INPUT_ATTR.size reports the native (fp16)
            // width and is not the expected size here (as in the other sidecars).
            inputs[index].type = RKNN_TENSOR_FLOAT32;
            inputs[index].fmt = RKNN_TENSOR_UNDEFINED;
            inputs[index].pass_through = 0;
            inputs[index].size = static_cast<uint32_t>(element_counts[index] * sizeof(float));
            inputs[index].buf = const_cast<float *>(buffers[index]);
        }
        if (const int r = rknn_inputs_set(ctx, input_count, inputs.data()); r != 0) {
            throw std::runtime_error(rknn_status("rknn_inputs_set", r));
        }
        if (const int r = rknn_run(ctx, nullptr); r != 0) {
            throw std::runtime_error(rknn_status("rknn_run", r));
        }
        rknn_output out = {};
        out.index = 0;
        out.is_prealloc = 0;
        out.want_float = 1;
        if (const int r = rknn_outputs_get(ctx, 1, &out, nullptr); r != 0) {
            throw std::runtime_error(rknn_status("rknn_outputs_get", r));
        }
        const auto * data = static_cast<const float *>(out.buf);
        const size_t got = std::min(spec.size(), static_cast<size_t>(out.size / sizeof(float)));
        std::copy_n(data, got, spec.begin());
        rknn_outputs_release(ctx, 1, &out);
        if (got != spec.size()) {
            throw std::runtime_error(
                "Kokoro generator RKNN: window returned " + std::to_string(got) + " values, expected " +
                std::to_string(spec.size()));
        }
    }
#else
    ~Impl() = default;
#endif
};

KokoroTtsRknnGenerator::KokoroTtsRknnGenerator(
    const std::string & model_path,
    int64_t latent_channels,
    int64_t feature_rows,
    int64_t samples_per_frame,
    int64_t overlap)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    impl_->load(model_path, latent_channels, feature_rows, samples_per_frame);
    if (overlap < 0 || overlap >= impl_->emitted) {
        throw std::runtime_error(
            "Kokoro generator RKNN: overlap " + std::to_string(overlap) +
            " must be in [0, " + std::to_string(impl_->emitted) + ") emitted frames");
    }
    impl_->overlap = overlap;
#else
    (void)model_path;
    (void)latent_channels;
    (void)feature_rows;
    (void)samples_per_frame;
    (void)overlap;
    throw std::runtime_error("Kokoro generator RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

KokoroTtsRknnGenerator::~KokoroTtsRknnGenerator() = default;

int64_t KokoroTtsRknnGenerator::window() const { return impl_->window; }

int64_t KokoroTtsRknnGenerator::emitted() const { return impl_->emitted; }

int64_t KokoroTtsRknnGenerator::columns_per_frame() const {
#if defined(ENGINE_HAVE_RKNN)
    return impl_->scale;
#else
    return 0;
#endif
}

std::vector<float> KokoroTtsRknnGenerator::run(
    const float * x,
    int64_t x_stride,
    const float * har,
    int64_t har_stride,
    const float * style,
    int64_t frames,
    const float * reference) {
#if defined(ENGINE_HAVE_RKNN)
    auto & impl = *impl_;
    const auto start = Clock::now();
    const int64_t columns = impl.scale * frames + 1;
    std::vector<float> out(static_cast<size_t>(impl.rows * columns), 0.0F);
    int64_t windows = 0;
    const int64_t window_columns = impl.scale * impl.window + 1;
    const int64_t stride = impl.emitted - impl.overlap;
    const int64_t mag_rows = impl.rows / 2;  // log-magnitude rows, then the phase rows
    for (int64_t emitted = 0; emitted < frames; emitted += stride) {
        // The window that reaches the end of the latent emits to the end and extends its
        // left context to fill the compiled shape, so the end of the utterance sits at
        // the same tensor boundary it has in a full-length decode.
        const bool tail = emitted + impl.emitted + kPadding >= frames;
        const int64_t lo = std::max(int64_t{0}, std::min(emitted - kPadding, frames - impl.window));
        const int64_t hi =
            tail ? frames : std::min(frames, emitted + impl.emitted + kPadding);
        impl.run_window(x, x_stride, har, har_stride, style, lo, hi - lo);
        const int64_t end = tail ? frames : std::min(frames, emitted + impl.emitted);
        const int64_t first = impl.scale * (emitted - lo);
        const int64_t count = impl.scale * (end - emitted) + 1;
        const int64_t dst = impl.scale * emitted;
        // Per-window loudness correction against a full-length CPU decode: an additive
        // offset on the log-magnitude rows that aligns this window's loudness with the
        // reference's over the same columns. The statistic is the linear-domain
        // loudness, the per-column sum of exp(log-magnitude) over the magnitude rows
        // (the waveform's energy proxy; a log-domain mean is dominated by the silence
        // floor, which is not comparable between the two normalizations). The gain is
        // the log ratio of the loudness sums over the columns the reference marks as
        // speech (within 40 dB of the region's loudest column); a region that is pure
        // silence in the reference gets no gain. The phase rows are untouched.
        float gain = 0.0F;
        if (reference != nullptr) {
            std::vector<float> loud_new(static_cast<size_t>(count), 0.0F);
            std::vector<float> loud_ref(static_cast<size_t>(count), 0.0F);
            float loud_max = 0.0F;
            for (int64_t row = 0; row < mag_rows; ++row) {
                const float * s = impl.spec.data() + row * window_columns + first;
                const float * r = reference + row * columns + dst;
                for (int64_t i = 0; i < count; ++i) {
                    loud_new[static_cast<size_t>(i)] += std::exp(s[i]);
                    loud_ref[static_cast<size_t>(i)] += std::exp(r[i]);
                }
            }
            for (int64_t i = 0; i < count; ++i) {
                loud_max = std::max(loud_max, loud_ref[static_cast<size_t>(i)]);
            }
            if (loud_max > 1e-3F) {  // the region carries speech in the reference
                double sum_new = 0.0;
                double sum_ref = 0.0;
                for (int64_t i = 0; i < count; ++i) {
                    if (loud_ref[static_cast<size_t>(i)] > 0.01F * loud_max) {
                        sum_new += loud_new[static_cast<size_t>(i)];
                        sum_ref += loud_ref[static_cast<size_t>(i)];
                    }
                }
                if (sum_new > 0.0 && sum_ref > 0.0) {
                    gain = static_cast<float>(std::log(sum_ref / sum_new));
                }
            }
        }
        // Overlap walk: the first `overlap` emitted frames of this window were also
        // emitted by its predecessor, so blend their spectrogram columns with a
        // raised-cosine ramp (0 = keep the previous window, 1 = take this one) instead
        // of overwriting them. With overlap = 0 the region is empty and the stitch is
        // the shipping hard-write.
        const int64_t blend =
            emitted > 0 && impl.overlap > 0
                ? std::min(impl.scale * impl.overlap, count - 1) + 1
                : 0;
        for (int64_t row = 0; row < impl.rows; ++row) {
            const float * s = impl.spec.data() + row * window_columns + first;
            float * d = out.data() + row * columns + dst;
            const float g = row < mag_rows ? gain : 0.0F;
            for (int64_t i = 0; i < blend; ++i) {
                const float w = 0.5F - 0.5F * std::cos(
                    3.14159265358979323846F * static_cast<float>(i) /
                    static_cast<float>(blend - 1));
                d[i] = (1.0F - w) * d[i] + w * (s[i] + g);
            }
            std::copy_n(s + blend, static_cast<size_t>(count - blend), d + blend);
            if (g != 0.0F) {
                for (int64_t i = blend; i < count; ++i) {
                    d[i] += g;
                }
            }
        }
        ++windows;
        if (tail) {
            break;
        }
    }
    debug::timing_log_scalar("kokoro.decoder.rknn.windows", windows);
    debug::timing_log_scalar("kokoro.decoder.rknn.step_ms", debug::elapsed_ms(start));
    return out;
#else
    (void)x;
    (void)x_stride;
    (void)har;
    (void)har_stride;
    (void)style;
    (void)frames;
    (void)reference;
    throw std::runtime_error("Kokoro generator RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

}  // namespace engine::models::kokoro_tts
