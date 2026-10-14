#include "engine/community_models/piper_tts/rknn_decoder.h"

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

namespace engine::models::piper_tts {

namespace {

using Clock = std::chrono::steady_clock;

// The window contract of tools/prepare_piper_tts_rknn_vocoder.py. The engine is
// compiled at `window` slices; every step emits `chunk` of them and keeps
// `padding` real slices of context on each side, which is what makes a window
// decode identically to a full-length decode. 24 is measured, not guessed: the
// inverse flow's receptive field is ~16 slices, and at the 8+8 paroli ships the
// interior windows score 8-60 dB against a full-length decode (93: 98+ dB, 77:
// 51+ dB).
constexpr int64_t kChunk = 45;
constexpr int64_t kPadding = 24;
// Seam stitching: compare this many samples of the previous window's tail against a
// search window in the next one, cut at the closest match, then blend the compare
// window. The search runs in steps of 2 samples.
constexpr int64_t kDepopCompare = 24;
constexpr int64_t kDepopSearch = 44;

#if defined(ENGINE_HAVE_RKNN)
std::string rknn_status(const char * what, int code) {
    std::ostringstream out;
    out << "Piper TTS decoder RKNN: " << what << " failed with code " << code;
    return out.str();
}

std::vector<char> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Piper TTS decoder RKNN: cannot open model " + path);
    }
    const auto size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("Piper TTS decoder RKNN: model " + path + " is empty");
    }
    std::vector<char> data(static_cast<size_t>(size));
    in.seekg(0);
    if (!in.read(data.data(), size)) {
        throw std::runtime_error("Piper TTS decoder RKNN: short read on " + path);
    }
    return data;
}
#endif

}  // namespace

struct PiperTtsRknnDecoder::Impl {
    int64_t window = 0;
#if defined(ENGINE_HAVE_RKNN)
    rknn_context ctx = 0;
    int64_t step = 0;  // audio samples per latent slice
    uint32_t input_count = 0;
    std::vector<float> latent;  // [channels][window]
    std::vector<float> mask;    // [window]
    std::vector<float> global;  // the voice's g input, zero (single-speaker Piper)
    std::vector<float> audio;

    ~Impl() {
        if (ctx != 0) {
            rknn_destroy(ctx);
        }
    }

    void load(const std::string & path, int64_t channels, int64_t samples_per_slice) {
        auto model = read_file(path);
        const int ret = rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
        if (ret != 0) {
            throw std::runtime_error(rknn_status("rknn_init", ret));
        }
        // AUTO lets the driver pick, and for a graph this small it picks one core
        // (/sys/kernel/debug/rknpu/load stays at 100/0/0). Ask for all three by name.
        if (const int m = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2); m != 0) {
            debug::timing_log_scalar("piper_tts.decoder.rknn.core_mask_ret", m);
            if (const int a = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO); a != 0) {
                debug::timing_log_scalar("piper_tts.decoder.rknn.core_mask_auto_ret", a);
            }
        }

        rknn_input_output_num io_num{};
        if (const int q = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(io_num)", q));
        }
        if (io_num.n_input < 2 || io_num.n_output != 1) {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: expected the decoder slice (z + y_mask, plus g for a "
                "multi-speaker voice), got " +
                std::to_string(io_num.n_input) + " inputs / " +
                std::to_string(io_num.n_output) + " outputs");
        }
        input_count = io_num.n_input;
        rknn_tensor_attr input{};
        input.index = 0;
        if (const int q = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(input_attr)", q));
        }
        rknn_tensor_attr mask_attr{};
        mask_attr.index = 1;
        if (const int q = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &mask_attr, sizeof(mask_attr));
            q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(input_attr)", q));
        }
        rknn_tensor_attr output{};
        output.index = 0;
        if (const int q = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)); q != 0) {
            throw std::runtime_error(rknn_status("rknn_query(output_attr)", q));
        }
        debug::timing_log_scalar(
            "piper_tts.decoder.rknn.io",
            std::string(input.name) + " type=" + std::to_string(input.type) +
                " elems=" + std::to_string(input.n_elems) +
                " + " + std::string(mask_attr.name) +
                " elems=" + std::to_string(mask_attr.n_elems) +
                (io_num.n_input > 2
                    ? " + g elems=" + std::to_string(global_elems()) : std::string{}) +
                " -> " + std::string(output.name) + " elems=" + std::to_string(output.n_elems));
        if (std::string(input.name) != "z" || std::string(mask_attr.name) != "y_mask") {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: model " + path + " inputs are '" +
                std::string(input.name) + "'/'" + std::string(mask_attr.name) +
                "', expected 'z'/'y_mask'; compile it with tools/prepare_piper_tts_rknn_vocoder.py");
        }
        if (input.n_dims != 3 || static_cast<int64_t>(input.dims[1]) != channels) {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: model " + path + " takes " +
                (input.n_dims == 3 ? std::to_string(input.dims[1]) : std::string{"?"}) +
                " latent channels, this voice has " + std::to_string(channels) +
                "; compile the engine from this voice (tools/prepare_piper_tts_rknn_vocoder.py)");
        }
        window = static_cast<int64_t>(input.dims[2]);
        // Exact, not a lower bound. A narrower engine (paroli's 45+5+5 = 55-slice
        // contract compiles from the same seam and passes every other gate) would let
        // `run_window` copy a 93-slice window into a 55-slice row: overrun into the next
        // channel and past the end of the latent buffer on the last one.
        if (window != kChunk + 2 * kPadding) {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: model " + path + " has a " + std::to_string(window) +
                "-slice window, expected " + std::to_string(kChunk) + " emitted + 2*" +
                std::to_string(kPadding) + " context = " + std::to_string(kChunk + 2 * kPadding) +
                "; compile it with tools/prepare_piper_tts_rknn_vocoder.py --window " +
                std::to_string(kChunk + 2 * kPadding));
        }
        if (mask_attr.n_elems != static_cast<uint32_t>(window)) {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: model " + path + " masks " +
                std::to_string(mask_attr.n_elems) + " frames per " + std::to_string(window) +
                "-slice window");
        }
        step = static_cast<int64_t>(output.n_elems) / window;
        if (step * window != static_cast<int64_t>(output.n_elems) || step != samples_per_slice) {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: model " + path + " voices " +
                std::to_string(output.n_elems) + " samples per " + std::to_string(window) +
                "-slice window, this voice expects " + std::to_string(samples_per_slice) +
                " samples per latent slice");
        }
        latent.assign(static_cast<size_t>(channels * window), 0.0F);
        mask.assign(static_cast<size_t>(window), 0.0F);
        if (io_num.n_input > 2) {
            rknn_tensor_attr global_attr{};
            global_attr.index = 2;
            if (const int q = rknn_query(
                    ctx, RKNN_QUERY_INPUT_ATTR, &global_attr, sizeof(global_attr)); q != 0) {
                throw std::runtime_error(rknn_status("rknn_query(input_attr)", q));
            }
            global.assign(static_cast<size_t>(global_attr.n_elems), 0.0F);
        }
        audio.assign(static_cast<size_t>(output.n_elems), 0.0F);
        debug::timing_log_scalar("piper_tts.decoder.rknn.window", window);
    }

    uint32_t global_elems() {
        rknn_tensor_attr attr{};
        attr.index = 2;
        if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)) != 0) {
            return 0;
        }
        return attr.n_elems;
    }

    // One window: `slice_count` slices of `source` (row-major [channels][frames]) starting at
    // slice `lo`, zero-padded to the compiled width and masked off over the padding, voiced into
    // `audio`.
    void run_window(
        const float * source,
        int64_t channels,
        int64_t frames,
        int64_t lo,
        int64_t slice_count) {
        std::fill(latent.begin(), latent.end(), 0.0F);
        for (int64_t channel = 0; channel < channels; ++channel) {
            std::copy_n(
                source + channel * frames + lo,
                slice_count,
                latent.begin() + channel * window);
        }
        // 1 over the real slices, 0 over the padding: the flow multiplies its activations by the
        // mask after every convolution, so a padded frame marked 1 propagates its conv biases
        // into the real frames and costs 10-20 dB there.
        std::fill(mask.begin(), mask.end(), 0.0F);
        std::fill_n(mask.begin(), static_cast<size_t>(slice_count), 1.0F);

        std::vector<rknn_input> inputs(static_cast<size_t>(input_count));
        for (size_t index = 0; index < inputs.size(); ++index) {
            inputs[index] = {};
            inputs[index].index = static_cast<uint32_t>(index);
            // The buffer is fp32 and pass_through = 0, so the width is the element count
            // times sizeof(float); INPUT_ATTR.size reports the native (fp16) width and is
            // not the expected size here (same as the qwen3_tts sidecar).
            inputs[index].type = RKNN_TENSOR_FLOAT32;
            inputs[index].fmt = RKNN_TENSOR_UNDEFINED;
            inputs[index].pass_through = 0;
        }
        inputs[0].buf = latent.data();
        inputs[0].size = static_cast<uint32_t>(latent.size() * sizeof(float));
        inputs[1].buf = mask.data();
        inputs[1].size = static_cast<uint32_t>(mask.size() * sizeof(float));
        if (inputs.size() > 2) {
            inputs[2].buf = global.data();
            inputs[2].size = static_cast<uint32_t>(global.size() * sizeof(float));
        }
        if (const int r = rknn_inputs_set(ctx, input_count, inputs.data()); r != 0) {
            throw std::runtime_error(rknn_status("rknn_inputs_set", r));
        }
        if (const int r = rknn_run(ctx, nullptr); r != 0) {
            throw std::runtime_error(rknn_status("rknn_run", r));
        }
        rknn_output out{};
        out.index = 0;
        out.is_prealloc = 0;
        out.want_float = 1;
        if (const int r = rknn_outputs_get(ctx, 1, &out, nullptr); r != 0) {
            throw std::runtime_error(rknn_status("rknn_outputs_get", r));
        }
        const auto * data = static_cast<const float *>(out.buf);
        const size_t count = std::min(audio.size(), static_cast<size_t>(out.size / sizeof(float)));
        std::copy_n(data, count, audio.begin());
        rknn_outputs_release(ctx, 1, &out);
        if (count != audio.size()) {
            throw std::runtime_error(
                "Piper TTS decoder RKNN: window returned " + std::to_string(count) +
                " samples, expected " + std::to_string(audio.size()));
        }
    }
#else
    ~Impl() = default;
#endif
};

PiperTtsRknnDecoder::PiperTtsRknnDecoder(
    const std::string & model_path,
    int64_t channels,
    int64_t samples_per_slice)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    impl_->load(model_path, channels, samples_per_slice);
#else
    (void)model_path;
    (void)channels;
    (void)samples_per_slice;
    throw std::runtime_error("Piper TTS decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

PiperTtsRknnDecoder::~PiperTtsRknnDecoder() = default;

int64_t PiperTtsRknnDecoder::window() const { return impl_->window; }

int64_t PiperTtsRknnDecoder::chunk() const { return kChunk; }

std::vector<float> PiperTtsRknnDecoder::decode(
    const float * latent,
    int64_t channels,
    int64_t frames) {
#if defined(ENGINE_HAVE_RKNN)
    auto & impl = *impl_;
    const auto start = Clock::now();
    std::vector<float> out;
    out.reserve(static_cast<size_t>(frames * impl.step));
    int64_t windows = 0;
    for (int64_t emitted = 0; emitted < frames; emitted += kChunk) {
        // The window that reaches the end of the utterance emits to the end, and extends its
        // left context to fill the compiled shape: the flow's receptive field reaches back
        // further than kPadding, and a tail window carrying only kPadding slices of context
        // scores -0.1 dB on its emitted slices.
        const bool tail = emitted + kChunk + kPadding >= frames;
        const int64_t lo = std::max(
            int64_t{0}, std::min(emitted - kPadding, frames - impl.window));
        const int64_t hi = tail ? frames : std::min(frames, emitted + kChunk + kPadding);
        const int64_t slice_count = hi - lo;
        const int64_t end_pad = tail ? 0 : kPadding;
        impl.run_window(latent, channels, frames, lo, slice_count);
        int64_t real_start = (emitted - lo) * impl.step;
        const int64_t real_end = (slice_count - end_pad) * impl.step;
        if (real_end <= real_start) {
            throw std::runtime_error("Piper TTS decoder RKNN: empty window emission");
        }
        if (out.size() >= static_cast<size_t>(kDepopCompare) &&
            real_end - real_start >= kDepopSearch * 2) {
            // Stitch: cut where the previous tail best matches this window, then blend
            // the compare window so the seam carries no step.
            const float * prev = out.data() + out.size() - kDepopCompare;
            const int64_t base = real_start - std::min(real_start, kDepopCompare);
            int64_t best_start = real_start;
            double best_diff = -1.0;
            for (int64_t j = 0; j < kDepopSearch * 2; j += 2) {
                if (base + j + kDepopCompare > static_cast<int64_t>(impl.audio.size())) {
                    break;
                }
                double diff = 0.0;
                for (int64_t k = 0; k < kDepopCompare; ++k) {
                    diff += std::abs(prev[k] - impl.audio[static_cast<size_t>(base + j + k)]);
                }
                if (best_diff < 0.0 || diff < best_diff) {
                    best_diff = diff;
                    best_start = base + j + kDepopCompare;
                }
            }
            real_start = best_start;
            for (int64_t k = 0; k < kDepopCompare; ++k) {
                const float weight = static_cast<float>(k) / static_cast<float>(kDepopCompare);
                out[out.size() - kDepopCompare + static_cast<size_t>(k)] =
                    out[out.size() - kDepopCompare + static_cast<size_t>(k)] * (1.0F - weight) +
                    impl.audio[static_cast<size_t>(real_start - kDepopCompare + k)] * weight;
            }
        }
        out.insert(
            out.end(),
            impl.audio.begin() + real_start,
            impl.audio.begin() + real_end);
        ++windows;
        if (tail) {
            break;
        }
    }
    debug::timing_log_scalar("piper_tts.decoder.rknn.windows", windows);
    debug::timing_log_scalar(
        "piper_tts.decoder.rknn.step_ms",
        debug::elapsed_ms(start));
    return out;
#else
    (void)latent;
    (void)channels;
    (void)frames;
    throw std::runtime_error("Piper TTS decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

}  // namespace engine::models::piper_tts
